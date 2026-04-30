# Path to 4K TPS/chip — DSv3 671B on 4×8×8 v7x

*Analysis date: 2026-03-29. Based on v10 profile (ring + A2A dispatch, GBS=1024, EP=8, FSDP=64).*

---

## Hardware Baseline

| Metric | Value |
|--------|-------|
| Cluster | 4×8×8 v7x, 256 physical chips, 512 JAX devices |
| Peak MXU | 1,028,750 GFLOP/s per chip |
| Peak HBM BW | 530 GB/s per chip |
| Arithmetic intensity (knee) | 1,940 FLOP/byte |
| ICI BW | ~59 TB/s (4.7% utilized in v10) |

---

## Current State — v10

**Config:** EP=8, FSDP=64, GBS=1024 (PDBS=2 seq/device), ring + A2A dispatch, gradient checkpoint, SGD

| Metric | Value |
|--------|-------|
| Step time | 35.24s |
| cluster_TPS | 119,016 |
| **TPS/chip** | **465** |
| MXU (when active) | 37% of peak |
| MXU wall-clock busy | 37% of time |
| Effective MFU | ~14% |
| HBM BW | 23.2% of wall time |
| ICI | 4.7% |

**Target:** 4,000 TPS/chip = cluster_TPS 1,024,000 = **8.6× improvement needed**

---

## Step Time Breakdown (v10 profile)

| Component | % of step | What it is |
|-----------|-----------|------------|
| `scatter_custom_fusion.29+.31` | **26.3%** | Backward through token_sort + combine — scatter-adds from ragged_dot bwd |
| `ragged-dot-none` (12 instances) | **35.0%** | `jax.vjp` in `_a2a_bwd` re-running ragged_dot **forward** unnecessarily |
| Splash attention (fwd + bwd recompute) | **~8%** | Attention forward + backward in MoE scan |
| FSDP all-gathers (5 non-overlapped) | **~7%** | Weight gather stalls — 0% overlapped |
| MoE forward | **11.9%** | Actual A2A + matmul forward pass |
| Other | **~12%** | Data formatting, copies, collectives |

**Root cause of 35% ragged-dot waste:**
`_a2a_bwd` calls `jax.vjp(fn)` which traces through `_expert_mlp_ring_ragged_body`.
`jax.vjp` always runs the forward first before computing gradients — so every backward
step runs a full ragged_dot **forward pass** (all_gather + matmul + combine) before the
actual gradient scatter-adds. With gradient checkpointing this forward is also run during
remat, so total ragged_dot forward passes per step = 2 (original remat + vjp internal).

---

## HBM Budget

```
Peak HBM:    64.1 GB / 101.7 GB  (63%)
Free:        37.6 GB
Stack:       57.2 GB  (XLA temporaries — activations + scan layer buffers)
Heap:         6.9 GB  (weights, persistent state)
```

Top scan-saved buffers (58-layer scan stores per-layer intermediates):
- `bf16[58, 32, 112, 2048]` × 2 = **1.7 GB** — attention K/V intermediates per layer
- `bf16[58, 256, 7168]` × 2 = **426 MB** — MoE projection intermediates per layer

**37.6 GB free = ample headroom for saving more activations to reduce recompute.**

Estimated cost of saving ragged_dot intermediates (dispatched + expert_out) per layer:
- `expert_out`: ~[65536, 7168] bf16 per device = ~0.9 GB per layer
- × 58 MoE layers = ~52 GB — **too large to save all layers at once**
- Selective: save every N layers (N=4 → 13 GB) or use `dots_saveable` policy

---

## Fix Roadmap — Ordered by Impact

### Fix 1: Eliminate ragged_dot forward in backward — **~1.5× speedup**

**Problem:** `_a2a_bwd` calls `jax.vjp(fn)` which re-runs ragged_dot forward (35% of step).

**Fix:** Restructure `_a2a_fwd` to save key intermediates as residuals so `_a2a_bwd`
can compute gradients directly without `jax.vjp`:

```python
def _a2a_fwd(flat_x, wi_0, wi_1, wo, flat_indices, flat_weights, K, axis_name):
    # Run ragged_dot forward here to capture intermediates for backward
    sorted_x, dispatched_x, expert_out = _ring_ragged_forward_with_intermediates(...)
    out = _a2a_gather(expert_out, ...)   # A2A gather in forward
    return out, (flat_x, wi_0, wi_1, wo, flat_indices, flat_weights,
                 sorted_x, dispatched_x, expert_out)

def _a2a_bwd(K, axis_name, res, g):
    *inputs, sorted_x, dispatched_x, expert_out = res
    # Compute dW directly from saved intermediates — no forward re-run
    d_wo = expert_out.mT @ g_unsorted         # weight grad
    d_wi_1 = dispatched_x.mT @ g_mid         # weight grad
    d_wi_0 = sorted_x.mT @ g_gate            # weight grad
    d_flat_x = _unsort_scatter(d_sorted, flat_indices)  # scatter-add still needed
    return (d_flat_x, d_wi_0, d_wi_1, d_wo, zeros_like(flat_indices), d_flat_weights)
```

**HBM cost:** Saving intermediates for 1 scan layer at a time (full-remat scan):
- `sorted_recv` (A2A dispatched tokens, local): T_recv × D × bf16 ≈ 700 MB/layer
- `gate_out_presilu`, `up_out`, `hidden`: T_recv × D_moe × bf16 ≈ 50-100 MB/layer each
- `expert_out_sorted`: T_recv × D × bf16 ≈ 700 MB/layer
- Plus metadata (indices, sizes): < 10 MB/layer
- **Total ≈ 1.5–2 GB** for one layer at a time (scan remat processes one layer at a time)
- Well within 37.6 GB free budget.

**Complexity note (2026-03-29):** Implementing this requires writing an explicit A2A backward
that calls `jax.lax.ragged_all_to_all` directly (not via autodiff) to avoid the JAX 0.9.2
backward bug. The local FFN backward can use `jax.vjp` on `sorted_recv` inputs (8× smaller
than ring path T_all*K). This approach reduces the ragged-dot forward from T_all*K to
T_recv ≈ T_all*K/EP tokens (~8× cheaper). Full explicit backward for dW requires
a segmented matmul which is not natively available in JAX (needs Pallas or `segment_sum`).
Estimated ~3-4 hours of careful implementation + testing.

**Effect (with explicit A2A backward):**
- Eliminates `ragged-dot-none` (ring forward) entirely — replaced by local-FFN backward
- Local-FFN forward (for vjp) is 8× cheaper: ~4% instead of 35%
- Step time: 35.24s × (1 - 0.31) = **~24.3s** (conservative)
- TPS/chip: 465 × 1.45 = **~674 TPS/chip** (conservative)

If dW computation is also made explicit (no vjp re-run):
- Step time: 35.24s × 0.65 = **~22.9s**
- TPS/chip: 465 × 1.54 = **~716 TPS/chip**

Note: scatter-add (26.3%) remains — it is fundamental to the combine backward.

---

### Fix 2: Increase GBS (amortize fixed overhead) — **~2–3× depending on GBS**

Communication and data-formatting overhead is largely independent of GBS.
Compute scales proportionally with GBS.

| GBS | Est. step time | cluster_TPS | TPS/chip |
|-----|---------------|-------------|----------|
| 1024 (current) | 35.24s | 119K | 465 |
| 2048 | ~60s | ~140K | ~547 |
| 4096 | ~110s | ~153K | ~597 |
| 8192 | ~205s | ~164K | ~640 |

*Estimate: step_time ≈ 35.24 × (GBS/1024)^0.9 (sub-linear due to fixed overhead).*
*HBM limit: each 2× GBS ≈ +2 GB activation stack — fits well within 37.6 GB free.*

GBS alone is insufficient — scatter-add and ragged_dot both scale with GBS (O(tokens)).

---

### Fix 3: FSDP all-gather overlap — **~1.1×**

5 non-overlapped all-gathers (0% overlap each), totaling ~7% of step time.
`--xla_tpu_enable_async_collective_fusion_multiple_steps=true` is set but not helping.

**Investigation:** Check whether these are FSDP weight gathers or EP token gathers.
Profile shows 3 × FSDP weight gathers (~12ms each) + 1 EP token gather (12ms) + 1 smaller.
- FSDP weight gathers happen before each layer's matmul — should overlap with prior layer
- EP token gather happens before ragged_dot backward — harder to overlap

**Potential fix:** `--xla_tpu_enable_ag_backward_pipelining=true` already set.
Try increasing `--xla_tpu_scoped_vmem_limit_kib` to give scheduler more freedom.

---

### Fix 4: Scatter-add (Pallas kernel or true A2A bwd) — **~1.4×**

The scatter-adds (`scatter_custom_fusion.29+.31`) are **16% of HBM peak bandwidth** —
pathologically memory-bound because random scatter write patterns defeat HBM burst modes.

**Option A: Pallas fused scatter kernel**
Fuse the scatter-add with adjacent elementwise ops into a single Pallas kernel that
processes tokens in HBM-tile-aligned chunks. Reduces random access penalty.
`kernel_opportunity` analysis: theoretical 1.97× speedup if all memory-bound ops fused.
Complexity: high (requires writing Pallas scatter kernel + custom backward).

**Option B: Wait for JAX bug fix (ragged_all_to_all native backward)**
JAX PR #26959 added ragged_all_to_all with autodiff. The backward generates an all_to_all
for metadata + ragged_all_to_all for data — mathematically replaces scatter-add with
a sparse collective (should be bandwidth-limited by ICI not HBM).
*Blocker:* JAX 0.9.2 bug: backward crashes with async collective fusion + checkpoint + scan.
Once fixed: drop custom_vjp, use native A2A backward → eliminates scatter-adds entirely.

**Effect (if scatter-add eliminated):** -26.3% of step time → **further 1.36× speedup**.

---

### Fix 5: `dots_saveable` checkpoint policy — **~1.1× on attention**

Apply `jax.checkpoint(..., policy=jax.checkpoint_policies.dots_saveable)` to attention
layers. Saves matmul outputs (cheap to store at ~100 MB/layer) to avoid recomputing them
during backward. Only affects non-MoE layers (~8% of step).

---

## Combined Speedup Projection

Starting from v10 (465 TPS/chip):

| Fix | Speedup | Cumulative TPS/chip |
|-----|---------|---------------------|
| Current v10 | — | **465** |
| Fix 1: No vjp fwd re-run | 1.54× | **716** |
| Fix 3: FSDP overlap | 1.10× | **788** |
| Fix 2: GBS=4096 | 1.29× | **1,016** |
| Fix 4A: Pallas scatter | 1.36× | **1,382** |
| Fix 5: dots_saveable attn | 1.05× | **1,451** |

**With current EP=8, FSDP=64:** reaching 4K TPS/chip requires either the JAX A2A backward
fix (Option B) OR a Pallas scatter kernel AND higher GBS AND better overlap — multiple
compounding improvements.

**Comparison with EP=32, FSDP=16**

The v4 run (EP=32, FSDP=16, GBS=4, jax_ep) measured **cluster_TPS=2317 → TPS/chip=9.1**
(2317 / 256 physical chips). No config has reached 4K TPS/chip yet — that remains the target.

| Config | Approach | Status | cluster_TPS | TPS/chip |
|--------|----------|--------|-------------|----------|
| EP=32, FSDP=16, GBS=4, jax_ep | baseline | measured (v4) | 2,317 | **9.1** |
| EP=8, FSDP=64, GBS=1024, ring+A2A | custom_vjp | measured (v10) | 119,021 | **465** |
| EP=8, FSDP=64, GBS=1024, ring+A2A | Fix 1+2+3 | projected | — | ~1,016 |
| EP=8, FSDP=64, GBS=1024, ring+A2A | Fix 1+2+3+4+5 | projected | — | ~1,451 |
| EP=8, FSDP=64, GBS=1024, true A2A bwd | JAX bug fix | projected | — | ~3,640 |

Note: v4 has tiny GBS=4 (8K total tokens) — low TPS/chip is expected. v10 at GBS=1024
processes 4M tokens per step and is the correct point of comparison for large-batch training.

---

## Recommended Next Steps (priority order)

1. **v11 (GBS=4096) — SUBMITTED:** Validate GBS scaling. Measures actual step time and
   TPS/chip at 4× GBS. Profile included at `gs://sivaibhav-exp/dsv3/profiles/4x8x8-ep8-fsdp64-gbs4096-v11`.

2. **Fix 1 (explicit A2A backward):** ~3-4 hours implementation.
   - Modify `_a2a_fwd` to save `sorted_recv`, `gate_out_presilu`, `up_out`, `hidden`, `expert_out_sorted`,
     exp sort indices, A2A metadata (offsets, sizes)
   - Write explicit `_a2a_bwd` using `jax.lax.ragged_all_to_all` directly (not autodiff)
   - Use `jax.vjp` on the local FFN only (8× fewer tokens)
   - For dW: use `jax.ops.segment_sum` or padded batched matmul

3. **Profile v11 then v12 (Fix 1 applied):** After GBS=4096 profile, implement Fix 1, run v12.
   Expect ragged-dot-none to drop from 35% → ~4% and step time to fall ~30-45%.

4. **Benchmark EP=32, FSDP=16 with A2A dispatch:** Does A2A forward help EP=32?
   With EP=32, all_gather dispatch is 4× more expensive → A2A forward benefit is larger.

5. **JAX A2A backward:** Monitor JAX issue tracker. When fixed, drop custom_vjp entirely.
   This is the cleanest long-term path.

---

## Key Invariants

- **TPS/chip** = cluster_TPS / 256 physical chips (NOT 512 JAX devices)
- **GBS** = global batch sequences; **PDBS** = GBS / N_JAX_devices (per device)
- v7x: 2 JAX devices per chip, 2 MXU cores per chip
- `--gbs` CLI flag in train.py (renamed from `--pdbs` 2026-03-29)
- Profile at `gs://sivaibhav-exp/dsv3/profiles/4x8x8-ep8-fsdp64-gbs1024-v10`
