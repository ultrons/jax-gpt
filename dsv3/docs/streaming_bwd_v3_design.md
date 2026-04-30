# streaming_bwd_v3: Fully-Fused Pallas Backward Kernel Design

**Date:** 2026-04-03  
**Status:** Design — pre-implementation  
**Target:** ~4× overall training speedup vs JAX on 4×8×8 EP=32 FSDP=16  
**Builds on:** streaming_bwd_v2 (v119c), profile analysis in `pallas_vs_jax_training_analysis.md`

---

## 1. Why v3

### v2 pathology (measured on 4×8×8, v119c, bench-v4 profile)

| Problem | Cost | Fix |
|---|---|---|
| 4× `all_gather("ep")` before kernel | 25s | Move A2A inside kernel (v3 core) |
| `psum("ep") + dynamic_slice` instead of `psum_scatter` | 42s | 4-line fix OR v3 handles internally |
| Sync barrier (all_reduce.311, 609KB in 7.8s) | 7.8s | Eliminated by psum_scatter fix |
| 0% collective overlap (vs 74% for JAX) | systemic | Eliminated when sync barrier is gone |
| `psum("fsdp")` all-reduce on weight grads | ~18s | Replace with `psum_scatter("fsdp")` |
| **Total Pallas backward** | **178.6s** | — |
| **JAX backward** | **79.3s** | — |

v3 goal: reduce backward to ~10–15s (op-time, same measurement basis), bringing overall
to ~4× faster than JAX.

### Why the forward is fast and the backward isn't

The forward kernel (`fused_ep_moe_fwd_streaming_v1`) does ALL EP communication via ICI DMA 
**inside the Pallas kernel**. From outside, it looks like one opaque compute op. XLA sees no 
collective and cannot break its pipeline. The kernel's internal A2A overlaps with MXU GEMMs via
double-buffering. XLA's `ag_backward_pipelining` flag is irrelevant — the kernel IS the pipeline.

The backward (v2) does the opposite: 4 explicit JAX collectives before calling the kernel, 3 more
after. XLA has no ability to pipeline these with Pallas custom-call ops (opaque to async fusion
analysis). Every collective runs serially → 0% overlap → 120s of serial collective time.

v3 brings the backward to parity: A2A lives inside the Pallas kernel, same structure as forward.

---

## 2. Mathematical Correctness (VJP derivation)

### Forward data flow (inside `shard_map(ep, fsdp)`)

```
Input: tokens_l  (T/(EP*FSDP), D)  ← local tokens, EP×FSDP sharded

[1] fx_full   = all_gather(tokens_l, "ep")         → (T/FSDP, D)
[2] out_partial = PallasFwdKernel(fx_full, w1, w2, gating)  → (T/FSDP, D)  ← partial in F
    |___ Internal A2A scatter: route tokens to experts
    |___ Expert FFN (w FSDP-sharded: only F_shard channels used)
    |___ Internal A2A gather: collect outputs
[3] out_ep  = psum_scatter(out_partial, "ep", scatter_dim=0)  → (T/(EP*FSDP), D)
[4] output  = psum(out_ep, "fsdp")                            → (T/(EP*FSDP), D)  ← FSDP sums
```

### VJP (applied in REVERSE order to forward)

```
Receive: d_out_l  (T/(EP*FSDP), D)

VJP[4]  psum("fsdp"):       d_out_l → d_out_l (identity — no-op)
VJP[3]  psum_scatter("ep"): d_out_l → all_gather(d_out_l, "ep") → d_out_full (T/FSDP, D)
VJP[2]  PallasFwdKernel:    d_out_full → PallasBwdKernel(d_out_full, tokens_full, w1, w2)
                              → d_tokens_partial (T/FSDP, D)  ← partial in F from FSDP-sharded w
                              → d_w1_shard (E_local, 2, D, F_shard)
                              → d_w2_shard (E_local, F_shard, D)
                              (The "all_gather ep" inside the bwd kernel is the VJP of the
                               "internal A2A scatter" — same infrastructure, reversed direction)
VJP[1]  all_gather("ep"):   psum_scatter(d_tokens_partial_fsdp_summed, "ep") → d_tokens_l
```

### Required reductions outside the kernel

```python
# After calling v3 kernel (which handles the EP A2A internally):
d_tok_fsdp_sum = lax.psum(d_tok_partial, "fsdp")          # sum F_shard contributions: (T/FSDP, D)
d_tok_l = lax.psum_scatter(d_tok_fsdp_sum, "ep",           # scatter EP-local slice: (T/(EP*FSDP), D)
                             scatter_dimension=0, tiled=True)

# Weight grad FSDP reduction (already local in EP, just need FSDP sum):
# d_w1 shape: (E_local, 2, D, F_shard) — already correct for this device's experts
# No EP psum needed (each EP device has its own E_local experts)
# FSDP: each FSDP shard computed partial contribution → psum or psum_scatter
d_w1_l = lax.psum_scatter(d_w1_partial, "fsdp", scatter_dimension=3, tiled=True)
d_w2_l = lax.psum_scatter(d_w2_partial, "fsdp", scatter_dimension=1, tiled=True)
```

**Total external collectives in v3 wrapper: 3 calls vs current 7+ calls.**
None of them are sync barriers. All can be pipelined by XLA's async collective fusion.

---

## 3. v3 Kernel Architecture

### Design principle: mirror the forward

```
FORWARD:                                  BACKWARD (v3):
─────────────────────────────────────     ──────────────────────────────────────────
tokens_l (T/(EP*FSDP), D) input           d_out_l (T/(EP*FSDP), D) input
                                          tokens_l (T/(EP*FSDP), D) input (for recompute)
    │                                           │
    ▼                                           ▼
[EP A2A scatter via DMA]                  [EP A2A scatter via DMA]
 tokens routed to expert devices           d_out + tokens routed to expert devices
 (T_expert tokens arrive per device)       (2D payload per token: packed scatter)
    │                                           │
    ▼                                           ▼
[Per-expert FFN — VMEM GEMMs]             [Per-expert FFN backward — VMEM GEMMs]
 h = silu(x@w1g) * (x@w1u)               recompute: h_g, h_u, h_act from tokens
 out = h @ w2                              d_h_act = d_out @ w2.T
                                           d_h_g = d_h_act * h_u * silu_grad(h_g)
                                           d_h_u = d_h_act * silu(h_g)
                                           d_tok = d_h_g @ w1g.T + d_h_u @ w1u.T
                                           d_w2  += h_act.T @ d_out_scaled
                                           d_w1g += tokens.T @ d_h_g
                                           d_w1u += tokens.T @ d_h_u
    │                                           │
    ▼                                           ▼
[EP A2A gather via DMA]                   [EP A2A gather via DMA]
 outputs returned to token-owner           d_tokens returned to token-owner devices
    │                                           │
    ▼                                           ▼
out_partial (T/FSDP, D)                   d_tok_partial (T/FSDP, D) ← partial in F
─────────────────────────────────────     ──────────────────────────────────────────
[External: psum_scatter("ep")]            [External: psum("fsdp") then psum_scatter("ep")]
[External: psum("fsdp")]
```

### A2A packing strategy

The forward scatters 1 × D bytes per token. The backward needs to scatter BOTH d_out AND tokens.
Instead of 2 separate A2A passes, pack them into a single scatter with 2× payload:

```python
# Pack before scatter
packed = jnp.stack([tokens_l, d_out_l], axis=0)  # (2, T/(EP*FSDP), D)

# A2A scatter → (2, T_expert, D) at each expert device
# Unpack inside kernel:
tokens_expert = packed_recv[0]   # (T_expert, D) — for FFN recompute
d_out_expert  = packed_recv[1]   # (T_expert, D) — for backward compute
```

Total A2A data per token per direction:
- Forward: 1D scatter + 1D gather = 2D
- Backward v3: 2D scatter (packed) + 1D gather = 3D = 1.5× forward

Expected backward ICI time ≈ 1.5× forward ICI time.

### VMEM budget (debug config: D=7168, F_shard=128, bt=16)

```
Forward VMEM (per double-buffer slot):
  a2a_s_x2:   2 × bt × EP × D × 2 bytes = 2 × 16 × 32 × 7168 × 2 = 14.6 MB per slot
  a2a_s_acc_x2: same = 14.6 MB
  b_w1_x2:    2 × 2 × D × F_shard × 2 = 7.3 MB
  b_w2_x2:    2 × F_shard × D × 2 = 3.7 MB
  b_acc:      2 × bt × EP × F_shard × 4 = 0.13 MB
  Total: ~40 MB of 64 MB VMEM budget

Backward VMEM additions (for 2D packed scatter):
  a2a_packed_x2: 2 × 2 × bt × EP × D × 2 = 29.2 MB (tokens + d_out)
  d_tok_acc:     2 × bt × EP × D × 4 = 14.6 MB (float32 accumulator for d_tokens)
  d_w1_acc:      D × F_shard × 4 × 2 (gate + up) = 7.3 MB (one expert at a time, float32)
  d_w2_acc:      F_shard × D × 4 = 3.7 MB

  Total: ~55 MB — fits in 64 MB with careful layout
  May need to reduce bt (token tile) slightly for v3
```

The bt reduction for v3 (from 16 to 8 for the packed A2A) keeps the budget:
- At bt=8: a2a_packed_x2 = 14.6 MB → total ~40 MB, well within budget.

---

## 4. Implementation Plan

### Phase 0: Fix streaming_bwd_v2 collectives (quick win, ~1 week)

**File:** `dsv3/mini_dsv3/model.py:1011–1020`

Replace `psum + dynamic_slice` with `psum_scatter`:
```python
# BEFORE (model.py:1011):
d_tok_full  = jax.lax.psum(d_tok_partial,  ep_axis_name)
d_topk_full = jax.lax.psum(d_topk_partial, ep_axis_name)
T_ep_local  = g_l.shape[0]
device_ep   = jax.lax.axis_index(ep_axis_name)
d_tok_l     = jax.lax.dynamic_slice(d_tok_full, (device_ep*T_ep_local,0), (T_ep_local,D))
d_topk_l    = jax.lax.dynamic_slice(d_topk_full,(device_ep*T_ep_local,0),(T_ep_local,K))

# AFTER:
d_tok_l  = jax.lax.psum_scatter(d_tok_partial,  ep_axis_name, scatter_dimension=0, tiled=True)
d_topk_l = jax.lax.psum_scatter(d_topk_partial, ep_axis_name, scatter_dimension=0, tiled=True)
```

Also replace `psum("fsdp")` with `psum_scatter("fsdp")` inside `fused_ep_moe_bwd_streaming_v2`
(lines 623 and 642 in backward_kernel.py):
```python
# BEFORE:
d_tokens = lax.psum(d_tokens, fsdp_axis_name)
d_top_k_weights = lax.psum(d_top_k_weights, fsdp_axis_name)

# AFTER (scatter_dim=0 since token dimension):
d_tokens        = lax.psum_scatter(d_tokens,        fsdp_axis_name, scatter_dimension=0, tiled=True)
d_top_k_weights = lax.psum_scatter(d_top_k_weights, fsdp_axis_name, scatter_dimension=0, tiled=True)
```

**Expected: backward 178.6s → ~80s, overall JAX parity (1.49×)**

Validate on debug config with correctness check (loss parity), then benchmark.

---

### Phase 1: Understand the forward A2A kernel internals (~1 week)

Before writing the backward kernel, need to understand:
1. How `_fused_ep_moe_kernel` dispatches ICI DMA (`async_remote_copy` or custom DMA)
2. How the A2A routing table is built from `d2e_count_x2` and `expert_offsets_x2`
3. How double-buffering is implemented (which VMEM scratch slots flip)
4. How the gather at the end returns token results to owners

Tasks:
- [ ] Read `tpu_inference/kernels/fused_moe/v1/kernel.py` A2A section
- [ ] Add `--xla_enable_custom_call_region_trace=true` profile (bench-v5, already submitted)
  to confirm A2A DMA bandwidth achieved in forward
- [ ] Prototype a minimal "reverse A2A" using the same infrastructure on local v4

---

### Phase 2: streaming_bwd_v3 Pallas kernel (~2–3 weeks)

**New file:** `dsv3/fused_moe_bwd/backward_kernel_v3.py`

Key implementation steps:

**Step 2a: Routing metadata (same as forward)**
```python
# Inside kernel: compute routing from gating_output + top_k_indices
# Same d2e_count, expert_offsets, expert_starts as forward
# These determine which tokens go to which expert device via A2A
```

**Step 2b: Packed A2A scatter (tokens + d_out → experts)**
```python
# Pack payload: (2, T_local, D) → A2A → (2, T_expert, D) at expert device
# DMA infrastructure: same async_remote_copy as forward, but 2× payload size
# VMEM scratch: a2a_packed_x2 (2 double-buffer slots × packed (tokens, d_out))
```

**Step 2c: Per-expert FFN backward (streaming, double-buffered)**
```python
# For expert e (with expert e+1 A2A prefetch in flight):
#   tokens_e = a2a_packed_recv[0][:, expert_e_slice]  (T_expert, D)
#   d_out_e  = a2a_packed_recv[1][:, expert_e_slice]  (T_expert, D)
#   
#   # Recompute forward activations (cheap vs saving them)
#   h_g  = tokens_e @ w1[e, 0]           (T_expert, F_shard)
#   h_u  = tokens_e @ w1[e, 1]           (T_expert, F_shard)
#   h_act = silu(h_g) * h_u              (T_expert, F_shard)
#   
#   # Backward
#   d_out_scaled = d_out_e * routing_weights_e  (T_expert, D)
#   d_h_act = d_out_scaled @ w2[e].T             (T_expert, F_shard)
#   d_h_g   = d_h_act * h_u * silu_grad(h_g)    (T_expert, F_shard)
#   d_h_u   = d_h_act * silu(h_g)               (T_expert, F_shard)
#   d_tok_e = d_h_g @ w1[e,0].T + d_h_u @ w1[e,1].T  (T_expert, D)
#   
#   # Weight grads — accumulated in VMEM per expert
#   d_w2_e  += h_act.T @ d_out_scaled      (F_shard, D)
#   d_w1g_e += tokens_e.T @ d_h_g         (D, F_shard)
#   d_w1u_e += tokens_e.T @ d_h_u         (D, F_shard)
```

**Step 2d: A2A gather (d_tokens → token owners)**
```python
# Same routing table as forward A2A gather, but direction reversed
# expert_e sends d_tok_e back to token-owner devices
# Result at each device: d_tok_partial (T/FSDP, D) partial in F dimension
```

**Step 2e: Output**
```python
# Write d_tok_partial to HBM output buffer
# Write d_w1[e], d_w2[e] to HBM weight grad buffers (per-expert)
```

---

### Phase 3: Integration and correctness (~1 week)

**File:** `dsv3/fused_moe_bwd/backward_kernel_v3.py` + `dsv3/mini_dsv3/model.py`

New wrapper `fused_ep_moe_bwd_streaming_v3`:
```python
def fused_ep_moe_bwd_streaming_v3(
    d_out_l,      # (T/(EP*FSDP), D) EP-local — NOT pre-gathered
    tokens_l,     # (T/(EP*FSDP), D) EP-local
    fi_l,         # (T/(EP*FSDP), K) routing indices, EP-local
    fw_l,         # (T/(EP*FSDP), K) routing weights, EP-local
    w1_shard,     # (E_local, 2, D, F_shard)
    w2_shard,     # (E_local, F_shard, D)
    *,
    ep_axis_name, fsdp_axis_name, K, max_tpe, E_global, ...
):
    # Calls the v3 Pallas kernel (handles EP A2A internally)
    d_tok_partial, d_w1_partial, d_w2_partial, d_topk_partial = _bwd_v3_kernel(...)
    
    # Sum FSDP partial contributions to d_tok
    d_tok_full = lax.psum(d_tok_partial, fsdp_axis_name)          # (T/FSDP, D)
    d_tok_l    = lax.psum_scatter(d_tok_full, ep_axis_name,
                                  scatter_dimension=0, tiled=True)  # (T/(EP*FSDP), D)
    
    # FSDP reduce-scatter for weight grads
    d_w1_l = lax.psum_scatter(d_w1_partial, fsdp_axis_name, scatter_dimension=3, tiled=True)
    d_w2_l = lax.psum_scatter(d_w2_partial, fsdp_axis_name, scatter_dimension=1, tiled=True)
    
    # d_topk: same FSDP reduction
    d_topk_l = lax.psum_scatter(d_topk_partial, fsdp_axis_name, scatter_dimension=0, tiled=True)
    
    return d_tok_l, d_w1_l, d_w2_l, d_topk_l
```

Update `_streaming_bwd_fn` in model.py to call v3 (no pre-gather needed):
```python
# BEFORE (4× all_gather before kernel, psum+slice after):
g_full  = jax.lax.all_gather(g_l,  ep_axis_name, ...)
fx_full = jax.lax.all_gather(fx_l, ep_axis_name, ...)
fw_full = jax.lax.all_gather(fw_l, ep_axis_name, ...)
fi_full = jax.lax.all_gather(fi_l, ep_axis_name, ...)
d_tok_partial, d_w1, d_wout, d_topk = _bwd_kernel(g_full, fx_full, w1, ...)
d_tok_l = psum(d_tok_partial, "ep") + dynamic_slice  # ← WRONG

# AFTER (no pre-gather, kernel handles A2A):
d_tok_l, d_w1, d_wout, d_topk_l = fused_ep_moe_bwd_streaming_v3(
    g_l, fx_l, fw_l, fi_l, w1_shard, wout_shard, ...)
return (d_tok_l, d_topk_l, d_w1, d_wout)
```

Correctness tests:
- [ ] EP=1 on local v4: compare v3 grads vs jax.vjp(ref_moe) — rtol=1e-3 (bf16)
- [ ] EP=4 on local v4: same test with EP>1
- [ ] 4×8×8 debug config: loss parity vs JAX at step 100 (within 0.1% of 25.084)

---

## 5. Risk Assessment

### Low risk
- **Phase 0 (psum_scatter fix)**: 4 lines, mathematically certain, high value
- **FSDP-sharded weights**: same approach as v2, already validated

### Medium risk
- **A2A reversed direction**: The forward A2A scatter and gather are asymmetric in the sense 
  that "which tokens go where" depends on routing. The backward reverses this: each expert device 
  sends d_tok BACK to the original token-owner devices. The routing table is the SAME as forward 
  (same top_k_indices), just used in the opposite DMA direction. Need to verify the forward kernel's 
  DMA code supports this reversed mode.

- **Packed A2A (2D payload)**: The forward kernel's VMEM scratch is sized for D per token. 
  Doubling to 2D (packing tokens + d_out) requires reducing bt (token tile). Need to find a 
  bt value that fits in 64 MB VMEM budget while still achieving good throughput.

### Higher risk
- **silu_grad numerical precision**: Recomputing h_g, h_u inside the backward from tokens (bf16) 
  will have rounding differences vs computing from the original activations. The v2 kernel already 
  does this recompute (matches v2's current approach), so it's not a new problem, but worth 
  verifying the numerical tolerance.

- **D-tiling with packed A2A**: If D=7168 doesn't fit in VMEM for a single expert pass with packed 
  A2A, will need both D-tiling AND packed A2A simultaneously. Adds implementation complexity. 
  May be addressed by reducing bt instead.

---

## 6. Projected Performance

### After Phase 0 (psum_scatter fix, no new kernel)

| Component | Before | After Fix 0 | Notes |
|---|---|---|---|
| MoE backward | 178.6s | ~80s | psum_scatter + eliminated sync barrier |
| Collective overlap | 0% | ~50–70% | barrier gone, XLA can pipeline again |
| Step time (debug) | 6.59s | ~4.5s | ~46% faster than current Pallas |
| TPS/chip (debug) | 2,484 | ~3,100 | closing in on JAX's 3,391 |

### After Phase 2 (v3 kernel, fully fused)

| Component | v2 | v3 | Notes |
|---|---|---|---|
| EP A2A (external) | 25s (4× all_gather) | 0s | moved inside kernel |
| EP reduction (external) | 42s (psum+slice) | ~0.5s (psum_scatter) | 1 fast call |
| FSDP reduction (external) | ~18s (psum) | ~0.5s (psum_scatter) | 1 fast call |
| Kernel internal A2A | ~35s (gather_custom_fusion) | ~8-12s | packed 2D A2A vs 1D fwd |
| **Total MoE backward** | **178.6s** | **~12-15s** | |

Estimated wall-clock step time with v3 backward:
- MoE fwd: 2.8s/step (unchanged)
- MoE bwd v3: ~4-5s/step (1.5× fwd A2A + similar compute)
- Dense + attn: 5.2s/step
- **Total: ~12s/step → 4× faster than JAX (4.83s × 4.1 = 19.8s? No...)**

Wait — JAX step time is 4.83s. If v3 total step time is ~12s that's SLOWER than JAX, not 4×.
The 4× estimate assumed op-time ratios translate directly to wall-clock ratios. They don't — the
overlap factor matters enormously.

Better estimate using JAX as anchor:
- JAX forward MoE wall-clock ≈ 50.2s op-time / 10.7 overlap factor = 4.7s (too high)

Hmm. Let me use the empirical forward speedup instead:
- Pallas fwd wall-clock: (8.4/154.5) × 4.83s per JAX step × 1 = 0.26s × 3 steps = 0.78s per step
- JAX fwd wall-clock: (50.2/154.5) × 4.83s = 1.57s per step × 3 steps? No...

Actually the simplest approach: Pallas current step time is 6.59s. Forward is 6× faster.
If backward is also ~6× faster (v3 goal), the step time would be:
- Current JAX step: 4.83s = 1.57s (fwd) + 2.62s (bwd) + 0.64s (dense/attn)
  [proportional to op-times: fwd 50.2/154.5=32.5%, bwd 79.3/154.5=51.4%, rest 16.1%]
- After v3: fwd/6 = 0.26s, bwd/6 = 0.44s, dense/attn 0.64s → 1.34s total
  That's 3.6× faster than JAX. 

Using the op-time ratio: JAX (T/FSDP) op-time sum = 154.5s.
With v3: fwd=8.4s, bwd~14s (assuming 6× reduction from 79.3s), rest=15.7s → total=38.1s
Ratio to JAX: 154.5/38.1 = 4.06×
Estimated step time: 4.83s / 4.06 ≈ **1.2s/step** → **TPS/chip ≈ 13,900 (debug)**
Full config: 1,083 × 4.06 ≈ **4,400 TPS/chip**

These projections assume the forward and backward A2A achieve the same efficiency.

---

## 7. Open Questions

1. **Does the forward DMA code support reversed A2A?** The forward kernel's DMA engine sends 
   tokens TO expert devices. For the backward, we need expert devices to send d_tok BACK to 
   token-owner devices. This may require new DMA scatter patterns or can reuse the existing 
   gather pattern. Need to audit `_fused_ep_moe_kernel` DMA section.

2. **Can we pack tokens + d_out in a single A2A?** The forward kernel's A2A buffer is sized 
   for D per token. Packing 2D requires either halving bt or adding new VMEM scratch. 
   Alternatively, do TWO separate A2A passes (tokens scatter + d_out scatter), which is cleaner 
   but costs 2× the A2A overhead vs a single packed pass.

3. **FSDP psum of d_tok: can it be avoided entirely?** 
   Current v2/v3: d_tok_partial (T/FSDP, D) is partial in F → psum("fsdp") needed.
   Alternative: gather full-F weights before the kernel (like v1 did), compute exact d_tok 
   (no FSDP partial). Cost: FSDP all_gather of weights before kernel (large: E_local × D × F). 
   Probably not worth it — psum("fsdp") on 3.8 GB is only ~0.35s.

4. **Routing weights d_topk handling**: Currently computed inside kernel and returned. In v3, 
   the routing is EP-local (before A2A), so d_topk can be computed locally from the backward 
   A2A results without any additional collectives. Need to ensure the routing weight gradient 
   is computed correctly in the packed A2A setting.

5. **Checkpoint compatibility**: v3 must work inside `jax.remat` with `everything_saveable` 
   policy (or the custom checkpoint_name approach from v119c). The v3 kernel takes EP-local 
   inputs, so there's no Pallas-in-checkpoint-context issue (unlike the original v115 problem). 
   The `_moe_pallas_sg` custom_vjp wrapper handles this.

---

## 8. First Step: Just Do Phase 0

Phase 0 is 4 lines of code, eliminates 50s of backward time, and is testable today.
No new kernel required. This is the right first move before committing to Phase 2.

Do Phase 0, benchmark it, and if it gets us to ~JAX parity (within 10%), 
assess whether Phase 2 is worth the implementation cost vs other priorities.

Phase 2 (v3 kernel) is a 3–4 week effort. The payoff is potentially 4× vs JAX, 
but Phase 0 gets 90% of the way to JAX parity with 1% of the work.
