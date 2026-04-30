# DSv3 Streaming Backward Integration Plan

## End goal

A single `fused_ep_moe_train_fsdp` function (named in `specs/fused_ep_moe_backward_spec.md`)
that replaces the current `moe_backend=jax` path with:
- **Pallas forward** (`fused_ep_moe` from `tpu_inference`) — faster than JAX shard_map forward
- **Streaming JAX backward** (`fused_ep_moe_bwd_streaming`) — eliminates scatter-add bottleneck
- **Async FSDP weight prefetch** — overlaps ICI all_gather with TensorCore compute
- **SparseCore token routing** — replaces indexed HBM token loads with SC gathers

Integrated in `model.py` as `moe_backend=fused_pallas` per the spec.

## Phased approach

Each phase is a separate cluster benchmark. Move to the next only after the current phase
passes correctness (loss matches jax baseline) and shows expected perf improvement.

```
Phase 1:  JAX fwd  + streaming bwd                   ✅ DONE — integrated in model.py
Phase 1b: Pallas fwd + streaming bwd                  ← add Pallas forward
Phase 2:  Pallas fwd + streaming bwd + FSDP overlap   ← overlap ICI and TensorCore
Phase 3:  All of above + SparseCore token routing     ← final fused_ep_moe_train_fsdp
```

**Why phase 1 before 1b**: isolates the backward change. If phase 1 works, any failure in
phase 1b is definitively in the Pallas forward, not the backward.

---

## Background and current state

### Production baseline (as of 2026-03-31)
- Training job: `k8s/dsv3-train-4x8x8-v38.yaml` (EP=8, FSDP=16, DP=4, GBS=1024, `moe_backend=jax`)
- Reference perf: ~1037 TPS/chip at v24 (EP=8, FSDP=64) — v38 targets improvement via wider GEMMs
- MoE backward path: `_moe_jax_ep_fn_bwd` in `model.py:890` — now has both `jax.vjp` and streaming paths
- Backward bottleneck: `scatter_custom_fusion` ops at ~87 GB/s HBM BW (16% efficiency) — ~1.18s/step
- Profile: `gs://sivaibhav-exp/dsv3/profiles/4x8x8-ep8-fsdp64-gbs1024-v24`

### Weight layout (actual, from `model.py:870-875`)

| Weight | Global shape | Partition spec | Per-device shape (EP=8, FSDP=16) |
|--------|-------------|----------------|----------------------------------|
| `w0` (gate) | `(E=256, D=7168, F=2048)` | `P("ep", None, "fsdp")` | `(32, 7168, 128)` |
| `w1` (up)   | `(E=256, D=7168, F=2048)` | `P("ep", None, "fsdp")` | `(32, 7168, 128)` |
| `wout` (down) | `(E=256, F=2048, D=7168)` | `P("ep", "fsdp", None)` | `(32, 128, 7168)` |

FSDP shards the **hidden (F=2048) dimension** — each device has F_local=128.
The input dimension D=7168 is **not** FSDP-sharded.

**Key design point**: The streaming backward uses these weights WITHOUT any all_gather.
Each device computes partial dot products over its F_local=128 slice, then a `psum("fsdp")`
combines all 16 partial contributions into the final d_tokens. No ICI for weights — only
for the final reduction.

### Why `jax.vjp` is slow
JAX autodiff through the shard_map forward generates per-token scatter-adds for the routing
backward. These are serialized, non-coalesced HBM writes → 16% efficiency. A per-expert
streaming loop eliminates them: each expert accumulates `d_w1` and `d_w2` locally (VMEM),
then does a single HBM write per expert.

### Why earlier Pallas attempts failed
See `docs/kernel_feedback_pallas_bwd.md` for full details. Short version:
- **v12**: `@jax.jit` on `fused_ep_moe_bwd` inside `custom_vjp` → 60+ min nested compile hang
- **v15**: `jax.vmap` over T×K=524,288 tokens materialized `(524288, 7168, 2048)` = 7.57 TB → OOM

### What was built and validated
- `fused_ep_moe_bwd_streaming` (`fused_moe_bwd/backward_kernel.py:563`) — per-expert streaming
  backward, **correctness validated on 4×8×8** at E=256, K=8, EP=8 via `test_grad_check_v7x.py`
- `fused_ep_moe_bwd_streaming_v2` (`fused_moe_bwd/backward_kernel.py:818`) — Phase 2 variant;
  **correctness validated locally** (EP=1, fsdp=4) via `test_grad_check_v2.py` — all norms 1.000000
- **Phase 1 integrated** in `model.py:890-999` — `use_streaming_bwd=True` path in
  `_moe_jax_ep_fn_bwd`; cluster run pending (compilation hang under investigation, see below)

---

## Phase 1 — Integrate streaming backward into training ✅ CODE DONE

**Goal**: `use_streaming_bwd=True` replaces `jax.vjp` with `fused_ep_moe_bwd_streaming`.
When this matches loss curves and runs faster, retire `jax.vjp`.

### How it is wired up (`model.py:855-999`)

`_moe_jax_ep_fn` (`model.py:855`) has `nondiff_argnums=(6,7,8,9,10,11)` — the 12th arg
`use_streaming_bwd: bool` is passed as a non-differentiable constant to both fwd and bwd.

```python
# model.py:890
def _moe_jax_ep_fn_bwd(mesh, K, act_spec, ep_axis_name, max_tpe, use_streaming_bwd, res, g):
    fx, fi, fw, w0, w1, wout = res   # fi=(T/fsdp,K) int indices; fw=(T/fsdp,K) weights

    if not use_streaming_bwd:
        # ... existing jax.vjp path ...

    # ---- Streaming backward path ----
    from backward_kernel import fused_ep_moe_bwd_streaming

    E_global = w0.shape[0]  # global expert count from un-sharded EP dimension

    def _streaming_bwd_fn(g_l, fx_l, fw_l, fi_l, w0_l, w1_l, wout_l):
        # 1. EP all_gather: reconstruct T/FSDP token view
        #    Each EP device has T/(EP*FSDP) tokens; all_gather → T/FSDP
        g_full  = lax.all_gather(g_l,  ep_axis_name, axis=0, tiled=True)   # (T/fsdp, D)
        fx_full = lax.all_gather(fx_l, ep_axis_name, axis=0, tiled=True)
        fw_full = lax.all_gather(fw_l, ep_axis_name, axis=0, tiled=True)   # (T/fsdp, K)
        fi_full = lax.all_gather(fi_l, ep_axis_name, axis=0, tiled=True)   # (T/fsdp, K) int

        # 2. Stack w0/w1 → (E_local, 2, D, F_local=128) — NO FSDP all_gather.
        #    Kernel works directly on the FSDP-local F slice; FSDP psum combines partials.
        w1_stk = jnp.stack([w0_l, w1_l], axis=1)  # (E_local, 2, D=7168, F_local=128)

        # 3. Streaming backward with precomputed routing (fi/fw from residuals).
        #    Use precomputed routing instead of recomputing from raw logits — necessary
        #    because DSv3 forward uses gate_bias + routed_scaling_factor which the kernel's
        #    compute_routing doesn't reproduce.
        d_tok_partial, d_w1_p, d_wo_p, d_topk_partial = fused_ep_moe_bwd_streaming(
            g_full, fx_full, w1_stk, wout_l,
            gating_output=None,
            top_k=K,
            scoring_fn="sigmoid",
            renormalize_topk_logits=True,
            act_fn="silu",
            ep_axis_name=ep_axis_name,
            max_tpe=max_tpe,
            top_k_indices_precomputed=fi_full,
            top_k_weights_precomputed=fw_full.astype(jnp.float32),
            return_dtopk=True,          # return d_top_k_weights (T/fsdp, K) not d_gating (T/fsdp, E)
            E_global_override=E_global,
        )

        # 4. FSDP psum: combine partial F_local contributions across 16 FSDP devices.
        d_tok_fsdp  = lax.psum(d_tok_partial,  "fsdp")   # (T/fsdp, D)
        d_topk_fsdp = lax.psum(d_topk_partial, "fsdp")   # (T/fsdp, K)

        # 5. EP psum: combine partial expert contributions across EP devices.
        d_tok_full  = lax.psum(d_tok_fsdp,  ep_axis_name)
        d_topk_full = lax.psum(d_topk_fsdp, ep_axis_name)

        # 6. Slice back to this EP device's T_ep_local token range.
        T_ep_local = g_l.shape[0]
        device_ep  = lax.axis_index(ep_axis_name)
        d_tok_l  = lax.dynamic_slice(d_tok_full,  (device_ep * T_ep_local, 0), (T_ep_local, D))
        d_topk_l = lax.dynamic_slice(d_topk_full, (device_ep * T_ep_local, 0), (T_ep_local, K))

        # 7. Weight grads: d_w1_p[:,0]/[:,1] are already F_local-sharded — no scatter needed.
        return (d_tok_l.astype(g_l.dtype),
                d_topk_l.astype(fw_l.dtype),
                d_w1_p[:, 0].astype(w0_l.dtype),
                d_w1_p[:, 1].astype(w1_l.dtype),
                d_wo_p.astype(wout_l.dtype))

    d_fx, d_fw, d_w0, d_w1, d_wout = shard_map(
        _streaming_bwd_fn, mesh=mesh,
        in_specs=(act_spec, act_spec, act_spec, act_spec,
                  P("ep", None, "fsdp"), P("ep", None, "fsdp"), P("ep", "fsdp", None)),
        out_specs=(act_spec, act_spec,
                   P("ep", None, "fsdp"), P("ep", None, "fsdp"), P("ep", "fsdp", None)),
        check_rep=False,
    )(g, fx, fw, fi, w0, w1, wout)

    return (d_fx, jnp.zeros_like(fi), d_fw, d_w0, d_w1, d_wout)
```

**Why no FSDP weight all_gather**: The forward's `_expert_mlp_ep_body_ep_sharded` computes
each expert's MLP as two partial GEMMs over F_local=128 hidden units, then does a
`psum("fsdp")` on the output. The backward mirrors this: partial GEMMs with F_local=128
weights, partial d_tokens accumulation, `psum("fsdp")` combines contributions. This avoids
any ICI for weights — the only communication is the final psum on d_tokens and d_topk.

Contrast with `_moe_pallas_bwd_bwd` (Pallas backward): that kernel needs the full F=2048
in VMEM for its tiled matmuls, so it does upfront `all_gather(w0, axis=1)` and
`all_gather(wout, axis=2)` before calling the Pallas kernel.

### Step 1.2 — Local sanity check before cluster

Run a single-device forward+backward to catch dtype/shape issues before rebuilding the image:

```bash
cd ~/dsv3
python -c "
import jax, jax.numpy as jnp, sys
sys.path.insert(0, 'fused_moe_bwd')
# Check: run one train_step with use_streaming_bwd=True at small config, assert loss finite
"
```

### ⚠ Known issue: compilation hang under investigation

A compilation hang has been observed when running `use_streaming_bwd=True` at training scale.
Root cause not yet identified. Candidates:
- `lax.fori_loop` over E_local=32 experts at D=7168 inside `shard_map` inside `custom_vjp`
  triggering excessive compile-time HBM usage
- Shape mismatch causing XLA to retry compilation indefinitely
- Token EP all_gather (`g_full, fx_full`) before the loop: each (T/FSDP, D=7168) at GBS=1024
  is `(64, 7168) × 4 = 1.8 MB` — should be fine

**Triage steps** before submitting cluster job:
1. Run with `--steps=1 --gbs=64` (reduce batch) to check if compile succeeds at small scale
2. Check XLA compiler dump: `XLA_FLAGS=--xla_dump_to=/tmp/xla_dumps` and look for loops
3. Try `jax.disable_jit()` on a single-host small-scale run to rule out shape errors

### Step 1.3 — Cluster test: correctness first

Build image and submit with `--steps=3`:

```yaml
# k8s/dsv3-train-4x8x8-v39-streaming.yaml
args:
  - "--config=full"
  - "--use_streaming_bwd=true"
  - "--attn_backend=splash"
  - "--ep=8"
  - "--fsdp=16"
  - "--gbs=1024"
  - "--steps=3"
```

Check:
1. Compilation completes (no 60+ min hang)
2. `train_step` returns without AssertionError (dtype/shape)
3. Loss is finite and tracks the `jax.vjp` baseline

### Step 1.4 — Performance benchmark

Once correctness passes, run 10 steps with profiling:

```yaml
args:
  - "--steps=10"
  - "--profile"
  - "--profile_dir=gs://sivaibhav-exp/dsv3/profiles/4x8x8-ep8-fsdp16-gbs1024-streaming-v1"
  - "--profile_skip=5"
  - "--profile_steps=2"
```

Expected: `scatter_custom_fusion` ops (1.18s/step, 16% BW) shrink — replaced by per-expert
VMEM accumulation + single HBM writes.

**Success criteria**:
- Loss matches `jax.vjp` baseline (within float noise)
- Step time < v38 baseline
- Profile: no large scatter ops in MoE backward

---

## Phase 1b — Pallas forward + streaming backward (after Phase 1 passes)

**Goal**: Swap the JAX shard_map forward for the Pallas forward kernel while keeping the
streaming backward. This produces `moe_backend=fused_pallas`.

**What changes vs Phase 1**: only the forward. The backward (`_streaming_bwd_fn`) is identical.

### Mesh note — do NOT call `fused_ep_moe` top-level

`fused_ep_moe` (`tpu_inference/kernels/fused_moe/v1/kernel.py:1243`) has a hard constraint:
```python
if len(mesh.shape) != 2: raise NotImplementedError("Only 2D mesh is supported.")
for axis in mesh.axis_names:
    if axis == ep_axis_name: continue
    if mesh.shape[axis] != 1: raise NotImplementedError(...)  # fsdp=16 fails this
```

Even with a 2D `("ep", "fsdp")` training mesh, `fsdp=16` violates this. Calling
`fused_ep_moe` directly from the training context will crash.

**Fix**: call `_fused_ep_moe_kernel` (the inner Pallas kernel) directly from inside a
`shard_map` over `("ep", "fsdp")`, after `all_gather`-ing the FSDP weight slice first.
This is the same pattern `_moe_pallas_bwd_bwd` uses (`model.py:786-827`). For the forward,
follow that pattern: all_gather w0/w1/wout → full F=2048 per device → call `_fused_ep_moe_kernel`.

The backward stays as Phase 1 (sharded F_local, no weight all_gather, FSDP psum on output).

### New `custom_vjp` wrapper

Create `fused_ep_moe_train_fsdp` in `fused_moe_bwd/backward.py`:

```python
@functools.partial(jax.custom_vjp, nondiff_argnums=(...))
def fused_ep_moe_train_fsdp(tokens, w0, w1, wout, gate_logits, fi, fw, mesh, ...):
    ...

def _fwd(tokens, w0, w1, wout, gate_logits, fi, fw, mesh, ...):
    # shard_map over ("ep", "fsdp")
    # all_gather w0/w1 on axis=2 (F_local→F), all_gather wout on axis=1 (F_local→F)
    # call _fused_ep_moe_kernel directly (bypass fused_ep_moe mesh check)

def _bwd(..., res, g):
    # identical to Phase 1 _streaming_bwd_fn
    # NO weight all_gather — uses F_local-sharded weights directly
    # FSDP psum on d_tokens + d_topk
```

**Success criteria**: same as Phase 1 plus additional gain from eliminating JAX forward scatter overhead.

---

## Phase 2 — FSDP async weight prefetch (after Phase 1b passes)

**Goal**: For Phase 1b's forward, the Pallas kernel requires full F=2048 weights via
`all_gather`. This gather blocks: all of `(E_local=32, 2, D=7168, F=2048)` = 9.4 GB must
arrive before computation begins. Phase 2 hides this latency by prefetching the next expert's
full-F weights while computing the current expert.

This is relevant only when we need full-F weights — i.e., Phase 1b forward and any backward
variant that works with full F (Pallas backward kernel or v2 streaming backward).

### Memory comparison at DSv3 671B (EP=8, FSDP=16)

| Approach | Upfront weight HBM | Per-expert ICI |
|---|---|---|
| Phase 1b forward (upfront all_gather) | 9.4 GB (full F, E_local experts) | none |
| Phase 2 (per-expert all_gather in scan) | **704 MB** × 2 double-buffer | overlap with TensorCore |
| Phase 1/1b backward (sharded F, no gather) | 352 MB (F_local, E_local experts) | none |

704 MB = 2 × `(2,7168,2048)×4` + 2 × `(2048,7168)×4` = 2×235 MB + 2×117 MB.

Note: Phase 1 streaming backward already uses sharded F_local and needs NO weight all_gather.
Phase 2 is primarily for the Pallas backward in Phase 1b (which needs full F=2048 for VMEM tiling).

### Implementation: `fused_ep_moe_bwd_streaming_v2` — ✅ WRITTEN & TESTED (`backward_kernel.py:818`)

**Correctness validated**: `test_grad_check_v2.py` PASS on fsdp=4, EP=1 — all norms 1.000000.
EP=8+fsdp=64 cluster test ready to run (`test_grad_check_v2.py --distributed` on 4×8×8).

Does NOT modify v1. Key differences:

| | v1 (`fused_ep_moe_bwd_streaming`) | v2 (`fused_ep_moe_bwd_streaming_v2`) |
|---|---|---|
| Weight input | `w1[E_local, 2, D, F_local]` — already sharded | `w1[E_local, 2, D, F_local]` — sharded |
| Loop primitive | `lax.fori_loop` — no prefetch | `lax.scan` — prefetch e+1 at top of body |
| Weight gather | none — uses F_local directly | `lax.all_gather` per expert inside loop |
| d_w output | `(E_local, 2, D, F_local)` — already sharded, no collective | `dynamic_slice` — each device takes its shard, no collective |
| Required param | — | `fsdp_axis_name: str` |
| Use case | Phase 1/1b backward | Phase 2 backward with full-F weights |

**Critical design note**: after `all_gather` + full-F backward, every FSDP device holds the
**same** `d_w1_full`/`d_w2_full` (no unique partial contribution per device). A `psum_scatter`
would sum fsdp_count identical copies → fsdp_count× overcounting. Correct operation:
`lax.dynamic_slice` to extract each device's own F_shard from the identical result.

The double-buffering mechanism (corrected):
```python
# w1: (E_local, 2, D, F_shard); w2: (E_local, F_shard, D)
# Seed expert 0 before scan
w1_seed = lax.all_gather(w1[0], fsdp_axis_name, axis=2, tiled=True)   # (2, D, F)
w2_seed = lax.all_gather(w2[0], fsdp_axis_name, axis=0, tiled=True)   # (F, D)
F = w1_seed.shape[2]  # full hidden dim

def process_expert(carry, e):
    ..., w1_curr, w2_curr = carry

    # Issue next expert's gather BEFORE matmuls below — XLA pipelines ICI + TensorCore
    e_next  = (e + 1) % E_local
    w1_next = lax.all_gather(w1[e_next], fsdp_axis_name, axis=2, tiled=True)
    w2_next = lax.all_gather(w2[e_next], fsdp_axis_name, axis=0, tiled=True)

    # ... matmuls using w1_curr, w2_curr — full F=2048 hidden dim ...

    new_carry = (..., w1_next, w2_next)
    return new_carry, None

# After scan: slice (not psum_scatter!) each device's own F shard
fsdp_idx = lax.axis_index(fsdp_axis_name)
d_w1_out = lax.dynamic_slice(d_w1_full, (0, 0, 0, fsdp_idx * F_shard), (E_local, 2, D, F_shard))
d_w2_out = lax.dynamic_slice(d_w2_full, (0, fsdp_idx * F_shard, 0),    (E_local, F_shard, D))
```

### Step 2.1 — Correctness test ✅ DONE

`test_grad_check_v2.py` written and passing locally:
- Test 1 (EP=1, fsdp=4/8): v2 via `shard_map("fsdp")` vs v1 full-weight reference — **PASS**
- Test 2 (EP=8, fsdp=64): needs 4×8×8 cluster run; `--distributed` arg triggers `jax.distributed.initialize()`

```bash
# Local smoke test (8 devices):
python fused_moe_bwd/test_grad_check_v2.py

# Cluster run (add --distributed to trigger jax.distributed.initialize):
kubectl --context gke_cloud-tpu-multipod-dev_us-central1_ninja-v7x-64-spot \
    apply -f k8s/test-v2-4x8x8.yaml
```

JAX 0.9.2 note: `shard_map` uses `check_vma=False` (not `check_rep=False`).

### Step 2.2 — Integration in Phase 1b backward

For Phase 1b: the Pallas forward needs full-F weights (after all_gather). The backward can
either (a) keep Phase 1 streaming with F_local — cheapest, or (b) use v2 to also work with
full-F weights and overlap ICI with TensorCore.

Decision: defer until Phase 1b benchmark shows backward is the bottleneck.

**Prerequisite**: Phase 1b must be clean and benchmarked first.

---

## Phase 3 — SparseCore token routing (after Phase 2)

**Goal**: Replace the TensorCore-based token gather (`jnp.take` / dynamic_slice at sorted
token positions) with a SparseCore indexed gather. On v7x, SC can load indexed rows from
HBM without materializing the full gather buffer.

Design sketch: `docs/sparsecore_moe_routing_sketch.md`

**Impact**: The current backward streams 58.7 MB per expert × 32 experts = 1.9 GB of token
data from HBM per backward call. SC gathers collapse this into a single indexed read per
expert, avoiding the sort → scatter → loop pattern entirely.

**Prerequisite**: Phase 2 must be clean and benchmarked first.

Phase 3 completes `fused_ep_moe_train_fsdp`: Pallas fwd, streaming bwd, async FSDP prefetch,
SC token routing — the full end goal.

---

## File map

| File | Role |
|------|------|
| `mini_dsv3/model.py:855` | `_moe_jax_ep_fn` — custom_vjp wrapper; `nondiff_argnums` includes `use_streaming_bwd` |
| `mini_dsv3/model.py:890` | `_moe_jax_ep_fn_bwd` — ✅ Phase 1 streaming path implemented |
| `mini_dsv3/model.py:780` | `_moe_pallas_bwd_bwd` — Pallas bwd with FSDP all_gather; contrast with streaming path |
| `fused_moe_bwd/backward_kernel.py:563` | `fused_ep_moe_bwd_streaming` (v1) — Phase 1/1b backward; uses F_local, no weight gather |
| `fused_moe_bwd/backward_kernel.py:818` | `fused_ep_moe_bwd_streaming_v2` (Phase 2) — per-expert all_gather; lax.scan double-buffering |
| `fused_moe_bwd/test_grad_check_v7x.py` | v1 correctness: EP=1 + EP=8 on 4×8×8 — ✅ passing |
| `fused_moe_bwd/test_grad_check_v2.py` | v2 correctness: fsdp-only ✅ passing; EP=8+fsdp cluster pending |
| `docs/kernel_feedback_pallas_bwd.md` | v12/v15 failure analysis — read before touching custom_vjp |
| `specs/fused_ep_moe_backward_spec.md` | Weight shapes, mesh axis names, routing details |
| `k8s/dsv3-train-4x8x8-v38.yaml` | Current baseline training config to match |

---

## Critical rules (learned from v12/v15)

1. **No `@jax.jit` inside `custom_vjp` backward** — causes 60+ min nested compile hang
2. **No `jax.vmap` over T×K tokens touching weight tensors** — materializes TB-scale tensors
3. **No pre-allocated `(E_local × max_tpe, D)` buffers** — 3.84 GB × 29 scan steps = OOM
4. Process **per-expert** (E_local iterations), not per-token
5. `ep_axis_name = "ep"` in training (not `"model"` used in standalone test scripts)
6. Cast all gradient outputs to match primal dtypes — `d_fx.astype(fx.dtype)` etc.
7. **No FSDP weight all_gather in streaming backward (v1)** — kernel works on F_local directly;
   FSDP psum on d_tokens + d_topk is sufficient (mirrors forward's column-parallel pattern)
8. Use **precomputed routing** (`top_k_indices_precomputed=fi, return_dtopk=True`) — DSv3
   forward uses gate_bias + routed_scaling_factor that `compute_routing` in the kernel doesn't
   replicate; routing from residuals avoids the mismatch
9. **No `psum_scatter` for weight grads after `all_gather`** (v2-specific) — after gathering
   full-F weights and computing the full backward, every FSDP device holds the SAME gradient.
   `psum_scatter` would sum fsdp_count identical copies = fsdp_count× overcounting.
   Use `lax.dynamic_slice(d_w1_full, (0,0,0, fsdp_idx*F_shard), ...)` instead — no collective.
10. **`check_vma=False` not `check_rep=False`** in JAX 0.9.2's `shard_map` — `check_rep` does
    not exist in this version; use `check_vma=False` to suppress representation checks.
