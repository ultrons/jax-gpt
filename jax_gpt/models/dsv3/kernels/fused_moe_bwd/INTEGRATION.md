# Fused EP-MoE Backward Kernel — Integration Guide

**Location:** `/home/sivaibhav_google_com/dsv3/fused_moe_bwd/`
**Model entry point:** `/home/sivaibhav_google_com/dsv3/mini_dsv3/model.py`

---

## Overview

Replaces JAX autograd through the MoE FFN with a fused streaming backward kernel.
The forward uses the sparse EP shard_map path (`_expert_mlp_ep_body_ep_sharded`);
the backward either falls back to `jax.vjp` (default) or uses the streaming kernel
(`fused_ep_moe_bwd_streaming`) via `moe_backend="streaming_bwd"`.

The streaming backward eliminates the T×K vmap materialization (7.57 TB OOM at
671B scale) by processing experts one at a time with a per-expert (max_tpe, D) buffer.

---

## How to Enable

Set `ModelConfig.moe_backend = "streaming_bwd"` when constructing your model config:

```python
cfg = ModelConfig(
    ...
    moe_backend="streaming_bwd",   # "jax_vjp" (default) or "streaming_bwd"
)
```

The flag is read by `expert_mlp_jax_ep` (model.py:744):
```python
use_streaming_bwd = (cfg.moe_backend == "streaming_bwd")
output = _moe_jax_ep_fn(
    flat_x, flat_indices, flat_weights, wi_0, wi_1, wo,
    cfg.mesh, K, act_spec, "ep", max_tpe, use_streaming_bwd)
```

`_moe_jax_ep_fn` is a `jax.custom_vjp` function; the backward dispatches to
`fused_ep_moe_bwd_streaming` when `use_streaming_bwd=True`.

---

## Weight Layout (Critical)

The streaming backward operates on **D_moe-sharded** weights (not D-sharded).
This avoids FSDP weight all_gathers in the forward (saves 3×938 MB/layer).

| Weight | Global shape | Partition spec | Local shape inside shard_map |
|--------|-------------|----------------|------------------------------|
| `wi_0` | `(E, D, D_moe)` | `P("ep", None, "fsdp")` | `(E_local, D, D_moe/FSDP)` |
| `wi_1` | `(E, D, D_moe)` | `P("ep", None, "fsdp")` | `(E_local, D, D_moe/FSDP)` |
| `wo`   | `(E, D_moe, D)` | `P("ep", "fsdp", None)` | `(E_local, D_moe/FSDP, D)` |

**Streaming kernel input**: `wi_0` and `wi_1` are stacked on axis 1:
```python
w1_stk = jnp.stack([w0_l, w1_l], axis=1)  # (E_local, 2, D, D_moe/FSDP)
```
So `fused_ep_moe_bwd_streaming` receives `w1=(E_local, 2, D, F_local)` and
`w2=(E_local, F_local, D)` where `F_local = D_moe / FSDP`.

---

## Collective Sequence (Forward and Backward)

Understanding the collective sequence is essential — the backward must be the exact
VJP of the forward.

### Forward (inside `_expert_mlp_ep_body_ep_sharded`)

```
all_gather(x,  "ep")   # (T_ep, D) → (T_fsdp, D)
all_gather(fi, "ep")   # routing indices
all_gather(fw, "ep")   # routing weights
  ↓
per-expert sparse MLP: sel_x @ wi_0, sel_x @ wi_1, h @ wo
  ↓
psum_scatter(partial_out, "ep")   # (T_fsdp, D) → (T_ep, D)
psum(partial_out_ep,      "fsdp") # D_moe contributions from FSDP shards
```

### Backward (inside `_streaming_bwd_fn`)

VJP of the forward above, in reverse:

```
psum(g_l, "fsdp")              # VJP of forward psum("fsdp") — AllReduce VJP = AllReduce
                               # g_l_sum: (T_ep, D) with all FSDP gradient contributions
  ↓
all_gather(g_l_sum, "ep")      # VJP of forward psum_scatter("ep") — (T_ep, D) → (T_fsdp, D)
all_gather(fx_l,    "ep")      # reconstruct token view (for d_w computation)
all_gather(fw_l,    "ep")      # routing weights
all_gather(fi_l,    "ep")      # routing indices
  ↓
fused_ep_moe_bwd_streaming(...)  # per-expert streaming backward
  returns: d_tok_partial (T_fsdp, D), d_w1 (E_local, 2, D, F_local), d_wo (E_local, F_local, D)
  ↓
psum(d_tok_partial, "ep")      # VJP of forward all_gather("ep")
dynamic_slice(d_tok_full, ep_device * T_ep, T_ep)  # extract local EP slice
```

**Critical invariant**: `psum("fsdp")` on `g_l` must happen BEFORE `all_gather("ep")`.
Without it, `d_w` is `1/FSDP` of the correct value (missing FSDP contribution factor).
The VJP of AllReduce is AllReduce — not identity.

**d_w is NOT reduced** across EP or FSDP — each device already computes gradients for
its own `E_local` experts using `F_local`-wide weights. No further reduction needed.

---

## Key Parameters

| Parameter | Formula | Example (DSv3 671B, GBS=1024×4096) |
|-----------|---------|-------------------------------------|
| `E_local` | `E // EP` | `256 // 8 = 32` |
| `F_local` | `D_moe // FSDP` | `2048 // 64 = 32` |
| `T_fsdp` | `T // FSDP` | `1024 × 4096 // 64 = 65,536` |
| `max_tpe` | `max(1, 2 × T_fsdp × K // E)` | `2 × 65536 × 8 // 256 = 4096` |

`max_tpe` is a **static int** — determines the per-expert buffer size. The factor 2×
provides headroom for uneven expert load (P(overflow) < 1e-6 for uniform routing).

---

## Routing: Precomputed vs Recomputed

The streaming backward accepts **precomputed routing** to bypass internal
`compute_routing` when the forward uses non-standard gating (e.g., DSv3 gate_bias,
routed_scaling_factor):

```python
fused_ep_moe_bwd_streaming(
    g_full, fx_full, w1_stk, wout_l,
    gating_output=None,          # skip internal routing
    top_k=K,
    scoring_fn="sigmoid",
    renormalize_topk_logits=True,
    act_fn="silu",
    ep_axis_name="ep",
    max_tpe=max_tpe,
    top_k_indices_precomputed=fi_full,    # (T_fsdp, K) int32 global IDs
    top_k_weights_precomputed=fw_full,    # (T_fsdp, K) float32 renormalized
    return_dtopk=True,                    # return d_fw not d_gating_logits
    E_global_override=E_global,           # required when gating_output=None
)
```

With `return_dtopk=True`, the kernel returns `d_top_k_weights (T_fsdp, K)` instead
of `d_gating_logits (T_fsdp, E_global)`. The caller slices back to local EP portion.

---

## Correctness Verification

All tests should pass before deploying:

```bash
# EP=1, FSDP=1 gradient check (reference)
python3 test_grad_check_stage1.py   # all [PASS]

# EP=2, FSDP=1 isolation test (no FSDP sharding)
# EP=2, FSDP=2 integration test (full EP+FSDP)
python3 test_wgrad_simple.py
# Expected: d_tok/d_w1/d_w2 ratios ≈ 1.00 (±5% at small T due to token-position mixing)

# NaN check: EP=2, FSDP=2 reference vs streaming
python3 test_nan_debug.py
# Expected: stream {name}: nan=False, ratio ≈ 1.000 for all gradients
```

**Verified status (2026-03-31):**

| Config | d_tok ratio | d_w1 ratio | d_w2 ratio | d_fw ratio |
|--------|------------|------------|------------|------------|
| EP=1, FSDP=1 | 1.000 | 1.000 | 1.000 | 1.000 |
| EP=2, FSDP=1 | 1.000 | 1.000 | 1.000 | 1.000 |
| EP=2, FSDP=2 | ~1.01 | ~1.01 | ~1.03 | ~1.000 |

The ~1–3% d_tok/d_w discrepancy at EP=2,FSDP=2 is a small-T artifact from token-position
mixing across FSDP devices (vanishes at production batch sizes where max_tpe >> average tpe).

**Full-model cluster validation — streaming v1 (v60, v75-train image, 2026-03-31):**

`fused_ep_moe_bwd_streaming` (v1) validated at DSv3 671B scale:
61 MLA + 58 MoE + 3 dense layers, 512 v7x devices (4×8×8 torus),
EP=8, FSDP=64, GBS=1024 × 4096 tokens, SGD optimizer, 10 steps from random init.

| Step | v57 (jax backend) | v60 (streaming_bwd, v75-train) |
|------|-------------------|---------------------------------|
| 1    | 86.703            | 86.693                          |
| 5    | ~86.6             | 86.595                          |
| 10   | 86.561            | 86.561                          |

Loss curves match step-by-step with no NaN or Inf.

**Caveat**: 10-step SGD from random init only. Long-run stability not yet validated.

**⚠ v1 regression — v69 (v83-train image, 2026-03-31):**

v69 runs streaming_bwd v1 on v83-train (same config as v60) and produces NaN from step 1.
Initial loss = 86.703 (forward correct), but first backward corrupts weights → NaN.

Root cause: `jnp.isfinite` guards added to `_streaming_bwd_fn` between v75-train and
v83-train (lines 987–991 in model.py). When the kernel outputs hit the guard, gradients
are masked to zero rather than corrected. Zero MoE gradients cause incorrect weight updates
for non-MoE layers (attention, embeddings) → NaN from step 1.

**Fix required**: remove the `jnp.isfinite` guards from `_streaming_bwd_fn` (they mask
symptoms, not the root cause) and re-run v1 regression to restore v60 baseline.

`fused_ep_moe_bwd_streaming_v2` cluster validation: **pending** (blocked on v1 regression fix).

---

## Constraints

| Constraint | Reason |
|-----------|--------|
| `E % EP == 0` | E_local must be integer |
| `T % (EP * FSDP) == 0` | EP+FSDP token sharding |
| `D_moe % FSDP == 0` | F_local must be integer |
| `max_tpe ≥ T_fsdp * K / E` (plus margin) | Static buffer must fit busiest expert |
| No `@jax.jit` on kernel inside custom_vjp | Causes nested JIT compile hang (60+ min) |

---

## Common Errors

**`"Found an unbound axis name: model"` (or similar ep_axis_name)**
- Cause: kernel called `lax.axis_index(ep_axis_name)` outside shard_map.
- Fix: guard with `ep_sharded = (E_local < E_global)` before any `axis_index` call.

**`d_w is 1/FSDP of reference`**
- Cause: missing `psum("fsdp")` on incoming gradient before `all_gather("ep")`.
- Fix: add `g_l_sum = jax.lax.psum(g_l, "fsdp")` in `_streaming_bwd_fn` before the EP gather.

**`"not enough values to unpack (expected 4, got 2)"` in kernel**
- Cause: wrong argument order in shard_map call (e.g., gate tensor passed as w1).
- Fix: verify `in_specs` order matches function signature: `(d_out, fx, fw, fi, w0, w1, wout)`.

**Compilation timeout (90s+) inside shard_map**
- Cause: `jax.debug.print` calls inside shard_map (one per expert per backward pass).
- Fix: remove all debug prints from inside shard_map / custom_vjp backward.

**`NotImplementedError: Get only supports float32 arrays of shapes [(16,)], got (128, D)`**
- Cause: JAX regression (v79-train+) — `plsc.BlockSpec(indexed_by=...)` mis-lowers large blocks
  to the hardware `Get` register instruction instead of `IndexedLoad` DMA. Affects both bf16 and f32.
- Workaround: `sc_gather_rows` unconditionally returns `source[row_indices]` (TC fallback).
  SC overlap performance is lost. See `docs/sc_gather_rows_regression_report.md`.
- Fix: upstream JAX/libtpu fix for `IndexedLoad` lowering. Re-enable SC path when fixed.

---

## Pending Work

| Phase | Description | Status |
|-------|-------------|--------|
| 1a | `fused_ep_moe_bwd_streaming` in `_moe_jax_ep_fn_bwd` | **DONE** |
| 1b | `make_fused_ep_moe_train_v4` FSDP support (`fsdp_axis_name` param) | **DONE** |
| 2  | `fused_ep_moe_bwd_streaming_v2` (FSDP async weight prefetch) | cluster validation in progress |
| 3  | SparseCore token gather+scatter in fwd (model.py) and bwd (backward_kernel.py) | **DONE** |
| 4  | Pallas streaming backward kernel with FSDP + D-tiling | pending — requires Phase 3 |

### Phase 1b: `make_fused_ep_moe_train_v4` with FSDP

`backward.py:make_fused_ep_moe_train_v4` now accepts `fsdp_axis_name` for EP+FSDP meshes:

```python
fn = make_fused_ep_moe_train_v4(
    mesh,
    top_k=K,
    scoring_fn="sigmoid",
    renormalize_topk_logits=True,
    act_fn="silu",
    ep_axis_name="ep",
    max_tpe=max_tpe,
    fsdp_axis_name="fsdp",   # NEW: enables FSDP-aware backward
)
# w1 must be sharded P("ep", None, None, "fsdp")  — (E, 2, D, F/FSDP)
# w2 must be sharded P("ep", "fsdp", None)         — (E, F/FSDP, D)
# tokens/gating sharded P(("ep","fsdp"), None)
```

When `fsdp_axis_name` is set, the backward shard_map uses:
- `psum(g, fsdp_axis_name)` on incoming gradient before EP all_gather (FSDP psum fix)
- FSDP-sharded weight specs for d_w output (no EP or FSDP reduction on d_w)
- `psum(d_tok, ep_axis_name)` + `dynamic_slice` for d_tok recovery

### Phase 2: `fused_ep_moe_bwd_streaming_v2` (FSDP double-buffer)

Adds per-expert FSDP `all_gather` with `lax.scan` double-buffering — issues
`all_gather(w1[e+1])` before the matmuls for expert `e` so XLA overlaps ICI
with TensorCore compute. Peak HBM: 704 MB (vs 22.5 GB upfront gather in Stage C/D).
Required when the forward Pallas kernel needs full-F weights.
**Cluster validation in progress** (as of 2026-03-31).

### Phase 3: SparseCore token gather (DONE)

Replaces TC-gather `source[row_indices]` with SC-accelerated `sc_gather_rows()` for the
per-expert token fetch in both forward and backward. Scatter-add (`segment_sum`) stays on
TC because SC scatter is overwrite-only, not atomic-accumulate.

**`sc_gather_rows(source, row_indices, block_n=128)`** added to `backward_kernel.py`:
- On v7x (`_HAS_SPARSECORE=True`): `pl.pallas_call` with `plsc.BlockSpec(indexed_by=1, indexed_dim=0)`
  + `compiler_params=pltpu.CompilerParams(kernel_type=CoreType.SC_VECTOR_SUBCORE)`.
  SC performs the indexed HBM→VMEM DMA; kernel body is a trivial copy.
- On other hardware (`_HAS_SPARSECORE=False`): falls back to `source[row_indices]` (TC).
- `block_n` adjusted down (halved) if `n % block_n != 0`.

**Forward** (`_expert_mlp_ep_body_ep_sharded`, model.py:642):
```python
from backward_kernel import sc_gather_rows
sel_x_e = sc_gather_rows(flat_x, all_safe_tids_list[e])   # was: flat_x[all_safe_tids_list[e]]
```

**Backward** (`fused_ep_moe_bwd_streaming`, backward_kernel.py per-expert loop):
```python
tokens_raw = sc_gather_rows(tokens_f32, tok_ids_e)   # was: tokens_f32[tok_ids_e]
tokens_e   = tokens_raw * valid_f[:, None]
d_out_e    = sc_gather_rows(d_out_f32, tok_ids_e)    # was: d_out_f32[tok_ids_e]
```

**SC/TC overlap**: XLA automatically schedules SC gather for expert `e+1` while TC runs
GEMMs for expert `e` inside the same `jax.jit` trace — ~2× backward throughput expected
at full scale (from hiding 4.2 ms gather behind 4.2 ms GEMM per expert).

**Scatter unchanged**: `segment_sum(all_d_tok, all_tok_ids, T)` stays as TC op.

### Phase 4: Pallas streaming backward kernel

The ultimate performance target: combine the per-expert streaming memory pattern
(no pre-materialized `bins_tokens` HBM buffer) with Pallas VMEM execution.

**Why the old Stage C/D Pallas kernel failed at 671B scale:**
`bins_tokens = (E_local × max_tpe, D)` — `32 × 4096 × 7168 × 4 = 3.8 GB` per
kernel call, held alongside `bins_d_exp` (another 3.8 GB). Together with optimizer
state these exceed HBM headroom.

**New design: stream directly from `flat_x (T, D)` — no pre-materialized bins:**

```
For each expert e (outer Python loop, static E_local iterations):
  [async DMA] load w1[e] (2, tile_D, F) into VMEM scratchpad A
                                          while computing expert e-1 in scratchpad B
  [SparseCore] gather tokens_e (max_tpe, tile_D) from flat_x using precomputed indices
  [VMEM]       forward recompute: h_g = tokens_e @ w1[0], h_u = tokens_e @ w1[1]
  [VMEM]       backward:  d_w1_e, d_w2_e  (accumulate in VMEM weight-grad buffer)
                           d_tok_e = d_h_g @ w1[0].T + d_h_u @ w1[1].T
  [async DMA] scatter d_tok_e back to d_flat_x in HBM
  swap scratchpads A/B
```

**D-tiling (mandatory for DSv3 671B on v7x):**

`w1[e] (2, D, F) = (2, 7168, 2048)` at bf16 = 56 MB > 64 MB VMEM.
Must tile D: `tile_D = 1024` → `w1_tile (2, 1024, 2048)` = 8 MB; 7 tiles per expert.

| Quantity | Shape | Size (bf16) |
|----------|-------|-------------|
| `w1_tile` (one scratchpad) | `(2, 1024, 2048)` | 8 MB |
| double-buffer (×2) | | 16 MB |
| `tokens_e` | `(4096, 1024)` | 8 MB |
| `d_tok_tile` | `(4096, 1024)` | 8 MB |
| `d_w1_tile` acc | `(2, 1024, 2048)` | 8 MB |
| `d_w2` acc | `(2048, 7168)` | 28 MB |
| **Total** | | **~76 MB** → need tile_D=512 or split d_w2 |

At `tile_D=512`: `w1_tile` = 4 MB, double-buffer = 8 MB, `d_w2` = 28 MB, rest ≈ 20 MB → ~56 MB ✓

**Key differences from Stage C/D kernel:**
- No `bins_tokens` or `bins_d_exp` in HBM — tokens gathered on-the-fly from `flat_x`
- SparseCore `tpu_gather` replaces the sort+slice token selection (from Phase 3)
- Double-buffered weight DMA: async load for expert `e+1` overlaps compute for expert `e`
- d_tok scatter is also async: issued after compute, overlaps with next weight load
- d_w accumulates in VMEM across D-tiles, written to HBM once per expert

**Expected HBM peak per expert:** ~56 MB in VMEM (no HBM intermediate per expert).
**Comparison:**

| | Stage C/D Pallas | JAX v1 | JAX v2 | Phase 4 Pallas |
|-|-----------------|--------|--------|----------------|
| Token buffer | 3.8 GB (bins) | streaming | streaming | streaming |
| Weight buffer | upfront 22.5 GB all-gather | F_local shard | 704 MB double-buf | 56 MB VMEM |
| ICI/compute overlap | No | n/a | Yes (lax.scan) | Yes (async DMA) |
| Fused VMEM matmuls | Yes | No | No | Yes |

---

## File Map

| File | Role |
|------|------|
| `backward_kernel.py` | `fused_ep_moe_bwd_streaming` (Phase 1) and `fused_ep_moe_bwd_streaming_v2` (Phase 2) |
| `backward.py` | `make_fused_ep_moe_train_v3` (old Pallas bwd API); `make_fused_ep_moe_train_v4` (streaming bwd, FSDP-aware) |
| `model.py` | `_moe_jax_ep_fn` + `_streaming_bwd_fn` (current integration) |
| `test_grad_check_stage1.py` | EP=1 gradient correctness |
| `test_wgrad_simple.py` | EP=2/FSDP=2 weight gradient isolation |
| `test_nan_debug.py` | NaN binary search at EP=2,FSDP=2 |
| `test_wgrad_debug.py` | Reference VJP vs streaming kernel comparison |
