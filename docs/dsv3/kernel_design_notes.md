# Kernel Migration Design Notes — Step A (gmm_v2) + Step B (scatter)

DSv3 671B on v7x — vendoring tpu_inference kernels into MoE training path.

## (a) Kernel APIs & Findings

### `tpu_inference.kernels.megablox.gmm_v2.gmm_v2`

**Signature** (gmm_v2.py:1120):
```python
gmm_v2(
    lhs,                # [size_m, size_k]            (tokens)
    rhs,                # [size_group, size_k, size_n] (weights per expert)
    group_sizes,        # int32[size_lhs_group]
    rhs_scale=None,     # FP8 quantization scales
    rhs_bias=None,      # FP8 quantization bias
    group_offset=None,
    *,
    tile_info=calculate_tiling,    # auto from tuned table or default
    vmem_limit_bytes=None,         # auto = 90% VMEM
    preferred_element_type=None,
    acc_dtype=None,
    maybe_quantize_lhs=True,
    zero_initialize=True,
    fuse_act=None,                 # 'silu' | 'gelu' | 'swigluoai' | None
)
```

**Key features:**
- Drop-in for `jax.lax.ragged_dot` API-wise.
- **`fuse_act='silu'` collapses gate+up+silu+multiply into ONE op**:
  - `gmm_v2(x, FusedWeightsRef(wi_0, wi_1), gs, fuse_act='silu')` returns `silu(x@wi_0) * (x@wi_1)`.
  - Saves the entire HBM round-trip for the `gate * up` intermediate (was 131072 × 2048 × bf16 = 0.5 GB per chunk).
- Triple-buffered weights internally — no extra prefetching needed.

**3 ragged-dots → 2 gmm_v2 calls per chunk:**
| current | with gmm_v2 |
|---|---|
| `gate=silu(rd(x, wi_0))` <br> `up=rd(x, wi_1)` <br> `hidden=gate*up` | `hidden = gmm_v2(x, FusedWeightsRef(wi_0, wi_1), gs, fuse_act='silu')` |
| `out = rd(hidden, wo)` | `out = gmm_v2(hidden, wo, gs)` |

**Tuned block sizes** (`megablox/tuned_block_sizes.py`):
- All 395 entries are FP8-keyed (`'float8_e4m3fn'`) — production tuning was for FP8 weights.
- bf16 weights fall to `get_default_gmm_block_sizes(...)` — kernel works but unhelpful tuning.
- **Implication**: Step A in bf16 = partial gain. Compounds with future FP8 conversion.

**No `custom_vjp`** — needs wrapping for training. See (b) below.

### `tpu_inference.kernels.sparse_core.ragged_scatter.ragged_scatter`

**Signature** (ragged_scatter.py:352):
```python
ragged_scatter(
    x,        # [num_rows, hidden_size]
    indices,  # [output_size,]
    start, end,  # scalar bounds for which indices to process
) -> output    # [output_size, hidden_size]
```

**Semantics**: `output[i] = x[indices[i]]` for `i` such that `indices[i] in [start, end)`.
This is a **gather**, not scatter-add. **Does NOT combine duplicates.**

**Per docstring**: replaces `gmm2_res[topk_argsort_revert_indices]` — the unsort/unpermute step.

**Production MoE pattern (inference)**:
1. After GMM: outputs in per-(t, k) sorted-by-expert layout, shape `(T*K, D)`.
2. `ragged_scatter` with inverse permutation → unsorted (T*K, D) in original (t, k) order.
3. `.reshape(T, K, D).sum(axis=1)` → `(T, D)`.

**Why this doesn't directly fit our EP>1 case**:
- Production assumes `(T*K, D)` intermediate (8× bigger than our `(max_local, D) = (T_all*K/EP, D)`).
- For our shape: `(T*K, D)` = `(524288, 7168) × bf16` = **7.5 GB per chunk per shard**. Too big.
- Our local-expert filtering keeps it at `(max_local, D) = (131072, 7168)` = 1.9 GB.
- Production EP=1 path doesn't have this issue.

**Step B options:**
- **B1**: Materialize (T*K, D) padded — clean code but +6 GB HBM per chunk per device. Probably blows budget.
- **B2**: Skip Step B, keep `at[].add()`. **Step A only.**
- **B3**: Use `gather_reduce` (we tried in v305) — known regression at our chunk frequency.
- **B4**: Write a per-EP-shard variant of ragged_scatter for the local-only case. Significant kernel work.

**Recommendation**: defer Step B until Step A is validated. Step A alone is the win.

### `gather_reduce` (from v305 read)

- Signature: `sc_gather_reduce(op, idx, reduce_group_size=K, single_sc=True)`.
- Constraint: `op.shape[0] % reduce_group_size == 0` (kernel constraint, can pad).
- v305 result: 5% TPS regression at our chunk frequency (~230 calls/step). Per-call overhead dominated.
- Now gated behind `--moe_use_sc_scatter` flag.

## (b) custom_vjp wrapper math for gmm_v2

### Plain gmm_v2 (no fuse_act)

`out[m, n] = sum_k (lhs[m, k] * rhs[g(m), k, n])` where g(m) = group of token m.

**Forward**: `out = gmm_v2(lhs, rhs, group_sizes)`

**Backward (cotangents `d_out`)**:
- `d_lhs[m, k] = sum_n (d_out[m, n] * rhs[g(m), k, n])`
  - = `gmm_v2(d_out, rhs.transpose(0,2,1), group_sizes)` — same kernel, transposed weights.
- `d_rhs[g, k, n] = sum_{m: g(m)=g} (lhs[m, k] * d_out[m, n])`
  - Per-group outer product. Standard megablox bwd: `gmm_v2(lhs.T_per_group, d_out, group_sizes)`.
  - The kernel handles the ragged grouping via group_sizes.

### Fused gmm_v2 (fuse_act='silu')

Forward: `out = silu(lhs @ wi_0) * (lhs @ wi_1) = silu(g) * u`, where:
- `g = gmm_v2_no_act(lhs, wi_0, group_sizes)`
- `u = gmm_v2_no_act(lhs, wi_1, group_sizes)`

Backward via chain rule with `d_out`:
- `d_g_silu = d_out * u` and `d_u = d_out * silu(g)`
- `d_g = d_g_silu * silu'(g)` where `silu'(g) = sigmoid(g) * (1 + g * (1 - sigmoid(g)))`
- `d_lhs += gmm_v2(d_g, wi_0_T, ...) + gmm_v2(d_u, wi_1_T, ...)`
- `d_wi_0 = gmm_v2(lhs_T, d_g, ...)`, `d_wi_1 = gmm_v2(lhs_T, d_u, ...)`

**Saves residuals**: must keep `g` and `u` (or `silu(g)`) for backward — same memory cost as current code that materializes `gate` and `up`. No regression.

### Approach for first cut

Strategy: **two-stage migration** to limit risk:
1. **Stage A.1**: Plain wrapper — forward uses gmm_v2, backward uses jax.lax.ragged_dot.
   - Easy bring-up: backward already works via existing _moe_gmm_ag_bwd wrapper (jax.vjp on ragged_dot).
   - Just need a custom_vjp for the gmm_v2 fwd → ragged_dot bwd direction.
   - Validates forward correctness + measures forward win.
2. **Stage A.2**: Upgrade backward to gmm_v2 (only after Stage A.1 passes).
   - Bigger code change but proven necessary if backward dominates step time.

## (c) Gate-1 AOT compile harness

To follow. Path: `~/ml-experiments/dsv3/kernels_test/aot_gmm_v2.py`.

Goals:
1. Import `gmm_v2`.
2. Build abstract inputs at DSv3 shapes:
   - lhs: `(131072, 7168)` bf16
   - rhs: `(64, 7168, 2048)` bf16
   - group_sizes: `(64,)` int32
3. AOT compile via `jax.experimental.topologies.get_topology_desc("tpu7x:4x4x4")`.
4. Confirm Mosaic compile succeeds without hitting documented constraints.
5. Get reported tile sizes / VMEM usage from the compile result.

Test cases:
- bf16 lhs + bf16 rhs (default block sizes)
- bf16 lhs + bf16 rhs + fuse_act='silu' (fused gate+up)
- bf16 lhs + FP8 rhs + scales (tuned block sizes path)

## Gate-1 AOT results (DSv3 671B shapes)

| case | result | compile | FLOPs reported |
|---|---|---:|---:|
| plain_bf16 (M=131072, K=7168, N=2048) | ✓ PASS | 1.4s | 3.85 TFLOPs |
| wo_bf16 (M=131072, K=2048, N=7168) | ✓ PASS | 1.3s | 3.85 TFLOPs |
| fused_silu_bf16 (vmem_limit=48M, N=2×2048) | ✓ PASS | 1.2s | 7.70 TFLOPs |

- Default tile picker for fused_silu picked `tile_n=768` → 69M VMEM, OOM by 5M.
  Setting `vmem_limit_bytes=48M` forces picker to smaller tiles, fits in 64M VMEM.
- FLOPs match hand calc: 2 × M × K × N for plain; 2× that for fused (gate+up).
- Iteration loop confirmed fast: AOT compile ~1.5s per change at full DSv3 shape.

## Open questions for after AOT compile passes

1. Does `gmm_v2(maybe_quantize_lhs=True)` re-quantize per call (overhead) or expects pre-quantized lhs?
2. What's the actual VMEM consumption at our shapes vs limit?
3. For bf16+bf16, do default block sizes hit reasonable MFU (~50%+) or do we need to tune?
4. For FP8 rhs path — what scale layout does it expect? (channel-wise? block-wise?)
5. For `fuse_act='silu'` — does the API accept TWO separate weight tensors or interleaved?
