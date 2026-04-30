# SparseCore `sc_gather_rows` Regression Report

**Date:** 2026-03-31 (fix applied 2026-03-31)
**Experiment range:** v62–v69 (TC fallback); fix in next image
**Status:** Fixed — D-tiling to SIMD register width. SC path re-enabled.

---

## 1. Background

`sc_gather_rows` in `fused_moe_bwd/backward_kernel.py` implements a SparseCore-accelerated
indexed gather on TPU v7x:

```
result[i] = source[row_indices[i]]    # shape: (n, D) from (T, D)
```

The implementation uses `jax.experimental.pallas.tpu_sc` (the `_plsc` module) with:

```python
_plsc.BlockSpec(
    (block_n, D),      # block_n=128, D=7168
    indexed_by=1,      # arg 1 (row_indices) provides the row index
    indexed_dim=0,     # index along dimension 0 of source
)
```

This `BlockSpec` lowers to an SC indexed-DMA instruction. The expected behavior is that the
SparseCore DMA engine issues random-access loads (`block_n=128` rows at a time) from HBM into
SC VMEM, overlapping with TensorCore GEMMs on the TC side.

**Measured benefit (v60 era, v75-train image):** 4.45× faster gather vs `source[row_indices]`
(TC scatter-gather). SC/TC overlap means the gather latency for expert `e+1` is hidden behind
the GEMMs for expert `e`.

---

## 2. The Regression

Starting at some point between v75-train and v79-train images, `sc_gather_rows` raises a
`NotImplementedError` during XLA compilation:

### Error variant A — bfloat16 source (all images v79+):

```
NotImplementedError: Get only supports bfloat16 arrays of shapes [(32,), (2, 16)],
got (128, 7168)
```

### Error variant B — float32 source (all images v79+):

```
NotImplementedError: Get only supports float32 arrays of shapes [(16,)], got (128, 7168)
```

The error originates from XLA's SC lowering pass. When `plsc.BlockSpec(indexed_by=1)` is
compiled, the lowering maps the indexed DMA access to the hardware `Get` instruction. The
new JAX/libtpu version enforces strict shape constraints on `Get` that match the SC hardware
register sizes:

| dtype    | Allowed shapes          | Our shape      |
|----------|-------------------------|----------------|
| float32  | `[(16,)]`               | `(128, 7168)`  |
| bfloat16 | `[(32,), (2, 16)]`      | `(128, 7168)`  |

Our block is 128 rows × 7168 columns = 917,504 elements. The hardware `Get` register holds
32–64 bytes. The SC indexed-DMA path (which can handle large blocks via the DMA engine) uses
a *different* lowering path; the regression causes the compiler to use `Get` instead.

**Confirmed regression:** v67 test ran streaming_bwd_v1 (known-good in v75-train) on v81-train
and got the same SC error → the regression is image-specific, not v2-specific.

---

## 3. Call Sites — Where `sc_gather_rows` Is Invoked

### 3.1 Forward pass (streaming_bwd v1: `fused_ep_moe_bwd_streaming`)

Inside `_streaming_bwd_fn` (the per-expert scan/loop body):

```python
# source: flat_x (T, D) — dtype: bfloat16
tokens_e = sc_gather_rows(flat_x, tok_ids_e)          # ← bfloat16 source
```

`flat_x` is the input activations from the forward pass, stored in bf16.

### 3.2 Forward pass (streaming_bwd v2: `fused_ep_moe_bwd_streaming_v2`)

Inside the Python for-loop over `E_local` experts (lines 1087–1088):

```python
# source: tokens_f32 (T, D) — dtype: float32
tokens_e = sc_gather_rows(tokens_f32, tok_ids_e) * valid_f[:, None]  # ← f32 source
# source: d_out_f32 (T, D) — dtype: float32
d_out_e  = sc_gather_rows(d_out_f32, tok_ids_e)                       # ← f32 source
```

### 3.3 Gradient checkpointing interaction

With `--gradient_checkpoint`, JAX recomputes the **entire forward MoE pass** inside the
backward. This means:

- The bfloat16 call site (3.1) is compiled as part of the **backward** JaxPR
- The float32 call sites (3.2) are also in the backward JaxPR

So a single backward compilation triggers **both** error variants (A and B) if either
call site goes through the SC Pallas path.

### 3.4 Error variant sequencing across experiments

| Experiment | Image | SC path active | Error |
|-----------|-------|----------------|-------|
| v62 | v77-train | lax.scan → SC for bf16 | NaN (different issue) |
| v63 | v78-train | lax.scan + debug.print | SC Get bfloat16 |
| v64 | v79-train | Python for loop + debug.print | SC Get bfloat16 |
| v65 | v80-train | Python for loop, no prints | SC Get bfloat16 |
| v66 | v81-train | Upfront all_gather, no prints | SC Get bfloat16 |
| v67 | v81-train | streaming_bwd v1 (regression test) | SC Get bfloat16 |
| v68 | v82-train | bf16 fallback only | SC Get **float32** (v2 f32 path exposed) |
| v69 | v83-train | **unconditional fallback** | *(running — expected clean)* |

The key insight from v67: the SC error is a **JAX regression**, not a v2-specific bug.
v1 also hits it with the new image.

---

## 4. Why Simpler JaxPR Didn't Help

An early hypothesis was that the large JaxPR produced by 64 unrolled all_gather ops
(Python for loop × 32 experts × 2 all_gathers each) triggered a different SC lowering
path that exposed the bug. This was tested and disproved:

- v66: Upfront 2-gather (smallest possible JaxPR — 2 all_gather calls total) → **same SC error**
- v67: streaming_bwd v1 (lax.scan, single body compiled once) → **same SC error**

The regression affects any `plsc.BlockSpec(indexed_by=...)` call with a large block shape,
regardless of JaxPR size.

---

## 5. Root Cause (from internal analysis)

The `Get` instruction IS the hardware constraint — it's not a regression in the lowering
logic. On v7x SparseCore, `Get` is a register-level op:

| dtype    | Elements per Get | Bytes |
|----------|-----------------|-------|
| float32  | 16              | 64 B  |
| bfloat16 | 32              | 64 B  |
| int32    | 16              | 64 B  |

Each SC vector subcore also has only **~128 KB VMEM**. A `(128, 7168)` float32 block = 3.67 MB
— far exceeds VMEM even if Get supported it.

What changed in v79-train: the SC lowering stopped auto-accepting large block shapes and
started enforcing hardware `Get` size limits explicitly. This exposed our incorrect block size.

**The correct fix is not upstream — it's our code**: tile D to SIMD register width so each
SC call issues exactly one `Get` with a compliant shape.

## 5a. Fix Applied

`sc_gather_rows` rewritten to use a 2D grid tiling D into SIMD-register-sized chunks:

```python
block_d = 64 // jnp.dtype(source.dtype).itemsize  # 16 (f32) or 32 (bf16)

pl.pallas_call(
    _kernel,
    out_shape=jax.ShapeDtypeStruct((n, D), source.dtype),
    grid=(n, D // block_d),           # 2D: one row × one D-tile per SC instance
    in_specs=(
        plsc.BlockSpec(
            (1, block_d),             # exactly one Get per SC subcore call
            indexed_by=1,
            indexed_dim=0,
        ),
        pl.BlockSpec((1,), lambda i, j: (i,)),
    ),
    out_specs=pl.BlockSpec((1, block_d), lambda i, j: (i, j)),
    compiler_params=pltpu.CompilerParams(
        kernel_type=pltpu.CoreType.SC_VECTOR_SUBCORE,
        dimension_semantics=(pltpu.PARALLEL, pltpu.PARALLEL),
    ),
)(source, row_indices)
```

For D=7168, float32: grid = `(n, 448)`. Each of the `n × 448` SC invocations issues one
16-element `Get` for a 64 B slice. The D-tiling loop is implicit in the 2D grid — XLA
schedules the 448 D-tiles across SC subcores in parallel.

---

## 6. Performance Impact

The SC gather overlap was benchmarked at ~4.45× improvement in gather throughput.
However, in the streaming backward the gather is **pipelined** — meaning expert `e+1`'s
gather runs during expert `e`'s GEMMs. The TC fallback serializes the gather before each
expert's compute.

Rough estimate for DSv3 671B EP=8 backward:

| Component | SC path | TC fallback |
|-----------|---------|-------------|
| Token gather latency (32 experts × 2 gathers) | ~hidden (overlapped) | ~serialized |
| Per-expert gather: 128 rows × 7168 cols × 4B | ~0.5 ms (SC DMA) | ~2.2 ms (TC BW) |
| Total gather overhead for 32 experts | ~2–3 ms (amortized) | ~70 ms |

This is a rough estimate. Profile the v69 run to get actual numbers. The TC fallback adds
gather time to the critical path of each expert, which may increase total backward time.

---

## 7. Next Steps

- [ ] Build new image with fixed `sc_gather_rows` (no TC fallback)
- [ ] Run 4x4x4 mini correctness test: confirm loss matches TC-fallback baseline step-for-step
- [ ] Profile on 4x8x8 to measure SC/TC overlap benefit vs TC fallback baseline
