# SparseCore for MoE Routing — Feasibility & Implementation Sketch

**Date**: 2026-03-30
**Context**: `fused_moe_bwd/backward_kernel.py` — Pallas backward kernel for EP MoE training
**v7x spec**: 2 SparseCores/chip, 16 vector subcores/SC, SIMD width 16(F32)/32(BF16), 192 GB HBM

---

## TL;DR

Yes, it's worth it — but not yet. SparseCore (SC) is a strong fit for the token gather/scatter
operations in MoE routing (measured **4.45× faster** than TensorCore for random gathers in the JAX
docs benchmark). The key architectural win is the **TC/SC overlap**: SC gathers tokens for expert `e+1`
while TC runs the GEMM for expert `e`, hiding gather latency entirely. This belongs in the Phase 3
Pallas backward design from the start, not as a retrofit.

---

## 1. What the Docs Say (JAX Pallas SparseCore)

SparseCore is designed for:
- Highly parallel, irregular, random data access
- Gather/scatter (indexed HBM fetch/store)
- Sorting, histograms, ragged operations
- Medium-to-low compute (not matmul)

Key v7x numbers:
- 2 SC chips, 16 subcores each → 32 parallel gather streams
- Gather benchmark: **4.05 ms** (SC) vs **18.1 ms** (TC) on 1024 × 4096 × 128 — **4.45× speedup**
- SC and TC can overlap when launched inside the same `jax.jit`; the compiler schedules them

The primary API is `plsc.BlockSpec` with `indexed_by`/`indexed_dim` for pipelined gather
(overlaps HBM→VMEM DMA with index lookup), or `pltpu.sync_copy(src.at[idx_vmem], dst_vmem)` inline.

---

## 2. Operations in the Current Kernel That SC Can Accelerate

The current kernel (`fused_ep_moe_bwd` in `backward_kernel.py`) has these gather/scatter hot paths,
all currently on TensorCore:

| Step | Code | Shape (full 671B EP=1) | SC fit? |
|---|---|---|---|
| Sort token→expert | `jnp.argsort(expert_ids_flat)` | (T×K,) = (4M,) int | Good (sorting) |
| Token gather | `tokens[sorted_token_ids]` | (4M, 7168) fp32 = **120 GB** | **Best fit** |
| d_out gather | `d_out[token_ids_flat]` | (4M, 7168) fp32 = 120 GB | **Best fit** |
| Scatter to bins | `.at[bin_flat_pos].set(sorted_tokens)` | (4M, 7168) → 120 GB buffer | **Best fit** |
| Scatter to bins | `.at[bin_flat_pos].set(d_sorted_exp)` | same | **Best fit** |
| Gather from bins | `d_bins[bin_flat_pos]` | (4M, 7168) fp32 | **Best fit** |
| Gradient unsort | `segment_sum(d_sorted, token_ids, T)` | scatter-add (T×K, D) | **Partial** |

The reason these are labeled "Best fit": SC has measured 4.45× speedup on exactly this pattern —
random row gathers from large HBM tables. The (4M, 7168) gathers are accessing 4M random rows
from a table, which is the workload SC was designed for.

The scatter-add (`segment_sum`) is partial because SC scatter is an **overwrite**, not an atomic
accumulate. This one stays on TensorCore.

---

## 3. Latency Model: Why TC/SC Overlap Matters

At full 671B training scale with per-expert streaming (Phase 3 design):

```
E_local = 32, T×K/E_local ≈ 131k tokens per expert, D=7168, F=2048
```

| Op | Data | Bandwidth | Time |
|---|---|---|---|
| SC gather (tokens_e from HBM) | 131k × 7168 × 4B = 3.75 GB | ~900 GB/s | ~4.2 ms |
| TC GEMM (d_out_e @ W2_e.T) | 131k × 7168 × 2048 FLOPs | ~900 TFLOPS bf16 | ~2.1 ms |
| TC GEMM (h_e.T @ d_out_e → dW2_e) | 7168 × 2048 × 131k FLOPs | ~900 TFLOPS | ~2.1 ms |

Without SC overlap: for each expert, gather (4.2 ms) → GEMM (4.2 ms) → total **8.4 ms × 32 = 268 ms**.

With SC/TC overlap: GEMM runs while SC gathers for the next expert → **~4.2 ms × 32 = 134 ms**.

That's roughly **2× faster** backward routing, just from hiding gather latency.

Note: these are rough calculations. At current EP=8 FSDP=64 training scale, `T = 524k`, so
`T×K/E_local = 131k` — the numbers above are realistic.

---

## 4. Phase 3 Design: Where SC Fits

The Phase 3 backward kernel has this per-expert streaming structure (the fix for the 120 GB OOM):

```
for e in range(E_local):
    [A]  gather  tokens_e   = tokens[is_local_e]    # (T×K/E_local, D) from (T, D)
    [B]  GEMM    d_tokens_e = d_out_e @ W2_e.T      # TC
    [C]  GEMM    dW2_e     += tokens_e.T @ d_out_e  # TC
    [D]  scatter d_tokens  += d_tokens_e * weight_e  # TC (segment_sum)
```

SC transforms this into a pipelined loop:

```
SC gathers expert e+1 while TC computes for expert e (overlap)

iter 0:  [SC: gather tokens_0]
iter 1:  [SC: gather tokens_1]  [TC: GEMM for tokens_0]  ← OVERLAPPED
iter 2:  [SC: gather tokens_2]  [TC: GEMM for tokens_1]  ← OVERLAPPED
...
iter e:  [SC: gather tokens_e]  [TC: GEMM + dW for e-1]
final:   [TC: GEMM + dW for E_local-1]
```

The XLA compiler handles the SC/TC scheduling when both are launched inside the same `jax.jit`.
From the docs: "just put them together inside a `jax.jit`. The XLA compiler will handle their
scheduling." The benchmark shows total time is less than the sum of the two separate kernels.

---

## 5. Implementation Sketch

### 5.1 SC Mesh Setup (one-time, outside the kernel)

```python
from jax.experimental.pallas import tpu_sc as plsc
from jax.experimental.pallas import tpu as pltpu

sc_info = pltpu.get_tpu_info().sparse_core
# v7x: 2 SparseCores, 16 subcores each
vector_mesh = plsc.VectorSubcoreMesh(
    core_axis_name="sc_core",
    subcore_axis_name="sc_subcore",
)
```

### 5.2 SC Token Gather Kernel

This replaces `tokens[sorted_token_ids]` in `sort_tokens_by_expert`.

```python
@jax.jit
def sc_gather_tokens(tokens_hbm, indices_hbm):
    """Gather rows from tokens_hbm using SC.

    Args:
        tokens_hbm:  (T, D) float32 — full token table in HBM
        indices_hbm: (TK,)  int32   — sorted token IDs (one per token-expert pair)

    Returns:
        (TK, D) float32 — sorted_tokens, equivalent to tokens_hbm[indices_hbm]
    """
    TK, D = indices_hbm.shape[0], tokens_hbm.shape[1]
    # SC gathers in windows of gather_window_size rows per pipeline step.
    # gather_window_size must divide TK evenly.
    gather_window_size = 128  # tunable; matches SC SIMD width × pipeline depth

    indices_2d = indices_hbm.reshape(1, TK)  # SC expects (1, TK) shape

    @pl.kernel(
        out_shape=jax.ShapeDtypeStruct((TK, D), tokens_hbm.dtype),
        mesh=vector_mesh,
    )
    def kernel(tokens_hbm_ref, idx_hbm_ref, out_hbm_ref):
        def body(idx_vmem, out_vmem):
            # The indexed_by pattern: gather tokens_hbm[idx_vmem[0]] → out_vmem
            pltpu.sync_copy(tokens_hbm_ref.at[idx_vmem.at[0]], out_vmem)

        pltpu.emit_pipeline(
            body,
            grid=(TK // gather_window_size,),
            in_specs=[
                pl.BlockSpec((1, gather_window_size), index_map=lambda i: (0, i))
            ],
            out_specs=[
                pl.BlockSpec((gather_window_size, D), index_map=lambda i: (i, 0))
            ],
            core_axis_name="sc_subcore",
            dimension_semantics=(pltpu.PARALLEL,),
        )(idx_hbm_ref, out_hbm_ref)

    return kernel(tokens_hbm, indices_2d)
```

Key details:
- `pltpu.sync_copy(tokens_hbm_ref.at[idx_vmem.at[0]], out_vmem)` is the SC gather op
- `emit_pipeline` pipelines DMA of indices into VMEM, gather, and output DMA out
- `core_axis_name="sc_subcore"` parallelizes across 16 subcores → 16× parallel gather streams
- `gather_window_size=128`: each SC step gathers 128 tokens → tune for SC DMA granule (64 bytes)

An alternative using `plsc.BlockSpec(indexed_by=1, indexed_dim=0)` for 4-stage pipelining:

```python
@partial(
    pl.pallas_call,
    out_shape=jax.ShapeDtypeStruct((TK, D), dtype),
    grid=(TK // gather_window_size,),
    in_specs=(
        plsc.BlockSpec(
            (gather_window_size, D),
            indexed_by=1,     # use arg index 1 (the indices) to index this input
            indexed_dim=0,    # index along dimension 0 (rows)
        ),
        pl.BlockSpec((gather_window_size,), lambda i: i),  # indices block
    ),
    out_specs=pl.BlockSpec((gather_window_size, D), lambda i: (i, 0)),
    compiler_params=pltpu.CompilerParams(
        kernel_type=pltpu.CoreType.SC_VECTOR_SUBCORE,
        dimension_semantics=(pltpu.PARALLEL,),
    ),
)
def sc_gather_kernel(gathered_ref, _idx_ref, out_ref):
    # gathered_ref already contains tokens[indices] thanks to plsc.BlockSpec
    @pl.loop(0, gather_window_size)
    def _(row):
        # Optional compute (e.g., multiply by weight) — can fuse here
        out_ref.at[pl.ds(row, 1)][...] = gathered_ref.at[pl.ds(row, 1)][...]
```

This 4-stage version overlaps: indices copy-in → gather → kernel compute → output copy-out.
Fusing a per-row weight multiplication into the kernel body is free because SC compute
is cheap and this is the only op running on SC subcores.

### 5.3 SC Token Scatter (bins layout)

This replaces the two scatter-set operations:
```python
bins_tokens = jnp.zeros((E_local*max_tpe, D)).at[bin_flat_pos].set(sorted_tokens)
```

SC scatter (`pltpu.sync_copy(src_vmem, dst_hbm.at[idx_vmem.at[0]])`):

```python
@jax.jit
def sc_scatter_to_bins(sorted_tokens, bin_flat_pos, bins_shape):
    """Scatter sorted_tokens into bins layout using SC.

    SC scatter is overwrite (not accumulate). Valid here because bins are
    zero-initialized and each position is written exactly once.
    """
    TK, D = sorted_tokens.shape
    scatter_window = 128

    @pl.kernel(
        out_shape=jax.ShapeDtypeStruct(bins_shape, sorted_tokens.dtype),
        mesh=vector_mesh,
    )
    def kernel(src_hbm, idx_hbm, out_hbm):
        def body(src_vmem, idx_vmem):
            pltpu.sync_copy(src_vmem, out_hbm.at[idx_vmem.at[0]])  # scatter

        pltpu.emit_pipeline(
            body,
            grid=(TK // scatter_window,),
            in_specs=[
                pl.BlockSpec((scatter_window, D), lambda i: (i, 0)),
                pl.BlockSpec((1, scatter_window), lambda i: (0, i)),
            ],
            out_specs=[],
            core_axis_name="sc_subcore",
            dimension_semantics=(pltpu.PARALLEL,),
        )(src_hbm, idx_hbm)

    return kernel(sorted_tokens, bin_flat_pos.reshape(1, TK))
```

Important: SC scatter here is **safe** because bin positions are unique (each token-expert
pair maps to exactly one bin slot). This is not the gradient scatter-add case.

### 5.4 TC/SC Overlap: Interleaving in the Per-Expert Loop (Phase 3)

The key design change for Phase 3: the expert loop is restructured so SC gather for expert `e+1`
launches while TC GEMM for expert `e` runs. XLA schedules this automatically when both are
inside the same `jax.jit`. In a `lax.fori_loop` / `lax.scan` context this requires careful
ordering of JAX ops (SC gather must be dispatched before the TC GEMM that consumes the next result).

Conceptual structure in Phase 3 Pallas forward kernel:

```python
# Pseudocode — inside jax.jit, XLA schedules SC+TC overlap automatically
def phase3_bwd_one_expert(carry, e):
    tokens_next = sc_gather_tokens(tokens_hbm, token_ids_for_expert(e + 1))  # SC
    # While SC gathers tokens_{e+1}, TC processes expert e from tokens_curr:
    d_tok_e, d_w1_e, d_w2_e = tc_ffn_backward(carry["tokens_curr"], w1[e], w2[e])  # TC
    d_tokens = d_tokens.at[token_ids_for_expert(e)].add(d_tok_e)               # TC
    return {"tokens_curr": tokens_next, "d_tokens": d_tokens, "d_w1": d_w1_e, "d_w2": d_w2_e}

init = {"tokens_curr": sc_gather_tokens(tokens_hbm, token_ids_for_expert(0)), ...}
final = lax.fori_loop(0, E_local, phase3_bwd_one_expert, init)
```

The XLA scheduler sees the SC and TC ops are independent within each loop body
and can pipeline them across iterations.

### 5.5 SC for Sort → Histogram (replace `jnp.argsort` + `jnp.bincount`)

The sorting + bincount in `sort_tokens_by_expert` can also use SC:

- `jnp.bincount(expert_ids_flat, length=E_local)` → SC histogram (one pass over (T×K,) indices)
- `jnp.argsort(expert_ids_flat)` → SC sort (SC is faster for small-value integer sorts)

With per-expert streaming (Phase 3), the full argsort is replaced by `expert_mask_e`:

```python
# Phase 3: no global sort needed
for e in range(E_local):
    mask_e = (expert_ids_flat == e)           # boolean (T×K,), cheap TC op
    token_ids_e = jnp.where(mask_e)[0]        # indices into flat array
    tokens_e = sc_gather_tokens(tokens, token_ids_e)  # SC
```

This eliminates `argsort` entirely. SC histogram for `bincount` is still useful to compute
`expert_sizes` (needed for loop bounds), but that's a small (T×K,) → (E_local,) reduction.

---

## 6. What Stays on TensorCore

| Operation | Reason |
|---|---|
| All matrix multiplications (dX@W2, dW1, dW2) | SC can't do matmul |
| `segment_sum(d_sorted_tokens, token_ids, T)` | SC scatter is overwrite, not atomic add |
| `top_k_weights * d_out` (scaling) | Element-wise, cheap on TC |
| `apply_scoring_fn_grad` (gating backward) | Element-wise, TC |
| `lax.psum(d_tokens, ep_axis_name)` (EP>1) | Collective, ICI interconnect |

The gradient scatter-add is the one SC can't help with. It stays as `segment_sum` on TC,
or can be fused into the TC GEMM pipeline as a post-GEMM scatter step.

---

## 7. Memory Implication: SC Eliminates the 120 GB bins Pre-Allocation

In the current kernel, the bins_tokens buffer is allocated upfront:
```python
bins_tokens = jnp.zeros((E_local * max_tpe, D))  # (4M, 7168) = 120 GB at full scale
```

With SC per-expert streaming (Phase 3 design), this buffer is **never materialized**:

```
Current (TC):  allocate bins(T×K, D) = 120 GB → fill via scatter → process
Phase 3 (SC):  for e in E_local: SC-gather tokens_e on demand → process → SC-scatter d_tokens_e
```

SC becomes both the enabler of Phase 3 (per-expert streaming) AND the hardware that makes
per-expert gathers fast. Without SC, each `tokens[token_ids_e]` gather is a slow TC `jnp.take`
call; with SC it's a hardware-accelerated random HBM fetch.

---

## 8. Implementation Difficulty

| Component | Complexity | Risk |
|---|---|---|
| SC gather kernel (token fetch) | Medium | Low — well-documented API, 1-to-1 with JAX example |
| SC scatter kernel (bins fill) | Medium | Low — same pattern, overwrite is safe |
| TC/SC loop overlap | Medium | Medium — requires careful op ordering inside lax.scan |
| SC sort/histogram | Low | Low — drop-in for argsort/bincount |
| Gradient scatter-add via SC | High | High — no atomic SC scatter, needs workaround |
| D-tiling within SC gather | Medium | Medium — D=7168 needs tiling within SC VMEM |

D-tiling in SC gather: with D=7168 and SC VMEM limited, we need to tile along D too.
`gather_window_size × D × dtype_size` must fit in SC subcore VMEM (each subcore has
its own VMEM; shared VMEM is "SPMEM" in docs). For v7x with 16 subcores per SC chip,
the per-subcore VMEM is much smaller than TC VMEM (16 MB). Tile D into 128 or 256 chunks
and run multiple gather passes per token window.

---

## 9. Recommendation and Sequencing

**Should we implement SC?** Yes, but in Phase 3, not as a standalone Phase 2.5.

**Sequencing:**
1. **Phase 3 (current priority)**: Get `_bwd_dX_kernel` + `_bwd_dW_kernel` working with
   per-expert streaming on TC only (no SC). This fixes the 120 GB OOM and validates correctness.
   SC gather can initially be `jnp.take` — correct, just slower.

2. **Phase 3 + SC (optimization pass)**: Replace `jnp.take` in the per-expert gather
   with `sc_gather_tokens`. Enable TC/SC overlap in the expert loop. Measure speedup.
   Target: reduce per-expert routing overhead from ~4 ms to ~0 ms (hidden by TC GEMM).

3. **Future**: SC histogram for `bincount`, SC sort elimination, SC scatter for bins layout.

**Do NOT** implement SC into the current Stage C/D kernel — it materializes the full (T×K, D)
bins buffer which is the OOM problem. SC can't fix that; per-expert streaming fixes it.

The right mental model: **Phase 3 = correctness fix (per-expert streaming); SC = Phase 3 performance optimization that makes the per-expert gather as fast as the TC GEMM**.

---

## 10. Detailed Pseudocode: Current Kernel vs SC Design

### Configuration (EP=8, FSDP=64, per JAX device)

```
T        = 524,288   tokens on this EP device (= GBS × S / EP = 1024 × 4096 / 8)
K        = 8         top-k experts per token
E        = 256       total experts
E_local  = 32        local experts (= E / EP)
TK       = T × K   = 4,194,304   all routing events (includes non-local with weight=0)
TK_local = T × K × E_local/E
         = 524,288   routing events for local experts only
tpe      = TK_local / E_local
         = 16,384    avg tokens per local expert  (= T × K / E)
D        = 7168      hidden size
F        = 2048      intermediate size
```

Note: the developer brief says "T = GBS×S / (EP×FSDP)" with math "= 524,288" — the
formula is wrong (1024×4096/512 = 8,192), but the number is right: T = GBS×S/EP = 524,288.

---

### 10A. Current Kernel (Stage C/D) — annotated pseudocode

```
Inputs (per device, after A2A receive from forward pass):
  tokens         : (T, D)         = (524,288, 7168)  fp32  =  15.0 GB
  d_out          : (T, D)         = (524,288, 7168)  fp32  =  15.0 GB   ← received via reverse A2A
  w1             : (E_local, 2, D, F) = (32, 2, 7168, 2048) fp32 = 3.69 GB
  w2             : (E_local, F, D)    = (32, 2048, 7168)    fp32 = 1.84 GB
  gating_output  : (T, E)         = (524,288, 256)   fp32  =   0.54 GB

─────────────────────────────────────────────────────────────────────────
STEP 1 — Routing backward [TC, cheap]
─────────────────────────────────────────────────────────────────────────
  scores           = softmax(gating_output)     (T, E)   = (524,288, 256)   fp32
  top_k_indices    = argtop_k(scores, K)        (T, K)   = (524,288, 8)     int32  =   16 MB
  top_k_weights    = scores[top_k_indices]      (T, K)   = (524,288, 8)     fp32   =   16 MB
  d_scores[top_k] += d_out × top_k_weights (routing backward, elementwise)
  d_gating         = scores × (d_scores - Σ(d_scores × scores)) * (renorm)
                                                (T, E)   = (524,288, 256)   fp32
  Memory: ~600 MB read + write.  FLOPs: ~2 GFLOPs.  Time: ~0.3 ms (memory-bound)

─────────────────────────────────────────────────────────────────────────
STEP 2 — Sort tokens by local expert [TC, HOT PATH]
─────────────────────────────────────────────────────────────────────────
  # Remap non-local expert IDs to 0, zero weights
  top_k_indices_local = where(is_local, top_k_indices - expert_offset, 0)  (T, K)
  top_k_weights_local = where(is_local, top_k_weights, 0.0)                (T, K)

  expert_ids_flat  = top_k_indices_local.reshape(TK)   (4,194,304,)  int32  =  16 MB
  token_ids_flat   = repeat(arange(T), K)               (4,194,304,)  int32  =  16 MB

  sort_order       = argsort(expert_ids_flat)            (4,194,304,)  int32  =  16 MB
    ↑ Sort of 4M int32 values on TC
    FLOPs: O(N log N) ≈ 92 MFLOPs  Bytes: 32 MB
    Time: ~5–15 ms  (TC argsort is slow for irregular access)

  sorted_token_ids = token_ids_flat[sort_order]          (4,194,304,)  int32  =  16 MB
  sorted_tokens    = tokens[sorted_token_ids]            (4,194,304, 7168)  fp32  ← ⚠ OOM
    Shape:  4,194,304 × 7168 × 4 bytes = 120 GB
    Reads:  524,288 unique rows (T=524,288) in random order = 15 GB unique data
    Writes: 120 GB to HBM
    Effective TC BW for large-table random gather: ~150 GB/s
    Time (if it fit): 120 GB / 150 GB/s ≈ 800 ms   ← ⚠ both OOM and slow

  d_exp_flat = d_out[token_ids_flat] * top_k_weights_local[token_ids_flat, k_ids_flat]
                                                        (4,194,304, 7168)  fp32  ← ⚠ OOM
    Shape: same 120 GB

  bins_tokens = zeros(E_local × max_tpe, D)             (4,194,304, 7168)  fp32  ← ⚠ OOM
             .at[bin_flat_pos].set(sorted_tokens)
    max_tpe  = ceil(TK / bte) * bte / E_local = TK / E_local = 131,072
    Shape:   32 × 131,072 × 7168 × 4 = 120 GB

  bins_d_exp  = zeros(E_local × max_tpe, D)             (4,194,304, 7168)  fp32  ← ⚠ OOM
             .at[bin_flat_pos].set(d_exp_flat)                                    ← 120 GB

  ⚠ TOTAL HBM for bins alone: 120 + 120 = 240 GB > 192 GB chip HBM
  ⚠ Including tokens (15 GB) + d_out (15 GB): ~270 GB — impossible to fit.

─────────────────────────────────────────────────────────────────────────
STEP 3 — Pallas GEMM kernel (E_local=32 experts, TC) [unreachable at full scale]
─────────────────────────────────────────────────────────────────────────
  Input:  bins_tokens  (4,194,304, 7168)   fp32   120 GB  (expert-sorted tokens)
          bins_d_exp   (4,194,304, 7168)   fp32   120 GB  (expert-sorted d_out×weight)
          w1           (32, 2, 7168, 2048) fp32   3.7 GB
          w2           (32, 2048, 7168)    fp32   1.8 GB

  For each expert e in [0, E_local=32):  # bte tokens at a time, tiled over D
    tok_start = e × max_tpe              # = e × 131,072
    tpe_e     = expert_sizes[e]          # actual tokens for expert e ≈ 16,384
    # (but max_tpe=131,072 slots are allocated; most are zero for non-local)

    # Load token block: (bte, tile_D) from bins_tokens[tok_start : tok_start+bte]
    # Load weight tile: W1_gate[tile_D, F], W1_up[tile_D, F], W2[F, tile_D]
    # (tile_D = 1024 for v7x, 7 tiles total for D=7168)

    For each D-tile d in [0, 7):    # 7 tiles of tile_D=1024
      For each token batch b in [0, max_tpe/bte):   # many batches, most zero-weight

        # ── Forward recompute (SwiGLU) ──
        h_gate = tok_buf @ W1_gate[d]     (bte, F)    GEMM: bte × tile_D × F FLOPs
        h_up   = tok_buf @ W1_up[d]       (bte, F)    GEMM: bte × tile_D × F FLOPs
        out    = silu(h_gate) * h_up      (bte, F)    elementwise

        # ── Backward through FFN ──
        d_h    = dexp_buf @ W2[d].T       (bte, F)    GEMM: bte × tile_D × F
        dW2[d] += out.T @ dexp_buf        (F, tile_D) GEMM: F × bte × tile_D
        d_gate = silu_grad(h_gate) * h_up * d_h  (bte, F)
        d_up   = silu(h_gate) * d_h       (bte, F)
        dtok   = d_gate @ W1_gate[d].T +
                 d_up   @ W1_up[d].T      (bte, tile_D) GEMM × 2
        dW1_gate[d] += tok_buf.T @ d_gate (tile_D, F) GEMM
        dW1_up[d]   += tok_buf.T @ d_up   (tile_D, F) GEMM

    # Accumulated over all batches and D-tiles:
    # d_bins_tokens for expert e ← dtok for all bte batches   (max_tpe, D) per expert
    # dW1_gate_e, dW1_up_e, dW2_e ← accumulated               (D, F), (D, F), (F, D)

  Output:
    d_bins_tokens  (4,194,304, 7168)   fp32  = 120 GB   ← written to HBM
    d_w1           (32, 2, 7168, 2048) fp32  = 3.7 GB
    d_w2           (32, 2048, 7168)    fp32  = 1.8 GB

  GEMMs (per expert, tpe = 16,384 actual tokens, 131,072 allocated, 7/8 zero-weight):
    bwd-W2:   (F=2048, tpe) @ (tpe, D=7168)     481 GFLOPs  (but wastes 7/8 on zeros)
    bwd-dh:   (tpe, D=7168) @ (D, F=2048)       481 GFLOPs
    bwd-dx×2: (tpe, F=2048) @ (F, D=7168) × 2   962 GFLOPs
    bwd-W1×2: (D=7168, tpe) @ (tpe, F=2048) × 2 962 GFLOPs
    Total:    2,886 GFLOPs per expert (but 7/8 wasted on zero-weight slots)
    Effective: 2,886 × 1/8 = 361 GFLOPs per expert (for actual tpe=16,384 tokens)
  For 32 experts: 32 × 2,886 = 92.4 TFLOPs gross (11.5 TFLOPs net useful work)

─────────────────────────────────────────────────────────────────────────
STEP 4 — Unsort d_tokens [TC, HOT PATH, OOM]
─────────────────────────────────────────────────────────────────────────
  d_sorted_tokens = d_bins_tokens[bin_flat_pos]       (4,194,304, 7168)  fp32  ← ⚠ OOM
    Random gather from 120 GB buffer
  d_tokens = segment_sum(d_sorted_tokens, sorted_token_ids, T)
                                                       (524,288, 7168)   fp32  =  15 GB
    Scatter-add: (4,194,304, 7168) → (524,288, 7168)
    Memory: 120 GB read + 15 GB write = 135 GB
    Time (if fit): ~900 ms at 150 GB/s effective TC BW for scatter-add

─────────────────────────────────────────────────────────────────────────
STEP 5 — EP>1 psum [ICI]
─────────────────────────────────────────────────────────────────────────
  d_tokens = lax.psum(d_tokens, ep_axis_name)        (T, D) all-reduce across EP=8
    Volume per device: T × D × 4 = 15 GB  (all-reduce = 2×(N-1)/N × vol = 26 GB ICI traffic)

─────────────────────────────────────────────────────────────────────────
Memory Summary (current kernel, peak simultaneous HBM usage):
─────────────────────────────────────────────────────────────────────────
  tokens         15 GB   (must be live from forward)
  d_out          15 GB   (received from reverse A2A)
  w1 + w2         5.5 GB
  bins_tokens   120 GB   ← ⚠ OOM alone
  bins_d_exp    120 GB   ← ⚠ OOM alone
  d_bins        120 GB   ← ⚠ OOM alone
  sorted_tokens 120 GB   ← ⚠ OOM alone
  d_tokens       15 GB
  ─────────────────────
  Total peak:  ~531 GB   — requires ~2.8× the 192 GB v7x HBM
```

---

### 10B. Phase 3 + SC Design — annotated pseudocode

Per-expert streaming: never materialize the full (TK, D) buffers.
SC gathers only the actual local tokens (TK_local = 524,288, not TK = 4,194,304).

```
Inputs: same as above (tokens 15 GB, d_out 15 GB, w1 3.7 GB, w2 1.8 GB, gating 0.54 GB)

─────────────────────────────────────────────────────────────────────────
STEP 1 — Routing backward [TC, identical to current]
─────────────────────────────────────────────────────────────────────────
  (same as current)   ~0.3 ms

─────────────────────────────────────────────────────────────────────────
STEP 2 — Compute local routing structure [TC, smaller than current]
─────────────────────────────────────────────────────────────────────────
  # Only consider local assignments (not all TK = 4M)
  is_local_mask     = (top_k_indices >= offset) & (< offset + E_local)
                                                   (T, K)  bool   =   4 MB
  local_expert_ids  = top_k_indices[is_local_mask] - offset
                                                   (TK_local,)  int32 = (524,288,)  =  2 MB
  local_token_ids   = token_positions[is_local_mask]
                                                   (524,288,)   int32               =  2 MB
  local_weights     = top_k_weights[is_local_mask] (524,288,)   fp32                =  2 MB

  # Sort only the LOCAL token-expert pairs (8× fewer than current)
  sort_order_local  = argsort(local_expert_ids)     (524,288,)   int32              =  2 MB
    Sort of 524,288 int32 values in [0, 32) on TC
    FLOPs: ~11 MFLOPs   Bytes: 4 MB
    Time: ~0.5 ms   (8× smaller sort than current)

  sorted_local_ids  = local_token_ids[sort_order_local]   (524,288,)  int32         =  2 MB
  sorted_weights    = local_weights[sort_order_local]      (524,288,)  fp32          =  2 MB
  expert_starts     = cumsum(bincount(local_expert_ids, 32)) (32,) int32
  expert_sizes_e    = bincount(local_expert_ids, 32)         (32,) int32

  Memory: ~20 MB total.   Time: ~0.5 ms

─────────────────────────────────────────────────────────────────────────
STEP 3 — Per-expert SC gather + TC GEMM (PIPELINED, SC overlaps TC)
─────────────────────────────────────────────────────────────────────────
  d_tokens = zeros(T, D)    (524,288, 7168) fp32 =  15 GB   (accumulation buffer)

  # Expert 0: prime the SC pipeline
  token_ids_0 = sorted_local_ids[expert_starts[0] : expert_starts[0]+tpe_0]
              shape: (tpe_0,)  ≈ (16,384,)  int32   =  64 KB
  tokens_0  [SC gather] = tokens[token_ids_0]       (tpe_0, D) = (16,384, 7168)  fp32  = 470 MB
  d_out_0   [SC gather] = d_out[token_ids_0]        (tpe_0, D) = (16,384, 7168)  fp32  = 470 MB
    SC reads 16,384 rows from (524,288, 7168) table — random access
    SC time: 940 MB / 500 GB/s ≈ 1.9 ms per expert

  for e in range(E_local=32):

    ┌─── SC (runs concurrently with TC below, XLA schedules) ─────────────────────────┐
    │ token_ids_{e+1} = sorted_local_ids[expert_starts[e+1] : ...]                    │
    │ tokens_{e+1}  [SC gather] = tokens[token_ids_{e+1}]   (tpe, D)  =  470 MB read │
    │ d_out_{e+1}   [SC gather] = d_out[token_ids_{e+1}]    (tpe, D)  =  470 MB read │
    │ SC time: 940 MB / 500 GB/s ≈ 1.9 ms                                             │
    └─────────────────────────────────────────────────────────────────────────────────┘

    ┌─── TC (runs concurrently with SC above) ────────────────────────────────────────┐
    │ # Load weights for expert e  (fits in 64 MB VMEM with tile_D=1024, 7 tiles)    │
    │ w1_gate_e  (D, F)  = w1[e, 0]  = (7168, 2048) fp32  = 59 MB                   │
    │ w1_up_e    (D, F)  = w1[e, 1]  = (7168, 2048) fp32  = 59 MB                   │
    │ w2_e       (F, D)  = w2[e]     = (2048, 7168) fp32  = 59 MB                   │
    │ Total weights per expert: 177 MB (streamed with D-tiling)                      │
    │                                                                                  │
    │ # SwiGLU backward (6 GEMMs per expert)                                         │
    │ h_gate_e  = tokens_e @ w1_gate_e  (tpe, D)@(D, F) = (tpe, F)    481 GFLOPs    │
    │ h_up_e    = tokens_e @ w1_up_e    same                             481 GFLOPs   │
    │ d_h_e     = d_out_e  @ w2_e.T    (tpe, D)@(D, F) = (tpe, F)    481 GFLOPs     │
    │ d_gate_e  = silu_grad(h_gate_e) * h_up_e * d_h_e    (tpe, F)  elementwise     │
    │ d_up_e    = silu(h_gate_e) * d_h_e                   (tpe, F)  elementwise     │
    │ d_tok_e   = d_gate_e @ w1_gate_e.T +                                           │
    │             d_up_e   @ w1_up_e.T   (tpe, F)@(F, D) → (tpe, D) 962 GFLOPs     │
    │ dw2_e    += out_e.T @ d_out_e      (F, tpe)@(tpe, D)→ (F, D)   481 GFLOPs     │
    │ dw1_gate_e += tokens_e.T @ d_gate  (D, tpe)@(tpe, F)→ (D, F)   481 GFLOPs     │
    │ dw1_up_e   += tokens_e.T @ d_up_e  same                          481 GFLOPs     │
    │ Total: 3,848 GFLOPs per expert (ALL on actual tpe=16,384 tokens, no wasted ops)│
    │ TC time (compute-bound): 3,848 GFLOPs / 459 TFLOPS ≈ 8.4 ms per expert        │
    └─────────────────────────────────────────────────────────────────────────────────┘

    # TC scatter-add (NOT SC — SC has no atomic accumulate)
    d_tokens[token_ids_e] += d_tok_e * sorted_weights_e   (tpe, D) scatter-add
      Writes: 470 MB scattered into (524,288, 7168) buffer  (random write)
      TC time: 470 MB / 150 GB/s ≈ 3.1 ms per expert

    tokens_e, d_out_e = tokens_{e+1}, d_out_{e+1}   # swap in SC-gathered next batch

  Peak HBM during loop: tokens_e (470 MB) + d_out_e (470 MB) + next batch (940 MB)
                        + w1_e/w2_e in VMEM (177 MB) + d_tokens (15 GB) + d_w (5.5 GB)
                      ≈ 22 GB   — fits comfortably within 192 GB ✓

  TC bottleneck per expert: max(GEMM=8.4, scatter=3.1) = 8.4 ms (GEMM dominates)
  SC gather per expert: 1.9 ms  ← fully hidden behind TC (8.4 >> 1.9)
  Pipeline efficiency: 8.4 / 8.4 = 100%  (SC gather never stalls TC)

─────────────────────────────────────────────────────────────────────────
STEP 4 — EP>1 psum [ICI, identical to current]
─────────────────────────────────────────────────────────────────────────
  d_tokens = lax.psum(d_tokens, ep_axis_name)   all-reduce over EP=8
    Volume per device: 15 GB.  ICI traffic: 26 GB/device.

─────────────────────────────────────────────────────────────────────────
Memory Summary (Phase 3 + SC, peak simultaneous HBM usage):
─────────────────────────────────────────────────────────────────────────
  tokens              15 GB   (must be live from forward)
  d_out               15 GB   (received from reverse A2A)
  w1 + w2              5.5 GB
  d_tokens (accum)    15 GB
  d_w1 + d_w2          5.5 GB
  tokens_e (current)   0.47 GB  (SC-gathered, in HBM scratch)
  d_out_e (current)    0.47 GB
  tokens_{e+1} (next)  0.47 GB  (SC prefetch)
  d_out_{e+1} (next)   0.47 GB
  ─────────────────────
  Total peak:  ~58 GB   — fits easily in 192 GB v7x HBM ✓
```

---

## 11. Roofline Performance Analysis — EP=8, FSDP=64, v7x

### Hardware Assumptions (v7x per JAX device = 1 TensorCore)

```
BF16 peak (TC):          459 TFLOPS    (= 918 TFLOPS/chip ÷ 2 TCs per chip)
HBM bandwidth (TC):      1,800 GB/s    (= 3,600 GB/s/chip, shared between 2 TCs)
HBM bandwidth (SC):        900 GB/s    (SC has dedicated HBM path, ~half of chip BW
                                        conservative for large-table random access)
ICI bandwidth:           200 GB/s (100 GB/s/link × 2 links/axis; A2A BW = 2B/(max(a,b,c)/4),
                         B=200 GB/s, EP=8 1D axis: 2×200/(8/4) = 200 GB/s)
Ridge point (TC):        255 FLOPs/byte  (= 459e12 / 1,800e9)
```

> ⚠ v7x specs are not fully public. The 918 TFLOPS/chip may be conservative if v7x is
> higher than v5p. All times below scale linearly with actual hardware parameters.

---

### 11.1 Operation-by-Operation Breakdown

#### Routing operations

| Operation | Shape | FLOPs | Bytes | AI (F/B) | Bound | Kernel | Time (est) |
|---|---|---|---|---|---|---|---|
| top-k scoring (softmax/sigmoid) | (524,288, 256) | 2 GF | 600 MB | ~3 | Memory | TC | 0.3 ms |
| argsort (current: TK=4M) | (4,194,304,) | 92 MF | 32 MB | ~3 | Memory | TC | 10–20 ms |
| argsort (Phase 3: TK_local=524K) | (524,288,) | 11 MF | 4 MB | ~3 | Memory | TC | 1–2 ms |
| sort 8× smaller in Phase 3 | | | | | | | **8× faster** |

#### Token gather (the SC opportunity)

| Operation | Reads | Writes | Total | TC time | SC time | Speedup |
|---|---|---|---|---|---|---|
| Current: sorted_tokens = tokens[sorted_token_ids] | 15 GB (random) | 120 GB | 135 GB | ~900 ms | — | — |
| Phase3: tokens_e = tokens[token_ids_e] per expert | 470 MB (random) | 470 MB | 940 MB | 6.3 ms | 1.9 ms | **3.3×** |
| Phase3 × 32 experts (SC, pipelined) | | | | 201 ms | **hidden** | — |

TC gather effective BW assumed: 150 GB/s (large-table random access, no L2 reuse)
SC gather effective BW assumed: 500 GB/s (SC designed for random HBM access)

#### SwiGLU backward GEMMs (per expert, tpe=16,384)

| GEMM | Dims (M,K,N) | FLOPs | AI (F/B, BF16) | Bound |
|---|---|---|---|---|
| dh  = d_out @ W2.T | 16384 × 7168 × 2048 | 481 GF | 1208 | **Compute** |
| dW2 += out.T @ d_out | 2048 × 16384 × 7168 | 481 GF | 725 | **Compute** |
| dx_gate = d_gate @ W1g.T | 16384 × 2048 × 7168 | 481 GF | 1208 | **Compute** |
| dx_up = d_up @ W1u.T | 16384 × 2048 × 7168 | 481 GF | 1208 | **Compute** |
| dW1g += x.T @ d_gate | 7168 × 16384 × 2048 | 481 GF | 725 | **Compute** |
| dW1u += x.T @ d_up | 7168 × 16384 × 2048 | 481 GF | 725 | **Compute** |
| **Total per expert** | | **2,886 GF** | **~1,000 avg** | **Compute** |

All GEMMs are compute-bound (AI ≫ ridge point 255 F/B). ✓

Time per expert: 2,886 GFLOPs / 459 TFLOPS = **6.3 ms**
For 32 experts: 32 × 6.3 = **201 ms**

Note: current kernel wastes 7/8 GEMM FLOPs on zero-weight non-local tokens.
Phase 3 operates only on tpe=16,384 actual tokens (8× better GEMM utilization).
Gross FLOPs current: 32 × 8 × 2,886 = 739 TFLOPs (but 7/8 wasted).
Net FLOPs Phase 3: 32 × 2,886 = 92.4 TFLOPs.

#### Scatter-add (d_tokens unsort)

| Operation | Data | TC time | SC support |
|---|---|---|---|
| Current: d_bins[4M, 7168] gather + segment_sum | 135 GB | ~900 ms | No (atomic) |
| Phase 3: d_tokens[token_ids_e] += per expert | 32 × 470 MB = 15 GB scattered | 99 ms | No (atomic) |

Scatter-add stays on TC in both cases. Phase 3 is faster because it scatters only
TK_local = 524,288 entries (not TK = 4,194,304).

#### ICI (A2A / psum)

| Operation | Volume per device | ICI traffic | Time (at 200 GB/s A2A BW) |
|---|---|---|---|
| Forward A2A (tokens → expert devices) | TK_local × D × 2 = 7.5 GB BF16 | 15 GB | **37.5 ms** |
| Backward A2A (d_out → token owners) | same | 15 GB | **37.5 ms** |
| EP psum (d_tokens) | T × D × 4 = 15 GB | 2×(EP-1)/EP × 15 = 26 GB | **~131 ms** |
| **Total ICI per MoE layer** | | **~56 GB** | **~206 ms** |

A2A BW = 2B/(max(a,b,c)/4) = 2×200/(8/4) = **200 GB/s** for EP=8 on 1D axis.
ICI (~206 ms) ≈ GEMM (~201 ms): ICI is now confirmed as the dominant bottleneck.

---

### 11.2 Roofline Summary: Current vs Phase 3 + SC

```
                        Current kernel          Phase 3 + SC kernel
                        (Stage C/D)             (proposed)
                        ────────────────────    ───────────────────────────
MEMORY STATUS           ❌ OOM (531 GB req)     ✓ ~58 GB peak

Sort (TC)               10–20 ms (4M sort)      1–2 ms  (8× smaller sort)

Token gather            ~900 ms (TC, 135 GB)    hidden (SC, 1.9 ms/expert
                        ⚠ OOM before running    pipelined behind GEMM)

GEMM (TC)               ~1,600 ms gross         201 ms
  (32 experts)          (7/8 wasted FLOPs,      (all FLOPs on real tokens,
                         effectively ~200 ms)    compute-bound ✓)

Scatter-add (TC)        ~900 ms (TC, 135 GB)    99 ms
                        ⚠ OOM before running

ICI A2A + psum          ~206 ms (same)          ~206 ms

Binning overhead        800 ms (TC scatter OOM) 0 ms (no bins buffer)

────────────────────────────────────────────────────────────────────────
Total (excl ICI)        ❌ OOM                  ~302 ms (201 + 99 + 2)
Total (incl ICI)        ❌ OOM                  ~508 ms
────────────────────────────────────────────────────────────────────────
```

**Key takeaways:**

1. **Phase 3 is not optional** — the current kernel cannot run at this scale. The OOM
   is fundamental (531 GB required, 192 GB available). Per-expert streaming is required.

2. **SC hides the gather** — per-expert gather (1.9 ms) is fully hidden behind GEMM
   (6.3 ms). Without SC, gather adds 2.0 ms/expert overhead = +64 ms total.
   SC saves ~64 ms (30% of compute time) on Phase 3 kernel.

3. **GEMM is the bottleneck** at Phase 3, all compute-bound at AI ≈ 1,000 FLOPs/byte
   (4× above ridge point 255). No memory-bandwidth bottleneck for GEMMs.

4. **Scatter-add is the next target** — 99 ms, memory-bound, no SC support. Fusing
   d_tokens accumulation into the GEMM tile loop (stay in VMEM) could reduce this.

5. **ICI dominates** at ~206 ms — ICI slightly exceeds GEMM (201 ms) and is the
   single largest contributor to per-layer latency. Overlapping A2A with the per-expert
   GEMM loop (async collective issued before the loop, completes during compute) would
   roughly halve effective ICI cost and is the highest-leverage optimization after Phase 3.

6. **GEMM utilization**: current kernel wastes 7/8 of GEMM FLOPs on zero-weight tokens.
   Phase 3 eliminates this waste entirely — equivalent to a free 8× GEMM efficiency gain.

### 11.3 Sensitivity to v7x Actual Specs

The roofline ratios (not absolute times) are hardware-independent:

| Ratio | Value | Meaning |
|---|---|---|
| GEMM AI / ridge point | ~4× | GEMMs are comfortably compute-bound |
| SC gather time / GEMM time | 1.9 / 6.3 = 0.30 | SC fully hidden even with 3× slower SC |
| Scatter-add time / GEMM time | 99 / 201 = 0.49 | Scatter-add = ~50% of GEMM time |
| ICI time / GEMM time | ~206 / 201 = 1.02 | ICI ≥ GEMM — A2A is the bottleneck, overlap critical |

These ratios hold as long as:
- GEMMs remain compute-bound (AI > ridge point) — true for any v7x variant
- SC gather is ≥ 2× faster than TC gather — confirmed by 4.45× benchmark
