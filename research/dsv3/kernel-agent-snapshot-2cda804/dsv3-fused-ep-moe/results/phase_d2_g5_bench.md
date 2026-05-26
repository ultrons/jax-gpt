---
slug: phase-d2-g5-bench-results
intent: results
status: snapshot 2026-05-12 (post-D.2)
sources:
  - targets/dsv3-fused-ep-moe/bench.py
  - targets/dsv3-fused-ep-moe/build/v_outside/expert_ffn.py (D.2: grid path)
  - targets/dsv3-fused-ep-moe/build/v_outside/moe_block_vjp.py
device: local TPU v7x (4 chips × 1 core = 4 jax devices)
---

# Phase D.2 — G5 perf signal after single-grid forward kernel

Re-run of the Phase C G5 bench after D.2 replaced the per-tile
`pallas_call` loop (B.1) with one `pallas_call(grid=(num_bt,), ...)`.
Same 3-layer stack, same shapes, same 2 warmup + 5 timed iters.

Bwd path is still **JAX-only** (D.3 will tile the Pallas bwd kernel
over M to make it usable at production-proxy shapes — until then bwd
runs through `_moe_block_bwd_jax`).

## small shape — `E=16, D=256, F=128, K=4, n_heads=4, head_dim=64`

| T    | jax_ref fwd ms | v_outside fwd ms | speedup (fwd) | jax_ref bwd ms | v_outside bwd ms | speedup (bwd) |
|------|---------------:|-----------------:|--------------:|---------------:|-----------------:|--------------:|
| 16   | 0.36           | 0.31             | 1.16×         | 0.88           | 0.92             | 0.96×         |
| 64   | 0.40           | 0.38             | 1.05×         | 1.01           | 1.08             | 0.94×         |
| 128  | 0.50           | 0.42             | 1.18×         | 1.26           | 1.32             | 0.95×         |
| 256  | 0.57           | 0.52             | 1.09×         | 1.44           | 1.63             | 0.88×         |
| 512  | 0.79           | 0.73             | 1.08×         | 2.11           | 2.26             | 0.93×         |
| 1024 | 1.14           | 1.14             | 1.00×         | 3.44           | 3.70             | 0.93×         |

## Delta vs Phase C (per-tile path)

| T    | fwd Phase C | fwd Phase D.2 | Δ      |
|------|------------:|--------------:|-------:|
| 16   | 1.02×       | 1.16×         | +0.14× |
| 64   | 1.06×       | 1.05×         | -0.01× |
| 128  | 1.04×       | 1.18×         | +0.14× |
| 256  | 1.05×       | 1.09×         | +0.04× |
| 512  | 1.07×       | 1.08×         | +0.01× |
| 1024 | 0.94×       | 1.00×         | +0.06× |

**Headline:** the T=1024 regression is closed (0.94× → 1.00× at par
with jax_ref). T=128 picks up an extra 0.14× from the dispatch
overhead reduction.

## Verdict

**G5-fwd SATISFIED across the full T sweep:** v_outside fwd is at par
or better than jax_ref at every T tested, including the previously
regressed T=1024 point.

**G5-bwd: still pending D.3.** Bwd path is pure JAX (Pallas bwd kernel
exists per D.1 but doesn't yet tile M; defaults to JAX). Numbers above
compare jax_ref autodiff against the hand-written JAX bwd — both run on
TC via XLA; they're roughly equivalent in compute, with v_outside
slightly slower due to the recompute + segment_sum scatter pattern.

After D.3 lands (Pallas bwd kernel tiled over M with persistent VMEM
accumulators), expect bwd numbers to converge to / exceed jax_ref.

## What changed in D.2

Per `targets/dsv3-fused-ep-moe/build/v_outside/expert_ffn.py`:

```python
# B.1 (Phase C): one pallas_call per M-tile
for i in range(num_bt):
    tok_i = lax.dynamic_slice_in_dim(sorted_tokens, i*bt, bt, axis=0)
    eids_i = lax.dynamic_slice_in_dim(sorted_eids, i*bt, bt, axis=0)
    out_pieces.append(_expert_ffn_one_tile(tok_i, eids_i, W1, W_d))
return jnp.concatenate(out_pieces, axis=0)

# D.2: ONE pallas_call with grid; Mosaic handles M-windowing + DMA double-buffer
return pl.pallas_call(
    body,
    grid=(num_bt,),
    in_specs=[
        pl.BlockSpec((bt, D), lambda i: (i, 0)),
        pl.BlockSpec((bt,),   lambda i: (i,)),
        pl.BlockSpec((E_local, D, 2*F), lambda i: (0,0,0)),   # full
        pl.BlockSpec((E_local, F, D),   lambda i: (0,0,0)),
    ],
    out_specs=pl.BlockSpec((bt, D), lambda i: (i, 0)),
    out_shape=jax.ShapeDtypeStruct((M, D), jnp.float32),
    scratch_shapes=[pltpu.VMEM((bt, D), jnp.float32)],
)(sorted_tokens, sorted_eids, W1, W_d)
```

## Constraint discovered

Mosaic rank-1 BlockSpec requires the block dim to be ≥ 128 (the lane
count) OR equal to the full array dim. At G2 test shape (M=32, bt=8),
neither holds — the kernel falls back to the per-tile path via
`impl="auto"`. Production bt_ffn=128 always picks the grid path.

## Notes

- Bench was at 4-core local TPU. Cluster runs (8-core bodaborg /
  128-core rbq) will exercise the grid path at production M.
- The `expert_ffn_v_outside(..., impl="auto")` default picks grid for
  bt ≥ 128, tile for bt < 128. Callers can force one or the other.
- W1 + W_d are loaded full per grid step (BlockSpec index_map returns
  `(0,0,0)`). At production scale where W1/W_d HBM footprint becomes
  large this may need slicing; current shapes fit VMEM comfortably.
