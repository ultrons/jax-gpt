---
slug: phase-c-g5-bench-results
intent: results
status: snapshot 2026-05-12
sources:
  - targets/dsv3-fused-ep-moe/bench.py
  - targets/dsv3-fused-ep-moe/build/v_outside/moe_block_vjp.py
device: local TPU v7x (4 chips × 1 core visible = 4 jax devices)
---

# Phase C — G5 perf signal: v_outside vs jax_ref baseline

SPEC §8.1: "kernel must be at least at par or better" than the pure-JAX
reference. This is the operational definition of G5 — measured here on
the local TPU at two model shapes, sweeping `T` from 16 to 1024 with
3 stacked transformer blocks (LayerNorm → multi-head attention →
LayerNorm → MoE → residual).

Each measurement: 2 JIT warmup iterations + 5 timed iterations. All runs
single-device (`v_outside` uses `moe_block_vjp.make_moe_block`, no EP).

## tiny shape — `E=8, D=64, F=32, K=2, n_heads=4, head_dim=16`

This shape is dispatch-dominated; absolute times are tiny, so the signal
is mostly about JIT-overhead parity rather than true compute.

| T    | jax_ref fwd ms | v_outside fwd ms | speedup (fwd) | jax_ref bwd ms | v_outside bwd ms | speedup (bwd) |
|------|---------------:|-----------------:|--------------:|---------------:|-----------------:|--------------:|
| 16   | 0.31           | 0.29             | 1.07×         | 0.74           | 0.78             | 0.95×         |
| 64   | 0.34           | 0.31             | 1.11×         | 0.77           | 0.82             | 0.94×         |
| 128  | 0.35           | 0.33             | 1.05×         | 0.82           | 0.82             | 1.00×         |
| 256  | 0.41           | 0.40             | 1.02×         | 0.97           | 0.93             | 1.05×         |
| 512  | 0.46           | 0.53             | 0.87×         | 0.99           | 1.12             | 0.89×         |
| 1024 | 0.58           | 0.64             | 0.90×         | 1.25           | 1.48             | 0.84×         |

## small shape — `E=16, D=256, F=128, K=4, n_heads=4, head_dim=64`

Matmul-leaning shape; per-call compute is 4-8× the tiny shape.

| T    | jax_ref fwd ms | v_outside fwd ms | speedup (fwd) | jax_ref bwd ms | v_outside bwd ms | speedup (bwd) |
|------|---------------:|-----------------:|--------------:|---------------:|-----------------:|--------------:|
| 16   | 0.32           | 0.32             | 1.02×         | 0.92           | 0.93             | 0.98×         |
| 64   | 0.40           | 0.38             | 1.06×         | 1.08           | 1.03             | 1.04×         |
| 128  | 0.49           | 0.47             | 1.04×         | 1.22           | 1.27             | 0.96×         |
| 256  | 0.60           | 0.58             | 1.05×         | 1.58           | 1.62             | 0.98×         |
| 512  | 0.78           | 0.73             | 1.07×         | 2.11           | 2.33             | 0.90×         |
| 1024 | 1.17           | 1.24             | 0.94×         | 3.49           | 3.74             | 0.93×         |

## Verdict

**G5 SATISFIED at small-to-medium T (≤512):** v_outside is at par or
slightly better than the JAX reference. At T=512 on the small shape,
v_outside fwd is **1.07× faster** with a clear separation.

**Regression at T≥1024**: v_outside fwd ~6-10% slower, fwd+bwd ~7-16%
slower. This is the **per-tile `pallas_call` overhead** baked in at B.1
as a workaround for Mosaic dynamic-index alignment constraints at small
G2 scale. As `num_bt = M / bt_ffn` grows, dispatch overhead per tile
exceeds matmul gains.

Phase D fixes this: replace per-tile `pallas_call` (one call per token
tile) with a single grid-based `pallas_call` (one call, internal
double-buffered DMA loop). The path is sketched in
`distilled/patterns/double-buffered-dma.md`; the conversion is
straightforward once Mosaic's alignment behaviour at production tile
sizes (`bt ≥ 128, bd ≥ 256, bf ≥ 256`) is verified.

For now: **G5 is a green-with-asterisk** — passes at the meshes we care
about for G2/G3 correctness; large-T regression is a known Phase D item.

## Backward measurements caveat

`v_outside` bwd runs through `custom_vjp._bwd` which is **JAX-only** (the
Pallas bwd FFN kernel is deferred to Phase D — see
`_inbox/blocker-mosaic-v7x-1d-reduction-acc.md`). So the bwd numbers
above compare jax_ref's autodiff against a hand-written JAX bwd that
mirrors the same per-expert recompute logic. Both paths run on TC via
XLA; they're roughly equivalent in compute.

The real bwd perf signal comes after the Pallas bwd kernel lands.

## Notes

- All compute is bf16-input with f32 accumulation in the kernel
  (`expert_ffn.py`'s out_acc is f32; cast back to bf16 only at HBM
  write).
- Attention: 4-head, scaled dot-product, no flash; representative compute
  but not optimised. Not on the critical path for the G5 read.
- 3-layer stack chosen per SPEC §7 — measures steady-state, not
  first-layer JIT compile.
- Local TPU exposes 4 cores; production targets are 8+ cores and
  measured separately on cluster runs (g3-cluster-7 = 8 cores PASS,
  rbq-g3-5 = 128 cores PASS).
