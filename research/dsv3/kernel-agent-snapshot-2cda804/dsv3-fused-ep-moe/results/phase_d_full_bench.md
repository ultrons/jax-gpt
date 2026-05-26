---
slug: phase-d-full-bench-results
intent: results
status: snapshot 2026-05-12 (D.1+D.2+D.3 complete)
sources:
  - targets/dsv3-fused-ep-moe/bench.py (--bwd-impl pallas)
  - targets/dsv3-fused-ep-moe/build/v_outside/expert_ffn.py (D.2 grid fwd)
  - targets/dsv3-fused-ep-moe/build/v_outside/expert_ffn_bwd.py (D.1+D.3 Pallas bwd)
  - targets/dsv3-fused-ep-moe/build/v_outside/moe_block_vjp.py (Pallas bwd path)
device: local TPU v7x (4 chips × 1 core = 4 jax devices)
---

# Phase D complete — G5 perf signal with full Pallas fwd+bwd

End-to-end Pallas kernels: D.2 single-grid forward + D.3 grid-tiled
backward (W1 workaround + persistent VMEM weight-grad accumulators).
Bench compares v_outside (Pallas fwd + Pallas bwd) against jax_ref
(JAX-only fwd + JAX autodiff bwd) in a 3-layer transformer stack.

Command: `python bench.py --sweep --shape small --bwd-impl pallas`

## small shape — `E=16, D=256, F=128, K=4, n_heads=4, head_dim=64`

| T    | jax_ref fwd ms | v_outside fwd ms | speedup (fwd) | jax_ref bwd ms | v_outside bwd ms | speedup (bwd) |
|------|---------------:|-----------------:|--------------:|---------------:|-----------------:|--------------:|
| 16   | 0.42           | 0.32             | **1.31×**     | 0.87           | 0.65             | **1.33×**     |
| 64   | 0.39           | 0.38             | 1.05×         | 1.03           | 0.76             | **1.36×**     |
| 128  | 0.47           | 0.42             | 1.12×         | 1.23           | 0.91             | **1.36×**     |
| 256  | 0.57           | 0.54             | 1.06×         | 1.47           | 1.17             | **1.25×**     |
| 512  | 0.78           | 0.71             | 1.09×         | 2.12           | 1.72             | **1.23×**     |
| 1024 | 1.15           | 1.15             | 1.00×         | 3.45           | 2.86             | **1.21×**     |

Note: "bwd ms" is fwd+bwd time (jax.grad includes a recompute fwd in
the bwd pass). The fwd column is fwd-only.

## Verdict

**G5-fwd SATISFIED:** v_outside fwd at par or better than jax_ref
across the full sweep (1.00× at T=1024, 1.31× at T=16).

**G5-bwd SATISFIED with margin:** v_outside fwd+bwd is **1.21-1.36×
faster** than jax_ref autodiff across the entire T sweep. The Pallas
bwd kernel's per-expert tiling + VMEM `+=` weight-grad accumulation
beats XLA's autodiff-generated bwd by a consistent ~25%.

## Progression across Phase D

| T    | C.G5 fwd | D.2 fwd  | D-full fwd | C.G5 bwd | D.2 bwd  | D-full bwd |
|------|---------:|---------:|-----------:|---------:|---------:|-----------:|
| 16   | 1.02×    | 1.16×    | 1.31×      | 0.98×    | 0.96×    | **1.33×**  |
| 64   | 1.06×    | 1.05×    | 1.05×      | 1.04×    | 0.94×    | **1.36×**  |
| 128  | 1.04×    | 1.18×    | 1.12×      | 0.96×    | 0.95×    | **1.36×**  |
| 256  | 1.05×    | 1.09×    | 1.06×      | 0.98×    | 0.88×    | **1.25×**  |
| 512  | 1.07×    | 1.08×    | 1.09×      | 0.90×    | 0.93×    | **1.23×**  |
| 1024 | 0.94×    | 1.00×    | 1.00×      | 0.93×    | 0.93×    | **1.21×**  |

- **C.G5** (Phase C snapshot): per-tile fwd loop (B.1), JAX bwd.
  T=1024 fwd regressed to 0.94×.
- **D.2** (single-grid fwd, JAX bwd): T=1024 fwd back to par.
  Bwd unchanged (JAX-only).
- **D-full** (D.2 grid fwd + D.1+D.3 grid bwd): both directions
  satisfied. Bwd jumps from regression to consistent 1.21-1.36× lead.

## What enabled the bwd speedup

The Pallas bwd kernel fuses several ops into one pallas_call dispatch
that the JAX autodiff bwd would emit as separate XLA ops:

1. **Per-expert recompute** of (gate, up, silu, act, out_e_unscaled) —
   fused into one kernel iteration; XLA bwd emits separate matmuls +
   element-wise ops + memory shuffles.
2. **VMEM `+=` weight-grad accumulation** — d_W1_acc and d_W_d_acc
   live in VMEM across the per-expert loop AND across grid iterations
   (D.3 persistent scratches). XLA autodiff would emit HBM read-modify-
   write per expert per tile.
3. **W1 workaround for d_sorted_w** — the (M,D) out_unscaled buffer
   avoids the Mosaic v7x sublane-gather blocker; reduce happens on TC
   in JAX glue (one op).
4. **Mask-based per-expert dispatch** — int-arithmetic masks (A3)
   avoid the broadcast / scatter patterns autodiff produces.

The cumulative effect: bwd FLOPS are similar but memory traffic drops
substantially, and fewer dispatches means less compiler/runtime overhead.

## What's still open

- **G3-bwd-pallas at full production-proxy shape** (E=32, D=512,
  F=128, EP=4) passes G3-bwd numerical gate; cluster-scale perf
  bench is a follow-up.
- **EP path bench** (multi-host A2A) — the bench above is single-device.
  Cluster-scale runs on rbq 4x4x4 are next (file separately, since
  cluster turnaround is much higher).
- **bt_ffn heuristic** — currently picks `min(1024, M//4)` rounded to
  a multiple of 128. Works well in this sweep; production may want
  per-shape tuning.

## Notes

- All compute is bf16-input with f32 accumulation; cast back to bf16
  only at HBM write.
- The bwd column reports fwd+bwd time (jax.grad's standard behavior).
  Computing bwd-only would require a residual-state pre-stage and
  isn't the typical training-loop shape we benchmark against.
- v_outside bwd benefit is in the Pallas kernel's reduction of memory
  traffic, not in raw FLOPS. Profile to confirm at production shape.
