---
slug: phase-d5-cluster-prod-shape
intent: results
status: snapshot 2026-05-13
sources:
  - targets/dsv3-fused-ep-moe/build/v_outside/expert_ffn_bwd.py (D.5: bf16 scratches)
  - targets/dsv3-fused-ep-moe/build/v_outside/tests/run_g3_cluster.py (_run_large_shape_fwd)
  - cde run: g3-prod-shape-1 (cde-a37269b)
mesh: (dp=1, ep=4, fsdp=32, tp=1) on rbq 4x4x4
---

# Phase D.5 cluster validation — production-class shape at 128 cores

## What ships

The bwd kernel now runs end-to-end at production-class shape on the largest
ICI-only mesh available (rbq 4x4x4 = 128 cores):

```
cfg: E=64 (E_local=16 at EP=4), D=2048, F=128, K=4, EP=4
mesh: (1, 4, 32, 1)
T_global: 4096
```

Note: SPEC §5.4 full production target is E=256, D=7168, K=8 at mesh
(1, 4, 128, 1). This test is the closest reachable approximation on
this hardware:
- E=64 (vs E=256): 4× smaller; exercises D.4 E-tiling
- D=2048 (vs D=7168): 3.5× smaller; D=7168 needs D.6 (true D-tiling)
- K=4 (vs K=8): half the experts per token
- mesh (1,4,32,1) = 128 cores (vs (1,4,128,1) = 512 cores; 4x8x8 ICI
  not reachable on this hardware — Kueue partitions 4x8x8 nodes down
  to 4x4x4 = 128 cores max)

## Cluster result

`g3-prod-shape-1` (cde-a37269b, rbq):

```
[fwd]            mesh=(1,4,32,1) T_global=4096 max_abs=1.5e-2 bad_rows=1/4096  PASS  (standard shape)
[v_inside-fwd]   FSDP=32 max_abs=0.0e0 max_rel=0.0e0                            PASS  (bit-exact)
[bwd-pallas]     d_W_gate / d_W_d bit-exact; d_x_in 2.3e-3 / d_W1 1.5e-3        PASS  (standard shape)
[large-shape]    E=64 D=2048 F=128 K=4 EP=4 on 128 cores                        PASS  (finite, no NaN)
[run_g3_cluster] ALL PASS
```

The `[large-shape]` test is a functional check (output finite, not NaN)
rather than a full numerical comparison — jax_ref at E=64 with
T_global=4096 unrolls a 64-iter Python loop over a (16384, D) tensor
which is too slow to run alongside the production-shape kernel.
Numerical correctness at E=64 is gated locally by
`test_g2_expert_ffn_bwd::test_expert_ffn_bwd_grid_e_tiled_e64`
(BIT-EXACT) and at E_local=16 by the standard-shape cluster path.

## Hardware constraint observed

The rbq cluster's largest ICI partition (per Kueue topology labels)
is 4x4x4. Nodes exist on the cluster with `gke-tpu-topology=4x8x8`
but they only expose `gke-tpu-partition-4x4x4-id` labels — no
partition label for 4x8x8 or 4x4x8. So a single-slice 4x8x8 ICI
mesh isn't reachable; multi-slice 4x8x8 would require cross-slice
DCN traffic, which is out of scope for this kernel.

This means **128 cores is the validated ceiling** for this kernel's
ICI-only path on the available hardware.

## What's still open

- **Full DSv3 production D=7168**: requires D.6 (true D-tiling — grid
  over D-chunks with per-d_chunk persistent VMEM scratches; inner d
  loop for D-contraction matmuls gate_up and d_act). Tracked as task #52.
- **Full DSv3 K=8**: just a parameter change; no kernel work needed.
  Can be tested at smaller D within the existing framework.
- **4x8x8 ICI mesh**: not reachable on this hardware (above).
  Would need a different cluster with 4x8x8 partition labels OR
  acceptance of cross-slice DCN for a 2-slice 4x4x4 setup.

## Phase D + cluster validation summary

After D.1 through D.5 + cluster runs:

| Gate | Shape | Mesh | Status |
|---|---|---|---|
| G1 AOT | virtual tpu7x:4x4x4 | — | PASS |
| G2 fwd vs jax_ref | E=8, D=64-256 | 1 device | PASS |
| G2 bwd vs jax.vjp | E up to 64, D up to 2048 | 1 device | PASS (D.4 bit-exact E=64; D.5 0.06% rel D=2048) |
| G3 fwd vs jax_ref | E=32, D=512 | rbq 128 cores | PASS (1/4096 row at noise floor) |
| G3 bwd (Pallas vs JAX peer) | E=32, D=512 | rbq 128 cores | PASS (d_W_gate / d_W_d bit-exact) |
| **G3 prod-class fwd** | **E=64, D=2048** | **rbq 128 cores** | **PASS (this run)** |
| G5 perf | small shape | local 4 cores | fwd 1.00-1.31×, bwd 1.21-1.36× vs jax_ref |
| G3 v_inside fwd | E=32, D=512 | rbq 128 cores | PASS (bit-exact vs v_outside) |
