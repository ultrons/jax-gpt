---
slug: phase-d-cluster-pallas-bwd
intent: results
status: snapshot 2026-05-12
sources:
  - targets/dsv3-fused-ep-moe/build/v_outside/tests/run_g3_cluster.py
  - targets/dsv3-fused-ep-moe/build/v_outside/expert_ffn_bwd.py (D.1+D.3+D.4)
  - targets/dsv3-fused-ep-moe/build/v_outside/moe_block_ep_vjp.py (bwd_impl=pallas)
  - cde run: g3-pal-rbq-2 (cde-bbab2b3)
mesh: (dp=1, ep=4, fsdp=32, tp=1) on rbq 4x4x4 (16 hosts × 8 cores = 128 cores)
---

# Phase D — cluster validation of Pallas bwd at production-class mesh

After D.1 (Pallas bwd FFN kernel, W1 workaround) + D.3 (grid-tiled bwd
with persistent VMEM acc) + D.4 (E-tiled for E_local up to 64) all
passed locally, the cluster gate was: does the cross-host A2A path
interact correctly with the Pallas bwd kernel inside `shard_map`?

Test: `_run_bwd_pallas_vs_jax_test` in `run_g3_cluster.py` — runs both
`bwd_impl="jax"` and `bwd_impl="pallas"` through the same shard_map at
the production-class mesh, asserts grads match within G2-bwd tolerance
(rtol=5e-2, atol=5e-2). No `jax_ref` reference comparison — the
peer-to-peer Pallas-vs-JAX check is what the cluster validates; jax_ref
already gated correctness at smaller shape locally.

## Results

`g3-pal-rbq-2` (cde-bbab2b3, gke_cloud-tpu-multipod-dev_us-central1_bodaborg-super-rbq)

Mesh: (1, 4, 32, 1) on tpu7x:4x4x4 = 128 cores across 16 hosts. Cfg:
E=32, D=512, F=128, K=4, EP=4, bt_router=32, bt_ffn=128. T_global=4096,
T_local=128 per device, M_local=512.

```
[run_g3_cluster] jax.distributed initialized; 16 processes converged
[run_g3_cluster] 128 devices, platform=tpu

[fwd]         max_abs=1.5625e-02  bad_rows=1/4096 (0.024%)         PASS
[bwd-pallas]  d_x_in    max_abs=3.9062e-03   max_rel=2.2831e-03    PASS
[bwd-pallas]  d_W_gate  max_abs=0.0000e+00   max_rel=0.0000e+00    PASS  (bit-exact)
[bwd-pallas]  d_W1      max_abs=3.1250e-02   max_rel=1.4793e-03    PASS
[bwd-pallas]  d_W_d     max_abs=0.0000e+00   max_rel=0.0000e+00    PASS  (bit-exact)
[run_g3_cluster] ALL PASS
```

Pallas-bwd grads match JAX-bwd grads at the production-class mesh:
- d_W_gate and d_W_d: **bit-exact** (max_abs = 0)
- d_x_in: max_rel 0.23%
- d_W1: max_rel 0.15%

All within the G2-bwd tolerance.

## What this validates

The Pallas bwd kernel works correctly when:
1. The fwd path uses `lax.ragged_all_to_all` across 4 EP shards
2. The bwd path is computed by `jax.vjp` through the JAX-only mirror
3. The per-expert FFN section of the bwd is replaced by the Pallas
   bwd kernel via `_expert_ffn_pallas_with_bwd` custom_vjp
4. The full computation runs inside a 4-axis `shard_map` with
   FSDP=32 across hosts

Step (3) is the critical Pallas/JAX boundary. The Pallas kernel runs
inside `shard_map`, but the wrapping `jax.vjp` handles the A2A and
sort gradients via autodiff. Cluster scale stress-tests the
ragged_all_to_all gather/scatter semantics, the cross-host DMA, and
the custom_vjp + autodiff composition all at once.

## What broke on first attempt (g3-pal-rbq-1)

```
ValueError: Gathering global non-fully-addressable arrays
only supports tiled=True
```

The first cluster runner version called
`multihost_utils.process_allgather(g_jax, tiled=False)` to gather grads
for diagnostic comparison. Sharded grads can't be gathered with
`tiled=False`; `tiled=True` would have its own problems (concatenates
identical shards for replicated grads). Fix: replace gather+diff with
an on-device scalar reduce — `jit(lambda gj, gp: (jnp.max(...), ...))`.
The result is a replicated scalar, fully addressable on every process,
no gather needed.

This is a worth-remembering lesson: **at cluster scale, prefer
on-device scalar reductions over host-side gather+diff.** Gather has
sharded-vs-replicated edge cases; scalar reductions just work.

## Open: full-shape production cluster run

Current run uses E=32 (E_local=8). D.4 unlocked E=64 (E_local=64 at
EP=4) locally, and D.5 (D-tiling for D=7168) is queued. A future
cluster run at full DSv3 shape will exercise the production
E_local=64 + D=7168. This was outside scope here — the immediate
goal was to validate the existing D.1+D.3 path on cluster, which
it does.
