---
slug: phase-e2-v_inside-cluster
intent: results
status: snapshot 2026-05-12
sources:
  - targets/dsv3-fused-ep-moe/build/v_inside/moe_block_ep.py
  - targets/dsv3-fused-ep-moe/build/v_outside/tests/run_g3_cluster.py
  - distilled/_inbox/blocker-v_inside-explicit-ag-inside-shardmap.md (the bug log)
  - cde run: g3-e2-rbq-9 (cde-3be3b2b)
mesh: (dp=1, ep=4, fsdp=32, tp=1) on rbq 4x4x4
---

# Phase E.2 — v_inside cluster validation via auto-AG

## What ships

v_inside MoE block accepts FSDP-sharded weights per SPEC §5.1 caller
contract:

```
W1:  (E_local, D, 2F_shard)   sharded P("ep", None, "fsdp")
W_d: (E_local, F_shard, D)    sharded P("ep", "fsdp", None)
```

Inside the wrapper's shard_map, `in_specs` declare these as fsdp-
REPLICATED — JAX auto-AGs at the shard_map boundary. The wrapper sees
full-F W and runs identical math to v_outside.

This is **Option β** per `_inbox/v_inside-fsdp-layout-decision.md`:
the caller contract change is what's new; the HBM footprint is the
same as v_outside (full-F W materializes inside the wrapper). Real
HBM peak win waits for Phase E.3 (streaming the full-F W into the
kernel chunk by chunk).

## Cluster result

`g3-e2-rbq-9` (cde-3be3b2b, gke_cloud-tpu-multipod-dev_us-central1_bodaborg-super-rbq):

```
mesh:     (1, 4, 32, 1) on tpu7x:4x4x4 = 128 cores × 16 hosts
cfg:      E=32, D=512, F=128, K=4, EP=4, bt_router=32, bt_ffn=128
T_global: 4096

[fwd]            mesh=(1, 4, 32, 1) max_abs=1.56e-2 bad_rows=1/4096 (0.024%)  PASS
[v_inside-fwd]   FSDP=32 max_abs=0.0000e+00 max_rel=0.0000e+00                 PASS
[bwd-pallas]     d_x_in 2.3e-3 / d_W_gate bit-exact / d_W1 1.5e-3 / d_W_d bit-exact  PASS
[run_g3_cluster] ALL PASS
```

v_inside output is **bit-exact** to v_outside at production-class mesh.

## The journey — 8 failed cluster runs before the pivot

The first v_inside.moe_block_ep version did an explicit
`lax.all_gather(W, "fsdp", ...)` inside the shard_map body. At
multi-host this produced consistent 12% rel error vs v_outside despite:

- Per-element `max(|AG_w1 - native_w1|)` = 0 (TRULY bit-exact bytes)
- Per-device `W1.sum()`, `sort_idx`, `expert_ids` all matching across paths
- `lax.optimization_barrier`, `tiled=False + jnp.concatenate`, and
  `mesh_utils.create_device_mesh` — all left the failure unchanged

The bit-identical failure value across 8 runs strongly suggested XLA
folded every source variant to the same wrong HLO. Bisection narrowed
to "byte-equivalent inputs, identical kernel, divergent output" —
characteristic of a physical-layout mismatch on the AG output tensor
that JAX-level reads dereference correctly but Pallas reads
misinterpret.

Full bug log:
`distilled/_inbox/blocker-v_inside-explicit-ag-inside-shardmap.md`.

## The pivot (Option β-final)

Dropped the explicit AG. Declared shard_map `in_specs` for W1/W_d as
fsdp-REPLICATED. JAX auto-AGs at the boundary via canonical
lowering — no in-shard_map AG triggered, no layout-mismatch path
exercised.

```python
sharded_inside = shard_map(
    moe_block_ep_v_inside_fwd,
    mesh=mesh,
    in_specs=(x_spec, P(None, None),
              P("ep", None, None),   # auto-AG on entry
              P("ep", None, None)),  # auto-AG on entry
    out_specs=x_spec,
)(x_in, W_gate, W1_F_sharded, W_d_F_sharded)
```

## What's still open

- **Real HBM win (Phase E.3):** keep W F-sharded inside the wrapper,
  stream the full-F portions into the kernel via the streaming-psum-
  scatter pattern. The E.2 pivot achieves caller-API parity with
  v_inside but the HBM peak inside the wrapper still has the full-F
  W materialized.
- **Root cause of the explicit-AG bug (deferred):** would need HLO
  diff or XLA instrumentation. Has minimal practical impact since
  the auto-AG path works and is the documented canonical pattern.

## Files

```
targets/dsv3-fused-ep-moe/build/v_inside/
├── expert_ffn.py            # E.1 kernel, identical math to v_outside
├── moe_block_ep.py          # E.2 wrapper, auto-AG via in_specs
└── tests/
    ├── test_g1_aot.py       # AOT compile gate
    ├── test_g2_fwd.py       # fsdp=1 vs v_outside bit-exact
    └── test_g3_ep_fwd.py    # fsdp>1 (auto-AG path)

distilled/_inbox/
├── blocker-spec-v_inside-sharding-vs-math.md  # SPEC §3 ↔ §5.1 reconciliation
├── v_inside-fsdp-layout-decision.md           # Option α/β/γ analysis
└── blocker-v_inside-explicit-ag-inside-shardmap.md  # the 8-run bug log
```
