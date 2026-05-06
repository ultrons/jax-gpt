# HALT — autoperf agent (dsv3_train_purefsdp track)

**Workload**: `autoperf/workloads/dsv3_train_purefsdp.yaml`
**Last iteration**: 0 (compile-failed at libtpu codegen; no training ran)
**Branch**: `autoperf/dsv3_train_purefsdp`
**Halted at**: 2026-05-06
**Reason**: `tool_blocked_libtpu` — pure FSDP at fsdp=512 with the v304
LIBTPU_INIT_ARGS hits a fatal libtpu assertion during XLA codegen for
the SparseCore async-collective offload path:

```
F0506 22:22:28.728338 async_collective_start_emitter.cc:65]
Check failed: smem_state_shape.dimensions(0) == num_states (26 vs. 6)
```

This is libtpu-internal code (`dev20260417` pin); not addressable from
jax-gpt source. The same flag-set compiles fine at v304's `ep=4 fsdp=128`
— the assertion is specific to the fsdp=512 mesh geometry's interaction
with SC offload.

Cluster cost: ~3 min admission + ~1.5 min compile attempt. ~30 pods
initialized cleanly, all crashed with the same assertion. No training
compute consumed. Full diagnostic in `autoperf/iter_log.md`. Cross-workload
note appended to `v7x_KNOWLEDGE.md` §5 ("`fsdp=512 + SC-offload flags →
libtpu CHECK fail`") so future agents on either track don't re-discover it.

## Recommended next human action

Three real options; cost/risk roughly increasing:

1. **Disable SC-offload flags for this workload (cheapest probe)** —
   parameterize `manifests/jobset.yaml.j2` to read SC-offload toggles from
   `cde_overrides`, then set them off in `dsv3_train_purefsdp.yaml`. Probably
   one to three flags need to come off (the AG-side ones are most likely):
   - `--xla_tpu_enable_sparse_core_collective_offload_all_gather=false`
   - `--xla_tpu_enable_sparse_core_collective_offload_2d_all_gather=false`
   - `--xla_tpu_enable_sparse_core_collective_offload_3d_all_gather=false`
   v304 baseline keeps its flags untouched. Risk: SC offload may have been
   contributing measurable TPS — disabling it changes the comparison vs
   v304 from "ep difference" to "ep + SC-offload difference," muddier
   attribution. Mitigation: also re-run dsv3_train_full's iter-0 with SC
   offload off, so both baselines are on the same flag set.

2. **Pivot to `ep=1 fsdp=256 tp=2` (v321 config)** — narrower FSDP axis
   may not trip the SC-offload assertion (untested). One-line workload
   yaml edit. Tests a different point in the design space — half FSDP, two
   TP — which is its own story, not strictly "pure FSDP". If the user's
   real question was "is non-EP a win," this is a defensible substitute.

3. **Wait for libtpu fix** — log a libtpu issue (out of our 4-agent system;
   would have to go through whatever Google-side channel handles libtpu
   bugs). Likely slowest. Both dsv3_train_full and dsv3_train_purefsdp
   tracks are working around their respective compile bugs already; a
   third tracked bug compounds the wait.

The autoperf agent's natural choice, if asked to keep moving without
human direction, would be **option 2** — single yaml edit, no template
work, retains "non-EP" intent. If the user wants the cleanest pure-FSDP
comparison, option 1 is correct but more invasive.

## Open follow-ups (parallel)

- `ultrons/perfsim#1` — resolved 2026-05-06.
- `ultrons/perfsim#3` — open, non-blocking, gemm_eff calibration. Carries
  over to whichever workload eventually generates a real headroom report.

## Cluster state at halt

- `dsv3train-pf-i0` JobSet: `Finished=True (Failed)`, ~30 pods crashed
  with the libtpu assertion. cde reaped.
- `dsv3train-i1` JobSet (prior iter-1 from dsv3_train_full): also failed
  earlier today; cde reaped.
- `cde history --status running`: zero (one slot pending,
  `v341b-nchunks4-barrier`, unrelated). Parallelism budget free.
- Logs preserved: `/tmp/dsv3train-pf-i0.log` (mostly empty — monitor
  startup race; full pod logs available via `kubectl logs <pod>`).

## What was committed on this branch

- `autoperf/workloads/dsv3_train_purefsdp.yaml` (the workload definition).
- `autoperf/iter_log.md` (iter-0 with full failure diagnostic).
- `autoperf/v7x_KNOWLEDGE.md` §5 ("fsdp=512 + SC-offload flags →
  libtpu CHECK fail" cross-workload entry, plus the
  `moe_xlayer_prefetch broken at ep=4 fsdp=128` entry carried over from
  the dsv3_train_full track).
- `autoperf/HALT.md` (this file).

No source files in `jax_gpt/` were touched. No manifest template edited
(option 1 above would be the first such edit if the user goes that route).
