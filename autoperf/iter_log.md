# autoperf iter_log — dsv3_train_full

Per `~/jax-gpt/autoperf/AGENT.md` §5b. One section per iteration. The
perfsim agent reads this when an issue references "iteration N".

---

## iter 0 — baseline (no autoperf change)

- **Reference profile**: `gs://max-experiments/dsv3/profiles/v304-cde-repro/`
  (image `cde-d08fbb8`, commit `3895a970`); host-localized at
  `autoperf/profiles/v304-cde-repro/...`.
- **Workload**: `autoperf/workloads/dsv3_train_full.yaml` baseline overrides
  (no `moe_xlayer_prefetch`).
- **Headroom report**: `autoperf/reports/dsv3_train_full_iter0.json`
  (regenerated locally with merged perfsim main, 2026-05-06).
- **Top-3 leaves** (`headroom_total_ms` desc):
  | leaf | predicted µs/step | measured µs/step | ratio | headroom µs/step |
  |---|---|---|---|---|
  | FSDP_AG | 0 | 263,681 | ∞ | +263,681 |
  | Router | 104,617 | 192,710 | 1.84× | +88,093 |
  | Norms | 207,282 | 275,286 | 1.33× | +68,004 |
- **Calibration caveat**: `meta.calibration_status="uncalibrated"`.
  Compute leaves all over-predicted by ~3.6× (Expert_gmm 54.9M predicted vs
  15.3M measured = 0.28×). Headroom = 0 for compute leaves until
  gemm_eff is fit. Comm/non-compute leaves are usable today.

## iter 1 — enable `moe_xlayer_prefetch`

- **Commit**: TBD on push
- **Change**: `autoperf/workloads/dsv3_train_full.yaml` adds
  `moe_xlayer_prefetch: true` to `cde_overrides`. Renders as bare
  `--moe_xlayer_prefetch` flag to `python -m jax_gpt.models.dsv3.train`.
- **Top-leaf targeted**: `FSDP_AG` (`headroom_total_ms` rank 1, +264 ms/step).
- **Lever per `auto-perf-guide.md` training subsection**:
  > `FSDP_AG` — `weight_allgather_f` over the F (expert) axis; ensure
  > `fsdp_uses_cores=True` and that prefetch_layers ≥ 1.
- **Cluster-side mapping**: jax-gpt's `cfg.moe_xlayer_prefetch=True` toggles
  `model.py:3061 _moe_scan_fn_pf` (3-tuple scan carry, prefetches layer
  N+1's `_ag_one_moe_layer` inside layer N's body). pcast/reduced bwd
  semantics avoid the v294-class 109 GB carry-grad stash. Gated on
  `moe_backend=gmm_ag`.
- **Hypothesis**: if the FSDP weight AG is exposed because the XLA
  scheduler can't lift it across the layer boundary on its own, this
  scan-level cross-layer carry will let it overlap with the next
  layer's compute and reduce measured FSDP_AG below 264 ms/step.
- **Risks (predicted)**:
  - **HBM**: extra ~7 GB of prefetched weight in scan carry. v304 baseline
    runs near limit; may push compile-time `RuntimeProgramAllocationFailure`.
  - **NaN**: untested at full+ga=1+n_chunks=2 scale despite codepath
    landing ~v300+. If NaN at step 1, halt + revert (AGENT.md §13).
  - **No-op**: if XLA scheduler was already lifting the AG
    cross-layer (just imperfectly), explicit prefetch may not move metric.
    That's signal that schedule-position-not-concurrency is the bottleneck;
    iter-2 candidate becomes Router/Norms or chase the calibration gap.
- **Expected next iter**: with FSDP_AG headroom resolved or shown
  unchanged-by-this-knob, top-leaf likely shifts to either Router (88 ms)
  or — if gemm_eff calibration lands — Expert_gmm.
- **Decision**: launch.
- **Result**: **FAILED at compile (no training step ran).** `cde run dsv3train-i1`
  → JobSet `Finished=True (Failed)` after first attempt; reason
  `ReachedMaxRestarts` (failurePolicy: maxRestarts: 0). All 64 pods
  reported the same Python exception during `train_step` tracing:
  ```
  ValueError: all_gather_reduced only accepts inputs that are varying.
  Got bf16[64,16,7168]
  ```
  Stack: `train_step` → `value_and_grad` → `_vjp` → `linearize` →
  `_all_gather_is_async` → `all_gather_reduced` →
  `_all_gather_reduced_effectful_abstract_eval`. Triggered during the
  bwd-transpose pass through `_ag_one_moe_layer` (called inside
  `_moe_scan_fn_pf` at `model.py:3079`).
  - Shape `bf16[64,16,7168]` = `[E_local=64, F_local=16, D=7168]` —
    one MoE layer's weight tile per FSDP shard (E=256/EP=4=64;
    F=2048/FSDP=128=16). Forward AG would gather along F (FSDP axis);
    error fires in bwd transpose where the gathered tensor is no longer
    "varying" along the AG axis under the new JAX sharding regime.
  - Path is gated on `cfg.moe_xlayer_prefetch and cfg.moe_backend == "gmm_ag"`
    — both true; cleanly hit. Same path with `moe_xlayer_prefetch=False`
    (default) compiles fine on `cde-9ea30df` baseline.
  - **No NaN, no OOM, no hardware failure.** Pure Python tracing exception.
  - Cluster cost: ~5 min admission + ~2 min compile attempt + propagation
    of failure across 64 pods. No training compute consumed.
- **Conclusion**: the `moe_xlayer_prefetch` code path
  (`model.py:3061-3094`) has a bug in its bwd-transpose under the production
  sharding (`fsdp=128 ep=4 tp=1` on v7x_4x8x8). Recorded as a known-broken
  experiment in `v7x_KNOWLEDGE.md` so future autoperf agents don't propose
  it again until the underlying jax-gpt bug is fixed.
- **Halt reason**: `broke_training` (compile-side, AGENT.md §13). Workload
  yaml change reverted. iter-2 candidate: `Router` (next-highest
  headroom, +88 ms/step on v304 baseline).
- **Reverted**: in commit immediately following this one (single-line
  removal of `moe_xlayer_prefetch: true` from `cde_overrides`; iter-1's
  iter_log + BLOCKED.md entries kept for audit).

---

## iter 0 (recalibrated, 2026-05-07) — perfsim#4 + #5 landed

After perfsim#4 (gmm_ag `batch_sharded_by_ep` wiring) and perfsim#5
(xplane-vs-perfsim dim validation) merged to perfsim main, re-ran the
headroom report on the same v304-cde-repro xplane. **The picture
flipped.** New report saved at
`autoperf/reports/dsv3_train_full_iter0.json` (post-fix); old
uncalibrated version preserved at
`autoperf/reports/dsv3_train_full_iter0_prefix.json` for audit.

| leaf | predicted µs/step | measured µs/step | ratio | headroom µs/step |
|---|---|---|---|---|
| **Expert_gmm** | 13,721,482 | 15,338,155 | **1.12×** | **+1,616,673** |
| FSDP_AG | 0 | 263,681 | ∞ | +263,681 |
| Router | 26,192 | 192,710 | 7.36× | +166,518 |
| Norms | 207,282 | 275,286 | 1.33× | +68,004 |
| Attn_scores | 8,527,927 | 5,901,344 | 0.69× | 0 (ratio<1) |
| (others all ratio<1, residual gemm_eff calibration outstanding) | | | | |

`meta.calibration_status: "partial — gmm_ag wiring landed (issue #4);
residual gemm_eff calibration outstanding"`. perfsim#5's dim check ran
clean (no shape mismatches).

**Top-3 shifted**: Expert_gmm (was masked at 0) is now #1 by 6× over
the previous top FSDP_AG. iter-1's lever pick was on stale data —
it would have been the wrong call even if the prefetch bwd-transpose
bug hadn't fired.

## iter 2 — enable `moe_use_gmm_v2`

- **Workload yaml change**: add `moe_use_gmm_v2: true` to `cde_overrides`.
  Renders as bare `--moe_use_gmm_v2` flag.
- **Top-leaf targeted**: `Expert_gmm` (`headroom_total_ms` rank 1,
  +1,617 ms/step on v304 baseline post-fix).
- **Lever per heuristic table**: "verify gmm_ag kernel registration over
  the default ragged-dot path" — this is exactly what `--moe_use_gmm_v2`
  controls (`model.py:1791-1799`). Routes the 3 ragged-dots per chunk
  through Pallas `gmm_v2_train` + `gmm_v2_fused_silu_train` (gate+up+silu
  fused into 2 calls instead of 3). Backward via `jax.vjp` on
  `jax.lax.ragged_dot` reference (Stage A.1 design).
- **Prior production evidence**: qwen35 uses `gmm_v2` at BS=4096 on v7x
  in production, 728 TPS/chip @ MFU. Different model, but same kernel
  family on the same hardware regime.
- **Pre-launch sanity**: imports verified locally with
  `PYTHONPATH=jax-gpt:jax-gpt/jax_gpt/models/dsv3` —
  `kernels.gmm_v2_train.gmm_v2_train` and
  `kernels.gmm_v2_train.gmm_v2_fused_silu_train` resolve. (Docker
  WORKDIR makes the second path implicit at runtime.)
- **Hypothesis**: if Pallas gmm_v2 reduces measured Expert_gmm time, the
  +1.6 sec/step headroom shrinks. Realistic gain per advisor's
  ratio-1.12× analysis: **160-480 ms/step** (10-30% of theoretical),
  likely larger than fixing FSDP_AG (264 ms ceiling).
- **Risks (predicted)**:
  - **Bwd path**: `jax.vjp` on `jax.lax.ragged_dot` reference may have
    different shape regime than fwd Pallas kernel. Could hit
    untested-at-scale compile bugs (the iter-1 / iter-purefsdp pattern).
    If compile fails, halt + revert.
  - **HBM**: gmm_v2 has its own scratch / VMEM allocation. May change
    peak HBM by ±1-2 GB. v304 runs near limit, but the math suggests
    gmm_v2 *reduces* per-call temp by fusing 3 ragged-dots into 2
    calls, so net should be neutral or favorable.
  - **Numerical drift**: bf16 fused silu may have slightly different
    rounding than 3 separate ragged-dots. If step-1 loss isn't 415.491
    (v304 baseline), that's a divergence to investigate (likely OK
    within bf16 tolerances).
  - **Calibration interaction**: residual gemm_eff calibration is still
    pending (perfsim#3 partial). After full calibration, Expert_gmm's
    ratio may shift; iter-2's actual measured improvement is what
    matters, not the predicted-side delta.
- **Decision**: launch.
- **Result**: TBD.
