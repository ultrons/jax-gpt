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
- **Result (after measurement)**: TBD.
