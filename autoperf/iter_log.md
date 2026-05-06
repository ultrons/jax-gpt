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
