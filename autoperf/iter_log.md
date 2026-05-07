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
- **Result (attempt 1, `dsv3train-i2`, ea60078)**: **FAILED at import**.
  ```
  ModuleNotFoundError: No module named 'kernels'
  ```
  Triggered by `from kernels.gmm_v2_train import ...` at `model.py:1793`,
  which only resolves when `jax_gpt/models/dsv3/` itself is on PYTHONPATH.
  The Dockerfile sets `PYTHONPATH=/app`; trainer is invoked as
  `python -m jax_gpt.models.dsv3.train`; only `/app` is on path. So the
  import fails before reaching any compile.

  This is a **real jax-gpt source bug** — the import is wrong as
  written, not just untested. My local sanity-check passed only because
  I extended PYTHONPATH manually. Same bug exists at `model.py:1642`
  (`from kernels.gather_reduce_pallas`) but that's a different code
  path (gated on `cfg.moe_use_sc_scatter`); not in scope for iter-2.

- **Fix (one-line, same iter)**: `model.py:1793` —
  `from kernels.gmm_v2_train` → `from .kernels.gmm_v2_train`. Relative
  import resolves correctly under `python -m`. Sanity-check: `grep -n
  'from .*kernels.gmm_v2_train'` shows the relative form lands.
  Cluster cost of attempt 1: ~3 min admission + import-time crash.
  No training compute.

- **Result (attempt 2, `dsv3train-i2b`, commit f0b34da)**: **SUCCESS — first
  end-to-end win of the session.** Steady-state numbers (steps 2-4):
  | metric | v304 baseline | iter-2 | delta |
  |---|---|---|---|
  | step time | 37.0 s | 34.65 s | **−2.35 s/step** |
  | TPS/chip | 1770 | 1882 | **+6.6%** |
  | MFU | 28.6% | 30.5% | **+1.9 pp** |
  | step-1 loss | 415.491 | 415.46-415.47 | within bf16 tolerance |

  Cluster cost: ~5 min admission + ~4 min compile + ~2 min steady-state
  + profile capture. Profile pulled to
  `autoperf/profiles/dsv3train-i2b/...`.
  Headroom report saved at
  `autoperf/reports/dsv3_train_full_iter2.json`.

- **Per-leaf attribution caveat (and the next-iter lever question)**:
  iter-2 headroom report shows Expert_gmm measured = 15.30M µs/step,
  basically unchanged from iter-0's 15.34M. But step time dropped by
  2.35 s. The cause is the bucketer's `LEAF_PATTERNS_TRAINING` rule:
  Expert_gmm matches fusions containing `ragged-dot`. The gmm_v2
  Pallas kernel produces a `tpu_custom_call` with a different fusion
  name, so iter-2's Expert_gmm leaf is under-counting. perfsim#5's
  defense-in-depth check correctly emitted:
  ```
  [info] Expert_gmm: no HLO op matched substring='ragged-dot' field='any'
  ```
  iter-2 top-3 from the (partially-stale) report:
  - Expert_gmm: 25,312 ms HR (16-pass total) — under-counted (see above)
  - FSDP_AG: 6,221 ms HR (16-pass) = 389 ms/step (got WORSE: was 264 ms)
  - Router: 3,300 ms HR (16-pass) = 206 ms/step (was 167 ms)

  **FSDP_AG getting worse is interesting**: with the faster expert
  kernel, the AG-vs-compute overlap window shifted. Schedule-position
  lever (vs concurrency) is real; the advisor flagged this earlier.

- **Decision for next iter**: pause for human review. Two open questions:
  1. File perfsim follow-up to extend `LEAF_PATTERNS_TRAINING` to
     match gmm_v2 fusion names, so iter-3's Expert_gmm bucketing is
     accurate.
  2. Pick iter-3 lever despite the stale Expert_gmm bucket — FSDP_AG
     (now 389 ms/step) or Router (206 ms/step) are usable signals.
     Either way, fixing the bucketer first means cleaner attribution.

---

## iter 3 — Tooling: BF16 microbench grid for perfsim#10

- **Class**: Tooling.
- **Goal**: produce empirical (M, K, N, n_groups, measured_efficiency) data for
  v7x BF16 to populate `HardwareSpec.gemm_eff_curve_bf16`. Today the curve is
  empty and BF16 falls back to scalar 0.90 — search results on training-side
  workloads are not trustworthy without this. perfsim#10 is open with status
  `needs-info` pending these measurements.
- **Hypothesis (deferred to next iter)**: the curve will reproduce the v304
  empirical anchor (M=131072, K=2048, N=7168, n_groups=64, eff=0.244) extracted
  from the v304-cde-repro xplane in perfsim#10's first comment, AND show
  efficiency drop at small M (latency-bound) and rise at large M (compute-bound)
  consistent with arithmetic-intensity intuition.
- **Pre-launch state**:
  - Bootstrap: `autoperf/bootstrap.sh` reported worktree creation errors
    because primary `~/perfsim`, `~/cde`, `~/xla-shell` already have
    `autoperf-loop` checked out (each branch can only be in one worktree).
    Worked in primary directly — functionally equivalent for committing/PR'ing
    since the user's `pip install -e` already imports from autoperf-loop.
  - Step-1 ritual: only open BLOCKED row was perfsim#10 (this iter's target).
    No open autoperf-loop PRs across any repo.
- **Change (durable; landed in perfsim PR #23)**:
  - `perfsim/bench_runner.py`: extend `benchmark_gemm` with optional
    `n_groups` field (default 1 = dense). When > 1, run `jax.lax.ragged_dot`
    with G equal-sized groups; total FLOPs `2*M*K*N` unchanged.
  - `perfsim/inference/configs/v7x_microbench/v7x_4x8x8_bf16_microbench.json`:
    35-workload grid covering (1) dense at LMHead/Router/QKV across M ∈
    {1024, 4096, 16384, 65536, 131072}; (2) grouped at QKV-symmetric
    (K=N=7168) at n_groups ∈ {8, 64}; (3) grouped at Expert_gmm production
    shapes (gate/up K=2048,N=7168 and down K=7168,N=2048) at n_groups=64.
  - `perfsim/inference/scripts/run_v7x_bf16_microbench.sh`: wrapper that
    runs bench_runner, emits marker-delimited JSON, uploads to
    `gs://max-experiments/autoperf/microbench/v7x_4x8x8_bf16_<date>/`.
  - `benchmarks/k8s/perfsim-bf16-microbench.yaml`: JobSet for
    `tpu7x-standard-1t` (1×1×1) on the bodaborg-tpu7x-inference cluster
    at priority `medium`. Mirrors `qwen3coder-calibration.yaml`.
  - perfsim autoperf-loop commit: `cb67ec0`.
- **Build**: image `gcr.io/tpu-vm-gke-testing/perfsim-bench:v25-bf16-microbench`
  built and pushed.
- **Result**: **HALT — `cluster_unhealthy`**. JobSet applied to
  `bodaborg-tpu7x-inference` stayed `Pending`: all 8 1t and 6 4t nodes
  occupied by long-running medium/very-high pods (other users' uBench
  servers, ages 21h–6d). Three other medium-priority pods already
  pending 25-81 min ahead of mine. Cluster-wide TPU quota exceeded for
  autoscale. Bumping priority to `high` would preempt another user's
  running workload — refused per AGENT.md §1 ("Halt when uncertain") +
  global CLAUDE.md ("ask for confirmation before proceeding" on actions
  affecting shared state). No JIT compile, no benchmark work performed.
- **PR opened**: https://github.com/ultrons/perfsim/pull/23 with the
  durable engineering artifact (spec + extension + yaml). The bench will
  run in a future session once cluster slots free up; perfsim#10 will be
  commented at that point — not yet, since the maintainer's needs-info
  gate is the data, not the spec.
- **Pending cleanup**: the still-Pending JobSet on the inference cluster
  (`v7x-bf16-microbench`) was not deleted — harness blocks `kubectl
  delete jobset` without explicit allow. Surfaced to user in HALT.md.
- **Decision for next iter**: human picks one of the next-actions in
  HALT.md (wait off-peak, authorize priority bump, pivot to a wasteful
  4×4×4 slice on `bodaborg-super-rbq`, or a different cluster). After
  the bench runs and JSON lands in GCS, the perfsim agent ingests +
  fits the curve in a follow-up commit on `autoperf-loop`; iter-4 is
  greedy on the new top-leaf.
- **Lesson for v7x_KNOWLEDGE.md** (added §10): node-existence is not
  node-availability. Always pair `kubectl get nodes -L
  cloud.google.com/gke-tpu-accelerator,topology` with `kubectl get pods
  -A -o json | jq '.spec.nodeSelector "tpu7x"'` to see which nodes are
  occupied before submitting.

### iter-3 RESUMED (2026-05-07 evening) — bench actually completed

The HALT was premature. The original JobSet (submit 06:27Z) sat in the Kueue
queue at medium priority and was admitted ~50 min later when one of the
medium pods released; bench completed at 07:56Z. I'd missed the success
because I was off pivoting to `tpu7x-inference-cluster` (which turned out to
be broken — see below). Recovered the data from `kubectl logs` of the
Completed pod 12h after-the-fact.

**Pivots attempted while the original job was actually running**:
1. **`tpu7x-inference-cluster` (`cloud-tpu-multipod-dev` project)** — resized
   `tpu7x-np` from 0 to 1; instance stuck `PENDING/CREATING` for >1 hour with
   `CONDITION_NOT_MET: Reservation 'cloudtpu-20251017124413-573252602' is
   incorrect for the requested resources.` Cluster has been broken for days
   (unrelated `vllm-tpu` pod Pending 4d19h on the same cause). Resized back
   to 0; cluster unusable until owner fixes the stale reservation pinning.
2. **No JobSet CRD on this cluster** — discovered mid-pivot. Converted yaml
   to plain `batch/v1 Job`; kept the multipod-variant yaml at
   `~/perfsim/benchmarks/k8s/perfsim-bf16-microbench-multipod.yaml` for
   future use if the reservation gets fixed.

**Results recovered**: 35 workloads measured, all `effective_dtype: bf16`,
cv_pct < 1% on most rows. JSON + log uploaded manually to
`gs://max-experiments/autoperf/microbench/v7x_4x8x8_bf16_2026-05-07/`. Local
copy at `autoperf/microbench-results/v7x_4x8x8_bf16_2026-05-07/`.

**v304 anchor outcome**: spec target eff=0.244 (xplane-derived per #10 first
comment); microbench measured **0.6226** at the same shape. Re-normalizing
the anchor to per-core peak (the maintainer used per-chip 2307 TFLOPS as
denominator; bench is single-core 1153.5 TFLOPS) cuts the gap to 0.488 vs
0.622 (1.27×). The remaining ~30% reflects in-training overhead the
standalone microbench doesn't capture — router dispatch, neighbor-activation
HBM contention, scheduler-induced overlap inefficiency. **Curve-fitter
recommendation**: fit `gemm_eff_curve_bf16` to the microbench; treat the
in-training overhead as a separate calibration term in the MoE leaf.

**Operational gotcha discovered for future calibration runs**: image
`Dockerfile.tpu` doesn't install `gsutil`/`gcloud storage`, so the wrapper
script's GCS upload was a no-op. Recovery via `kubectl logs` worked because
the script also emits the JSON marker-delimited to stdout. Follow-up on
perfsim side: image-fix to install `google-cloud-cli` so future runs
self-publish. NOT done in iter-3 (out of scope; surfaced in #10 comment).

**PR #23 + #10 commented**:
- ultrons/perfsim#23 — bench_runner grouped-MM extension + workload JSON +
  k8s yaml. Open, not merged. Reviewer should validate FLOPs accounting and
  whether to extend the curve schema for `n_groups` discriminator.
- ultrons/perfsim#10 — measurements posted, status flipped from
  needs-info → ready for curve fit. Closing autoperf-side task.

**Decision for iter-4**: once perfsim agent fits the curve and lands an
update on `autoperf-loop`, regenerate iter-2 headroom on the v304 xplane to
refresh the trust table. Most likely top-leaf is still **Expert_gmm** (top-3
#1 post-PR#22) — but the new curve may shift other leaves. iter-4 picks the
greedy lever from the refreshed trusted set.
