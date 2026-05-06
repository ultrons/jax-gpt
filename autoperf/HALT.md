# HALT — autoperf agent

**Workload**: `autoperf/workloads/dsv3_train_full.yaml`
**Last iteration**: 1 (halted at orient/sanity-check, before any cluster
launch — no profile, no diary entry, no measurement-side work performed)
**Branch**: `autoperf/dsv3_train_full`
**Halted at**: 2026-05-06
**Reason**: `tool_blocked_perfsim#1` — perfsim's `headroom_report.py` does
not support training-regime workloads. Three concrete gaps documented in
[ultrons/perfsim#1](https://github.com/ultrons/perfsim/issues/1):

1. `headroom_report.py:_predict_per_leaf` only calls
   `inference.simulator.run_decode`. No training-regime path is wired up,
   even though `perfsim.simulator.run` + `TrainingConfig`/`ParallelismConfig`
   exist in the core library.
2. `LEAF_PATTERNS` are vLLM-named (`VllmSharedFusedMoE`,
   `VllmRowParallelLinear`, `QKVParallelLinear`, `compute_logits_func`).
   The jax-gpt training stack uses different op names — only `jit(gmm_v2)`
   would match. The bucketer would silently drop most training-stack ops.
3. Workload yaml's `model: dsv3_671b` is not in `perfsim.specs.model.MODEL_PRESETS`.
   The closest match is `deepseek_v3` (61L, 256E, k=8, hidden=7168 — same
   shape as DSv3 671B). Need either a preset alias or a workload-yaml
   correction.

## Why halt rather than launch

Step 11 of `autoperf/AGENT.md` §3 is the headroom report — that's how
each iteration picks the next lever. With the headroom step broken for
this workload, a cluster run (5 training steps × full 4×8×8 v7x slice =
512 devices) would produce a profile we couldn't rank, so launching
would burn cluster cycles for no decision input. AGENT.md §1 explicit
principle: "Halt when uncertain. Lost cluster cycles cost real money."

## Recommended next human action

1. **If the perfsim agent fixes #1 quickly** (likely path — the changes are
   bounded: route `headroom_report` through training simulator, generalize
   `LEAF_PATTERNS`, add the preset alias):
   - `git -C ~/perfsim pull` to pick up the fix
   - mark `BLOCKED.md` row `open` → `resolved`
   - re-run iter 1 from step 1 of `AGENT.md` §3 — first cluster job will
     reproduce the v304-postrefactor baseline (full + ga=1 + n_chunks=2 +
     gbs=4096) and the headroom report will tell us where to start
2. **If the issue is going to take longer**: human can either (a) run the
   `dsv3_train_mini.yaml` workload through the existing inference-style
   path if a calibration anchor is needed, or (b) wait. There's no other
   training workload defined that would not hit the same gaps.

## What was done in this iter-1 session

- Read the four required references (`AGENT.md`, `v7x_KNOWLEDGE.md`,
  `auto-perf-guide.md`, `perfsim-protocol.md`).
- Verified perfsim presets and core APIs (`MODEL_PRESETS`, `HW_PRESETS`,
  `simulator.run`, `TrainingConfig`, `ParallelismConfig`).
- Read `~/perfsim/perfsim/FEEDBACK_TO_JAX_GPT_TRAINING.md` (perfsim agent's
  prior reply to a v304 calibration round; confirms perfsim-side training
  fixes have already landed on `auto_perf` — but the headroom-report
  wrapper is still inference-only).
- Reaped 31 stale `running` rows in `cde history` (hangover from prior
  bisection sessions; only `v341b-nchunks4-barrier` was actually still
  running). Free parallelism budget now: 1/2 slots used.
- Created branch `autoperf/dsv3_train_full`, pushed initial state.
- Filed [ultrons/perfsim#1](https://github.com/ultrons/perfsim/issues/1)
  with three gaps bundled, copy-pasteable repro, and a definition-of-done
  with three sub-criteria.
- Updated `autoperf/BLOCKED.md` with the issue row.

No source files in `jax_gpt/` were touched. No cluster job submitted. No
git history rewritten.
