# HALT — autoperf agent (dsv3_train_purefsdp track)

**Workload**: `autoperf/workloads/dsv3_train_purefsdp.yaml`
**Last iteration**: 0 attempt 2 (compile-OOM; no training ran)
**Branch**: `autoperf/dsv3_train_purefsdp`
**Halted at**: 2026-05-06
**Reason**: `compile_oom` — `ep=1 fsdp=256 tp=2` at gbs=4096 needs
126.45 GB compile-time HBM; v7x conservative limit is 94.75 GB
(+32 GB over). Resource exhaustion, not a tooling bug.

The pure-FSDP / no-EP track has now hit two distinct compile-side
failures back-to-back:

1. iter-0 attempt 1 (`fsdp=512 ep=1 tp=1`): libtpu SC-offload assertion
   (mesh-geometry-specific codegen bug).
2. iter-0 attempt 2 (`fsdp=256 ep=1 tp=2`): compile-time HBM OOM
   (every device handling all 256 experts' AG transient = 4× the v304
   per-chunk AG memory; aggregate exceeds 94.75 GB compile-conservative
   limit).

**The structural argument from the dsv3_train_full pivot is now
empirically validated**: at gbs=4096 on v7x_4x8x8, EP is not a free
choice — it's forced by HBM. v304's ep=4 fsdp=128 sits in a tight HBM
operating point that ep=1 variants don't fit into without other
concessions.

The historical pure-FSDP / fsdp=256 profiles (v262/v263/v272 at
gbs=2048; v327b at gbs=1024) all ran at HALF or QUARTER our batch size,
which halves/quarters the per-device activation memory. None apply
apples-to-apples at gbs=4096.

## Recommended next human action

Three real options. The autoperf agent's natural choice would be **(1)**
— concede the empirical evidence, return to the originally-blocked
dsv3_train_full track, and either fix the moe_xlayer_prefetch bug or
pick the next-best lever.

1. **Concede & resume dsv3_train_full** (recommended) — the no-EP
   experiment is conclusive at gbs=4096: v304's ep=4 is HBM-forced. Switch
   back to `autoperf/dsv3_train_full` branch and either:
     - 1a. Fix the moe_xlayer_prefetch bwd-transpose bug, retry the
       lever, unlock the +264 ms FSDP_AG headroom (largest single win).
     - 1b. Skip moe_xlayer_prefetch, pick Router (+88 ms) or Norms
       (+68 ms) as iter-2's lever. Smaller per-iter win but no source
       diagnostic needed.
   The dsv3_train_purefsdp track stays parked, branch preserved, two
   iter-0 attempts in iter_log.md as audit history.

2. **Try one more no-EP geometry: `ep=1 fsdp=128 tp=4`** — TP=4 splits
   activation memory across 4 cores instead of 2; per-device activation
   shrinks to ~1/4 of attempt-2's. AG transient ~unchanged. PDBS = 4096
   / 128 = 32 (matches v304). Untested; HBM math unverified. Risk:
   another compile-side failure (TP=4 with 256 experts is its own
   regime). Expected win if it compiles: a real no-EP baseline at
   gbs=4096, comparable to v304.

3. **Lower gbs to 2048 for the no-EP track** — establishes a no-EP
   baseline that matches the historical v262/v263/v272 conditions.
   Compiles for sure (those runs have real xplanes). Loses
   apples-to-apples comparison with v304's gbs=4096 — different
   workload, different optimizer state size, different HBM regime. The
   leaf-level optimization story still works within the smaller-batch
   workload, but the cross-track TPS comparison gets messy.

## What was committed on this branch

- iter-0 attempt 1 → HALT → user-elected option 2 → iter-0 attempt 2
  → this HALT. Three commits documenting the no-EP track's failure to
  establish a baseline at gbs=4096:
  - `f77ce8b`: open dsv3_train_purefsdp track + iter-0 setup
  - `06a77e3`: iter-0 attempt 1 HALT (libtpu SC-offload CHECK fail)
  - `7477cf3`: iter-0b pivot to fsdp=256 tp=2
  - (this commit): iter-0b HALT (compile HBM OOM)
- `autoperf/iter_log.md`: full diagnostic of both attempts.
- `autoperf/v7x_KNOWLEDGE.md` §5: two new cross-workload entries
  ("fsdp=512 + SC-offload → libtpu CHECK" and "ep=1 at gbs=4096 doesn't
  fit compile-time HBM"). Future agents won't re-run these.

## Open follow-ups (parallel)

- `ultrons/perfsim#1` — resolved 2026-05-06.
- `ultrons/perfsim#3` — open, non-blocking, gemm_eff calibration.

## Cluster cost so far this session

- `dsv3train-i1` (dsv3_train_full iter-1): ~7 min, broke at compile.
- `dsv3train-pf-i0` (dsv3_train_purefsdp iter-0): ~5 min, broke at compile.
- `dsv3train-pf-i0b` (dsv3_train_purefsdp iter-0b): ~3 min, broke at compile.
- **Zero training compute.** All three failures were on the autoperf
  loop's compile-side gates — exactly where AGENT.md §1 says it's
  cheaper to halt than to keep iterating without information.
