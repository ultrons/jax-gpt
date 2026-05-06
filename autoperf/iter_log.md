# autoperf iter_log — dsv3_train_purefsdp

Per `~/jax-gpt/autoperf/AGENT.md` §5b. One section per iteration. The
perfsim agent reads this when an issue references "iteration N".

Sister log: `dsv3_train_full` (paused at HALT after iter-1 broke at
compile; see that branch's iter_log + HALT.md). This track was opened
2026-05-06 to verify pure-FSDP empirically rather than concede the
ep=4 headroom report's leaf-level math.

---

## iter 0 — baseline establishment (no autoperf change)

iter-0 in this track is **not** a free profile: there is no
v304-postrefactor pure-FSDP profile (the v288/v318/v327/v336 attempts
predate v304-postrefactor's gmm_v2 / einsum deltanet / auxar
global-psum fixes), so we have to launch a cluster cycle to capture
one.

- **Workload**: `autoperf/workloads/dsv3_train_purefsdp.yaml`
  baseline overrides (no autoperf changes — vanilla pure-FSDP run).
- **Image**: TBD on `cde build` (auto-pinned hash of current commit).
- **Cluster context**: `gke_cloud-tpu-multipod-dev_us-central1_bodaborg-super-rbq`.
- **Expected step time**: unknown. v304 baseline is 37 s/step at 1770
  TPS/chip. Pure-FSDP delta is the experiment's whole point — record
  the steady-state TPS/chip as the new baseline number.
- **Pre-launch risks (predicted)**:
  - **HBM peak**: with 256 experts gathered per FSDP_AG and n_chunks=2,
    transient gathered tile ≈ 128 × F × D × 2 = 3.6 GiB peak (vs ep=4's
    1.78 GiB / chunk). v304 runs near the 101.7 GB runtime limit; +1.8
    GiB peak HBM might trigger `RuntimeProgramAllocationFailure`.
  - **NaN at PDBS=8**: Bug 4 says PDBS=1 NaNs at mini config; PDBS=8 at
    full has not been explicitly tested but should be well above the
    floor. Watch for fwd or bwd NaN at step 1.
  - **Compile-time blowup**: AG over fsdp=512 (vs 128) is a wider mesh
    collective; XLA may take longer to schedule. If compile exceeds
    20 min, halt as a hang.
- **Decision**: launch.
- **Result**: **FAILED at compile (no training step ran).** `cde run dsv3train-pf-i0`
  → `Finished=True (Failed)` after first attempt; reason `ReachedMaxRestarts`.
  Init succeeded (all 58 MoE layers initialized; 670B params @ 1341 GB bf16
  reported); compile started, then libtpu hit a fatal assertion:
  ```
  F0506 22:22:28.728338 async_collective_start_emitter.cc:65]
  Check failed: smem_state_shape.dimensions(0) == num_states (26 vs. 6)
  ```
  Stack: `xla::jellyfish::AsyncCollectiveStartEmitter::Emit() →
  LoweringEmitter::HandleCustomCall → HandleFusion`. This is libtpu
  internal code-gen for the SparseCore async-collective offload path —
  the manifest's `--xla_tpu_enable_sparse_core_collective_offload_*`
  flags route AG/RS through SC, and SC's emitter expected `num_states=6`
  but got `smem_state_shape[0]=26` at the fsdp=512 mesh geometry.
  - **Not a jax-gpt source bug** — the same code path compiles fine at
    v304's ep=4 fsdp=128. The bug is the SC-offload emitter's assumption
    breaking at the wider mesh shape.
  - **Not a jaxlib/Python issue** — fatal in-libtpu CHECK, not a Python
    exception. SIGABRT in the trainer pod.
  - Cluster cost: ~3 min admission + ~1.5 min compile attempt. ~30 pods
    initialized, all crashed with the same assertion. No training compute.
- **Halt reason**: `tool_blocked_libtpu` (compile-side; libtpu is binary
  pinned at dev20260417, autoperf doesn't own it). Workload yaml is
  unchanged; revert isn't applicable here.
- **What this rules out**: `dp=1 fsdp=512 ep=1 tp=1` with the current
  v304 LIBTPU_INIT_ARGS won't compile. Either (a) disable
  SC-offload flags for this workload (manifest template edit), or
  (b) pivot to a different parallelism that doesn't trip the assertion
  (e.g., v321's `ep=1 fsdp=256 tp=2`, or `dp=2 fsdp=256 ep=1 tp=1`).
- **Iter-0 retry strategy**: human chose HALT.md option 2 (pivot to
  v321 geometry `ep=1 fsdp=256 tp=2`). See iter-0 attempt 2 below.

---

## iter 0 (attempt 2) — pivot to ep=1 fsdp=256 tp=2

Following HALT.md option 2 from attempt 1's libtpu compile fail. Same
"no EP" intent, different geometry. v321 / v262 / v263 / v272 historical
runs in `gs://max-experiments/dsv3/profiles/` all used this geometry and
have real xplanes — empirically known to compile, even if the TPS numbers
predate v304-postrefactor's improvements.

- **Workload**: `autoperf/workloads/dsv3_train_purefsdp.yaml` updated to
  `dp=1 fsdp=256 ep=1 tp=2`. Branch name kept as `autoperf/dsv3_train_purefsdp`.
- **Image**: TBD on `cde build`.
- **Per-device economics**:
  - PDBS = gbs/fsdp = 4096/256 = **16** (vs v304's 32, vs attempt-1's 8)
  - Expert AG: 256 experts gathered, mesh fsdp=256 (vs attempt-1's 512)
  - TP comm is new: TP=2 introduces TP_AR after attn/MoE blocks
- **Pre-launch risks**:
  - **HBM**: TP=2 splits weights across 2 cores; per-device weight
    footprint similar to v304 (~44 MiB per expert). Expert AG transient
    is the same total volume as attempt-1 (256 × F × D × 2 = 7.13 GiB),
    just gathered over 256 instead of 512 — could be larger per-device
    receive buffer.
  - **TP_AR overhead**: with TP=2, each layer's attention and MoE have a
    TP all-reduce. This is a NEW comm leaf vs v304 (which has tp=1).
  - **Compile time**: pivoting from attempt-1 with same image is fast
    re-compile (Docker layers identical); JAX cache may not transfer if
    sharding differs.
- **Decision**: launch.
- **Result**: TBD.
