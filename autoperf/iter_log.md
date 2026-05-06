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
- **Result**: TBD.
