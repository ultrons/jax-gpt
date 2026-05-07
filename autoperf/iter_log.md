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
- **Result**: **FAILED at compile (no training step ran).**
  `cde run dsv3train-pf-i0b` → `Finished=True (Failed)`. Init succeeded
  (all 58 MoE layers, 670B params reported); compile started, then JAX
  raised `CompileTimeHbmOom`:
  ```
  JaxRuntimeError: RESOURCE_EXHAUSTED: E1000: CompileTimeHbmOom:
  XLA:TPU compile permanent error. Ran out of memory in memory space
  hbm. Used 126.45G of 94.75G hbm. Exceeded hbm capacity by 31.70G.
  ```
  Compile-time conservative HBM limit is 94.75 GB (v7x_KNOWLEDGE.md §1);
  we needed 126.45 GB — **+32 GB over budget**.
  - **Why this fails vs v304** (which fits at the same gbs=4096):
    - v304: ep=4 → 64 experts AGed per FSDP shard; per-chunk transient
      ~1.78 GiB.
    - iter-0b: ep=1 → all 256 experts AGed; per-chunk transient ~3.56 GiB
      (4× larger). With 61 layers + activations + program binary, the
      total compile-time peak crosses the 94.75 GB conservative limit.
    - The historical v262/v263/v272 (`fsdp=256 tp=2`) runs that fit DID
      run at **gbs=2048** (PDBS=8) — half the activation memory of our
      gbs=4096 (PDBS=16). v321 likely the same. Apples-to-oranges
      comparison; their compile-time HBM was lower.
  - **Empirical conclusion**: ep=1 at gbs=4096 on v7x_4x8x8 doesn't fit
    in compile-time HBM at this code revision. The structural argument
    that EP relieves HBM pressure (ep=4 → 4× smaller AG transient) is
    confirmed empirically.
- **Halt reason**: `compile_oom` (resource exhaustion, not a tooling bug).
  The no-EP track can't establish a baseline at gbs=4096 in this geometry.
- **Iter-0 retry strategy**: needs human direction. Options:
  - Pivot to ep=1 fsdp=128 tp=4 (more TP cuts AG transient further; same
    devices). Untested at this code revision; HBM math unverified.
  - Lower gbs to 2048 for the no-EP track (validates non-EP empirically
    but loses apples-to-apples vs v304's gbs=4096).
  - Concede the structural argument; resume dsv3_train_full track with
    iter-2 lever pick (Router or Norms), or fix the moe_xlayer_prefetch
    bwd-transpose bug to unlock the +264 ms FSDP_AG lever.

---

## iter 0 (attempt 3) — pivot to ep=1 fsdp=128 tp=4

User chose HALT.md option 2 from attempt 2's compile OOM. TP=4 splits
the F dim 4 ways, dropping per-device AG transient to ~1.79 GiB/chunk
— matching v304's 1.78 GiB/chunk exactly. PDBS = 4096/128 = 32, also
matching v304. The per-device HBM profile should be near-identical
to v304; only the comm pattern differs (no EP_AG_dispatch, but a new
TP_AR after attn/MoE).

- **Workload**: `autoperf/workloads/dsv3_train_purefsdp.yaml` updated
  to `dp=1 fsdp=128 ep=1 tp=4`.
- **HBM expectation**:
  - AG transient/chunk: 256 experts × F/4=512 × D=7168 × 2 = 1.79 GiB
    (matches v304 exactly).
  - Activations/layer: PDBS=32 × seq=4096 × D=7168 × 2 = 1.88 GiB
    (matches v304 exactly).
  - Per-device weights: 256 × 2048 × 7168 × 3 × 2 / (128 × 4) = 88 MiB
    (matches v304).
- **Risk**: TP=4 has not been tested at this code revision. The
  program binary may be different size at TP=4 sharding. SC-offload
  flags should still apply (fsdp=128 axis is the same as v304).
- **Decision**: launch.
- **Result**: **FAILED in Python init (before compile, before cluster
  admission overhead).** `cde run dsv3train-pf-i0c` → trainer pod
  hit a `ValueError` immediately:
  ```
  ValueError: tp=4 requested but cores axis nc=2 doesn't match.
  Implement multi-axis TP placement if needed.
  ```
  Source: `jax_gpt/models/dsv3/model.py:247` in
  `ShardConfig.create_mesh`. The mesh constructor has a hardcoded
  invariant: TP must equal the v7x cores-per-chip axis (`nc=2`). TP=4
  would need "multi-axis TP placement" (not implemented). So
  effectively, **TP ∈ {1, 2} in this codebase**, full stop.
  - Cluster cost: trivial — error in Python init, no compile, no
    significant admission overhead.
  - This is now the third failure on the no-EP track and the second
    consecutive structural one. Combined with attempt 2's HBM OOM, it
    means: at TP ≤ 2, no-EP doesn't fit at gbs=4096, and TP ≥ 4 isn't
    code-supported. The space is exhausted from the autoperf side.
- **Halt reason**: `code_constraint` (jax-gpt source doesn't support
  the required TP). Workaround would be a non-trivial source change
  (implement multi-axis TP placement), comparable in scope to fixing
  the moe_xlayer_prefetch bwd-transpose bug on the dsv3_train_full
  track. Both are jax-gpt-side surgery; neither is a per-leaf lever.
- **Conclusion superseded**: see iter 0 attempt 4. The "TP > 2 not
  supported" maxim was over-broad; on closer reading the constraint was
  "TP must equal cores axis", which is fixable with a small patch to
  `create_mesh` (extending the single-axis-match logic that EP already
  has, to TP).

---

## iter 0 (attempt 4) — patch `create_mesh` to allow TP-on-X, retry

User pointed out the iter-0c failure was an implementation choice in
`ShardConfig.create_mesh` (TP hardcoded to cores axis), not a fundamental
limitation. The constraint at `model.py:245-249` was a missing single-axis
match for TP — EP already had it (lines 258-262), TP didn't.

**Patch** (`model.py:230-303`): replaced the `tp_takes_cores` hardcode with
single-axis TP placement: prefer cores when `nc == tp` (preserves all
existing v304 / v321 behavior), else scan X/Y/Z for any axis whose size
matches `self.tp`. Multi-axis TP placement is still not implemented (would
need TP spanning multiple physical axes — not what we need here).

Sanity-check traces on v7x_4x8x8: `tp=1 → None`, `tp=2 → C`,
`tp=4 → X`, `tp=8 → Y`. No regressions for v304 (tp=1) or v321
(tp=2). For the present workload, `tp=4 → X` puts TP on the X(4) axis
of the physical mesh.

- **Workload**: `autoperf/workloads/dsv3_train_purefsdp.yaml` unchanged
  from attempt 3 (still `dp=1 fsdp=128 ep=1 tp=4`).
- **HBM expectation**: same as attempt 3's analysis — per-device profile
  matches v304 by construction (AG transient 1.79 GiB/chunk, PDBS=32,
  weights 88 MiB/device).
- **TP perf caveat**: TP-AR now lands on ICI (X axis) instead of D2D
  (cores). v7x_KNOWLEDGE.md §1 says w-axis ICI is 7-13× faster than
  others for D2D, but TP-AR happens after every attn AND every MoE
  block × 61 layers × fwd+bwd. Off-chip TP-AR will show up as a much
  larger TP_AR_* leaf than ep=4 baseline's (which had no TP). This is
  the empirical question: whether saving EP_AG_dispatch (~25 ms/step)
  beats paying TP_AR on ICI.
- **Risk**: TP=4 has not been tested at this code revision. The
  program binary at TP=4 sharding may differ in size; expert sharding
  patterns may interact differently. Watch for compile errors or NaN
  signals.
- **Decision**: launch.
- **Result**: **`create_mesh` patch worked** — the program reached XLA's
  HLO allocator (proving TP-on-X mesh construction is correct). Compile
  then OOM'd with **even more HBM excess than attempt 2** (135.89 G vs
  126.45 G; +41 GB over the 94.75 GB conservative limit). The HALT.md
  prediction "TP=4 cuts AG transient to v304-equivalent" was **wrong**;
  the actual XLA-allocator data shows a different root cause.

  Top allocations at OOM (extracted from rank-0 pod log):
  ```
  1. 7.00 GB weight_allgather_f shape bf16[1,256,2048,7168]
  2. 7.00 GB reduce_sum shape bf16[524288,7168]
  3. 7.00 GB copy of all-gathered weights bf16[256,2048,7168]
  4. 7.00 GB weight_allgather_f (2nd copy)
  5-10. 7.00 GB each (more remat copies + bwd ag)
  11-13. 7.00 GB each (gather_offload_custom_fusion, all 256 experts × D)
  ```
  Just the AG transient + remat copies stack to ~35 GB before activations.

- **What I had wrong about TP=4 cutting the AG transient**: I assumed TP
  shards the AG output (so per-device transient = `#experts × F/TP × D`).
  The XLA allocator shows the AG materializes the **full pre-TP-split
  tensor** `bf16[1,256,2048,7168]` = 7 GB, then TP-sharding happens
  downstream. So AG transient is determined entirely by
  `#experts_per_FSDP_shard = total_experts / EP`. **TP doesn't help.**
  Only EP reduces the per-device expert count.

- **Empirical conclusion (now definitive at HLO-allocator level)**:
  no-EP at gbs=4096 on v7x_4x8x8 cannot fit compile-time HBM, regardless
  of TP/FSDP geometry. The 7 GB AG transient × 5+ live copies in bwd-
  transpose alone exceeds the +41 GB excess. EP=4 in v304 is HBM-forced
  by the linear scaling of AG transient with `#experts/EP`.

- **Other knobs that won't help (verified by reading train.py CLI flags)**:
  - `--moe_no_weight_ag`: gated on `EP>1+TP>1`. We have EP=1.
  - `--moe_shard_e_with_fsdp`: shards a different axis but total AG
    bytes are unchanged.
  - `--moe_shard_d_with_fsdp`: same total bytes.
  - `--moe_n_chunks > 2`: Bug 3 (n_chunks=4 NaNs in bwd). Off-limits.

- **Halt reason**: `track_exhausted` (now genuinely conclusive — four
  attempts, four distinct walls, with the final attempt providing the
  HLO-allocator data that explains why none of the others could have
  worked either). The structural argument from the original ep=4
  → no-EP pivot is now backed by concrete XLA evidence, not just
  reasoning.

- **The patch stays** — `create_mesh` extension to support single-axis
  TP placement on X/Y/Z is a real correctness improvement (TP > nc was
  silently rejected before, even when a single-axis match exists). It
  just doesn't unlock no-EP at this batch size; the limit is HBM, not
  mesh construction.
