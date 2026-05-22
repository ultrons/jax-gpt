# STATE — dsv3-fused-ep-moe

This file is the **kernel-phase-runner**'s working memory for this target.
Keep it concise — every line load-bearing.

```yaml
target: targets/dsv3-fused-ep-moe
spec_path: SPEC.md
spec_version: v0.6
auto_push: true

# Phases, oldest first. status ∈ {complete, deferred, blocked}.
# Each entry MUST cite a results doc + commit sha.
phases:
  - id: A-E.2
    status: complete
    summary: "v_outside fwd + bwd, v_inside Option β, cluster fwd PASS"
    results_doc: results/phase_a_e.md
    committed_sha: f392aa8
    cluster_validated: true

  - id: D.6-lite + E.3
    status: complete
    summary: "D=3840 cluster ceiling, streaming-psum-scatter standalone validated"
    results_doc: results/phase_d6_e3_cluster.md
    committed_sha: 37436d9
    cluster_validated: true

  - id: E.4
    status: complete
    summary: "Megatron column+row parallel fwd (F-sharded W + lax.psum)"
    results_doc: results/phase_e4_through_f1.md
    committed_sha: 9702ba6
    cluster_validated: true   # f1-bench-3 [megatron] FSDP=32 PASS

  - id: E.5
    status: complete
    summary: "Megatron bwd VJP via jax.vjp on JAX-only mirror"
    results_doc: results/phase_e4_through_f1.md
    committed_sha: 634e668
    cluster_validated: false  # local-only (fsdp=1/2/4 PASS)

  - id: D.6
    status: complete-with-gap   # see known_issues
    summary: "D-tiled kernel reaches D=7168 at F=128 (test shape)"
    results_doc: results/phase_e4_through_f1.md
    committed_sha: 9a27693
    cluster_validated: false

  - id: E.6-step-1
    status: complete
    summary: "Megatron with lax.psum_scatter + lax.all_gather (no overlap yet)"
    results_doc: results/phase_e4_through_f1.md
    committed_sha: ff693f5
    cluster_validated: true   # via f1-bench-3 perf table

  - id: F.1
    status: complete
    summary: "variant perf table on rbq 4x4x4 (E=32 D=2048 F=128 K=4 EP=4 FSDP=32)"
    results_doc: results/phase_e4_through_f1.md
    committed_sha: b4b63d1
    cluster_validated: true   # f1-bench-3

  - id: D.7
    status: complete                       # cluster gate PASSED on d7-fix-5
    summary: "F-tiled expert FFN; grid d-outermost fixes the D-axis RMW bug. Production F=2048 D=7168 fits 64 MB VMEM AND bit-equivalent to JAX f32 reference at production E_local=64 on real v7x (d7-fix-5: max_rel=3.28e-4 at all F_tile geometries)."
    results_doc: results/phase_d7.md
    committed_sha: 917ce01
    cluster_validated: full                # d7-fix-5 PASS: d7-prod-sanity at E_local=64, bisect-E4-D7168, bisect-E16-D2048, Ftile128, Ftile256 all PASS (max_rel <= 3.28e-4 vs JAX f32 reference)

# What to do next. The phase runner reads this and runs ONE.
next_phase:
  id: D.7-megatron-wire
  description: |
    Point `moe_block_ep_v_inside_megatron_fwd` (and the scatter variant)
    at `expert_ffn_v_outside_f_tiled` instead of `expert_ffn_v_inside`,
    unlocking production F=2048 D=7168 for the v_inside Megatron path.

    Mechanical change: the Megatron wrapper today reshapes
    `W1 (E_local, D, 2, F_shard) → (E_local, D, 2*F_shard)` before
    calling `expert_ffn_v_inside`. The F-tiled kernel takes the
    `(E_local, D, 2, F)` layout natively — skip the reshape and the
    F-axis tiling falls out automatically.

    Steps:
      1. In `build/v_inside/moe_block_ep_megatron.py`, remove the
         `W1.reshape(E_local, D, 2 * F_shard)` and call
         `expert_ffn_v_outside_f_tiled(sorted_local_tokens,
         sorted_local_eids, W1, W_d, bt=cfg.bt_ffn)` directly.
      2. Same change in `moe_block_ep_megatron_scatter.py`.
      3. Local test_g3_megatron.py at production-ish shape.
      4. Cluster gate at production shape on x8p.

    Why this is the right next step:
      - Pure plumbing; no new kernel work.
      - D.7 cluster gate proved correctness at production E_local=64
        D=7168 F=2048 (d7-fix-5 max_rel <= 3.28e-4).
      - Unblocks production F=2048 for the path the F.1 perf table
        already proved is the real W-side HBM win.
  spec_refs: ["SPEC.md §5.4 (production shape)", "build/v_inside/moe_block_ep_megatron.py"]
  evidence:
    - "D.7 kernel cluster-validated on d7-fix-5 at E_local=64 D=7168 F=2048 (max_rel=3.28e-4 at F_tile=128/256; bisect-E4-D7168 max_rel=1.76e-4; bisect-E16-D2048 max_rel=2.26e-7)"
    - "Local + cluster bit-equivalent (modulo bf16 noise) at all tested F_tile geometries"
    - "F_tile=F=2048 at production D=7168 is INTENTIONALLY untested — that's the autoperf OOM shape D.7 exists to avoid"
  blocked_by: null

# What was the D.7 ticket (preserved for traceability):
previous_next_phase:
  id: D.7
  description: |
    F-tiling for production F=2048. D.6 closes D=7168 but only at F=128
    (W1 window 3.7 MB). At production F=2048 the W1 window is
    (1, 7168, 2*2048) bf16 = 56 MiB → 112 MiB double-buffered → exceeds
    64 MiB VMEM. Need an F-output tile axis (or equivalent), being
    careful that silu(gate) * up consumes the full 2F so the activation
    layer can't be naively F-tiled.
  description: |
    F-tiling for production F=2048. D.6 closes D=7168 but only at F=128
    (W1 window 3.7 MB). At production F=2048 the W1 window is
    (1, 7168, 2*2048) bf16 = 56 MiB → 112 MiB double-buffered → exceeds
    64 MiB VMEM. Need an F-output tile axis (or equivalent), being
    careful that silu(gate) * up consumes the full 2F so the activation
    layer can't be naively F-tiled.
  spec_refs: ["SPEC.md §3.3 (gate+up matmul)", "SPEC.md §5.4 (production shape)"]
  prior_art:
    - targets/dsv3-fused-ep-moe/build/v_outside/expert_ffn_d_tiled.py  # D.6 base
    - corpus/kernels/megablox__gmm_v2.py  # emit_pipeline with K-axis streaming
    - distilled/patterns/streaming-ag-into-matmul.md  # if W1 needs to stream
  gates:
    - aot_compile@tpu7x:4x8x8: E=256 D=7168 F=2048 K=8 (autoperf's exact failing shape — Mosaic check, hardware-free)
    - ep1_exec at F=2048 small shape vs JAX reference (single-host, local 4 cores)
    - cluster_4x8x8: autoperf's exact shape on real hardware. NOW REACHABLE per ~/infra/INSTRUCTIONS.md §3 (cluster-queue has ~320 chips free; 4x8x8 = 256 chips fits). This is the strong gate — if AOT passes here we know production-shape D.7 is real.
  cluster:
    instructions: ~/infra/INSTRUCTIONS.md          # AUTHORITY — read on every submission
    context: gke_cloud-tpu-multipod-dev_us-central1_bodaborg-super-xpk-x8p
    routing: default                                # default/multislice-queue/medium → cluster-queue
    namespace: default
    queue: multislice-queue
    priority_class: medium
    slice_topology: 4x8x8                           # 256 chips, fits in cluster-queue
    cde_yaml: ~/infra/cde.yaml
    template: ~/infra/manifests/jobset.yaml.j2
    known_good_reference: ~/infra/manifests/sanity_4x8x8.yaml   # diff against this if submission misbehaves
  blocked_by: null
  evidence:
    - "Autoperf agent reproduced VMEM OOM at production shape (E=256 D=7168 F=2048 K=8 BS=4096 seq=4096 on tpu7x:4x8x8): single-expert W1 window bf16[1, 7168, 4096] = 56 MiB → 112 MiB double-buffered > 64 MiB VMEM"
    - "f1-bench-3 confirms F=128 production-class shapes work at FSDP=32 cluster scale"

# Deferred (not blocking but tracked).
deferred:
  - id: E.5-pallas-bwd
    why: "JAX-only Megatron bwd is 4× fwd on cluster vs 1.26× for v_outside. Pallas bwd would close the gap."
  - id: E.6-step-2
    why: "Fused down-matmul + streaming scatter for real comm-compute overlap. Standalone primitive validated in E.3."
  - id: D.6-d7168-cluster-smoke
    why: "Local PASS at D=7168 F=128; cluster smoke at full DSv3 shape postponed pending D.7 (otherwise W footprint OOMs)."
  - id: custom_vjp-v_inside-and-scatter
    why: "v_inside Option β + Megatron-scatter are fwd-only. Bwd would complete F.1 table."

# Cross-cutting hazards the runner should know.
known_blockers:
  - "Cluster ops authority: ~/infra/INSTRUCTIONS.md (re-read every submission — it evolves). Historical phases (A-F.1) ran on rbq-super-bodaborg/multislice-queue. NEW phases (D.7+) submit to bodaborg-super-xpk-x8p via the SAME default/multislice-queue/medium routing pattern (per the updated INSTRUCTIONS — earlier note that poc-dev was canonical was wrong)."
  - "Dynamic slice composition is supported on x8p — single template covers 4x4x4/4x4x8/4x8x8/8x8x8 via overrides.slice_topology. The 4x4x4 sub-block annotations (podset-slice-required-topology, podset-slice-size: 16) stay constant; only gke-tpu-slice-topology + parallelism change per size."
  - "Cluster-queue has ~320 chips free, so 4x4x4 (64) / 4x4x8 (128) / 4x8x8 (256) all fit. 8x8x8 (512) is close to cluster total — don't try without coordinating."
  - "Image tag must be rebuilt+pushed after every commit before submitting — cde-<sha> drift causes ImagePullBackOff. cluster-ops checks for this within 10 min of submission."
  - "Verified-working bare manifest for 4x8x8: ~/infra/manifests/sanity_4x8x8.yaml (~60s admit→run→complete). Diff against it if a submission stalls."
  - "Megatron variant: x_spec must be P('ep', None), NOT P(('ep','fsdp'), None). fsdp peers must see same tokens or psum mixes garbage. (Cost us hours in E.4.)"
  - "Bench scripts launched from inside run_g3_cluster.py must not re-call jax.distributed.initialize() — gate on jax.process_count()."
  - "Size-2 axis + multi-lane-block F_tile in 4-axis BlockSpec is broken (or at least numerically silent-wrong): `pl.BlockSpec((1, D, 2, F_tile), ...)` slicing axis 2 to peel gate vs up produces numerically WRONG results at F_tile > 128 (= more than one lane block) on real v7x, even though AOT@4x8x8 compiles cleanly and the kernel reports finite output. Local 4-core tpu7x:2x2x1 misses this (only 1 lane block fits anyway). New debugging-runbook entry needed."
  - "x8p cluster pulls images from gcr.io/cloud-tpu-multipod-dev/ — kernel-agent's existing registry gcr.io/tpu-vm-gke-testing/ is a different GCP project and likely not pullable. Always re-tag for the new cluster before submitting."

# Patterns we wish we'd captured before this session (now living in
# distilled/debugging-runbooks/ once authored).
debugging_runbook_seeds:
  - title: "wrapper looks broken but kernel + math are fine"
    body: "Symptom: full wrapper gives error >tolerance, but pure-JAX mirror and isolated kernel match. → Check sharding specs and replication contracts of every input. (Source: E.4 x_spec bug.)"
  - title: "VMEM math says fits but compile OOMs"
    body: "Symptom: hand-computed VMEM < cap but pallas_call fails RESOURCE_EXHAUSTED. → Double-buffer multiplier (×2) on every IO + output block. Output f32 blocks are often the dominant term. (Source: D.6 design path.)"
  - title: "cluster job admitted but pods Error/ImagePullBackOff"
    body: "Symptom: kueue admits, pods schedule, then never start. → Verify the image tag pushed matches the JobSet's image: field. Image tag is a content hash of the build context, not git SHA. (Source: e4-megatron-2 stall.)"
```
