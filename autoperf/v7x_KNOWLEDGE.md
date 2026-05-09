# v7x_KNOWLEDGE.md — what we know about v7x + JAX/XLA

The autoperf agent's anti-hallucination ledger. **Read this at session start.**
**Append to this when you learn something new.** Do not delete entries; mark
them stale or contradicted with a follow-up entry.

This file complements (does not duplicate):
- **perfsim** — encodes the *numerical* TPU model (HBM BW, MXU shape, kernel-only ceilings)
- **`~/.claude/CLAUDE.md`** (global) — JAX/TPU/Pallas rules that apply across all projects
- **`~/jax-gpt/CLAUDE.md`** (repo) — model file paths, build commands

This file holds the *operational, narrative* knowledge that lets a fresh agent
make sensible experiments without re-discovering known-broken paths.

---

## 1. Hardware

**v7x physical:**
- 2 cores per chip. (Common footgun: GKE machine names like `tpu7x-standard-4t`
  count CHIPS = 4. JAX/vLLM see CORES = 8. A single 4t host is TP=8 single-host.)
- MXU: **256×256** per core. (NOT 128 — corrected 2026-05-04. perfsim and any
  tile-tuning code must respect this.)
- Per-core HBM: **101.733 GB** runtime, **94.75 GB** compile-time conservative limit
- Per-core HBM BW: **3.65 TB/s** (= shared-per-chip 7.3 TB/s ÷ 2 cores)
- ICI: w-axis is **7-13× faster** for D2D than other axes — use it for bandwidth-heavy
  comm dimensions (typically EP for MoE)

**Production cluster topology**: `4×8×8 = 256 chips = 512 devices`.
- Mesh assignment used by v304: `X=4 Y=8 Z=8 C=2` → `EP=X(4), FSDP=Y·Z·C(128), TP=—`
- 512 devices total (256 chips × 2 cores)

**perfsim repo path** (canonical): `~/ml-experiments-perfsim/`. Host
convention is to set up `~/perfsim` → `~/ml-experiments-perfsim` as a
symlink (`ln -sfn ~/ml-experiments-perfsim ~/perfsim` once per machine).
AGENT.md files use the symlinked path `~/perfsim/perfsim/...` for
readability; fall back to the canonical path on machines without the
symlink. Known duplicate: `~/ml-experiments/perfsim/` is a separate
git checkout of currently-identical perfsim content — treat
`~/ml-experiments-perfsim/` as authoritative for autoperf purposes.

**Cluster context**: `gke_cloud-tpu-multipod-dev_us-central1_bodaborg-super-rbq`
- Namespace: `poc-dev`
- Priority class: `poc-dev-priority`
- Other contexts (don't use unless explicitly required):
  - `gke_cloud-tpu-multipod-dev_us-central1_ninja-v7x-64-spot` — older spot cluster
  - `gke_cloud-tpu-multipod-dev_us-central1_tpu7x-inference-cluster` — vLLM inference (no JobSet CRD)
  - `gke_cloud-tpu-multipod-dev_europe-west4_mlperf-rl-v5p-128` — v5p, eval only

---

## 2. Software stack pin (as of 2026-05-05)

```
jax     0.10.0
jaxlib  0.10.0
libtpu  0.0.41.dev20260417+nightly   ← reverted from dev20260504 due to Bug 1
```

**Why pinned**: `dev20260504` introduced a step-1 NaN at the production baseline
(`full + grad_accum=1`). See Bug 1 below. Don't bump libtpu without:
1. running `full + grad_accum=1` on a known-good config and verifying step-1 finite
2. updating this section + Bug-table date

**Image (known-good baseline)**: `gcr.io/tpu-vm-gke-testing/jax-gpt-dsv3:cde-9ea30df`
- builds from commit `b178a67` (libtpu reverted)
- v304-postrefactor reproducible: full + ga=1 + n_chunks=2 + gbs=4096 → step-1 loss 415.491

---

## 3. Known active bugs (workaround in place)

| ID | One-line | Workaround | Filed | Discovered |
|---|---|---|---|---|
| Bug 1 | NEW libtpu (dev20260504) NaN at production baseline | Pinned libtpu to dev20260417 | (none — fixed in-tree) | 2026-05-05 |
| Bug 2 | scan grad_accum > 1 + remat + vjp produces NaN cotangents | Use `--grad_accum=1` | task #43 | 2026-05-05 |
| Bug 3 | n_chunks=4 MoE body produces NaN cotangents in bwd (any libtpu) | Use `--moe_n_chunks=2` | task #44 | 2026-05-05 |
| Bug 4 | PDBS=1 (gbs=512 at fsdp=128 ep=4) produces NaN at mini config | Use `--gbs ≥ 1024` (PDBS ≥ 2) | task #45 | 2026-05-05 |
| Bug 5 | `attn_proj_out` checkpoint offload produces NaN at step 1 (production scale) | Don't add to offload list (jax-gpt model.py:3054) | jax-gpt#2 | 2026-05-08 |
| Bug 6 | `jax.checkpoint(prevent_cse=True)` produces NaN at step 1; same family as #2 | Keep `prevent_cse=False` (current code default) | jax-gpt#3 | 2026-05-08 |

**Implication for autoperf**: the levers `moe_n_chunks` and `grad_accum` are
**off-limits** (locked at 2 and 1 respectively). `attn_proj_out` offload and
`prevent_cse=True` are also off-limits until jax-gpt#2/#3 are fixed. Don't
propose changes to these until the corresponding bug is closed. The workload
yamls bake these in.

### Operational findings (NOT bugs, just shape-of-the-bottleneck)

- **moe_gmm_ag chunk pipelining without compute/collective overlap** (iter-12
  HLO finding, 2026-05-09). The `_expert_mlp_gmm_ag_body` runs n_chunks=2;
  HLO shows 2 distinct `reduce-scatter` ops at IDs 84/94 (single-operand
  each — XLA combiner is correctly **not merging** them). However, each
  reduce-scatter has ~6.8 s self-time; combined ~13.4 s exposed stall
  (~2.4% of step). XLA scheduler is failing to overlap these collectives
  with the TC compute on the other chunk. The `optimization_barrier` was
  intentionally REMOVED in v323 to allow overlap (model.py:1745-1748
  comment), but the overlap isn't achieving its intended value. Multi-iter
  scope to investigate (collective ordering hints, alternative barrier
  placement, staggered_call wrapper). **Implication**: ~2.4% TPS sits in
  this overlap gap and is recoverable IF the scheduling can be coerced.
- **White paper as a lever-source has training-coverage gaps** (iter-12
  meta-finding, 2026-05-09). The Qwen3.5-Coder white paper's optimization
  catalog skews INFERENCE-side. Training-specific bottlenecks
  (autodiff transpose of collectives, recompute-policy levers,
  weight-sharding-for-grad accumulators) are systematically
  underrepresented. AGENT.md §13 "≥1 white-paper-pattern attempt" before
  cumulative halt should be reinterpreted on training workloads as
  "diagnose-from-white-paper-and-find-it-doesn't-apply" being a valid
  attempt outcome.

---

## 4. JAX/XLA operational pitfalls (learned the hard way)

### `jax.debug.print` perturbs XLA scheduling
- Even with `ordered=False`, the side-effect token can flip XLA's scheduler
  and expose latent NaN-producing patterns
- Observed 2026-05-04 in `_expert_mlp_gmm_ag_body` (added prints → fwd NaN at n_chunks=4)
- Observed 2026-05-05 in `_vocab_ce_bwd` (always-on prints → step-1 NaN at full+ga=1)
- **Rule**: gate every `jax.debug.print` behind an env var. Default OFF in checked-in code.
- Pattern:
  ```python
  if os.environ.get("MY_FINITE_CHECK", "").lower() in ("1", "true", "yes"):
      jax.debug.print(...)
  ```

### scan + value_and_grad inside body + remat = NaN
- `jax.lax.scan(_accum_body, ...)` where `_accum_body` calls
  `jax.value_and_grad(compute_loss with checkpointed layers)` produces NaN
  cotangents at random init
- Restructuring (vjp outside scan, MaxText-style params-in-carry) doesn't help
- Python for-loop avoids it (Bug 2 mini repro confirmed)
- **Conclusion**: the toxic combo is `scan + checkpoint + outer-vjp` at any
  vjp nesting. Workaround: use grad_accum=1.

### argsort / fancy indexing in vjp path
- `arr[idx]` for integer idx → bwd via `scatter_p`. On TPU, scatter_p has
  known correctness/performance issues at certain shapes.
- MaxText hand-writes a custom_vjp on `_sort_activations_custom` (uses
  `argsort(idx)` to invert) specifically to avoid this.
- Our routing (`moe_routing` in dsv3) uses `sort_key_val` (sort op, not gather)
  to dodge the SC packed-element-gather error at TP=1. Different bug, same
  family of TPU-fancy-indexing landmines.

### `dataclasses.replace()` doesn't free HBM
- The old object stays live until ref count hits 0 AND JAX deallocates.
- To actually free a JAX array: `del x; gc.collect(); jax.effects_barrier()`
- Critical during staged setup (weight loading, cache transitions). See
  `~/.claude/CLAUDE.md` global JAX HBM section.

### `donate_argnums` on benchmark JIT breaks multi-run loops
- The donated buffer is consumed; subsequent calls receive a deleted buffer.
- For benchmarks: separate non-donating warmup + donating hot-loop.

### `RuntimeProgramAllocationFailure`
- Program binary (~15 GB for DSv3) needs CONTIGUOUS HBM at runtime
- Triage: `jax.live_arrays()` to find what's still live; `del` + `gc.collect()`
  before compilation step
- HLO temps scale with `page_B = batch_size / micro_batches` (paged-decode case)

---

## 5. MoE / DSv3-specific operational knowledge

### Sharding that works (v304 production baseline)
- `dp=1, fsdp=128, ep=4, tp=1` on 4×8×8 v7x
- `--config=full --gbs=4096 --grad_accum=1 --moe_n_chunks=2 --gradient_checkpoint --no_cp`
- `--optimizer=sgd --grad_clip=1.0` (AdamW now also gets grad_clip via `_maybe_clip_grads`)
- Throughput: **1770 TPS/chip @ 28.6% MFU** (v304-postrefactor)

### Adding `--moe_use_gmm_v2` (autoperf iter-2, 2026-05-07): +6.6% TPS
- Same baseline + `--moe_use_gmm_v2`: **1882 TPS/chip @ 30.5% MFU**
  (step 34.65 s vs 37.0 s baseline; loss 415.46 vs 415.49 — within bf16
  tolerance).
- Routes the 3 ragged-dots/chunk through Pallas `gmm_v2_train` /
  `gmm_v2_fused_silu_train` (gate+up+silu fused into 2 calls). Bwd via
  `jax.vjp` on `jax.lax.ragged_dot` reference.
- Required two fixes that landed alongside the experiment:
  - perfsim#4 (gmm_ag `batch_sharded_by_ep` wiring) — without it,
    Expert_gmm's predicted time was 4× too high and the leaf appeared
    masked (headroom = 0).
  - jax-gpt `model.py:1793`: `from kernels.gmm_v2_train` →
    `from .kernels.gmm_v2_train` (relative import). The bare-`kernels`
    form had been broken whenever `cfg.moe_use_gmm_v2=True` was set.
- **Bucketer caveat for next iter**: `LEAF_PATTERNS_TRAINING` rule for
  Expert_gmm matches `ragged-dot`-named fusions; gmm_v2's Pallas
  custom_call has a different fusion name. So Expert_gmm measured time
  in any post-gmm_v2 xplane is under-counted; iter-3 either files a
  perfsim follow-up to extend the pattern, or picks from the
  remaining (correctly-bucketed) leaves.

### Headroom-leaf trust state (updated 2026-05-07 post-PR#22)

**Update:** ultrons/perfsim PR #22 landed both #19 (per-op model_builder
port) and #7 (thin swap of headroom_report's training path to
`model_builder`). The 25-48% sim-vs-bld divergence on attention compute is
now closed within ±15% on the v304 xplane. **All compute leaves are now
TRUSTED.** The trust state below reflects the post-#22 ratios.

| Leaf | Trust | Pre-#22 ratio | Post-#22 ratio | Note |
|---|---|---|---|---|
| `Expert_gmm` | TRUSTED | 1.12 | **1.18** | NEW top-headroom leaf (positive headroom). Bucketer caveat for post-gmm_v2 xplanes still applies — `LEAF_PATTERNS_TRAINING` may under-count `tpu_custom_call` fusion names. |
| `Attn_scores` | TRUSTED | 0.69 | **0.82** | Recovered post-port. |
| `O_proj` | TRUSTED | 0.75 | **0.92** | Recovered post-port. |
| `QKV_proj` | TRUSTED | 0.40 | **0.48** | Largest remaining gap; partial calibration debt expected to close as more BF16 microbench data lands (perfsim#10). |
| `EP_AG_dispatch` | TRUSTED | 0.94 | **0.99** | Tightened. |
| `FSDP_AG` | TRUSTED | — | — | Comm leaf; iter-2 saw 264→389 ms schedule-position shift (perfsim#16 ADR-002 design pass; impl deferred). |
| `Router` | TRUSTED with caveat | 8.88 | tbd | Re-verify on next iter-3 headroom report. |
| `Norms` | **NOT TRUSTED** (post-PR#22 regression; perfsim#25 filed 2026-05-07) | 1.28 | 1994× | predicted collapsed to 0.1 ms (was 207 ms iter-2). Top-3 inclusion is an artifact; do NOT pick as iter target until #25 closes. |
| `Embed_lookup` | TRUSTED | — | — | predicted=0 (out of scope), measured small. |
| `LMHead` | TRUSTED with caveat | 0.25 | tbd | Re-verify on next iter-3 headroom report; not explicitly in PR#22's per-op leaf list. |

**Top-3 ranking shifted** from pre-#22 `[FSDP_AG, Router, Norms]` (small
comm/elem leaves dominating when all compute ratios were <1.0) →
**`[Expert_gmm, Norms, FSDP_AG]`** post-#22. Expert_gmm now has positive
headroom — exactly the diagnostic autoperf wanted from the start.

**Implication for iter-3 lever pick**: 
- **First action: launch the BF16 microbench grid** (autoperf-side task
  blocking perfsim#10; see BLOCKED.md). Tooling-class iteration; no on-
  cluster perf measurement. Output unblocks the only remaining open
  perfsim issue.
- **After microbench lands**: regenerate iter-2 headroom report against the
  v304 xplane to refresh the trust table (especially Router/Norms/LMHead
  which need fresh post-#22 ratios), then pick the top-headroom lever from
  the now-fully-trusted set. **Most likely candidate is Expert_gmm**
  (top-3 #1 post-#22) via further kernel/scheduling work.
- **Alternative if Expert_gmm has no remaining lever**: FSDP_AG (schedule-
  position experiment, exposed by iter-2's 264→389 ms shift) or Router
  (investigate the 8.88× ratio if it persists).

### iter-3 BF16 microbench landed (2026-05-07)

35-workload BF16 grid measured on `bodaborg-tpu7x-inference` 1×1×1 single
chip. Results at
`gs://max-experiments/autoperf/microbench/v7x_4x8x8_bf16_2026-05-07/` and
`autoperf/microbench-results/v7x_4x8x8_bf16_2026-05-07/`. cv_pct < 1% on
most rows, all `effective_dtype: bf16` (no fp8 fallbacks).

**Per-core efficiency convention (from perfsim agent's PR#23 review)**:
`HardwareSpec.bf16_tflops_per_chip = 2307` is per-CHIP (both v7x cores
combined). `GemmEfficiencyConfig` values are fractions of per-CORE peak
(`bf16_tflops_per_chip / d2d_size = 1153.5 TFLOPS`). When deriving
efficiency from xplane data: achieved-per-chip / peak-per-chip equals
achieved-per-core / peak-per-core BY SYMMETRY, since both cores contribute
equally to gmm_ag in production. Don't double-count the d2d_size factor.

**Top-leaf efficiency anchors** (per-core, BF16, the new
`gemm_eff_curve_bf16` ground truth):

| Block | Shape (K, N) | n_groups | M=1024 | M=4096 | M=16384 | M=65536 | M=131072 |
|---|---|---|---|---|---|---|---|
| LMHead dense | (7168, 129280) | 1 | 0.805 | 0.768 | 0.770 | 0.783 | 0.786 |
| QKV dense    | (7168, 7168)   | 1 | 0.764 | 0.755 | 0.759 | 0.770 | 0.743 |
| Router dense | (7168, 64)     | 1 | 0.025 | 0.100 | 0.131 | 0.145 | 0.147 |
| QKV grouped g=8  | (7168, 7168) | 8  | 0.233 | 0.469 | 0.613 | 0.669 | 0.674 |
| QKV grouped g=64 | (7168, 7168) | 64 | 0.036 | 0.131 | 0.470 | 0.558 | 0.602 |
| Expert gate/up   | (2048, 7168) | 64 | 0.036 | 0.138 | 0.494 | 0.577 | 0.623 |
| Expert down      | (7168, 2048) | 64 | 0.036 | 0.130 | 0.459 | 0.556 | 0.611 |

**iter-3 → iter-4: the "2.5× ceiling-vs-measured" framing has been
RETRACTED** as apples-to-oranges. iter-4's bisection (Tooling iter,
2026-05-07) of `moe_experts/moe_gmm_ag` on the iter-2b xplane via
`xla_shell list_sources` showed:
- The microbench (iter-3 PR#23) measured `jax.lax.ragged_dot` (Mosaic GMM,
  the **pre-iter-2** kernel) FORWARD ONLY at 0.6226 efficiency.
- The v304 xplane "0.244 in-training efficiency" derivation took total
  `moe_gmm_ag` per-call wall time and divided by ragged_dot fwd FLOPs —
  but that total time **includes dispatch, scatter, weight AG, AND the
  backward path through remat**. The factor-of-~2 from forward to
  backward (with remat=full) accounts for most of the gap.
- iter-2b is on **gmm_v2 Pallas**, not ragged_dot. Neither side of the
  iter-3 ratio is current for iter-2b state.

**Real per-band breakdown of `moe_experts/moe_gmm_ag` on iter-2b** (full
detail in `research/dsv3/iter4_moe_gmm_ag_bisection.md`):

| band | ms/step | what |
|---|---|---|
| total moe_gmm_ag | 16,656 | 48.1% of 34.65s step |
| fwd: jit(gmm) (gmm_v2 kernel) | 1,845 | 33.9% of fwd; **at-ceiling** per microbench bucket |
| fwd: scatter (EP psum_scatter) | 1,685 | 31% of fwd; iter-5 candidate A |
| fwd: ep_token_gather (dispatch AG) | 998 | 18% of fwd |
| fwd: weight_allgather (FSDP AG) | 389 | matches FSDP_AG leaf in headroom |
| fwd: other (gather, shard_map, jvp) | 519 | mostly small |
| **bwd**: transpose(moe...) | 7,913 | XLA autodiff bwd-transpose pass |
| **bwd**: jvp() | 3,306 | cotangent prep + accumulation |

**gmm_v2 forward kernel is at-ceiling.** Per-call forward time ≈ 663 µs;
microbench 0.61–0.62 efficiency at this shape; back-of-envelope agrees
within sigma. **Tile-tuning gmm_v2 forward is unlikely to land >5–10% on
the kernel call.**

**Backward path is dominant** (67% of expert time). With remat=full,
forward is recomputed in bwd, then actual bwd compute runs.
`gmm_v2_train.py:33-61` has tokamax-tuned tile sizes (`_gmm_tiles` /
`_tgmm_tiles`) for 4 specific (M,K,N) shapes; unmatched shapes use a
generic fallback. **HLO inspection of bwd needed to know if any call
hits the fallback.**

**iter-5 lever candidates**:
- **A. EP scatter fusion** (1,685 ms fwd headroom + bwd savings) — replace
  EP `psum_scatter` with fused combine, or move scatter into gmm_v2
  kernel. Architectural; needs Pallas correctness + AOT compile gate.
- **B. moe_xlayer_prefetch**: blocked by iter-1's unresolved compile bug
  (don't propose).
- **C. Backward tile audit**: ❌ **iter-5 HALT — lever_blocked_at_library**
  (2026-05-07). Sizing showed 3,026 ms/step in tgmm (sized via
  `xla_shell list_fusions`, 6× tgmm.12-17 fusions ~8M us each over
  16-pass run). AOT compile-gate survey on tpu7x:2x2x1 confirmed
  **megablox.tgmm has hardcoded 32 MB scoped VMEM and no
  `vmem_limit_bytes` parameter** (unlike gmm_v2 which accepts 48 MB).
  All tile choices that improve iter-count over the generic
  `(2048, 1024, 1024)` exceed scoped-VMEM (40-48 MB) and FAIL AOT.
  **Generic `_tgmm_tiles` is provably-optimal within library cap**; no
  tile-tuning win possible. Future iters targeting tgmm need either
  upstream JAX `vmem_limit_bytes` exposure or a custom in-tree Pallas
  tgmm. Multi-iter follow-up; out of scope for single-iter Greedy.
- **D. Remat scope reduction**: switch from `remat=full` to `remat=attn_only`
  for MoE chunks. Saves ~5,400 ms/step in fwd recompute. **Requires HBM
  headroom analysis** — model already runs near runtime-compile limit.

### megablox.tgmm: production VMEM is 64 MB, not 32 MB (iter-5, 2026-05-08)

**INITIAL CLAIM RETRACTED**: my earlier note in this section claimed
"megablox.tgmm has hardcoded 32 MB scoped VMEM and no vmem_limit_bytes
parameter". The 32 MB figure was the AOT default; production runs with
64 MB via `LIBTPU_INIT_ARGS=--xla_tpu_scoped_vmem_limit_kib=65536` (set
in `manifests/jobset.yaml.j2:112`). Always check the LIBTPU_INIT_ARGS
in the production manifest before concluding a tile change is
library-blocked.

**Tgmm IS memory-bound at the v7x DSv3 production shapes**, confirmed
empirically by iter-5:
  - **iter-2b baseline** (tile `(2048, 1024, 1024)`): tgmm 48.4M us =
    3,026 ms/step, step time 34.65 s, TPS/chip 1882.
  - **iter-5 attempt** (tile `(4096, 1024, 1024)`, 2× iter reduction):
    tgmm 66.8M us = **4,175 ms/step (+38%)**, step time 35.30 s,
    TPS/chip 1856 (-1.4%).
  - The iter-count win (896 → 448 tile iters) became a wall-time loss
    because bigger tile_m grew the LHS input window per call without
    amortizing compute — added HBM bandwidth pressure.

**Operating rule for v7x DSv3 tgmm**: at the production shapes
`(K=7168, M=131072, N=2048)` and `(K=2048, M=131072, N=7168)`, the
generic `_tgmm_tiles = (2048, 1024, 1024)` is empirically optimal.
Don't tile-tune without microbench data — heuristic shots regress here.

**Future leverage** (if tgmm time becomes a target again):
1. Microbench tgmm via `bench_runner` (extend with `kind: "tgmm"`)
   to find tiles that actually improve wall time at these shapes.
2. Investigate if megablox.tgmm has compute-bound sweet spots at other
   shapes; the v7x DSv3 path may be a uniquely-bad case.

### v7x XLA flags worth knowing about

`LIBTPU_INIT_ARGS` in `manifests/jobset.yaml.j2:110-115`:
- `--xla_tpu_scoped_vmem_limit_kib=65536` — 64 MB scoped VMEM. Default
  is 32 MB; production needs 64 MB for `gmm_v2` and other Pallas
  kernels at production shapes. Documented use: `kernel_feedback_pallas_bwd.md`
  mentions 80 MB at one point.
- `--xla_tpu_bf16_emission_mode=NATIVE_EMISSION` — bf16 native emission.
- `--xla_tpu_num_sparse_cores_for_gather_offloading=2` — SC offload knob.
- `--xla_tpu_enable_sparse_core_collective_offload_all_gather=true`
- `--xla_tpu_enable_sparse_core_collective_offload_2d_all_gather=true`

**AOT compile gates should mirror these flags** to avoid false "lever
blocked" verdicts. Pattern:
```python
import os
os.environ["LIBTPU_INIT_ARGS"] = "--xla_tpu_scoped_vmem_limit_kib=65536"
# ... rest of AOT script
```
Without this override, AOT verdict diverges from production runtime.

**Cluster-discovery footnotes for next session**:
1. Bench JSON marker-delimited stdout is the durable data channel —
   `Dockerfile.tpu` lacks gsutil/gcloud, so the wrapper script's GCS
   upload is a no-op. Recovery via `kubectl logs` + manual `gcloud
   storage cp` from local works.
2. `tpu7x-inference-cluster` (`cloud-tpu-multipod-dev` project) has been
   broken for days due to stale reservation pinning on the MIG template
   (`cloudtpu-20251017124413-573252602` doesn't match current resource
   specs). Don't waste cycles there until owner fixes the node pool.
3. Halt declarations should re-poll JobSet state on session resume —
   Kueue admits asynchronously; iter-3's halt was premature, the bench
   actually completed in the queue while autoperf was off pivoting.

### Agent architecture: 1-agent + reviewer (2026-05-07)

Pivoted from 4-agent (autoperf + 3 maintainer fixers) to 1-agent + reviewer
on 2026-05-07. The autoperf agent now fixes inline across all 4 repos
(jax-gpt, perfsim, cde, xla-shell) on `autoperf-loop` branches in dedicated
worktrees under `~/autoperf/repos/<repo>/`. Maintainer agents move to a
review-only role: hourly pass over `autoperf-loop` PRs, comment, never merge.
Human gates daily merges.

Reasons for the pivot:
- Cross-repo handoffs (autoperf → maintainer agent → close issue) added
  multi-hour latency for fixes the autoperf agent could have made in 30
  seconds (e.g., the `from kernels.gmm_v2_train` relative-import bug on
  iter-2 attempt 1 — found, fixed, re-launched in one turn).
- Context fragmentation: the maintainer agents had to re-derive autoperf's
  diagnostic context from issue bodies; the autoperf agent had it in head.
- The 4-agent design's main benefit (second-pair-of-eyes review) is
  preserved by reviewer-agent role; we trade synchronous handoffs for
  async PR review.

Bootstrap: `autoperf/bootstrap.sh` creates the 3 worktrees on first run.
Invocations of perfsim/xla-shell scripts use
`PYTHONPATH=~/autoperf/repos/<repo>` to override the user's `pip install -e`
without disturbing the user's primary checkout. See AGENT.md §6 for the
worktree pattern.

### Aux loss
- `cfg.moe_aux_loss_coeff = 1e-4` by default
- Initial aux at random init: `coeff × E (=256) × sum(f·P) ≈ 7` per MoE layer
- Full config (58 MoE layers): initial aux ≈ 403, total initial loss ≈ 415
- Mini config (1 MoE layer): initial aux ≈ 7, total ≈ 19
- To disable: `--moe_aux_loss_coeff=0`. NOT `--aux_loss_weight=0` (that flag was
  dead and was removed in commit eb30b24)
- IMPORTANT: with coeff=0, aux=NaN at step 1+ is a SYMPTOM (`0 * NaN = NaN`),
  not a cause — the underlying NaN is in scores

### MaxText's aux-loss formula differs slightly
- Ours: `coeff × E × sum_e (avg_b f × avg_b P)` — average over batch first
- MaxText: `coeff × E² × mean_b mean_e (f_b × P_b)` — per-batch
- NOT bit-equivalent (Jensen's inequality). Forward magnitudes are similar
  (both ~7 per layer for mini, ~400 for full); bwd gradients through the
  routing softmax differ.

### MoE backend choices
- `gmm_ag` (production) — AllGather tokens on EP + AllGather F-dim weights on FSDP
- `jax` — does NOT work for DSv3 mini config (D=7168, D_moe=2048 contracting-dim mismatch)
- `fused_ep_moe_v4`, `fused_ep_moe_v4_jax_fwd`, `pfwd_jbwd` — Pallas v4 backends; test before relying

### `n_chunks` constraint
- Default `cfg.moe_n_chunks = 2` (per `model.py`)
- `n_chunks=4` triggers Bug 3 (bwd NaN). Don't propose without resolving the bug.
- `n_chunks=8` untested as of 2026-05-05.

### `prevent_cse=True` on jax.checkpoint produces NaN (autoperf iter8, 2026-05-08)

- Setting `prevent_cse=False → True` on the moe-scan `jax.checkpoint`
  call (model.py:3084 and :3101) produces **NaN at step 1** on the
  production baseline, with the existing `moe_layer_input`-only
  offload policy (no new offload markers added).
- jax-gpt#3 (filed) — different mechanism than jax-gpt#2 (attn_proj_out)
  but same NaN-at-step-1 symptom.
- **Working hypothesis (in jax-gpt#3 body)**: production is in a narrow
  stable groove where `prevent_cse=False` lets XLA silently CSE
  between fwd and recomputed-fwd, effectively bypassing the offload-
  restore path. With `prevent_cse=True`, the offload-restore path is
  genuinely exercised and exposes the same async-DMA race / layout
  drift issue that broke `attn_proj_out` in jax-gpt#2. If correct,
  fixing #2 likely fixes #3.
- **Don't propose `prevent_cse=True`** until the underlying offload-
  restore correctness issue is fixed. iter-2b state requires
  `prevent_cse=False` AND only `moe_layer_input` in offload — both
  load-bearing for stability.

### `attn_proj_out` offload broken at production scale (autoperf iter7, 2026-05-08)

- Adding `"attn_proj_out"` to `_ckpt_policy.names_which_can_be_offloaded`
  at `model.py:3054` produces **NaN at step 1** on the production
  baseline (`full + ga=1 + n_chunks=2 + gbs=4096` on `fsdp=128 ep=4
  tp=1` v7x_4x8x8). Reverted on iter-7 (commit `8245f5d` → revert
  `bbfe974`). dsv3train-i7 step-4 reported TPS/chip 1920 (+2% vs
  iter-2b's 1882) but **all steps had loss=nan; the speedup is illusory
  (NaN-bit propagation is fast, no real computation)**.
- The marker `out = ck(out, "attn_proj_out")` at `model.py:560` (CP path)
  and `:636` (non-CP path) is in the code but the v315 author's policy
  at `model.py:3052-3057` only includes `moe_layer_input`. The
  comment at line 3047-3051 reads "Only large activations
  (moe_layer_input, attn_proj_out) have favorable DUS:save ratio" —
  this is now confirmed to be a **hint not a verified lever**. The
  author likely encountered this NaN behavior too and the comment
  documents the candidate without endorsing the wired-in version.
- AOT compile gate + jaxpr verification (`dot_general` count drops
  10→9 with the change; `attn_proj_out` appears as `from_residual` in
  bwd jaxpr) PASSED. The NaN is runtime-only. Hypotheses:
  1. Async `pinned_host` offload race (bwd reads stale buffer).
  2. bf16↔fp32 conversion drift on offload roundtrip combining with
     attention's softmax to produce NaN.
  3. Sharded-layout mismatch when offloaded activation is restored.
- **Don't propose `attn_proj_out` offload until the underlying NaN
  bug is fixed.** It's the iter-6-bisection-identified lever for
  4,216 ms/step of attention-recompute headroom; until the offload
  correctness is fixed, that headroom is unaddressable from the
  autoperf side. Same class of "marker-exists-policy-doesn't" might
  also apply to `q_a` (`model.py:568`), `kv_a`, `shared_hidden`
  (`model.py:2676`); they were tested + rejected by v315 for small
  DUS:save ratio, but the NaN failure mode might also apply.
- Filed as autoperf iter-7 result in `autoperf/iter_log.md`.
  Pre-existing `moe_layer_input` offload remains the only verified
  positive-ROI offload at iter-2b state.

### `--moe_xlayer_prefetch` broken at production scale (autoperf iter1, 2026-05-06)
- `cfg.moe_xlayer_prefetch=True` (cross-layer FSDP weight AG prefetch via
  3-tuple scan carry, `model.py:3061-3094`) fails to compile at
  `full + ga=1 + n_chunks=2 + gbs=4096` on `fsdp=128 ep=4 tp=1` v7x_4x8x8.
- Error: `ValueError: all_gather_reduced only accepts inputs that are
  varying. Got bf16[64,16,7168]` during bwd-transpose abstract eval of
  `_ag_one_moe_layer`. Stack: `train_step → value_and_grad → _vjp →
  linearize → _all_gather_is_async → all_gather_reduced → abstract_eval`.
- Cause: the `pcast/reduced` AG primitive's bwd transpose requires the
  input to be sharded ("varying") along the gather axis. The prefetch
  carry threads a gathered weight tile `[E_local=64, F_local=16, D=7168]`
  into a position where its bwd cotangent appears replicated, not
  sharded. JAX/jaxlib 0.10.0 (current pin) rejects this; whether older
  versions accepted it (silently) is unknown.
- **Don't propose `moe_xlayer_prefetch=True` until the underlying
  jax-gpt bug is fixed.** It's the heuristic-table lever for `FSDP_AG`
  top-headroom; until the fix lands, iter-N's `FSDP_AG` headroom is
  unaddressable from the autoperf side.
- Filed as autoperf iter-1 result in `autoperf/iter_log.md`. Reverted on
  branch `autoperf/dsv3_train_full`. Default (`moe_xlayer_prefetch=False`)
  compiles and trains cleanly — that's the v304-postrefactor baseline.

---

## 6. cde / kubectl operational knowledge

### Image tags drift
- `cde build` computes the image tag from a hash of the build context
- Untracked files (PDFs, PNGs) in working tree can poison the hash → cde tag drift
- Fixed via `.dockerignore` + `.gitignore` patterns for `*.pdf *.png *.pptx`
  (commit 7dbd659)

### `cde profile` paths
- `cde profile path <run_id>` returns `gs://...` URI of pulled profile
- Localize: `gsutil -m cp -r <uri>/* <local-dir>/`
- xplane.pb dirs typically nested under `plugins/profile/`

### `cde history --status running` for parallel-job count
- Use this to enforce max-parallel-jobs constraint
- Example: `[ $(cde history --status running 2>/dev/null | wc -l) -ge 2 ]`

### Eviction vs failure (cde-side signals)
- `Finished=True (Succeeded)` → ok
- `Finished=True (Failed)` + `Requeued=True | NodeFailures` → eviction (retry)
- `Finished=True (Failed)` without those → real failure (read logs, halt)

### Image pull races
- Pod pull failures often racing with cde tag updates
- Workaround: rebuild explicitly (`cde build`), then resubmit with new tag

### Cluster-availability check protocol (added 2026-05-07 from iter-3)

**Node-existence ≠ node-availability.** Always pair these two checks BEFORE
submitting a job to a non-cde-managed cluster (`kubectl apply -f` direct, e.g.
calibration jobs that run `bench_runner`):

```bash
# (1) which TPU nodes exist?
kubectl --context <ctx> get nodes \
    -L cloud.google.com/gke-tpu-accelerator,cloud.google.com/gke-tpu-topology \
    --no-headers | grep tpu7x | awk '{print $(NF-1), $NF}' | sort | uniq -c

# (2) which of those nodes are already occupied?
kubectl --context <ctx> get pods -A -o json | \
    jq -r '.items[] | select(.spec.nodeSelector and
        (.spec.nodeSelector."cloud.google.com/gke-tpu-accelerator"=="tpu7x")) |
        "\(.spec.nodeName // "PENDING")\t\(.metadata.namespace)/\(.metadata.name)\t\(.spec.priorityClassName // "none")\t\(.status.phase)"' \
    | sort | head -30
```

Iter-3 hit this trap on `bodaborg-tpu7x-inference`: 8× tpu7x-standard-1t nodes
visible, all 8 held by long-running medium-priority `ubench` workloads (ages
21h–6d). Three other medium pods already pending 25-81 min ahead. Cluster-wide
TPU quota exceeded for autoscale, so submitting a new medium-priority pod just
adds to the queue. Bumping to `high` would preempt another user's running
workload — that's the user's call, not autoperf's; halt instead.

**Calibration-friendly clusters and their typical state**:

| Cluster | Useful pools | Pattern |
|---|---|---|
| `bodaborg-tpu7x-inference` (project `tpu-prod-env-automated`) | 8× 1t (1×1×1), 6× 4t (2×2×1) | First choice for single-host calibration. Runs on `medium` priority during low-usage windows; full during business-hours. |
| `bodaborg-super-rbq` (project `cloud-tpu-multipod-dev`) | 16× 4×4×4, 1088× 4×8×8 | Production training cluster. **No 1t/4t nodes**. A 1-chip calibration on the 4×4×4 pool burns 64 chip-min for a 5-15 min run — wasteful but always available. |
| `ninja-v7x-64-spot` | 8× 2×4×4 | Spot/preemptible. Skip for calibration (unreliable). |

**Image cross-project pulls**: images at `gcr.io/tpu-vm-gke-testing/...` pull
fine across both `cloud-tpu-multipod-dev` and `tpu-prod-env-automated`
projects (verified iter-3, qwen3coder-calibration earlier). No project-local
mirroring needed.

---

## 7. Filing tool issues (vs working around)

When a tool failure feels like a missing feature or a bug rather than a config
mistake, file an issue against the upstream repo. Repos and labels:

| tool | repo | issue label |
|---|---|---|
| perfsim | `ultrons/perfsim` | `autoperf-blocking` |
| cde | `ultrons/cde` | `autoperf-blocking` |
| xla-shell | `ultrons/xla-shell` | `autoperf-blocking` |

Always `gh issue list --repo <r> --search "<keyword>"` first to dedupe. Always
include a copy-pasteable repro and "definition of done" criterion.

(See AGENT.md §5 for the exact issue body template.)

---

## 8. How to use this file (the meta-rule)

- **Read at session start** — every autoperf session, first thing.
- **Append when you learn something** — a new pitfall, a fix that landed, a
  perfsim preset that's now reliable. Add a row or section.
- **Don't delete entries** — mark them stale with a date if superseded.
- **Cross-link, don't duplicate** — for things in `~/.claude/CLAUDE.md` global
  (Mosaic constraints, vLLM weight-loading via runai-streamer), reference rather
  than copy. This file is for *operational knowledge an autoperf agent needs to
  not repeat known-bad experiments*.

When a tool agent (cde/perfsim/xla-shell) closes one of your blocking issues,
update the corresponding Bug row above (status `active` → `fixed in <repo>#<pr>`)
and the stack-pin section if the fix changes a version requirement.

---

## 9. Architectural future: A2A (Agent2Agent) migration considerations

The current 4-agent system uses **GitHub issues as the message bus**
(autoperf files; cde/perfsim/xla-shell maintainer agents fix + close).
This is intentional — GH issues give us free durable audit trail, async
delivery (agents don't need to be live simultaneously), and LLM-agnostic
operation (any model with `gh` CLI works).

**Google's A2A protocol** (announced 2025) is the structured alternative:
HTTP + JSON-RPC 2.0 + SSE, Agent Cards at `/.well-known/agent.json`,
Task lifecycle states, multi-turn negotiation. SDKs in Python/JS/Java/Go.

**When to consider migrating** (not now; future-self read):
1. **High-frequency interactive debugging** — if autoperf and a tool
   agent need to bounce hypotheses in seconds rather than at iteration
   boundaries (today: minutes/hours).
2. **Auto-delegation chains** — perfsim agent's "tax mis-fit → file at
   xla-shell" pattern is currently manual cross-ref + close-as-duplicate.
   A2A's `forwardTo` would automate this and wire the response back to
   the original issue's filer (autoperf).
3. **Third-party agent integration** — if a vLLM-side or training-stability
   monitor wants to consume autoperf's "is this leaf a known top-leaf"
   query, A2A is the clean interface.

**What would need to change for migration** (rough scope):
- Each maintainer agent runs as a long-lived A2A server, not a one-shot
  Claude Code session. (Hosting concern: 4× always-on LLM cost.)
- Agent Cards published per repo describing skills (e.g., perfsim's
  `diagnose_mispredict`, `extend_microbench_grid`, `add_preset`).
- autoperf becomes an A2A client that routes Tasks to the right agent
  instead of running `gh issue create`.
- We lose GH-issue audit trail unless we double-write (issues + A2A
  Tasks). A2A doesn't have native persistence.

**Recommendation**: stay on GH issues until specific friction surfaces.
The protocol overhead, hosting cost, and lost durability aren't worth it
for an intermittent perf-optimization loop. Revisit if any of (1)/(2)/(3)
above become a recurring need. If revisiting, the GH-issue protocol we
have now translates 1:1 to A2A Task semantics — the migration is pattern-
preserving, just more infrastructure.
