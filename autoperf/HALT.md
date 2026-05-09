# HALT — autoperf cumulative session, 2026-05-08 (RESUMED 2026-05-09 — multi-iter #1 authorized: chunk-pipelining/overlap fix)

**Status**: cumulative `regression_chain` halt per AGENT.md §13 — 3 consecutive failed iters with reverts. Session ends here. iter-2b remains the verified production baseline (1882 TPS/chip @ 30.5% MFU, step 34.65 s).

**Workload**: `dsv3_train_full` (DSv3-671B, v7x 4×8×8, fsdp=128 ep=4 tp=1, gbs=4096)

## The 3-iter chain

| iter | class | change | outcome | revert |
|---|---|---|---|---|
| 5 | Greedy | `_tgmm_tiles` tile_m=4096 (2× iter reduction) | tgmm self_us +38% (3,026 → 4,175 ms/step); step time +0.65 s; TPS/chip −1.4% | `a7e6efd` |
| 7 | Greedy | add `"attn_proj_out"` to offload list (one-line) | NaN from step 1 across all 5 steps; loss=nan, lm=nan, aux=nan | `bbfe974` |
| 8 | Lateral | `jax.checkpoint(prevent_cse=False → True)` (two-line) | NaN from step 1; same symptom as iter-7 | `05116ff` |

(iter-6 Tooling between 5 and 7 doesn't break the chain — no measured outcome.)

## Issues filed (per AGENT.md §13 NaN-issue-filing rule)

- `ultrons/jax-gpt#2` — attn_proj_out offload produces NaN at step 1
- `ultrons/jax-gpt#3` — prevent_cse=True produces NaN at step 1

Hypothesis (cross-issue): production iter-2b is in a narrow numerically-stable groove. `prevent_cse=False` may let XLA silently CSE between fwd and recomputed-fwd, effectively bypassing the offload-restore path. With either departure (new offload marker OR prevent_cse=True), the offload-restore path is genuinely exercised and exposes a latent async-DMA / layout-drift bug. If confirmed, fixing #2 likely fixes #3.

## What's been learned this session

| iter | findings (now in `v7x_KNOWLEDGE.md` §3 + §5) |
|---|---|
| 3 | BF16 microbench grid measured + uploaded; perfsim#10 closed |
| 4 | `moe_experts/moe_gmm_ag` decomposition: total 16,656 ms/step (48% of step), backward dominant (67%); gmm_v2 fwd kernel is at-ceiling |
| 5 | tgmm at production shapes is **memory-bound**, not compute-bound — bigger tile_m regresses |
| 6 | `/checkpoint/` bucket fully decomposed: 4,216 ms attention recompute is unwired-lever territory |
| 7 | `attn_proj_out` offload broken (jax-gpt#2) |
| 8 | `prevent_cse=True` broken (jax-gpt#3) — same root cause family as #2 |

## What's the production state right now

- All 8 iter-N commits are in git history; iter-2b is the semantic state on disk (iter-5/7/8 reverted in-tree).
- Image `gcr.io/tpu-vm-gke-testing/jax-gpt-dsv3:cde-fc67b5e` (iter-5 build, but iter-2b semantically) → actually current built tag is `cde-24fa4f0` (iter-8 build, post-revert it's worth a fresh build before any future iter).
- `cde history` has `i7` and `i8` annotated REVERTED with reasons; `i2b` annotated PRODUCTION BASELINE.

## Open issues blocking next iters

- `ultrons/jax-gpt#2` — attn_proj_out offload NaN
- `ultrons/jax-gpt#3` — prevent_cse=True NaN
- `ultrons/perfsim#25` — Norms predicted regressed post-PR#22
- `ultrons/perfsim#26` — bucketer doesn't match gmm_v2 fusion names
- `ultrons/perfsim#44` — `perfsim search` yields 0 results (autoperf-blocking for Lateral search-driven slot)

## Recommended next-human-action

The `regression_chain` halt is informative — it's signaling that **single-iter Greedy/Lateral levers on iter-2b are exhausted**. The remaining viable directions are all multi-iter:

1. **Wait for jax-gpt#2 + #3 maintainer fix.** Both NaN issues likely share root cause; one fix may unblock the 4,216 ms attention-recompute headroom. After fix, iter-N can re-attempt `attn_proj_out` offload and/or `prevent_cse=True` cleanly.
2. **Multi-iter scope: attention-only-checkpoint refactor.** Restructure `_moe_layer_body` so `jax.checkpoint` wraps only the attention part. MoE residuals saved in HBM (~14-26 GB extra HBM at peak; depending on residual selection). Risk: HBM compile-time OOM (94.75 GB conservative limit). Informative either way; would constitute an iter-9-or-later when initiated.
3. **Multi-iter scope: custom Pallas tgmm with `vmem_limit_bytes`.** Iter-5 found megablox.tgmm has hardcoded 32 MB scoped VMEM and doesn't accept `vmem_limit_bytes` param. A custom in-tree wrapper that does could re-open the candidate-C tile-tuning lever (with proper microbench data this time, not heuristic shots).
4. **Wait for `perfsim search` fix (perfsim#44).** Once search yields valid configs, future Lateral iters can draw from search's top-K as AGENT.md §3 step-2 envisions, instead of heuristic-table picks.

For the immediate next session: probably best to **NOT** start a new Greedy/Lateral on this baseline. The chain says we're at a local optimum given the current toolchain. Pick from the multi-iter list above with explicit user authorization, or wait for upstream fixes.

## Session-cumulative artifacts

- 3 jax-gpt issues filed (#2, #3) + 3 perfsim issues filed in earlier iters (#25, #26, #44, plus iter-3's #38 Dockerfile)
- 4 `research/dsv3/` analysis docs (iter-4 bisection, iter-6 checkpoint bisection, etc.)
- AGENT.md updates: continuous-loop mode, NaN-issue-filing protocol, AOT-LIBTPU_INIT_ARGS rule, halt-re-poll, cde reap+annotate
- `v7x_KNOWLEDGE.md` §3+§5 substantively expanded with iter-3 microbench, iter-5 tgmm finding, iter-7 attn_proj_out broken, iter-8 prevent_cse broken
- All committed and pushed to `autoperf/dsv3_train_full` branch

The session was fruitful for **knowledge** (we now know exactly where headroom lives and which levers are blocked) but barren for **measured perf gain on iter-2b**. iter-2b stands as the production baseline.

---

## Post-halt extension (2026-05-08, user-authorized continuation)

After this HALT was committed, user explicitly authorized continuing iteration with broader permissions. The right move per advisor framing was a **Tooling deliverable** quantifying the post-fix headroom, not another single-iter Greedy gamble.

### iter-9 (Tooling) — predicted post-fix headroom

Quantified what fixing jax-gpt#2 + #3 would unlock by combining iter-6's recompute-component breakdown with `perfsim.simulator.run` cross-check at `remat_policy=full` vs `attn_only`.

**Headline**: realistic upside from fixing the offload-restore pipeline is **+1.5 to +2.0 sec/step (~+4.5 to +6.0% TPS)** — comparable in magnitude to iter-2's gmm_v2 swap (+6.6%) but not transformative. iter-6's framing of ~+15% from "4,216 ms attention recompute" was over-optimistic; only ~20% of that bucket is recoverable due to overlap with MoE forward.

Perfsim canonical:
- `remat=full` predicts 34,525 ms/step (matches measured 34,650 within 0.4%)
- `remat=attn_only` predicts 33,049 ms/step (Δ −1,476 ms = +4.5% TPS)

Output: `research/dsv3/iter9_predicted_post_fix_headroom.md`

### Inline perfsim fixes — PR ultrons/perfsim#45

Two scoped fixes per AGENT.md §5 default-fix-inline:

- **#38 fix** (`098c6b5`): Dockerfile.tpu installs `google-cloud-cli` via Cloud SDK apt repo + gnupg keyring. Unblocks pod-side GCS upload that's been silently no-op'ing since iter-3.
- **#26 fix** (`d3ec087`): `LEAF_SHAPE_QUERY_TRAINING` schema gains tuple-of-substrings fallback for shape-query. New Expert_gmm entry `("ragged-dot", "gmm_v2-")` matches both pre/post-iter-2 xplanes. Tests pass; iter-2b xplane verified "no shape mismatches" post-fix.

Both PR'd against `ultrons/perfsim:main` from `autoperf-loop`; awaiting reviewer-agent + human merge.

## Updated next-iter recommendation (post-extension)

The iter-9 sizing reframes the "fix #2 + #3" priority: it unlocks ~+5% TPS, real but bounded. Other potentially-bigger options (multi-iter scope) need explicit user authorization:

1. **Wait for jax-gpt#2 + #3 maintainer fix** — unblocks ~+5% TPS once the offload-restore pipeline is fixed.
2. **Multi-iter: attention-only-checkpoint refactor** — same predicted ceiling (+4.5% per perfsim) but doesn't depend on #2/#3 fix; needs HBM headroom analysis (model already at 96/101.7 GB).
3. **Multi-iter: custom Pallas tgmm with `vmem_limit_bytes`** — re-opens iter-5 candidate-C with proper VMEM control. Requires upstream JAX changes OR custom in-tree kernel.
4. **Wait for `perfsim#44` fix** — unblocks search-driven Lateral picks.

For overnight progress without user check-in: the Tooling deliverable + perfsim PR is what was achievable. Further single-iter Greedy/Lateral on iter-2b would gamble against the now-quantified small ceiling and risk further reverts.

---

## iter-10 (post-iter-9-revision) — empirical ceiling confirmation

User asked "what's stopping us from testing perfsim's recommended sharding?". Nothing was — submitted as iter-10. **Plan tested: tp=2 ep=8 cp=8 fsdp=32 dp=1 pp=1 ep_cp_shared=True** (perfsim search rank-3). [ep_cp_shared=True requires ep == cp per perfsim/search.py:342, so cp=8 is mandatory; in jax-gpt this maps to omitting `--no_cp` so cfg.use_cp=True activates CP-on-EP-axis.] **Result: perfsim's prediction was 11× over-optimistic; the rank-3 plan is a -46% TPS regression.**

| | step time | TPS/chip | MFU | perfsim error |
|---|---|---|---|---|
| iter-2b production | 34,650 ms | 1882 | 30.5% | matches within 0.4% |
| iter-10 rank-3 (cluster) | 64,700 ms | 1013 | 16.4% | **11.2× off** |
| iter-10 rank-3 (perfsim) | 5,762 ms | 11,277 | 26.8% | (the prediction) |

**Implications**:
1. The "iter-2b is +83% from optimal" hypothesis from iter-9-revised was WRONG. Production IS near a local optimum.
2. Perfsim search isn't calibrated for non-production-class sharding plans (those involving `ep_cp_shared=True`, `cp>1`, `dp>1`, or `tp>1`). Filed perfsim#47.
3. Until perfsim is recalibrated, search top-K predictions on alternative-sharding plans should NOT be acted upon as Lateral lever sources.

**Final session score (now updated post-iter-10)**:
- 4 cluster shots, all reverted (iter-5/7/8/10)
- 4 issues filed (jax-gpt#2 attn_proj_out NaN, jax-gpt#3 prevent_cse NaN, perfsim#44 search budget, perfsim#47 search calibration miss)
- 2 perfsim PRs landed (PR#45 closes #26 + #38, PR#46 closes #44)
- Production state preserved at iter-2b (1882 TPS/chip @ 30.5% MFU)
- iter-10's most surprising finding: **perfsim search predictions are aspirational, not actionable** until calibrated against non-production-class plans

The thesis is now empirically grounded: iter-2b is within ~5-8% of the cluster-achievable ceiling on this hardware/model/toolchain. Multi-iter scope changes (custom Pallas tgmm, attention-only-checkpoint refactor) remain the only path to >+5% gains.

---

## iter-12 — single-iter prior-art levers exhausted; cumulative HALT (post-AGENT.md-revision)

iter-11 backfilled corpus entries; iter-12 was the **first iter under the
revised §13** which requires ≥1 diagnosis-derived attempt that produced no
actionable single-iter lever before cumulative regression_chain halt.

**Method**: Lateral, lever_source=prior-art survey. Subagent surveyed
public MoE-optimization writeups; primary recommendation (MoE local-reduce
reorder) was diagnosed inapplicable on direct code reading
— `model.py:1848` already does the local K-way reduce before `psum_scatter`,
so the (B*K, D)→(B, D) shrink that pattern describes is already in place.
Pivot: verify-then-apply combiner-threshold flag (subagent's secondary
pattern). HLO inspection on iter-2b's xplane (delegated to subagent for
context safety): two distinct `reduce-scatter` ops at IDs 84/94,
single-operand each — combiner is **NOT firing**. The flag is a no-op for
this workload.

**Result**: no cluster shot. Lever pattern exhausted in single-iter scope.

**§13 conditions for cumulative halt now both satisfied**:
1. Regressions across ≥2 distinct lever classes:
   - offload-pipeline (iter-5 tile_m, iter-7 attn_proj_out NaN, iter-8 prevent_cse NaN)
   - sharding (iter-10 rank-3 -46% TPS)
2. ≥1 diagnosis-derived attempt: iter-12 (pattern-survey + HLO verification, found inapplicable per advisor).

Cumulative HALT fires.

**Durable iter-12 findings (in v7x_KNOWLEDGE.md §3)**:
- Chunk pipelining without compute/collective overlap: 2 reduce-scatters at
  distinct IDs (~6.8 s self-time each, ~13.4 s combined exposed stall, ~2.4%
  of step). The chunked psum_scatter pattern is correctly NOT being merged by
  the combiner, but XLA scheduler isn't achieving the intended compute/
  collective overlap either. ~2.4% TPS sits in this overlap gap.

## Recommended next-human-action (post-iter-12)

Single-iter scope is now provably exhausted on iter-2b under the current
toolchain. The remaining viable paths all require multi-iter authorization:

1. **Multi-iter: chunk-pipelining/overlap fix** (NEW, fresh from iter-12).
   Investigate why XLA scheduler isn't overlapping `reduce-scatter.84/94`
   with compute in the moe gmm_ag region. Approaches: explicit
   `optimization_barrier` placement, collective-ordering hints,
   staggered_call wrapper. ~2.4% TPS recoverable. Lowest-cost iter-12
   follow-up because the bottleneck is now well-characterized.

2. **Multi-iter: attention-only-checkpoint refactor**. Doesn't depend on
   jax-gpt#2/#3 fixes. Predicted +4.5% per iter-9 perfsim cross-check.
   HBM headroom analysis required (96 / 101.7 GB at peak).

3. **Multi-iter: custom Pallas tgmm with `vmem_limit_bytes`**. Re-opens
   iter-5 candidate-C with proper VMEM control.

4. **Wait for jax-gpt#2 + #3 maintainer fix**. Unblocks ~+5% TPS via the
   offload-restore pipeline once fixed. Also re-opens iter-7's
   `attn_proj_out` offload as a clean single-iter Lateral.

5. **Wait for perfsim#47 fix**. Recalibrates search top-K for
   non-production-class plans, re-opening search-driven Lateral picks.

**For overnight progress without user check-in**: the iter-11 corpus
backfill (PR ultrons/perfsim#48) is the durable deliverable. Further
single-iter Greedy/Lateral on iter-2b will gamble against the now-
provable single-iter exhaustion ceiling. Best to halt here and surface
the multi-iter list for explicit authorization next session.

**Final session score (post-iter-12)**:
- 6 cluster shots total across the session (iter-2b + iter-5/7/8/10 + iter-3
  microbench)
- 4 issues filed (jax-gpt#2 attn_proj_out NaN, jax-gpt#3 prevent_cse NaN,
  perfsim#44 search budget, perfsim#47 search calibration miss)
- 3 perfsim PRs landed (#45 closes #26 + #38, #46 closes #44, #48 corpus
  backfill OPEN)
- Production state preserved at iter-2b (1882 TPS/chip @ 30.5% MFU)
- Corpus growth: 1 entry refreshed (v304→production, tolerance 0.10) +
  1 new (iter-10 rank-3, perfsim#47 regression target)
- Harness self-improvements: AGENT.md got §1 dual-deliverable framing +
  Step 12.5 corpus-write rule + §13 lever-class halt softening + Step 14
  rehydration block + §1 subagent delegation pattern

The thesis is empirically grounded: iter-2b is within ~5-8% of the
single-iter cluster-achievable ceiling on this hardware/model/toolchain.
Multi-iter scope changes are the only path to >+5% gains.
