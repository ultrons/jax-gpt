# HALT — autoperf cumulative session, 2026-05-08

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
