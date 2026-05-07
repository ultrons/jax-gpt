# BLOCKED.md — open tool-bug issues blocking autoperf iterations

The autoperf agent maintains this table. One row per filed issue. On each
iteration's step-1, the agent re-checks `open` rows; closed ones get marked
`resolved`, the relevant tool repo gets `git pull`ed, and the originally
blocked iteration's change is retried.

| iter | workload | repo#issue | filed | status |
|---|---|---|---|---|
| 1 | dsv3_train_full | ultrons/perfsim#1 | 2026-05-06 | resolved 2026-05-06 |
| 2 | dsv3_train_full | ultrons/perfsim#3 | 2026-05-06 | resolved 2026-05-07 (gemm_eff calibration; partial fix landed alongside #4) |
| 3 | dsv3_train_full | ultrons/perfsim#4 | 2026-05-07 | resolved 2026-05-07 (gmm_ag batch_sharded_by_ep wiring; user-filed) |
| 4 | dsv3_train_full | ultrons/perfsim#5 | 2026-05-07 | resolved 2026-05-07 (xplane HLO dim validation; user-filed) |

## perfsim deep-review issues (filed 2026-05-07; nearly all resolved by 06:02 UTC)

The deep review of perfsim (autoperf session 2026-05-07) surfaced 12 gaps; sibling #19 added per maintainer's analysis on #7. **The 12 issues went from filed → 12-of-13 resolved in ~3 hours via two PRs (perfsim PR#20, PR#22).** Iter-3 prerequisite cluster (#7 + #19) cleared.

| iter | workload | repo#issue | filed | status |
|---|---|---|---|---|
| 3-prerequisite | dsv3_train_full | ultrons/perfsim#7 | 2026-05-07 | resolved 2026-05-07 06:02 (PR#22 — `bad2c271`; thin swap headroom_report→model_builder) |
| 3-prerequisite | dsv3_train_full | ultrons/perfsim#19 | 2026-05-07 | resolved 2026-05-07 06:02 (PR#22 — `689a01bd`; per-op port into model_builder layers) |
| 3-secondary | dsv3_train_full | ultrons/perfsim#8 | 2026-05-07 | resolved 2026-05-07 05:26 (PR#20 — `a1344a45`; per-leaf confidence) |
| 3-secondary | dsv3_train_full | ultrons/perfsim#9 | 2026-05-07 | resolved 2026-05-07 05:26 (PR#20 — `a1344a45`; bundled with #8) |
| 3-secondary | dsv3_train_full | ultrons/perfsim#10 | 2026-05-07 | resolved 2026-05-07 (PR#24 — `eea991f`; bucketed `gemm_eff_curve_bf16` for v7x; pinned to microbench from PR#23) |
| 3-secondary | dsv3_train_full | ultrons/perfsim#11 | 2026-05-07 | resolved 2026-05-07 05:26 (PR#20 — `a1344a45`; bundled with #8 + #9; extrapolation flag) |
| (search) | dsv3_train_full | ultrons/perfsim#12 | 2026-05-07 | resolved 2026-05-07 05:26 (PR#20 — `8a24f380`; cmd_search wired) |
| (search) | dsv3_train_full | ultrons/perfsim#13 | 2026-05-07 | resolved 2026-05-07 05:26 (PR#20 — `78f7fdc6`; batch_sharded_by_ep in search) |
| (search) | dsv3_train_full | ultrons/perfsim#14 | 2026-05-07 | resolved 2026-05-07 05:26 (PR#20 — `b19bb7af`; HBM ceiling 0.90→0.85) |
| (search) | dsv3_train_full | ultrons/perfsim#15 | 2026-05-07 | resolved 2026-05-07 05:26 (PR#20 — `98650f95`; profile_corpus.json index + CLI) |
| (long) | dsv3_train_full | ultrons/perfsim#16 | 2026-05-07 | resolved 2026-05-07 05:26 (PR#20 — `962a42a0`; ADR-002 design pass; impl deferred) |
| (long) | dsv3_train_full | ultrons/perfsim#17 | 2026-05-07 | resolved 2026-05-07 05:59 (closed as duplicate; subsumed by #18 ADR-001 inference scope) |
| (long) | dsv3_train_full | ultrons/perfsim#18 | 2026-05-07 | resolved 2026-05-07 05:59 (closed as partially-completed; remaining inference migration deferred) |

**Empirical impact post-PR#22 on v304 headroom (the trust restoration):**

| leaf | pre-PR#22 ratio | post-PR#22 ratio |
|---|---|---|
| Expert_gmm | 1.12 | **1.18** ← new top-headroom |
| Attn_scores | 0.69 | **0.82** |
| O_proj | 0.75 | **0.92** |
| QKV_proj | 0.40 | **0.48** |
| EP_AG_dispatch | 0.94 | **0.99** |

Top-3 shifted: `[FSDP_AG, Router, Norms]` → **`[Expert_gmm, Norms, FSDP_AG]`**. See v7x_KNOWLEDGE.md §5 for the full updated trust table.

**Iter-3 readiness:** prerequisite cluster cleared. Only blocker is perfsim#10 (BF16 curve) — gated on the autoperf-side microbench grid (below). That's a Tooling-class iteration; can run anytime.

**Iter-4 readiness (2026-05-07 evening):** All deep-review issues closed. perfsim#10 (BF16 curve) resolved via PR#24. NO open BLOCKED rows. iter-4 can run as Greedy class on Expert_gmm (top-leaf #1 across iter-2 and post-PR#24).

**Iter-4 step-1.5 findings (2026-05-07 evening, post-PR#24 trust validation):**

Two new perfsim issues filed. Both autoperf-blocking class for trust-table coherence; iter-4 proceeds as Tooling-class anyway, so they're parallel housekeeping:

| iter | workload | repo#issue | filed | status |
|---|---|---|---|---|
| 4-housekeeping | dsv3_train_full | ultrons/perfsim#25 | 2026-05-07 | **OPEN** — P1: Norms predicted collapsed to ~0.1ms post-PR#22 (was 207ms iter-2). Top-3 inclusion of Norms is artifact, not real signal. Until fixed, Norms NOT-trusted. |
| 4-housekeeping | dsv3_train_full | ultrons/perfsim#26 | 2026-05-07 | **OPEN** — P2: LEAF_PATTERNS_TRAINING Expert_gmm rule needs gmm_v2/tpu_custom_call fusion-name match. Post-iter-2b xplanes under-count Expert_gmm measured. Workaround: compare against v304 (pre-gmm_v2) xplane for clean Expert_gmm bucketing. |

## Autoperf-side tasks blocking perfsim issues

| iter | autoperf task | unblocks perfsim# | added | status |
|---|---|---|---|---|
| 3-tooling | Launch `bench_runner` on v7x_4x8x8 with the (M, K, N) grid agreed on perfsim#10's needs-info | #10 (BF16 GEMM curve) | 2026-05-07 | **resolved 2026-05-07 evening** — original JobSet on `bodaborg-tpu7x-inference` actually completed at 07:56Z (Kueue admitted it ~50 min after submit when a medium pod released). 35 workloads measured, cv<1%; results at `gs://max-experiments/autoperf/microbench/v7x_4x8x8_bf16_2026-05-07/`. Commented on perfsim#10 (status: needs-info → ready for curve fit) and on PR#23 with the headlines. v304 anchor measured at 0.6226 vs spec 0.244 (1.27× after per-core peak renorm); ~30% remaining gap = in-training overhead the standalone microbench doesn't capture. |

Microbench grid spec (from perfsim#10 maintainer comment):
- M ∈ {1024, 4096, 16384, 65536, 131072}
- Dense and grouped-MM (`n_groups`)
- (K, N) coverage: LMHead (7168, 129280), Router (7168, 64), attention QKV (7168, 7168)
- Output: `gs://...autoperf/microbench/v7x_4x8x8_bf16_<date>/`
- Validate by re-running iter-2 headroom and confirming LMHead/Router ratios normalize

**Step-1 ritual update**: when an issue here transitions from `open` to closed,
the agent that picks up the next session should:
1. `git -C ~/perfsim pull` (worktree at `~/autoperf/repos/perfsim` likewise)
2. Re-run iter-2's `headroom_report` against the same xplane to see the new
   ratios (especially for QKV/O/Attn/LMHead — these should normalize once #7+#19 land)
3. Update the trust table in `v7x_KNOWLEDGE.md` §5
4. Mark the corresponding row above as `resolved <date>` with one-line summary

(Add rows below this header. Don't delete `resolved` rows — they're
audit history.)
