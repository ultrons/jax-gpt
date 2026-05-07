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
| 3-secondary | dsv3_train_full | ultrons/perfsim#10 | 2026-05-07 | **OPEN** — needs-info; awaits microbench grid (autoperf-side task below) |
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

## Autoperf-side tasks blocking perfsim issues

| iter | autoperf task | unblocks perfsim# | added | status |
|---|---|---|---|---|
| 3-tooling | Launch `bench_runner` on v7x_4x8x8 with the (M, K, N) grid agreed on perfsim#10's needs-info | #10 (BF16 GEMM curve) | 2026-05-07 | **partial 2026-05-07** — spec + bench_runner grouped-MM extension landed in ultrons/perfsim PR#23 (commit `cb67ec0` on autoperf-loop). Image `perfsim-bench:v25-bf16-microbench` built/pushed. JobSet applied to `bodaborg-tpu7x-inference` 1×1×1 stayed Pending (cluster full at medium priority, refused to preempt). HALT.md filed. Next session: resubmit when 1t nodes free up OR authorize priority bump; on success, comment on perfsim#10 with the GCS results path. |

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
