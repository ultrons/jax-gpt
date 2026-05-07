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

## perfsim deep-review issues (filed 2026-05-07; gating iter-3 resumption)

The deep review of perfsim (autoperf session 2026-05-07) surfaced 12 gaps; sibling #19 added per maintainer's analysis on #7. Iter-3 is paused until at least the P0 cluster (#7 + #19) lands. P1+ can land in parallel on the perfsim side without blocking iter-3.

| iter | workload | repo#issue | filed | status |
|---|---|---|---|---|
| 3-prerequisite | dsv3_train_full | ultrons/perfsim#7 | 2026-05-07 | open — P0; **blocked-by #19** (per maintainer analysis); ADR-001 migration |
| 3-prerequisite | dsv3_train_full | ultrons/perfsim#19 | 2026-05-07 | open — P0-blocker; per-op efficiency port into model_builder layers |
| 3-secondary | dsv3_train_full | ultrons/perfsim#8 | 2026-05-07 | open — P1; per-leaf confidence propagation |
| 3-secondary | dsv3_train_full | ultrons/perfsim#9 | 2026-05-07 | open — P1; surface BreakdownNode.assumptions in headroom report |
| 3-secondary | dsv3_train_full | ultrons/perfsim#10 | 2026-05-07 | open — P1; BF16 GEMM efficiency curve (no scalar fallback) |
| 3-secondary | dsv3_train_full | ultrons/perfsim#11 | 2026-05-07 | open — P1; gemm_efficiency_at extrapolation flag |
| (search) | dsv3_train_full | ultrons/perfsim#12 | 2026-05-07 | open — P2; wire cmd_search CLI |
| (search) | dsv3_train_full | ultrons/perfsim#13 | 2026-05-07 | open — P2; enumerate batch_sharded_by_ep in search |
| (search) | dsv3_train_full | ultrons/perfsim#14 | 2026-05-07 | open — P2; HBM fragmentation in search filter |
| (search) | dsv3_train_full | ultrons/perfsim#15 | 2026-05-07 | open — P2; profile corpus index/query API |
| (long) | dsv3_train_full | ultrons/perfsim#16 | 2026-05-07 | open — P3-design; overlap_compute_us static (FSDP_AG schedule-position blind) |
| (long) | dsv3_train_full | ultrons/perfsim#17 | 2026-05-07 | open — P3; inference _comm_node shallow audit |
| (long) | dsv3_train_full | ultrons/perfsim#18 | 2026-05-07 | open — P3-umbrella; unify inference + training simulator paths |

## Autoperf-side tasks blocking perfsim issues

| iter | autoperf task | unblocks perfsim# | added | status |
|---|---|---|---|---|
| 3-tooling | Launch `bench_runner` on v7x_4x8x8 with the (M, K, N) grid agreed on perfsim#10's needs-info | #10 (BF16 GEMM curve) | 2026-05-07 | pending — first action of iter-3 |

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
