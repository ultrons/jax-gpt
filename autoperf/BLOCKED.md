# BLOCKED.md — open tool-bug issues blocking autoperf iterations

The autoperf agent maintains this table. One row per filed issue. On each
iteration's step-1, the agent re-checks `open` rows; closed ones get marked
`resolved`, the relevant tool repo gets `git pull`ed, and the originally
blocked iteration's change is retried.

| iter | workload | repo#issue | filed | status |
|---|---|---|---|---|
| 1 | dsv3_train_full | ultrons/perfsim#1 | 2026-05-06 | resolved (training-regime wiring; PR merged 2026-05-06) |
| 1 | dsv3_train_full | ultrons/perfsim#4 | 2026-05-07 | fix-proposed (perfsim-agent session branch, awaiting human merge — Expert_gmm now 1.12× measured, total step 6.7% err; iter 1 unblocks on merge) |
| (architectural) | — | ultrons/perfsim#5 | 2026-05-07 | unblocked (xla-shell#1 closed; perfsim-agent has empirical substrings + per-operand comparison strategy in #5 comments; awaiting perfsim-side wiring DoD #2-5) |
| (architectural) | — | ultrons/xla-shell#1 | 2026-05-07 | resolved (get_op_shape API shipped + tested against v304 xplane; PR merged 2026-05-07) |

(Add rows below this header. Don't delete `resolved` rows — they're
audit history.)
