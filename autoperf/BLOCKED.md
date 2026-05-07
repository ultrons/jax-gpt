# BLOCKED.md — open tool-bug issues blocking autoperf iterations

The autoperf agent maintains this table. One row per filed issue. On each
iteration's step-1, the agent re-checks `open` rows; closed ones get marked
`resolved`, the relevant tool repo gets `git pull`ed, and the originally
blocked iteration's change is retried.

| iter | workload | repo#issue | filed | status |
|---|---|---|---|---|
| 1 | dsv3_train_full | ultrons/perfsim#1 | 2026-05-06 | resolved (training-regime wiring; PR merged 2026-05-06) |
| 1 | dsv3_train_full | ultrons/perfsim#4 | 2026-05-07 | open (BLOCKING — predicted > measured for compute leaves; batch_sharded_by_ep wiring missing for gmm_ag) |
| (architectural) | — | ultrons/perfsim#5 | 2026-05-07 | open (defense-in-depth: validate perfsim dims against xplane HLO; not blocking iter 1) |

(Add rows below this header. Don't delete `resolved` rows — they're
audit history.)
