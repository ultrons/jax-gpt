# BLOCKED.md — open tool-bug issues blocking autoperf iterations

The autoperf agent maintains this table. One row per filed issue. On each
iteration's step-1, the agent re-checks `open` rows; closed ones get marked
`resolved`, the relevant tool repo gets `git pull`ed, and the originally
blocked iteration's change is retried.

| iter | workload | repo#issue | filed | status |
|---|---|---|---|---|
| 1 | dsv3_train_full | ultrons/perfsim#1 | 2026-05-06 | resolved 2026-05-06 |
| 2 | dsv3_train_full | ultrons/perfsim#3 | 2026-05-06 | open (non-blocking; needed for iter-2 compute-leaf ranking) |

(Add rows below this header. Don't delete `resolved` rows — they're
audit history.)
