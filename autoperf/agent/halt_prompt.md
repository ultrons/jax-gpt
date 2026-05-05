# autoperf — HALT contract

When the iter prompt asks you to emit a HALT verdict, write the JSON in this
exact shape:

```json
{"action": "HALT", "reason": "<one-line categorical reason>"}
```

## Standard halt reasons (use one of these where it fits)

| reason | when to use |
|---|---|
| `no_lever_for_top_leaf` | top-headroom leaf isn't in the heuristic table; no obvious code change to attempt |
| `workload_at_ceiling` | predicted ≈ measured for all leaves > 5% step share |
| `novel_failure_<short-desc>` | NaN / OOM / libtpu issue / unexpected error during eval; needs human |
| `regression_chain` | last N iterations all regressed; we're digging a hole |
| `kernel_rewrite_required` | only viable lever is a Pallas kernel write; needs human review |
| `outside_perfsim_scope` | leaf is in section 7 of perfsim-protocol.md (continuous-batching, comm headroom not anchored, indexer cost, cross-host PP) |

## Things HALT is NOT for

- "I'd rather try X tomorrow" — propose X today, that's the loop's job.
- "Compile took longer than expected" — that's a budget concern, the budget
  check fires separately.
- "I'm not sure my change worked" — measure it; the headroom report is the
  authority.
