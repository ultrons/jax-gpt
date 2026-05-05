# autoperf — agent-driven perf optimization

A long-running LLM session optimizes one workload at a time on TPU v7x by
iterating: apply ONE change → build → run with profile → measure → next change.
Stop when top-leaf headroom is below 0.5 ms total or when the agent halts
(novel failure, exhausted levers, etc).

## Files

```
AGENT.md             the harness — system prompt the LLM reads at start
README.md            this file
workloads/*.yaml     per-workload spec (model, hw, parallelism, cde overrides)
BLOCKED.md           ledger of GitHub issues blocking iteration progress
diary/<wl>.log       one line per iteration (commit-tracked audit trail)
reports/<wl>_iter<N>.{md,json}  per-iter headroom report (gitignored)
profiles/<run-id>/   pulled cluster profile (gitignored)
```

That's the whole repo. The agent IS the orchestrator. There is no Python loop.

## How a human kicks it off

In a long-running Claude Code session in this repo:

```
Read autoperf/AGENT.md and run as the autoperf agent on
workload autoperf/workloads/dsv3_train_full.yaml.
```

The agent then reads AGENT.md, the workload yaml, and starts iterating per
the steps in §3 of AGENT.md. Same instruction works in any LLM with shell
access (Gemini CLI, codex, etc).

## Concurrency model

`AGENT.md` has the agent enforce **max-parallel = 2 cluster jobs** by polling
`cde history --status running | wc -l` before launching anything. No worker
pool, no Python threading — the agent is single-threaded by nature, but lets
2 jobs run concurrently by launching one and continuing to the next iteration's
prep work without waiting.

## Tool ecosystem

`autoperf` treats `cde`, `perfsim`, and `xla-shell` as black-box tools owned by
sibling agents (see `~/cde/AGENT.md`, `~/xla-shell/AGENT.md`,
`~/ml-experiments-perfsim/AGENT.md`). When `autoperf` hits a structured tool
bug, it files a GitHub issue against the offending repo via `gh issue create`
and HALTs the iteration with reason `tool_blocked_<repo>#<issue>`.

The corresponding tool agent picks up the issue, fixes it, comments + closes.
On the next iteration, autoperf checks `gh issue view <repo>#<issue>` — if
closed, pulls the fixed tool's repo and retries the previously-blocked change.

## Stop conditions (autoperf self-enforces)

- top-leaf headroom < 0.5 ms total
- 3 consecutive regressions
- all leaves > 5% step-share at predicted ceiling
- training broke (NaN, OOM, no metric progress) — auto-revert + halt
- cluster unhealthy (3 evictions in a row)
- novel failure → file issue, halt with reason `tool_blocked_...`

Each halt writes `autoperf/HALT.md` with diagnosis + recommended next human
action.
