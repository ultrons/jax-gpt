# autoperf — agent-driven perf optimization

A long-running LLM session optimizes one workload at a time on TPU v7x by
iterating: apply ONE change → build → run with profile → measure → next change.
Stop when top-leaf headroom is below 0.5 ms total or when the agent halts
(novel failure, exhausted levers, etc).

## Architecture (as of 2026-05-07)

**1 doer + 3 reviewers.** The autoperf agent owns `jax-gpt` directly AND has
fix-inline authority on `perfsim`, `cde`, `xla-shell` for scoped bugs (via
worktree on `autoperf-loop` branch in each repo, opens PR, never merges).

The three sibling tool repos have **reviewer agents** that run hourly review
on `autoperf-loop` PRs, comment, never merge. Humans gate daily merges.

This replaces an earlier 4-agent design (autoperf + 3 maintainer fixers) where
cross-repo handoff via GitHub issues added multi-hour latency. See
`autoperf/MAINTAINER_REVIEWERS.md` for the role specification.

## Files

```
AGENT.md                  the harness — system prompt the LLM reads at start
README.md                 this file
MAINTAINER_REVIEWERS.md   reviewer-agent role spec (applies to sibling repos)
bootstrap.sh              one-time worktree setup
v7x_KNOWLEDGE.md          operational TPU/JAX knowledge ledger (anti-hallucination)
workloads/*.yaml          per-workload spec (model, hw, parallelism, cde overrides)
BLOCKED.md                ledger of GitHub issues + autoperf-side tasks blocking iter
iter_log.md               structured per-iter log (class, hypothesis, result, decision)
diary/<wl>.log            one line per iteration (commit-tracked audit trail)
reports/<wl>_iter<N>.json per-iter headroom report (gitignored)
profiles/<run-id>/        pulled cluster profile (gitignored)
```

The agent IS the orchestrator. There is no Python loop.

## One-time host setup

```bash
# 1. Symlink perfsim so AGENT.md path references resolve
ln -sfn ~/ml-experiments-perfsim ~/perfsim

# 2. Create cross-repo worktrees on autoperf-loop branches
bash ~/jax-gpt/autoperf/bootstrap.sh
```

The bootstrap script is idempotent — safe to re-run. It creates:
- `~/autoperf/repos/perfsim/` (worktree on `autoperf-loop` branch)
- `~/autoperf/repos/cde/` (worktree on `autoperf-loop`)
- `~/autoperf/repos/xla-shell/` (worktree on `autoperf-loop`)

Your primary checkouts at `~/perfsim`, `~/cde`, `~/xla-shell` stay on
whatever branch you're using; autoperf operates in the worktrees, doesn't
disturb your main work. Invocations from autoperf scripts use
`PYTHONPATH=~/autoperf/repos/<repo>` to override your editable install.

(There's also a third copy of perfsim under `~/ml-experiments/perfsim/` in
a separate git repo — currently bit-identical to the canonical copy, but
they may drift. Treat `~/ml-experiments-perfsim/` as authoritative.)

## How a human kicks it off

In a long-running Claude Code session in this repo:

```
Read autoperf/AGENT.md and run as the autoperf agent on
workload autoperf/workloads/dsv3_train_full.yaml.
```

**One iteration per session.** When the iteration completes (succeeded,
halted, or paused for review), the agent commits all state and ends the
session. You start the next iteration in a fresh session. Multi-iteration
sessions accumulate context that dilutes attention on the load-bearing
thread.

## Iteration policy split

Each iteration is one of three classes (logged in `iter_log.md`):

- **Greedy (60% / 40% post-search):** top-headroom from the trusted-leaf
  set in `v7x_KNOWLEDGE.md` §5. Map via `auto-perf-guide.md` heuristic table.
- **Lateral (25% / 40% post-search):** schedule-position experiment, second-
  best trusted leaf, untried lever. Once `perfsim search` is wired
  (perfsim#12), this draws from search's top-K.
- **Tooling (15% / 20% post-search):** invest in the cost model itself —
  bucketer fix, calibration improvement (run a calibration job, not a perf
  job), perfsim issue you've been deferring. **No on-cluster perf
  measurement.** Output is one or more PRs on sibling-repo `autoperf-loop`
  branches.

Note: search-driven mode (40/40/20) is gated on perfsim#12 (cmd_search CLI),
plus the P0/P1 cluster of perfsim#7-#11 for prediction trust.

## Concurrency model

`AGENT.md` has the agent enforce **max-parallel = 2 cluster jobs** by polling
`cde history --status running | wc -l` before launching anything.

## Stop conditions (autoperf self-enforces)

- top-leaf headroom < 0.5 ms total
- 3 consecutive regressions
- all leaves > 5% step-share at predicted ceiling
- training broke (NaN, OOM, no metric progress) — auto-revert + halt
- cluster unhealthy (3 evictions in a row)
- perfsim's reasoning didn't make sense and `--explain` didn't resolve it
  → halt with reason `perfsim_unverifiable`
- novel failure → file issue, halt with reason `tool_blocked_<repo>#<issue>`

Each halt writes `autoperf/HALT.md` with diagnosis + recommended next human
action.
