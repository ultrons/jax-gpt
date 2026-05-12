# AUTONOMOUS_RUN.md — autoperf agent session contract

This file is the **session contract** for an autonomous autoperf run. When
present, the agent reads it on every Step 1 and continues self-authorized
iteration without per-iter user prompt, within the risk envelope below.

When the agent decides the session is over (or the time budget expires),
it deletes this file at iter closeout — the next session starts in
normal user-paced mode.

---

## Session metadata

- **Started**: 2026-05-12
- **Workload**: `autoperf/workloads/dsv3_train_full.yaml`
- **Branch**: `autoperf/dsv3_train_full`
- **Time budget**: 12 hours from start (hard halt)
- **Cluster shot budget**: 8 (hard cap; each ~30-40 min wall)
- **Author**: human (Vaibhav)

## Pre-authorized scope

The following are auto-authorized — no user prompt needed:

**Resume authorization 2026-05-12 04:13 UTC (post-HALT_FOR_AUTH)**:
User selected Option A. Bisect jax-gpt#4 NaN class via single-name
SAVE adds, smallest HBM first: kv_a (2.1 GB) → q_a (5.6 GB) →
shared_hidden (7.4 GB). Each iter compares to iter-16's measurement.
If a name lands clean with measured improvement, KEEP it on the save
list for the next iter (cumulative stacking). If a name NaN's,
DROP it and proceed to the next candidate. Stop if all 3 fail OR
when cluster budget exhausts.


1. **Iter-17 ratchet of iter-16.** Same image (cde-b950f34), same flags,
   second measurement. Per AGENT.md §5b: gain within ±0.3% of 34,200 ms
   → promote to PRODUCTION BASELINE + ratchet corpus. Outside band →
   file noise finding, hold baseline at iter-2b.

2. **Multi-iter scope #3 — Chunk-pipelining/overlap fix.** Diagnosed
   in iter-13 (~2.4% TPS in body-tail exposure). Sub-iter budget: up
   to 4 iters across diagnosis + implementation + cluster validation.
   Per the iter-13 lever catalog, EXCLUDED variants: n_chunks=4
   (blocked by Bug 3, multiple-failed-fix-attempts) and ragged_a2a
   collective fusion (gate-rejected in iter-14 on engine-assignment
   evidence). Permitted variants: cross-iter prefetch (Lever B,
   moe_scan body refactor), or a NEW variant the agent surfaces via
   diagnosis.

3. **Greedy levers on trusted-leaf set** drawn from the heuristic
   table or diagnosis. Trusted set per `v7x_KNOWLEDGE.md` §5.

4. **Tooling iters** (no cluster spend). Free to spawn.

5. **Inline tool fixes** on sibling-repo `autoperf-loop` branches. PR
   open, log in BLOCKED.md, continue.

## NOT pre-authorized (must halt + surface to user)

- Code changes touching >5 files or >100 lines in a single iter
  (call advisor() and write a HALT_FOR_AUTH.md instead of proceeding)
- Multi-iter scope OPENING beyond #3 (after #3 closes, halt for next-
  scope authorization unless the time budget already permits a 4th)
- Any modification to `manifests/jobset.yaml.j2` template
- Any operation on `main` or shared branches
- Any `gcloud auth` or credential operations
- Force-push, branch deletion, `kubectl delete` of resources not
  created by this session

## Hard halt conditions (write HALT.md, stop, do not continue)

1. **Time budget exceeded** — 12 hours elapsed from session start
2. **Cluster shot budget exceeded** — 8 cluster runs counted
3. **§13 cumulative HALT** — regression chain + diagnosis-derived attempt
4. **NaN class not seen before** — file jax-gpt issue, halt for user
   (don't continue trying offload-pipe variants until #2/#3 lands)
5. **Cluster_unhealthy** — 3 evictions in a row OR cluster context
   unresponsive (`kubectl get ns` timeout)
6. **Synthesis step concludes the search space is exhausted** — see
   §3 below

## §1 Three autonomous primitives

These augment AGENT.md (not replace). When this file is present,
they're active.

### Primitive A — Synthesis-every-3-iters

Every 3 iters (or after every halt-with-revert, whichever first),
spawn a `general-purpose` subagent with the prompt:

> Review the last 3-5 iter sections in `autoperf/iter_log.md` +
> any new issues filed + any new corpus anchors. Find ONE
> cross-iter pattern that no single iter would surface. Suggest
> ONE specific experiment to test the pattern. Output budget:
> 400 words.

The subagent's hypothesis lands as a comment block at the top of
`autoperf/lever_queue.md` (file auto-created). The next iter's
lever-pick draws from this queue first, then falls back to
heuristic table.

### Primitive B — Parallel iter pipelining

After submitting cluster run for iter-N (background via cde), the
agent IMMEDIATELY starts designing iter-N+1: read iter-N's predicted
profile (perfsim), survey heuristic table, draft candidate edit.
When iter-N completes:
- Run completed successfully → evaluate, close iter-N, decide:
  apply iter-N+1 draft OR pivot based on iter-N findings
- Run NaN'd / OOM'd → revert iter-N, file issue, decide iter-N+1
  from queue (don't use the now-stale draft)

Limit: max 1 cluster job in flight at a time. The "parallel" is
agent design vs cluster execution, not parallel cluster submits.
(AGENT.md max-2 parallel applies if a multi-iter scope warrants
batched submissions — for this session keep single-stream.)

### Primitive C — Self-authorization within risk envelope

For each candidate iter:

1. Match against "Pre-authorized scope" above → auto-proceed
2. If not matched: check if it's a simple Greedy/Lateral with
   diagnosis-derived mechanism → call advisor(); if advisor returns
   "proceed", run; if "concerns/halt", write HALT_FOR_AUTH.md
3. If multi-iter scope OPENING or >5-file change: write
   HALT_FOR_AUTH.md, stop autonomous run

Self-authorization is logged in iter_log: `auth=auto` (matched
pre-auth scope), `auth=advisor-proceed` (advisor approved),
`auth=halt-for-user` (could not self-authorize).

## §2 Cadence + wake-ups

- Cluster wait → ScheduleWakeup(delaySeconds=900, reason="poll iter-N status")
  every 15 min. Cache stays warm (under 5 min would be best but
  cluster compile is ~10+ min so 15 min is the right cadence)
- Synthesis step → run inline (not scheduled)
- Iter closeout → commit + push immediately; rehydration block
  durable BEFORE next iter starts
- Between iters → no scheduling; agent loops directly

## §3 Session conclusion criteria

Agent declares session complete when ANY of:

- Time budget exhausted (write HALT.md "time_budget_exceeded")
- Cluster shot budget exhausted ("cluster_budget_exceeded")
- Hard halt fires (see above)
- Multi-iter scope #3 closes AND no further pre-authorized scope remains
- Synthesis step concludes "no lever class with positive expected
  information value remaining within budget"

At conclusion: write final HALT.md with session score (iters run,
cluster shots used, perf gains, issues filed, corpus anchors,
PR queue), delete THIS file (AUTONOMOUS_RUN.md), commit + push.

## §4 What the user comes back to

A clean HALT.md with the session summary, a corpus PR queue (perfsim
PR#49 + any iter-17/18+ new ones), iter_log entries through the last
iter, a synthesized lever_queue.md the next user-paced session can
start from, and a clear "what's the next user decision required"
recommendation.

The user does NOT come back to mid-iter state — every iter is closed
cleanly or the session is halted between iters.
