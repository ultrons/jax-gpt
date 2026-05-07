# Maintainer reviewer agents — role under the 1-agent autoperf design

**Audience:** the `~/perfsim/AGENT.md`, `~/cde/AGENT.md`, `~/xla-shell/AGENT.md`
files. This doc lives in `jax-gpt` because autoperf coordinates the cross-repo
workflow, but the actual content needs to be applied to each sibling repo's
AGENT.md by the user (or by each maintainer agent at the start of its next
session).

## What changed (2026-05-07)

The 4-agent design (autoperf + 3 maintainer fixers) is being collapsed to:
- **1 doer:** autoperf agent has fix-inline authority across all 4 repos.
- **3 reviewers:** perfsim/cde/xla-shell agents move from "fix issues" to
  "review autoperf-loop branches and PRs" + "handle structural / non-scoped
  bugs filed as issues."

Why: cross-repo handoff latency in the 4-agent design (autoperf → maintainer
→ close → autoperf retries) added multi-hour delay for fixes the autoperf
agent could land in 30 seconds. The reviewer role preserves second-pair-of-
eyes safety without the latency.

## Reviewer role — what changed in your AGENT.md

Each maintainer agent's AGENT.md should now describe:

### Primary responsibilities

1. **Hourly review pass over `autoperf-loop` PRs in this repo.** Read diff,
   run tests if scoped, post a PR comment. **Never merge.** Humans (or a
   dedicated reviewer-agent with merge authority) gate merges.

2. **Handle structural bugs filed as issues.** If autoperf opens an issue
   (rather than a PR) against this repo with label `autoperf-blocking`,
   that's a sign the bug is not scoped — design discussion, cross-cutting
   refactor, etc. These remain your traditional fix-and-PR responsibility.

3. **Validate autoperf's fixes against repo-specific norms.** AGENT.md §1
   norms — boundary discipline (no kernel-zoo creep for perfsim, no
   workload-specific fudge factors, etc.) — apply equally to autoperf's
   inline fixes. Push back via PR comment when a fix violates them. The
   autoperf agent has explicit instructions to defer to your review on
   norm-conformance questions.

### Secondary responsibilities (unchanged from prior AGENT.md)

- Maintain repo health: tests, docs, validation corpus, ADRs.
- Respond to direct user requests for feature work that's not autoperf-loop-
  scoped.
- Tighten validation tolerances when ratchet requests come in (perfsim).

### Removed responsibilities

- **No more "fix issues filed by autoperf"** as the default flow. Autoperf
  fixes scoped bugs inline; you review those PRs. Issues are reserved for
  unscoped work (which you still own).
- **No autoperf-blocking-as-default issue mode.** Many former autoperf-
  blocking issues will now arrive as PRs you review instead.

## Reviewer's PR review checklist (per-repo)

When reviewing an `autoperf-loop` PR, a maintainer agent should answer:

1. **Scope check.** Does this PR change one thing that's clearly localized?
   If it's bigger than that — refuse to review, request a redesign as an
   issue first. Don't let autoperf accumulate technical debt by sneaking
   refactors through "fix" PRs.
2. **Norm conformance.** Does this PR violate any AGENT.md §1 invariants?
   - perfsim: no kernel-zoo creep, no workload-specific calibration overrides
   - cde: no skipping CI hooks, no force-pushes
   - xla-shell: no scatter_p on JAX arrays, no breaking xplane parsing API
3. **Test coverage.** Did autoperf add tests for the change, or update
   existing ones? If the fix is a calibration tweak, did it update the
   validation corpus snapshot?
4. **Cross-repo consistency.** Does the fix in this repo coordinate with
   any related fix in another repo (e.g., perfsim shape-cross-check needs
   xla-shell get_op_shape API to land at compatible version)?

## Hourly cadence — implementation suggestion

Reviewer agent runs as a long-lived Claude Code session with a self-paced
loop:

```
1. gh pr list --repo ultrons/<repo> --state open --head autoperf-loop --json number,updatedAt
2. For each PR with comments older than 1 hour OR no comments yet:
   - read diff
   - run repo-local tests if available (perfsim: pytest; cde: mypy + tests; xla-shell: pytest)
   - post PR comment with checklist verdict (approve / request-changes / question)
3. ScheduleWakeup or Cron 1h
```

Concrete invocation pattern depends on whether you use Claude Code's
CronCreate or a manual `/loop` self-pacing — both work; pick one.

## What stays in your AGENT.md

Everything else: branch conventions, repo file paths, tool-specific
operational knowledge, ADRs, validation corpus discipline. The scope of the
agent narrows; the discipline does not.

## What autoperf knows about you

The autoperf agent's AGENT.md §5 instructs it to:
- Default to fix-inline for scoped bugs (not file-an-issue).
- Open PRs from `autoperf-loop` branch, never merge.
- Defer to maintainer-agent PR review for norm-conformance questions.
- File issues only for non-scoped work (design changes, structural refactors).

So if you find autoperf opening many tiny PRs that should have been one
larger issue, push back on the FIRST one with "this is part of a larger
refactor; please file an issue and we'll plan it together." Autoperf will
read your comment in the next iteration's step-1 and adjust.

## Migration steps for each maintainer's next session

1. Read this file once.
2. Update your repo's AGENT.md per the "What changed" sections above.
3. Confirm `autoperf-loop` branch exists in the repo (autoperf creates it
   on first cross-repo fix; if missing, it'll be created by autoperf's
   bootstrap).
4. Set up the hourly review loop (or document that a human triggers it
   manually for now).
5. Commit the AGENT.md update on whatever branch you typically work on.

## When to abandon this design

If reviewer-agent feedback latency starts becoming the bottleneck (e.g.,
autoperf has to wait >24h for a PR review on a normal change), that's a
signal to move the reviewer cadence faster, not to revert to 4-agent. The
4-agent design's coordination tax was higher than its review value.
