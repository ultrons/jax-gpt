# HALT_FOR_AUTH.md — autonomous run paused for user decision

**When**: 2026-05-12 04:10 UTC (~3 hours into 12-hour autonomous window)
**Reason**: AUTONOMOUS_RUN.md hard halt #4 — NaN class not seen before
**Recommended next**: user picks one of the options below; agent resumes

---

## What happened (concise)

| iter | outcome | data point |
|---|---|---|
| 17 | EVICTED (ImagePullBackOff) | inline fix: .dockerignore exclude autoperf/+cde.yaml |
| 17b | EVICTED (preempted mid-step-5) | 4 valid steps: 33,611 ms avg / 1949 TPS-chip / 31.5% MFU; CONFIRMS iter-16 direction (+3.6% vs iter-2b) |
| 18 | REVERTED (nan_at_step1) | Stack SAVE list (4 names) NaN's; new failure class (SAVE-path, not OFFLOAD) |

iter-18 filed as `ultrons/jax-gpt#4` with full repro + untried-alternative-paths enumeration per AGENT.md §13.

## Why halted

AUTONOMOUS_RUN.md §1 hard halt #4: "NaN class not seen before → file jax-gpt issue, halt for user". The autonomous loop's purpose is to surface novel NaN-classes for user judgement, not blindly burn cluster slots on family-of-broken-levers.

The iter-18 NaN is in the SAVE code path (not OFFLOAD), which is genuinely new — iter-16 worked, iter-18 broke it by adding 3 names. We don't yet know which name is culprit.

## Cluster shot budget

Used: 3 of 8 (iter-17 evict, iter-17b evict-with-data, iter-18 NaN-with-revert).
Remaining: 5 cluster shots within 9 hours.

## Options for user

### Option A — Authorize iter-19 bisect (recommended)

Pre-authorize the agent to bisect jax-gpt#4 via single-name SAVE adds, in HBM-budget order (smallest first to maximize success probability):

1. iter-19: `kv_a` alone (+2.1 GB) — smallest HBM addition
2. iter-20 (if iter-19 lands clean): `q_a` alone (+5.6 GB)
3. iter-21 (if iter-20 lands clean): `shared_hidden` alone (+7.4 GB)

If all three land clean individually, the NaN is a combination-only effect. If one specific name NaN's, we've isolated the culprit + can keep the safe ones on the SAVE list. Worst case: 3 wasted iters but full information.

**Expected gain if at least one name is safe**: +0.3 to +1% additional TPS stacking on iter-16's +1.8%.

### Option B — Authorize multi-iter #3 (chunk-pipelining)

Skip the iter-18 bisect, pivot to the multi-iter scope #3 candidate that's already pre-authorized in AUTONOMOUS_RUN.md (cross-iter prefetch via moe_scan body refactor). ~+0.5-1% TPS expected.

### Option C — Authorize iter-19 = clean ratchet of iter-16

Repeat iter-16 (just attn_proj_out alone) one more time to get a full clean profile + corpus-anchorable measurement, then promote to PRODUCTION BASELINE. Conservative — confirms iter-16 before any further exploration.

### Option D — End the session

Write final HALT.md with session score, delete AUTONOMOUS_RUN.md. iter-16 stays as IMPROVED candidate; ratchet attempt postponed to future session.

## How to resume

User edits AUTONOMOUS_RUN.md to either:
- Append "iter-19 bisect (single-name SAVE) authorized" → agent picks up Option A on next session
- Or replace this file's recommendation with a different scope

Then ping the agent (next session). Agent reads HALT_FOR_AUTH.md at Step 1a0b, sees the authorization edit, and resumes.

## Other findings to log

**Variance noise band**: iter-16 (34,203 ms) and iter-17b (33,611 ms) differ by 1.7%. The ±0.3% ratchet noise band in AUTONOMOUS_RUN.md is too tight given per-iter variance. Future versions should set ratchet band to ~±1.5%.

**Cluster preemption risk**: 2 evictions in 3 cluster shots this session. Higher-priority workloads (mk-q30b-0508) are taking slices. iter-18 succeeded because it admitted faster (20 min vs iter-17b's 55 min) and completed before next preemption window.

**.dockerignore fix worth noting**: autoperf/ in build context was a quiet build-hash poison that bit iter-16's first try AND iter-17. Now fixed for all future iters.
