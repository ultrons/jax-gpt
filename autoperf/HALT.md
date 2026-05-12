# HALT — autoperf autonomous run, 2026-05-12

**Reason**: time budget exceeded (903 min elapsed vs 720 min / 12 h target).
Session also hit hard halt #4 mid-run (NaN class not seen — iter-18).

**Window**: 2026-05-12 01:14 → 16:48 UTC (15h 34min wall)
**Workload**: dsv3_train_full, iter-2b baseline preserved
**Branch**: autoperf/dsv3_train_full HEAD `37c6b8c`
**Autonomous-mode contract**: AUTONOMOUS_RUN.md (deleted at session close)

---

## Session score

| metric | count |
|---|---|
| iters attempted | 4 (17, 17b, 18, 19) |
| cluster shots used | 4 of 8 budget |
| successful trains (loss finite + ran) | 0 |
| partial trains (steps captured) | 1 (iter-17b: 4 valid steps before preempt) |
| NaN halts | 1 (iter-18) |
| evictions | 3 (iter-17, iter-17b, iter-19) — broken by iter-18 NaN |
| issues filed | 1 (jax-gpt#4) |
| inline harness fixes | 2 (.dockerignore exclude autoperf/+cde.yaml; HALT_FOR_AUTH protocol exercised) |
| perfsim PRs OPEN | 1 (#49 — iter-16 IMPROVED candidate corpus update) |
| perf gain measured + clean | 0 |
| perf gain measured + partial | +3.6% TPS (iter-17b vs iter-2b, partial profile) |
| commits pushed | 12 (e52bfcb → 37c6b8c) |

**Production state unchanged**: iter-2b remains baseline (1882 TPS/chip @ 30.5% MFU, 34,659 ms/step). iter-16 IMPROVED candidate (1916 TPS/chip @ 31.1% MFU) still awaits clean repeat for ratchet — iter-17b's partial measurement (1949 TPS/chip @ 31.5% MFU) is suggestive but not full-iter ratifying.

## What happened per iter

### iter-17 — EVICTED (ImagePullBackOff)

Submit 01:14, evicted 02:14 via PodsReady timeout. Root cause: autoperf/ files in docker build context caused tag drift between submit and rebuild. **Inline fix** (`1629109`): `.dockerignore` excludes `autoperf/` + `cde.yaml`. Durable harness improvement — same root cause bit iter-16's first attempt on 2026-05-11.

### iter-17b — EVICTED (preempted)

Retry as iter-17b on `cde-41b3703`. Admitted 03:18 (55 min queue), ran 4 valid steps then preempted by `mk-q30b-0508` workload at 03:30. Step results captured:

| step | seconds | TPS/chip | MFU |
|---|---|---|---|
| 2 | 33.528 | 1955 | 31.6% |
| 3 | 33.733 | 1943 | 31.4% |
| 4 | 33.571 | 1952 | 31.6% |

Avg 33,611 ms / **1949 TPS/chip**. Confirms iter-16 direction (+3.6% vs iter-2b; +1.7% over iter-16's own 34,203 ms). Loss matches iter-2b exactly. NOT ratcheted to baseline per AGENT.md §5b — eviction = partial data.

### iter-18 — REVERTED (nan_at_step1, NEW NaN class)

Greedy stack of SAVE list to 4 names (`attn_proj_out + q_a + kv_a + shared_hidden`). Drafted via Primitive B during iter-17b wait. Compile clean (234s), step 1 loss=nan. Reverted (`e3ee729`). Filed **`ultrons/jax-gpt#4`** with full repro + untried-alternative-paths enumeration. Distinct from jax-gpt#2/#3 (OFFLOAD path) — this is the SAVE path failing on names beyond `attn_proj_out`.

### iter-19 — EVICTED (PodsReadyTimeout + cluster image-pull issues)

User authorized Option A (bisect, smallest-first single-name): tried `kv_a` alone. Image built and pushed 04:13. Queued in Kueue for 12 hours (cluster contended). Admitted 16:17, evicted 16:47 — pods never reached Ready (cluster-wide `FailedToRetrieveImagePullSecret` warnings affecting multiple non-autoperf workloads at the same time). No measurement.

## Demonstration findings — insights that emerged

### 1. Per-iter variance is ~1.7%, not ±0.3%

The ratchet noise band in AUTONOMOUS_RUN.md was set at ±0.3%. iter-16 measured 34,203 ms; iter-17b (identical config, just repeat) measured 33,611 ms — a 1.7% delta. **Insight**: single-iter measurements can't reliably distinguish gains smaller than ~2%. Future ratchet criterion should be ±1.5% over ≥2 clean repeats. Logged in `lever_queue.md` synthesis log.

### 2. SAVE-list expansion has a NaN failure mode

iter-16 established that SAVE (HBM) is distinct from OFFLOAD (host) — different code path, different known failures. iter-18 disproved the implicit assumption that SAVE is safe-by-default. The SAVE path also has failure modes that show up only on combination expansion — likely interaction with the broader `_ckpt_policy` recompute logic when multiple names are persisted. Filed jax-gpt#4 with bisect plan.

### 3. Cluster preemption is a hidden cost at medium priority

3 of 4 cluster shots evicted this session. Higher-priority workloads (`mk-q30b-0508`, `cloud-t-ubji2w`, `olmo3-s1-2192`) compete for slices. Even when admitted (iter-17b), preemption mid-run cost ~half the measurement (step 5 profile capture cut). **Implication**: contended-cluster sessions need eviction-survival design — frequent step-time logging, partial-data acceptance, possibly priority-class bump for time-sensitive ratchet runs.

### 4. Build context discipline is load-bearing

The `autoperf/` directory being in docker build context caused 2 ImagePullBackOff failures across the session (iter-16's first try, iter-17). Each cost ~30 min wall. The systematic fix (`.dockerignore` exclusion) prevents recurrence forever and is exactly the kind of inline tool-friction fix AGENT.md §1 calls out as "tools mature with iteration".

## Three autonomous primitives — how they performed

### Primitive A — Synthesis-every-3-iters

**Status**: Did not fire formally (session halted at iter-18 + iter-19 before reaching iter count for synthesis). Hypotheses surfaced informally in `lever_queue.md` synthesis log (variance noise band, ratchet criterion update). For next session: synthesis should trigger off the iter-18 NaN halt, not just iter count.

### Primitive B — Parallel iter pipelining

**Status**: Worked. iter-18 was fully drafted in `lever_queue.md` during iter-17b's queue wait. Saved ~10 min wall when iter-17b completed and the agent immediately applied the patch + submitted. Single-stream cluster execution + parallel agent design is the right balance for medium-priority contended clusters.

### Primitive C — Self-authorization within risk envelope

**Status**: Worked, and **HALT_FOR_AUTH protocol triggered correctly** on iter-18's new NaN class. Agent did NOT continue blindly bisecting; halted and surfaced 4 options. User selected Option A, agent resumed. This is the primitive working as designed — autonomous within envelope, halts on novel failure class.

## State on disk at session close

- `autoperf/iter_log.md`: iter-17/17b/18 sections + rehydration block (iter-19 not logged because no measurement)
- `autoperf/lever_queue.md`: iter-18 candidate + iter-17b synthesis findings + iter-19 plan
- `autoperf/BLOCKED.md`: jax-gpt#4 needs adding (file existed before session; add this iter)
- `autoperf/v7x_KNOWLEDGE.md`: 3 new operational findings (SAVE-vs-OFFLOAD, cde.yaml build-hash, cluster reset)
- `jax_gpt/models/dsv3/model.py:3059`: SAVE list at `("attn_proj_out", "kv_a")` from iter-19 commit `37c6b8c`. This is **un-tested live state** — kv_a was never validated on cluster. Next session should either (a) re-submit iter-19 for kv_a bisect, or (b) revert to iter-16's single-name state.
- `autoperf/AUTONOMOUS_RUN.md`: DELETED at session close.
- `autoperf/HALT_FOR_AUTH.md`: DELETED (was the iter-18 surface; resolved by user Option A authorization).
- jax-gpt issues open: #2 (offload attn_proj_out NaN), #3 (prevent_cse NaN), #4 (SAVE-list expansion NaN, NEW this session).
- perfsim PR open: #49 (iter-16 IMPROVED corpus update, needs review/merge).

## Recommended next user actions

1. **Resolve iter-19's un-validated state**: either revert `model.py:3059` back to iter-16's single-name SAVE list, or rerun iter-19 in a fresh session (cluster permitting). Currently the on-disk state has `kv_a` added — if you build + submit without verification, that's the iter-18-class NaN risk.

2. **Merge perfsim PR#49** to land the iter-16 IMPROVED candidate in the corpus.

3. **Triage jax-gpt#4** with the bisect plan in the issue body. If a maintainer can identify which name(s) NaN individually, autoperf can pick up the safe ones in a future session.

4. **Cluster scheduling improvement (optional)**: the medium-priority preemption pattern bit 3/4 cluster shots this session. Consider either (a) higher priority class for ratchet/critical iters, (b) different cluster context with less contention, or (c) accept the eviction-survival overhead.

## Demo verdict — "insights emerge from running experiments"

**Met partially.** 4 insights emerged that single-iter mode would not have surfaced:
- Per-iter variance is ~1.7% (cross-iter comparison)
- SAVE list has expansion-mode NaN failure (combination probe)
- Build-context discipline is load-bearing (failure-pattern repetition)
- Cluster preemption hidden cost (eviction-rate statistic)

**Did NOT meet**: no NEW perf measurement landed cleanly. iter-2b remains baseline. iter-16's +1.8% candidate remains un-ratcheted. The session's measured value is the insights + harness improvements, not a new perf number.

The autonomous primitives functioned correctly — the limiting factor was cluster availability (3 of 4 cluster shots evicted) plus user-authorization gates (correct behavior on novel NaN, but a real wall-clock cost on the Option A delay).
