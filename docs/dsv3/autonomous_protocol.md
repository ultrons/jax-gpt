# DSv3 autonomous-execution protocol

**Context**: Vaibhav focuses on the v4 fused MoE routing kernel. Claude executes the
EP=4, FSDP=128 perf path (chunks pipeline + scatter Pallas + tile/MFU work) without
needing per-iteration sign-off.

Edit anything below — this is a contract. Save changes here so they persist.

---

## Authority — what Claude can do without asking

- Build / push docker images: `gcr.io/tpu-vm-gke-testing/mini-dsv3:vXXX-<tag>`
- Apply / delete JobSets in `poc-dev` namespace on bodaborg cluster
- Modify these files freely:
  - `mini_dsv3/model.py`: MoE backend bodies (`_expert_mlp_gmm_*`, `_moe_gmm_ar`,
    `_moe_gmm_ag`, `_moe_jax_ep_rd`, all bodies they call)
  - `mini_dsv3/model.py`: XLA flag dispatch helpers
  - `kernels/`: anything except `forward_kernel.py`
  - YAMLs in `k8s/` (new versions)
- Commit to current branch (`g_fix`) with descriptive messages
- Update `docs/profile_inventory.md`, `docs/daily_status.md`, memory entries
- Schedule autonomous-loop wakeups
- Try arbitrary XLA flag combinations (within `LIBTPU_INIT_ARGS`)

## Hard stops — wait for Vaibhav

- Push to remote / open PR / merge / touch other branches
- Touch the kernel work: `forward_kernel.py`, `backward_kernel_v4.py`,
  `fused_ep_moe_v4*`, anything under `tpu_inference/kernels/`
- Modify `init_params` signature OR moe weight storage shape (cascades)
- Modify `_carry_spec`, `mla_attention`, `_dense_layer_body` (system-wide effects)
- 3 consecutive failed iterations on the same concept → pause + write a
  "stuck on X, here's what I tried" note to `daily_status.md`
- Negative perf regression >10% from v304 baseline (1948 TPS/chip) that
  Claude can't explain in one paragraph
- Touch other clusters (`sivaibhav-exp-v7x`, etc.) — bodaborg/poc-dev only

## Disk maintenance setup (installed 2026-04-27)

**Background**: dev VM disk fills from xla_shell GCS cache (12 GB), uv/pip caches
(19 GB), profile downloads, and continuously-growing system logs (~6 GB/day).

**Two layers of defense**:

1. **systemd timer** (auto, every 30 min): truncates `/var/log/kern.log` and
   `/var/log/syslog` to keep them bounded. Survives reboots via `Persistent=true`.
   - Service: `/etc/systemd/system/log-truncate.service`
   - Timer:   `/etc/systemd/system/log-truncate.timer`
   - Inspect: `systemctl list-timers log-truncate.timer`
   - History: `journalctl -u log-truncate.service --since today`
   - Disable: `sudo systemctl disable --now log-truncate.timer`

2. **`~/cl.sh`** (manual, run on demand): clears the BIG one-off offenders.
   Recovers ~32 GB on a typical filled disk.
   - `xla_shell_gcs_cache` (~12 GB)
   - `~/.cache/uv`, `~/.cache/pip` (~19 GB combined)
   - `/tmp/v*_prof`, `/tmp/llo_*`, `/tmp/tpu_logs` (~3 GB)
   - Plus docker prune + log truncation
   - Run as: `sudo ~/cl.sh`

**Trigger for cl.sh**: shell returns exit 1 on `echo`, or `df -h /` shows >90% used.
With the timer running, expect to need cl.sh maybe weekly instead of daily.

## Recovery procedures Claude will execute

| Symptom | Action |
|---|---|
| Shell returns exit 1 on `echo` | Disk full. Run `sudo ~/cl.sh` (recovers ~32 GB; see "Disk maintenance" above) |
| `SliceCreationFailed` on jobset | Stale MIG attachment. `kubectl delete jobset` + `sleep 15` + `kubectl apply` |
| `JobSet immutable` on apply | Wait 30s for Kueue cleanup, retry |
| Pod OOMKilled | Distinguish HBM compile-time (HLO temps) vs host pinned (offload × 8 devices/pod) |
| NaN at step 1 | Likely sharding/spec mismatch between MoE output and carry_spec |
| RuntimeProgramAllocationFailure | Compiled binary too big; try smaller GBS or simpler backend variant |
| Cluster congested (no slices available) | Wait 1hr, retry. If still failing 3hr later, write status and pause |

## Per-experiment loop

```
1. build image (sudo docker build) → push (sudo docker push)
2. apply YAML (kubectl -n poc-dev apply -f ...)
3. ScheduleWakeup ~9 min
4. Check pods + logs:
   - if NaN/error/OOM → diagnose, propose fix, build vXXXb, retry (max 3 attempts/concept)
   - if running → wait for steps 2-5 stable
   - if regression → revert to last known-good baseline (v304-auxar), document why
5. Record TPS/chip + step time + MFU in profile_inventory.md
6. Clean up: kubectl delete jobset
7. Move to next planned step
```

**Cap: 3 attempts per concept.** If still broken after 3 versioned tries, write a
"stuck on X" entry in `daily_status.md` and move to the next item in the plan
(don't keep hammering on the same wall).

## Daily status format

Claude writes `docs/daily_status.md` (overwriting each day) so Vaibhav can catch up
in 2 minutes:

```
# Day YYYY-MM-DD

## Versions tried today
- vXXX: <one-line description, result>
- ...

## Best so far
v304: 1948 TPS/chip @ 33.5s, 31.6% MFU (baseline, unchanged)
vXXX: <new best if achieved>, delta vs baseline

## Stuck on
<anything where I want Vaibhav's input — be specific, with what I tried>

## Next planned step
<concrete next vXXX>
```

## Resource budget

- **Max 1 active jobset at a time** (no parallel slice attempts; cluster is shared)
- **Max ~6 hr cluster wall time per day** (stop and wait for Vaibhav if I'm churning)
- **Max ~50 GB/day GCS profile data** (clean stale profiles in `gs://max-experiments/dsv3/profiles/` if approaching)

## Branching setup (replaces carve-out list)

```
g_fix       ← Vaibhav's branch (kernel work). Claude never pushes here.
auto_perf   ← Claude's branch (broader authority). Rebase from g_fix daily.
```

**On `auto_perf` Claude can touch any file under**:
- `dsv3/mini_dsv3/` — including `init_params`, `_carry_spec`, `mla_attention`,
  `_dense_layer_body`, all MoE backend bodies, dispatch helpers
- `dsv3/k8s/` — all YAMLs
- `dsv3/docs/` — inventory, status, design notes
- `dsv3/kernels/` — except the kernel files below

**Files Claude never touches** (Vaibhav's kernel domain):
- `forward_kernel.py`
- `backward_kernel_v4.py`
- `fused_ep_moe_v4*` (any version)
- Anything under `tpu_inference/kernels/`

**Workflow**:
1. Start of session: `git checkout auto_perf && git rebase g_fix` (absorb kernel changes)
2. Commit frequently with descriptive messages (clean cherry-pick / revert surface)
3. Merge `auto_perf` → `g_fix` only when Vaibhav decides there's a leap worth merging
4. Merge conflicts on `model.py`: resolve in Vaibhav's favor by default for kernel-adjacent code

## Week 1 plan (locked-in target: 2400-2700 TPS/chip in bf16)

| Day | Workstream | Goal | Expected TPS gain |
|---|---|---|---:|
| Mon | B start: read `gather_reduce_pallas.py`, v305 profile, sketch integration | scoping doc | 0 |
| Mon-Tue | B: integrate kernel into `_expert_mlp_gmm_ag_body` (~line 1488 scatter), validate correctness, measure | working v322 | +200-400 |
| Tue-Wed | A start: profile v304 chunk boundaries, locate problematic optimization_barriers | scoping doc | 0 |
| Wed | A: try scheduling_group annotations / barrier rewrites | working v324 | +100-300 |
| Thu | Compose A+B → v325, retune tiles | composed result | +0-200 (margin) |
| Fri | Compute push: re-test gmm_v2 fused-silu on top, evaluate `attn_proj_out` offload now that scatter cheaper | final v326 | +0-200 |

**Honest ceiling without FP8: 2700 TPS/chip.** Anything above is upside.

## What Vaibhav needs to confirm/fill in

1. **`~/cl.sh` contents** (or confirm the fallback `docker system prune -a -f && truncate /tmp/*.log` is OK)
2. **Anything missing from the "discuss before touching" list?**
3. **Resource budget OK?** (1 jobset at a time, ~6hr/day cluster, ~50GB/day profile data)
4. **OK with daily_status.md format?** Or want richer/sparser?
5. **How often will you check in?** (Affects how patient I am with "stuck" pauses)

## What Vaibhav does NOT need to do

- Approve each experiment iteration
- Watch the autonomous loop
- Chase down OOMs / cluster surprises (Claude handles)
- Tell Claude to start the next experiment after one finishes (Claude self-paces)

## Self-discipline rules (informed by failures Vaibhav has caught)

These are real instances where Claude was wrong and Vaibhav's intervention saved a
wasted day. Each rule is a procedural check to catch the same pattern without him.

### Rule 1 — Hypothesis log before each experiment

**Pattern**: built a theory from one data point and ran with it without asking what
would falsify it. (v315: claimed dropping small offloads would save 5s of DUS overhead;
result was identical to baseline — the offloads' actual win was recompute, not DUS.)

**Rule**: before launching a vXXX experiment, write to `daily_status.md`:
- Hypothesis (one sentence)
- Predicted result (specific TPS/chip range)
- Falsification: what observation would prove me wrong?

If the actual result matches "wrong" prediction more than "right", STOP and re-think
the premise — don't tweak parameters and retry.

### Rule 2 — Read code before modifying flags or specs

**Pattern**: assumed code does what it should and changed flags / dispatch logic
without reading the implementation. (v311 SMEM crash: bumped `max_concurrent_async_all_gathers`
without understanding the lowering bug. v317 OOM: assumed gmm_ag had TP weight sharding
without reading line 1518 where it's clearly P("ep","fsdp",None) — no TP at all.)

**Rule**: any change to a spec / flag / shard_map dispatch requires `Read`-ing the
relevant function body in this session first. A grep + assumption is not enough.
If the function has comments saying "uses X pattern", verify by reading X's
implementation too.

### Rule 3 — Check memory + experiment_log_v2.md before "X doesn't work" claims

**Pattern**: dismissed possibilities Vaibhav knew worked historically. (Claimed
ragged_dot at TP=2 wouldn't work; Vaibhav pushed me to dig and found v264 = 1740
TPS/chip with col/row pattern — never dismissed in memory because I hadn't written it.)

**Rule**: before declaring "X doesn't work" or "X is broken", do these two greps:
- `grep -ri "X" memory/`
- `grep -A 5 "X" dsv3/docs/experiment_log_v2.md`

If history has a counter-example, surface it before dismissing. If memory doesn't
have it but I know it from this session, ADD it to memory before next iteration.

### Rule 4 — 2-attempt rule (not 3) before structural reset

**Pattern**: kept tweaking same-concept fixes after first didn't work. (v321a/b/c:
3 iterations of psum/AG fixes, all NaN at step 1, all the same initial loss 208 —
I should have stopped at attempt 2 and asked "is the spec layout itself wrong?"
instead of adjusting psum placement.)

**Rule**: cap is **2 attempts per concept** (revised down from 3). After 2 same-concept
failures with no improvement signal:
1. Stop
2. Write to `daily_status.md` what was tried and what specifically didn't change
3. Pivot to next planned item (don't sit idle)
4. Either ask Vaibhav at next check-in OR escalate by reading the actual working
   reference (v264's body, not the abstraction)

### Rule 5 — Always scale per-device → per-pod

**Pattern**: did arithmetic on per-device numbers without scaling. (v314: computed
67 GB host pinned per-device → forgot 8 devices/pod → pod OOM at 544 GB.)

**Rule**: when reasoning about HBM or host memory, **always also compute per-pod**
(× 8 devices/pod for v7x) and per-host-RAM-budget. Add to the OOM diagnosis flow:
- HBM compile-time: per-device, vs 94.75 GB
- HBM runtime program: per-device, vs ~95 GB allocatable contiguous
- Host pinned: per-pod = per-device × 8, vs node memory limit (~1 TB on bodaborg)

### Rule 6 — Verify when result surprises me

**Pattern**: when v315 matched v313 EXACTLY (1540 vs 1540), I treated it as
confirmation rather than as an anomaly worth investigating. The exact match should
have triggered "wait, my theory predicted -5s gain — why is it identical?"

**Rule**: when an experimental result matches the **null hypothesis** (no change
from baseline), that's a STRONGER signal than a noisy improvement. Stop and ask
why my predicted change didn't materialize before queuing the next experiment.

## Token-efficient operation

Strategies to make autonomous iteration cost less:

| Practice | Why |
|---|---|
| Always pipe `kubectl logs` through `grep -E "completed step\|Compilation\|loss\|OOM\|Error"` before consuming | OOM dumps are 25-35 KB because of full replica_groups lists (512 device IDs each) |
| Never `kubectl describe pod` without `\| head -50` or grep for specific keys | Pod descriptions are ~5 KB even when nothing's wrong |
| Don't re-Read the same file range I already saw this session | Track via mental notes; if uncertain, just proceed |
| Use `persisted-output` / `tool-results` saved files: grep them, don't re-fetch from cluster | Saves the second round of log fetching |
| Status updates capped at ~10 lines unless Vaibhav asks for detail | Be terse; he reads diff from the inventory anyway |
| Reference YAMLs by version, don't paste flag lists | "v304 YAML" → 5 chars vs ~3 KB of XLA flags |
| One TaskCreate per concept (not per attempt) | v321a/b/c = one task with status notes, not three |
| `profile_inventory.md` is append-only; new rows at end | Avoid rewriting existing rows even if I learn more |

## Check-in cadence (Vaibhav's schedule)

- **8 AM**: overnight summary in `daily_status.md`, propose morning plan
- **12 PM**: midday status, ≤ 1 decision point if needed
- **5 PM**: evening status, queue overnight work

Between check-ins (8→12, 12→17, 17→08): autonomous run with 2-attempt rule. Never
sit idle — always pivot to next planned item.

**Most aggressive window**: overnight (17 → 08 = 15 hrs). Plan ~3-4 experiments
queued, each with clear fallback if it fails. Use the time on slower workstreams
(profile analysis, code review of complex changes, kernel integration).
