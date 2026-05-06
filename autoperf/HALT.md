# HALT — autoperf agent (dsv3_train_purefsdp track) — TRACK EXHAUSTED

**Workload**: `autoperf/workloads/dsv3_train_purefsdp.yaml`
**Last iteration**: 0 attempt 3 (failed in Python init; no compile, no cluster admission overhead)
**Branch**: `autoperf/dsv3_train_purefsdp`
**Halted at**: 2026-05-06
**Reason**: `track_exhausted` — three structurally distinct failure modes
across three iter-0 attempts have left the no-EP design space empty
within the current codebase.

## Three attempts, three different walls

| attempt | geometry | failure |
|---|---|---|
| 1 | `fsdp=512 ep=1 tp=1` | libtpu SC-offload CHECK fail at fsdp=512 mesh (codegen bug) |
| 2 | `fsdp=256 ep=1 tp=2` | compile-time HBM OOM, +32 GB over conservative limit |
| 3 | `fsdp=128 ep=1 tp=4` | jax-gpt source: TP must equal cores axis (nc=2); TP > 2 unimplemented |

Each failure is independent and structural — different layer of the
stack, different root cause. Together they exhaust the parallelism
options open to the autoperf-side loop:

- TP ∈ {1, 2} is a hard code-side constraint. TP > 2 needs a real
  jax-gpt source change (multi-axis TP placement) that's outside per-iter
  scope.
- TP=1 + ep=1 forces fsdp=512, which trips the libtpu codegen path.
- TP=2 + ep=1 forces fsdp=256, which doesn't fit in the 94.75 GB
  compile-time HBM budget at gbs=4096.

The structural argument I made when you proposed pure FSDP is now
empirically validated three different ways. **EP=4 in the v304
baseline isn't a chosen-out-of-many — it's HBM-and-libtpu-forced**
on this hardware/batch combo.

## Recommended next human action

The autoperf agent's strong recommendation: **resume `dsv3_train_full`**.
The v304 ep=4 baseline is the only viable workload on v7x_4x8x8 at
gbs=4096; the headroom report against it is what should drive iter-2.

Two options inside that track:

1. **Fix the moe_xlayer_prefetch bwd-transpose bug** — unlocks the
   +264 ms FSDP_AG lever (top headroom by a wide margin). Likely a
   small jax-gpt source change in `model.py:3061-3094`: either an
   explicit `jax.lax.with_sharding_constraint` on the carry's `ws_ag`
   to keep its PartitionSpec stable, or restructure the prefetch so
   the AG result is built inside the body's `shard_map` rather than
   threaded through scan carry. One change, single commit, comparable
   in shape to a normal autoperf iter.

2. **Skip the bug, take the next-best lever** — `Router` (+88 ms) or
   `Norms` (+68 ms). Smaller per-iter win but no source diagnostic.
   Could run iter-2 in under an hour.

The autoperf agent's natural choice if asked to keep moving without
intervention: **(1)** — the headroom math says fixing
moe_xlayer_prefetch buys 3× what (2) does, and the fix is well-scoped.

## Other defensible options

- **Open a third workload track** at smaller gbs (e.g.
  `dsv3_train_purefsdp_gbs2048.yaml`) — would let the no-EP comparison
  succeed empirically, but loses apples-to-apples vs v304's gbs=4096.
  The two-track comparison gets messy: different optimizer state,
  different HBM regime, different convergence implications.
- **File a libtpu issue for the SC-offload CHECK** — out of our 4-agent
  system; would need a Google-side channel. Doesn't unblock anything
  short-term.
- **Implement multi-axis TP placement in jax-gpt** to enable TP > 2 —
  much larger source change than fixing moe_xlayer_prefetch; likely a
  few days' work for a domain expert. Doesn't feel proportionate to
  the headroom-driven loop's iter cadence.

## Track summary on `autoperf/dsv3_train_purefsdp`

Branch is preserved for audit. iter_log.md has full diagnostics for all
three attempts. v7x_KNOWLEDGE.md gained three cross-workload entries
that future autoperf agents on either track will benefit from:
- "fsdp=512 + SC-offload flags → libtpu CHECK fail"
- "ep=1 at gbs=4096 doesn't fit compile-time HBM"
- "TP is hardcoded to align with v7x cores axis: TP ∈ {1, 2} only"

## Cluster cost across this track

- attempt 1: ~5 min (admission + libtpu CHECK SIGABRT)
- attempt 2: ~5 min (admission + JAX CompileTimeHbmOom)
- attempt 3: trivial — Python init error, ~30s pod startup
- **Zero training compute consumed across the entire no-EP track.**

Per AGENT.md §1, halt-when-uncertain has paid off — three potential
multi-hour debugging tangents avoided, all useful negative evidence
captured in iter_log + v7x_KNOWLEDGE for future sessions.

## Open follow-ups

- `ultrons/perfsim#1` — resolved.
- `ultrons/perfsim#3` — open, non-blocking, gemm_eff calibration.
  Will become relevant when dsv3_train_full's iter-N starts ranking
  compute leaves.

## Cluster state

- All `dsv3train-*` runs reaped (`failed`).
- `cde history --status running`: zero runs in this project.
- Parallelism budget: free.
