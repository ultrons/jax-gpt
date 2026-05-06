# HALT — autoperf agent

**Workload**: `autoperf/workloads/dsv3_train_full.yaml`
**Last iteration**: 1 (compile-failed; reverted)
**Branch**: `autoperf/dsv3_train_full`
**Halted at**: 2026-05-06
**Reason**: `broke_training` per AGENT.md §13. iter-1 lever
(`moe_xlayer_prefetch=True`) failed at compile with
```
ValueError: all_gather_reduced only accepts inputs that are varying.
Got bf16[64,16,7168]
```
in the bwd transpose of `_ag_one_moe_layer` via `_moe_scan_fn_pf`
(`model.py:3079`). Pure Python tracing exception — no NaN, no OOM, no
hardware failure. Cluster cost: ~7 min (admission + compile attempt + 64-pod
failure propagation); zero training compute.

Full diagnostic in `autoperf/iter_log.md` iter-1 entry. Anti-hallucination
note appended to `v7x_KNOWLEDGE.md` §5 ("`--moe_xlayer_prefetch` broken at
production scale") so future agents don't repeat the experiment until the
underlying jax-gpt bug is fixed.

## Recommended next human action

This is a fork-in-the-road decision; the autoperf agent halts because §13
calls for it, but the next iteration is well-scoped:

1. **Continue the loop on the next-best lever (cheap, immediate)**:
   iter-2's top-headroom leaf is `Router` (+88 ms/step exposed on
   v304 baseline). Per `auto-perf-guide.md` training-subsection lever for
   `Router`: "gate-logits dot_general (`bsd,de->bse`) tile choice; avoid
   scatter on the TC path." Concretely: re-tile the Router GEMM, or
   verify it isn't dispatching to the SC scatter path for top-k.
   Implementation lives at `model.py:moe_routing/gate_logits/`. ~1 cluster
   cycle to test.

2. **Fix the jax-gpt `moe_xlayer_prefetch` bug (correct, scoped)**:
   The bwd transpose's sharding constraint mismatch is in jax-gpt source,
   not autoperf scope. A small jax-gpt-side change (e.g., adding an
   explicit shard_map wrapper around `_ag_one_moe_layer` in the prefetch
   path, or pinning the carry's PartitionSpec) likely fixes it. Once
   fixed, `FSDP_AG` (top headroom +264 ms/step) becomes addressable
   again. This is the largest single lever on this workload but is gated
   on the bug.

3. **Both, in order**: option 2 first (unlocks the +264 ms lever), then
   resume autoperf at iter-2 with `moe_xlayer_prefetch=True` retried.
   Skip option 1.

The autoperf agent's natural choice (if instructed to keep going without
human intervention) would be **option 1** — keep the loop moving on the
next-best lever. But if the user prefers to chase the larger headroom,
option 2 is the bigger win.

## Open follow-ups (parallel)

- `ultrons/perfsim#3` (filed 2026-05-06, OPEN, non-blocking): fit
  `gemm_eff` for `dsv3_671b` training-regime corpus entry. Without it,
  compute leaves stay masked at headroom=0; iter-2's lever pick on
  `Router` doesn't depend on this, but iter-3+ may.

## Cluster state at halt

- `dsv3train-i1` JobSet: `Finished=True (Failed)`, all 64 pods exited.
- `cde history --status running`: only `v341b-nchunks4-barrier`
  (unrelated, pre-autoperf bisection job). Parallelism budget free for
  next launch.
- Logs preserved at `/tmp/dsv3train-i1.log`.

## What was reverted

- `autoperf/workloads/dsv3_train_full.yaml`: removed
  `moe_xlayer_prefetch: true` and accompanying comments from
  `cde_overrides` block. Single line + 6 comment lines. Restored to
  pre-iter-1 baseline.

## What was kept (audit trail)

- iter-1 commit (`449a8e2`) stays in branch history — reverted commits
  stay in history per AGENT.md §4 push rules.
- `autoperf/iter_log.md` iter-1 section retains full diagnostic.
- `v7x_KNOWLEDGE.md` §5 has the new "broken at production scale" entry
  so the experiment isn't repeated.
- `autoperf/BLOCKED.md`: perfsim#1 = `resolved`, perfsim#3 = `open`.
- `autoperf/reports/dsv3_train_full_iter0.json` = baseline headroom
  report; reusable by iter-2.
