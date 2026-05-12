# Lever Queue — prioritized backlog for autoperf

Maintained across sessions. Synthesis-every-3-iters subagent (per
AGENT.md §8b Primitive A) prepends a `## Hypothesis` block at the top
of this file with a cross-iter pattern + suggested experiment.

The next iter's lever-pick draws from this queue FIRST (highest
expected information value), then falls back to heuristic table.

When a queued lever lands or is rejected, mark it `[done iter-N]` or
`[rejected iter-N: reason]` instead of deleting — history matters.

---

## Queue (priority order, 2026-05-12)

### multi-iter-#3 — chunk-pipelining/overlap fix

**Source**: iter-13 diagnosis (body-tail exposure under TC serialization);
iter-14 ruled out collective fusion via ragged_a2a (no SC offload on v7x).
Remaining viable variants:

1. **Cross-iter prefetch (moe_scan body refactor)** — hoist next-layer
   attention QKV before `jnp.concatenate` at model.py:1871, so chunk-1's
   RS has next-layer compute to overlap behind. Estimated +0.5-1% TPS.
   Multi-iter scope (touches moe_scan body structure). Subagent design
   needed before patch.
2. **Explicit barrier reorder** — try inserting an `optimization_barrier`
   between chunk1's scatter-add and its psum_scatter, to force XLA to
   schedule chunk0's RS earlier in the body. Single-iter scope. Lower
   expected impact (~0.3%) but cheap probe.
3. **Staggered-call wrapper** — replicate MaxText's staggered_call
   pattern at the moe_scan level so chunks run as producer-consumer
   pipeline. Multi-iter, higher risk.

### Single-iter Greedy candidates

- (after iter-17 ratchet) Promote iter-16 to BASELINE, ratchet corpus
- **iter-18 candidate** (drafted via Primitive B during iter-17 wait):
  add q_a + kv_a + shared_hidden to `names_which_can_be_saved` in
  `_ckpt_policy`. Sizes: 5.6 + 2.1 + 7.4 = 15.1 GB additional HBM
  across L_moe=58 layers. Combines with iter-16's attn_proj_out (26 GB)
  for a total ~41 GB SAVE-list footprint. Worst case: compile OOM
  (fast-fail ~10 min), pivot to smallest first. Best case: +0.5-1.5%
  additional TPS stacking on iter-16. Patch is a 1-line edit to
  `model.py:3053` extending the saved tuple from `("attn_proj_out",)`
  to `("attn_proj_out", "q_a", "kv_a", "shared_hidden")`. The
  rejection logic at model.py:3047-3051 only applies to OFFLOAD
  (per-DUS overhead); SAVE has no DUS so the comment doesn't bind.

### Tooling candidates

- Add cde.yaml to .dockerignore (build-hash leak fix; iter-16 retrospective)
- Investigate Bug 3 (n_chunks=4 NaN with line-1859 barrier) — multi-iter
  but only if chunk-pipelining lever class proves viable; otherwise
  this unlocks n_chunks=4 = smaller per-chunk RS body-tail (~+1% TPS)
- Update perfsim training-regime remat model to know about
  attn_proj_out SAVE (predicts attn_only at +4.5% but actual is +1.8%;
  perfsim should model save-only-attn-out as a distinct policy)

### Rejected / blocked

- [rejected iter-14] Collective fusion via ragged_a2a (RaggedAllToAllEmitter
  not fusion-aware on v7x; 32× scoped-memory delta vs RS-on-SC)
- [rejected iter-5] tgmm tile_m=4096 (memory-bound at production shapes)
- [blocked Bug 3] n_chunks=4 (NaN cotangents; failed multiple fixes:
  3756666 barrier, fb17291 v341n serializing, ff41a58 scan conversion)
- [blocked jax-gpt#2] OFFLOAD attn_proj_out (NaN; broken offload-restore
  pipe). Will unblock when #2 lands.
- [blocked jax-gpt#3] prevent_cse=True (NaN; same family as #2)

---

## Synthesis log

Synthesis-every-3-iters subagent (Primitive A) writes its observations
here. Empty until first synthesis fires (after iter-19 or after a
halt-with-revert).

### iter-17b finding (2026-05-12, partial)

iter-17b evicted by preemption (mk-q30b-0508 higher-priority workload).
4 valid steps captured before kill:
- step 2: 33.528 s (1955 TPS/chip @ 31.6% MFU)
- step 3: 33.733 s (1943 TPS/chip @ 31.4% MFU)
- step 4: 33.571 s (1952 TPS/chip @ 31.6% MFU)
- avg: 33,611 ms / **1949 TPS/chip / 31.5% MFU**

vs iter-16 (34,203 ms / 1916 TPS/chip): **+1.7% faster** — outside ±0.3%
ratchet noise band. Two reads:
1. iter-16 was on slow end of variance; iter-17b is "true" gain
2. Per-iter variance is ~1-2%, ratchet criterion needs revisiting

Direction confirmed regardless: SAVE attn_proj_out gives **+3.6% vs iter-2b**
(was +1.8% at iter-16). Conservative move: do NOT ratchet corpus baseline yet
(eviction = partial data). Wait for 1 more clean measurement before promoting.

**Note for synthesis** (Primitive A trigger after iter-19): variance of ~1.7%
across iter-16 vs iter-17b is the new noise band estimate — tighter than the
±0.3% used in AUTONOMOUS_RUN.md. Updating the ratchet criterion to ±1% may
be appropriate.
