# iter-9 Tooling: predicted post-fix headroom analysis

**Date**: 2026-05-08
**Workload**: dsv3_train_full (DSv3-671B, v7x 4×8×8, gmm_v2-enabled iter-2b baseline)
**Class**: Tooling (no jax-gpt source change, no cluster run, perfsim cross-check)
**Inputs**: iter-2b xplane, iter-6 `/checkpoint/` bisection, iter-7+8 NaN findings (jax-gpt#2, #3)

## Why this iter

Three single-iter Greedy/Lateral attempts (iter-5/7/8) reverted on `dsv3_train_full`, triggering AGENT.md §13 cumulative `regression_chain` halt. The chain is signaling that single-iter levers on iter-2b are exhausted. Before pursuing multi-iter scope changes (HALT.md options 2-4) or additional gambling, this iter quantifies what fixing the **identified upstream blockers** (jax-gpt#2, #3) would unlock — without burning a cluster slot.

Concrete question: **once the offload-restore pipeline is fixed, what's the realistic headroom available from the existing checkpoint markers?**

## Method

1. Take the recompute-component breakdown from iter-6's `/checkpoint/` bisection (research/dsv3/iter6_checkpoint_bisection.md):

| recompute component | ms/step (16-pass on iter-2b) |
|---|---|
| attn fwd recompute (q_proj fusions) | 1,779 |
| attn fwd recompute (splash_mha_fwd_residuals) | 1,644 |
| attn fwd recompute (convert_reduce_fusion) | 545 |
| attn fwd recompute (slice_negate_fusion / rope) | 248 |
| MoE gmm_v2 fwd recompute | 1,820 |
| MoE all-gather (chunk0+1+weight) recompute | 1,007 |
| MoE all-reduce (gate_logits) recompute | 137 |
| **Total** | **7,180** |

2. Estimate two scenarios:
   - **Pessimistic**: 75% of recompute eliminated, full DUS overhead realized (2 names × 30 ms × 58 layers = 3,480 ms)
   - **Optimistic**: 90% of recompute eliminated, only marginal DUS overhead beyond what `moe_layer_input` already pays (~1,044 ms)

3. Cross-check against `perfsim.simulator.run(hw, model, train, par)` with `remat_policy ∈ {full, attn_only, none}`.

## Results

### Heuristic estimate range

| | recompute eliminated | DUS overhead | net savings | predicted step | predicted TPS/chip |
|---|---|---|---|---|---|
| **Pessimistic (75% / full DUS)** | 5,385 ms | 3,480 ms | **+1,905 ms** | 32,745 ms | 1,991 (+5.8%) |
| **Optimistic (90% / marginal DUS)** | 6,462 ms | 1,044 ms | **+5,418 ms** | 29,232 ms | 2,231 (+18.5%) |

### Perfsim cross-check (canonical)

`perfsim.simulator.run` with workload-yaml-matching parallelism (tp=1, ep=4, fsdp=128, batch_sharded_by_ep=True, ep_comm=ag) and SGD optimizer:

| remat | predicted step (ms) | Δ vs full | predicted TPS/chip Δ |
|---|---|---|---|
| `full` (production) | 34,525 | baseline | — |
| **`attn_only`** | **33,049** | **−1,476 ms** | **+4.5%** |
| `none` | 33,049 | (same) | — |

**Interpretation**:
- Perfsim's `remat=attn_only` model lands closer to the **pessimistic** end of my heuristic range (1,476 vs 1,905 estimate). Perfsim is more conservative because it correctly accounts for the fact that attention recompute is partially-overlapped with MoE forward already in the production scheduling.
- Perfsim shows **`attn_only` and `none` are identical** — meaning perfsim's model doesn't differentiate between "save attention residuals to host" and "save everything to HBM." This is because perfsim doesn't model the DUS overhead at all; it only models compute time of the saved-vs-recomputed paths.

### Triangulating the realistic upside

Combining both:
- Heuristic upper bound (optimistic, no model overhead): +5.4 sec/step (+18.5%)
- Perfsim model (compute-only, no DUS): **+1.5 sec/step (+4.5%)**
- Heuristic with full DUS overhead: +1.9 sec/step (+5.8%)
- **Most realistic estimate: +1.5 to +2.0 sec/step, ~+4.5 to +6.0% TPS**

Note that perfsim's prediction of 34,525 ms is **0.4% off the measured 34,650 ms** — well within the validation corpus tolerance, lending confidence to the relative attn_only delta.

## Implications

1. **The fix-#2-#3-unlocks-headroom story is real but bounded**. Expect ~+5% TPS, not +15-20% as iter-6's framing implied. Iter-6 mistakenly conflated "recompute time" with "savable time" — much of the recompute is partially-overlapped today.

2. **The cumulative `regression_chain` halt was correct.** Even if iter-7's `attn_proj_out` offload had worked numerically, the upside was ~+5% TPS, not the ~+6% iter-6 estimated. iter-7's NaN AND iter-5's regression both consumed cluster slots for changes whose ceiling was already small.

3. **iter-2b at 1882 TPS/chip is much closer to the local optimum than the bisection numbers suggested.** The 7,180 ms "recompute" bucket is real spend but only ~20% of it is recoverable. Honest estimate of remaining iter-2b headroom: **+5–8% TPS via offload-fixes once #2/#3 land** plus whatever multi-iter scope projects yield.

4. **The realistic next-iter ROI on this baseline is limited.** Multi-iter scope changes are required for >5% gains:
   - **Attention-only-checkpoint refactor** (HALT.md option 2): could land ~+4.5% if HBM fits, but is multi-iter risk
   - **Custom Pallas tgmm with `vmem_limit_bytes`** (HALT.md option 3): re-opens iter-5 candidate-C with proper VMEM control; potentially +1-2% if microbench-validated
   - **Sharding plan reconsideration**: untouched in this session; perfsim search doesn't yet help (perfsim#44)

## Cross-references

- iter-6 bisection: `research/dsv3/iter4_moe_gmm_ag_bisection.md`, `research/dsv3/iter6_checkpoint_bisection.md`
- iter-7 NaN: `autoperf/iter_log.md` § iter 7, jax-gpt#2
- iter-8 NaN: `autoperf/iter_log.md` § iter 8, jax-gpt#3
- v7x_KNOWLEDGE.md §3 entries: attn_proj_out broken, prevent_cse=True broken
- HALT.md cumulative session state

## Post-PR#45 update — sharding-plan sweep via direct simulator.run

After perfsim PR#45 was rebased + reviewed, ran `perfsim.simulator.run`
directly to scan candidate sharding plans (bypassing perfsim#44 search-
bug; root cause: `search.py:147` uses `hw.cell_size × d2d_size = 140,000`
instead of `hw.slice_size × d2d_size = 512`).

Top sharding candidates by perfsim-predicted step time:

| plan (tp, ep, fsdp, remat) | step ms (predicted) | TPS/chip ratio | Δ vs production |
|---|---|---|---|
| (1, 1, 512, full) — ep=1 | 32,503 | ~1992 | **+5.9% ⚠ HBM dubious** |
| (1, 4, 128, **attn_only**) — post-fix#2/#3 | 33,049 | ~1963 | **+4.3%** |
| (1, 2, 256, full) — sharding-only | 33,486 | ~1938 | **+3.0%** |
| (1, 4, 128, full) — production | 34,525 | 1882 measured | 0% baseline |
| (2, 4, 64, full) — tp=2 d2d | 39,572 | ~1639 | −14.6% |
| ep=8/16/32 various | 42K–213K | catastrophic | huge regressions |

### Two orthogonal levers above production

1. **Remat policy** (post-jax-gpt#2/#3 fix): `attn_only` saves +4.3%
2. **Sharding** (with HBM caveats): ep=2 saves +3.0%

These are structurally orthogonal; combined ceiling is ~**+8% TPS**
(estimated, untested). HBM caveat: perfsim's HBM estimate is
under-counted by ~80 GB on production sharding (perfsim says 16 GB,
actual production peak is 96 GB) because perfsim doesn't model XLA
program binary or HLO temps. Treat ep=2/ep=1 plans as "may OOM at
compile-time" until empirically validated.

### Final answer: how far from best possible

iter-2b at 1882 TPS/chip is **within ~8% of the perfsim-predicted
ceiling** for this hardware/model/toolchain combination. The cheap
single-iter Greedy/Lateral headroom is genuinely close to gone. Remaining
gains require either (a) upstream offload-pipeline fix (jax-gpt#2/#3),
(b) HBM-aware sharding redesign with empirical compile-time validation,
or (c) multi-iter design-space projects (custom Pallas tgmm,
attention-only-checkpoint refactor).

The cumulative `regression_chain` halt was a correct signal that the
local optimum is reached. Future autoperf sessions on this baseline
should **wait for upstream** rather than chase additional single-iter
gambles.

## Recommendation for the maintainer agent reviewing jax-gpt#2 + #3

This analysis quantifies exactly what's at stake: **fixing #2 + #3 (or just #2 if they share root cause) unlocks ~+5% TPS** on the v304-class production training baseline. That's a real win, comparable in magnitude to iter-2's gmm_v2 swap (+6.6%). Worth prioritizing.

The fix should also unblock the `q_a` / `kv_a` / `shared_hidden` markers (model.py:568, 580-area, 2676) that the v315 author tested and rejected for "small DUS:save ratio" — the rejection rationale assumed these markers worked correctly, which iter-7/8 now suggests was incorrect. If the offload-restore pipeline gets its async-DMA / layout drift fixed, those markers may also become positive-ROI and add another 1-3% TPS on top of attn_proj_out.
