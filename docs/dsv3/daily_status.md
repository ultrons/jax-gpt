# Daily status — 2026-04-27

## Active workstream: B (scatter Pallas integration)

## Hypothesis log (Rule 1)

**Hypothesis**: replacing the `at[].add(...)` scatter at `model.py:_expert_mlp_gmm_ag_body`
(line ~1488) with the production `gather_reduce_pallas` kernel will save 3-5s/step.
Current scatter is 19% of step at 6.5s/chip, HBM-bound at 1 FLOP/B.

**Predicted**: TPS/chip 2150-2330 (vs v304 baseline 1948).

**Falsification**: if TPS/chip stays ≤ 1980 (within run-to-run noise of v304), then either:
- (a) the kernel itself isn't faster than `at[].add`
- (b) the scatter wasn't on the critical path
- (c) per-call overhead dominates (like v305 hit at 230 calls/step)

If any of (a)-(c), STOP, profile, write findings, pivot to Workstream A (chunk pipeline).

## Versions tried today
- v322/v324 BLOCKED on old cluster (bodaborg-super-alpha) for ~20 hr — metadata.google.internal DNS failures. Vaibhav switched cluster to **bodaborg-super-rbq** which works.
- **v324b** (combined) hit `pl.Indirect` API gap — production gather_reduce_pallas kernel needs newer JAX than 0.10.0. Blocked on JAX upgrade.
- **v325** (chunk barriers ALL removed, no scatter Pallas): **2034 TPS/chip @ 32.2s, 32.9% MFU (+4.5%) BUT NaN loss**.
- **v326** (only post-AG barrier kept, 3/4 removed): **1940 TPS/chip @ 33.8s, 31.4% MFU = v304 baseline, valid loss**.
- **Bisection conclusion**: post-AG barrier alone provides the +4.5% perf gain when removed AND is correctness-critical. Can't remove it without NaN.
- **Built JAX nightly Apr 24 image** (`gcr.io/tpu-vm-gke-testing/mini-dsv3:nightly-apr24`) with `jax==0.10.1.dev20260424` + `libtpu-0.0.41.dev20260424` — has both `pl.Indirect` and `needs_layout_passes` APIs.
- **v304-nightly** (sanity, no scatter Pallas): **1936 TPS/chip @ 33.8s, MFU 31.4%** — baseline confirmed, JAX nightly is safe.
- **v322-nightly-scpallas** attempt 1: hit `TypeError: kernel() got an unexpected keyword argument 'out_shape'`. Apr 24 nightly renamed `pl.kernel`'s kwarg to `out_type`. Patched.
- **v322-nightly-scpallas** attempt 2 (commit cd6e9a2 — out_type + needs_layout_passes): hit `ValueError: function wrapper at .../helpers.py:201 traced for jit returned a mutable array reference of type Ref{bfloat16[65536,7168]}`. The new `_make_kernel` wrapper expects body to mutate out_refs in-place + return None; somewhere along the way an input ref is leaking through the JIT trace. Need to either (a) refactor `sc_gather_reduce` to bypass `pl.kernel` and use `pl.pallas_call` / `core_map` directly, or (b) try a more recent nightly where this might be fixed.

## Workstream B fully blocked (2026-04-28 evening)

After exhausting Apr 16/17/21/24 nightlies and 3 wrapper-bypass rewrites:

| Attempt | Result |
|---|---|
| Apr 24 nightly + kernel as-is | `out_shape=` kwarg renamed to `out_type=` |
| Apr 24 + `out_type=` patch | `Ref{bfloat16[65536,7168]}` leak from `pl.kernel` wrapper line 201 |
| Apr 21 nightly (original `out_shape=` API) | Same leak from line 201 |
| Apr 17 nightly | Same leak from line 193 (different line, identical mechanic) |
| Apr 17 + bypass `pl.kernel`, custom core_map+freeze | Leak from MY line 173 (`freeze(out_ref)`) |
| Apr 17 + drop @jit, jnp.zeros init, lax.copy on freeze | Leak from custom_vjp boundary line 1425 (`_sc_combine_with_vjp`) |

Each fix moves the leak one frame up the stack. The pattern of "create Ref → core_map mutates it → freeze and return" is fundamentally rejected by JAX 0.10.x JIT/custom_vjp tracing — refs can't escape any traced function boundary, and successive workarounds (freeze, lax.copy) don't actually break the ref-tracking attribute.

**Next-day path** (need Vaibhav input):
1. Was the kernel authored against an internal/unreleased JAX build with different ref semantics? If so, get that build pinned.
2. OR rewrite `sc_gather_reduce` to a pure-functional `pl.pallas_call` (no internal refs; takes inputs and returns outputs). Sidesteps refs entirely. Cost: ~half-day port.
3. OR drop Workstream B and focus on Workstream A (scheduling_group annotations) or FP8 weights.

Images on GCR: `nightly-apr24`, `nightly-apr21`, `nightly-apr17`, `nightly-apr17-v2`, `nightly-apr17-v3` — all ready, all blocked at the ref-leak.

## Old open question (superseded by above)
The new `pl.kernel` wrapper does:
```python
@api.jit
def wrapper(*operands):
    arg_refs = tree_util.tree_map(jax_core.new_ref, operands)
    out_refs = tree_util.tree_map(_get_empty_ref, out_type)
    @pl_core.core_map(mesh, scratch_shapes=scratch_types, **mesh_kwargs, name=...)
    def _(*scratch_refs, **scratch_kwrefs):
      return body(*arg_refs, *out_refs, *scratch_refs, **scratch_kwrefs)
    outs = tree_util.tree_map(lambda ref: ref[...], out_refs)
    return outs[0] if unwrap_out else outs
```
Our body's last statement is `pltpu.emit_pipeline(...)(idx_hbm_ref, ...)` which is an
expression-statement (Python returns None). Yet the wrapper trace claims a Ref of
input shape (65536, 7168) is being returned. Suspect: the inner emit_pipeline call
may be returning a Ref tree that JAX tracing treats as a leaked output. Try adding
explicit `return None` at end of body? Or wrap in `del _ = pltpu.emit_pipeline(...)`?

## Cluster state (as of 04:41 UTC)
- Persistent `metadata.google.internal` DNS failures during JAX distributed init across multiple nodes.
- 4× retry pattern: pod 60→60→44→44→44.
- Bad node identified earlier: `gke-tpu-804094a1-ncj2` — but other pods (44) also failing now, so not isolated to one node.
- Other recent runs (v300 / v301 / v321 sessions) had similar but isolated occurrences.

## Best so far
v304: 1948 TPS/chip @ 33.5s, 31.6% MFU (baseline, unchanged)

**Locked-in measurement (v325)**: chunk pipeline overlap gain ceiling at +4.5%
(2034 TPS/chip) — currently unreachable without NaN. To unlock, need a way
to give XLA "may interleave AG output reads" hint without removing the
correctness-essential barrier. Candidates:
- `scheduling_group_id` annotations
- `with_compute_on('sc'/'tc')` directives
- New XLA flag enabling safe reorder around opt_barriers

## Integration sketch (v322)

**Wiring already exists**: `cfg.moe_use_sc_scatter` (config) → `_moe_gmm_ag` →
body's `use_sc_scatter` flag → calls `_sc_combine_with_vjp` → `sc_gather_reduce`.

**v305's loss**: 1840 TPS/chip @ 35.6s (vs v304's 1948), 5% regression.
Cause: 230 calls/step (2 chunks × 58 layers × 2 fwd/bwd) of the older MLIR-direct kernel.

**v322 hypothesis**: production Pallas kernel (`gather_reduce_pallas.py`) has lower
per-call overhead via cleaner pipeline/subcore mesh design. Same signature, drop-in.

**Changes**:
1. Copy `dsv3/gather_reduce_pallas.py` → `dsv3/mini_dsv3/kernels/gather_reduce_pallas.py`
   (so Dockerfile auto-includes it via `COPY mini_dsv3/kernels /app/kernels`)
2. Replace import in `_sc_combine_fwd` (model.py:1442) from `gather_reduce_sc`
   to `gather_reduce_pallas`. Signature is identical.
3. New YAML v322 = v304 + `--moe_use_sc_scatter` enabled.

**Predicted**: if production kernel cuts per-call overhead by ≥50%, the 5%
regression v305 saw becomes a 0-2% regression OR a small gain (depends on
scatter vs HBM scatter-add tradeoff). Even break-even validates the path
before chunk pipeline work (Workstream A).

If gain ≥3% TPS/chip: keep, move to A. If flat: profile to see if scatter
is actually the bottleneck or not.

## Stuck on
- Bad-node metadata DNS issue blocking cluster runs. Not a code problem. Need
  Vaibhav to either cordon the node or wait for it to be rotated out.
  Affected node: `gke-tpu-804094a1-ncj2`.

## Workstream A finding (v304 profile analysis)

**Key data** (from xla_shell on v304 profile, 32-host slice):

| Op | Time / host (cumulative) | Per chip per step | % of step |
|---|---:|---:|---:|
| splash_mha_dkv_no_residuals | 38.4s | 2.4s | 7.2% |
| splash_mha_fwd_residuals (×2 = chunks 0+1) | 49.0s | 3.1s | 9.2% |
| **scatter_custom_fusion (×4) + offload_async_done** | **52s + 19.4s = 71.4s** | **~4.5s** | **13.5%** |
| ragged-dot-none (16 ops × ~6s each) | 96s | 6s | 18% |

**Exposed collective stall: 31.6s/host = 1.97s/chip = 5.9% of step**

Top exposed: 4× `ep_token_gather/all_gather` totaling **20.2s/host = 1.26s/chip**.
These are inside `_expert_mlp_gmm_ag_body`'s `_process_chunk` (model.py:1413-1418).

**Workstream A target**: each chunk has barriers around its AG (model.py:1410, 1417)
that prevent XLA from interleaving chunk0 AG with chunk1 compute. The 1.26s/chip
exposed AG IS the chunk pipeline failure to overlap. Removing/replacing barriers
should let XLA pipeline → -1 to -2s/chip.

**Workstream B target**: scatter Pallas SC kernel still validates — scatter is 4.5s/chip
HBM-bound. Replacement could save 3-5s.

**Combined ceiling**: -4 to -7s → 27-29s step → **2250-2480 TPS/chip**.

## Next planned step

When cluster recovers (Vaibhav check-in or cluster ops):
1. **First attempt**: v324 (combined scatter Pallas + barrier removal). If both
   workstreams compose cleanly → ~2200-2600 TPS/chip.
2. **If v324 NaN/breaks**: bisect — apply v322-scpallas alone (isolates scatter),
   then a hypothetical v323-only image (barriers off, original scatter).
3. **If cluster stays broken**: ping ops or wait until tomorrow.

## Images ready (pushed to GCR)
- `gcr.io/tpu-vm-gke-testing/mini-dsv3:v322-scpallas` — v304 + scatter Pallas only
- `gcr.io/tpu-vm-gke-testing/mini-dsv3:v324-scpallas-nobar` — v304 + scatter Pallas + barriers off

## Code committed
- v322 (scatter Pallas): commit `d7f109e` on auto_perf
- v323 (barrier removal): commit `9986ee3` on auto_perf
- v324 YAML: commit `e6fffac` on auto_perf
All pushed to remote `auto_perf` branch.
