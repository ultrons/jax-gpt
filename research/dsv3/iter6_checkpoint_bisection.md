# iter-6 Tooling: deeper bisection of /checkpoint/ ops on iter-2b xplane

**Date**: 2026-05-08
**Workload**: dsv3_train_full (DSv3-671B, v7x 4×8×8, gmm_v2-enabled iter-2b baseline)
**Class**: Tooling (no jax-gpt source change, no cluster run)
**Input xplane**: `autoperf/profiles/dsv3train-i2b/plugins/profile/2026_05_07_02_55_30/`

## Why this iter

iter-4's bisection lumped the bwd bucket as `transpose 7,913 + jvp 3,306 ms = 11,219 ms`. iter-5 / iter-6 candidate D analysis showed that lump hides multiple distinct cost classes (true bwd vs recompute vs setup). Without finer decomposition, single-iter Greedy levers keep firing too low — iter-5 picked tile-tuning that matched a forward bucket, missing the recompute-vs-bwd distinction. This iter sizes each /checkpoint/ family by `tf_op_name` to surface real levers.

## Method

`xla_shell list_fusions --json --top 200` on iter-2b xplane, family-grouped by stripping trailing `.NN` from fusion names. Categorized each family by sample `tf_op_name`:
- `transpose(jvp(...))/.../moe_experts/...` → MoE bwd or fwd-recompute (depends on inner path)
- `/checkpoint/rematted_computation/mla_attention/...` → attention forward recompute (explicit JAX recompute)
- `splash_mha_dkv_no_residuals` → splash attention bwd (true bwd)
- `splash_mha_fwd_residuals` → splash attention forward recompute (explicit JAX recompute)
- `tgmm.NN` / `gmm.NN` → MoE megablox bwd (true bwd, identified iter-4)
- `gmm_v2-...` → MoE forward recompute (gmm_v2 fwd shape)

## Results

**Total /checkpoint/ bucket: 20,599 ms/step** (vs iter-4's transpose+jvp lump of 11,219 ms — the rest is non-MoE-only ops; iter-4 filtered to `moe_experts/moe_gmm_ag` source path only).

### Decomposition by category

| category | ms/step | % of /chkpt/ | levers? |
|---|---|---|---|
| **True bwd (cannot reduce)** | 7,448 | 36% | No — already at-tile-ceiling per iter-5 / iter-2 |
| **Attention forward recompute** | **4,216** | **20%** | **Yes — `attn_proj_out` already marked, not in policy** |
| **MoE forward recompute (gmm_v2 + dispatch AG)** | ~3,000 | 15% | Limited — fused-silu kernel internals not Python-exposed (iter-6 analysis) |
| **MoE bwd-only ops (scatter bwd, gather, multiply_reduce)** | ~3,300 | 16% | Bwd-side; tile-tuning blocked at megablox (iter-5) |
| **Other (rope, fusions, small ops)** | ~2,635 | 13% | Mostly small individually |

### True bwd (7,448 ms — fixed cost)

| family | ms/step | what |
|---|---|---|
| `tgmm.12-17` | 3,028 | d_rhs MoE bwd (megablox.tgmm); generic tile is provably-optimal at 64 MB scoped VMEM (iter-5) |
| `splash_mha_dkv_no_residuals` | 2,619 | attention dkv bwd; splash kernel internal |
| `gmm.12-17` | 1,801 | d_lhs MoE bwd (megablox.gmm); already tokamax-tuned |

### Attention forward recompute (4,216 ms — **iter-7 lever target**)

All under `/checkpoint/rematted_computation/mla_attention/`:

| family | ms/step | what |
|---|---|---|
| `fusion` (q_proj/k_proj/v_proj recompute) | 1,779 | mla projection forward recompute (30 distinct fusions) |
| `splash_mha_fwd_residuals` | 1,644 | splash forward-with-residuals being recomputed |
| `convert_reduce_fusion` | 545 | out_proj fp32→bf16 conversion fusions in recompute path |
| `slice_negate_fusion` | 248 | rope `neg` fusions in recompute path |

**The lever exists and is one-line**: `attn_proj_out` is already `checkpoint_name`-marked at `model.py:560` (CP path) and `model.py:636` (non-CP path) with explicit comment:
```python
out = ck(out, "attn_proj_out")  # offload: 448 MB/layer, skip Splash bwd recompute
```

The active checkpoint policy at `model.py:3052-3057` only includes `moe_layer_input`:
```python
_ckpt_policy = jax.checkpoint_policies.save_and_offload_only_these_names(
    names_which_can_be_saved=(),
    names_which_can_be_offloaded=("moe_layer_input",),
    offload_src="device", offload_dst="pinned_host",
)
```

The v315 author's comment at line 3047-3051 explicitly endorses both as favorable:
> "Only large activations (moe_layer_input, attn_proj_out) have favorable DUS:save ratio."

So `attn_proj_out` is a known-good lever that just isn't wired. Adding it requires one-line edit of the offload tuple.

**Sizing**:
- HBM: 448 MB/layer × 58 layers = **26 GB host offload total** (was 27 GB for moe_layer_input, comparable).
- DUS overhead: per v315 comment, ~25-35 ms × 58 = 1.5-2 sec/step (matching moe_layer_input pattern; same shape class).
- Recompute saved: most of the 4,216 ms attention recompute. Best case ~4,000 ms saved if all four sub-families short-circuit.
- **Net potential**: 4,000 - 1,800 = **~2,200 ms/step (~6% TPS)**.

Caveat: depends on JAX's bwd dependency analysis. The 1,644 ms splash_mha_fwd_residuals recompute may persist if the splash bwd (`dkv_no_residuals` family) is invoked AFTER the saved `attn_proj_out` value has already been used — i.e., if splash_mha_dkv_no_residuals needs Q/K/V residuals separately. Verify post-cluster by checking the iter-7 xplane for whether `splash_mha_fwd_residuals` still appears under `/rematted_computation/`.

### MoE forward recompute (3,000 ms — limited leverage)

| family | ms/step | what |
|---|---|---|
| `gmm_v2-` (fwd shape, in /checkpoint/) | 1,820 | MoE Pallas forward kernels recomputed |
| `all-gather.NNN.cloned.1.call-done` (chunk0+1+weight) | 1,007 | dispatch AG + weight AG recompute |
| `all-reduce.278.cloned.1.call-done` | 137 | router psum recompute |

The advisor's earlier analysis (iter-6 advisor call) showed: saving `hidden = silu(gate)*up` via `checkpoint_name` would only short-circuit the down-projection's bwd (~600 ms savings), because gate/up are kernel-internal residuals inside `gmm_v2_fused_silu_train`'s custom_vjp — not Python-level tensors that JAX can save.

**Lever (limited)**: Save `hidden`, accept ~600 ms gain at ~870 ms DUS overhead = potentially negative. Not worth a single-iter Greedy.

**Lever (bigger, multi-iter)**: rewrite `_expert_mlp_gmm_ag_body` to NOT use the fused silu kernel, allowing gate/up/hidden to be Python-level tensors that can be marked + offloaded individually. Loses iter-2's gmm_v2 fused win (+6.6%). Probably net-negative.

### MoE bwd-only ops (3,300 ms — bwd-side, fixed cost)

| family | ms/step | what |
|---|---|---|
| `scatter_custom_fusion` | 1,628 | bwd-of-scatter (the EP psum_scatter's bwd) |
| `multiply_reduce_fusion` | 1,291 | mostly attention out_proj bwd-side reductions per sample tf_op_name |
| `gather_offload_async_done` | 256 | async offload bwd |
| `slice / pad_add / add` | ~770 | small bwd setup ops |

Most are bwd-side (true bwd of fwd ops); tile-tuning of scatter would be candidate-A territory and is architectural.

### Other (~2,600 ms — small, distributed)

`convolution_bitcast_fusion` (657 ms) is in `mla_attention/out_proj` — it's actually a DOT general fused with bitcast ops; "convolution" here is just XLA's internal naming for matmul-via-conv-rewrite. Not a real conv op. Likely part of attention recompute and would be subsumed by `attn_proj_out` save.

## iter-7 Greedy lever (recommended)

**One-line change at `model.py:3054`**:
```python
names_which_can_be_offloaded=("moe_layer_input", "attn_proj_out"),
```

Rationale:
- Marker already exists (lines 560, 636).
- v315 author's comment endorses it as favorable DUS:save ratio.
- Sized at ~2,200 ms/step potential gain (~6% TPS).
- HBM impact bounded — host offload, no HBM growth (moe_layer_input pattern proven).
- AOT compile gate per AGENT.md §3 step-4b mandatory before submit (mirror production `LIBTPU_INIT_ARGS=--xla_tpu_scoped_vmem_limit_kib=65536`).

Pre-flight optional but recommended:
- `jax.make_jaxpr(_moe_layer_body)` after change → verify `attn_proj_out` appears as `from_residual(...)` in bwd jaxpr instead of being recomputed (iter-5 lesson: jaxpr-verify before cluster).

## iter-8 Lateral lever (deferred — option #1 from iter-6 framing)

Refactor `_moe_layer_body` so `jax.checkpoint` wraps only the attention part; MoE runs outside checkpoint. Bigger upside (~4 sec/step if HBM fits) but compile-time HBM risk; depends on iter-7 outcome (after attn recompute is eliminated, the multi-iter remat-policy redesign is cleaner).

## Cross-references

- iter-4 bisection: `research/dsv3/iter4_moe_gmm_ag_bisection.md`
- iter-5 retrospective (tile-tune regression): `autoperf/iter_log.md` § iter 5
- v315 author's policy decision: `model.py:3046-3057` + comment
- Existing checkpoint markers: `model.py:560`, `:568`, `:636`, `:2676`, `:2923`, `:2939`
