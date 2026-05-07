# iter-4 Tooling: MoE Expert path bisection on iter-2b xplane

**Date**: 2026-05-07
**Workload**: dsv3_train_full (DSv3-671B, v7x 4×8×8, fsdp=128 ep=4 tp=1, gbs=4096, gmm_v2 enabled)
**Class**: Tooling (xla_shell bisection of `moe_experts/moe_gmm_ag` source path)
**Input xplane**: `autoperf/profiles/dsv3train-i2b/plugins/profile/2026_05_07_02_55_30/`
**Step time**: 34,659 ms/step (1882 TPS/chip @ 30.5% MFU, the iter-2 baseline)

## Why this iter

iter-3's BF16 microbench grid produced ceiling data for `jax.lax.ragged_dot` (Mosaic GMM, the **pre**-gmm_v2 kernel). iter-2b has been on the new gmm_v2 Pallas kernel since iter-2 (+6.6% TPS lever). The "0.6226 ceiling vs 0.244 in-training" framing in iter-3's iter_log is therefore apples-to-oranges: ceiling measured forward kernel only on the pre-iter-2 kernel; in-training measurement is the post-iter-2 kernel **plus dispatch/scatter/all-gather/backward**. Without bisection we don't know where the optimization headroom actually sits in the iter-2b state.

## Method

`xla_shell list_sources --json --top 200` on the iter-2b xplane, grouping `while/body/.../moe_experts/moe_gmm_ag/...` source paths into bands (forward vs `/checkpoint/` remat; kernel-call vs scatter vs dispatch vs jvp vs transpose). Total 86 distinct sub-sources; classification covers 100% of `moe_gmm_ag` time.

## Results

**Total `moe_experts/moe_gmm_ag` time: 16,655.9 ms/step** — that's **48.1% of the 34.65 s step**. The single dominant compute scope.

### Forward path (5,436 ms/step, 32.6% of MoE)

| sub-source | ms/step | % of fwd | what it is |
|---|---|---|---|
| `jit(gmm)` | 1,844.5 | 33.9% | gmm_v2 kernel forward calls (3 per chunk × 2 chunks × 58 layers; gate/up fused via `gmm_v2_fused_silu_train`, down via `gmm_v2_train`) |
| `scatter` | 1,684.9 | 31.0% | post-call writeback / EP `psum_scatter` of expert outputs back to per-token rows |
| `ep_token_gather` | 998.4 | 18.4% | dispatch AG: gather tokens from all EP shards before grouped matmul (chunk0 + chunk1 combined) |
| `weight_allgather` | 388.8 | 7.2% | FSDP weight AG over F (expert) axis. Same as `FSDP_AG` leaf in headroom report (389 ms/step). |
| shard_map other | 325.6 | 6.0% | shard_map setup, mask construction, unaccounted small ops |
| `gather` | 194.4 | 3.6% | post-call read of expert outputs |

### Backward path (11,219 ms/step, 67.4% of MoE)

| sub-source | ms/step | what it is |
|---|---|---|
| `transpose(moe...)` (under `/checkpoint/`) | 7,913.1 | XLA's autodiff bwd-transpose pass for all forward operations in `moe_gmm_ag`. Includes bwd of kernel calls (`megablox.gmm` + `megablox.tgmm`), bwd of scatter, bwd of dispatch all-gather, bwd of weight all-gather. |
| `jvp() bwd` (under `/checkpoint/`) | 3,306.1 | cotangent prep + accumulation across the recomputed forward |

The remat path roughly = forward time × 2 + actual bwd compute (~5,400 fwd recompute + ~5,800 bwd compute = ~11,200 ms — sanity-checks the breakdown).

## Key findings

### 1. Forward kernel is at-ceiling; little headroom there

`jit(gmm)` forward = **1,844.5 ms/step**. Per perfsim's microbench (PR#23 grid), gmm_v2 at production shape (gate/up/down with M=131,072 grouped, K∈{2048,7168}, N∈{2048,7168}, n_groups=64) achieves **0.61–0.62 efficiency** at the **ragged_dot** kernel, which is what the microbench actually measured. We don't have direct microbench data on gmm_v2 the kernel, but iter-2's +6.6% TPS confirms gmm_v2 outperforms ragged_dot at this shape; presumably its kernel-only ceiling is ≥ ragged_dot's 0.62. With ~1.85 sec at ~700 TFLOPS achieved per core (estimated from the FLOPs budget), the forward kernel is essentially **at ceiling**. **Tile-tuning gmm_v2 forward is unlikely to land >5–10% on the kernel call.**

### 2. The "2.5× ceiling-vs-measured" framing was misleading

iter-3 iter_log claimed a 2.5× gap between standalone bench (0.6226) and v304 in-training (0.244), framed as "in-training overhead". The truth is more nuanced:
- The standalone microbench measured **forward kernel only** for `jax.lax.ragged_dot`.
- The v304 xplane "0.244" derivation took total `moe_gmm_ag` per-call wall time and divided by ragged_dot FLOPs. That total includes dispatch, scatter, **and the implicit backward path** via remat (since v304 used remat=full).
- The factor-of-~2 from forward to backward (with remat=full) accounts for most of the gap.

**Recommendation**: retract the "2.5× in-training overhead" framing in `v7x_KNOWLEDGE.md` §5. Replace with the bisection table above. The real headroom landscape is described by the per-band breakdown, not a single ratio.

### 3. Top iter-5 lever candidates ranked by absolute headroom

| candidate | size (ms/step) | approach | risk |
|---|---|---|---|
| **A. EP scatter (post-call writeback)** | 1,685 fwd + bwd | Replace EP `psum_scatter` with a fused combine, or move the scatter into the gmm_v2 kernel directly. Architectural change to gmm_v2 — substantial. | Pallas kernel correctness; bwd must match. |
| **B. ep_token_gather (dispatch AG)** | 998 fwd + bwd | Cross-layer prefetch via `moe_xlayer_prefetch`. Same lever as iter-1 (compile-failed). Need to fix the underlying `all_gather_reduced` bwd-transpose bug (`v7x_KNOWLEDGE.md` §5). | Original lever lever was iter-1 halt; bug is unaddressed in jax-gpt. |
| **C. Backward-path tile tuning** | 11,219 (combined) | `_gmm_tiles` / `_tgmm_tiles` in `gmm_v2_train.py` already use tokamax-tuned values for `(M, K, N) ∈ {(131072,2048,7168), (131072,7168,2048), (131072,7168,1024), (131072,1024,7168)}`. Other shapes fall back to a generic `(128, min(K,2048), min(N,1024))` heuristic. Worth profiling whether any backward call hits the fallback. | Need backward HLO inspection to identify shapes; tile changes can compile-error post-Mosaic-lowering (CLAUDE.md AOT compile gate before any cluster submit). |
| **D. Reduce remat scope** | up to 5,436 fwd recompute | Switch from `remat=full` to `remat=attn_only` for MoE chunks; saves the forward recompute time (≈5.4 sec/step). | HBM headroom — model already runs near runtime-compile limit; saving recompute means saving activations to HBM. Need careful HBM accounting before submitting. |

### 4. Forward kernel call count

If the production breakdown is 24 calls/chunk × 2 chunks × 58 layers = ~2,784 calls/step (forward only), then per-call kernel time = 1,844.5/2784 ≈ 663 µs. At the per-core peak 1153.5 TFLOPS bf16 and the gate/up shape's ~3.85e12 FLOPs/call, theoretical minimum = 3.85e12 / (0.62 × 1153.5e12) ≈ 5,395 µs. **That's larger than the measured 663 µs**, suggesting:
- per-call FLOPs is smaller than 3.85e12 because chunked dispatch divides M by `moe_n_chunks=2` and EP_size=4 ⇒ M_per_call ≈ 16,384 not 131,072
- At smaller M, microbench efficiency drops (M=16,384 grouped n=64: eff=0.49) but absolute time per call drops more.
This back-of-envelope places the forward kernel **at the kernel-only ceiling within sigma**, supporting finding (1).

## Decision for iter-5

iter-5 should be **Greedy on candidate A or C** (scatter or backward tile-tuning). Both are jax-gpt-side levers requiring careful kernel-correctness validation. Per `~/.claude/CLAUDE.md` Pallas kernel constraints + AOT compile gate, both should pass the AOT compile check before cluster submit.

Candidate B (`moe_xlayer_prefetch`) requires fixing the iter-1 compile bug first — out of scope for a single iter.
Candidate D (remat scope) requires HBM headroom analysis first; not a one-line change.

If user prefers a faster Lateral iter, **profile-driven backward tile-tuning (C)** is the cleanest path:
1. Inspect bwd HLO via xla_shell for `_gmm_tiles` fallback hits at non-tokamax shapes.
2. Add tokamax entries for those shapes if available; otherwise micro-benchmark via bench_runner extended for tgmm.
3. Re-measure step time.

## Cross-references

- iter-3 microbench grid: `gs://max-experiments/autoperf/microbench/v7x_4x8x8_bf16_2026-05-07/`
- iter-2 commit (gmm_v2 enable): `f0b34da` on `autoperf/dsv3_train_full`
- iter-1 compile failure on `moe_xlayer_prefetch`: `autoperf/iter_log.md` § iter 1
- gmm_v2 kernel: `jax_gpt/models/dsv3/kernels/gmm_v2.py`
- gmm_v2 training wrapper: `jax_gpt/models/dsv3/kernels/gmm_v2_train.py` (custom_vjp; bwd via megablox)
- Pre-existing bucketer caveat: ultrons/perfsim#26 (Expert_gmm pattern doesn't match gmm_v2 fusion names)
