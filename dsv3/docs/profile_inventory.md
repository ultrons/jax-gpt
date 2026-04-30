# DSv3 671B Profile Inventory (recent runs)

Cluster: `gke_cloud-tpu-multipod-dev_us-central1_bodaborg-super-alpha-cluster`, namespace `poc-dev`.
Common defaults across all runs below: model `full_671b` (E=256, K=8, S=4096, D=7168, F_moe=2048, L=58 MoE + 3 dense),
optimizer SGD, dtype bf16, gradient_checkpoint=True, attn `splash` with `block_q=2048` + `use_fused_bwd_kernel=True`,
profile_skip=2, profile_steps=1.

All profiles live under `gs://max-experiments/dsv3/profiles/<run-tag>/plugins/profile/<timestamp>/<host>.xplane.pb`.

## Recent perf series (chunking → kernel work)

| ver | tag (GCS dir name) | topo | EP | FSDP | TP | GBS | n_chunks | special | step time | TPS/chip | MFU |
|---|---|---|---:|---:|---:|---:|---:|---|---:|---:|---:|
| v292 | `v292-nosched-ep4-fsdp128-4x8x8` | 4×8×8 | 4 | 128 | 1 | 1024 | 1 | revert scheduling_group + opt_barrier | — | — | — |
| v296 | `v296-gbs4k-ep4-fsdp128-4x8x8` | 4×8×8 | 4 | 128 | 1 | **4096** | 1 | best practical baseline | 33.5s | **1962** | 31.7% |
| v298 | `v298-ep8-ep8-fsdp64-4x8x8` | 4×8×8 | **8** | 64 | 1 | 4096 | 1 | EP=8 + A2A dispatch | (regression) | — | — |
| v300 | (no profile flushed) | 4×8×8 | 4 | 128 | 1 | 4096 | 1 | MaxText SC offload XLA flags only | — | — | — |
| v301b | `v301b-chunk2-ep4-fsdp128-4x8x8` | 4×8×8 | 4 | 128 | 1 | 4096 | **2** | Python-unrolled token chunking | 35.1s | 1862-1869 | 30.1% |
| v302 | `v302-chunkbar-ep4-fsdp128-4x8x8` | 4×8×8 | 4 | 128 | 1 | 4096 | 2 | + per-chunk AG + intra-chunk barriers | 35.3s | 1848-1856 | 30.0% |
| v303 | `v303-2sc-ep4-fsdp128-4x8x8` | 4×8×8 | 4 | 128 | 1 | 4096 | 2 | + 2-SC AG offload + async-RS XLA flags | 33.5s | **1948-1955** | 31.6% |
| v304 | `v304-auxar-ep4-fsdp128-4x8x8` | 4×8×8 | 4 | 128 | 1 | 4096 | 2 | + deferred aux_loss AR (one global at end) | 33.5s | 1948-1955 | 31.6% |
| v305 | `v305-scscatter-ep4-fsdp128-4x8x8` | 4×8×8 | 4 | 128 | 1 | 4096 | 2 | + SC gather-reduce kernel for scatter | 35.6s | 1840 | 29.8% |
| v307 | `v307-gmmv2-ep4-fsdp128-4x8x8` | 4×8×8 | 4 | 128 | 1 | 4096 | 2 | + Pallas gmm_v2 fwd (Stage A.1, jax.vjp bwd) | 34.0s | 1921-1928 | 31.1% |
| v308 | `v308-gmmv2-n1-ep4-fsdp128-4x8x8` | 4×8×8 | 4 | 128 | 1 | 4096 | **1** | gmm_v2 + chunks=1 (Stage A.1 bwd) | 33.9s | 1930 | 31.3% |
| v309 | (NaN, no profile) | 4×8×8 | 4 | 128 | 1 | 4096 | 1 | gmm_v2 + Stage A.2 (gmm/tgmm bwd, default tiles) | 558s | NaN | — |
| v309b | (NaN, no profile) | 4×8×8 | 4 | 128 | 1 | 4096 | 1 | + stable silu' (still NaN) | 553s | NaN | — |
| v309c | (job killed; partial) | 4×8×8 | 4 | 128 | 1 | 4096 | 1 | + saved g,u as residuals (finite) | 391s | **167** (12× regression) | 2.7% |
| v310 | `v310-tuned-ep4-fsdp128-4x8x8` | 4×8×8 | 4 | 128 | 1 | 4096 | 1 | + tokamax-tuned gmm/tgmm tiles (fixed 12× regression) | 33.8s | 1939-1940 | 31.4% |

## EP-shape sweep + remat exploration (v311-v316)

| ver | tag (GCS dir name) | topo | EP | FSDP | TP | GBS | n_chunks | special | step time | TPS/chip | MFU |
|---|---|---|---:|---:|---:|---:|---:|---|---:|---:|---:|
| v311 | (compile OOM, no profile) | 4×8×8 | **1** | **512** | 1 | 4096 | 1 | pure FSDP, fix SMEM crash via fuse_RS=false + 3d_decomposer=false | OOM @ 115.87G | — | — |
| v312 | `v312-ep2cores-ep2-fsdp256-4x8x8` | 4×8×8 | **2 (cores)** | 256 | 1 | 4096 | 2 | EP=2 on cores axis (intra-chip free A2A) | 44.9s | 1461 | 23.6% |
| v313 | `v313-ep2c-n1-ep2-fsdp256-4x8x8-chunks1` | 4×8×8 | 2 (cores) | 256 | 1 | 4096 | **1** | + chunks=1 (bigger AG, fewer of them) | 42.5s | 1540 | 24.9% |
| v314 | (host OOM at step 1, killed) | 4×8×8 | 2 (cores) | 256 | 1 | 4096 | 1 | + offload q_a, kv_a, attn_proj_out, shared_hidden — 544 GB/pod host pinned | host OOM | — | — |
| v314a | `v314a-remat-ep2-fsdp256-4x8x8-offload` | 4×8×8 | 2 (cores) | 256 | 1 | 4096 | 1 | drop attn_proj_out (208 GB/pod too much); keep q_a+kv_a+shared_hidden (336 GB/pod) | 41.2s | **1591** | 25.8% |
| v315 | `v315-1off-ep2-fsdp256-4x8x8` | 4×8×8 | 2 (cores) | 256 | 1 | 4096 | 1 | drop ALL small offloads (DUS overhead test) | 42.5s | 1540 | 24.9% |
| v316 | (NaN, no profile) | 4×8×8 | 4 | 128 | 1 | 4096 | 2 | v304 baseline + small offloads + 3d_RS_decomposer=true + offloading_copy_to_sparsecore=true | 33.98s | NaN @ 1928 | 31.2% |
| v316a | (NaN, no profile) | 4×8×8 | 4 | 128 | 1 | 4096 | 2 | v316 minus the 2 new flags — flags weren't the cause; same NaN | 33.9s | NaN @ 1934 | 31.3% |
| v316b | `v316b-n1-ep4-fsdp128-4x8x8-smalloff-rs/` | 4×8×8 | 4 | 128 | 1 | 4096 | **1** | + chunks=1 (avoids the chunks=2 + checkpoint_name NaN bug) — finite loss but slower | 34.6s | 1893 | 30.6% |

| v316c | `v316c-sccopy-ep4-fsdp128-4x8x8-smalloff-rs/` | 4×8×8 | 4 | 128 | 1 | 4096 | 1 | v316b + offloading_copy_to_sparsecore=true (isolated) | 34.5s | 1900 | 30.8% |

### Findings v316/v316a/v316b/v316c
- **chunks=2 + small offloads + EP=4 = NaN** (checkpoint_name interaction with chunks=2 control flow bug). chunks=1 fixes the NaN.
- **Small offloads HURT at EP=4** (−2% vs v308). Helped at EP=2 (+3% vs v313) only because EP=2's looser overlap had idle windows to hide DUS. v304's tighter FSDP overlap leaves no slack.
- **`xla_tpu_enable_3d_reduce_scatter_decomposer=true`** breaks at EP=4 (v316 NaN). Keep permanently OFF.
- **`xla_tpu_enable_offloading_copy_to_sparsecore=true`** is SAFE and gives +0.4% (v316c vs v316b at 1900 vs 1893). Free win to add to any baseline.
- **v304 remains bf16 optimum at 1948 TPS/chip**. Topology and remat exploration exhausted; **FP8 weights is the next major lever**.

### Findings from v311-v315
- **v311**: Pure FSDP=512 fundamentally OOMs in bf16. 15× weight buffer copies (3 in-flight AGs + 12 layout permutations) of bf16[256,2048,7168]=7GB. Only fixable with FP8.
- **v312/v313**: EP=2 cross-cores LOSES vs EP=4 (v304). FSDP doubles (5.6→11.25 GB/layer) and ring doubles (128→256 dev). Free intra-chip A2A wasn't a win because A2A was already 98% hidden in v304.
- **v313→v314a**: Adding small offloads (q_a/kv_a/shared_hidden) gives +3% (1540→1591). DUS per offload is 25-35 ms × 58 layers = 1.5-2 s pure overhead each, but it hides behind other work; recompute saved > DUS overhead.
- **v315**: Confirms (matches v313 exactly at 1540) — dropping small offloads regresses to no-offload baseline. The +3% in v314a was real net positive.
- **v304's EP=4 remains the bf16 sweet spot** (1948 TPS/chip). EP-shape changes don't beat it.

## Earlier runs (kernel correctness era — fewer perf-comparable)

| ver | tag | topo | EP | FSDP | TP | GBS | notes |
|---|---|---|---:|---:|---:|---:|---|
| v286 | `v286-spcp-ep16-fsdp32-4x4x16` | 4×4×16 | 16 | 32 | 1 | — | CP+A2A+Splash, pre-token-chunking |
| v287 | `v287-agep4-ep4-fsdp128-4x8x8` | 4×8×8 | 4 | 128 | 1 | 1024 | first AG-dispatch on 4×8×8 |
| v288 | `v288-fsdp1k-fsdp1024-8x8x8` | 8×8×8 | 1 | 1024 | 1 | — | EP=1 ablation |
| v289 | `v289-nopad-ep4-fsdp128-4x8x8` | 4×8×8 | 4 | 128 | 1 | 1024 | no padding tuning |
| v290 | `v290-aghoist-ep4-fsdp128-4x8x8` | 4×8×8 | 4 | 128 | 1 | 1024 | AG hoisting attempt |
| v291 | `v291-fusedbwd-ep4-fsdp128-4x8x8` | 4×8×8 | 4 | 128 | 1 | 1024 | Splash use_fused_bwd_kernel + block_q=2048 |
| v293b | `v293b-sck-ep4-fsdp128-4x8x8` | 4×8×8 | 4 | 128 | 1 | 1024 | first SC kernel attempt (vendored) |

## Pre-v286 (different model/sharding regimes)

| ver | tag | topo | sharding |
|---|---|---|---|
| v250-v275 | `v250-rd-…` through `v275-agf-…` | 4×8×8 | various ragged_dot + offload + tp explorations on GBS 2k-4k |
| v276-v283 | `v276-ep-ep16-fsdp32-…` series | 4×4×16 | EP=16 / FSDP=32 explorations on bodaborg |
| v281-v283 | `v281-tm-`, `v282-tf-`, `v283-a2a-` (all on `4x4x16`) | 4×4×16 | A2A + ep16 path |
| bench-v4-{jax,pallas} | inference benchmarks | n/a | reference benchmarks |

## Key analysis findings (running through these profiles)

1. **v303 trace overlap (canonical baseline measurement)**: AG 99.4% overlap, RS 75.8% overlap, **overall 98.2% comm overlap, exposed stall 3.5s of 33.5s step (0.6%)**. Comm is NOT the bottleneck.
2. **v304 confirmed compute is the constraint**: deferred aux_loss removed 8.6s of cumulative AR time but step time unchanged → AR was already opportunistically hidden.
3. **v307 profile**: gmm_v2 forward 38s/host vs ragged_dot's 156s — 2× compute speedup verified, but wall time barely moved (compute was already hidden).
4. **v309c profile (12× slowdown)**: localized to gmm/tgmm Pallas backward kernels using default `(128, 128, 128)` tiles. For our M=131072 shape that's ~920K tile iterations per kernel call.
5. **scatter_custom_fusion ≈ 28.8 ms × 58 layers × 2 = 3.3s/step (~10% of step)** — second-largest chunk after splash attention (32%). HBM-bound at ~1.0 FLOP/B.

## Current best
**v304 / v303**: ~1948-1955 TPS/chip, 31.6% MFU, 33.5s step. With an honest target of 2400-2700 TPS/chip in bf16 = potentially externally-publishable best on TPU.

## Compute breakdown (per chip per step, approximate from v307 profile)

| op category | aggregate (8 dev) | per-chip | step share |
|---|---:|---:|---:|
| Splash attention (fwd + bwd) | ~87s | ~10.7s | **~32%** |
| ragged-dot-none (incl. bwd) | ~156s (v304) / ~44s (v307) | ~5.5s / ~3.0s | 16% / 9% |
| scatter_custom_fusion | ~52s | ~6.5s | ~19% (on TC HBM bus) |
| gmm_v2 fused-silu (v307) | ~38s | ~4.7s | ~14% |
| FSDP weight AG (already 99.5% hidden) | — | — | — |
| Token AG (latency-bound; some exposed) | ~10.6s exposed | ~1.3s | ~4% |
| Other (norms, optimizer, copies, ICI, ...) | residual | residual | ~22% |

## How to load a profile

```bash
source ~/xdb/.xprof/bin/activate

# CLI analysis (no UI needed)
cd ~/ml-experiments/timing
python -m xla_shell -c "read_xplane gs://max-experiments/dsv3/profiles/<tag>; parse_trace --collectives"

# Or browse with xprof server
GOOGLE_CLOUD_PROJECT=cloud-tpu-multipod-dev xprof --port=9090 \
    --logdir=gs://max-experiments/dsv3/profiles/<tag>
```
