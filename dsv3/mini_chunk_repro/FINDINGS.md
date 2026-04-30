# Mini-repro findings — token chunking inside MoE body

## Setup
- 4 v4 cores, mesh dp=1, fsdp=2, ep=2, tp=1
- Toy dims: T_local=512, D=512, F_moe=256, E=8, K=4
- Mirrors v301's `_expert_mlp_gmm_ag_body`: AG once, then per-chunk { dynamic_slice, sort, ragged_dot×3, psum_scatter }

## Result

| n_chunks | wall (us) | total chip time (ms) | non-collective (ms) | RS time (ms) |
|---:|---:|---:|---:|---:|
| 1 | 304 | 11.8 | 9.4 | 1.25 |
| 2 | 354 (+16%) | 13.8 (+17%) | 11.3 (+1.9) | 1.4 |

Mirrors v301 cluster regression (1962 → 1862 TPS/chip, –5%).

## Why chunking doesn't pay

1. **RS is synchronous on v4.** Even with `xla_tpu_enable_async_collective_fusion_fuse_reduce_scatter=true` and `xla_enable_async_reduce_scatter_fusion=true`, the HLO emits plain `reduce-scatter` (not `reduce-scatter-start/-done`). On v7x cluster, RS *should* go async via `xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=true` (offload to SC), but that path isn't available locally.

2. **HLO ordering is "all chunk-0 then all chunk-1".** With async AGs but sync RS:
   ```
   AGs start (chunk-shared, hoisted)
   chunk-0 ragged-dot×3
   chunk-0 reduce-scatter  ← BLOCKING
   chunk-1 ragged-dot×3
   chunk-1 reduce-scatter
   ```
   Chunk 0 RS happens *between* chunk 0 and chunk 1 ragged-dots. There's no scheduling cycle to overlap them because the TC is doing both, and the RS sits on the critical path.

3. **+1.9 ms (+20%) non-collective overhead from the chunking itself**: 2× ragged-dot-metadata, 2× dynamic_slice, 2× sort+gather_custom_fusion, smaller GEMMs run less efficiently per byte.

4. **AGs already overlapped fine in n_chunks=1** (only 0.76 ms exposed stall out of 1.1 ms AG time). There's nothing left to overlap with the AG path.

## Implication for v301 on cluster

On v7x with SC offload, RS *might* go async — that would create the pipelining opportunity we want. But chunking still pays an overhead cost and shrinks GEMMs. For chunking to win, the savings from `chunk-0-RS overlapping chunk-1-compute` must exceed:
- chunking machinery overhead (~10–20%)
- ragged_dot efficiency loss at smaller LHS

In v301 (T_local≈8192 → 2 chunks of 4096), GEMMs are still big enough that #2 should be small. So the v301 regression on cluster is likely the **same RS-stays-sync issue** — SC offload didn't pick up these RSes for some reason (different shape? bf16 vs f32? the gmm_ag custom_call boundary?).

## Next steps (cheap → expensive)

1. **Check actual cluster HLO from v301 dump** for `reduce-scatter-start` vs `reduce-scatter`. Confirms whether SC offload fired. (zero cost)

2. **Add explicit `optimization_barrier` between chunks** to *force* the scheduler to interleave chunk-0 RS with chunk-1 compute. Sometimes the latency-hiding scheduler needs a hint. (cheap)

3. **Switch from `psum_scatter` to manual `collective_permute` ring** for RS. Ring decomposition produces P-1 small permutes that are inherently async-able. (medium effort, may break SC offload)

4. **Layer-level micro-batching (batchsplit)**: split batch in 2, run layer L on mb0 while layer L-1 finishes mb1's RS. This has the structural benefit (RS happens *before* attention of next mb, not between MoE compute), at the cost of re-running attention. (high effort — full pipeline rewrite)

The right question for the cluster is: **does v301's actual HLO show `reduce-scatter-start` on cluster v7x, and if so, does the profile show overlap?**
