# DSv3 671B Training — Experiment Log v2

## Best Verified Result

**v264: 1740 TPS/chip, 18.8s/step, 28.2% MFU**
Config: EP=1, FSDP=256, TP=2, GBS=2048, `gmm` backend, psum on FSDP.
Profile: `gs://max-experiments/dsv3/profiles/v264-tp-d-fsdp256-tp2-gbs2048/`

## Current Active Experiments (4×4×16)

All use EP=16, FSDP=32, TP=1, GBS=2048 with topology-aware mesh (EP on XY, FSDP on Z×TC).

| Run | Dispatch | Sharding | Key Change | Profile |
|-----|----------|----------|------------|---------|
| v282-tf | AllGather | Batch on EP | Topology fix | v282-tf-ep16-fsdp32-4x4x16 |
| v283-a2a | A2A | Batch on EP | A2A dispatch | v283-a2a-ep16-fsdp32-4x4x16 |
| v284-cp | A2A | **Sequence on EP (CP)** | Context parallelism | v284-cp-ep16-fsdp32-4x4x16 |

## Run History

### Phase 1: Recovering v251 Baseline

| Run | TPS/chip | Step | Config | Finding |
|-----|----------|------|--------|---------|
| v251 | 1541 | 21.3s | EP=1 FSDP=256 TP=2, psum | Original baseline |
| v258-rd | 924 | 71.0s | EP=1 FSDP=256 TP=2, AllGather-D | 2.75× regression |
| v262-2k | 560 | 58.6s | Same as v258 at GBS=2048 | Confirmed regression |
| v263-v251 | 1540 | 21.3s | v251 image repro | Baseline confirmed |

**Root cause**: v258 changed MoE from column/row parallel (psum) to AllGather-D,
introducing 10.5 GB AllGather per MoE layer vs 1.8 GB psum.

### Phase 2: TP on D Optimization (v264)

| Run | TPS/chip | Step | Change |
|-----|----------|------|--------|
| v264 | **1740** | 18.8s | TP on D (reduction dim), FSDP on D_moe |

**Key insight**: put TP on the reduction dimension (D) so activation D/tp matches weight D/tp.
No AllGather needed. psum("fsdp") combines partial F contributions.

### Phase 3: AllGather-F Experiments (v268-v275)

Attempt to replace psum (batch-scaling) with AllGather (fixed cost).

| Run | TPS/chip | Step | Issue |
|-----|----------|------|-------|
| v268 | 502 | 65.2s | AllGather-F strided DMA: 44 GB/s (should be 200+) |
| v271 | 530 | 61.9s | AllGather-E: same 44 GB/s despite contiguous E |
| v272 | 571 | 57.4s | Flatten-gather-reshape: marginal improvement |
| v273 | 531 | 61.7s | MaxText layout flags: no improvement |
| v275 | 261 | — | SC offload flags: made it worse |
| v279 | 634 | 51.7s | Pure FSDP=512: 29.8s all_reduce on weight shape |
| v280 | crash | — | `to="reduced"` needs pcast (not implemented) |

**Root cause (AllGather slowness)**:
1. v7x Megacore (2,1) element packing forces strided DMA on gathered dimensions
2. ragged_dot `operand_layout_constraints` force F as minor dim regardless of storage order
3. AllGather inside shard_map: XLA can't pipeline across scan iterations
4. Missing `to="reduced"`: XLA fuses fwd AllGather + bwd ReduceScatter → single expensive all_reduce

**T(8,128) Padding**: F_local=8 at FSDP=256 → 93.8% waste (~14 GB). Unavoidable with ragged_dot.

### Phase 4: EP=16 Experiments (v276+)

| Run | TPS/chip | Step | Config | Issue |
|-----|----------|------|--------|-------|
| v276 (4×8×8) | 2155* | 15.2s | EP=16 FSDP=32, batch on EP | *WRONG: psum instead of A2A |
| v277-adam | 1590* | 20.6s | Same + Adam | *Same correctness bug |
| v278 (4×4×16) | 975* | 33.6s | Same, correct A2A dispatch | Mesh mapped EP to Z (wrong axis) |
| v282 | pending | — | AllGather + topology fix | EP on XY now |
| v283 | pending | — | A2A + topology fix | EP on XY now |
| v284 | pending | — | A2A + CP (sequence sharding) | 16× less A2A |

*v276/v277 had incorrect EP dispatch (psum instead of AllGather+psum_scatter). Loss appeared correct
at 5 steps from random init but would diverge during real training.

## Key Architectural Findings

### 1. Mesh Construction Must Be Topology-Aware

`np.array(devices).reshape(...)` ignores physical topology.
`mesh_utils.create_device_mesh()` maps largest logical axis to highest-BW physical axis — but
doesn't know which axis is on the critical path.

**Fix**: explicit topology mesh from device coordinates:
- EP → XY plane (4×4=16, ~360 GB/s bisection)
- FSDP → Z×TC (16×2=32, ~180 GB/s single axis)

### 2. AllGather vs A2A for EP Token Dispatch

**AllGather**: broadcast all tokens to all EP devices. Volume = T_local × EP × D.
Simple but O(EP) data per device regardless of routing.

**A2A**: send each token only to devices with its selected experts.
Volume = T_local × K/EP × D. ~2× less than AllGather, but A2A BW ≈ half AG BW → same wall time.

**Real win**: combine A2A with sequence sharding (CP). T_local drops by EP, so A2A volume
drops by EP² vs non-sharded AllGather.

### 3. Context Parallelism (CP) = Sequence Sharding on EP

**Batch sharding on EP**: Each device has B/(f×e) full sequences. T_local = B/(f×e) × S.
- Attention: no EP comm (independent sequences)
- MoE A2A: T_local × K/EP × D per device

**Sequence sharding on EP (CP)**: Each device has B/f sequences × S/EP positions.
T_local = B/f × S/EP = same count BUT:
- Attention: AllGather compressed KV across EP (~288 MB via MLA compression)
- MoE A2A: T_local drops by 1/EP → volume drops 16× → ~110 MB per layer!

**Net**: saves ~1.65 GB/layer MoE A2A, costs ~288 MB/layer attention KV gather. Big win.

### 4. EP + FSDP Communication Balance

With EP=16 (XY, 360 GB/s) and FSDP=32 (Z, 180 GB/s):

| Collective | Volume | Axis | Est. Time |
|-----------|--------|------|-----------|
| FSDP weight AG | 1.34 GB | Z | 7.4 ms |
| EP token A2A (batch) | 1.76 GB | XY | 4.9 ms |
| EP token A2A (CP) | 0.11 GB | XY | 0.3 ms |
| EP KV gather (CP) | 0.29 GB | XY | 0.8 ms |

CP makes EP comm negligible → FSDP weight AG becomes the bottleneck.

### 5. `to="reduced"` Pattern (MaxText)

MaxText uses `jax.lax.all_gather(..., to="reduced")` to keep fwd AllGather and bwd ReduceScatter
as separate pipelineable ops. Without it, XLA fuses them into a single all_reduce (29.8s in v279).

Requires `jax.lax.pcast(x, axis_name="ep", to="reduced")` first for axes where the tensor
is replicated. We haven't implemented this yet.

### 6. XLA Flags for v7x

MaxText's flag groups:
```
LAYOUT_FOR_ALL_REDUCE_SCATTER:
  --xla_tpu_use_minor_sharding_for_major_trivial_input=true
  --xla_tpu_relayout_group_size_threshold_for_reduce_scatter=1
  --xla_tpu_assign_all_reduce_scatter_layout=true

SPARSECORE_OFFLOAD_FOR_ALL_GATHER:
  --xla_tpu_enable_sparse_core_collective_offload_all_gather=true
  --xla_tpu_enable_all_gather_offload_tracing=true
  --xla_sc_disable_megacore_partitioning=true
  --xla_tpu_enable_async_collective_fusion_fuse_all_gather=false  (disable TC fusion)

MEGACORE:
  --xla_tpu_megacore_fusion_allow_ags=false
```

These flags didn't significantly help our AllGather-F approach (v273: 531 TPS/chip).
MaxText avoids the problem by keeping E_local ≥ 2 (FSDP=128 not 256).

### 7. carry_bytes Bug (Still Unfixed)

model.py:1682-1684: `carry_bytes = cfg.L_moe * B_l * 2` missing `× S × D`.
Host offload never triggers. Doesn't matter at GBS=2048 (carry fits), but blocks GBS=4096.

## Profile Locations

All at `gs://max-experiments/dsv3/profiles/<dir>/`

## Next Steps

1. Analyze v282/v283/v284 profiles on 4×4×16
2. Implement `to="reduced"` with pcast for proper AllGather pipelining
3. Compare CP implementation with MaxText's context parallelism
4. D-sharded carry (ReduceScatter at output) — halves carry
5. Fix carry_bytes bug for GBS=4096 path
6. Splash Attention for CP (asymmetric Q/K support)
