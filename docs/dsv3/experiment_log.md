# DSv3 671B Training — Experiment Log

Config: 4×8×8 (256 chips, 512 devices), FSDP=256, EP=1, TP=2, GBS=2048, `gmm` backend.

## Best Result

**v264 (TP on D, psum): 1740 TPS/chip, 18.8s/step, 28.2% MFU**

Profile: `gs://max-experiments/dsv3/profiles/v264-tp-d-fsdp256-tp2-gbs2048/`

Key: TP shards D (reduction dim), FSDP shards D_moe. No weight AllGather.
Column/row parallel: psum("fsdp") combines partial F contributions.
+13% over v251 baseline (1541 TPS/chip).

## All Runs

| Run | Image | Change | TPS/chip | Step | MFU | Profile |
|-----|-------|--------|----------|------|-----|---------|
| v251 (original) | v251-rd | Baseline: col/row parallel, D full, F by fsdp×tp | 1541 | 21.3s | 25.0% | v251-rd-proper-tp |
| v258-rd | v258-rd | AllGather D, offload | 924 | 71.0s | 14.9% | v258-rd-offload-gbs4096 |
| v259-rd | v258-rd | Same code, FSDP=512 TP=1 | 923 | 71.0s | 14.9% | v259-rd-fsdp512-gbs4096 |
| v261-rd | v261-rd | Newer code, FSDP=512 TP=1 | OOM | — | — | — |
| v262-2k | v258-rd | v258 code at GBS=2048 | 560 | 58.6s | 9.1% | — |
| v263-v251 | v251-rd | v251 repro (confirmed) | 1540 | 21.3s | 24.9% | v263-v251-fsdp256-tp2-gbs2048 |
| **v264** | **v264-tp-on-d** | **TP on D, FSDP on D_moe, no AG** | **1740** | **18.8s** | **28.2%** | **v264-tp-d-fsdp256-tp2-gbs2048** |
| v265-lay | v265-layout | + Layout API per-layer | 1282 | 25.6s | 20.7% | v265-lay-fsdp256-tp2-gbs2048 |
| v267-xpose | v267-transpose | Store wi as (E,F,D) | 1740 | 18.8s | 28.2% | v267-xpose-fsdp256-tp2-gbs2048 |
| v268-agf | v268-agf | AllGather F, no psum | 502 | 65.2s | 8.1% | v268-agf-fsdp256-tp2-gbs2048 |
| v271-age | v271-age | AllGather E, no psum | 530 | 61.9s | 8.5% | v271-age-fsdp256-tp2-gbs2048 |
| v272-fgr | v272-fgr | Flatten-gather-reshape | 571 | 57.4s | 9.2% | v272-fgr-fsdp256-tp2-gbs2048 |
| v272-fgr (flags) | v272-fgr | + minor_sharding + exp_sched flags | 571 | 57.4s | 9.2% | — |
| v273-mxf | v273-maxflags | Straight AG + MaxText flags | pending | — | — | v273-mxf-fsdp256-tp2-gbs2048 |

## Key Findings

### 1. MoE Weight Sharding — v251 → v258 Regression (2.75×)

v251 used column/row parallel MoE: weights `P("ep", None, ("fsdp","tp"))` — D full, F by fsdp×tp.
No weight AllGather. One psum(("fsdp","tp")) per layer ≈ 1.8 GB.

v258 changed to AllGather-D: weights `P("ep", "fsdp", "tp")` — D by fsdp, F by tp.
AllGather D inside shard_map: 10.5 GB/layer. Caused 560 TPS/chip at GBS=2048.

### 2. TP on D (v264) — Best Approach

Put TP on reduction dimension (D), FSDP on D_moe (non-reduction).
Contraction dim D/tp matches between activation and weight → no AllGather needed.
psum("fsdp") combines partial F contributions: 224 MB/layer.
1740 TPS/chip = +13% over v251.

### 3. T(8,128) Padding Waste — 14 GB Unavoidable at F_local=8

With FSDP=256: F_local = D_moe/256 = 8. ragged_dot kernel forces F=8 as minor dim → 
pads to 128 → 93.8% waste. ~14 GB on stacked params.

Tried: transpose storage, Layout API, different gather axis. XLA's ragged_dot 
`operand_layout_constraints` always force F as minor. Confirmed via AOT compile on v7x.

Only fix: F_local ≥ 128 (requires FSDP ≤ 16) — changes entire config.
At GBS=2048 we have headroom. For GBS=4096: fix carry offload instead.

### 4. AllGather-F/E Approach — 3× Slower Than psum

Replacing psum with AllGather (to eliminate batch-scaling comm):
- AllGather F (v268): 502 TPS/chip — strided DMA at 44 GB/s
- AllGather E (v271): 530 TPS/chip — still 44 GB/s despite contiguous E
- Flatten trick (v272): 571 TPS/chip — marginal improvement

Root cause: 256-way AllGather of 3.5 GB per weight (10.5 GB total) is 47× more data
than the 224 MB psum. Even at full ICI BW (250 GB/s), 3× AllGather = 42ms vs 16ms psum.
Only wins at GBS > ~6000 where psum scales past the fixed AG cost.

v7x Megacore (2,1) packing + size-1 major dim → strided DMA → 44 GB/s effective.
MaxText uses specific flags to fix this — testing in v273.

### 5. Layout API Issues

- Per-layer Layout constraint (v265): XLA inserts relayout copies per scan iteration → 26% regression.
- Stacked params Layout (v266): int32 allocation overflow — tensor too large for the 6D retiling.
- XLA always overrides our layout hints for ragged_dot operands (confirmed via AOT v7x compile).

### 6. carry_bytes Bug

Current code (model.py:1682-1684) has broken host offload formula:
```python
carry_bytes = cfg.L_moe * B_l * 2  # MISSING: × S × D
```
Missing D=7168, divides by TP incorrectly. Offload never triggers.
v251 didn't need offload (25.4 GB fits). For GBS=4096: must fix this.

### 7. Device Ordering — TP on d2d Confirmed

AOT topology check confirmed TP=2 correctly maps to intra-chip d2d (same coords, 
different core_on_chip). The 171 GB/s AR(tp) bandwidth is the actual d2d AllReduce 
throughput, not a mapping issue.

### 8. v251 Profile Analysis

Step time: 21.3s. Key per-layer costs:
- splash_mha_fwd: 23.9 ms (28%)
- **psum("fsdp"): 16.15 ms (19%)** ← dominant non-attention cost
- scatter-add: 10.83 ms (13%)
- ragged_dot compute: 4.26 ms (5%) — terrible MXU at F=8
- Attn weight AGs: 3.47 ms (4%)

Collective overlap: AG 81.5%, AR 5.4% (1.89s stall), RS 66.8%.

## Open Questions

1. Can MaxText's XLA flags (`minor_sharding_for_major_trivial`, `megacore_fusion_allow_ags=false`) 
   fix the AllGather bandwidth? → v273 testing now.

2. Can we implement MaxText's explicit weight prefetch pattern (AllGather layer i+1 in carry 
   while computing layer i) with `scheduling_group` hints?

3. D-sharded carry (ReduceScatter at output, AllGather at input) — halves carry from 25.4→12.7 GB.
   Not yet tested.

4. GBS=4096 path: fix carry offload formula + the 14 GB padding waste is tolerable if carry is 
   offloaded to host.

## Profile Locations

All at: `gs://max-experiments/dsv3/profiles/<dir>/`
