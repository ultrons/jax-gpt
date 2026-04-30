# DSv3 671B — Sharding Trade-off Analysis

Topology: 4×4×16 = 256 chips × 2 TCs = 512 devices.
Model: D=7168, D_moe=2048, E=256, K=8, H=128, S=4096, L=61 (3 dense + 58 MoE).

## Configurations

| Config | A (current) | B | C |
|--------|-------------|---|---|
| **EP** | 1 | 4 (4×1×1) | 16 (4×4×1) |
| **FSDP** | 256 | 64 (1×8×8) | 16 (1×1×16) |
| **TP** | 2 | 2 | 2 |
| **DP** | 1 | 1 | 1 |
| **Total** | 512 | 512 | 512 |
| EP axis ICI | — | x-axis (200 GB/s) | x×y plane (200 GB/s) |
| FSDP axis ICI | all ICI | y×z plane (200 GB/s) | z-axis (50 GB/s) |
| TP axis | d2d intra-chip | d2d intra-chip | d2d intra-chip |

## Per-device MoE weight shapes

| | A (FSDP=256) | B (FSDP=64) | C (FSDP=16) |
|---|---|---|---|
| E_local | 256 (EP=1) | 64 (EP=4) | 16 (EP=16) |
| D_local (D/TP) | 3584 | 3584 | 3584 |
| F_local (D_moe/FSDP) | **8** | **32** | **128** |
| wi per-dev | (256, 3584, 8) | (64, 3584, 32) | (16, 3584, 128) |
| wo per-dev | (256, 8, 3584) | (64, 32, 3584) | (16, 128, 3584) |
| T(8,128) padding on F | 8→128 = **93.8%** | 32→128 = **75%** | 128=128 = **0%** |
| wi HBM per-dev (logical) | 14.6 MB | 14.6 MB | 14.6 MB |
| wi HBM per-dev (padded) | 234 MB | 58.4 MB | **14.6 MB** |
| wi stacked ×58 (padded) | **13.3 GB** | **3.3 GB** | **0.85 GB** |

## Per-device activation shapes

| | A | B | C |
|---|---|---|---|
| GBS | 2048 | 2048 | 2048 |
| B_local (GBS/FSDP) | 8 | 32 | 128 |
| T_fsdp (B_local × S) | 32768 | 131072 | 524288 |
| x per-dev (T_fsdp, D_local) | (32768, 3584) | (131072, 3584) | (524288, 3584) |
| x size per-dev | 224 MB | 896 MB | 3.5 GB |
| Scan carry per layer | 224 MB | 896 MB | 3.5 GB |
| Scan carry total (×58) | **12.7 GB** | **50.8 GB** | **204 GB** ✗ |

## MoE compute per device (per layer, forward)

ragged_dot: (max_local, D_local) × (E_local+1, D_local, F_local) → (max_local, F_local)

| | A | B | C |
|---|---|---|---|
| max_local (T_fsdp×K/EP) | 262144 | 262144 | 262144 |
| Gate matmul | (262K, 3584) × (257, 3584, 8) | (262K, 3584) × (65, 3584, 32) | (262K, 3584) × (17, 3584, 128) |
| Gate FLOP | 2×262K×3584×8 = **15.0 GFLOP** | 2×262K×3584×32 = **60.1 GFLOP** | 2×262K×3584×128 = **240 GFLOP** |
| Up matmul | same | same | same |
| Down matmul | (262K, 8) × (257, 8, 3584) | (262K, 32) × (65, 32, 3584) | (262K, 128) × (17, 128, 3584) |
| Down FLOP | 2×262K×8×3584 = **15.0 GFLOP** | 2×262K×32×3584 = **60.1 GFLOP** | 2×262K×128×3584 = **240 GFLOP** |
| **Total MoE FLOP/dev** | **45 GFLOP** | **180 GFLOP** | **720 GFLOP** |
| MXU efficiency (F_local) | F=8 → **very poor** | F=32 → **poor** | F=128 → **good** |

Note: total cluster FLOP is the same (each device does 1/FSDP of the work, but FSDP devices
do the same experts with different token batches). The MXU utilization differs because F_local
determines the matmul tile efficiency.

## MoE communication per layer

### Column/row parallel psum (FSDP axis)

Each device computes partial output for its F_local columns, then psum("fsdp") sums all contributions.

| | A | B | C |
|---|---|---|---|
| psum tensor | (T_fsdp, D_local) | (T_fsdp, D_local) | (T_fsdp, D_local) |
| psum size | (32768, 3584) = **224 MB** | (131072, 3584) = **896 MB** | (524288, 3584) = **3.5 GB** |
| psum devices | 256-way | 64-way | 16-way |
| FSDP ICI BW | all ICI (~200 GB/s avg) | y×z plane (200 GB/s) | z-axis (**50 GB/s**) |
| psum time est. | 224/200 ≈ **1.1s** | 896/200 ≈ **4.5s** | 3500/50 ≈ **70s** ✗ |

### EP psum (EP axis) — only configs B and C

Each EP device handles E_local experts. EP psum aggregates partial token contributions.

| | A | B | C |
|---|---|---|---|
| EP psum tensor | — | (T_fsdp, D_local) | (T_fsdp, D_local) |
| EP psum size | — | (131072, 3584) = **896 MB** | (524288, 3584) = **3.5 GB** |
| EP devices | — | 4-way | 16-way |
| EP ICI BW | — | x-axis (200 GB/s) | x×y plane (200 GB/s) |
| EP psum time est. | — | 896/200 ≈ **4.5s** | 3500/200 ≈ **17.5s** |

### TP communication (attention w_out + MoE output AG)

| | A | B | C |
|---|---|---|---|
| AR(tp) per layer | ~1.1 GB | ~1.1 GB | ~1.1 GB |
| TP axis | d2d (~540 GB/s) | d2d | d2d |
| Time est. | ~2 ms | ~2 ms | ~2 ms |

### Attention FSDP AllGather (weight gather)

| | A | B | C |
|---|---|---|---|
| Attn weight AG/layer | 134 MB | 134 MB | 134 MB |
| AG devices | 256-way | 64-way | 16-way |
| FSDP ICI BW | ~200 GB/s | 200 GB/s | 50 GB/s |
| Time est. | 0.67 ms | 0.67 ms | 2.7 ms |

## MoE layer communication summary

| | A | B | C |
|---|---|---|---|
| FSDP psum | 224 MB / 256-way | 896 MB / 64-way | 3.5 GB / 16-way |
| EP psum | — | 896 MB / 4-way | 3.5 GB / 16-way |
| TP comms | ~1.1 GB / d2d | ~1.1 GB / d2d | ~1.1 GB / d2d |
| Attn AG(fsdp) | 134 MB / 256-way | 134 MB / 64-way | 134 MB / 16-way |
| **Total comm/layer** | **~1.5 GB** | **~3.0 GB** | **~8.2 GB** |

## Full step estimates (58 MoE + 3 dense layers, fwd+bwd ≈ 3×)

| | A | B | C |
|---|---|---|---|
| MoE comm (58 layers × 3×) | ~260 GB | ~522 GB | ~1.4 TB |
| MoE compute (58 layers × 3×) | 7.8 TFLOP | 31.3 TFLOP | 125 TFLOP |
| MXU efficiency | very poor (F=8) | poor (F=32) | good (F=128) |
| Scan carry (58 layers) | 12.7 GB | 50.8 GB | 204 GB ✗ |
| Wi padding waste | 13.3 GB | 3.3 GB | 0 GB |
| **Measured TPS/chip** | **1740** | — | — |

## Trade-off Summary

| Metric | A (best TPS so far) | B (middle) | C (best MXU) |
|--------|-------|---|---|
| F_local | 8 | 32 | 128 |
| Padding waste | 13.3 GB | 3.3 GB | 0 |
| MXU efficiency | worst | medium | best |
| Comm volume/layer | lowest (1.5 GB) | 2× (3 GB) | 5.5× (8.2 GB) |
| Scan carry | 12.7 GB (fits) | 50.8 GB (needs offload) | 204 GB (impossible) |
| FSDP BW | high (all ICI) | high (y×z) | **low (z-only 50 GB/s)** |

### Key observations

1. **Config C is infeasible**: 204 GB scan carry at GBS=2048, and FSDP on z-axis only (50 GB/s)
   makes the 3.5 GB psum take ~70s/layer. Dead on arrival.

2. **Config B is the interesting middle ground**:
   - F_local=32 reduces padding from 93.8% → 75% (saves 10 GB)
   - 4× better MXU than A (F=32 vs F=8)
   - But 2× more comm volume (psum + EP psum)
   - Scan carry 50.8 GB → needs host offload (proven to work in v258)
   - EP=4 introduces A2A-like routing (each device handles 64 of 256 experts)

3. **Config A wins on simplicity**: EP=1 = no token dispatch, no EP routing overhead.
   The low MXU is compensated by minimal communication. 1740 TPS/chip is hard to beat.

4. **For GBS=4096**: Config A carry doubles to 25.4 GB (still fits at FSDP=256),
   but the padding waste (13.3 GB) eats into headroom. Config B carry at GBS=4096
   = 101 GB → needs aggressive offload.

### Recommendation

Stay with **Config A** (EP=1, FSDP=256, TP=2) for now — it's proven at 1740 TPS/chip.
For GBS=4096, fix the carry offload formula (the bug we found earlier) rather than
changing the sharding config. The 13.3 GB padding waste is tolerable if carry is offloaded.

Config B worth exploring later if MXU becomes the bottleneck after communication
optimizations are exhausted.
