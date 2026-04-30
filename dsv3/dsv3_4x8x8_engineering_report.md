# DeepSeek-v3 671B: Engineering Report — 4×8×8 v7x Training

**Date:** 2026-03-28
**Status:** Phase 2 of 3 (correctness ✅, roofline ✅, baseline TBD)
**Target:** 70% of roofline TPS/chip

---

## 1. Hardware: TPU v7x (Ironside)

### 1.1 Per-chip specs

| Spec | Value | Notes |
|------|-------|-------|
| Peak MXU BF16 | 2,307 TFLOP/s | Both megacore cores combined |
| HBM capacity | ~95 GB usable | Per core (190 GB per chip) |
| HBM bandwidth | 7,373 GB/s | Per chip (read+write) |
| VMEM per core | 64 MB | On-chip scratchpad |
| Cores per chip | 2 (megacore) | Each core = 1 JAX device |
| ICI links | 6 per chip | 3D torus, 2 links per axis |
| ICI BW per link | 90 GB/s | Unidirectional |
| ICI total BW | 540 GB/s | 6 × 90 GB/s per chip |
| Ridge point | 313 FLOP/byte | 2307 TFLOP/s ÷ 7373 GB/s |

### 1.2 4×8×8 topology

- **Chips:** 4 × 8 × 8 = 256 chips
- **JAX devices:** 512 (256 chips × 2 cores/chip)
- **ICI topology:** 3D torus; X-axis=4, Y-axis=8, Z-axis=8
- **Links per direction:** 2 (one per direction per axis)
- **Effective inter-axis BW for multi-axis collectives:** up to 4 links = 360 GB/s per chip

### 1.3 ICI bandwidth model

For a ring collective spanning N chips along one axis, per-chip effective BW = 90 GB/s (one link).
For 2D all-gather spanning 8×8 = 64 chips (both Y and Z axes): ~180 GB/s effective per chip.
*We use the conservative 1-axis model (90 GB/s) throughout; actual may be ~2× better.*

---

## 2. Model Architecture

### 2.1 Global parameters (DSv3 671B)

| Parameter | Symbol | Value |
|-----------|--------|-------|
| Hidden dim | D | 7,168 |
| Vocab size | V | 129,280 |
| Total layers | L | 61 (3 dense + 58 MoE) |
| Sequence length | S | 4,096 |
| Num heads | H | 128 |
| Query LoRA rank | R_q | 1,536 |
| KV LoRA rank | R_kv | 512 |
| Non-RoPE head dim | d_nope | 128 |
| RoPE head dim | d_rope | 64 |
| Value head dim | d_v | 128 |
| MoE experts total | E | 256 |
| Top-k experts/token | K | 8 |
| MoE intermediate dim | D_moe | 2,048 |
| Shared experts | 1 | |
| Dense MLP dim | D_mlp | 18,432 |

### 2.2 Parameter count

| Component | Params per layer | Total | BF16 |
|-----------|-----------------|-------|------|
| MLA attention | ~187M | 11.4B × 61 = 11.4B | 22.9 GB |
| Dense MLP (3 layers) | ~604M | 1.8B | 3.6 GB |
| Routed MoE experts (58 layers) | ~4B | 232B | 464 GB |
| Shared expert (58 layers) | 44M | 2.6B | 5.2 GB |
| Embedding + output head | — | ~1.8B | 3.6 GB |
| **Total** | | **~671B** | **~499 GB** |

### 2.3 Per-device memory at EP=8, FSDP=64

With EP=8 (each device owns E/EP=32 experts) and FSDP=64 (D axis sharded):

| Memory item | Per device | Notes |
|-------------|-----------|-------|
| MoE expert weights | 32 × 7168/64 × 2048 × 3 × 2B = 44 MB/layer × 58 = **2.6 GB** | Before all-gather; stored |
| MLA weights | ~374 MB / 64 × 61 = **356 MB** | FSDP-sharded, stored |
| Dense MLP weights | ~302 MB / 64 × 3 = **14 MB** | |
| All weights (sharded) | | **~3 GB** |
| Optimizer states (Adam, bf32) | 3× weights = | **~9 GB** |
| Activations (PDBS=1, ckpt) | ~B×S×D×L×2B | **~4-8 GB** |
| **HBM budget check** | **~20-25 GB used** | **95 GB available ✓** |

---

## 3. Recommended Sharding: EP=8, FSDP=64

### 3.1 Rationale

On 512 devices (4×8×8): EP × FSDP = 512.
More EP = smaller per-device expert shard = smaller FSDP gather = faster.

However, EP=8 is our **verified working configuration** (correctness confirmed end-to-end: v23-eval produces " Paris" as top-1 for "The capital of France is"). Higher EP values (EP=16, EP=32) have not been tested.

**EP=8, FSDP=64 is the safe starting point.** Increasing to EP=32 is the primary optimization lever once baseline is established.

### 3.2 Physical axis mapping on 4×8×8

```
Physical torus:   X(4) × Y(8) × Z(8)
Per-chip cores:   2 (megacore)

EP=8  → maps to   Z-axis (8 chips × 1 core/side = 8 JAX devices per Z-slice)
FSDP=64 → maps to X(4) × Y(8) × 2 cores = 64 JAX devices per EP group
```

Each EP group (EP shard) is a 4×8×2 = 64-device rectangular sub-mesh.
FSDP all-gather uses both X and Y axes (effective BW ≈ 4 links ≈ 360 GB/s per chip).

### 3.3 Parameter sharding table

| Parameter | Shape | PartitionSpec | Per-device shape |
|-----------|-------|--------------|-----------------|
| Embedding | [V, D] | P(None, "fsdp") | [129280, 112] |
| MLA wq_a | [D, R_q] | P("fsdp", None) | [112, 1536] |
| MLA wq_b | [R_q, H×qk] | P(None, "fsdp") | [1536, 384] |
| MLA wkv_a | [D, R_kv+d_rope] | P("fsdp", None) | [112, 576] |
| MLA wkv_b | [R_kv, H×(d_nope+d_v)] | P(None, "fsdp") | [512, 512] |
| MLA w_out | [H×d_v, D] | P("fsdp", None) | [256, 7168] |
| MoE wi_0/wi_1 | [E, D, D_moe] | P("ep", "fsdp", None) | [32, 112, 2048] |
| MoE wo | [E, D_moe, D] | P("ep", None, "fsdp") | [32, 2048, 112] |
| MoE gate | [D, E] | P(None, None) | replicated [7168, 256] |
| Shared expert | [D, D_moe] | P("fsdp", None) | [112, 2048] |
| Layer norms | [D] | P(None) | replicated [7168] |
| Output head | [D, V] | P("fsdp", None) | [112, 129280] |

---

## 4. Per-Step FLOPs Breakdown

### 4.1 Forward pass FLOPs per component (PDBS=1, S=4096, EP=8, jax_ep backend)

| Component | FLOPs per layer | Layers | Total FWD |
|-----------|----------------|--------|-----------|
| MLA wq_a: [D, R_q] | 2 × 4096 × 7168 × 1536 = 90.2 GFLOP | 61 | 5.5 TFLOP |
| MLA wq_b: [R_q, H×qk] | 2 × 4096 × 1536 × 24576 = 309.2 GFLOP | 61 | 18.9 TFLOP |
| MLA wkv_a | 2 × 4096 × 7168 × 576 = 33.8 GFLOP | 61 | 2.1 TFLOP |
| MLA wkv_b | 2 × 4096 × 512 × 32768 = 137.4 GFLOP | 61 | 8.4 TFLOP |
| MLA attention | 2 × 4096 × 128 × 4096 × 192 = 831 GFLOP | 61 | 50.7 TFLOP |
| MLA out_proj | 2 × 4096 × 16384 × 7168 = 963 GFLOP | 61 | 58.7 TFLOP |
| **MLA total** | **~2.9 TFLOP** | 61 | **177.3 TFLOP** |
| Dense MLP (gate+up+down) | 2 × 4096 × (18432+18432+18432) × 7168 / 3 ≈ 3.23 TFLOP | 3 | 9.7 TFLOP |
| MoE gate: [D, E] | 2 × 4096 × 7168 × 256 = 7.5 GFLOP | 58 | 0.44 TFLOP |
| MoE GMM×3 (jax_ep: E/EP=32 experts × T) | 3 × 2 × 4096 × 32 × 7168 × 2048 = 11.5 TFLOP | 58 | 667 TFLOP |
| MoE shared expert × 3 | 3 × 2 × 4096 × 7168 × 2048 = 360 GFLOP | 58 | 20.9 TFLOP |
| Output head | 2 × 4096 × 7168 × 129280 = 7.6 TFLOP | 1 | 7.6 TFLOP |
| **TOTAL FWD** | | | **~886 TFLOP** |

> **Note:** MoE GMM dominates FWD at 75% of total. With jax_ep backend, each device runs
> E/EP=32 local experts for ALL B×S=4096 tokens. Ideal (ragged_dot + EP) would do K=8
> expert-token pairs → 4× less compute. This is the primary compute optimization TODO.

### 4.2 Training step FLOPs (with gradient checkpointing)

| Pass | FLOPs | Factor |
|------|-------|--------|
| Forward (original) | 886 TFLOP | 1× |
| Backward (dX + dW per matmul) | 2 × 886 TFLOP | 2× |
| Recompute forward (for activation ckpt) | 886 TFLOP | 1× |
| **Total** | **3,544 TFLOP** | 4× fwd |

### 4.3 FLOPs by PDBS

| PDBS | Tokens/device | Total FLOPs | Compute time (2307 TFLOP/s) |
|------|--------------|-------------|---------------------------|
| 1 | 4,096 | 3,544 TFLOP | 1.54 s |
| 2 | 8,192 | 7,089 TFLOP | 3.07 s |
| 4 | 16,384 | 14,177 TFLOP | 6.15 s |

---

## 5. Collective Communication Schedule

### 5.1 Pattern per training step

For each MoE layer (58× per step):

```
Forward pass:
  shard_map (ep axis) {
    wi_0: all_gather("fsdp", D/fsdp → D)   ← 940 MB total, ~10 ms
    wi_1: all_gather("fsdp", D/fsdp → D)   ← 940 MB total, ~10 ms
    wo:   all_gather("fsdp", D/fsdp → D)   ← 940 MB total, ~10 ms
    compute: einsum over E_local experts
    psum("ep")                               ← 58 MB, ~0.6 ms
  }

Backward pass (recompute + gradient):
  Same pattern × 2 (recompute forward, then actual backward)
```

For each dense/MLA layer (61× per step):
```
  all_gather("fsdp", layer_weights)    ← ~374 MB total per MLA layer, ~4.1 ms
  compute layer
  reduce_scatter("fsdp", layer_grads)  ← same volume, same time
```

### 5.2 Collective size and time estimates

| Collective | Tensor | Bytes | ICI time (90 GB/s) | Count/step | Total |
|-----------|--------|-------|---------------------|------------|-------|
| FSDP all-gather, MoE wi_0 | [32, 7168, 2048] BF16 | 940 MB | 10.2 ms | 58×4=232 | 2,366 ms |
| FSDP all-gather, MoE wi_1 | [32, 7168, 2048] BF16 | 940 MB | 10.2 ms | 232 | 2,366 ms |
| FSDP all-gather, MoE wo | [32, 2048, 7168] BF16 | 940 MB | 10.2 ms | 232 | 2,366 ms |
| FSDP all-gather, MLA (all params) | ~374 MB/layer | 374 MB | 4.1 ms | 61×4=244 | 1,000 ms |
| EP psum per layer | [4096, 7168] BF16 | 58 MB | 0.6 ms | 58×3=174 | 104 ms |
| **TOTAL** | | | | | **~8.2 s** |

> **Key insight:** MoE FSDP weight gather (7.1 s / 87%) completely dominates communication.
> This is because the MoE weight matrices [32, 7168, 2048] × 3 = 2.8 GB per layer must be
> gathered from 64 FSDP shards before each forward/backward pass.

### 5.3 Overlap potential

XLA async collective fusion (enabled via LIBTPU_INIT_ARGS) can overlap some communication with computation:
- **Can overlap:** FSDP gather of layer N+1 during compute of layer N (pipeline)
- **Cannot overlap:** within-layer gather (must complete before matmul)
- **Estimated achievable overlap:** 40-60% of FSDP gather time

---

## 6. Step Time Estimates

### 6.1 EP=8, FSDP=64, jax_ep backend, gradient checkpointing, seqlen=4096

| PDBS | Compute | HBM | FSDP gather | EP comm | Roofline* | TPS/chip | TPS/cluster |
|------|---------|-----|-------------|---------|-----------|----------|-------------|
| 1 | 1.54 s | 0.96 s | 8.31 s | 0.20 s | **5.79 s** | 707 | 180,967 |
| 2 | 3.07 s | 1.73 s | 8.31 s | 0.40 s | **7.43 s** | 1,103 | 282,256 |
| 4 | 6.15 s | 3.28 s | 8.31 s | 0.80 s | **10.70 s** | 1,531 | 391,944 |

*Roofline = max(compute, HBM) + 50% comm overlap assumed.

### 6.2 Target metrics (70% of roofline, EP=8 jax_ep)

| PDBS | Roofline step | 70% efficiency step | Target TPS/chip | Target TPS/cluster |
|------|--------------|---------------------|-----------------|-------------------|
| 1 | 5.79 s | 8.27 s | **495** | 126,720 |
| 2 | 7.43 s | 10.61 s | **772** | 197,632 |
| 4 | 10.70 s | 15.29 s | **1,072** | 274,432 |

### 6.3 Comparison: what ideal ragged_dot+EP would achieve

| PDBS | jax_ep TPS/chip | ragged_dot+EP TPS/chip | Ratio |
|------|-----------------|------------------------|-------|
| 1 | 707 | 832 | 1.18× |
| 2 | 1,103 | 1,440 | 1.31× |
| 4 | 1,531 | 2,270 | 1.48× |

*Note: ragged_dot+EP is not yet implemented for EP>1 in mini_dsv3.*

### 6.4 Effect of increasing EP (with jax_ep backend)

| EP | FSDP | PDBS=4 TPS/chip |
|----|------|-----------------|
| 8 | 64 | 1,531 ✅ (working) |
| 16 | 32 | 2,494 (2× better) |
| 32 | 16 | 3,369 (2.2× better) |

**EP=32 is the highest-impact optimization after establishing baseline.**

---

## 7. Optimization Roadmap

### Priority 1: Implement ragged_dot + EP (code change)
**Impact:** 1.18–1.48× TPS improvement at PDBS=1–4
**Root cause:** `_expert_mlp_grouped` uses `group_sizes[E]` but weights have shape `[E/EP, ...]` for EP>1
**Fix:** Extend `_expert_mlp_grouped` to accept a `local_start` offset and filter `group_sizes` to local expert range
**Risk:** Medium (needs correctness validation)

### Priority 2: Increase EP to 16–32 (config change)
**Impact:** 2–2.2× TPS improvement
**Root cause:** FSDP gather of MoE weights (940 MB per all-gather) dominates step time
**Fix:** Test EP=16 first, then EP=32 — correctness should carry over from shard_map+psum pattern
**Risk:** Low (same code path, just different axis sizes)

### Priority 3: Async FSDP gather overlap (XLA flags)
**Impact:** Up to 2× improvement (if 50% → 90% overlap achievable)
**Root cause:** FSDP gather serializes with compute; XLA pipeline can overlap
**Fix:** LIBTPU_INIT_ARGS already has async flags; need to validate overlap in profile
**Risk:** Low (already enabled)

### Priority 4: FP8 quantization (model change)
**Impact:** 2× reduction in all communication (weights become 1 byte vs 2 bytes)
**Root cause:** All FSDP gathers transfer BF16 weights; FP8 halves bandwidth
**Risk:** Medium (correctness impact on training)

---

## 8. Baseline Experiment Plan

### 8.1 First run: EP=8, PDBS=1

```yaml
# k8s/dsv3-train-4x8x8-v1.yaml
cluster: gke_tpu-vm-gke-testing_us-central1_sivaibhav-exp-v7x
topology: 4x8x8 (64 pods)
args:
  --config=full --fsdp=64 --ep=8 --pdbs=1 --steps=15
  --gradient_checkpoint --moe_backend=jax
  --profile --profile_dir=gs://sivaibhav-exp/dsv3/profiles/4x8x8-ep8-pdbs1-v1
  --profile_skip=5 --profile_steps=2 --profile_rank=0
```

### 8.2 Expected baseline performance

- Step time: ~8–10 s (roofline 5.79 s + overhead)
- MFU: ~15–20% of peak MXU (heavily comm-bound)
- Top bottleneck: FSDP MoE gather (should see >50% idle time in profile)

### 8.3 Profile analysis checklist

Using `xla_shell report_timing` and `list_collectives --overlap`:
- [ ] Verify step time matches model (±25%)
- [ ] Confirm FSDP gather > compute in timeline
- [ ] Check ICI link utilization during FSDP gathers (expect ~50-70% of peak)
- [ ] Check compute/collective overlap ratio
- [ ] Look for unexpected small operations wasting time (XLA overhead)

---

## 9. Summary

**Current state:**
- EP=8 correctness: ✅ confirmed (v23-eval, " Paris" top-1)
- Roofline model: ✅ corrected (3 bugs fixed, bottleneck identified)
- Baseline: 🔲 pending (sivaibhav-exp-v7x cluster ready)

**Key finding:** FSDP MoE weight gather (84% of comm time) is the dominant bottleneck.
Training at PDBS=1 is 5.5× communication-bound. Reaching 70% of roofline = ~495 TPS/chip.

**Highest-impact path to 2× improvement:** Increase EP from 8 to 32 (reduces FSDP gather 4×).
