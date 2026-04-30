# DeepSeek-v3 671B: Parametric Roofline Analysis

**Goal:** Estimate theoretical performance limits for training on a 4x8x8 v7x slice (256 chips, 512 cores). Then build a mini version for iteration on 4x4x4.

---

## 1. Model Architecture (from deepseek3-671b-2dfsdp.yml)

### Global parameters
| Parameter | Value |
|-----------|-------|
| `base_emb_dim` (D) | 7168 |
| `base_num_decoder_layers` (L) | 61 (3 dense + 58 MoE) |
| `first_num_dense_layers` | 3 |
| `vocab_size` (V) | 129280 |
| `max_target_length` (S) | 4096 |
| `base_num_query_heads` (H_q) | 128 |
| `base_num_kv_heads` (H_kv) | 128 |

### MoE parameters
| Parameter | Value |
|-----------|-------|
| `num_experts` (E) | 256 |
| `num_experts_per_tok` (K) | 8 |
| `shared_experts` | 1 |
| `base_moe_mlp_dim` (D_moe) | 2048 |
| `base_mlp_dim` (D_mlp) | 18432 (dense layers only) |
| `routed_scaling_factor` | 2.5 |

### MLA (Multi-head Latent Attention)
| Parameter | Value |
|-----------|-------|
| `attention_type` | mla |
| `q_lora_rank` (R_q) | 1536 |
| `kv_lora_rank` (R_kv) | 512 |
| `qk_nope_head_dim` (d_nope) | 128 |
| `qk_rope_head_dim` (d_rope) | 64 |
| `v_head_dim` (d_v) | 128 |

Derived: `qk_head_dim = d_nope + d_rope = 192`

### Parallelism (4x8x8 v7x, 256 chips)
| Axis | Size | Purpose |
|------|------|---------|
| `data` | 1 | Data parallelism |
| `stage` | 1 | Pipeline (unused) |
| `fsdp` | 128 | Weight sharding |
| `fsdp_transpose` | 1 | (unused at size 1) |
| `expert` | 4 | Expert parallelism |
| `context` | 1 | Context parallelism |

Total: 1 × 1 × 128 × 1 × 4 × 1 = 512 devices (256 chips × 2 cores)

---

## 2. Hardware Specs: v7x (Ironside)

| Spec | Value | Notes |
|------|-------|-------|
| Peak MXU BF16 | 2,307 TFLOP/s per chip | Both cores combined |
| HBM capacity | ~95 GB usable | From profile |
| HBM BW peak | 7,373 GB/s per chip | TODO: confirm uni/bidirectional |
| VMEM per core | 64 MB | |
| Cores per chip | 2 (megacore) | |
| ICI links | 6 per chip | 3D torus, 2 per axis |
| ICI BW per link | 80 GB/s | Unidirectional |
| ICI total per chip | 480 GB/s | 6 × 80 GB/s |
| Ridge point (MXU/HBM) | 313 FLOP/byte | 2307e12 / 7373e9 |

### HBM BW scaling with transfer size
| Transfer size | Achieved BW (approx) |
|---------------|---------------------|
| < 1 KB | ~10% of peak |
| 1-10 KB | ~30% of peak |
| 10 KB - 1 MB | ~60% of peak |
| 1-10 MB | ~90% of peak |
| > 10 MB | ~100% of peak |

### ICI topology: 4x8x8 torus
- X axis: 4 chips, torus (wrap-around)
- Y axis: 8 chips, torus
- Z axis: 8 chips, torus
- Each chip: 2 links per axis (one in each direction)
- Bisection bandwidth per axis: 2 × N/2 × 80 GB/s (N = axis size)

---

## 3. Per-Layer Operation Breakdown

### 3.1 MLA Attention (all 61 layers)

For batch B tokens of sequence length S:
- B = PDBS × S (tokens per device before EP gather)
- After EP all-gather: B_ep = B × EP = B × 4

**Query projection (LoRA):**
```
wq_a: [D, R_q] = [7168, 1536]
    FLOPs: 2 × B × D × R_q = 2 × B × 7168 × 1536
    Bytes: B × D × 2 + R_q × D × 2 + B × R_q × 2 (activations + weights + output)

q_norm: RMSNorm on [B, R_q]
    FLOPs: ~5 × B × R_q (negligible)

wq_b: [R_q, H_q × (d_nope + d_rope)] = [1536, 128 × 192] = [1536, 24576]
    FLOPs: 2 × B × R_q × H_q × qk_head_dim = 2 × B × 1536 × 24576
```

**KV projection (LoRA):**
```
wkv_a: [D, R_kv + d_rope] = [7168, 576]
    FLOPs: 2 × B × D × 576

kv_norm: RMSNorm on [B, R_kv]
    FLOPs: ~5 × B × R_kv

wkv_b: [R_kv, H_kv × (d_nope + d_v)] = [512, 128 × 256] = [512, 32768]
    FLOPs: 2 × B × 512 × 32768
```

**RoPE:**
```
Apply YaRN to q_pe [B, H, d_rope] and k_rope [B, 1, d_rope]
FLOPs: ~4 × B × H × d_rope (sin/cos multiply) — negligible
```

**Attention (Splash):**
```
QK^T: [B, H, S, S] — but Splash uses blocked attention
    FLOPs: 2 × B × H × S × S × qk_head_dim (per head: 2 × S² × 192)
    Total: 2 × B × 128 × S² × 192

Softmax: ~5 × B × H × S²

AV: 2 × B × H × S² × d_v = 2 × B × 128 × S² × 128
```

**Output projection:**
```
out: [H × d_v, D] — but MLA compresses this
    FLOPs: 2 × B × H × d_v × D = 2 × B × 128 × 128 × 7168
```

### 3.2 Dense MLP (layers 0-2 only)

```
wi_gate: [D, D_mlp] = [7168, 18432]
    FLOPs: 2 × B × D × D_mlp

wi_up: [D, D_mlp] = [7168, 18432]
    FLOPs: 2 × B × D × D_mlp

SiLU activation: ~2 × B × D_mlp
Element-wise multiply: B × D_mlp

wo: [D_mlp, D] = [18432, 7168]
    FLOPs: 2 × B × D_mlp × D
```

### 3.3 Routed MoE (layers 3-60, 58 layers)

**Router:**
```
Gate logits: [D, E] = [7168, 256]
    FLOPs: 2 × B × D × E = 2 × B × 7168 × 256

Top-k selection: O(B × E × K) — negligible
Sigmoid + normalize: O(B × K) — negligible
```

**Expert dispatch (all-gather across EP):**
```
All-gather activations: B × D × 2 bytes × (EP-1)/EP
All-gather routing info: B × K × 4 bytes × (EP-1)/EP
Sort by expert: O(B × EP × K × log(E))
```

**Routed expert MLP (per device handles E/EP = 64 experts):**
```
Tokens per device after gather: B_total = B × EP × K = B × 4 × 8 = 32B
(but distributed across 64 local experts)

wi_0 (gate): [E_local, D, D_moe] where weights are [256, 7168, 512] per device
    FLOPs: 2 × B_total × D × D_moe = 2 × 32B × 7168 × 2048
    Note: D_moe is actually 2048, but per-device weight has D_moe/EP = 512
    So actual: 2 × 32B × 7168 × 512 per device

wi_1 (up): same as wi_0

SiLU(gate) × up: 2 × B_total × D_moe (per device: × 512)

wo (down): [E_local, D_moe, D] where weights are [256, 512, 7168] per device
    FLOPs: 2 × 32B × 512 × 7168

Weighted combine: B_total × D
```

**Shared expert:**
```
shared_wi_0: [D, D_moe] = [7168, 2048]
    FLOPs: 2 × B × D × D_moe

shared_wi_1: [D, D_moe] = [7168, 2048]
    FLOPs: 2 × B × D × D_moe

shared_wo: [D_moe, D] = [2048, 7168]
    FLOPs: 2 × B × D_moe × D

shared_gate: sigmoid([D, 1])
    FLOPs: ~B × D (negligible)
```

**Expert undispatch:**
```
Unsort + reduce-scatter: B_total × D × 2 bytes × (EP-1)/EP
```

### 3.4 Layer norms (every layer)
```
Pre-attention RMSNorm: ~5 × B × D
Post-attention RMSNorm: ~5 × B × D
```

### 3.5 Embedding + output head
```
Embedding lookup: B × D (memory only, no FLOPs)
Output projection: [D, V] = [7168, 129280]
    FLOPs: 2 × B × D × V
```

---

## 4. FLOP Counting (Forward Pass)

Let B = PDBS × S = tokens per device per step.

### Per MLA attention layer:
```
q_proj_a:     2 × B × 7168 × 1536     = 22,020,096 × B
q_proj_b:     2 × B × 1536 × 24576    = 75,497,472 × B
kv_proj_a:    2 × B × 7168 × 576      =  8,257,536 × B
kv_proj_b:    2 × B × 512 × 32768     = 33,554,432 × B
attention_qk: 2 × B × 128 × S × 192  = 49,152 × B × S
attention_av: 2 × B × 128 × S × 128  = 32,768 × B × S
out_proj:     2 × B × 128 × 128 × 7168 = ... wait, need to reconsider
```

TODO: Complete the FLOP table parametrically for PDBS = {1, 2, 4, 8}

---

## 5. Collective Communication Model

### ICI topology: 4x8x8 torus
- 6 links per chip, 80 GB/s each (unidirectional)
- Total bisection BW per axis depends on axis size and algorithm

### All-reduce on a torus axis of size N:
- Ring all-reduce: 2 × (N-1)/N × M bytes, latency: 2 × (N-1) × α
- For axis size 4: overhead factor = 2 × 3/4 = 1.5
- For axis size 8: overhead factor = 2 × 7/8 = 1.75
- Time = M × factor / BW_per_link + latency

### All-gather on a torus axis of size N:
- Ring: (N-1)/N × M bytes
- Time = M × (N-1)/N / BW_per_link + (N-1) × α

### Reduce-scatter on a torus axis of size N:
- Ring: (N-1)/N × M bytes (same as all-gather)

### For FSDP (axis size 128 = mapped across all 3 physical axes):
The FSDP axis spans the entire 4×8×8 = 256 chip mesh.
All-gather for FSDP weight reconstruction uses multi-axis collectives.

### For EP (axis size 4):
EP=4 maps to one axis of the torus (likely X axis with 4 chips).
All-gather/reduce-scatter: 3/4 × M at 80 GB/s per link.

---

## 6. Memory Model

### Per-chip weight memory (with FSDP=128, EP=4):

**MLA weights per layer (FSDP-sharded):**
```
wq_a: [7168, 1536] / FSDP = [7168/128, 1536] × 2B = ~172 KB
wq_b: [1536, 24576] / FSDP = ...
kv_a, kv_b, out: similar
Total MLA per layer: ~X MB (FSDP sharded)
```

**MoE weights per layer (from profile: [256, 7168, 512] per device):**
```
wi_0: [256, 7168, 512] × 2B = 1.75 GB
wi_1: same = 1.75 GB
wo:   [256, 512, 7168] × 2B = 1.75 GB
Total routed per layer: 5.25 GB
```

**Shared expert per layer:**
```
shared_wi_0, shared_wi_1, shared_wo: ~84 MB total (FSDP sharded)
```

**Optimizer states (Adam: params + momentum + variance = 3×):**
Per-device param size × 3 × dtype_size

---

## TODO: Complete sections

- [ ] Parametric FLOP table for PDBS = {1, 2, 4, 8}
- [ ] Backward pass FLOP multiplier (typically ~2× forward for matmuls, 1× for attention)
- [ ] Memory budget: weights + optimizer + activations + KV cache (training doesn't use KV cache)
- [ ] Collective schedule: when does each collective happen, what overlaps with what
- [ ] Roofline time estimate per component
- [ ] Mini model spec for 4x4x4 iteration
