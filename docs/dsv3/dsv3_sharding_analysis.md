# DSv3 671B — Sharding, Collectives & HBM Analysis

**Config**: EP=32, FSDP=16, 4×8×8 (512 JAX devices, 256 physical chips)
**Target run**: v6, GBS=1024 (1024 seq × 4096 tokens = 4,194,304 global tokens)
**Model**: Full 671B — D=7168, H=128 heads, E=256 experts, K=8, L=3+58=61 layers

---

## 1. Mesh Setup

```
512 JAX devices = dp × ep × fsdp = 1 × 32 × 16

  dp   = 1   (no data parallelism — effectively unused)
  ep   = 32  (expert parallelism — cross-host, inter-chip ICI)
  fsdp = 16  (weight sharding — intra-host, fast ICI)
```

Each device is at position `(dp=0, ep_i, fsdp_j)` for `ep_i ∈ [0,31]`, `fsdp_j ∈ [0,15]`.

**Batch distribution at GBS=1024:**
- Each FSDP group processes `GBS / fsdp = 64 sequences`
- Each EP group replicates those 64 sequences and distributes experts
- Local tokens per device: `64 × 4096 = 262,144`

---

## 2. Weight Sharding

### 2a. MLA Attention (per layer, all 61 layers)

| Weight | Shape | Sharding | Per-device size |
|--------|-------|----------|-----------------|
| `wq_a` | (7168, 1536) | `P("fsdp", None)` | (448, 1536) = 1.37 MB |
| `wq_b` | (1536, 128×192) | `P(None, "fsdp")` | (1536, 1536) = 4.72 MB |
| `q_norm_scale` | (1536,) | `P(None)` | replicated |
| `wkv_a` | (7168, 576) | `P("fsdp", None)` | (448, 576) = 0.52 MB |
| `wkv_b` | (512, 128×256) | `P(None, "fsdp")` | (512, 2048) = 2.10 MB |
| `kv_norm_scale` | (512,) | `P(None)` | replicated |
| `w_out` | (128×128, 7168) | `P("fsdp", None)` | (1024, 448) = 0.92 MB |
| norms | (7168,) × 2 | `P(None)` | replicated |

MLA subtotal per layer per device: **~10 MB** → 61 layers = **~610 MB**

### 2b. Dense MLP (3 dense layers only)

| Weight | Shape | Sharding | Per-device size |
|--------|-------|----------|-----------------|
| `wi_gate` | (7168, 18432) | `P("fsdp", None)` | (448, 18432) = 16.5 MB |
| `wi_up` | (7168, 18432) | `P("fsdp", None)` | (448, 18432) = 16.5 MB |
| `wo_mlp` | (18432, 7168) | `P("fsdp", None)` | (18432, 448) = 16.5 MB |

Dense MLP subtotal: **~50 MB** × 3 layers = **~150 MB**

### 2c. MoE Expert Weights (58 MoE layers)

Each device owns `E/ep = 256/32 = 8 experts`.

| Weight | Shape | Sharding | Per-device shape | Per-device size |
|--------|-------|----------|------------------|-----------------|
| `wi_0` | (256, 7168, 2048) | `P("ep", "fsdp", None)` | (8, 448, 2048) | 14.7 MB |
| `wi_1` | (256, 7168, 2048) | `P("ep", "fsdp", None)` | (8, 448, 2048) | 14.7 MB |
| `wo` | (256, 2048, 7168) | `P("ep", None, "fsdp")` | (8, 2048, 448) | 14.7 MB |
| `gate` | (7168, 256) | `P(None, None)` | (7168, 256) replicated | 3.7 MB |
| `gate_bias` | (256,) | `P(None)` | replicated | negligible |

Expert weights per layer per device: **~44 MB** → 58 layers = **~2,560 MB = 2.5 GB**

### 2d. Shared Expert (per MoE layer)

| Weight | Shape | Sharding | Per-device size |
|--------|-------|----------|-----------------|
| `shared_wi_0` | (7168, 2048) | `P("fsdp", None)` | (448, 2048) = 1.84 MB |
| `shared_wi_1` | (7168, 2048) | `P("fsdp", None)` | (448, 2048) = 1.84 MB |
| `shared_wo` | (2048, 7168) | `P("fsdp", None)` | (2048, 448) = 1.84 MB |

Shared expert per layer: **~5.5 MB** → 58 layers = **~320 MB**

### 2e. Embedding + Output Head

| Weight | Shape | Sharding | Per-device size |
|--------|-------|----------|-----------------|
| `embed` | (129280, 7168) | `P(None, "fsdp")` | (129280, 448) = 116 MB |
| `output_head` | (7168, 129280) | `P("fsdp", None)` | (448, 129280) = 116 MB |

### Parameter Memory Summary (per JAX device)

| Component | Size |
|-----------|------|
| MLA weights (61 layers) | ~610 MB |
| Dense MLP (3 layers) | ~150 MB |
| MoE expert weights (58 layers) | ~2,560 MB |
| Shared experts (58 layers) | ~320 MB |
| Embedding + output | ~232 MB |
| Norms + misc | ~50 MB |
| **Total parameters** | **~3.9 GB** |
| **Gradients (same layout)** | **~3.9 GB** |
| **Params + grads** | **~7.8 GB** |

> Note: The full model is 1342 GB bf16. With 512 devices and mixed FSDP/EP sharding,
> average per-device = 1342/512 = 2.6 GB, but the above estimate (~3.9 GB) accounts
> for partial replication (gate weights, norms, embeddings are replicated across ep axis).

---

## 3. Collectives — What, Where, Why

### 3a. FSDP All-Gather (intra-host, fast ICI)

Triggered whenever a weight with `P("fsdp", None)` is multiplied by an input:
XLA reconstructs the full weight by gathering from all 16 FSDP peers.

| Location | Weight gathered | Data volume per gather |
|----------|----------------|------------------------|
| MLA `q_low = x @ wq_a` | wq_a (7168, 1536) | 7168×1536×2 = 22 MB |
| MLA `kv_low = x @ wkv_a` | wkv_a (7168, 576) | 8.3 MB |
| Dense MLP `x @ wi_gate/up` | wi_gate, wi_up (×2) | 2 × 264 MB = 528 MB |
| Dense MLP `hidden @ wo_mlp` | wo_mlp (18432, 7168) | 264 MB |
| MoE expert (inside shard_map) | wi_0, wi_1, wo per layer | 3 × 14.7 MB×32÷16 = 88 MB |
| Shared expert `x @ shared_wi_*` | shared_wi_0, wi_1 (×2) | 2 × 29 MB = 58 MB |
| Output head `x @ output_head` | output_head (7168, 129280) | 1,849 MB |

**All FSDP all-gathers are intra-host (within 16 devices on same ICI fabric). Fast.**

### 3b. FSDP Reduce-Scatter (intra-host, fast ICI)

Triggered on the output of any einsum that contracts over the FSDP-sharded D dimension.
After the reduce-scatter, the output is again `P("fsdp", None)` sharded on the batch.

| Location | Output shape | Data volume |
|----------|-------------|-------------|
| MLA `attn_flat @ w_out` | (64, 4096, 7168) | 3.76 GB |
| Dense MLP `hidden @ wo_mlp` | (64, 4096, 7168) | 3.76 GB |
| Shared expert `shared_h @ shared_wo` | (64, 4096, 7168) | 3.76 GB |
| Output head `x @ output_head` | (64, 4096, 129280) | 67.6 GB |

Each reduce-scatter at 3.76 GB per layer × (61 MLA + 3 MLP + 58 shared) = 122 layers × 3.76 GB = **459 GB** per forward pass (intra-host, so fast).

**Output head RS is expensive at 67.6 GB — once per step.**

### 3c. EP All-Reduce / psum (cross-host, slow ICI)

The core bottleneck. Happens **once per MoE layer** in `expert_mlp_jax()`:

```python
# Inside shard_map over "ep" axis (model.py:447):
return psum(partial_out, "ep")
# partial_out: (local_tokens=262144, D=7168) per device
```

Each device holds partial expert outputs for its 8 experts. The psum sums across all 32 EP devices so every device gets the complete expert output for all tokens.

| Parameter | Value |
|-----------|-------|
| Tensor shape per device | (262144, 7168) |
| Tensor size | 262144 × 7168 × 2 = **3.76 GB** |
| All-reduce with EP=32 | each device sends/receives ~2 × 3.76 GB = 7.52 GB |
| Number of MoE layers | 58 |
| **Total EP psum traffic per forward pass** | **58 × 7.52 GB ≈ 436 GB** |
| Backward pass (grad of psum = psum) | another **436 GB** |
| **Total per training step** | **~872 GB** |

EP psums cross host boundaries → latency-sensitive. At v5 (GBS=8), EP psum was
7.2% of step at 0% overlap. At GBS=1024 (128× larger batch but same communication
structure), EP psum bytes scale proportionally, but compute scales even more —
so EP psum should be a smaller fraction of step time.

### 3d. No All-to-All

Unlike ring-of-experts (MaxText MoE), this implementation uses `shard_map + psum`:
- Tokens are NOT physically routed to expert devices
- Every device runs ALL tokens through its LOCAL experts (dispatch matrix masks out non-local)
- One psum at the end combines results
- **No all-to-all.** The all-to-all in v1-v5 profiles was routing sync (tiny, ~6ms/step).

---

## 4. HBM Estimate for GBS=1024

### 4a. Static Memory (always resident)

| Component | Per-device |
|-----------|-----------|
| Parameters | ~3.9 GB |
| Gradients | ~3.9 GB |
| Optimizer state (SGD = none) | 0 GB |
| JAX/XLA runtime overhead | ~1.0 GB |
| **Static total** | **~8.8 GB** |

### 4b. Peak Activation Memory (per forward layer, gradient_checkpoint=True)

With `gradient_checkpoint=True`, only the **layer input x** is saved per layer.
Everything else is recomputed in the backward pass.

**Saved between layers (carry):**
- `x`: (64, 4096, 7168) = **3.76 GB** — always live

**Peak during MLA attention (forward recompute):**
- `q`: (64, 4096, 128, 192) = 0.77 GB
- `k, v`: (64, 4096, 128, 192) = 0.77 GB
- `attn_weights` (**current bottleneck**): (64, 4096, 128, 4096) in fp32 = **137 GB** ← OOM cause
- With flash attention: never materialized → **peak ~0.77 GB for QKV**

**Peak during MoE forward:**
- `expert hidden`: (262144, 8, 2048) = 8.59 GB  _(all tokens × local experts × D_moe)_
- `expert_out_all`: (262144, 8, 7168) = 30.1 GB  _(all tokens × local experts × D)_

> The `expert_out_all` tensor at 30 GB is large — this is before the dispatch/combine step.
> May need to fuse or chunk.

**Peak during Dense MLP:**
- `hidden (gate * up)`: (64, 4096, 18432) = 9.66 GB

### 4c. Summary: HBM Budget at GBS=1024

| Component | Size | Notes |
|-----------|------|-------|
| Parameters | 3.9 GB | always |
| Gradients | 3.9 GB | always |
| Layer carry (x) | 3.76 GB | always during fwd/bwd |
| **With naive attention** | +137 GB | **OOM — current blocker** |
| **With flash attention** | +0.77 GB | replaces 137 GB |
| MoE expert_out_all | 30.1 GB | peak during MoE layer |
| Dense MLP hidden | 9.66 GB | peak during dense layer |
| **Peak with flash attention** | **~42 GB** | MoE layer dominates |

**v7x HBM per JAX device: ~48 GB**

With flash attention, peak estimate is ~42 GB — fits, but tight (~87% HBM).
The `expert_out_all` tensor (30 GB) is the next concern after flash attention.

---

## 5. Collective Summary

| Collective | Axis | Type | Volume/step | Topology | Overlap |
|-----------|------|------|-------------|----------|---------|
| Weight all-gather (MLA) | fsdp | AG | ~30 MB/layer | intra-host | can overlap |
| Weight all-gather (MLP) | fsdp | AG | ~528 MB/layer | intra-host | can overlap |
| Weight all-gather (expert) | fsdp | AG | ~88 MB/layer | intra-host | can overlap |
| Activation reduce-scatter | fsdp | RS | 3.76 GB/layer | intra-host | ~99% overlapped (v4) |
| **EP expert psum** | **ep** | **AR** | **7.52 GB/layer × 58** | **cross-host** | **0% overlapped** |
| Output head RS | fsdp | RS | 67.6 GB | intra-host | overlapped |

**Critical bottleneck: EP psum, 0% overlapped, 436 GB per fwd pass.**

---

## 6. Path to High MFU

1. **Flash attention** (immediate blocker): replace `jnp.einsum` in `mla_attention` with
   Splash attention (Pallas). Eliminates 137 GB materialization, unblocks GBS=1024.

2. **MoE expert_out_all chunking**: 30 GB tensor may OOM on tight HBM budget.
   Consider computing expert dispatch in chunks of experts.

3. **EP psum overlap**: Pipeline EP psum with compute using async collectives.
   Already have `--xla_enable_async_all_reduce` — need to restructure scan to expose
   inter-layer overlap.

4. **Larger GBS**: GBS=1024 gives 262k tokens/device. At ~50% MFU:
   - FLOPs/step (global): 22,828,181 TFLOP (from v6 log)
   - Step time at 50% MFU: 22,828,181 / (0.5 × 2307 × 256) = **77 s/step**
   - cluster_TPS: 4,194,304 / 77 = **54,472**
   - TPS/physical chip: 54,472 / 256 = **213 TPS/chip**

   At 70% MFU: **step ~55s**, **TPS/chip ~299**
