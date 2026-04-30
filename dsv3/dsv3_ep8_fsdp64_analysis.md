# DSv3 671B — EP=8 / FSDP=64 / GBS=1024 Analysis

**Proposed config**: EP=8, FSDP=64, GBS=1024 sequences, 4×8×8 (512 JAX devices, 256 chips)
**Model**: Full 671B — D=7168, H=128, E=256, K=8, L=61 (3 dense + 58 MoE)

---

## 1. Physical Torus Mapping

v7x 4×8×8 is a 3D torus with axis sizes **4 × 8 × 8** in physical chips.
Each chip has **2 cores** (JAX devices), so the effective device grid is:

```
Physical chips : 4  × 8  × 8  = 256 chips
JAX devices   : 8  × 8  × 8  = 512 devices  (one axis doubled by 2 cores/chip)
               OR
               4  × 8  × 16 = 512 devices  (last axis doubled)
               OR
               4  × 16 × 8  = 512 devices  (middle axis doubled)
```

The torus has **intra-axis ICI links**: communication along each torus axis uses the
physical links for that axis — fast and bandwidth-proportional to hop count.
Cross-axis communication is slower (more hops).

### Proposed mesh: `(dp=1, ep=8, fsdp=64)`

```
ep=8   → maps to the physical axis of size 4 (4 chips × 2 cores = 8 JAX devices)
         ⟹ all EP all-reduce hops stay on ONE physical axis — minimum latency

fsdp=64 → maps to the remaining 8×8 = 64 physical locations × (1 core each)
          ⟹ FSDP traffic on the fast 8×8 torus subgrid (well-connected ring)

dp=1   → unused
```

This is good topology hygiene:
- **EP traffic (cross-host, latency-sensitive)** → shortest physical axis (4 hops max)
- **FSDP traffic (weight all-gather/reduce-scatter)** → dense 8×8 local mesh

### Compare: current EP=32, FSDP=16

```
ep=32 → must span all three physical axes (4×8 = 32 chips → 64 JAX devices)
         ⟹ EP all-reduce hops cross multiple axis boundaries — higher latency

fsdp=16 → only 16 devices, 1 physical axis of size 8 × 2 cores
```

EP=8 on the 4-chip axis is strictly better torus alignment.

---

## 2. Batch Sharding — The Key Correction

**FSDP shards the batch dimension** (`act_spec = P("fsdp", None)`, model.py:501).
This means:

| Config | Local sequences/device | Local tokens/device |
|--------|------------------------|---------------------|
| EP=32, FSDP=16, GBS=8 (v4/v5, old code) | 8/16 → **invalid** (used P(None,None)) | 32768 replicated |
| EP=32, FSDP=16, GBS=1024 (v6 attempt) | 1024/16 = **64 seq** | 262,144 tokens |
| **EP=8, FSDP=64, GBS=1024 (proposed)** | 1024/64 = **16 seq** | 65,536 tokens |

> **v4/v5 note**: the v26-train image used `P(None, None)` for activations, so the batch
> was NOT sharded — all 512 devices held the full GBS=8 batch. The v27-train fix introduced
> `P("fsdp", None)`. GBS must be ≥ FSDP for this to work (requires 1024 ≥ 64 ✓).

---

## 3. Ring of Experts — Dispatch Options

### Current approach: replicated tokens + psum

```
All EP devices (same FSDP position) hold IDENTICAL tokens.
Each runs its local experts on ALL tokens (dispatch mask zeros non-local).
psum("ep") at end combines expert contributions.

Collective: 1× allreduce per MoE layer
Volume:     (local_T, D) per device = (65536, 7168) × 2 = 939 MB
```

### Ring-of-experts: true token routing

Tokens are pre-distributed across both EP and FSDP axes:
- Each device starts with `GBS / (ep × fsdp) = 1024 / 512 = 2 sequences = 8192 tokens`
- Two dispatch options to get tokens to expert devices:

#### Option A: All-to-All dispatch

```
Step 1 [all2all]: route tokens to the EP device owning the selected expert
  - Input per device: (8192, D) own tokens, sorted by destination EP device
  - Send to each of 8 EP devices: 8192 × K/E × E_local = 8192 × 8/256 × 32 ≈ 8192 tokens
  - All2all volume: (8192, D) × 8 = 8 × 8192 × 7168 × 2 = 939 MB send + 939 MB receive

Step 2 [local compute]: run received tokens through local 32 experts
  - Received: ~8192 tokens per device (E=256, K=8, E_local=32 → K×E_local/E = 1.0)
  - expert_out: (8192, D_moe=2048) per expert × 32 = (262144, 2048) = 1.07 GB

Step 3 [all2all back]: return outputs to token-owner devices
  - Same volume as Step 1: 939 MB each way

Total per MoE layer: 2 × 939 MB = 1.88 GB  (same as allreduce, but more compute-efficient)
```

**Advantage**: each device only computes K×T/E tokens (vs all T tokens in psum approach).
Compute per device: 8192 × K×E_local/E = 8192 tokens through 32 experts.
No wasted compute on tokens that route elsewhere.

#### Option B: All-Gather dispatch

```
Step 1 [all-gather]: broadcast all tokens to all EP devices
  - Each device gathers from 7 peers: 7/8 × 65536 × 7168 × 2 = 820 MB received
  - After gather: each device has ALL 65536 tokens (8 × 8192)

Step 2 [local compute]: run ALL 65536 tokens through local 32 experts
  - Wasteful: runs tokens that don't route to local experts (masked to 0)
  - expert_out_all: (65536, 32, D_moe) = 65536 × 32 × 2048 × 2 = 8.59 GB  ← large

Step 3 [reduce-scatter]: shard output back
  - 820 MB

Total per MoE layer: ~1.64 GB comm, but 8× more compute than all2all
```

**Advantage**: simpler to implement; no token-routing bookkeeping.
**Disadvantage**: 8× wasted expert computation, large intermediate tensor.

#### Recommendation: All-to-All for EP=8

At EP=8, all2all is strongly preferred:
- Same communication volume as all-gather
- 8× less expert compute wasted
- Smaller intermediates (8192 tokens vs 65536)
- At EP=8, all2all is simple — only 8-way exchange

For future larger EP (EP=32+), all-gather may be better if the all2all routing
overhead (token permutation, variable bucket sizes) dominates.

### Configurable dispatch interface

```python
def moe_expert_dispatch(mode: str):
    # mode = "all2all" | "all_gather" | "psum" (current)
    if mode == "all2all":
        tokens_recv = lax.all_to_all(tokens_sorted, "ep", ...)
        out = run_local_experts(tokens_recv)
        return lax.all_to_all(out, "ep", ...)   # scatter back
    elif mode == "all_gather":
        tokens_all = lax.all_gather(tokens, "ep", ...)
        out_all = run_local_experts(tokens_all)  # wasteful but simple
        return lax.psum_scatter(out_all, "ep", ...)
    else:  # psum (current)
        out = run_local_experts(tokens)  # tokens already replicated
        return lax.psum(out, "ep")
```

---

## 4. Communication Volume Comparison

### Per-MoE-layer, per forward pass

| Config | EP psum/AR volume | All2all option | FSDP RS volume |
|--------|-------------------|----------------|----------------|
| EP=32 FSDP=16 GBS=8 (v5, no batch shard) | 469 MB | N/A | 469 MB × 61 = 28 GB |
| EP=32 FSDP=16 GBS=1024 (v6, with batch shard) | 3.76 GB | N/A | 939 MB × 61 = 57 GB |
| **EP=8 FSDP=64 GBS=1024 (proposed, psum)** | **939 MB** | — | **235 MB × 61 = 14 GB** |
| **EP=8 FSDP=64 GBS=1024 (proposed, all2all)** | — | **2 × 939 MB** | **235 MB × 61 = 14 GB** |

> FSDP RS volume per layer = (local_T, D) = (local_seq × S, D) × 2 bytes
> EP=8 FSDP=64: (16 × 4096, 7168) × 2 = 939 MB per layer → ÷4 vs EP=32 (smaller local batch)

**Per full forward pass (58 MoE layers):**
- EP=32 FSDP=16 GBS=1024: EP AR = 58 × 3.76 GB = **218 GB** (cross-host!)
- **EP=8 FSDP=64 GBS=1024**: EP AR = 58 × 939 MB = **53 GB** (cross-host, shorter axis)

EP=8 gives **4× less EP communication** than EP=32 at the same GBS.

---

## 5. HBM Estimate at EP=8, FSDP=64, GBS=1024

### Static memory (per JAX device)

| Component | FSDP=16 (current) | FSDP=64 (proposed) |
|-----------|-------------------|---------------------|
| Non-expert params (FSDP-sharded) | ~31 GB | ~7.8 GB |
| Expert params (EP×FSDP-sharded) | ~2.5 GB | ~2.5 GB (unchanged — EP×FSDP=512 constant) |
| Gradients | same as params | same as params |
| **Total static** | **~67 GB** | **~20.6 GB** |

FSDP=64 shards the D dimension 4× more aggressively → 4× less non-expert weight memory.
Expert weights unchanged since EP×FSDP product = 512 is the same.

v7x HBM: **~48 GB per JAX device** (96 GB per physical chip / 2 cores)

→ EP=32 FSDP=16 is at **~67 GB of 48 GB = OOM for parameters alone!**
→ EP=8 FSDP=64 is at **~20.6 GB of 48 GB = 43% utilization** for static memory ✓

> Note: EP=32 FSDP=16 worked in earlier runs (v4/v5) because those were GBS=4/8 where
> activations were tiny. With GBS=1024 the activation memory would also blow up.

### Peak activation memory (per JAX device, with gradient_checkpoint)

| Tensor | EP=32 FSDP=16 GBS=1024 | EP=8 FSDP=64 GBS=1024 |
|--------|------------------------|------------------------|
| Layer carry x | (64, 4096, 7168) = **3.76 GB** | (16, 4096, 7168) = **939 MB** |
| Attn (naive) | (64×128×4096×4096) fp32 = **137 GB** ← OOM | (16×128×4096×4096) fp32 = **34 GB** ← still OOM |
| Attn (flash) | ~0.19 GB | ~0.05 GB ✓ |
| MoE all2all recv | (65536, 7168) = 939 MB | (8192, 7168) = **235 MB** |
| Expert hidden | (65536, 2048) × 32 = 4.3 GB | (8192, 2048) × 32 = **1.07 GB** |
| Expert out | (65536, 7168) × 32 = 30 GB | (8192, 7168) × 32 = **7.5 GB** |

> Flash attention is required regardless of config — naive attention OOMs at both.
> With all2all dispatch, the `expert_out_all` tensor shrinks from 30 GB to 7.5 GB.

### Total HBM budget (with flash attention + all2all dispatch)

| Component | EP=32 FSDP=16 | EP=8 FSDP=64 |
|-----------|---------------|---------------|
| Static (params + grads) | ~67 GB | **~20.6 GB** |
| Layer carry | 3.76 GB | **0.94 GB** |
| Flash attention | 0.19 GB | **0.05 GB** |
| MoE expert_out peak | 30 GB | **7.5 GB** |
| Misc overhead | ~1 GB | **~1 GB** |
| **Total peak** | **~102 GB** ❌ | **~30 GB ✓** |

EP=8 FSDP=64 fits comfortably: **30 GB of 48 GB = 63% HBM utilization**.

---

## 6. Compute Analysis at GBS=1024

From v6 log: `Estimated FLOPs/step (global): 22,828,181 TFLOP`

With 256 physical chips at 2307 TFLOP/s peak each:
- At 50% MFU: `22,828,181 / (0.5 × 2307 × 256) = 77s/step`
- At 70% MFU: `22,828,181 / (0.7 × 2307 × 256) = 55s/step`

**Key question: is 16 sequences (65,536 tokens) enough for good MXU?**

Expert matmul shape (per device, all2all dispatch):
```
Received tokens ≈ 8192 per device (K=8, E=256, E_local=32 → K×E_local/E = 1.0 tokens/device)
Expert matmul: (8192, D=7168) @ (32, D=7168, D_moe=2048)
  → 8192 × 7168 × 2048 × 2 FLOPs × 32 experts = 7.7 TFLOP per device per MoE layer
  → MXU 128×128 tiles: shape is thick enough (8192 rows >> 128) ✓
```

Attention matmul shape (per device, with flash attention):
```
Q@K^T: (16, 4096, 128, 192) — 16 local seqs, 128 heads, qk_dim=192
  → (16×128) × 4096 × 4096 × 192 FLOPs ≈ 2.0 TFLOP per device per attn layer
  → Tile size: 4096 >> 128 ✓ good MXU utilization
```

Both shapes are large enough for good MXU efficiency. 16 sequences is sufficient.

---

## 7. Summary: Does EP=8 FSDP=64 GBS=1024 Make Sense?

**Yes. It's a significantly better configuration than EP=32 FSDP=16.**

| Property | EP=32 FSDP=16 GBS=1024 | EP=8 FSDP=64 GBS=1024 |
|----------|------------------------|------------------------|
| Torus alignment | EP crosses 3 axes ❌ | EP on 1 axis (4-chip) ✓ |
| Param memory/device | ~102 GB ❌ | ~20.6 GB ✓ |
| Peak HBM (flash+all2all) | ~102 GB ❌ | ~30 GB ✓ |
| EP comm per fwd pass | 218 GB ❌ | 53 GB ✓ |
| Experts per device | 8 (256/32) | 32 (256/8) ✓ more work/device |
| Local sequences | 64 | 16 (still fine for MXU) ✓ |
| MoE dispatch | psum (OK) | all2all (efficient) ✓ |

**Required changes to implement:**
1. Flash attention (Splash) — needed regardless of EP config
2. Ring-of-experts with all2all/all-gather dispatch modes
3. Update mesh to `(dp=1, ep=8, fsdp=64)`
4. Ensure GBS ≥ FSDP (1024 ≥ 64 ✓) for batch sharding
5. Ensure GBS × K / E is divisible per device for balanced all2all

**One concern**: FSDP=64 means 64-way all-gather for weights at every layer.
The weight all-gather for MLA wq_a: (7168, 1536) = 22 MB total across 64 devices.
On the 8×8 torus, max hop count = 4+4 = 8 hops. This is fine at ICI speeds (~900 GB/s).
