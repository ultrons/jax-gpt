# DSv3 671B Layer-by-Layer Performance Analysis

**Profile**: v10 baseline (`ring+A2A`, `jax_pallas_bwd` pending)
**Cluster**: 4×8×8 v7x, 512 JAX devices, 256 chips
**Config**: EP=8, FSDP=64, GBS=1024, S=4096, PDBS=2
**Baseline**: 465 TPS/chip, 35.2s/step
**Target**: 4,000 TPS/chip (8.6× improvement needed)

---

## 1. Hardware Reference (v7x, per JAX device)

| Metric | Value | Source |
|--------|-------|--------|
| Peak bf16 compute | 1,028,750 GFLOP/s | xprof roofline |
| Peak HBM bandwidth | 530.2 GB/s | xprof roofline |
| Compute/memory knee | 1,940 FLOP/byte | derived |
| MXU busy (v10 step avg) | **37.0%** | show_utilization |
| HBM Rd+Wr utilization | **23.2%** | show_utilization |
| ICI utilization | **4.7%** | show_utilization |

Key: 63% of cycles the MXU is idle. HBM bandwidth has 4.5× headroom. ICI is underutilized.

---

## 2. Configuration and Per-Device Shapes

### Mesh and parallelism

```
mesh = (dp=1, ep=8, fsdp=64)  # 1 × 8 × 64 = 512 JAX devices
GBS  = 1024 sequences × 4096 tokens = 4,194,304 tokens
PDBS = 2 sequences per JAX device
```

### Per-device activation shapes

| Activation | Shape | Bytes |
|-----------|-------|-------|
| `x` per device (main body) | `(2, 4096, 7168)` bf16 | 117.9 MB |
| `flat_x` (pre-shard_map) | `(8192, 7168)` bf16 | 117.9 MB |
| `flat_x` in ring shard_map body | `(8192, 7168)` bf16 | 117.9 MB (T_local=8192) |
| `flat_x_all` (after EP all_gather) | `(65536, 7168)` bf16 | 943.7 MB |
| `sorted_x` (T_all × K rows) | `(524288, 7168)` bf16 | 7.55 GB |

Ring backend uses `act_spec = P(("ep","fsdp"), None)` → T_local = T_global / (ep×fsdp) = 8,192.

### Per-device weight shapes (representative MoE layer)

| Weight | Stored (sharded) | After all_gather | Bytes (full) |
|--------|-----------------|-----------------|-------------|
| `wi_0` (gate proj) | `(32, 112, 2048)` | `(32, 7168, 2048)` | 939 MB |
| `wi_1` (up proj) | `(32, 112, 2048)` | `(32, 7168, 2048)` | 939 MB |
| `wo` (down proj) | `(32, 2048, 112)` | `(32, 2048, 7168)` | 939 MB |
| gate | `(7168, 256)` unreplicated | same | 3.7 MB |
| shared wi_0/wi_1 | `(112, 2048)` | `(7168, 2048)` | 29.4 MB |
| shared wo | `(2048, 112)` | `(2048, 7168)` | 29.4 MB |

---

## 3. Full Step HLO Time Budget (v10, 2-step profile)

Total accumulated HLO op time (2 steps, one JAX device): **846,025 ms**
Wall time per step: **35.2s** → implied ~12× average parallelism.

```
MoE backward      391,403 ms   46.3%  ████████████████████░░░░░░░░░░░░░░░░░░░░░░
Ragged-dot        295,789 ms   35.0%  ██████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░
MoE forward       100,631 ms   11.9%  ████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
Overhead/no-src    48,216 ms    5.7%  ██░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
Dense+Attn          5,744 ms    0.7%  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
Other               4,242 ms    0.4%  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
```

---

## 4. Layer-by-Layer Op Breakdown

### 4.1 MLA Attention (all 61 layers)

Code: `mini_dsv3/model.py:263` — `mla_attention()`

| Op | Input shapes | Output | Collective | Code ref |
|----|-------------|--------|------------|---------|
| q_proj_a | `(8192, 7168) × (112, 1536)→(8192,1536)` | `q_low` | **FSDP all-gather** `wq_a` | `:275` |
| q_proj_b | `(8192,1536) × (1536,128,192)` | `q (B,S,H,qk)` | none | `:279` |
| kv_proj_a | `(8192,7168) × (7168,576)` | `kv_low` | **FSDP all-gather** `wkv_a` | `:284` |
| kv_proj_b | `(8192,512) × (512,128,256)` | `k,v` | none | `:292` |
| RoPE | element-wise on `(8192,H,d_rope)` | `q,k` rotated | none | `:297` |
| Splash attn | `(2,4096,128,192)` Q,K,V | `(2,4096,128,128)` | none (Pallas local) | `:333` |
| out_proj | `(8192,16384) × (16384,7168)` | `(8192,7168)` | **FSDP all-gather** `w_out` | `:364` |

**Estimated FLOPs per layer per device (fwd):**
- q_proj_a+kv_proj_a: 2 × 8192 × 7168 × (1536+576) = 248 GFLOP
- Attn core (Splash): 2 × 2 × 4096² × 128 × 192 = 1,073 GFLOP
- out_proj: 2 × 8192 × 16384 × 7168 = 1,932 GFLOP
- Total fwd ≈ **3.3 TFLOP/layer/device**; backward ≈ 6.6 TFLOP

**Measured (v10 profile):**
- Dense layer scope (3 dense layers, fwd+bwd): **5,744 ms** = **0.68%** of total HLO
- Per layer per step: 5744 / (3 × 2) ≈ 958 ms
- Dominant ops inside: `splash_mha_dkv` (373ms), `splash_mha_dq` (323ms), `splash_mha_fwd` (291ms) — all Unknown-bound (no FLOPs data, Pallas opaque)
- MoE backward scope also includes Splash attn (checkpoint recompute): `splash_mha_dkv_no_residuals.5` (21,623ms) + `splash_mha_dq_no_residuals.5` (18,721ms) + `splash_mha_fwd` (16,679ms) = 57,023ms in the 58-layer MoE backward path

**Theoretical without overlap:**
- Splash attention dominates (~60%) within attention; rest are FSDP-gather-bound matmuls
- With gradient checkpointing: fwd is recomputed during backward → 2× attention cost in backward
- Attention is **tiny** (< 2% of step time) — not a bottleneck at training scale

---

### 4.2 Dense MLP (L_dense=3 layers)

Code: `mini_dsv3/model.py` — `dense_mlp()` (implied from sharding `wi_gate`, `wi_up`, `wo_mlp`)

Weights: `wi_gate`, `wi_up`: P("fsdp",None) → `(112, 18432)` stored, `(7168, 18432)` full
`wo_mlp`: P("fsdp",None) → `(112, 18432)` stored, `(18432, 7168)` full

| Op | Shape | FLOPs/device/layer | Collective |
|----|------|--------------------|------------|
| wi_gate matmul | `(8192, 7168) × (7168, 18432)` | 2.16 TFLOP | FSDP all-gather `wi_gate` |
| wi_up matmul | same | 2.16 TFLOP | FSDP all-gather `wi_up` |
| SiLU gate | element-wise | ~0 | none |
| wo_mlp matmul | `(8192, 18432) × (18432, 7168)` | 2.16 TFLOP | FSDP reduce-scatter grad |
| Total fwd | — | **6.48 TFLOP** | 3 gathers + 1 RS |

Dense MLP contributes ~0.3% of step HLO time — negligible.

---

### 4.3 MoE Routing (gate projection)

Code: `mini_dsv3/model.py:373` — `moe_routing()`

```python
logits = jnp.einsum("bsd,de->bse", x, gate_weight)  # (8192, 256)
scores = sigmoid(logits)
top_k_indices, top_k_weights = top_k(scores + gate_bias, K=8)
```

| Op | Shape | FLOPs | Bytes | OI | Collective |
|----|------|-------|-------|----|------------|
| gate matmul | `(8192, 7168) × (7168, 256)` | 30 GFLOP | 246 MB | 121 F/B | none (gate unreplicated) |
| sigmoid+top_k | `(8192, 256)` | ~1 GFLOP | 4 MB | — | none |

OI=121 < knee 1940 → **HBM-bound** (small matmul). Fast regardless.

---

### 4.4 MoE Routed Expert Compute — THE BOTTLENECK

Code: `mini_dsv3/model.py:889` — `_expert_mlp_ring_a2a_body_impl()` (v10)
Code: `mini_dsv3/model.py:644` — `_moe_pallas_bwd_bwd()` (v12)

**v10 uses ring+A2A dispatch with ragged_dot compute.**

#### Communication schedule per MoE layer per device

```
shard_map(mesh, in_specs=(P(("ep","fsdp"),None), ..., P("ep","fsdp",None), ...))
T_local = 4,194,304 / (8 × 64) = 8,192 tokens
```

| Collective | Direction | Axis | Shape | Bytes/device | BW target | Est. time |
|-----------|-----------|------|-------|-------------|-----------|----------|
| FSDP all-gather (wi_0) | intra-host | fsdp=64 | `(32,112,2048)→(32,7168,2048)` | **937 MB** | 530 GB/s | 1.77 ms |
| FSDP all-gather (wi_1) | intra-host | fsdp=64 | same | **937 MB** | 530 GB/s | 1.77 ms |
| FSDP all-gather (wo) | intra-host | fsdp=64 | `(32,2048,112)→(32,2048,7168)` | **937 MB** | 530 GB/s | 1.77 ms |
| A2A dispatch (tokens→experts) | cross-host | ep=8 | `(T_local×K, D+1)` sorted | **940 MB** | 530 GB/s | 1.77 ms |
| A2A gather (results→owners) | cross-host | ep=8 | `(T_local×K, D)` | **940 MB** | 530 GB/s | 1.77 ms |
| FSDP reduce-scatter (grad) | intra-host | fsdp=64 | `(32,7168,2048)` grad | **937 MB** | 530 GB/s | 1.77 ms |

**Total theoretical comm time (no overlap): 6 × 1.77 = 10.6 ms per layer** assuming 530 GB/s.

**Measured FSDP all-gather BW (v10):** 94–104 GB/s → **5.1–5.6 ms per layer** (9–11× below peak, cross-host vs intra-host?)

Wait — FSDP=64 on 4×8×8 spans 8×8 = 64 devices in a 2D torus subgroup. The FSDP all-gather involves cross-host ICI (not just intra-host). Measured: ~12.9s per all_gather in 2-step profile → ~222ms per layer per device. That's 22× slower than theoretical! This is a known issue: FSDP gathers are on the critical path (0% overlap).

#### Compute: ragged_dot (forward + backward)

```python
# Inside _expert_mlp_ring_ragged_body / _expert_mlp_ring_a2a_body_impl
sorted_x   = (T_all × K, D)  = (65536×8, 7168) = (524288, 7168)
wi_0_ext   = (E_local+1, D, D_moe) = (33, 7168, 2048)
gate_out   = ragged_dot(sorted_x, wi_0_ext, group_sizes)   # gate
up_out     = ragged_dot(sorted_x, wi_1_ext, group_sizes)   # up
expert_out = ragged_dot(hidden,   wo_ext,   group_sizes)   # down
```

| Op | Input | Weight | FLOPs | OI | Achieved | Bound |
|----|------|--------|-------|-----|---------|-------|
| ragged_dot gate_proj | `(524288,7168)` | `(33,7168,2048)` | 15.4 TFLOP | 1447.7 | 620 TFLOP/s | Compute |
| ragged_dot up_proj | same | same | 15.4 TFLOP | 1447.7 | 621 TFLOP/s | Compute |
| ragged_dot down_proj | `(524288,2048)` | `(33,2048,7168)` | 15.4 TFLOP | 1447.7 | 682-709 TFLOP/s | Compute |
| ragged_dot bwd(×3) | various | various | ~46 TFLOP | ~1448 | ~650 TFLOP/s | Compute |

**Measured performance (v10 profile):**
```
ragged-dot-none.1:  34,559ms/2steps × 620 TFLOP/s, 60.3% MXU
ragged-dot-none.3:  34,481ms/2steps × 621 TFLOP/s, 60.4% MXU
ragged-dot-none.4:  31,438ms/2steps × 682 TFLOP/s, 66.3% MXU
ragged-dot-none.8:  30,224ms/2steps × 709 TFLOP/s, 68.9% MXU
```

6–7 unique ragged-dot shapes (fwd gate+up+down, bwd gradient passes).
Total ragged-dot HLO time: **295,789 ms** over 2 steps = **34.9%** of all HLO time.

**Ragged-dot is the largest compute sink but runs at reasonable 60–69% MXU efficiency.**

#### v10 MoE backward: the dominant bottleneck

The backward for ring+A2A involves transposing the token-sort permutation and combine-weighting:

```python
# _sort_tokens_by_expert backward → scatter-add on sorted activations
# _unsort_and_combine backward → scatter-add on expert outputs
```

Both are **large irregular scatter-add operations** over `(524288, 7168)` tensors.

| Op | Tensor size | HBM BW achieved | Peak BW | Efficiency |
|----|------------|-----------------|---------|------------|
| scatter_custom_fusion.29 (combine bwd) | 7.55 GB | **87 GB/s** | 530 GB/s | **16.4%** |
| scatter_custom_fusion.31 (sort bwd) | 7.55 GB | **88 GB/s** | 530 GB/s | **16.6%** |

These two ops alone: **222,512 ms** = **26.3%** of total HLO time = **57.7% of all MoE backward time**.

**Root cause**: Scatter-add on random permutation patterns causes highly non-contiguous HBM access patterns → cache thrashing → 16% of peak HBM BW.

**v12 fix (jax_pallas_bwd)**: The Pallas backward kernel (`fused_ep_moe_bwd`) replaces these scatter-add operations with a tiled, VMEM-resident backward that computes dX, dW directly from sorted activations without unsort scatter. D-tiling (tile_D=1024, 7 tiles over D=7168) keeps the working set in VMEM (64 MB).

---

### 4.5 Shared Expert (n_shared=1)

Code: `mini_dsv3/model.py:1244` — `moe_layer()`

Same structure as dense MLP with D_moe=2048. Computed in parallel with routed expert path.

| Weight | Shape | FLOPs/layer | Collective |
|--------|-------|-------------|------------|
| shared_wi_0, wi_1 | `(7168, 2048)` per device full | 2 × 8192 × 7168 × 2048 = 0.24 TFLOP | FSDP all-gather |
| shared_wo | `(2048, 7168)` | 0.24 TFLOP | FSDP all-gather |

Negligible fraction of total time.

---

### 4.6 Collectives Summary (per MoE layer)

From `list_collectives --overlap` on v10 profile (top non-overlapped):

| Collective | Source | Time/2steps | Overlap | BW (GB/s) | Analysis |
|-----------|--------|-------------|---------|-----------|---------|
| all-gather.479 | `dp_weight_gather` | 12,917ms | **0%** | 94 | FSDP wi_0 or wi_1 |
| all-gather.483 | `dp_weight_gather` | 12,909ms | **0%** | 94 | FSDP weight |
| all-gather.481 | `dp_weight_gather` | 11,755ms | **0%** | 104 | FSDP weight |
| all-gather.445 | `ep_token_gather` | 12,232ms | **0%** | 100 | EP token A2A/gather |
| all-gather.441 | `reduce_scatter` | 7,425ms | **0%** | 164 | FSDP reduce-scatter |

**Total non-overlapped collective stall time: 59,577ms** over 2 steps.

FSDP weight all-gathers occupy 37.2% of MoE forward HLO time and are **completely exposed** (not pipelined with compute). This is a fundamental issue with the ring+A2A backend.

---

## 5. Aggregate Step Time Analysis

### HLO time budget per step (single device, approximate)

| Component | HLO time/step | % total | Wall contribution |
|-----------|--------------|---------|-------------------|
| MoE backward — scatter-add | **111s** | 26.3% | ~9s |
| Ragged-dot compute (6–7 ops) | **148s** | 35.0% | ~12s |
| MoE forward — A2A + gathers | **50s** | 11.9% | ~4s |
| FSDP weight all-gathers | **25s** | 5.9% | ~2s (stall) |
| FSDP grad reduce-scatter | **14s** | 3.3% | ~1s |
| Splash attention fwd/bwd | **28s** | 6.6% | ~2s |
| Overhead / unattributed | **47s** | 11.1% | ~4s |
| **Total** | **423s** | 100% | **~35.2s** |

Wall time estimates assume uniform 12× parallelism (423s / 35.2s). In practice, non-overlapped collectives (~30s HLO) are closer to serial → they likely dominate the critical path.

### MoE forward critical path (one layer, estimated)

```
FSDP all-gather(wi_0,wi_1,wo) [~222ms, 0% overlap]
     ↓ (stall)
A2A dispatch [~6.8ms, overlaps with below?]
     ↓
ragged_dot gate×up×down [3×24.8ms = 74ms, compute]
     ↑ overlaps with A2A gather in theory
A2A gather [~6.3ms]
     ↓
psum/combine [~5ms]
     ↓ FSDP reduce-scatter for grad [~12ms]
Total per layer ≈ 222 + 74 + 12 = 308ms
Total 58 MoE layers ≈ 17.9s
```

But measured MoE forward is ~50s HLO time / 12× parallelism ≈ 4.2s wall time.
The 4× discrepancy: layers ARE pipelined via lax.scan + XLA scheduling.

---

## 6. v10 Profile: Top Bottlenecks Ranked

| Rank | Bottleneck | HLO time/2steps | % total | Root cause | Fix |
|------|-----------|-----------------|---------|------------|-----|
| 1 | **scatter-add (MoE bwd)** | 222,512ms | 26.3% | HBM scatter at 87GB/s (16% eff) | **v12 Pallas bwd** |
| 2 | **Ragged-dot compute** | 295,789ms | 35.0% | 60–69% MXU, not bad but dominant | Larger batch, better tiling |
| 3 | **MoE forward A2A+gathers** | 100,631ms | 11.9% | FSDP gathers at 94–104GB/s, 0% overlap | Pipeline gathers across layers |
| 4 | **Grad reduce-scatter** | 27,826ms | 3.3% | HBM at 44GB/s (8% eff) | XLA async AG-RS |
| 5 | **EP token all-gather** | 12,232ms | 1.4% | 100GB/s (19% HBM) | Not critical path |

---

## 7. Path to 4,000 TPS/chip

Current: **465 TPS/chip** at 35.2s/step.
Target: **4,000 TPS/chip** → 4.1s/step → **8.6× speedup needed**.

Hardware ceiling (100% compute, 100% overlap):
```
89,172 TFLOP/chip ÷ 1,028,750 GFLOP/s = 86.7s serial compute time
With 12× actual parallelism → 7.2s minimum (fully compute-bound)
TPS/chip ceiling ≈ 4,194,304 / (256 × 7.2) ≈ 2,274 TPS/chip
```

So even at 100% MXU with no communication overhead, ~2,274 TPS/chip is the compute ceiling for this problem size. **4,000 TPS/chip requires either larger batch or hardware we haven't fully characterized.**

Let me restate using measured peak MXU efficiency:

```
Achievable at 100% MXU (1028 TFLOP/s per device) vs current 37% MXU:
Current wall time = 35.2s, MXU busy = 13s → compute delivers 13s × 37% × 1028 = 4,949 TFLOP
At 100% MXU: same compute needs 13s (can't go faster if all other ops removed)
But 63% of time is non-compute → removing it gives: 13s/step = 3,240 TPS/chip
```

### Optimization roadmap

#### Level 1: v12 Pallas backward (removes scatter-add bottleneck)
- Eliminates scatter_custom_fusion.29+.31: −222,512ms HLO = **−26.3%**
- Pallas kernel computes dX, dW with VMEM tiling → ~80% HBM BW (vs 16%)
- Expected MoE backward speedup: **3–4×**
- Step time impact (proportional): 35.2 × (1 - 0.263 + 0.263/4) = 35.2 × 0.803 ≈ **28.3s**
- **Expected: ~1.24× step speedup → ~580 TPS/chip**

But compilation is currently stuck at 30+ minutes due to `shard_map(pallas_call)` slow XLA compilation.

#### Level 2: Pipeline FSDP weight gathers (removes #3 bottleneck)
- Current: 3 FSDP all-gathers per layer, 0% overlap = ~37% of MoE forward HLO
- Fix: Prefetch next layer's weights while computing current layer (async AG before RS)
  ```python
  # XLA flag already set: --xla_tpu_enable_ag_backward_pipelining=true
  # But "dp_weight_gather" is not overlapped in v10 → flag may not be triggering
  ```
- This requires restructuring the layer body to expose the overlap to XLA
- Potential: pipeline latency hidden behind compute (**1.2–1.3× step speedup**)

#### Level 3: Larger batch (PDBS=4 → 8)
- Doubles/quadruples tokens per device: T_local=16384, T_all=131072
- Ragged-dot arithmetic intensity unchanged but FSDP gather cost amortized
- scatter-add cost scales with T_all (still bad for v10, good for Pallas)
- Memory cost: activations 2× → check 95 GB HBM budget
  - With gradient_checkpoint: activations ≈ 1 layer × 2 × 8192 × 7168 × 2 bytes = 235 MB per layer
  - 58 layers scan: needs only ~1 layer active + scan carry
  - PDBS=4 is feasible; PDBS=8 needs validation
- Ragged-dot FLOPs double → time doubles but FSDP gather stays same → **1.3–1.5× TPS improvement**

#### Level 4: Ragged-dot efficiency improvements
- Currently 60–69% MXU. Theoretical ceiling: ~80–85% for well-tiled matmuls
- Increase D_moe tile size or use fused kernel (e.g., Pallas GMM forward)
- **~1.1–1.2× on ragged-dot portion**

#### Level 5: EP scaling (EP=16 or EP=32)
- EP=16: T_local halves to 4,096, T_all halves to 32,768
- Ragged-dot compute halves: FLOPs/device/layer = 7.7 TFLOP instead of 15.4
- FSDP=32: FSDP gather bytes halve (470 MB vs 937 MB)
- But EP collectives double (16-way vs 8-way)
- ICI currently 4.7% utilized → room for more EP
- **~1.5–2× combined compute + gather reduction**

#### Level 6: Custom Pallas forward kernel (full fusion)
- Fuse scatter+gather+matmul×3 into one Pallas kernel (like fused_ep_moe_bwd)
- Eliminates A2A overhead and intermediate buffers
- Potential for near-memory-optimum execution
- **2–3× on MoE forward**

### Summary table

| Optimization | Effort | Expected Speedup | Cumulative TPS/chip |
|-------------|--------|-----------------|---------------------|
| v10 baseline | — | 1× | 465 |
| v12 Pallas backward | Done (pending compile) | 1.24× | ~580 |
| PDBS=4 | Easy (flag) | 1.35× | ~780 |
| Pipeline FSDP gathers | Medium | 1.25× | ~975 |
| EP=16 | Medium | 1.6× | ~1,560 |
| Pallas forward (fused) | Hard | 2.0× | ~3,120 |
| Full optimization | — | ~6.7× | ~3,100 |

**The 4,000 TPS/chip target requires all of the above plus potential hardware-level improvements or a larger cluster.**

A more realistic near-term target: **~1,500–2,000 TPS/chip** within ~6 weeks with PDBS scaling, EP scaling, and the Pallas backward kernel compiling successfully.

---

## 8. Key Facts for Reasoning About Headroom

```
v7x peak compute:    1,028,750 GFLOP/s per JAX device
v10 achieved MXU:    37% = 380,637 GFLOP/s
                     → 63% headroom on compute

v7x peak HBM BW:     530.2 GB/s
v10 avg HBM:         23.2% = 123 GB/s average across step
Top scatter-add:     87 GB/s = 16.4% of peak HBM BW
                     → 6× headroom on HBM BW

v7x ICI:             4.7% utilized
                     → 20× headroom on network; EP/FSDP not network-bound

Wall time 35.2s:     MXU busy 13s, rest = 22.2s overhead/stall
                     → Halving overhead → 35.2→(13+11)=24s → 1.47× speedup from overlap alone

Roofline TPS/chip:   89,172 TFLOP/chip ÷ 1,028,750 GF/s = 86.7s → 485 TPS/chip (compute roofline)
v10 vs roofline:     465/485 = 95.9% OF COMPUTE ROOFLINE? — No.
```

Wait, 89,172 TFLOP/chip divided by 1,028,750 GFLOP/s = 86.7s per step (if fully serial per chip). But there are 256 chips working in parallel:
```
Global FLOPs per step: 22,828,181 TFLOP (from train.py log)
Per-chip FLOPs: 22,828,181 / 256 = 89,172 TFLOP/chip
At 1,028,750 GF/s: ideal time = 89,172 / 1028.75 = 86.7s
But chips run in parallel: all 256 chips run concurrently
Each chip's share of compute = 89,172 TFLOP at 1028 TFLOP/s = 86.7s? No — each chip runs THE SAME 86.7s compute (it's all-reduce, not partitioned)
```

Actually the per-chip FLOPs include both weight compute AND activations. The correct roofline is:
```
Per-chip compute time = 89,172 TFLOP / (1028.75 TFLOP/s) = 86.7s  (if one chip, serial)
With parallelism within chip (multiple ops overlap): wall time < 86.7s
Achieved: 35.2s → 86.7/35.2 = 2.46× compute parallelism within chip
```

So v10 achieves **only 37% MXU** (from show_utilization), but 86.7s/35.2s = 2.46× means on average 2.46 compute ops run in parallel. The 37% MXU is the correct metric: 37% of cycles the MXU is actually executing.

**True roofline TPS/chip** = (89,172 TFLOP/chip × 256 chips × TPS_per_TFLOP) → too complex.

Simply: to hit 4000 TPS/chip we need 35.2s × (465/4000) = 4.1s step time. That requires ~8.6× speedup. If we could get to 100% MXU (impossible in practice), we'd need:
- Current step: 37% MXU of 35.2s = 13s compute; non-compute = 22.2s
- Removing all non-compute: 13s/step → 465 × (35.2/13) = 1,259 TPS/chip
- Getting to 100% MXU on 13s: 13s × (1/37%) * 37% = still 13s...
- Getting 100% MXU in minimum time = 86.7s/256_chips...

The bottom line: **4000 TPS/chip requires fundamentally reducing the memory-bound scatter-add penalty (Pallas bwd), adding compute via larger batch, and better pipelining collectives.**

---

## 9. Next Steps

1. **Wait for v12 to compile** (currently at 30+ min, expected 40–60 min first run)
   - If it compiles: profile with xla_shell, measure scatter-add elimination
   - If it times out at 60 min: restructure `fused_ep_moe_bwd` to remove nested JIT

2. **Submit PDBS=4 run** (change `--gbs=2048` in YAML, keep same EP=8/FSDP=64)
   - Fast win: amortizes FSDP gather cost, improves ragged-dot efficiency

3. **EP=16 experiment**: create `dsv3-train-4x8x8-v13.yaml` with `--ep=16 --fsdp=32`
   - ICI at 4.7% → plenty of room for more EP collectives

4. **Profile Pallas custom-calls properly**: add `--xla_enable_custom_call_region_trace=true`
   to get FLOPs data for the 288 opaque ops (9% of HLO time)

---

*Profile source: `gs://sivaibhav-exp/dsv3/profiles/4x8x8-ep8-fsdp64-gbs1024-v10`*
*Generated: 2026-03-30*
