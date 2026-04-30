# Pallas A2A vs JAX: Training Performance Analysis

**Date:** 2026-04-03  
**Config:** debug (D=7168, L=13, 10 MoE layers), EP=32, FSDP=16, GBS=1024, S=4096  
**Cluster:** 4×8×8 v7x (512 JAX devices, 256 chips)  
**Image:** v119c-train  
**Profiles:** `gs://max-experiments/dsv3/profiles/bench-v4-jax`, `bench-v4-pallas`  

---

## 1. Headline Numbers

| Metric | JAX backend | Pallas backend | Delta |
|---|---|---|---|
| TPS/chip (steady state) | **~3391** | **~2484** | Pallas **27% slower** |
| Step time | 4.83s | 6.59s | +1.76s/step |
| MFU | 141% | 109% | |
| Loss @ step 100 | 25.084 | 25.091 | +0.007 |
| Profile total (3 steps) | 154.5s | 202.7s | +48.2s |

The JAX backend at ~3391 TPS/chip is the **best measured training performance** on this config to date.
The roofline for EP=32, FSDP=16, GBS=4 (ragged_dot+fp8) predicts 3,880 TPS/chip — we are at 87% of that ceiling with the plain JAX backend.

---

## 2. Component Breakdown

| Component | JAX | Pallas | Notes |
|---|---|---|---|
| MoE forward | 50.2s (32.5%) | **8.4s (4.2%)** | Pallas **6× faster** |
| MoE backward | 79.3s (51.4%) | **178.6s (88.1%)** | Pallas **2.25× slower** |
| Dense + attention | ~9.9s (6.4%) | ~10.0s (4.9%) | Identical |
| Total collective time | 70.4s (45.6%) | 91.9s (45.3%) | |
| MXU utilization | 11.2% | 7.6% | Both severely underutilized |
| ICI utilization | 24.6% | 22.2% | Both communication-bound |

**The Pallas fused A2A kernel delivers a real 6× forward speedup.  
The backward regression fully cancels it and then some.**

---

## 3. Full Computation Graph

### 3a. JAX Forward (`_expert_mlp_ep_body_ep_sharded`, shard_map)

```
all_gather(tokens, "ep")      9.76s × 2  ← FSDP scatter-reduce (dominant)
all_gather(indices, "ep")  
all_gather(weights, "ep")     
scatter_custom_fusion          4.96s × 2  ← EP token scatter (SparseCore-offloaded)
local expert einsum            0.33s × 2  ← compute-bound @ 932K GFLOP/s
all_gather(tokens, "ep")      3.54s × 4  ← ep_token_gather (result routing back)
psum_scatter(out, "ep")       }
psum(out, "fsdp")             } fused into reduce-scatter seen in profile
splash_mha_fwd                 2.03s × 2
```
Total forward: 50.2s. FSDP reduce-scatter alone = 19.5s (39% of forward).

### 3b. JAX Backward (jax.vjp automatic transpose)

```
reduce-scatter.37 (FSDP grad)    8.5s       vjp of FSDP weight gathers
all-gather.609/.575 (ep_psum_scatter)  7.1s × 2  vjp of psum_scatter("ep")
scatter_custom_fusion.48/.43     5.0s × 2   token routing grad
all-gather.611/.577 (ep_token_gather)  3.5s × 2  vjp of all_gather("ep")
splash_mha_dkv + splash_mha_dq   ~10s       attention backward
splash_mha_fwd remat             2.0s × 2   attention recompute (nothing_saveable)
gather_offload_async_done        ~3.7s      async FSDP prefetch completions
```
Total backward: 79.3s.  
Key: JAX VJP produces **reduce-scatter** (psum_scatter) for both EP and FSDP — half the data of all-reduce.

### 3c. Pallas Forward (`_moe_pallas_fwd_ep_fn`, shard_map)

```
all_gather(x, "ep")             }
all_gather(g, "ep")             } All fused into...
fused_ep_moe_fwd_streaming_v1   } one Pallas A2A+GEMM kernel (not visible without region trace)
psum_scatter(out, "ep")         }
psum(out, "fsdp")               }
splash_mha_fwd                   2.03s × 2   (same as JAX)
```
Total forward: 8.4s. The Pallas kernel eliminates 4 separate EP/FSDP collectives that cost 43.8s in JAX.

### 3d. Pallas Backward (`streaming_bwd_v2`, shard_map)

```
# 4 separate all_gathers (inside _streaming_bwd_fn):
all_gather(grad, "ep")          }
all_gather(fx, "ep")            } 7.1–7.3s each × 2 large
all_gather(fw, "ep")            } small
all_gather(fi, "ep")            } small

gather_custom_fusion.215/.238    18.2s + 17.1s  ← Pallas weight gather (no FLOPs data yet)

fused_ep_moe_bwd_streaming_v2   (Pallas kernel — handles per-expert weight grads internally)

# Token grad reduction — THE PRIMARY BOTTLENECK:
psum(d_tok, "ep")               14.4s + 13.9s  ← all-reduce over EP=32 (2× data of reduce-scatter)
psum(d_gate, "ep")              12.3s + 12.2s  ← all-reduce
psum(d_gate2, "ep")             7.8s           ← 609KB in 7.8s = SYNC BARRIER

# Inside backward kernel (fsdp_axis_name="fsdp"):
psum(d_w0, "fsdp")              }
psum(d_w1, "fsdp")              } all-reduce (should be reduce-scatter)
psum(d_wo, "fsdp")              }

splash_mha_dkv + splash_mha_dq  ~10s  (identical to JAX)
scatter_custom_fusion            5.1s × 2
```
Total backward: 178.6s.

---

## 4. Root Causes of Backward Regression

### 4a. `psum("ep")` all-reduce instead of `psum_scatter("ep")` reduce-scatter — **~42s wasted**

In `_streaming_bwd_fn` (model.py:1011–1020):
```python
# CURRENT (bad):
d_tok_full = jax.lax.psum(d_tok_partial, ep_axis_name)      # all-reduce (T/FSDP, D) = 938MB
d_tok_l    = jax.lax.dynamic_slice(d_tok_full, ...)          # then slice

# SHOULD BE:
d_tok_l = jax.lax.psum_scatter(d_tok_partial, ep_axis_name,  # reduce-scatter: 2× less data
                                scatter_dimension=0, tiled=True)  # directly produces local slice
```
All-reduce does 2× the data movement of reduce-scatter AND produces a full 938MB tensor that must then be sliced anyway. This is the VJP of `all_gather` — the mathematically correct operation is `reduce_scatter`, not `psum + slice`.

### 4b. Sync barrier: all-reduce.311 — **7.8s for 609KB = 0.0001 GB/s**

The `psum(d_topk_partial, "ep")` for gate gradients is a tiny tensor but creates a global synchronization point that blocks the entire pipeline. This is why **ALL collectives show 0% overlap** in the Pallas backward (vs 74% average overlap for all-gathers in JAX). A single serialization point forces everything into a sequential chain.

### 4c. 4× all_gather("ep") vs 1× in JAX — **+17.9s**

streaming_bwd_v2 explicitly gathers `(grad, fx, fw, fi)` separately:
- 2 large: grad + fx ≈ 938MB each → 7+ seconds each
- 2 small: fw + fi → negligible

JAX backward only needs 1 all_gather (the VJP of the forward's psum_scatter). 

### 4d. `psum("fsdp")` all-reduces inside backward kernel — **~18s**

`fused_ep_moe_bwd_streaming_v2` with `fsdp_axis_name="fsdp"` does 3 all-reduces (d_w0, d_w1, d_wo). JAX's GSPMD automatically uses reduce-scatter for these.

### 4e. SparseCore / async flags don't help Pallas

| | JAX | Pallas |
|---|---|---|
| Async FSDP prefetch | `gather_offload_async_done` at 157–303 GB/s | `gather_custom_fusion` at 0 GB/s |
| `--xla_tpu_enable_ag_backward_pipelining` | 74% avg all-gather overlap | 0% overlap everywhere |
| `--xla_tpu_enable_async_collective_fusion` | Reduces effective FSDP time | No effect on shard_map custom ops |

XLA cannot pipeline or overlap Pallas custom-call ops — they are opaque to the async fusion analysis. The `--xla_tpu_enable_ag_backward_pipelining=true` flag that gives JAX 74% overlap is completely ineffective for the streaming_bwd_v2 shard_map.

---

## 5. Cost Summary (per profile step, 3-step average)

### JAX backward collectives: 39.6s
| Op | Time | Data | Pattern |
|---|---|---|---|
| reduce-scatter (FSDP) | 8.5s | 5.7 GB | efficient |
| all-gather ×2 (EP psum_scatter) | 14.1s | 16.9 GB | efficient |
| all-gather ×2 (EP token gather) | 7.1s | 0.8 GB | efficient |
| scatter_custom_fusion ×2 | 9.9s | — | HBM-bound |

### Pallas backward collectives: 120.9s
| Op | Time | Data | Pattern | Problem |
|---|---|---|---|---|
| psum("ep") ×2 (d_tok) | 28.2s | 16.1 GB | all-reduce | 2× data of RS |
| psum("ep") ×2 (d_gate) | 24.6s | 23.5 GB | all-reduce | same |
| psum("ep") (sync barrier) | 7.8s | 609 KB | all-reduce | serializes pipeline |
| all-gather ×4 (inputs) | 25.0s | ~17 GB | — | 4× more than needed |
| gather_custom_fusion ×2 | 35.3s | — | Pallas (no data) | unknown until v5 profile |

---

## 6. Optimization Roadmap

### Fix 1 — Replace `psum + slice` with `psum_scatter` (easy, high impact)
**File:** `model.py:1011–1020`  
**Expected savings:** ~42s backward → brings backward to ~136s  
**Also eliminates the sync barrier** (all-reduce.311)

```python
# model.py:1011
- d_tok_full  = jax.lax.psum(d_tok_partial,  ep_axis_name)
- d_topk_full = jax.lax.psum(d_topk_partial, ep_axis_name)
- T_ep_local = g_l.shape[0]
- device_ep  = jax.lax.axis_index(ep_axis_name)
- d_tok_l  = jax.lax.dynamic_slice(d_tok_full,  (device_ep * T_ep_local, 0), (T_ep_local, D))
- d_topk_l = jax.lax.dynamic_slice(d_topk_full, (device_ep * T_ep_local, 0), (T_ep_local, K))

+ d_tok_l  = jax.lax.psum_scatter(d_tok_partial,  ep_axis_name, scatter_dimension=0, tiled=True)
+ d_topk_l = jax.lax.psum_scatter(d_topk_partial, ep_axis_name, scatter_dimension=0, tiled=True)
```

### Fix 2 — Stack `grad+fx` all_gather (medium, some impact)
Combine the two 938MB all_gathers into a single stacked gather:
```python
gfx = jnp.stack([g_l, fx_l], axis=0)  # (2, T_ep, D)
gfx_full = jax.lax.all_gather(gfx, "ep", axis=1, tiled=True)  # one collective
g_full, fx_full = gfx_full[0], gfx_full[1]
```
Halves the number of large EP all_gathers. Same data volume but may improve overlap and scheduling.

### Fix 3 — Replace kernel's `psum("fsdp")` with `psum_scatter("fsdp")` (hard, kernel change)
Inside `fused_ep_moe_bwd_streaming_v2`, change weight gradient reduction from all-reduce to reduce-scatter. Requires modifying the Pallas backward kernel. Expected savings: ~18s.

### Fix 4 — Profile with `--xla_enable_custom_call_region_trace=true` (diagnostic, in flight)
Profile v5 already submitted: `gs://max-experiments/dsv3/profiles/bench-v5-pallas-traced`.  
Will reveal actual bandwidth of `gather_custom_fusion` (35.3s). If HBM-bound and slow, may need kernel optimization. If at hardware BW limits, this cost is fundamental.

---

## 7. Projected Performance After Fixes

| State | Backward time | TPS/chip (est.) |
|---|---|---|
| Current (v119c) | 178.6s | 2,484 |
| After Fix 1 (psum_scatter) | ~136s | ~3,100 |
| After Fix 1+2 (stack gather) | ~128s | ~3,200 |
| After Fix 1+2+3 (kernel psum_scatter) | ~110s | ~3,400 |
| JAX baseline | 79.3s | 3,391 |
| Roofline (EP=32, ragged_dot+fp8) | — | 3,880 |

Fix 1 alone should bring Pallas training within ~10% of JAX. Fix 3 would make Pallas competitive. The Pallas forward is already 6× faster — if backward parity is achieved, the overall benefit is real.

---

## 8. Open Questions

1. **What is `gather_custom_fusion`?** (35.3s, 0 GB/s) — v5 profile will answer this.
2. **Why is psum("ep") BW only 0.6–1.1 GB/s?** EP=32 spanning 2 physical torus dimensions (4×8=32) may give suboptimal ICI routing. On v7x, EP=32 means devices span the full 4×8 ICI grid — the effective bisection BW may be much lower than single-axis EP=8.
3. **Can async collective fusion help after Fix 1?** Once the sync barrier is removed, XLA's async pipelining flags may unlock overlap for the remaining all-gathers.
4. **Is `everything_saveable` checkpoint the right policy?** It saves ~4s of attention remat but costs additional HBM. For the 61-layer production model, HBM budget may force switching to `nothing_saveable` + the Pallas-safe custom_vjp approach.
