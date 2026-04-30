# Fused EP MoE Backward Pass — Developer Brief

## Background

A production Pallas kernel for EP MoE inference exists at:
```
~/tpu-inference/tpu_inference/kernels/fused_moe/v1/kernel.py  (1698 lines)
```

It uses **sparse A2A routing**: sends only *routed* tokens to expert devices instead of broadcasting all tokens (all_gather + dense). This eliminates the ~32× wasted compute in the current all_gather+dense path (which computes a dense `T_all × E_local` matmul where only `K/EP ≈ 1` expert per token is actually active).

We attempted to add a backward pass to enable training use. The implementation lives at `~/dsv3/fused_moe_bwd/backward_kernel.py`.

---

## What Broke: Two Fatal Scalability Issues at Full DSv3 Scale

### System configuration at failure

- Model: DeepSeek-v3 671B
- D=7168, D_moe=2048, E=256, K=8
- GBS=1024, S=4096
- EP=8, FSDP=64 on 4×8×8 v7x (512 JAX devices)
- **Tokens per device**: T = GBS × S / (EP × FSDP) = 1024 × 4096 / 512 = **524,288**

---

### Issue 1 — jax.vmap weight gather creates a 247 TB tensor

```python
# backward_kernel.py, line ~411
w1g_pairs = w1_f32[expert_ids_flat_kernel, 0]  # shape: (T×K, D, F)
```

At full scale:
- T × K = 524,288 × 8 = **4,194,304** token-expert pairs per device
- Gather shape: `(4,194,304, 7168, 2048)` float32 = **247 TB**

XLA either hangs in compilation (tries to fuse the gather-GEMM into a single op) or OOMs immediately. There is no flag or memory tuning that fixes this — the tensor is fundamentally `T×K×D×F` which is ~500,000× larger than HBM.

---

### Issue 2 — `bins_tokens` pre-allocation → 120 GB (or 3.85 TB with a bug)

```python
# backward_kernel.py, line ~479
bins_tokens = zeros((E_local * max_tpe, D))
```

`bins_tokens` is a flat buffer holding ALL token-expert pairs sorted by expert index. At full scale:
- `E_local` = E / EP = 256 / 8 = 32 local experts
- `max_tpe` = T × K / E_local = 4,194,304 / 32 = 131,072 tokens per expert
- Buffer shape: `(32 × 131,072, 7168)` = `(4,194,304, 7168)` = **120 GB**

There is also a bug on line ~472:
```python
max_tpe = cdiv(TK, bte) * bte  # should divide by E_local, not by bte
# BUG: max_tpe = TK (not TK/E_local) → buffer becomes 3.85 TB
```

Even with the bug fixed, 120 GB exceeds the ~95 GB HBM budget.

---

### Why tests passed at mini scale

The kernel was designed for inference (small batch). At mini config (`T ≈ 512`):
- `bins_tokens`: `(512 × 8 / 32, 7168)` = `(128, 7168)` = **0.7 MB** ✓
- Weight gather: `(4096, 7168, 2048)` = **24 GB** (still large but XLA handled it)

Tests passed at mini scale and failed only when scaled to full 671B training.

---

## Root Cause

The backward kernel materializes the full `(T×K, D)` expert-sorted buffer **globally** before processing. This was a reasonable design choice for inference (small T), but becomes a hard OOM wall at training scale.

Every operation that has `T×K` as a leading dimension needs to be restructured.

---

## Required Redesign: Per-Expert Streaming

The fix is to replace the global `T×K` materialization with a streaming loop over `E_local` experts:

```
Current (broken at training scale):
  pre-allocate bins_tokens (T×K, D)   ← 120 GB OOM
  process all E_local experts at once
  gather all weights (T×K, D, F)      ← 247 TB OOM

Correct (streaming):
  for e in range(E_local):
      token_mask_e = (expert_ids == e)           # which tokens route to expert e
      tokens_e     = x[token_mask_e]             # shape: (~T*K/E_local, D) ≈ 3.75 GB ✓
      d_tokens_e   = ffn_backward(tokens_e, w_e) # local matmuls, weights for expert e only
      accumulate_d_x(token_mask_e, d_tokens_e)   # scatter grads back to token positions
      accumulate_d_w_e(tokens_e, d_tokens_e)     # weight grad for expert e
```

Peak memory per expert pass: `(T×K / E_local, D)` = 120 GB / 32 = **3.75 GB** ✓

The forward kernel already operates this way (async DMA, per-expert compute). The backward follows the same structure with reversed A2A routing (gradients flow from expert devices back to token-owner devices).

---

## Backward Pass Architecture (Full Design)

The backward consists of two kernels mirroring the forward:

### `bwd_dX_kernel` — gradient w.r.t. input tokens

```
1. Reverse scatter (async_remote_copy): pull dY from expert devices back to token-owner
2. Per-expert: dX_e = dY_e @ W2_e.T  (same shape as forward, same streaming structure)
3. Weighted sum over top-k: dX += routing_weight_e × dX_e
4. Routing score backward: d_score_e = sum(dY_e × expert_out_e, dim=-1)
```

### `bwd_dW_kernel` — gradient w.r.t. expert weights

```
1. For each local expert e:
   tokens_e  = stored/recomputed activations for expert e
   d_out_e   = dY for expert e (from reverse A2A)
   dW1_e    += tokens_e.T @ d_out_e   (D×F accumulation)
   dW2_e    += intermediate_e.T @ d_out_e
```

Both kernels require `E_local` iterations (not `T×K` global pre-allocation).

---

## Integration Plan

**Phase 1 (current)**: Ring + ragged_dot in `model.py` — full autodiff support, no Pallas complexity. This is what runs today.

**Phase 2**: Hybrid `jax.custom_vjp` wrapper:
- `fwd`: use `fused_ep_moe` Pallas kernel (fast, sparse A2A)
- `bwd`: use JAX reference through autodiff (correct, not Pallas-fast)
- Correctness verified before committing to Phase 3.

**Phase 3**: Full Pallas backward (`bwd_dX_kernel` + `bwd_dW_kernel`). Requires the per-expert streaming redesign above. Estimated ~800–1200 lines each.

Full spec: `~/dsv3/specs/fused_ep_moe_backward_spec.md`

---

## EP/FSDP Integration Gap (Separate Issue)

The forward kernel uses `P(ep_axis_name)` for weights — no FSDP awareness. For EP=8 FSDP=64:

```python
# Before calling fused_ep_moe, must FSDP all_gather weights first:
w = fsdp_all_gather(w_sharded)   # reconstruct full weight for this EP shard
out = fused_ep_moe(x, w)         # kernel sees fully-gathered weights
```

This is the same pattern used in the current `_expert_mlp_ring_body`. The kernel's internal shard_map only has an `ep` axis; FSDP must be handled externally.
