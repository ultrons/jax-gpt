# MoE Forward + Backward — Per-Device Operation Whiteboard

**Config**: GBS=1024, T=4096, D=7168, F=2048, E=256, K=8, EP=8, FSDP=64
**Per device**: B=8 sequences, T_local = B×T = 32,768 tokens, E_local = E/EP = 32 local experts
**Avg tokens per expert** (tpe): T_local × K / E_local = 32,768 × 8 / 32 = **8,192** (varies by routing)

---

## Forward Pass

### Phase 1 — Routing (local, no communication)

```
Input:  x_flat  [32768, 7168]   (attention output, flattened B×T)
        W_gate  [256,  7168]    (router weights, replicated across devices)

Step 1  gate_logits = x_flat @ W_gate.T          [32768, 256]   matmul (TC)
Step 2  weights     = sigmoid(gate_logits)        [32768, 256]   elementwise

Step 3  expert_ids, top_weights = top_k(weights, K=8)
           expert_ids   [32768, 8]   int32, values in [0..255]
           top_weights  [32768, 8]   float32

Step 4  expert_ids_flat = expert_ids.reshape(-1)  [262144]   T*K token-expert pairs
        token_ids_flat  = repeat(arange(T), K)    [262144]   which token each pair belongs to

Step 5  sort_idx         = argsort(expert_ids_flat)        [262144]   groups pairs by expert
        sorted_expert_ids = expert_ids_flat[sort_idx]      [262144]
        sorted_token_ids  = token_ids_flat[sort_idx]       [262144]

Step 6  send_counts  = histogram(sorted_expert_ids, bins=256)   [256]   pairs per expert
        send_offsets = cumsum(send_counts)                       [257]   slice boundaries
```

After step 6: pairs 0..send_offsets[e] belong to expert e. Pairs for expert e on
EP device d live in send_offsets[d*32] .. send_offsets[(d+1)*32].

---

### Phase 2 — Pack + DMA Send to Expert Devices

```
Step 7  sorted_tokens = x_flat[sorted_token_ids]   [262144, 7168]   gather (index into T dim)

        For each EP peer device d in 0..7:
Step 8    lo, hi = send_offsets[d*32], send_offsets[(d+1)*32]
          async_remote_copy(
              src = sorted_tokens[lo:hi],          [count_d, 7168]
              dst = recv_buf[my_slot] on device d
          )
          # also ships: sorted_expert_ids[lo:hi], top_weights[lo:hi]
```

Point-to-point DMA — only tokens that have at least one assignment to device d's
32 experts are sent. Tokens with no assignment to d are never transmitted.

---

### Phase 3 — Expert FFN (on received tokens, streaming over E_local)

```
recv_buf  [~32768, 7168]   tokens received from all 8 EP peers
                           (avg same count as sent; ragged per-peer slices)

For each local expert e in 0..31:

Step 9    tok_e = recv_buf[expert_offset[e] : expert_offset[e+1]]   [tpe, 7168]

Step 10   gate_up = tok_e @ W1[e].T        [tpe, 4096]   fused gate+up projection (TC)
          gate, up = split(gate_up, -1)    each [tpe, 2048]
          act      = silu(gate) * up       [tpe, 2048]   elementwise (save for bwd)

Step 11   out_e    = act @ W2[e].T         [tpe, 7168]   (TC)
          out_e   *= weight_e              [tpe, 7168]   elementwise scale by routing weight
                                                         weight_e = top_weights for expert e
          store out_e → output_buf[expert_offset[e] : expert_offset[e+1]]
```

Residuals to save for backward: `tok_e`, `gate`, `up`, `act`, `out_e` (pre-weight-scale).

---

### Phase 4 — DMA Send Back + Weighted Combine

```
        For each EP peer device d in 0..7:
Step 12   async_remote_copy(
              src = output_buf[d_slice],     [count_d, 7168]
              dst = result_buf[my_slot] on device d (token owner)
          )

result_buf  [262144, 7168]   expert outputs, in sorted_token_ids order

Step 13  moe_out = zeros([32768, 7168])
         moe_out[sorted_token_ids] += result_buf      scatter-add (segment_sum)
         # each token accumulates contributions from its K=8 experts

Step 14  x = x + moe_out                             residual add
```

---

## Backward Pass

Gradient flowing in: `d_moe_out [32768, 7168]`
Goal: compute `d_x_flat`, `d_W1[E_local]`, `d_W2[E_local]`, `d_W_gate`

---

### Phase 4 Backward — Reverse Combine + DMA Receive

```
Step B1  d_x_flat += d_moe_out                          residual passthrough [32768, 7168]

         Reverse of scatter-add (step 13):
Step B2  d_result_buf = d_moe_out[sorted_token_ids]     [262144, 7168]   gather
         # scatter-add reverses to a gather: one d_result_buf row per token-expert pair

         For each EP peer device d:
Step B3    async_remote_copy(
               src = d_result_buf[d_slice],     [count_d, 7168]
               dst = d_output_buf[my_slot] on device d (expert owner)
           )
```

---

### Phase 3 Backward — Expert FFN Backward (on expert device)

```
d_output_buf  [~32768, 7168]   received from token-owner devices

For each local expert e in 0..31:

Step B4   d_out_e_scaled = d_output_buf[expert_offset[e] : expert_offset[e+1]]
                                                          [tpe, 7168]

          Reverse of weight scale (step 11):
Step B5   d_weight_e = (d_out_e_scaled * out_e_pre).sum(-1)   [tpe]   routing weight grad
          d_out_e    = d_out_e_scaled * weight_e               [tpe, 7168]

          Reverse of down projection (step 11):
Step B6   dW2[e]  += act.T @ d_out_e        [2048, 7168]   weight grad (TC)
          d_act    = d_out_e @ W2[e]         [tpe, 2048]   input grad (TC)

          Reverse of SiLU + elementwise multiply (step 10):
Step B7   d_up   = d_act * silu(gate)                     [tpe, 2048]
          d_gate = d_act * up * silu_grad(gate)            [tpe, 2048]
          d_gate_up = concat(d_gate, d_up, axis=-1)        [tpe, 4096]

          Reverse of up projection (step 10):
Step B8   dW1[e]  += tok_e.T @ d_gate_up    [7168, 4096]  weight grad (TC)
          d_tok_e  = d_gate_up @ W1[e]       [tpe, 7168]  input grad (TC)

          store d_tok_e   → d_sorted_tokens[expert_offset[e] : ...]
          store d_weight_e → d_weights_sorted[expert_offset[e] : ...]
```

---

### Phase 2 Backward — DMA Send Back to Token Owners

```
        For each EP peer device d:
Step B9   async_remote_copy(
              src = d_sorted_tokens[d_slice],    [count_d, 7168]
              dst = d_recv_tokens[my_slot] on device d (token owner)
          )
          # also ships d_weights_sorted[d_slice] for routing grad
```

---

### Phase 1 Backward — Routing Gradient (local, on token-owner device)

```
d_recv_tokens    [262144, 7168]   token grads, in sorted order
d_weights_sorted [262144]         routing weight grads, in sorted order

         Reverse of gather (step 7):
Step B10 d_x_from_experts = zeros([32768, 7168])
         d_x_from_experts[sorted_token_ids] += d_recv_tokens      scatter-add

Step B11 d_x_flat += d_x_from_experts

         Assemble routing weight grad in original (T, K) layout:
Step B12 d_top_weights = zeros([32768, 8])
         d_top_weights.reshape(-1)[sort_idx] = d_weights_sorted    unsort

         Reverse of top_k (step 3):
         d_weights_full = zeros([32768, 256])
         d_weights_full[arange(T)[:,None], expert_ids] = d_top_weights

         Reverse of sigmoid (step 2):
Step B13 d_gate_logits = d_weights_full * weights * (1 - weights)   [32768, 256]

         Reverse of gate matmul (step 1):
Step B14 dW_gate  += x_flat.T @ d_gate_logits    [7168, 256]   (TC)
         d_x_flat += d_gate_logits @ W_gate       [32768, 7168] (TC)
```

---

## Operation Summary

### Forward

| Step | Operation | Shape in → out | Where |
|---|---|---|---|
| 1 | Gate matmul | [32768,7168] × [7168,256] → [32768,256] | TC (local) |
| 2 | Sigmoid | [32768,256] | elementwise |
| 3 | Top-K | [32768,256] → [32768,8] × 2 | local |
| 5 | Argsort | [262144] → [262144] | local |
| 7 | Gather tokens | [262144] → [262144,7168] | index (SC candidate) |
| 8 | DMA send tokens | [count_d, 7168] × 7 peers | point-to-point ICI |
| 10 | FFN up+gate | [tpe,7168] × [7168,4096] → [tpe,4096] | TC × 32 experts |
| 11 | FFN down | [tpe,2048] × [2048,7168] → [tpe,7168] | TC × 32 experts |
| 12 | DMA send outputs | [count_d, 7168] × 7 peers | point-to-point ICI |
| 13 | Scatter-add | [262144,7168] → [32768,7168] | index (not SC-able) |

### Backward

| Step | Operation | Shape | Where |
|---|---|---|---|
| B2 | Gather d_moe_out | [32768,7168] → [262144,7168] | index (SC candidate) |
| B3 | DMA send d_out | [count_d,7168] × 7 peers | point-to-point ICI |
| B5 | Weight scale grad | [tpe,7168] | elementwise |
| B6 | dW2, d_act | [tpe,7168]↔[2048,7168] | TC × 32 experts |
| B7 | SiLU grad | [tpe,2048] | elementwise |
| B8 | dW1, d_tok | [tpe,7168]↔[7168,4096] | TC × 32 experts |
| B9 | DMA send d_tokens | [count_d,7168] × 7 peers | point-to-point ICI |
| B10 | Scatter-add d_x | [262144,7168] → [32768,7168] | index (not SC-able) |
| B13 | Sigmoid grad | [32768,256] | elementwise |
| B14 | dW_gate, d_x | [32768,7168]↔[7168,256] | TC (local) |

### ICI Traffic Summary (per device, per MoE layer)

| Direction | Volume (BF16) | Time @ 200 GB/s A2A |
|---|---|---|
| Fwd: tokens → expert devices | 7.5 GB | 37.5 ms |
| Fwd: outputs → token owners | 7.5 GB | 37.5 ms |
| Bwd: d_out → expert devices | 7.5 GB | 37.5 ms |
| Bwd: d_tokens → token owners | 7.5 GB | 37.5 ms |
| Bwd: EP psum d_x (all-reduce) | 26 GB traffic | ~131 ms |
| **Total** | **~90 GB** | **~281 ms** |

Note: forward has 2 ICI transfers (fwd tokens + fwd outputs). Backward has 3 (d_out
send, d_tokens send, EP psum). The psum dominates.

---

## Step Time Estimate — Full Pallas Kernel (Optimized)

**Assumptions:** Full Pallas fwd+bwd, FSDP all-gather overlapped with compute,
SC-offloaded scatter/gather DMAs, A2A point-to-point DMAs pipelined across layers.
No EP all-reduce (replaced by A2A). K=6 (DSv3 actual, not K=8 above).

### Hardware

| Resource | Per device (v7x, 1 core) |
|---|---|
| MXU | 1,153 TFLOP/s (2,307 TFLOP/s per chip ÷ 2 cores) |
| HBM BW | ~3,686 GB/s (7,373 GB/s per chip ÷ 2) |
| Intra-chip ICI (co-core) | **600 GB/s** |
| Cross-chip ICI | ~90 GB/s per link |

### Per-device token counts (K=6)

| Quantity | Value | Derivation |
|---|---|---|
| T_local | 8,192 | GBS×seqlen / 512 devices = 1024×4096/512 |
| T_fsdp (after EP all-gather) | 65,536 | T_local × EP = 8192 × 8 |
| Total token-expert pairs | 393,216 | T_fsdp × K = 65536 × 6 |
| Pairs for local EP rank | 49,152 | 393216 / EP — no ICI |
| tpe (avg tokens per expert) | 1,536 | 49152 / E_local = 49152 / 32 |

### EP topology and traffic breakdown

EP=8 spans 4 chips × 2 cores. Each chip holds 2 EP ranks.

| Destination | Pairs | Volume | BW | Time |
|---|---|---|---|---|
| Local EP rank (own 32 experts) | 49,152 | 0 — no ICI | — | 0 ms |
| Co-chip EP rank (1 of 7 remote) | 49,152 | 703 MB | 600 GB/s | ~1 ms (free) |
| 6 cross-chip EP ranks | 49,152 each | 703 MB each | ~90 GB/s/link | bottleneck |

On a 4-chip ring, each chip has 2 direct ICI links in the EP dimension (to its 2
ring neighbors). The 2 ranks on the opposite chip require 2-hop routing, sharing
bandwidth with adjacent-chip transfers. Effective aggregate for 6 cross-chip peers
≈ 150 GB/s.

Cross-chip ICI per direction per MoE layer:
  6 peers × 703 MB = 4.2 GB @ 150 GB/s ≈ **28 ms**

### Compute (expert FFN, per device per MoE layer)

| Op | FLOPs | Time @ 1153 TFLOP/s |
|---|---|---|
| W1 fwd: 32 experts × tpe × D × 2F × 2 | 2,899 GFLOP | |
| W2 fwd: 32 experts × tpe × F × D × 2 | 1,449 GFLOP | |
| **Fwd total** | **4,348 GFLOP** | |
| **Fwd + bwd + recompute (4×)** | **17,391 GFLOP** | **15 ms** |

Expert FFN compute (15 ms) << A2A ICI (28 ms) → **ICI-dominated**.

With perfect layer-to-layer A2A pipelining (bwd d_tokens send of layer n overlaps
with fwd token send of layer n+1), the per-layer critical path ≈ 28 ms per direction.
With fwd and bwd further overlapped: effective ≈ **~56 ms per MoE layer**.

### Step time budget (58 MoE + 61 attention + 3 dense layers)

| Component | Time |
|---|---|
| 58 MoE layers × 56 ms (ICI-bound, pipelined A2A) | **3.2 s** |
| 61 MLA attention layers (compute-bound) | ~0.13 s |
| 3 dense MLP layers | ~0.02 s |
| FSDP weight all-gather (overlapped with compute) | 0 on critical path |
| **Total** | **~3.4 s** |

### Summary

| Scenario | Step time | TPS/chip |
|---|---|---|
| Optimistic (perfect A2A pipeline, 150 GB/s agg) | 3.4 s | ~4,800 |
| Realistic (60% pipeline efficiency) | 5.5 s | ~2,960 |
| Conservative (40% efficiency) | 8 s | ~2,040 |
| v10 ragged_dot baseline | ~35 s | 465 |

**Key levers in order of impact:**
1. A2A DMA pipeline efficiency (biggest unknown — sets 3-8 s range)
2. Elimination of EP all-reduce (saves ~131 ms/layer × 58 = 7.6 s vs current streaming_bwd)
3. Co-chip 600 GB/s path reduces effective remote ICI from 7 to 6 cross-chip hops
4. FSDP all-gather overlap hides ~5.6 GB/layer weight gather in compute shadow
5. SC offload frees MXU from scatter/gather (marginal — compute is not the bottleneck)
