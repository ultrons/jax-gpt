---
target: dsv3-fused-ep-moe
status: DRAFT v0.6 — §3 / §4 v_inside reconciled with §5.1 F-sharded layout (Megatron column+row parallel); supersedes v0.5
authors: vaibhav (architecture), claude (formalization)
last-updated: 2026-05-12
---

# DSv3 Fused EP-MoE Kernel — SPEC

The architecture the kernel-agent builds. **The agent does not derive any of this**; it reads this SPEC and produces kernels that satisfy it.

Where this SPEC says `[DECIDE]`, the user picks; where it states a value, that's binding architecture. Where it says `[DERIVED]`, the agent computes the value at build time.

---

## 1. Scope

**Two Pallas kernel variants** (see §2 — Variants), each performing an entire MoE block per layer:

- **Input**: `x_in` (post-attention activations) + per-layer weight tensors (router, experts), all with logical-axis sharding.
- **Output**: `x_out = x_in + moe_residual_contribution`.

Inside each kernel: routing, sort, top-K filtering, per-expert FFN, A2A scatter+gather, weighted unsort+combine, residual add. The variants differ in whether FSDP all-gather of expert weights is fused inside the kernel (`v_inside`) or handled by JAX outside (`v_outside`).

**Sharding model.** The kernel is integrated via `shard_map`. The contract uses **logical axis names** (e.g. `embed`, `experts`, `mlp`, `joined_heads`); the mapping logical → physical mesh axes is defined by the user's `LOGICAL_AXIS_RULES` outside this SPEC and is not the kernel's concern. Inside the kernel, axes are referred to by the `shard_map` axis name (e.g. `ep_axis_name="ep"`).

### What's IN scope
- Forward + backward (both, from day 1 — see §4)
- BF16 throughout
- Two AG variants (inside / outside; see §2)
- A bench harness with attention glue and weight AG (§7) — kernel is tested in a realistic block, not standalone

### What's explicitly OUT of scope
- Attention itself (separate kernel/JAX) — but the bench harness includes a representative attention block before and after the MoE call so that we exercise the surrounding XLA scheduling
- Layer norms (JAX, before/after kernel input/output)
- Residual stream itself (kernel writes into it; doesn't own the global stream)
- FP8 / Int4 / FP4 / quantized variants — separate targets
- Flash-attention numerical refinements

---

## 2. Variants

We ship two kernel variants in this target. Both same algorithm; differ only in where the FSDP all-gather of `W_gate`, `W1`, `W_d` happens.

### `v_inside` — primary deliverable

FSDP all-gather of `W_gate`, `W1`, `W_d` is **fused inside** the Pallas kernel. Each weight is streamed in shard-by-shard via `async_remote_copy` from FSDP peers; matmul accumulates `+=` over shards in VMEM. This is the "absolute control" path — the kernel owns the entire weight-data-movement schedule.

### `v_outside` — baseline / counterfactual

FSDP all-gather is done by JAX/XLA **outside** the kernel (`jax.lax.all_gather` on each weight before the kernel call). The kernel receives full-F weights and does only the local matmul. This is what `fused_moe_kernel_explainer §5.1` describes for v1 W1/W2.

**Why both:** v_inside owns the schedule; v_outside lets XLA schedule. Empirically — does fusing inside the Pallas/Mosaic boundary box XLA out of useful interleaving with surrounding compute (attention prefetches, etc.)? Or does XLA still do meaningful work after the Mosaic lowering pass to LLO? **The bench harness measures both with the same attention glue.** Whichever wins becomes the recommended path; if v_inside wins by a small margin and v_outside is much simpler, we may keep v_outside as the default and v_inside as the "absolute control" specialty.

Both variants share §3-§4 architecture; §5 contracts; §7 bench harness; §8-§9 perf and budget targets.

---

## 3. Architecture — Forward Pass

Four phases, all inside one Pallas kernel. Phase boundaries are logical; in the kernel body they may overlap (Phase 2 DMA send proceeds while Phase 1 routing finishes for later tokens).

### Phase 1 — Routing (local, no comms)

| Step | Op | In | Out | Engine |
|---|---|---|---|---|
| 1.1 | gate matmul | `x_in [T_local, D]`, `W_gate [E, D]` | `gate_logits [T_local, E]` | TC MXU |
| 1.2 | scoring | `gate_logits` | `weights = sigmoid(gate_logits) [T_local, E]` | VPU |
| 1.3 | top-K | `weights` | `expert_ids [T_local, K]`, `top_weights [T_local, K]` | TC iterative-argmax (K iterations of `argmax + mask-to-neg-inf`; Mosaic doesn't natively support bf16 argmax — explainer §5.7) |
| 1.4 | renormalize | `top_weights` | `top_weights /= top_weights.sum(-1, keepdims=True)` (DSv3-style) | VPU |
| 1.5 | flatten | `expert_ids`, `token_ids` | `expert_ids_flat [T_local*K]`, `token_ids_flat [T_local*K]` | VPU |
| 1.6 | sort by expert | `expert_ids_flat` | `sort_idx`, `sorted_expert_ids`, `sorted_token_ids` | SC (sort/scatter is its design use case; chunked per C1: ≤65,536 rows per SC gather) |
| 1.7 | histogram + cumsum | `sorted_expert_ids` | `send_counts [E]`, `send_offsets [E+1]` | VPU |
| 1.8 | send-counts metadata exchange | `send_counts [E]` | `all_send_counts [EP, E] int32` (every peer's send_counts visible to every peer) | ICI all-gather along EP axis (small payload: EP·E·4 = ~4 KB at production) |

**Routing math (binding):** `weights = sigmoid(gate_logits)`; top-K by score; `top_weights[i, :] /= top_weights[i, :].sum()`. Matches `fused_ep_moe_v1__kernel.py` `apply_scoring_fn(scoring_fn="sigmoid")` + `renormalize_topk_logits=True`.

**Step 1.8 rationale:** before Phase 2 can issue `async_remote_copy` writes, every peer must know how many tokens every other peer is sending it (to compute its own `recv_offsets` and detect completion). Production v1 (`fused_ep_moe_v1__kernel.py:393` `all_reduce_metadata`) handles this as an all-gather of per-peer send_counts. Cost is trivial (a few KB along the EP axis) vs the data A2A in 2.2.

### Phase 2 — Pack + A2A scatter

| Step | Op | Detail |
|---|---|---|
| 2.1 | gather sorted tokens | `sorted_tokens = x_in[sorted_token_ids]` shape `[T_local*K, D]` (TC gather; no SC fallback in v0 — antipattern C3 mis-lowers `plsc.BlockSpec` for 2-D sources, and a pure-Pallas SC gather is post-v0 work) |
| 2.2 | per-peer remote DMA | for each EP peer `d`: `async_remote_copy(sorted_tokens[lo:hi] → recv_buf[my_slot] on device d)`, also ships `sorted_expert_ids[lo:hi]` and `top_weights[lo:hi]`. Recv-side offsets derived from `all_send_counts` (step 1.8) |
| 2.3 | barrier | all EP peers signal+wait on shared barrier semaphore before Phase 3 reads `recv_buf` |

**Idiom:** point-to-point `async_remote_copy`, NOT collective `all_to_all`. Tokens with no assignment to peer `d` are not transmitted (sparse pattern).

### Phase 3 — Expert FFN with weight AG

For each local expert `e ∈ [0, E_local)` (Python for-loop unrolled at trace time per anti-pattern G3):

| Step | Op | Shape | `v_inside` | `v_outside` |
|---|---|---|---|---|
| 3.1 | extract per-expert tokens | `tok_e = recv_buf[exp_off[e]:exp_off[e+1]]` shape `[tpe_e, D]` | VMEM ref slice | VMEM ref slice |
| 3.2 | fused gate+up | `gate_up = tok_e @ W1[e]` → `gate, up = split` shape each `[tpe_e, F]` | TC, **column-parallel**: each device's `W1_shard[e]: (D, 2F_shard)` contracts on D, produces `gate_up_local: (tpe_e, 2F_shard)` — local F slice, **no comm** | TC, W1[e] is full-F (already AG'd outside); single matmul |
| 3.3 | activation + multiply | `act = silu(gate) * up` shape `[tpe_e, F_shard]` (v_inside) or `[tpe_e, F]` (v_outside) | VPU | VPU |
| 3.4 | down matmul | `out_e = act @ W_d[e]` shape `[tpe_e, D]` | TC, **row-parallel**: each device's `W_d_shard[e]: (F_shard, D)` contracts on F_shard, produces `out_e_partial: (tpe_e, D)` partial-sum, then `lax.psum` (or streaming-psum-scatter — `distilled/patterns/streaming-psum-scatter.md`) across fsdp axis to reduce partials | TC, W_d[e] is full-F; single matmul |
| 3.5 | route weight scale | `out_e *= top_weights_e_per_token` (per-token scalar; broadcast over D) | VPU | VPU |
| 3.6 | store | write `out_e → output_buf[exp_off[e]:exp_off[e+1]]` | VMEM | VMEM |

**Activation math (binding):** `silu(gate) * up` where `silu(x) = x * sigmoid(x)`. The "fused gate+up" naming refers to the layout `W1 = concat(W_gate_proj, W_up_proj, axis=-1)` — one matmul produces both halves.

### Phase 4 — A2A gather + weighted combine

| Step | Op | Detail |
|---|---|---|
| 4.1 | per-peer remote DMA back | for each EP peer `d`: `async_remote_copy(output_buf[d_slice] → result_buf[my_slot] on device d)` |
| 4.2 | barrier | wait for all EP peers' results |
| 4.3 | scatter-add (unsort + combine) | `moe_out = zeros([T_local, D]); moe_out[sorted_token_ids] += result_buf` — `segment_sum` over the K=8 contributions per token. **Implemented as an explicit SC `pallas_call` (gather_reduce) invoked from JAX glue between the main TC kernel's `result_buf` output and the residual add in 4.4 — NOT inside the main TC `pallas_call`.** Reference: `corpus/kernels/sparse_core_upstream__gather_reduce.py` (canonical upstream) or `dsv3_prod__gather_reduce_sc.py` (production wrapper). A single `pl.pallas_call` body cannot span TC and SC backends; the SPEC's "one Pallas kernel" framing in §1 is approximate — see PHASE_A_PLAN.md §10.3 for the decomposition (1× TC pallas_call + thin JAX glue + 1-2 SC pallas_calls, all composed under one `custom_vjp`). |
| 4.4 | residual add | `x_out = x_in + moe_out` (JAX glue, after the SC scatter-add in 4.3) |

---

## 4. Architecture — Backward Pass

Same 4 phases run in reverse, each conjugate of forward. **Routing residuals** (small) are saved per E4. **Activation residuals** (large) follow the policy in §5.2 — by default everything >32 MB is recomputed in bwd; with optional host-offload as a third path. The per-expert bwd loop in Phase 3' recomputes `gate`, `up`, `act`, `out_e_pre_scale` from saved `tok_e`-rederivation and `W1[e]`, `W_d[e]`.

### 4.0 Bwd preamble — reconstruct large activations

**Step 0a — re-execute Phase 2 A2A scatter to reconstruct `tok_e_buf`.**

`tok_e_buf` is 30 GB at production (per device); we don't save it. The bwd starts by re-running the forward A2A scatter using saved routing residuals (`sorted_token_ids`, `expert_offsets`, etc.):

```
for each EP peer d:
    async_remote_copy(
        src = x_in[sorted_token_ids][lo:hi],   # x_in is the kernel input; saved routing tells us what to send
        dst = recv_buf[my_slot] on device d
    )
barrier
# recv_buf now contains tok_e_buf, identical to forward
```

This costs an extra A2A (~37.5 ms / layer per whiteboard §ICI) but saves 30 GB of HBM. Below the 32 MB threshold this trade-off would be unfavorable; above it, recompute wins.

**Step 0b — per-expert recompute happens inside Phase 3' loop**, not as a global preamble. See Phase 3' below.

### Phase 4 backward — reverse residual + un-combine

| Step | Op | Detail |
|---|---|---|
| 4'.1 | residual | `d_x_in += d_x_out` (carried out of kernel) and `d_moe_out = d_x_out` |
| 4'.2 | un-combine | for each token, K-way duplicate: `d_result_buf = d_moe_out[sorted_token_ids]` |
| 4'.3 | per-peer remote DMA back | for each EP peer `d`: `async_remote_copy(d_result_buf[my_slot] → d_output_buf[d_slice] on device d)` |
| 4'.4 | barrier | |

### Phase 3 backward — expert FFN backward (with per-expert recompute)

For each local expert `e ∈ [0, E_local)`:

| Step | Op | Shape | Notes |
|---|---|---|---|
| 3'.0a | extract tok_e | `tok_e = recv_buf[exp_off[e]:exp_off[e+1]]` (rederived in §4.0a) | VMEM ref slice |
| 3'.0b | **recompute** gate+up | `gate_up = tok_e @ W1[e]`; `gate, up = split` | TC; column-parallel local matmul (`v_inside`, no comm) or full-F single matmul (`v_outside`) — same as fwd 3.2 |
| 3'.0c | **recompute** activation | `act = silu(gate) * up` | VPU |
| 3'.1 | extract per-expert d_out | `d_out_e = d_output_buf[exp_off[e]:exp_off[e+1]]` | VMEM ref slice |
| 3'.2a | un-scale d_out | `d_out_e_unscaled = d_out_e * top_weights_e_per_token` | VPU |
| 3'.2b | d_top_weights (per-token) | for each token-block: `out_e_pre_scale_block = act_block @ W_d[e]` (block-by-block, never materialized full); `d_top_weights_e_block += sum(d_out_e_block * out_e_pre_scale_block, axis=-1)` | TC matmul re-run; D-vector dot product per token, cheap |
| 3'.3 | down-matmul backward | `d_act = d_out_e_unscaled @ W_d[e]`; `d_W_d[e] += act.T @ d_out_e_unscaled` | E1: VMEM `+=` for `d_W_d` |
| 3'.4 | activation backward | `d_gate = d_act * up * silu'(gate)` where `silu'(x) = sigmoid(x) * (1 + x*(1-sigmoid(x)))`; `d_up = d_act * silu(gate)` | VPU |
| 3'.5 | gate+up matmul backward | `d_tok_e = concat(d_gate, d_up) @ W1[e]`; `d_W1[e] += tok_e.T @ concat(d_gate, d_up)` | E1: VMEM `+=` for `d_W1`. **No HBM bin pre-allocation** (E2). **No (T×K, D, F) intermediate** (E3). |
| 3'.6 | store d_tok_e | write `d_tok_e → d_recv_buf[exp_off[e]:exp_off[e+1]]` | |

The 3'.0a-c steps (recompute) cost approximately one fwd Phase 3 per expert, doubling the kernel's compute relative to "save everything." At HBM-bound MoE workloads, the extra compute is hidden behind the same HBM reads we'd do anyway — recompute is closer to free than the 2× FLOP count suggests.

After loop: write accumulated `d_W1`, `d_W_d` from VMEM to HBM (single write per weight per expert).

For `v_inside` (Megatron column+row parallel; see `_inbox/blocker-spec-v_inside-sharding-vs-math.md` for the §5.1↔§3 reconciliation): each device's accumulator is sized for its local F shard. `d_W1` accumulator is `(E_local, D, 2F_shard) f32` (column-parallel: each device owns a F-slice of d_W1, no comm needed for d_W1). `d_W_d` accumulator is `(E_local, F_shard, D) f32`. The cross-fsdp `d_x_in` gradient (from the row-parallel down-matmul's bwd) is a partial along fsdp that requires `lax.psum` (or streaming-psum-scatter — `distilled/patterns/streaming-psum-scatter.md`). For `v_outside`, the kernel writes full-F grads and JAX outside does the psum_scatter.

### Phase 2 backward — A2A return d_tok to token-owner devices

| Step | Op | Detail |
|---|---|---|
| 2'.1 | per-peer remote DMA back | for each EP peer `d`: `async_remote_copy(d_recv_buf[d_slice] → d_sorted_tokens_buf[my_slot] on device d)` |
| 2'.2 | barrier | |
| 2'.3 | unsort | `d_x_pre_residual = zeros([T_local, D]); d_x_pre_residual[sorted_token_ids] += d_sorted_tokens_buf` (segment_sum over K=8). **Same pattern as fwd 4.3 — explicit SC `pallas_call` (gather_reduce), invoked from JAX glue, NOT inside the main TC bwd kernel.** |

### Phase 1 backward — routing gradient

| Step | Op | Detail |
|---|---|---|
| 1'.1 | gather d_top_weights for non-local slots | E5: zero non-local slots before scatter (`d_top_weights = where(is_local, d_top_weights, 0)`) |
| 1'.2 | un-renormalize (canonical VJP for `y = x / sum(x)`) | `s = sum(top_weights_unnorm, axis=-1, keepdims=True)`; `inner = sum(d_top_weights * top_weights_renorm, axis=-1, keepdims=True)`; `d_top_weights_unnorm = (d_top_weights - inner) / s` — see citation below |
| 1'.3 | scatter d_top_weights_unnorm back to E-wide | `d_weights = zeros([T_local, E]); d_weights.at[token_idx, expert_ids].add(d_top_weights_unnorm)` (E5 zeroing applies). **Same pattern as fwd 4.3 — explicit SC `pallas_call` (gather_reduce variant), invoked from JAX glue, NOT inside the main TC bwd kernel.** |
| 1'.4 | sigmoid backward | `d_gate_logits = d_weights * sigmoid(gate_logits) * (1 - sigmoid(gate_logits))` | VPU |
| 1'.5 | gate matmul backward | `d_x_routing = d_gate_logits @ W_gate`; `d_W_gate += x_in.T @ d_gate_logits` | TC; `d_W_gate` accumulated VMEM `+=` |
| 1'.6 | combine input grads | `d_x_in += d_x_pre_residual + d_x_routing` (residual contributes via 4'.1) | VPU |

**All 5 backward anti-patterns enforced** (E1-E5; see `distilled/antipatterns/jax-mosaic-rules.md §E`).

**Citation for §4 Phase 1' step 1'.2 (renormalize VJP):** The canonical formula
`d_top_w_unnorm = (d_top_w - <d_top_w, top_w_renorm>) / s` is implemented at
`corpus/kernels/fused_moe_bwd__backward.py:279-280` in production code:
```python
d_logits = (d_logits - jnp.sum(d_logits * top_k_logits, axis=-1, keepdims=True)) / s
```
NOTE: that file's comment block on lines 270-271 contains an incorrect informal derivation (says `renorm * sum(d_renorm)` instead of `<d_renorm, renorm>`); the **code is correct, the comment is buggy**. SPEC v0.2 was wrong against the code; v0.3 fixes to match the code.

The math: for `y[k] = x[k] / sum(x)`, the VJP is `d_x[k] = (d_y[k] - sum_j(d_y[j] * y[j])) / sum(x)`. This is a standard L1-normalize backward (softmax-without-exp). JAX autodiff produces the same formula when applied to `y = x / x.sum(-1, keepdims=True)`.

---

## 5. Contracts (kernel boundary)

### 5.1 Inputs

Logical-axis names (mapped to physical mesh by user's `LOGICAL_AXIS_RULES`). Inside `shard_map`, axes referenced by `shard_map` axis name.

| Name | Logical shape | Logical axis sharding | Inside-shard_map shape (at EP=4 FSDP=128) | Dtype |
|---|---|---|---|---|
| `x_in` | `(T_global, D)` | `(seq, embed)` → `(fsdp, None)` | `(T_local=T_global/fsdp, D)` | bf16 |
| `W_gate` | `(E, D)` | `(experts_router, embed)` → `(None, None)` (replicated) | `(E, D)` | bf16 |
| `W1` (gate+up fused) | `(E, D, 2F)` | `(experts, embed, mlp)` → `(ep, None, fsdp)` | `(E_local=E/ep, D, 2F_shard=2F/fsdp)` | bf16 |
| `W_d` | `(E, F, D)` | `(experts, mlp, embed)` → `(ep, fsdp, None)` | `(E_local, F_shard, D)` | bf16 |

`v_inside` receives the inside-shard_map shapes (FSDP-sharded) and gathers internally. `v_outside` receives full-F shapes (JAX has already done the AG before calling).

### 5.2 Outputs + residual policy

#### Output

| Name | Logical shape | Sharding | Inside-shard_map shape | Dtype |
|---|---|---|---|---|
| `x_out` | `(T_global, D)` | `(seq, embed)` → `(fsdp, None)` | `(T_local, D)` | bf16 |

#### Residual policy (per residual: SAVE-HBM / OFFLOAD-HOST / RECOMPUTE)

**Rule (binding):** any residual >32 MB per device is RECOMPUTE by default. SAVE-HBM is allowed only for residuals ≤32 MB. OFFLOAD-HOST is an opt-in alternative for the recompute set.

The agent honors per-residual policy declarations from this table; the default column is what's used unless the user overrides via a policy file (`targets/dsv3-fused-ep-moe/residual_policy.yaml`).

Sizes computed at production scale (DSv3 671B, EP=4, FSDP=128, BS=2048, seq=4096 → T_local=65,536, E_local=64, F_shard=16, max_tpe=16,384):

| Residual | v_inside size | v_outside size | Default policy | Notes |
|---|---|---|---|---|
| `tok_e_buf` (E_local, max_tpe, D) bf16 | 15 GB | 15 GB | RECOMPUTE | re-A2A in §4.0a; saved routing residuals tell us what to send |
| `gate_buf` (E_local, max_tpe, F_or_Fshard) bf16 | 32 MB | 4 GB | RECOMPUTE | recompute in §3'.0b inside per-expert loop. v_inside lands exactly at the 32 MB SAVE-HBM threshold; kept RECOMPUTE for symmetry with v_outside and zero residency cost |
| `up_buf` | 32 MB | 4 GB | RECOMPUTE | same as gate_buf |
| `act_buf` (post SiLU) | 32 MB | 4 GB | RECOMPUTE | recompute in §3'.0c |
| `out_e_pre_scale` (E_local, max_tpe, D) bf16 | 15 GB | 15 GB | RECOMPUTE | computed block-by-block in §3'.2b, never materialized full |
| `top_weights_renorm` (T_local, K) bf16 | 1.0 MB | 1.0 MB | SAVE-HBM | needed for §3.5 scaling & §1'.2 inner-product |
| `top_weights_unnorm_sum` (T_local, 1) bf16 | 0.13 MB | 0.13 MB | SAVE-HBM | the `s` denominator for §1'.2 |
| `expert_ids` (T_local, K) int32 | 2.1 MB | 2.1 MB | SAVE-HBM | needed for §1'.3 scatter and §4.0a re-A2A |
| `sorted_token_ids` (T_local*K,) int32 | 2.1 MB | 2.1 MB | SAVE-HBM | §4.0a re-A2A |
| `sort_idx` (T_local*K,) int32 | 2.1 MB | 2.1 MB | SAVE-HBM | invertible sort permutation |
| `expert_offsets` (E_local+1,) int32 | trivial | trivial | SAVE-HBM | per-expert slicing |
| `send_offsets`, `send_counts` (E+1,) int32 | trivial | trivial | SAVE-HBM | A2A peer slicing |

`max_tpe` is computed at trace time as `cdiv(2 * T_local * K / E_local, 128) * 128` (per fused_moe_kernel_explainer §5.6 — 2× avg with rounding).

#### OFFLOAD-HOST option

For any residual currently marked RECOMPUTE, the user may override to OFFLOAD-HOST in `residual_policy.yaml`. The kernel then DMAs the residual to host RAM via PCIe asynchronously during forward, and prefetches back asynchronously during backward.

**PCIe budget:** v7x PCIe per chip ≈ 12 GB/s, per-core ~6 GB/s (~600× slower than HBM, ~30× slower than ICI). Practical use:

| Residual size | Recompute cost | OFFLOAD-HOST cost | When to choose offload |
|---|---|---|---|
| `tok_e_buf` 15 GB | ~18.75 ms re-A2A | ~1.25 s PCIe (per chip) | NEVER — re-A2A is ~65× faster |
| `gate_buf, up_buf, act_buf` 32 MB (v_inside) | ~few ms recompute | ~2.7 ms PCIe | comparable; recompute is the safer default |
| `gate_buf, up_buf, act_buf` 4 GB (v_outside) | ~few ms recompute | ~333 ms PCIe | recompute always wins |
| Hypothetical residual 50-100 MB that's expensive to recompute (e.g. requires a full-D matmul on a subset of the kernel) | ~10 ms recompute | ~5-8 ms PCIe (hidden behind compute if overlapped) | offload may win |

**Offload is pragmatically rare for this kernel** — recompute always beats it for our specific residual shapes. We ship the option for completeness and for future kernels where the trade-off changes (e.g. attention KV cache, where recompute IS expensive).

The agent must implement OFFLOAD-HOST as a working code path even if no residual chooses it by default, because future targets will use it.

### 5.3 Static parameters (compile-time)

| Param | Default for DSv3 671B | Notes |
|---|---|---|
| `E` (global experts) | 256 | DSv3 paper |
| `D` (hidden) | 7168 | DSv3 paper |
| `F` (FFN intermediate) | 2048 | DSv3 paper |
| `K` (top-k) | 8 | matches v1 production + DSv3 paper |
| `BS` (batch size) | 2048 | production training shape |
| `seq` (sequence length) | 4096 | production training shape |
| `T_global = BS × seq` | 8,388,608 | total tokens per step |
| `bt`, `bd`, `bf` (block sizes) | from `tuned_block_sizes.py` | `[DERIVED]` by agent |

Derived (set by mesh shape):
- `E_local = E / ep_size`
- `F_shard = F / fsdp_size`
- `T_local = T_global / fsdp_size`

### 5.4 Mesh contract

```
mesh = Mesh(devs, ('dp', 'ep', 'fsdp', 'tp'))
```

| Surface | Mesh | Total devices |
|---|---|---|
| Production | `(dp=1, ep=4, fsdp=128, tp=1)` | 512 (= 256 chips × 2 cores) on 4×8×8 v7x |
| Iteration (bodaborg) | `(dp=1, ep=2, fsdp=8, tp=1)` | 16 (= 8 chips × 2 cores), 2× tpu7x-standard-4t cross-host |
| AOT virtual | `tpu7x:2x2x2 / 4x8x8` | matches the cluster of intent |

`tp=1` for v0; tensor-parallel inside the MoE block isn't part of the v0 architecture.

**Production-derived constants** (with §5.3 + production mesh):
- `T_local = 8,388,608 / 128 = 65,536`
- `E_local = 256 / 4 = 64`
- `F_shard = 2048 / 128 = 16`
- `max_tpe = cdiv(2 × T_local × K / E_local, 128) × 128 = cdiv(2 × 65,536 × 8 / 64, 128) × 128 = 16,384` (well below the SC C1 ceiling of 65,536)

**v0.3 → v0.4 changelog:** v0.3 specified production as `(ep=8, fsdp=64)` with `BS=4096, seq=2048`. The corrected production sharding is `(ep=4, fsdp=128)` with `BS=2048, seq=4096`. `T_global = BS × seq` is unchanged (8.4M tokens). §5.1 caption, §5.2 residual size table, and §5.2 PCIe budget table are all recomputed at v0.4 sharding. Phase A plan §9.1 has the per-shape recompute and §9.2 confirms Q11 (HBM OOM at v0.3 sharding) is resolved at v0.4 sharding.

**v0.4 → v0.5 changelog:** Dropped the `.T` in §3 step 3.2 (`gate_up = tok_e @ W1[e]`) and step 3.4 (`out_e = act @ W_d[e]`), plus the matching §4 step 3'.0b recompute and §4 step 3'.2b reference. v0.4's `.T` notation was inconsistent with the §5.1 weight shapes (`W1: (E, D, 2F)`, `W_d: (E, F, D)`): `tok_e @ W1[e].T` would need W1[e].T of shape `(2F, D)` but `W1[e]` is `(D, 2F)`. The implementation (`build/v_outside/expert_ffn.py` since B.1) has always used the no-transpose form per `_inbox/blocker-spec-matmul-transpose-nit.md`. Pure cosmetic SPEC fix; no algorithmic or shape change. Bwd matmuls in §3'.3 / §3'.5 / §1'.5 keep their `.T` on activations/inputs (e.g. `tok_e.T @ d_concat`) — those are LEGITIMATE transposes (transposing the data tensor, not the weight).

**v0.5 → v0.6 changelog:** Reconciled §3 (and §4 bwd) v_inside math with §5.1 F-sharded weight layout. v0.5's v_inside column described "streaming AG of W1[e] with `gate_up += tok @ W1_shard[s]`", which is consistent only with D-sharded W (each peer holds a D-slice; `+=` sums partials over D contractions). The §5.1 sharding is F-sharded: each peer holds a 2F-slice; `tok @ W1_shard` produces an output F-slice, not a partial sum — `+=` is the wrong operator. v0.6 replaces "streaming AG with `+=`" with **Megatron column-parallel (gate+up; no comm)** + **row-parallel (down; psum across fsdp)**. The optional streaming optimization for the down-matmul psum is documented in `distilled/patterns/streaming-psum-scatter.md`. Full reconciliation: `distilled/_inbox/blocker-spec-v_inside-sharding-vs-math.md`. No change to §5.1 sharding or §3 fwd math (only the v_inside operator description).

---

## 6. Math reference (the JAX equivalent)

`targets/dsv3-fused-ep-moe/jax_ref.py` (to be written): a pure-JAX implementation of §3-§4 math, same scoring/normalization/activation/A2A pattern, used as the numerical ground truth for G2 / G3.

**The same JAX file also serves as the §8 perf baseline** (with no kernel call — "pure JAX with same architecture, no fusion glue before/after"). The kernel must be ≥ this baseline.

---

## 7. Bench harness

The kernel is **never tested in isolation.** Bench code emulates a realistic transformer block:

```
for each MoE layer:
  x = LayerNorm(x)
  q, k, v = attention_qkv_proj(x)         # JAX, with own weight AG
  attn_out = attention(q, k, v)            # JAX or stub kernel
  x = x + attn_out                         # residual

  x = LayerNorm(x)
  x = moe_kernel(x, W_gate, W1, W_d)       # the kernel under test
  # x already has residual baked in by the kernel (Phase 4.4)
```

Bench harness specifies:
- Attention glue: own weight AG, own logical-axis sharding, representative compute volume
- Logical axis rules matching the production setup
- `shard_map` integration of the MoE kernel (NOT raw `pallas_call` from the top level)
- Repeated layers (e.g. 3 MoE layers) so we measure steady-state, not first-layer warmup
- Both forward and backward measured (when bwd ships)

This is what the user means by "real setup" — the bench tests whether `v_inside`'s schedule fights or composes with the surrounding XLA scheduling.

`targets/dsv3-fused-ep-moe/bench.py` (to be written) is the artifact.

---

## 8. Performance targets

Three reference points against which the kernel is measured. **Targets apply to BOTH variants** (`v_inside`, `v_outside`); we report all three for both.

### 8.1 Pure JAX baseline (lower bound — kernel must beat this)

Same architecture in pure JAX (`jax_ref.py`), same sharding, same bench harness, no fusion. The kernel must be **at least at par or better**. This is a low bar — if a Pallas kernel can't beat naive JAX-with-collectives, it's not earning its complexity.

### 8.2 Production v1 (target — match or close to)

`fused_ep_moe_v1` (forward only — bwd is JAX in production). Within 2× of v1 fwd time on production shapes is the v0 success criterion; matching v1 is the v1 success criterion. For backward there is no v1 production to compare to (v1 falls back to JAX bwd) — bwd target is "≤ pure-JAX backward time".

### 8.3 Roofline (upper bound — how much headroom remains)

xla-shell `report_roofline` + `llo_analysis` per phase. For each phase, report:
- `bound_by` (HBM, COMPUTE, or DMA)
- `mxu_util` (target >40%)
- `dma_overlap` (target >70%)
- `gap_to_roofline` (measured / max(roofline_compute, roofline_hbm))

Headroom ≤ 30% means we're done; >30% means there's a phase we should investigate.

---

## 9. Memory budget (per device)

| Bucket | Budget | Notes |
|---|---|---|
| VMEM total | 64 MB per core | hardware-spec.md §1 |
| Per-buffer (double-buffered) | ≤30 MB | half VMEM minus headroom |
| HBM weight residency | `[DERIVED]` | EP=8, FSDP=64 → ~135 MB W1+W_d per device (`v_inside`); ~8.6 GB per device (`v_outside`, full-F W1+W_d) |
| Program binary contiguous | reserve ≥15 GB contiguous | RuntimeProgramAllocationFailure prevention |
| Activation HBM (forward residency for bwd) | `[DERIVED]` | T_local × D × ~10 saved tensors × 2 bytes |

`[ACTION]`: agent computes per-buffer VMEM allocation budget at Phase A and **fails the design check** if any allocation exceeds 30 MB per buffer. This is the gate that would have caught v3's 917 GB OOM at design time.

---

## 10. Idioms (which substrate patterns to use)

The agent draws from these. Production v1 source code in `corpus/kernels/` is **reference for understanding**, not a blueprint to copy.

| Idiom | Substrate doc | Used in phase |
|---|---|---|
| pallas_call skeleton | `distilled/patterns/pallas-call-skeleton.md` | All |
| AOT compile gate | `distilled/patterns/aot-compile-gate.md` | Pre-submit |
| Mosaic constraints (20 rules) | `distilled/antipatterns/jax-mosaic-rules.md` | Self-lint |
| Double-buffered DMA | `distilled/patterns/double-buffered-dma.md` `[2B-PENDING]` | Phase 3 weights |
| async_remote_copy + EP barrier | `distilled/patterns/async-remote-copy-ep.md` `[2B-PENDING]` | Phases 2, 4, AG inside Phase 3 (`v_inside`), all of bwd phases 4', 2' |
| Streaming AG fused into matmul | `distilled/patterns/streaming-ag-into-matmul.md` `[2B-PENDING]` | Phase 3 matmuls (`v_inside` only) |
| Streaming psum_scatter (dual of AG) | `distilled/patterns/streaming-psum-scatter.md` `[2B-PENDING]` | Phase 3 backward (`v_inside`); writes d_W1 / d_W_d back to FSDP-sharded HBM |
| Iterative argmax for top-K | `distilled/patterns/iterative-argmax-topk.md` `[2B-PENDING]` | Phase 1 step 1.3 |
| Scatter-add via segment_sum | `distilled/patterns/scatter-add-segment-sum.md` `[2B-PENDING]` | Phase 4 step 4.3, Phases 2'.3, 1'.3 |
| VMEM `+=` weight grad accumulation | `distilled/patterns/vmem-plus-equals-weight-grad.md` `[2B-PENDING]` | Phase 3 backward (E1) |
| Residual policy: save / offload / recompute | `distilled/patterns/residual-policy.md` `[2B-PENDING]` | All residuals per §5.2 — three-way choice with PCIe budget reasoning |
| PCIe host-offload async DMA | `distilled/patterns/pcie-host-offload-dma.md` `[2B-PENDING]` | Per-residual offload path |
| Re-A2A scatter for tok_e reconstruction | `distilled/patterns/re-a2a-scatter-recompute.md` `[2B-PENDING]` | §4.0a; same DMA pattern as §3 Phase 2, just invoked at bwd start |

---

## 11. Validation plan (how the agent knows it's done)

| Gate | Pass criterion | Source |
|---|---|---|
| G0 self-lint | 20-rule checklist clean | jax-mosaic-rules §H |
| G1 AOT compile | Mosaic compiles cleanly on `tpu7x:2x2x2` (and 4x8x8 for production AOT) | aot-compile-gate.md |
| G2 math correctness fwd | `assert_allclose(kernel_out, jax_ref_out, rtol=1e-2)` at small shapes, 1 chip | jax_ref.py |
| G2-bwd math correctness bwd | `jax.grad` round-trip vs jax_ref autograd, rtol=5e-2 | jax_ref.py + custom_vjp |
| G3 EP=2 round-trip | G2 + G2-bwd at `tpu7x:2x2x2` (16 devices, EP=2) on bodaborg | bench.py |
| G4 production scale numerical | G2 + G2-bwd at production shapes | ninja-v7x-64 |
| G5 perf vs pure-JAX baseline | both variants ≥ JAX baseline | bench.py |
| G6 perf vs v1 (fwd) | both variants within 2× of v1 fwd | bench.py side-by-side |
| G7 roofline analysis | `gap_to_roofline ≤ 30%` per phase, OR root cause documented | xla-shell |

---

## 12. Open decisions summary

All resolved unless noted.

| # | Question | Answer |
|---|---|---|
| 1 | AG W_gate, W1, W_d outside or inside? | **Both:** `v_inside` primary, `v_outside` baseline. Test against same bench harness. |
| 2 | BF16-only for v0? | **Yes** (FP8 follow-up target) |
| 3 | top-K via SC or TC iterative-argmax? | **TC iterative** (Mosaic doesn't natively support bf16 argmax) |
| 4 | sort by expert via SC or TC? | **SC** (its design use case; chunked per C1) |
| 5 | scoring fn? | **sigmoid** (DSv3 production + paper) |
| 6 | top-K weight normalization? | **renormalize to sum=1** (DSv3-style; matches v1 `renormalize_topk_logits=True`) |
| 7 | activation function? | **silu(gate) * up** |
| 8 | AG W_d strategy? | **(b) streaming inside** for `v_inside`; **(a) outside** for `v_outside` |
| 9 | AG W1 strategy? | same as D8 |
| 10 | kernel owns residual add? | **Yes** |
| 11 | fwd-only first, or fwd+bwd together? | **Fwd+bwd together** (v3 stuck in bwd, v4 stuck in fwd — substrate must be coherent across boundary) |
| 12 | W_gate sharding? | **Replicated** |
| 13 | accept `top_k_indices_precomputed`? | **No, fwd computes internally**; bwd consumes residuals from fwd |
| 14 | K? | **8** (matches v1 + DSv3 paper) |
| 15 | production mesh? | **EP=8 FSDP=64 TP=1** on 4×8×8 v7x |
| 16 | perf targets? | **§8 — three reference points: pure-JAX baseline (lower bound), v1 (target), roofline (upper bound)** |

---

## 13. What this SPEC explicitly does NOT specify

- Block sizes (`bt`, `bd`, `bf`) — agent derives via tuned_block_sizes lookup or microbench
- VMEM allocation order, semaphore IDs, register assignments — implementation detail
- Compile flags, container layout — out of scope
- Specific Mosaic ops (e.g. `lax.axis_index` vs `pl.program_id` — these are pattern choices)
- Inline performance tuning (loop unroll factors, prefetch distances) — Phase D xla-shell signal
- The mapping from logical axis names to physical mesh axes — that's the user's `LOGICAL_AXIS_RULES`, outside the SPEC

These are ALL implementation; the SPEC defines architecture. The agent's job in Phase B is to fill them in, guided by the substrate patterns referenced in §10.

---

## 14. Followups / parking lot

Things mentioned during SPEC discussion that aren't in v0.2:
- DSv3 training-time bias-adjusted gating (auxiliary-loss-free load balancing) — not in v0; inference math doesn't use it
- DSv3 shared-expert (1 always-on expert in addition to top-K routed) — not in v0; can be added as a separate trivial Pallas call or JAX glue if needed
- FP8 / quantized variants — separate target after BF16 lands
- Multi-node DCN benchmarking — `bodaborg-tpu7x-inference` exposes this naturally (2 nodes, cross-host)
