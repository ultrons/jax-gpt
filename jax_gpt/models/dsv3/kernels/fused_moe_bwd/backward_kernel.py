"""Pallas backward kernel for fused_ep_moe (Stage C/D).

Design:
  - EP=1 and EP>1 supported.
  - EP>1: called inside shard_map; each device processes its E_local experts.
    Non-local expert slots are zeroed (zero weight mask).  Partial d_tokens and
    d_gating are psum'd by the caller (make_fused_ep_moe_train_v3._bwd).
  - Pre-sort tokens by expert assignment in JAX before the pallas_call.
  - Pallas kernel processes each expert's tokens as a contiguous VMEM block,
    doing batched matmuls instead of per-token-pair individual matmuls.
  - Activation checkpointing: h_gate/h_up are recomputed inside the kernel.
  - d_w1/d_w2 accumulate in VMEM and are written to HBM once per expert.
  - d_tokens are written back in sorted order; JAX unsorts them afterward.

D-tiling:
  - When D > tile_D, the kernel tiles the hidden dimension to stay within VMEM.
  - tile_D must divide D evenly and satisfy 6 * tile_D * F * 4 < vmem_limit_bytes.
  - For full DSv3 671B (D=7168, F=2048) on v7x (64 MB VMEM):
      tile_D=1024 → 6 * 1024 * 2048 * 4 = 50 MB < 64 MB  ✓  (7 tiles)
  - D-tiling adds an outer loop over d_tile in process_expert.  Each iteration:
      loads w1[:, d_tile*tile_D:(d_tile+1)*tile_D, :] into VMEM,
      slices tok_buf to tok_tile (bte, tile_D),
      accumulates partial d_tok_tile contributions,
      writes final d_tok (sum over tiles) back to d_bins_tokens_hbm.
"""

import functools

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

cdiv = pl.cdiv

# ---------------------------------------------------------------------------
# sc_gather_rows is a TC-fallback gather. SC IndexedLoad was attempted but
# plsc.BlockSpec mis-lowers 2D indexed specs to Get (register) instead of DMA
# in current JAX; see sc_gather_rows docstring for full explanation.

def sc_gather_rows(source, row_indices):
    """Gather rows from source using row_indices.

    result[i] = source[row_indices[i]],  shape (n, D) from (T, D).

    TC fallback (unconditional): The original intent was to use SparseCore
    IndexedLoad DMA to overlap per-expert gather with the previous expert's
    TC GEMMs. However, plsc.BlockSpec with indexed_by mis-lowers to the
    hardware Get register instruction instead of IndexedLoad DMA for 2D
    sources in the current JAX version. Get only supports 1D register-sized
    shapes; all 2D block specs fail at compile time with NotImplementedError.
    The system is ICI-dominated (28 ms A2A vs 15 ms expert FFN per layer),
    so SC gather overlap is not on the critical path — TC fallback is correct
    and has negligible performance impact.

    Args:
        source:      (T, D) array in HBM.
        row_indices: (n,)  int32 row indices.
    Returns:
        (n, D) gathered array.
    """
    return source[row_indices]


# ---------------------------------------------------------------------------
# Routing helpers (pure JAX, cheap)
# ---------------------------------------------------------------------------

def compute_routing(gating_output, top_k, scoring_fn, renormalize_topk_logits):
    """Compute top-k routing from gating logits. Pure JAX, O(T*E).

    Returns:
      top_k_indices  (T, K) int32
      top_k_weights  (T, K) float32 (post renorm)
    """
    if scoring_fn == "softmax":
        scores = jax.nn.softmax(gating_output.astype(jnp.float32), axis=-1)
    elif scoring_fn == "sigmoid":
        scores = jax.nn.sigmoid(gating_output.astype(jnp.float32))
    else:
        raise NotImplementedError(scoring_fn)

    top_k_weights, top_k_indices = lax.top_k(scores, top_k)
    if renormalize_topk_logits:
        top_k_weights = top_k_weights / top_k_weights.sum(axis=-1, keepdims=True)
    return top_k_indices, top_k_weights


def sort_tokens_by_expert(tokens, top_k_indices, top_k_weights, num_experts):
    """Pre-sort tokens by expert assignment for the backward kernel.

    Returns a dict of arrays that are cheap residuals for the backward:
      sorted_tokens      (T*K, D) — token embeddings sorted by expert id
      d_exp_weights      (T*K,)   — top_k_weights in sorted order (for d_out weighting)
      sort_order         (T*K,)   — argsort of expert_ids_flat (for unsort)
      sorted_token_ids   (T*K,)   — which original token each sorted slot came from
      expert_starts      (E,) int32
      expert_sizes       (E,) int32
    """
    T, D = tokens.shape
    K = top_k_indices.shape[1]
    TK = T * K

    expert_ids_flat = top_k_indices.reshape(TK)         # (T*K,) expert for each slot
    token_ids_flat  = jnp.repeat(jnp.arange(T), K)     # (T*K,) original token index
    k_ids_flat      = jnp.tile(jnp.arange(K), T)       # (T*K,) which k-slot

    # Sort by expert id (stable so tokens within an expert keep their order)
    sort_order       = jnp.argsort(expert_ids_flat, stable=True)   # (T*K,)
    sorted_expert_ids  = expert_ids_flat[sort_order]               # (T*K,)
    sorted_token_ids   = token_ids_flat[sort_order]                # (T*K,)
    sorted_k_ids       = k_ids_flat[sort_order]                    # (T*K,)

    sorted_tokens  = tokens[sorted_token_ids]                      # (T*K, D)

    # d_out weighting: for slot (t, k), upstream gradient is d_out[t] * weight[t,k]
    flat_weights  = top_k_weights[token_ids_flat, k_ids_flat]      # (T*K,)
    d_exp_weights = flat_weights[sort_order]                       # (T*K,) sorted

    # Expert boundaries
    expert_sizes  = jnp.bincount(sorted_expert_ids, length=num_experts).astype(jnp.int32)
    expert_starts = jnp.concatenate([
        jnp.zeros(1, jnp.int32),
        jnp.cumsum(expert_sizes[:-1]),
    ])  # (E,)

    return dict(
        sorted_tokens     = sorted_tokens,
        d_exp_weights     = d_exp_weights,
        sort_order        = sort_order,
        sorted_token_ids  = sorted_token_ids,
        sorted_expert_ids = sorted_expert_ids,  # (T*K,) expert id for each sorted slot
        expert_starts     = expert_starts,
        expert_sizes      = expert_sizes,
    )


# ---------------------------------------------------------------------------
# Stage 1: per-expert streaming JAX backward (no T*K*D materialization)
# ---------------------------------------------------------------------------

def fused_ep_moe_bwd_streaming(
    d_out,           # (T, D) float32
    tokens,          # (T, D)
    w1,              # (E_local, 2, D, F)  — may be E//EP when called inside shard_map
    w2,              # (E_local, F, D)
    gating_output,   # (T, E_global) raw logits — OR None when top_k_indices_precomputed given
    top_k: int,
    *,
    scoring_fn: str = "softmax",
    renormalize_topk_logits: bool = False,
    act_fn: str = "silu",
    ep_axis_name: str = "model",
    max_tpe: int | None = None,
    # Precomputed routing: bypass internal compute_routing (use when forward routing
    # differs from kernel's compute_routing, e.g. DSv3 gate_bias / routed_scaling_factor).
    top_k_indices_precomputed=None,  # (T, K) int32 global expert IDs, optional
    top_k_weights_precomputed=None,  # (T, K) float32 routing weights, optional
    # When True, return (d_tokens, d_w1, d_w2, d_top_k_weights) instead of d_gating.
    # Use with top_k_indices_precomputed so the caller gets grad wrt flat_weights (T, K).
    return_dtopk: bool = False,
    # Required when gating_output is None (needed to determine E_global).
    E_global_override: int | None = None,
    # Pallas-kernel params accepted but ignored (not needed for JAX implementation)
    bte: int | None = None,
    tile_D: int | None = None,
    vmem_limit_bytes: int = 16 * 1024 * 1024,
    # Accepted for API compatibility; no longer used (loop is always list-based).
    use_python_loop: bool = False,
):
    """Stage 1: Per-expert streaming JAX backward for fused_ep_moe.

    Key insight: keep the bins *index* layout (1D, cheap ~33 MB) but eliminate
    bins *data* buffers (which would be E_local * max_tpe * D = 120 GB).
    Per expert, tokens are gathered on-the-fly from original HBM via indices
    — ~470 MB per expert, never pre-materialized for all experts at once.

    EP=1: called directly; processes all E experts.
    EP>1: called inside shard_map; each device processes its E_local experts.
          Returns *partial* d_tokens and d_gating — caller must lax.psum them.

    Args:
      bte, tile_D, vmem_limit_bytes: accepted for API compatibility but ignored.
      top_k_indices_precomputed: when provided, skip internal compute_routing.
      top_k_weights_precomputed: matching routing weights (must accompany above).
      return_dtopk: when True, return d_top_k_weights (T, K) instead of d_gating (T, E).
      E_global_override: required when gating_output is None.

    Returns (d_tokens, d_w1, d_w2, d_gating) or (d_tokens, d_w1, d_w2, d_top_k_weights).
    """
    T, D = tokens.shape
    E_local, _, _, F = w1.shape
    if gating_output is not None:
        E_global = gating_output.shape[1]
    else:
        assert E_global_override is not None, "E_global_override required when gating_output is None"
        E_global = E_global_override

    tokens_f32  = tokens.astype(jnp.float32)
    w1_f32      = w1.astype(jnp.float32)
    w2_f32      = w2.astype(jnp.float32)
    d_out_f32   = d_out.astype(jnp.float32)

    # ---- 1. Routing (pure JAX, global expert IDs) ----
    if top_k_indices_precomputed is not None:
        # Use caller-provided routing (correct for DSv3 with gate_bias + routed_scaling_factor)
        top_k_indices = top_k_indices_precomputed.astype(jnp.int32)
        top_k_weights = top_k_weights_precomputed.astype(jnp.float32)
    else:
        gating_f32  = gating_output.astype(jnp.float32)
        top_k_indices, top_k_weights = compute_routing(
            gating_f32, top_k, scoring_fn, renormalize_topk_logits)
    # top_k_indices: (T, K) global expert IDs [0, E_global)
    # top_k_weights: (T, K)

    # ---- 1b. EP>1: remap global IDs to local; mask non-local slots ----
    ep_sharded = (E_local < E_global)
    if ep_sharded:
        device_id     = lax.axis_index(ep_axis_name)
        expert_offset = jnp.int32(device_id) * jnp.int32(E_local)
        is_local      = (top_k_indices >= expert_offset) & (
                         top_k_indices < expert_offset + E_local)
        is_local_flat = is_local.reshape(-1)              # (TK,)
        # Non-local → E_local: sorts after all local experts; OOB writes to padded bins dropped.
        top_k_indices_kernel = jnp.where(is_local, top_k_indices - expert_offset, E_local)
        top_k_weights_kernel = jnp.where(is_local, top_k_weights, 0.0)
    else:
        expert_offset        = jnp.int32(0)
        is_local_flat        = jnp.ones(T * top_k, dtype=bool)
        top_k_indices_kernel = top_k_indices
        top_k_weights_kernel = top_k_weights

    TK = T * top_k

    # ---- 2. Sort tokens by local expert ----
    expert_ids_flat = top_k_indices_kernel.reshape(TK)       # (TK,) local IDs
    token_ids_flat  = jnp.repeat(jnp.arange(T, dtype=jnp.int32), top_k)  # (TK,)
    k_ids_flat      = jnp.tile(jnp.arange(top_k, dtype=jnp.int32), T)    # (TK,)
    flat_weights    = top_k_weights_kernel[token_ids_flat, k_ids_flat]     # (TK,)

    sort_order          = jnp.argsort(expert_ids_flat, stable=True)        # (TK,)
    sorted_expert_ids   = expert_ids_flat[sort_order]                      # (TK,)
    sorted_token_ids    = token_ids_flat[sort_order]                       # (TK,)
    d_exp_weights       = flat_weights[sort_order]                         # (TK,)
    is_local_sorted     = is_local_flat[sort_order]                        # (TK,)

    expert_sizes  = jnp.bincount(sorted_expert_ids, length=E_local).astype(jnp.int32)
    expert_starts = jnp.concatenate([
        jnp.zeros(1, jnp.int32),
        jnp.cumsum(expert_sizes[:-1]),
    ])  # (E_local,)

    # ---- 3. Compute max_tpe_single (static Python int for lax.dynamic_slice) ----
    if max_tpe is None:
        avg_tpe = max(cdiv(TK, E_local) * 2, 128)
        max_tpe_single = min(avg_tpe, TK)
        # Round up to nearest 128
        max_tpe_single = cdiv(max_tpe_single, 128) * 128
    else:
        max_tpe_single = max_tpe

    # ---- 4. Build 1D padded bins (index/scalar arrays only — no (TK, D) buffer) ----
    # Expert e's slots are at [e*max_tpe_single, (e+1)*max_tpe_single)
    expert_starts_per_slot = expert_starts[sorted_expert_ids]              # (TK,)
    local_indices          = jnp.arange(TK, dtype=jnp.int32) - expert_starts_per_slot  # (TK,)
    bin_positions          = sorted_expert_ids * max_tpe_single + local_indices         # (TK,)

    pad_to = E_local * max_tpe_single

    bins_tok_ids  = jnp.zeros(pad_to, jnp.int32 ).at[bin_positions].set(sorted_token_ids)
    bins_weights  = jnp.zeros(pad_to, jnp.float32).at[bin_positions].set(d_exp_weights)
    bins_is_local = jnp.zeros(pad_to, bool       ).at[bin_positions].set(is_local_sorted)
    bins_valid    = jnp.zeros(pad_to, bool       ).at[bin_positions].set(
                        jnp.ones(TK, bool))

    # ---- 5. Activation helpers (defined once, reused in loop) ----
    def silu_grad(x):
        sig = jax.nn.sigmoid(x)
        return sig * (1.0 + x * (1.0 - sig))

    # ---- 6. Per-expert loop — list-based (no carry) ----
    #
    # Mirrors the forward pass pattern (_expert_mlp_ep_body_ep_sharded Phase 3+4):
    # collect per-expert results independently, then single batched gather/stack
    # at the end. Avoids chained scatter-add and accumulator updates that create
    # long sequential dependency chains → XLA liveness analysis hanging (50+ min).
    #
    # Key insight: each expert's backward is INDEPENDENT (no shared state).
    # The carry pattern (fori_loop or Python for) creates N versions of 13 GB of
    # accumulators, whereas the list pattern lets XLA see N independent operations.
    all_d_tok_e_list  = []   # per-expert: (max_tpe, D)
    all_tok_ids_list  = []   # per-expert: (max_tpe,)  int32
    d_w1g_list        = []   # per-expert: (D, F)
    d_w1u_list        = []   # per-expert: (D, F)
    d_w2_list         = []   # per-expert: (F, D)
    d_tw_bin_list     = []   # per-expert: (max_tpe_single,)

    for e in range(E_local):
        start = e * max_tpe_single

        tok_ids_e  = lax.dynamic_slice(bins_tok_ids,  (start,), (max_tpe_single,))
        weights_e  = lax.dynamic_slice(bins_weights,  (start,), (max_tpe_single,))
        is_local_e = lax.dynamic_slice(bins_is_local, (start,), (max_tpe_single,))
        valid_e    = lax.dynamic_slice(bins_valid,    (start,), (max_tpe_single,))

        valid_f = valid_e.astype(jnp.float32)  # (max_tpe_single,)

        # Gather tokens and d_out from original HBM (SC-accelerated on v7x).
        # sc_gather_rows uses SparseCore indexed DMA (4.45× faster than TC for random
        # access). XLA overlaps SC gather for expert e+1 with TC GEMMs for expert e.
        tokens_raw = sc_gather_rows(tokens, tok_ids_e).astype(jnp.float32)
        tokens_e   = tokens_raw * valid_f[:, None]             # (max_tpe, D)
        d_out_e    = sc_gather_rows(d_out, tok_ids_e).astype(jnp.float32)  # (max_tpe, D)
        d_out_es = d_out_e * (weights_e * valid_f)[:, None]    # weight-scaled for FFN bwd

        # FFN forward recompute (activation checkpointing)
        h_g  = tokens_e @ w1_f32[e, 0]          # (max_tpe, F)
        h_u  = tokens_e @ w1_f32[e, 1]          # (max_tpe, F)

        if act_fn == "silu":
            h_act   = jax.nn.silu(h_g) * h_u    # (max_tpe, F)
        elif act_fn == "gelu":
            h_act   = jax.nn.gelu(h_g) * h_u
        else:
            raise NotImplementedError(act_fn)

        # FFN backward
        d_h_act = d_out_es @ w2_f32[e].T         # (max_tpe, F)
        d_w2_e  = h_act.T @ d_out_es             # (F, D)

        if act_fn == "silu":
            d_h_u = d_h_act * jax.nn.silu(h_g)
            d_h_g = d_h_act * h_u * silu_grad(h_g)
        elif act_fn == "gelu":
            d_h_u = d_h_act * jax.nn.gelu(h_g)
            d_h_g = d_h_act * h_u * jax.vmap(jax.vmap(jax.grad(jax.nn.gelu)))(h_g)

        d_w1g_e = tokens_e.T @ d_h_g             # (D, F)
        d_w1u_e = tokens_e.T @ d_h_u             # (D, F)

        # d_tok contribution from this expert (zeroed for padding slots via valid_f)
        d_tok_e = (d_h_g @ w1_f32[e, 0].T + d_h_u @ w1_f32[e, 1].T) * valid_f[:, None]

        # d_routing_weights: (d_out * out) summed over D, only for local slots
        out_e   = h_act @ w2_f32[e]               # (max_tpe, D)
        d_tw_e  = (d_out_e * out_e * valid_f[:, None]).sum(-1) * is_local_e.astype(jnp.float32)

        # Collect — no carry, no sequential dependency between iterations
        all_d_tok_e_list.append(d_tok_e)
        all_tok_ids_list.append(tok_ids_e)
        d_w1g_list.append(d_w1g_e)
        d_w1u_list.append(d_w1u_e)
        d_w2_list.append(d_w2_e)
        d_tw_bin_list.append(d_tw_e)

    # Single batched scatter for d_tokens (mirrors forward's Phase 4 scatter-add).
    # segment_sum handles overlapping tok_ids across experts (K>1 routing).
    all_d_tok  = jnp.concatenate(all_d_tok_e_list, axis=0)   # (E_local * max_tpe, D)
    all_tok_ids = jnp.concatenate(all_tok_ids_list, axis=0)  # (E_local * max_tpe,)
    d_tokens = jax.ops.segment_sum(all_d_tok, all_tok_ids, T)  # (T, D)

    # Stack weight grads — each expert writes its own slice; independent operations.
    d_w1_out = jnp.stack(
        [jnp.stack([d_w1g_list[e], d_w1u_list[e]], axis=0) for e in range(E_local)],
        axis=0)   # (E_local, 2, D, F)
    d_w2_out = jnp.stack(d_w2_list, axis=0)  # (E_local, F, D)

    # d_tw_padded in bin layout: expert 0 occupies [0:max_tpe], expert 1 next, etc.
    d_tw_padded = jnp.concatenate(d_tw_bin_list, axis=0)   # (E_local * max_tpe_single,)

    # ---- 7. Recover d_top_k_weights from bin layout → (T, K) ----
    # d_tw_padded is in bin order (sorted by expert); gather back to sorted order,
    # then invert the sort to recover (T*K,) in original (token, k-slot) order.
    d_tw_sorted     = d_tw_padded[bin_positions]              # (TK,) sorted order
    sort_order_inv  = jnp.argsort(sort_order)
    d_tw_flat       = d_tw_sorted[sort_order_inv]             # (TK,) original flat order
    d_top_k_weights = d_tw_flat.reshape(T, top_k)             # (T, K)
    if ep_sharded:
        # Non-local bin_positions are OOB → gathered arbitrary values; zero them out.
        # Without this, routing gradients are overcounted by (EP-1)*FSDP → NaN at step 1.
        d_top_k_weights = jnp.where(is_local, d_top_k_weights, 0.0)

    # ---- 8. Routing backward ----
    if return_dtopk:
        # Caller wants grad wrt flat_weights (T, K) not gate logits (T, E).
        # Used when integrating with custom_vjp that has flat_weights as residual.
        return d_tokens, d_w1_out, d_w2_out, d_top_k_weights

    from backward import moe_bwd_routing

    assert gating_output is not None, "gating_output required when return_dtopk=False"
    gating_f32_for_bwd = gating_output.astype(jnp.float32)
    gating_scores = (jax.nn.softmax(gating_f32_for_bwd, axis=-1)
                     if scoring_fn == "softmax"
                     else jax.nn.sigmoid(gating_f32_for_bwd))

    # Use global top_k_indices for the scatter step (full (T, E_global) output).
    # d_top_k_weights is already zero for non-local slots → partial d_gating.
    d_gating = moe_bwd_routing(
        d_top_k_weights, top_k_indices, top_k_weights,
        gating_scores, gating_f32_for_bwd,
        scoring_fn=scoring_fn,
        renormalize_topk_logits=renormalize_topk_logits,
    )

    return d_tokens, d_w1_out, d_w2_out, d_gating


# ---------------------------------------------------------------------------
# Stage 2: FSDP async weight prefetch streaming backward
# ---------------------------------------------------------------------------

def fused_ep_moe_bwd_streaming_v2(
    d_out,           # (T, D) float32
    tokens,          # (T, D)
    w1,              # (E_local, 2, D, F/fsdp) — FSDP-shards the hidden (F) dim, axis=3
    w2,              # (E_local, F/fsdp, D)    — FSDP-shards the hidden (F) dim, axis=1
    gating_output,   # (T, E_global) raw logits — OR None when top_k_indices_precomputed given
    top_k: int,
    *,
    fsdp_axis_name: str,               # REQUIRED — collective axis for all_gather / psum_scatter
    scoring_fn: str = "softmax",
    renormalize_topk_logits: bool = False,
    act_fn: str = "silu",
    ep_axis_name: str = "model",
    max_tpe: int | None = None,
    top_k_indices_precomputed=None,
    top_k_weights_precomputed=None,
    return_dtopk: bool = False,
    E_global_override: int | None = None,
    # Pallas-compat params accepted but ignored
    bte: int | None = None,
    tile_D: int | None = None,
    vmem_limit_bytes: int = 16 * 1024 * 1024,
):
    """Streaming backward v2: conjugate-collective backward with FSDP-sharded weights.

    The forward MoE uses FSDP-sharded weights (F/fsdp per device) and applies silu
    to the partial F/fsdp activations independently per FSDP shard:
      out_j = silu(x @ w1_j[0]) * (x @ w1_j[1]) @ w2_j   per FSDP shard j
      out   = psum("fsdp")(out_j)

    The conjugate backward uses sharded weights directly (no FSDP all_gather):
      d_h_act_j = d_out_es @ w2_j.T              (F/fsdp columns)
      d_tok_j   = d_h_g_j @ w1_j[0].T + ...      partial D_moe contribution
      d_tokens  = psum("fsdp")(segment_sum(d_tok_j))   ← conjugate of fwd psum

    Weight layout (from P("ep", None, "fsdp") / P("ep", "fsdp", None)):
      w1: (E_local, 2, D, F/fsdp)  — D is full, F is FSDP-sharded (axis 3)
      w2: (E_local, F/fsdp, D)     — F is FSDP-sharded (axis 1)

    Must be called within a shard_map that covers fsdp_axis_name (and ep_axis_name if
    EP>1) so the psum collective is valid.

    Returns (d_tokens, d_w1_sharded, d_w2_sharded, d_gating or d_top_k_weights)
    where d_w1_sharded is (E_local, 2, D, F/fsdp) and d_w2_sharded is (E_local, F/fsdp, D).
    """
    T, D = tokens.shape
    E_local, _, _, F_shard = w1.shape   # w1: (E_local, 2, D, F/fsdp)
    if gating_output is not None:
        E_global = gating_output.shape[1]
    else:
        assert E_global_override is not None, "E_global_override required when gating_output is None"
        E_global = E_global_override

    # ---- 1. Routing ---- (identical to v1)
    if top_k_indices_precomputed is not None:
        top_k_indices = top_k_indices_precomputed.astype(jnp.int32)
        top_k_weights = top_k_weights_precomputed.astype(jnp.float32)
    else:
        gating_f32 = gating_output.astype(jnp.float32)
        top_k_indices, top_k_weights = compute_routing(
            gating_f32, top_k, scoring_fn, renormalize_topk_logits)

    # ---- 1b. EP>1: remap global IDs to local; mask non-local slots ---- (identical to v1)
    ep_sharded = (E_local < E_global)
    if ep_sharded:
        device_id     = lax.axis_index(ep_axis_name)
        expert_offset = jnp.int32(device_id) * jnp.int32(E_local)
        is_local      = (top_k_indices >= expert_offset) & (
                         top_k_indices < expert_offset + E_local)
        is_local_flat = is_local.reshape(-1)
        # Non-local → E_local: sorts after all local experts; OOB writes to padded bins dropped.
        top_k_indices_kernel = jnp.where(is_local, top_k_indices - expert_offset, E_local)
        top_k_weights_kernel = jnp.where(is_local, top_k_weights, 0.0)
    else:
        expert_offset        = jnp.int32(0)
        is_local_flat        = jnp.ones(T * top_k, dtype=bool)
        top_k_indices_kernel = top_k_indices
        top_k_weights_kernel = top_k_weights

    TK = T * top_k

    # ---- 2. Sort tokens by local expert ---- (identical to v1)
    expert_ids_flat = top_k_indices_kernel.reshape(TK)
    token_ids_flat  = jnp.repeat(jnp.arange(T, dtype=jnp.int32), top_k)
    k_ids_flat      = jnp.tile(jnp.arange(top_k, dtype=jnp.int32), T)
    flat_weights    = top_k_weights_kernel[token_ids_flat, k_ids_flat]

    sort_order        = jnp.argsort(expert_ids_flat, stable=True)
    sorted_expert_ids = expert_ids_flat[sort_order]
    sorted_token_ids  = token_ids_flat[sort_order]
    d_exp_weights     = flat_weights[sort_order]
    is_local_sorted   = is_local_flat[sort_order]

    expert_sizes  = jnp.bincount(sorted_expert_ids, length=E_local).astype(jnp.int32)
    expert_starts = jnp.concatenate([
        jnp.zeros(1, jnp.int32),
        jnp.cumsum(expert_sizes[:-1]),
    ])

    # ---- 3. max_tpe_single ---- (identical to v1)
    if max_tpe is None:
        avg_tpe = max(cdiv(TK, E_local) * 2, 128)
        max_tpe_single = min(avg_tpe, TK)
        max_tpe_single = cdiv(max_tpe_single, 128) * 128
    else:
        max_tpe_single = max_tpe

    # ---- 4. 1D padded bins ---- (identical to v1)
    expert_starts_per_slot = expert_starts[sorted_expert_ids]
    local_indices  = jnp.arange(TK, dtype=jnp.int32) - expert_starts_per_slot
    bin_positions  = sorted_expert_ids * max_tpe_single + local_indices

    pad_to = E_local * max_tpe_single

    bins_tok_ids  = jnp.zeros(pad_to, jnp.int32 ).at[bin_positions].set(sorted_token_ids)
    bins_weights  = jnp.zeros(pad_to, jnp.float32).at[bin_positions].set(d_exp_weights)
    bins_is_local = jnp.zeros(pad_to, bool       ).at[bin_positions].set(is_local_sorted)
    bins_valid    = jnp.zeros(pad_to, bool       ).at[bin_positions].set(
                        jnp.ones(TK, bool))

    # ---- 5. Activation helpers ---- (identical to v1)
    def silu_grad(x):
        sig = jax.nn.sigmoid(x)
        return sig * (1.0 + x * (1.0 - sig))

    # ---- 6. Per-expert backward with FSDP-sharded weights ----
    # Use sharded weights directly — no FSDP all_gather needed.
    #
    # The forward computes, per FSDP shard j:
    #   h_g_j = tokens @ w1_j[0]          (max_tpe, F_shard) — partial F
    #   h_act_j = silu(h_g_j) * h_u_j
    #   out_j   = h_act_j @ w2_j          (max_tpe, D)
    # then psum("fsdp") sums partial D_moe contributions.
    #
    # Backward (conjugate collectives):
    #   d_h_act_j = d_out_es @ w2_j.T     (max_tpe, F_shard) — uses sharded w2
    #   d_tok_j   = d_h_g_j @ w1_j[0].T + d_h_u_j @ w1_j[1].T  (max_tpe, D) — partial
    # Then psum("fsdp") on d_tokens sums partial D_moe contributions → correct d_tok.
    #
    # Weight grads are already FSDP-local (each shard owns its F_shard columns).
    # No dynamic_slice needed — h_act, d_h_g, d_h_u are already F_shard-wide.

    all_d_tok_list   = []
    all_tok_ids_list = []
    d_w1_list        = []   # per-expert (2, D, F_shard)
    d_w2_list        = []   # per-expert (F_shard, D)
    d_tw_list        = []   # per-expert (max_tpe_single,)

    for e in range(E_local):
        w1_e = w1[e]   # (2, D, F_shard) — bf16, same as forward
        w2_e = w2[e]   # (F_shard, D) — bf16

        start      = e * max_tpe_single
        tok_ids_e  = lax.dynamic_slice(bins_tok_ids,  (start,), (max_tpe_single,))
        weights_e  = lax.dynamic_slice(bins_weights,  (start,), (max_tpe_single,))
        is_local_e = lax.dynamic_slice(bins_is_local, (start,), (max_tpe_single,))
        valid_e    = lax.dynamic_slice(bins_valid,    (start,), (max_tpe_single,))

        valid_f   = valid_e.astype(tokens.dtype)  # bf16 mask — avoids upcasting tokens_e

        tokens_e  = sc_gather_rows(tokens, tok_ids_e) * valid_f[:, None]
        d_out_e   = sc_gather_rows(d_out, tok_ids_e)
        # weights_e is fp32 (routing weight); cast scaled mask back to bf16 so matmuls stay bf16.
        d_out_es  = d_out_e * (weights_e * valid_f).astype(tokens.dtype)[:, None]

        # Forward recompute with sharded weights — matches actual forward.
        h_g = tokens_e @ w1_e[0]   # (max_tpe, F_shard)
        h_u = tokens_e @ w1_e[1]   # (max_tpe, F_shard)

        if act_fn == "silu":
            h_act = jax.nn.silu(h_g) * h_u
        elif act_fn == "gelu":
            h_act = jax.nn.gelu(h_g) * h_u
        else:
            raise NotImplementedError(act_fn)

        # Backward with sharded weights.
        d_h_act = d_out_es @ w2_e.T   # (max_tpe, F_shard)

        if act_fn == "silu":
            d_h_u = d_h_act * jax.nn.silu(h_g)
            d_h_g = d_h_act * h_u * silu_grad(h_g)
        elif act_fn == "gelu":
            d_h_u = d_h_act * jax.nn.gelu(h_g)
            d_h_g = d_h_act * h_u * jax.vmap(jax.vmap(jax.grad(jax.nn.gelu)))(h_g)

        # d_tok partial (F_shard contribution). psum("fsdp") below completes it.
        d_tok_e = (d_h_g @ w1_e[0].T + d_h_u @ w1_e[1].T) * valid_f[:, None]

        # d_tw partial (F_shard contribution to out_e). psum("fsdp") below completes it.
        out_e = h_act @ w2_e   # (max_tpe, D) partial
        d_tw_e = (d_out_e * out_e * valid_f[:, None]).sum(-1) * is_local_e.astype(jnp.float32)

        # Weight grads: h_act/d_h_g/d_h_u are already F_shard-wide — no slicing needed.
        d_w2_e  = h_act.T @ d_out_es   # (F_shard, D)
        d_w1g_e = tokens_e.T @ d_h_g   # (D, F_shard)
        d_w1u_e = tokens_e.T @ d_h_u   # (D, F_shard)

        all_d_tok_list.append(d_tok_e)
        all_tok_ids_list.append(tok_ids_e)
        d_w1_list.append(jnp.stack([d_w1g_e, d_w1u_e], axis=0))  # (2, D, F_shard)
        d_w2_list.append(d_w2_e)
        d_tw_list.append(d_tw_e)

    all_d_tok   = jnp.concatenate(all_d_tok_list,   axis=0)   # (E_local * max_tpe_single, D)
    all_tok_ids = jnp.concatenate(all_tok_ids_list, axis=0)   # (E_local * max_tpe_single,)
    # segment_sum gives partial d_tokens (this FSDP shard's F_shard contribution).
    # psum("fsdp") sums all shards' partial contributions → correct full d_tokens.
    d_tokens = jax.ops.segment_sum(all_d_tok, all_tok_ids, T)  # (T, D) partial
    d_tokens = lax.psum(d_tokens, fsdp_axis_name)              # (T, D) full

    d_w1_full   = jnp.stack(d_w1_list, axis=0)           # (E_local, 2, D, F_shard)
    d_w2_full   = jnp.stack(d_w2_list, axis=0)           # (E_local, F_shard, D)
    d_tw_padded = jnp.concatenate(d_tw_list, axis=0)     # (E_local * max_tpe_single,)

    # ---- 8. Weight grads: local shard only, already correct ----
    d_w1_out = d_w1_full   # (E_local, 2, D, F_shard)
    d_w2_out = d_w2_full   # (E_local, F_shard, D)

    # ---- 9. Recover d_top_k_weights from bin layout → (T, K) ----
    d_tw_sorted     = d_tw_padded[bin_positions]
    sort_order_inv  = jnp.argsort(sort_order)
    d_tw_flat       = d_tw_sorted[sort_order_inv]
    d_top_k_weights = d_tw_flat.reshape(T, top_k)
    if ep_sharded:
        # Non-local bin_positions are OOB → gathered arbitrary values; zero them out.
        d_top_k_weights = jnp.where(is_local, d_top_k_weights, 0.0)
    # psum("fsdp") sums partial d_tw contributions across FSDP shards.
    d_top_k_weights = lax.psum(d_top_k_weights, fsdp_axis_name)

    # ---- 10. Routing backward ---- (identical to v1)
    if return_dtopk:
        return d_tokens, d_w1_out, d_w2_out, d_top_k_weights

    from backward import moe_bwd_routing

    assert gating_output is not None, "gating_output required when return_dtopk=False"
    gating_f32_for_bwd = gating_output.astype(jnp.float32)
    gating_scores = (jax.nn.softmax(gating_f32_for_bwd, axis=-1)
                     if scoring_fn == "softmax"
                     else jax.nn.sigmoid(gating_f32_for_bwd))

    d_gating = moe_bwd_routing(
        d_top_k_weights, top_k_indices, top_k_weights,
        gating_scores, gating_f32_for_bwd,
        scoring_fn=scoring_fn,
        renormalize_topk_logits=renormalize_topk_logits,
    )

    return d_tokens, d_w1_out, d_w2_out, d_gating
