"""Mixture of Experts layer with ragged_dot.

Implements top-k expert routing with a shared expert.
Uses jax.lax.ragged_dot for sparse expert computation.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from jax_gpt.models.qwen35.fp8 import matmul_maybe_fp8


def moe_routing(
    x: jax.Array,
    gate_weight: jax.Array,
    n_experts_per_token: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Top-k expert routing.

    Args:
        x: (M, D) flattened tokens (M = B*T).
        gate_weight: (D, n_experts) router weight matrix.
        n_experts_per_token: k — how many experts each token selects.

    Returns:
        expert_indices: (M, k) selected expert indices per token.
        expert_weights: (M, k) softmax weights for selected experts.
        sorted_indices: (M*k,) token indices sorted by expert assignment.
        group_sizes: (n_experts,) number of tokens assigned to each expert.
    """
    n_experts = gate_weight.shape[1]

    # Router: softmax over ALL experts, then top-k, then renormalize
    # (matches HF Qwen3.5 router)
    logits = matmul_maybe_fp8(x, gate_weight)  # (M, n_experts)
    probs = jax.nn.softmax(logits.astype(jnp.float32), axis=-1)
    top_k_values, top_k_indices = jax.lax.top_k(probs, n_experts_per_token)
    # Renormalize selected expert weights
    expert_weights = top_k_values / jnp.sum(top_k_values, axis=-1, keepdims=True)

    # Build sorted token indices and group sizes for ragged_dot.
    # For each expert, we need to know which tokens are assigned to it
    # and sort tokens by expert assignment.
    M = x.shape[0]
    k = n_experts_per_token

    # Flatten assignments: each token contributes k assignments
    flat_token_ids = jnp.repeat(jnp.arange(M), k)  # (M*k,)
    flat_expert_ids = top_k_indices.reshape(-1)      # (M*k,)

    # Sort by expert id to group tokens per expert
    sort_order = jnp.argsort(flat_expert_ids)
    sorted_token_ids = flat_token_ids[sort_order]
    sorted_expert_ids = flat_expert_ids[sort_order]

    # Count tokens per expert
    group_sizes = jnp.zeros(n_experts, dtype=jnp.int32)
    group_sizes = group_sizes.at[flat_expert_ids].add(1)

    return top_k_indices, expert_weights, sorted_token_ids, group_sizes


def expert_forward(
    x: jax.Array,
    sorted_token_ids: jax.Array,
    group_sizes: jax.Array,
    expert_indices: jax.Array,
    expert_weights: jax.Array,
    gate_proj: jax.Array,
    up_proj: jax.Array,
    down_proj: jax.Array,
) -> jax.Array:
    """Expert computation using ragged_dot with SwiGLU.

    Each expert applies: down @ (silu(x @ gate) * (x @ up))
    Uses ragged_dot for the batched matmuls across experts.

    Args:
        x: (M, D) all tokens.
        sorted_token_ids: (M*k,) token indices sorted by expert.
        group_sizes: (n_experts,) tokens per expert.
        expert_indices: (M, k) which experts each token selected.
        expert_weights: (M, k) routing weights.
        gate_proj: (n_experts, D, intermediate_dim) expert gate weights.
        up_proj: (n_experts, D, intermediate_dim) expert up weights.
        down_proj: (n_experts, intermediate_dim, D) expert down weights.

    Returns:
        (M, D) weighted sum of expert outputs.
    """
    M, D = x.shape
    k = expert_indices.shape[1]

    # Extract raw arrays from fp8 dicts if quantized
    # For ragged_dot, dequantize expert weights back to activation dtype
    # (native fp8 ragged_dot can be added later)
    def _get_expert_weight(w):
        if isinstance(w, dict) and 'w' in w:
            # Dequantize: w is (E, out, in), scale_inv is (E, out, 1)
            # Transpose back to (E, in, out) for ragged_dot
            w_f = (w['w'].astype(jnp.float32) * w['scale_inv'])
            return jnp.transpose(w_f, (*range(w_f.ndim - 2), w_f.ndim - 1, w_f.ndim - 2)).astype(x.dtype)
        return w

    gate_w = _get_expert_weight(gate_proj)
    up_w = _get_expert_weight(up_proj)
    down_w = _get_expert_weight(down_proj)
    n_experts = gate_w.shape[0]

    # Gather tokens in expert-sorted order
    x_sorted = x[sorted_token_ids]  # (M*k, D)

    # SwiGLU expert computation via ragged_dot:
    gate_out = jax.nn.silu(jax.lax.ragged_dot(x_sorted, gate_w, group_sizes))
    up_out = jax.lax.ragged_dot(x_sorted, up_w, group_sizes)
    hidden = gate_out * up_out
    expert_out = jax.lax.ragged_dot(hidden, down_w, group_sizes)  # (M*k, D)

    # Scatter back: accumulate weighted expert outputs per token
    # Build the inverse mapping: for each sorted position, which token and which expert slot
    flat_expert_slot_weights = expert_weights.reshape(-1)  # (M*k,)

    # We need to reconstruct: for sorted position i, the weight is expert_weights[token_id, slot]
    # Since sorted_token_ids and the sort order came from flattening (token, slot),
    # we need the original flat weights in sorted order.
    flat_token_ids = jnp.repeat(jnp.arange(M), k)
    flat_expert_ids = expert_indices.reshape(-1)
    sort_order = jnp.argsort(flat_expert_ids)
    sorted_weights = flat_expert_slot_weights[sort_order]  # (M*k,)

    # Weight the expert outputs
    weighted_out = expert_out * sorted_weights[:, None]  # (M*k, D)

    # Scatter-add back to token positions
    output = jnp.zeros((M, D), dtype=x.dtype)
    output = output.at[sorted_token_ids].add(weighted_out.astype(x.dtype))

    return output


def shared_expert_forward(
    x: jax.Array,
    gate_proj: jax.Array,
    up_proj: jax.Array,
    down_proj: jax.Array,
) -> jax.Array:
    """Shared expert (always active, standard SwiGLU MLP).

    Args:
        x: (M, D)
        gate_proj: (D, intermediate_dim)
        up_proj: (D, intermediate_dim)
        down_proj: (intermediate_dim, D)

    Returns:
        (M, D)
    """
    gate = jax.nn.silu(matmul_maybe_fp8(x, gate_proj))
    up = matmul_maybe_fp8(x, up_proj)
    return matmul_maybe_fp8(gate * up, down_proj)


def moe_layer(
    x: jax.Array,
    params: dict,
    n_experts_per_token: int,
) -> jax.Array:
    """Full MoE layer: route + routed experts + shared expert.

    Args:
        x: (B, T, D) input.
        params: dict with keys:
            gate_weight: (D, n_experts)
            gate_proj: (n_experts, D, intermediate_dim)
            up_proj: (n_experts, D, intermediate_dim)
            down_proj: (n_experts, intermediate_dim, D)
            shared_gate_proj: (D, shared_intermediate_dim)
            shared_up_proj: (D, shared_intermediate_dim)
            shared_down_proj: (shared_intermediate_dim, D)
        n_experts_per_token: top-k.

    Returns:
        (B, T, D)
    """
    B, T, D = x.shape
    M = B * T
    x_flat = x.reshape(M, D)

    with jax.named_scope('moe_routing'):
        expert_indices, expert_weights, sorted_token_ids, group_sizes = moe_routing(
            x_flat, params['gate_weight'], n_experts_per_token,
        )

    with jax.named_scope('moe_experts'):
        routed_out = expert_forward(
            x_flat, sorted_token_ids, group_sizes,
            expert_indices, expert_weights,
            params['gate_proj'], params['up_proj'], params['down_proj'],
        )

    with jax.named_scope('moe_shared_expert'):
        shared_out = shared_expert_forward(
            x_flat,
            params['shared_gate_proj'],
            params['shared_up_proj'],
            params['shared_down_proj'],
        )
        shared_gate = jax.nn.sigmoid(matmul_maybe_fp8(x_flat, params['shared_expert_gate_weight']))
        shared_out = shared_gate * shared_out

    output = routed_out + shared_out
    return output.reshape(B, T, D).astype(x.dtype)
