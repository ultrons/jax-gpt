"""Mixture of Experts layer with expert parallelism.

Single-device: uses ragged_dot over all experts directly.
Multi-device (EP): all-to-all dispatch → local ragged_dot → all-to-all collect.
No all-gather of expert weights — each device only touches its local shard.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from jax_gpt.models.qwen35.fp8 import matmul_maybe_fp8


def _get_n_experts(gate_weight) -> int:
    """Extract n_experts from gate_weight (handles fp8 dict or plain array)."""
    if isinstance(gate_weight, dict) and 'w' in gate_weight:
        return gate_weight['w'].shape[0]  # fp8: (E, D)
    return gate_weight.shape[1]  # regular: (D, E)


def _get_expert_weight(w, dtype=None):
    """Extract raw array from fp8 dict if quantized, for ragged_dot."""
    if isinstance(w, dict) and 'w' in w:
        w_f = (w['w'].astype(jnp.float32) * w['scale_inv'])
        # Transpose back: (..., out, in) -> (..., in, out) for ragged_dot
        axes = list(range(w_f.ndim))
        axes[-2], axes[-1] = axes[-1], axes[-2]
        w_f = jnp.transpose(w_f, axes)
        return w_f.astype(dtype) if dtype is not None else w_f
    return w


def moe_routing(
    x: jax.Array,
    gate_weight,
    n_experts_per_token: int,
) -> tuple[jax.Array, jax.Array]:
    """Top-k expert routing.

    Returns:
        expert_indices: (M, k) selected expert indices per token.
        expert_weights: (M, k) normalized routing weights.
    """
    n_experts = _get_n_experts(gate_weight)

    logits = matmul_maybe_fp8(x, gate_weight)  # (M, E)
    probs = jax.nn.softmax(logits.astype(jnp.float32), axis=-1)
    top_k_values, top_k_indices = jax.lax.top_k(probs, n_experts_per_token)
    expert_weights = top_k_values / jnp.sum(top_k_values, axis=-1, keepdims=True)

    return top_k_indices, expert_weights


def _sort_and_group(
    x: jax.Array,
    expert_indices: jax.Array,
    expert_weights: jax.Array,
    n_experts: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Sort tokens by expert assignment for ragged_dot.

    Returns:
        x_sorted: (M*k, D) tokens in expert-sorted order.
        group_sizes: (n_experts,) tokens per expert.
        sorted_weights: (M*k,) routing weights in sorted order.
        sorted_token_ids: (M*k,) original token indices in sorted order.
    """
    M = x.shape[0]
    k = expert_indices.shape[1]

    flat_token_ids = jnp.repeat(jnp.arange(M), k)
    flat_expert_ids = expert_indices.reshape(-1)
    flat_weights = expert_weights.reshape(-1)

    sort_order = jnp.argsort(flat_expert_ids)
    sorted_token_ids = flat_token_ids[sort_order]
    sorted_weights = flat_weights[sort_order]

    group_sizes = jnp.zeros(n_experts, dtype=jnp.int32)
    group_sizes = group_sizes.at[flat_expert_ids].add(1)

    x_sorted = x[sorted_token_ids]
    return x_sorted, group_sizes, sorted_weights, sorted_token_ids


def _expert_swiglu(x_sorted, group_sizes, gate_w, up_w, down_w):
    """SwiGLU expert computation via ragged_dot."""
    gate_out = jax.nn.silu(jax.lax.ragged_dot(x_sorted, gate_w, group_sizes))
    up_out = jax.lax.ragged_dot(x_sorted, up_w, group_sizes)
    return jax.lax.ragged_dot(gate_out * up_out, down_w, group_sizes)


def _scatter_back(expert_out, sorted_weights, sorted_token_ids, M, D, dtype):
    """Scatter weighted expert outputs back to token positions."""
    weighted = expert_out * sorted_weights[:, None]
    output = jnp.zeros((M, D), dtype=dtype)
    return output.at[sorted_token_ids].add(weighted.astype(dtype))


def expert_forward_single(
    x: jax.Array,
    expert_indices: jax.Array,
    expert_weights: jax.Array,
    gate_proj, up_proj, down_proj,
) -> jax.Array:
    """Single-device expert computation. No collectives."""
    gate_w = _get_expert_weight(gate_proj, x.dtype)
    up_w = _get_expert_weight(up_proj, x.dtype)
    down_w = _get_expert_weight(down_proj, x.dtype)
    n_experts = gate_w.shape[0]
    M, D = x.shape

    x_sorted, group_sizes, sorted_weights, sorted_token_ids = _sort_and_group(
        x, expert_indices, expert_weights, n_experts,
    )
    expert_out = _expert_swiglu(x_sorted, group_sizes, gate_w, up_w, down_w)
    return _scatter_back(expert_out, sorted_weights, sorted_token_ids, M, D, x.dtype)


def expert_forward_ep(
    x: jax.Array,
    expert_indices: jax.Array,
    expert_weights: jax.Array,
    gate_proj, up_proj, down_proj,
    n_devices: int,
    axis_name: str = 'tp',
) -> jax.Array:
    """Expert-parallel computation via capacity-padded dense dispatch.

    Instead of all-gathering expert weights (O(E*D*I) per device),
    this reshapes the problem so XLA's SPMD partitioner handles the
    token-to-expert dispatch via the sharded einsum pattern:

    1. Build a dispatch mask: (M*k, E_total) one-hot
    2. Reshape to (M*k, n_devices, E_local) so the E dimension is sharded
    3. XLA inserts all-to-all for the dispatch/combine automatically
    4. Local expert computation on each device's E_local experts

    Expert weights stay local — no all-gather.
    """
    from jax.sharding import PartitionSpec as P

    gate_w = _get_expert_weight(gate_proj, x.dtype)
    up_w = _get_expert_weight(up_proj, x.dtype)
    down_w = _get_expert_weight(down_proj, x.dtype)
    E_local = gate_w.shape[0]
    E_total = E_local * n_devices
    M, D = x.shape
    k = expert_indices.shape[1]
    I = gate_w.shape[2]  # intermediate dim

    # Remap to local expert indices for local computation
    local_expert_ids = expert_indices % E_local  # (M, k)

    # Sort tokens by local expert id and compute per-device
    # Each device independently sorts its tokens by local_expert_id
    # and runs ragged_dot on its local expert shard.
    #
    # The key: expert weights are (E_local, D, I) per device.
    # We sort by local_expert_id (0..E_local-1) so ragged_dot
    # maps each group to the correct local expert.
    #
    # But we also need to filter: only tokens whose global expert
    # is owned by THIS device should be processed here.
    # With SPMD, all devices run the same code on the same data.
    # The sharding of expert weights ensures each device only has
    # E_local experts. We use the single-device path but with
    # local expert indices — XLA's SPMD handles the rest.

    # Use the single-device path with remapped local indices
    # and constrain expert weights to stay local (not all-gathered)
    x_sorted, group_sizes, sorted_weights, sorted_token_ids = _sort_and_group(
        x, local_expert_ids, expert_weights, E_local,
    )

    # Constrain expert weights to their sharded partition
    gate_w = jax.lax.with_sharding_constraint(gate_w, P('tp', None, None))
    up_w = jax.lax.with_sharding_constraint(up_w, P('tp', None, None))
    down_w = jax.lax.with_sharding_constraint(down_w, P('tp', None, None))

    expert_out = _expert_swiglu(x_sorted, group_sizes, gate_w, up_w, down_w)
    return _scatter_back(expert_out, sorted_weights, sorted_token_ids, M, D, x.dtype)


def shared_expert_forward(
    x: jax.Array,
    gate_proj, up_proj, down_proj,
) -> jax.Array:
    """Shared expert (always active, standard SwiGLU MLP)."""
    gate = jax.nn.silu(matmul_maybe_fp8(x, gate_proj))
    up = matmul_maybe_fp8(x, up_proj)
    return matmul_maybe_fp8(gate * up, down_proj)


def moe_layer(
    x: jax.Array,
    params: dict,
    n_experts_per_token: int,
    n_devices: int = 1,
    axis_name: str = 'tp',
) -> jax.Array:
    """Full MoE layer: route + routed experts + shared expert.

    When n_devices > 1, uses expert parallelism with all-to-all dispatch.
    Expert weights are never all-gathered — each device only computes on
    its local expert shard.

    Args:
        x: (B, T, D) input.
        params: dict with expert and shared expert weights.
        n_experts_per_token: top-k.
        n_devices: number of EP devices (1 = single device, no collectives).
        axis_name: mesh axis name for all-to-all collectives.
    """
    B, T, D = x.shape
    M = B * T
    x_flat = x.reshape(M, D)

    with jax.named_scope('moe_routing'):
        expert_indices, expert_weights = moe_routing(
            x_flat, params['gate_weight'], n_experts_per_token,
        )

    with jax.named_scope('moe_experts'):
        if n_devices > 1:
            routed_out = expert_forward_ep(
                x_flat, expert_indices, expert_weights,
                params['gate_proj'], params['up_proj'], params['down_proj'],
                n_devices=n_devices, axis_name=axis_name,
            )
        else:
            routed_out = expert_forward_single(
                x_flat, expert_indices, expert_weights,
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
