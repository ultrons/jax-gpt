"""Mixture of Experts layer with expert parallelism.

Single-device: uses ragged_dot over all experts directly.
Multi-device (EP): shard_map + psum following MaxText/Megablox pattern.
Each device handles its local expert shard — no all-gather of weights.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P

from jax_gpt.models.qwen35.fp8 import matmul_maybe_fp8


def _get_n_experts(gate_weight) -> int:
    """Extract n_experts from gate_weight (handles fp8 dict or plain array)."""
    if isinstance(gate_weight, dict) and 'w' in gate_weight:
        return gate_weight['w'].shape[0]
    return gate_weight.shape[1]


def _get_expert_weight(w, dtype=None):
    """Extract raw array from fp8 dict if quantized, for ragged_dot."""
    if isinstance(w, dict) and 'w' in w:
        w_f = (w['w'].astype(jnp.float32) * w['scale_inv'])
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
    """Sort tokens by expert assignment for ragged_dot."""
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
    mesh,
    axis_name: str = 'tp',
) -> jax.Array:
    """Expert-parallel MoE using shard_map.

    Following the MaxText/Megablox pattern from Robert Dyro's presentation:
    - shard_map gives each device explicit control via axis_index
    - Each device slices group_sizes to its local experts
    - Local ragged_dot on local expert weights (no all-gather)
    - psum to reduce across EP devices

    Args:
        x: (M, D) tokens.
        expert_indices: (M, k) global expert indices.
        expert_weights: (M, k) routing weights.
        gate_proj, up_proj, down_proj: expert weights sharded along expert dim.
        mesh: device mesh.
        axis_name: mesh axis for EP.
    """
    from jax.experimental.shard_map import shard_map

    gate_w = _get_expert_weight(gate_proj, x.dtype)
    up_w = _get_expert_weight(up_proj, x.dtype)
    down_w = _get_expert_weight(down_proj, x.dtype)
    E_local = gate_w.shape[0]
    M, D = x.shape
    k = expert_indices.shape[1]
    n_devices = mesh.shape[axis_name]
    E_total = E_local * n_devices

    # Compute group_sizes for ALL experts (on each device, same result)
    flat_expert_ids = expert_indices.reshape(-1)  # (M*k,)
    group_sizes_all = jnp.zeros(E_total, dtype=jnp.int32)
    group_sizes_all = group_sizes_all.at[flat_expert_ids].add(1)

    # Sort tokens by global expert id
    flat_token_ids = jnp.repeat(jnp.arange(M), k)
    flat_weights = expert_weights.reshape(-1)
    sort_order = jnp.argsort(flat_expert_ids)
    sorted_token_ids = flat_token_ids[sort_order]
    sorted_weights = flat_weights[sort_order]
    x_sorted = x[sorted_token_ids]  # (M*k, D) sorted by global expert

    # The shard_map expert function: each device processes its local shard
    # Precompute cumulative offsets for each device's expert range
    # group_offsets[i] = sum of group_sizes for experts 0..i-1
    group_offsets = jnp.concatenate([jnp.array([0]), jnp.cumsum(group_sizes_all)])

    @partial(shard_map, mesh=mesh,
             in_specs=(P(), P(), P(), P(), P(axis_name, None, None),
                       P(axis_name, None, None), P(axis_name, None, None), P()),
             out_specs=P(),
             check_rep=False)
    def _expert_fn(x_sorted, sorted_weights, group_sizes_all, group_offsets,
                   local_gate, local_up, local_down, sorted_token_ids):
        # Which device am I?
        my_idx = jax.lax.axis_index(axis_name)

        # E_local from the actual local weight shape (after shard_map splits)
        e_local = local_gate.shape[0]

        # My local group sizes
        group_sizes_local = jax.lax.dynamic_slice(
            group_sizes_all, (my_idx * e_local,), (e_local,))

        # Start offset in the sorted array for my experts
        start_offset = group_offsets[my_idx * e_local]

        # Slice my tokens from the sorted array (use M*k as max, mask later)
        x_local = jax.lax.dynamic_slice(x_sorted, (start_offset, 0), (M * k, D))
        weights_local = jax.lax.dynamic_slice(sorted_weights, (start_offset,), (M * k,))
        token_ids_local = jax.lax.dynamic_slice(sorted_token_ids, (start_offset,), (M * k,))

        # How many tokens are actually mine
        n_local_tokens = jnp.sum(group_sizes_local)

        # Compute SwiGLU on local experts
        local_out = _expert_swiglu(x_local, group_sizes_local,
                                   local_gate, local_up, local_down)

        # Mask out padding tokens (beyond my actual count)
        mask = jnp.arange(M * k) < n_local_tokens
        local_out = jnp.where(mask[:, None], local_out, 0.0)
        weights_local = jnp.where(mask, weights_local, 0.0)

        # Scatter weighted outputs back to token positions
        weighted = local_out * weights_local[:, None]
        output = jnp.zeros((M, D), dtype=x_sorted.dtype)
        output = output.at[token_ids_local].add(weighted.astype(output.dtype))

        # psum across all EP devices to combine expert contributions
        output = jax.lax.psum(output, axis_name)
        return output

    return _expert_fn(x_sorted, sorted_weights, group_sizes_all, group_offsets,
                      gate_w, up_w, down_w, sorted_token_ids)


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
    mesh=None,
) -> jax.Array:
    """Full MoE layer: route + routed experts + shared expert.

    When n_devices > 1 and mesh is provided, uses shard_map EP.
    Expert weights are never all-gathered — each device only computes on
    its local expert shard.
    """
    B, T, D = x.shape
    M = B * T
    x_flat = x.reshape(M, D)

    with jax.named_scope('moe_routing'):
        expert_indices, expert_weights = moe_routing(
            x_flat, params['gate_weight'], n_experts_per_token,
        )

    with jax.named_scope('moe_experts'):
        if n_devices > 1 and mesh is not None:
            routed_out = expert_forward_ep(
                x_flat, expert_indices, expert_weights,
                params['gate_proj'], params['up_proj'], params['down_proj'],
                mesh=mesh, axis_name=axis_name,
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
