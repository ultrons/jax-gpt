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

from jax_gpt.models.qwen35.fp8 import matmul_maybe_fp8, FP8_DTYPE, dynamic_quantize_fp8


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


def _is_fp8_weight(w) -> bool:
    """Check if weight is an fp8 quantized dict."""
    return isinstance(w, dict) and 'w' in w and w['w'].dtype == FP8_DTYPE


def _fp8_expert_components(w):
    """Extract fp8 weight in ragged_dot layout (E, K, N) + per-output scale.

    Stored: (E, N, K) fp8 + (E, N, 1) scale_inv.
    Returns: (E, K, N) fp8, (E, N) scale_inv.
    """
    w_fp8 = jnp.transpose(w['w'], (0, 2, 1))
    scale = w['scale_inv'].squeeze(-1)
    return w_fp8, scale


def _fp8_ragged_dot_rescaled(x_fp8, x_scale, w_fp8, group_sizes, w_scale,
                              n_tokens):
    """Native fp8 ragged_dot with activation + weight rescaling."""
    out = jax.lax.ragged_dot(
        x_fp8, w_fp8, group_sizes,
        preferred_element_type=jnp.float32,
    )
    out = out * x_scale
    w_scale_per_token = jnp.repeat(
        w_scale, group_sizes, axis=0, total_repeat_length=n_tokens)
    return out * w_scale_per_token


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


def _expert_swiglu(x_sorted, group_sizes, gate_w, up_w, down_w,
                    gate_scale=None, up_scale=None, down_scale=None):
    """SwiGLU expert computation via ragged_dot.

    When weights are fp8 and scales are provided, uses native fp8 hardware.
    """
    if gate_w.dtype == FP8_DTYPE:
        n = x_sorted.shape[0]
        x_fp8, x_scale = dynamic_quantize_fp8(x_sorted)
        gate_out = jax.nn.silu(
            _fp8_ragged_dot_rescaled(
                x_fp8, x_scale, gate_w, group_sizes, gate_scale, n))
        up_out = _fp8_ragged_dot_rescaled(
            x_fp8, x_scale, up_w, group_sizes, up_scale, n)
        intermediate = gate_out * up_out
        int_fp8, int_scale = dynamic_quantize_fp8(intermediate)
        return _fp8_ragged_dot_rescaled(
            int_fp8, int_scale, down_w, group_sizes, down_scale, n)
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
    M, D = x.shape

    if _is_fp8_weight(gate_proj):
        gate_fp8, gate_s = _fp8_expert_components(gate_proj)
        up_fp8, up_s = _fp8_expert_components(up_proj)
        down_fp8, down_s = _fp8_expert_components(down_proj)
        n_experts = gate_fp8.shape[0]
        x_sorted, group_sizes, sorted_weights, sorted_token_ids = (
            _sort_and_group(x, expert_indices, expert_weights, n_experts))
        expert_out = _expert_swiglu(
            x_sorted, group_sizes,
            gate_fp8, up_fp8, down_fp8, gate_s, up_s, down_s)
        return _scatter_back(
            expert_out, sorted_weights, sorted_token_ids, M, D, x.dtype)

    gate_w = _get_expert_weight(gate_proj, x.dtype)
    up_w = _get_expert_weight(up_proj, x.dtype)
    down_w = _get_expert_weight(down_proj, x.dtype)
    n_experts = gate_w.shape[0]

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

    Memory-efficient implementation following MaxText/Megablox pattern:
    - x (M, D) is passed replicated into shard_map — only ~1 GB all-gather
      instead of the old approach which all-gathered the expanded (M*k, D)
    - Sort, gather, and ragged_dot happen INSIDE shard_map per device
    - Each device processes all tokens against its local expert shard
    - psum combines results across EP devices

    Memory per device: O(M * k * D) where M = B_local * T.
    Old approach was O(M * k * TP * D) due to all-gather of expanded tensor.

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
    M, D = x.shape
    k = expert_indices.shape[1]

    # Detect data-parallel axes: all non-trivial mesh axes except the EP axis.
    # x is sharded along these axes (batch dim), and must stay sharded —
    # using P() would force an all-gather of x across dp, blowing up memory.
    dp_axes = tuple(name for name in mesh.axis_names
                    if name != axis_name and mesh.shape[name] > 1)
    if len(dp_axes) == 0:
        act_pspec = P(None, None)          # no dp axis, x is replicated
    elif len(dp_axes) == 1:
        act_pspec = P(dp_axes[0], None)    # single dp axis (common case)
    else:
        act_pspec = P(dp_axes, None)       # multi-axis dp (dp + fsdp)

    @partial(shard_map, mesh=mesh,
             in_specs=(act_pspec,                    # x: dp-sharded, tp-replicated
                       act_pspec,                    # expert_indices: same as x
                       act_pspec,                    # expert_weights: same as x
                       P(axis_name, None, None),     # gate_w: E-sharded
                       P(axis_name, None, None),     # up_w: E-sharded
                       P(axis_name, None, None)),    # down_w: E-sharded
             out_specs=act_pspec,
             check_rep=False)
    def _expert_fn(x, indices, weights, local_gate, local_up, local_down):
        my_idx = jax.lax.axis_index(axis_name)
        e_local = local_gate.shape[0]
        m_local, d = x.shape  # per-dp-shard token count

        # Flatten (M_local, k) → (M_local*k,)
        flat_idx = indices.reshape(-1)   # global expert ids
        flat_w = weights.reshape(-1)

        # Map to local expert range; non-local → e_local (sorts to end)
        local_start = my_idx * e_local
        valid = (flat_idx >= local_start) & (flat_idx < local_start + e_local)
        mapped = jnp.where(valid, flat_idx - local_start, e_local)

        # Sort by mapped expert id — local expert tokens first
        order = jnp.argsort(mapped)

        # Gather sorted tokens (order // k gives original token index)
        x_sorted = x[order // k]  # (M_local*k, D)

        # Group sizes for local experts via masked scatter-add
        local_idx = jnp.where(valid, flat_idx - local_start, 0)
        group_sizes = jnp.zeros(e_local, dtype=jnp.int32)
        group_sizes = group_sizes.at[local_idx].add(valid.astype(jnp.int32))

        # SwiGLU via ragged_dot (processes first sum(group_sizes) rows)
        gate_out = jax.nn.silu(
            jax.lax.ragged_dot(x_sorted, local_gate, group_sizes))
        up_out = jax.lax.ragged_dot(x_sorted, local_up, group_sizes)
        expert_out = jax.lax.ragged_dot(
            gate_out * up_out, local_down, group_sizes)

        # Zero non-local positions and apply routing weights
        sorted_valid = valid[order]
        expert_out = jnp.where(sorted_valid[:, None], expert_out, 0.0)
        expert_out = expert_out * flat_w[order][:, None]

        # Scatter weighted outputs back to token positions
        output = jnp.zeros((m_local, d), dtype=x.dtype)
        output = output.at[order // k].add(expert_out.astype(x.dtype))

        # Sum across EP devices
        output = jax.lax.psum(output, axis_name)
        return output

    return _expert_fn(x, expert_indices, expert_weights,
                      gate_w, up_w, down_w)


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
