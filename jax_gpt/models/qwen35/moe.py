"""Mixture of Experts layer with expert parallelism.

Single-device: uses ragged_dot over all experts directly.
Multi-device (EP): shard_map + psum following MaxText/Megablox pattern.
Each device handles its local expert shard — no all-gather of weights.

Supports two MoE backends via `moe_backend` parameter:
- 'ragged_dot' (default): Uses jax.lax.ragged_dot (XLA-compiled).
- 'gmm': Uses megablox gmm_v2 Pallas kernel. Bypasses XLA tiling —
  reads HBM directly, eliminating squeeze/retiling overhead in decode.
"""

from __future__ import annotations

from functools import partial
from typing import Literal

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P

from jax_gpt.models.qwen35.fp8 import matmul_maybe_fp8, FP8_DTYPE, dynamic_quantize_fp8

MoeBackend = Literal['ragged_dot', 'gmm']


def _get_n_experts(gate_weight) -> int:
    """Extract n_experts from gate_weight (handles fp8 dict or plain array)."""
    if isinstance(gate_weight, dict) and 'w' in gate_weight:
        return gate_weight['w'].shape[0]
    return gate_weight.shape[1]


def _get_expert_weight(w, dtype=None):
    """Extract raw array from fp8 dict if quantized, for ragged_dot.

    Expert weights are stored in ragged_dot layout (E, K, N) with
    scale_inv (E, 1, N). Dequant: w_real = w_fp8 / scale_inv.
    """
    if isinstance(w, dict) and 'w' in w:
        w_f = w['w'].astype(jnp.float32) / w['scale_inv']
        return w_f.astype(dtype) if dtype is not None else w_f
    return w


def _is_fp8_weight(w) -> bool:
    """Check if weight is an fp8 quantized dict."""
    return isinstance(w, dict) and 'w' in w and w['w'].dtype == FP8_DTYPE


def _fp8_expert_components(w):
    """Extract fp8 weight + rescale factor for native fp8 ragged_dot.

    Expert weights stored in ragged_dot layout (E, K, N) fp8 with
    scale_inv (E, 1, N) where scale_inv = 1/scale.
    Returns: (E, K, N) fp8 (no copy), (E, N) rescale factor (= scale).
    """
    w_fp8 = w['w']  # already in ragged_dot layout
    w_scale = (1.0 / w['scale_inv']).squeeze(-2)
    return w_fp8, w_scale


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


def _fp8_gmm_components(w):
    """Extract fp8 weight + rhs_scale in gmm_v2 format.

    Expert weights stored as {'w': (...,K,N) fp8, 'scale_inv': (...,1,N) float32}.
    gmm_v2 expects rhs_scale: (..., num_blocks, 1, N). Our weights use a single
    scale block (block_size = K), so num_blocks = 1.

    Handles both 3D (E,K,N) and 4D (N_L,E,K,N) stacked weights.
    """
    w_fp8 = w['w']  # (..., K, N) fp8
    # scale_inv = 1/scale → scale = 1/scale_inv
    # Insert num_blocks=1 dim: (..., 1, N) → (..., 1, 1, N)
    rhs_scale = jnp.expand_dims(1.0 / w['scale_inv'], axis=-2)
    return w_fp8, rhs_scale


def _gmm_matmul(x, w, group_sizes, rhs_scale=None, group_offset=None,
                rhs_group_offset=None):
    """Single gmm_v2 matmul. Lazy-imported to avoid top-level Pallas import."""
    from jax_gpt.models.qwen35.megablox import gmm_v2
    return gmm_v2(
        x, w, group_sizes,
        rhs_scale=rhs_scale,
        group_offset=group_offset,
        rhs_group_offset=rhs_group_offset,
        maybe_quantize_lhs=rhs_scale is not None,
    )


def _expert_swiglu_gmm(x_sorted, group_sizes, gate_w, up_w, down_w,
                         gate_scale=None, up_scale=None, down_scale=None,
                         group_offset=None, rhs_group_offset=None):
    """SwiGLU expert computation via gmm_v2 Pallas kernel.

    When rhs_scale is provided, gmm_v2 handles lhs quantization internally.
    """
    with jax.named_scope('expert_gate_up'):
        gate_out = jax.nn.silu(
            _gmm_matmul(x_sorted, gate_w, group_sizes, gate_scale,
                         group_offset, rhs_group_offset))
        up_out = _gmm_matmul(x_sorted, up_w, group_sizes, up_scale,
                              group_offset, rhs_group_offset)
        intermediate = (gate_out * up_out).astype(x_sorted.dtype)
    with jax.named_scope('expert_down'):
        return _gmm_matmul(intermediate, down_w, group_sizes, down_scale,
                            group_offset, rhs_group_offset)


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

    with jax.named_scope('moe_router'):
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
        with jax.named_scope('expert_gate_up'):
            gate_out = jax.nn.silu(
                _fp8_ragged_dot_rescaled(
                    x_fp8, x_scale, gate_w, group_sizes, gate_scale, n))
            up_out = _fp8_ragged_dot_rescaled(
                x_fp8, x_scale, up_w, group_sizes, up_scale, n)
            intermediate = gate_out * up_out
            int_fp8, int_scale = dynamic_quantize_fp8(intermediate)
        with jax.named_scope('expert_down'):
            return _fp8_ragged_dot_rescaled(
                int_fp8, int_scale, down_w, group_sizes, down_scale, n)
    with jax.named_scope('expert_gate_up'):
        gate_out = jax.nn.silu(jax.lax.ragged_dot(x_sorted, gate_w, group_sizes))
        up_out = jax.lax.ragged_dot(x_sorted, up_w, group_sizes)
    with jax.named_scope('expert_down'):
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
    moe_backend: MoeBackend = 'ragged_dot',
    layer_idx: int | None = None,
    n_experts_override: int | None = None,
) -> jax.Array:
    """Single-device expert computation. No collectives.

    Args:
        layer_idx: When set, expert weights are stacked (N_L*E, K, N).
            rhs_group_offset = layer_idx * n_experts is passed to gmm_v2.
        n_experts_override: Routing expert count when weights are stacked
            (since weight dim-0 > E).
    """
    M, D = x.shape
    swiglu_fn = _expert_swiglu_gmm if moe_backend == 'gmm' else _expert_swiglu

    rhs_group_offset = None
    if layer_idx is not None and moe_backend == 'gmm':
        n_experts = n_experts_override
        rhs_group_offset = jnp.array([layer_idx * n_experts], dtype=jnp.int32)
    else:
        n_experts = None  # derived from weight shape below

    if _is_fp8_weight(gate_proj):
        if moe_backend == 'gmm':
            gate_fp8, gate_s = _fp8_gmm_components(gate_proj)
            up_fp8, up_s = _fp8_gmm_components(up_proj)
            down_fp8, down_s = _fp8_gmm_components(down_proj)
        else:
            gate_fp8, gate_s = _fp8_expert_components(gate_proj)
            up_fp8, up_s = _fp8_expert_components(up_proj)
            down_fp8, down_s = _fp8_expert_components(down_proj)
        if n_experts is None:
            n_experts = gate_fp8.shape[0]
        x_sorted, group_sizes, sorted_weights, sorted_token_ids = (
            _sort_and_group(x, expert_indices, expert_weights, n_experts))
        if moe_backend == 'gmm':
            expert_out = swiglu_fn(
                x_sorted, group_sizes,
                gate_fp8, up_fp8, down_fp8, gate_s, up_s, down_s,
                rhs_group_offset=rhs_group_offset)
        else:
            expert_out = swiglu_fn(
                x_sorted, group_sizes,
                gate_fp8, up_fp8, down_fp8, gate_s, up_s, down_s)
        return _scatter_back(
            expert_out, sorted_weights, sorted_token_ids, M, D, x.dtype)

    gate_w = _get_expert_weight(gate_proj, x.dtype)
    up_w = _get_expert_weight(up_proj, x.dtype)
    down_w = _get_expert_weight(down_proj, x.dtype)
    if n_experts is None:
        n_experts = gate_w.shape[0]

    x_sorted, group_sizes, sorted_weights, sorted_token_ids = _sort_and_group(
        x, expert_indices, expert_weights, n_experts,
    )
    if moe_backend == 'gmm':
        expert_out = swiglu_fn(x_sorted, group_sizes, gate_w, up_w, down_w,
                               rhs_group_offset=rhs_group_offset)
    else:
        expert_out = swiglu_fn(x_sorted, group_sizes, gate_w, up_w, down_w)
    return _scatter_back(expert_out, sorted_weights, sorted_token_ids, M, D, x.dtype)


def _ep_inner_body(x, indices, weights, k, axis_name,
                    gate_w, up_w, down_w,
                    gate_s=None, up_s=None, down_s=None,
                    moe_backend: MoeBackend = 'ragged_dot',
                    rhs_group_offset=None, n_local_experts=None):
    """Core EP expert logic, called inside shard_map.

    When scales are provided, gate_w/up_w/down_w are fp8 in (E, K, N) layout.
    moe_backend selects ragged_dot (XLA) or gmm (Pallas kernel).

    Args:
        rhs_group_offset: Optional int32[1] offset for stacked weight indexing.
        n_local_experts: When weights are stacked (rhs dim-0 > E_local), pass
            the actual local expert count here. Otherwise derived from gate_w.shape[0].
    """
    my_idx = jax.lax.axis_index(axis_name)
    e_local = n_local_experts if n_local_experts is not None else gate_w.shape[0]
    m_local, d = x.shape

    flat_idx = indices.reshape(-1)
    flat_w = weights.reshape(-1)

    local_start = my_idx * e_local
    valid = (flat_idx >= local_start) & (flat_idx < local_start + e_local)
    mapped = jnp.where(valid, flat_idx - local_start, e_local)

    order = jnp.argsort(mapped)
    x_sorted = x[order // k]

    local_idx = jnp.where(valid, flat_idx - local_start, 0)
    group_sizes = jnp.zeros(e_local, dtype=jnp.int32)
    group_sizes = group_sizes.at[local_idx].add(valid.astype(jnp.int32))

    if moe_backend == 'gmm':
        expert_out = _expert_swiglu_gmm(x_sorted, group_sizes,
                                         gate_w, up_w, down_w,
                                         gate_s, up_s, down_s,
                                         rhs_group_offset=rhs_group_offset)
    else:
        expert_out = _expert_swiglu(x_sorted, group_sizes,
                                     gate_w, up_w, down_w,
                                     gate_s, up_s, down_s)

    sorted_valid = valid[order]
    expert_out = jnp.where(sorted_valid[:, None], expert_out, 0.0)
    expert_out = expert_out * flat_w[order][:, None]
    output = jnp.zeros((m_local, d), dtype=x.dtype)
    output = output.at[order // k].add(expert_out.astype(x.dtype))
    output = jax.lax.psum(output, axis_name)
    return output


def expert_forward_ep(
    x: jax.Array,
    expert_indices: jax.Array,
    expert_weights: jax.Array,
    gate_proj, up_proj, down_proj,
    mesh,
    axis_name: str = 'tp',
    moe_backend: MoeBackend = 'ragged_dot',
    layer_idx: int | None = None,
) -> jax.Array:
    """Expert-parallel MoE using shard_map.

    Memory-efficient implementation following MaxText/Megablox pattern:
    - x (M, D) is passed replicated into shard_map — only ~1 GB all-gather
      instead of the old approach which all-gathered the expanded (M*k, D)
    - Sort, gather, and ragged_dot/gmm happen INSIDE shard_map per device
    - Each device processes all tokens against its local expert shard
    - psum combines results across EP devices

    Args:
        x: (M, D) tokens.
        expert_indices: (M, k) global expert indices.
        expert_weights: (M, k) routing weights.
        gate_proj, up_proj, down_proj: expert weights sharded along expert dim.
            When layer_idx is set, these are 4D stacked (N_L, E, K, N) with
            sharding P(None, axis_name, None, None).
        mesh: device mesh.
        axis_name: mesh axis for EP.
        moe_backend: 'ragged_dot' or 'gmm'.
        layer_idx: When set, expert weights are stacked across layers (4D).
            gmm_v2 uses rhs_group_offset to index the right layer's portion.
    """
    from jax.experimental.shard_map import shard_map

    M, D = x.shape
    k = expert_indices.shape[1]

    # Data-parallel axes: only the 'dp' axis partitions the batch.
    # Explicitly exclude 'tp' (tensor-parallel weight axis) and the EP axis —
    # on a 3D mesh (dp, tp, ep), 'tp' does not split activations even though
    # M may happen to be divisible by mesh.shape['tp'].
    dp_axes = tuple(name for name in mesh.axis_names
                    if name not in (axis_name, 'tp')
                    and mesh.shape[name] > 1
                    and M % mesh.shape[name] == 0)
    if len(dp_axes) == 0:
        act_pspec = P(None, None)
    elif len(dp_axes) == 1:
        act_pspec = P(dp_axes[0], None)
    else:
        act_pspec = P(dp_axes, None)

    # Stacked weight path: 4D (N_L, E, K, N) sharded on expert dim.
    # Inside shard_map, reshape to 3D (N_L*E_local, K, N) and use
    # rhs_group_offset = layer_idx * E_local.
    is_stacked = layer_idx is not None

    w3d = P(axis_name, None, None)
    w4d = P(None, axis_name, None, None)
    w_spec = w4d if is_stacked else w3d

    if _is_fp8_weight(gate_proj):
        if moe_backend == 'gmm':
            gate_fp8, gate_s = _fp8_gmm_components(gate_proj)
            up_fp8, up_s = _fp8_gmm_components(up_proj)
            down_fp8, down_s = _fp8_gmm_components(down_proj)
            # gmm rhs_scale: (E, num_blocks, 1, N) or stacked (N_L, E, num_blocks, 1, N)
            w_scale_spec = P(None, axis_name, None, None, None) if is_stacked \
                else P(axis_name, None, None, None)
        else:
            gate_fp8, gate_s = _fp8_expert_components(gate_proj)
            up_fp8, up_s = _fp8_expert_components(up_proj)
            down_fp8, down_s = _fp8_expert_components(down_proj)
            # ragged_dot scale is 2D: (E, N) or stacked (N_L, E, N)
            w_scale_spec = P(None, axis_name, None) if is_stacked \
                else P(axis_name, None)

        @partial(shard_map, mesh=mesh,
                 in_specs=(act_pspec, act_pspec, act_pspec,
                           w_spec, w_scale_spec, w_spec, w_scale_spec, w_spec, w_scale_spec),
                 out_specs=act_pspec,
                 check_rep=False)
        def _expert_fn(x, indices, weights,
                        lg, lgs, lu, lus, ld, lds):
            if is_stacked:
                # Reshape 4D→3D: (N_L, E_local, ...) → (N_L*E_local, ...)
                e_local = lg.shape[1]
                lg = lg.reshape(-1, *lg.shape[2:])
                lu = lu.reshape(-1, *lu.shape[2:])
                ld = ld.reshape(-1, *ld.shape[2:])
                lgs = lgs.reshape(-1, *lgs.shape[2:])
                lus = lus.reshape(-1, *lus.shape[2:])
                lds = lds.reshape(-1, *lds.shape[2:])
                rhs_offset = jnp.array([layer_idx * e_local], dtype=jnp.int32)
            else:
                e_local = None
                rhs_offset = None
            return _ep_inner_body(x, indices, weights, k, axis_name,
                                  lg, lu, ld, lgs, lus, lds,
                                  moe_backend=moe_backend,
                                  rhs_group_offset=rhs_offset,
                                  n_local_experts=e_local)

        return _expert_fn(x, expert_indices, expert_weights,
                          gate_fp8, gate_s, up_fp8, up_s, down_fp8, down_s)

    gate_w = _get_expert_weight(gate_proj, x.dtype)
    up_w = _get_expert_weight(up_proj, x.dtype)
    down_w = _get_expert_weight(down_proj, x.dtype)

    @partial(shard_map, mesh=mesh,
             in_specs=(act_pspec, act_pspec, act_pspec,
                       w_spec, w_spec, w_spec),
             out_specs=act_pspec,
             check_rep=False)
    def _expert_fn(x, indices, weights, local_gate, local_up, local_down):
        if is_stacked:
            e_local = local_gate.shape[1]
            local_gate = local_gate.reshape(-1, *local_gate.shape[2:])
            local_up = local_up.reshape(-1, *local_up.shape[2:])
            local_down = local_down.reshape(-1, *local_down.shape[2:])
            rhs_offset = jnp.array([layer_idx * e_local], dtype=jnp.int32)
        else:
            e_local = None
            rhs_offset = None
        return _ep_inner_body(x, indices, weights, k, axis_name,
                              local_gate, local_up, local_down,
                              moe_backend=moe_backend,
                              rhs_group_offset=rhs_offset,
                              n_local_experts=e_local)

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
    moe_backend: MoeBackend = 'ragged_dot',
    stacked_expert_weights: dict | None = None,
    layer_idx: int | None = None,
) -> jax.Array:
    """Full MoE layer: route + routed experts + shared expert.

    When n_devices > 1 and mesh is provided, uses shard_map EP.
    Expert weights are never all-gathered — each device only computes on
    its local expert shard.

    Args:
        moe_backend: 'ragged_dot' (XLA) or 'gmm' (Pallas kernel).
        stacked_expert_weights: When provided, a dict with 'gate_proj',
            'up_proj', 'down_proj' containing stacked expert weights
            (N_L, E, K, N) for EP or (N_L*E, K, N) for single device.
            These are used INSTEAD of params['gate_proj'] etc, and gmm_v2's
            rhs_group_offset indexes the right layer without JAX-level slicing.
        layer_idx: Index into the stacked weights for the current layer.
            Required when stacked_expert_weights is set.
    """
    B, T, D = x.shape
    M = B * T
    x_flat = x.reshape(M, D)

    with jax.named_scope('moe_routing'):
        expert_indices, expert_weights = moe_routing(
            x_flat, params['gate_weight'], n_experts_per_token,
        )

    # Select expert weight source: stacked (no squeeze) or per-layer (standard).
    use_stacked = stacked_expert_weights is not None and moe_backend == 'gmm'
    if use_stacked:
        gate_proj = stacked_expert_weights['gate_proj']
        up_proj = stacked_expert_weights['up_proj']
        down_proj = stacked_expert_weights['down_proj']
        # Derive n_experts from gate_weight (routing gate), not expert weights.
        n_experts = _get_n_experts(params['gate_weight'])
    else:
        gate_proj = params['gate_proj']
        up_proj = params['up_proj']
        down_proj = params['down_proj']
        layer_idx = None
        n_experts = None

    # When mesh has a separate 'ep' axis (3D mesh), use it for expert routing
    # instead of axis_name (which is the TP axis). Expert weights are sharded
    # on 'ep' (per AXIS_RULES_EP), so the shard_map must use 'ep' as its axis.
    if mesh is not None and 'ep' in mesh.axis_names:
        ep_axis = 'ep'
        ep_n = mesh.shape['ep']
    else:
        ep_axis = axis_name
        ep_n = n_devices

    with jax.named_scope('moe_experts'):
        if ep_n > 1 and mesh is not None:
            routed_out = expert_forward_ep(
                x_flat, expert_indices, expert_weights,
                gate_proj, up_proj, down_proj,
                mesh=mesh, axis_name=ep_axis,
                moe_backend=moe_backend,
                layer_idx=layer_idx,
            )
        else:
            routed_out = expert_forward_single(
                x_flat, expert_indices, expert_weights,
                gate_proj, up_proj, down_proj,
                moe_backend=moe_backend,
                layer_idx=layer_idx,
                n_experts_override=n_experts,
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
