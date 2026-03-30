"""Transformer block: 4-layer group (3 DeltaNet + 1 GQA), each with MoE.

Pure functions operating on param dicts and pytree caches.
All functions annotated with jax.named_scope for TPU profile visibility.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from jax_gpt.models.qwen35.config import Qwen35Config
from jax_gpt.models.qwen35.deltanet import deltanet_prefill, deltanet_recurrent_step
from jax_gpt.models.qwen35.gqa import gqa_attention
from jax_gpt.models.qwen35.moe import moe_layer, MoeBackend
from jax_gpt.models.qwen35.primitives import rms_norm

try:
    from jax_gpt.models.qwen35.gqa_rpa import gqa_attention_rpa
    HAS_RPA = True
except ImportError:
    HAS_RPA = False


def deltanet_layer_forward(
    x: jax.Array,
    params: dict,
    delta_M: jax.Array,
    delta_conv: jax.Array,
    config: Qwen35Config,
    is_decode: bool,
    n_devices: int = 1, mesh=None,
    axis_name: str = 'tp',
    moe_backend: MoeBackend = 'ragged_dot',
    stacked_expert_weights: dict | None = None,
    layer_idx: int | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Single DeltaNet layer: pre-norm -> attention -> residual -> pre-norm -> MoE -> residual."""

    with jax.named_scope('deltanet_attn'):
        normed = rms_norm(x, params['attn_norm'], config.rms_norm_eps)
        if is_decode:
            attn_out, new_M, new_conv = deltanet_recurrent_step(
                normed, params['attn'], delta_M, delta_conv,
                config.delta_n_qk_heads, config.delta_n_v_heads,
                config.delta_qk_head_dim, config.delta_v_head_dim,
                mesh=mesh,
            )
        else:
            attn_out, new_M, new_conv = deltanet_prefill(
                normed, params['attn'],
                config.delta_n_qk_heads, config.delta_n_v_heads,
                config.delta_qk_head_dim, config.delta_v_head_dim,
                config.delta_conv_kernel,
                chunk_size=config.delta_chunk_size,
            )
        # Ensure state dtypes match input cache for scan carry compatibility
        new_M = new_M.astype(delta_M.dtype)
        new_conv = new_conv.astype(delta_conv.dtype)

    x = x + attn_out

    with jax.named_scope('deltanet_moe'):
        normed = rms_norm(x, params['moe_norm'], config.rms_norm_eps)
        moe_out = moe_layer(normed, params['moe'], config.n_experts_per_token,
                            n_devices=n_devices, axis_name=axis_name, mesh=mesh,
                            moe_backend=moe_backend,
                            stacked_expert_weights=stacked_expert_weights,
                            layer_idx=layer_idx)

    x = x + moe_out
    return x, new_M, new_conv


def gqa_layer_forward(
    x: jax.Array,
    params: dict,
    gqa_k: jax.Array,
    gqa_v: jax.Array,
    cache_pos: jax.Array | None,
    config: Qwen35Config,
    rope_freqs: jax.Array,
    n_devices: int = 1, mesh=None,
    axis_name: str = 'tp',
    moe_backend: MoeBackend = 'ragged_dot',
    stacked_expert_weights: dict | None = None,
    layer_idx: int | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Single GQA layer: pre-norm -> attention -> residual -> pre-norm -> MoE -> residual."""

    with jax.named_scope('gqa_attn'):
        normed = rms_norm(x, params['attn_norm'], config.rms_norm_eps)
        attn_out, new_k, new_v = gqa_attention(
            normed, params['attn'],
            config.gqa_n_q_heads, config.gqa_n_kv_heads, config.gqa_head_dim,
            rope_freqs, config.gqa_rope_dim,
            cache_k=gqa_k, cache_v=gqa_v, cache_pos=cache_pos,
        )

    x = x + attn_out

    with jax.named_scope('gqa_moe'):
        normed = rms_norm(x, params['moe_norm'], config.rms_norm_eps)
        moe_out = moe_layer(normed, params['moe'], config.n_experts_per_token,
                            n_devices=n_devices, axis_name=axis_name, mesh=mesh,
                            moe_backend=moe_backend,
                            stacked_expert_weights=stacked_expert_weights,
                            layer_idx=layer_idx)

    x = x + moe_out
    # Ensure cache dtypes match input for scan carry compatibility
    if new_k is not None:
        new_k = new_k.astype(gqa_k.dtype)
        new_v = new_v.astype(gqa_v.dtype)
    return x, new_k, new_v


def gqa_layer_forward_rpa(
    x: jax.Array,
    params: dict,
    kv_cache: jax.Array,
    kv_lens: jax.Array,
    page_indices: jax.Array,
    cu_q_lens: jax.Array,
    distribution: jax.Array,
    cache_pos: jax.Array,
    config: Qwen35Config,
    rope_freqs: jax.Array,
    n_devices: int = 1, mesh=None,
    axis_name: str = 'tp',
    moe_backend: MoeBackend = 'ragged_dot',
    stacked_expert_weights: dict | None = None,
    layer_idx: int | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Single GQA layer using RPA v3 kernel for decode."""

    with jax.named_scope('gqa_attn_rpa'):
        normed = rms_norm(x, params['attn_norm'], config.rms_norm_eps)
        attn_out, updated_cache = gqa_attention_rpa(
            normed, params['attn'],
            config.gqa_n_q_heads, config.gqa_n_kv_heads, config.gqa_head_dim,
            rope_freqs, config.gqa_rope_dim,
            kv_cache=kv_cache,
            kv_lens=kv_lens,
            page_indices=page_indices,
            cu_q_lens=cu_q_lens,
            distribution=distribution,
            cache_pos=cache_pos,
            mesh=mesh,
            axis_name=axis_name,
        )

    x = x + attn_out

    with jax.named_scope('gqa_moe'):
        normed = rms_norm(x, params['moe_norm'], config.rms_norm_eps)
        moe_out = moe_layer(normed, params['moe'], config.n_experts_per_token,
                            n_devices=n_devices, axis_name=axis_name, mesh=mesh,
                            moe_backend=moe_backend,
                            stacked_expert_weights=stacked_expert_weights,
                            layer_idx=layer_idx)

    x = x + moe_out
    return x, updated_cache


def group_forward(
    x: jax.Array,
    group_params: dict,
    delta_Ms: jax.Array,
    delta_convs: jax.Array,
    gqa_k: jax.Array,
    gqa_v: jax.Array,
    cache_pos: jax.Array | None,
    config: Qwen35Config,
    rope_freqs: jax.Array,
    is_decode: bool,
    n_devices: int = 1, mesh=None,
    axis_name: str = 'tp',
    moe_backend: MoeBackend = 'ragged_dot',
    delta_moe_list: list[dict] | None = None,
    gqa_moe: dict | None = None,
    stacked_delta_expert_weights: dict | None = None,
    stacked_gqa_expert_weights: dict | None = None,
    group_idx: int | None = None,
    delta_layer_list: list[dict] | None = None,
    gqa_layer_params: dict | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Forward pass through one 4-layer group (3 DeltaNet + 1 GQA).

    Args:
        delta_moe_list: Optional list of n_delta MoE param dicts, one per
            DeltaNet layer. When provided, overrides group_params['delta_layers']['moe']
            to avoid XLA retiling from stacked array slicing.
        gqa_moe: Optional MoE param dict for the GQA layer. When provided,
            overrides group_params['gqa_layer']['moe'].
        stacked_delta_expert_weights: Optional dict with stacked expert weights
            (gate_proj, up_proj, down_proj) across all delta layers, shape
            (total_delta_layers, E, K, N). Used with gmm backend to eliminate
            squeeze entirely — gmm_v2's rhs_group_offset indexes directly.
        stacked_gqa_expert_weights: Same for GQA layers.
        group_idx: Group index, used to compute per-layer indices for stacked weights.
    """

    # Unrolled DeltaNet layers (3 iterations) — see group_forward_rpa comment.
    # delta_layer_list: pre-sliced per-layer dicts (built outside JIT with static Python
    # indexing). When provided, no dynamic_index_in_dim is needed → eliminates retiling.
    if delta_layer_list is None:
        delta_layer_params = group_params['delta_layers']
    n_delta = config.full_attention_interval - 1
    new_Ms_list, new_convs_list = [], []
    for i in range(n_delta):
        if delta_layer_list is not None:
            layer_p = delta_layer_list[i]  # Python tuple index — zero JAX ops, no retiling
        else:
            i_idx = jnp.int32(i)
            # Use dynamic_index_in_dim (not static a[i]) to prevent XLA from
            # fusing all 3 layer slices into one slice_bitcast_fusion.
            layer_p = jax.tree.map(
                lambda a: jax.lax.dynamic_index_in_dim(a, i_idx, axis=0, keepdims=False),
                delta_layer_params)
            if delta_moe_list is not None:
                layer_p = {**layer_p, 'moe': delta_moe_list[i]}

        stacked_ew = None
        l_idx = None
        if stacked_delta_expert_weights is not None and group_idx is not None:
            stacked_ew = stacked_delta_expert_weights
            l_idx = group_idx * n_delta + i

        x, new_M, new_conv = deltanet_layer_forward(
            x, layer_p,
            delta_Ms[i],
            delta_convs[i],
            config, is_decode,
            n_devices=n_devices, mesh=mesh, axis_name=axis_name,
            moe_backend=moe_backend,
            stacked_expert_weights=stacked_ew,
            layer_idx=l_idx,
        )
        new_Ms_list.append(new_M)
        new_convs_list.append(new_conv)
    new_Ms = tuple(new_Ms_list)
    new_convs = tuple(new_convs_list)

    if gqa_layer_params is not None:
        gqa_p = gqa_layer_params
    else:
        gqa_p = group_params['gqa_layer']
        if gqa_moe is not None:
            gqa_p = {**gqa_p, 'moe': gqa_moe}

    stacked_gqa_ew = None
    gqa_l_idx = None
    if stacked_gqa_expert_weights is not None and group_idx is not None:
        stacked_gqa_ew = stacked_gqa_expert_weights
        gqa_l_idx = group_idx

    x, new_gqa_k, new_gqa_v = gqa_layer_forward(
            x, gqa_p,
            gqa_k, gqa_v, cache_pos,
            config, rope_freqs,
            n_devices=n_devices, mesh=mesh, axis_name=axis_name,
            moe_backend=moe_backend,
            stacked_expert_weights=stacked_gqa_ew,
            layer_idx=gqa_l_idx,
        )

    return x, new_Ms, new_convs, new_gqa_k, new_gqa_v


def group_forward_rpa(
    x: jax.Array,
    group_params: dict,
    delta_Ms: jax.Array,
    delta_convs: jax.Array,
    kv_cache: jax.Array,
    kv_lens: jax.Array,
    page_indices: jax.Array,
    cu_q_lens: jax.Array,
    distribution: jax.Array,
    cache_pos: jax.Array,
    config: Qwen35Config,
    rope_freqs: jax.Array,
    n_devices: int = 1, mesh=None,
    axis_name: str = 'tp',
    moe_backend: MoeBackend = 'ragged_dot',
    delta_moe_list: list[dict] | None = None,
    gqa_moe: dict | None = None,
    stacked_delta_expert_weights: dict | None = None,
    stacked_gqa_expert_weights: dict | None = None,
    group_idx: int | None = None,
    delta_layer_list: list[dict] | None = None,
    gqa_layer_params: dict | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Forward pass through one group using RPA for GQA decode.

    Args:
        delta_moe_list: Optional list of n_delta MoE param dicts, one per
            DeltaNet layer. When provided, overrides group_params['delta_layers']['moe']
            to avoid XLA retiling from stacked array slicing.
        gqa_moe: Optional MoE param dict for the GQA layer. When provided,
            overrides group_params['gqa_layer']['moe'].
        stacked_delta_expert_weights: Optional dict with stacked expert weights
            across all delta layers. See group_forward docstring.
        stacked_gqa_expert_weights: Same for GQA layers.
        group_idx: Group index for computing per-layer indices.

    Returns (x, new_Ms, new_convs, updated_kv_cache).
    """

    # Unrolled DeltaNet layers (3 iterations).
    # Python loop avoids jax.lax.scan tiling constraints on expert weights:
    # scan forces XLA to tile stacked [3,E,K,N] as T(8,128), but ragged_dot
    # needs T(32,128), causing a ~386ms copy per step. Unrolling lets XLA
    # tile each layer's [E,K,N] weights independently.
    #
    # delta_layer_list: pre-sliced per-layer dicts (built outside JIT with static Python
    # indexing). When provided, no dynamic_index_in_dim is needed → eliminates retiling.
    if delta_layer_list is None:
        delta_layer_params = group_params['delta_layers']
    n_delta = config.full_attention_interval - 1
    new_Ms_list, new_convs_list = [], []
    for i in range(n_delta):
        if delta_layer_list is not None:
            layer_p = delta_layer_list[i]  # Python list index — zero JAX ops, no retiling
        else:
            i_idx = jnp.int32(i)
            # Use dynamic_index_in_dim (not static a[i]) to prevent XLA from
            # fusing all 3 layer slices into one slice_bitcast_fusion.
            layer_p = jax.tree.map(
                lambda a: jax.lax.dynamic_index_in_dim(a, i_idx, axis=0, keepdims=False),
                delta_layer_params)
            if delta_moe_list is not None:
                layer_p = {**layer_p, 'moe': delta_moe_list[i]}

        stacked_ew = None
        l_idx = None
        if stacked_delta_expert_weights is not None and group_idx is not None:
            stacked_ew = stacked_delta_expert_weights
            l_idx = group_idx * n_delta + i

        x, new_M, new_conv = deltanet_layer_forward(
            x, layer_p,
            delta_Ms[i],
            delta_convs[i],
            config, is_decode=True,
            n_devices=n_devices, mesh=mesh, axis_name=axis_name,
            moe_backend=moe_backend,
            stacked_expert_weights=stacked_ew,
            layer_idx=l_idx,
        )
        new_Ms_list.append(new_M)
        new_convs_list.append(new_conv)
    new_Ms = tuple(new_Ms_list)
    new_convs = tuple(new_convs_list)

    if gqa_layer_params is not None:
        gqa_p = gqa_layer_params
    else:
        gqa_p = group_params['gqa_layer']
        if gqa_moe is not None:
            gqa_p = {**gqa_p, 'moe': gqa_moe}

    stacked_gqa_ew = None
    gqa_l_idx = None
    if stacked_gqa_expert_weights is not None and group_idx is not None:
        stacked_gqa_ew = stacked_gqa_expert_weights
        gqa_l_idx = group_idx

    x, updated_cache = gqa_layer_forward_rpa(
        x, gqa_p,
        kv_cache, kv_lens, page_indices,
        cu_q_lens, distribution, cache_pos,
        config, rope_freqs,
        n_devices=n_devices, mesh=mesh, axis_name=axis_name,
        moe_backend=moe_backend,
        stacked_expert_weights=stacked_gqa_ew,
        layer_idx=gqa_l_idx,
    )

    return x, new_Ms, new_convs, updated_cache
