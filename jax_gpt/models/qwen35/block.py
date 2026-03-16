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
from jax_gpt.models.qwen35.moe import moe_layer
from jax_gpt.models.qwen35.primitives import rms_norm


def deltanet_layer_forward(
    x: jax.Array,
    params: dict,
    delta_M: jax.Array,
    delta_conv: jax.Array,
    config: Qwen35Config,
    is_decode: bool,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Single DeltaNet layer: pre-norm -> attention -> residual -> pre-norm -> MoE -> residual."""

    with jax.named_scope('deltanet_attn'):
        normed = rms_norm(x, params['attn_norm'], config.rms_norm_eps)
        if is_decode:
            attn_out, new_M, new_conv = deltanet_recurrent_step(
                normed, params['attn'], delta_M, delta_conv,
                config.delta_n_qk_heads, config.delta_n_v_heads,
                config.delta_qk_head_dim, config.delta_v_head_dim,
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
        moe_out = moe_layer(normed, params['moe'], config.n_experts_per_token)

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
        moe_out = moe_layer(normed, params['moe'], config.n_experts_per_token)

    x = x + moe_out
    # Ensure cache dtypes match input for scan carry compatibility
    if new_k is not None:
        new_k = new_k.astype(gqa_k.dtype)
        new_v = new_v.astype(gqa_v.dtype)
    return x, new_k, new_v


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
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Forward pass through one 4-layer group (3 DeltaNet + 1 GQA)."""

    # Scan over DeltaNet layers
    def _delta_step(carry, layer_inputs):
        x_carry = carry
        layer_params, M_i, conv_i = layer_inputs
        x_carry, new_M, new_conv = deltanet_layer_forward(
            x_carry, layer_params, M_i, conv_i, config, is_decode,
        )
        return x_carry, (new_M, new_conv)

    delta_layer_params = group_params['delta_layers']
    x, (new_Ms, new_convs) = jax.lax.scan(
        _delta_step, x, (delta_layer_params, delta_Ms, delta_convs),
    )

    x, new_gqa_k, new_gqa_v = gqa_layer_forward(
            x, group_params['gqa_layer'],
            gqa_k, gqa_v, cache_pos,
            config, rope_freqs,
        )

    return x, new_Ms, new_convs, new_gqa_k, new_gqa_v
