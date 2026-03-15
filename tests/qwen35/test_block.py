"""Tests for transformer block (4-layer group)."""

import jax
import jax.numpy as jnp
import pytest

from jax_gpt.models.qwen35.config import Qwen35Config
from jax_gpt.models.qwen35.model import init_params
from jax_gpt.models.qwen35.block import group_forward
from jax_gpt.models.qwen35.primitives import precompute_rope_freqs


def test_group_forward_prefill_shape():
    """Group forward (prefill) should produce correct shapes."""
    cfg = Qwen35Config.mini()
    params = init_params(cfg, jax.random.key(0))
    B, T = 1, 8

    # Extract one group's params (first group)
    group_params = jax.tree.map(lambda x: x[0], params['groups'])

    key_dim = cfg.delta_n_qk_heads * cfg.delta_qk_head_dim
    value_dim = cfg.delta_n_v_heads * cfg.delta_v_head_dim
    conv_dim = key_dim * 2 + value_dim
    n_delta = cfg.full_attention_interval - 1

    delta_Ms = jnp.zeros((n_delta, B, cfg.delta_n_v_heads, cfg.delta_qk_head_dim, cfg.delta_v_head_dim))
    delta_convs = jnp.zeros((n_delta, B, conv_dim, cfg.delta_conv_kernel))
    gqa_k = jnp.zeros((B, cfg.gqa_n_kv_heads, T, cfg.gqa_head_dim))
    gqa_v = jnp.zeros((B, cfg.gqa_n_kv_heads, T, cfg.gqa_head_dim))

    rope_freqs = precompute_rope_freqs(cfg.gqa_rope_dim, T, cfg.gqa_rope_theta)
    x = jax.random.normal(jax.random.key(1), (B, T, cfg.d_model))

    x_out, new_Ms, new_convs, new_gk, new_gv = group_forward(
        x, group_params, delta_Ms, delta_convs,
        gqa_k, gqa_v, None, cfg, rope_freqs, is_decode=False,
    )

    assert x_out.shape == (B, T, cfg.d_model)
    assert new_Ms.shape == delta_Ms.shape
    assert new_convs.shape == delta_convs.shape


def test_group_forward_no_nan():
    """Group forward should not produce NaN."""
    cfg = Qwen35Config.mini()
    params = init_params(cfg, jax.random.key(2))
    B, T = 1, 4

    group_params = jax.tree.map(lambda x: x[0], params['groups'])

    key_dim = cfg.delta_n_qk_heads * cfg.delta_qk_head_dim
    value_dim = cfg.delta_n_v_heads * cfg.delta_v_head_dim
    conv_dim = key_dim * 2 + value_dim
    n_delta = cfg.full_attention_interval - 1

    delta_Ms = jnp.zeros((n_delta, B, cfg.delta_n_v_heads, cfg.delta_qk_head_dim, cfg.delta_v_head_dim))
    delta_convs = jnp.zeros((n_delta, B, conv_dim, cfg.delta_conv_kernel))
    gqa_k = jnp.zeros((B, cfg.gqa_n_kv_heads, T, cfg.gqa_head_dim))
    gqa_v = jnp.zeros((B, cfg.gqa_n_kv_heads, T, cfg.gqa_head_dim))

    rope_freqs = precompute_rope_freqs(cfg.gqa_rope_dim, T, cfg.gqa_rope_theta)
    x = jax.random.normal(jax.random.key(3), (B, T, cfg.d_model))

    x_out, _, _, _, _ = group_forward(
        x, group_params, delta_Ms, delta_convs,
        gqa_k, gqa_v, None, cfg, rope_freqs, is_decode=False,
    )

    assert not jnp.any(jnp.isnan(x_out))
