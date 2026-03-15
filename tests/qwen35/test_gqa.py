"""Tests for Grouped Query Attention."""

import jax
import jax.numpy as jnp
import pytest

from jax_gpt.models.qwen35.config import Qwen35Config
from jax_gpt.models.qwen35.gqa import gqa_attention
from jax_gpt.models.qwen35.primitives import precompute_rope_freqs


def _make_gqa_params(cfg: Qwen35Config, key: jax.Array) -> dict:
    """Create random GQA params for testing."""
    keys = jax.random.split(key, 5)
    D = cfg.d_model
    q_dim = cfg.gqa_n_q_heads * cfg.gqa_head_dim
    kv_dim = cfg.gqa_n_kv_heads * cfg.gqa_head_dim

    return {
        'q_proj': jax.random.normal(keys[0], (D, q_dim * 2)) * 0.02,  # query + gate
        'k_proj': jax.random.normal(keys[1], (D, kv_dim)) * 0.02,
        'v_proj': jax.random.normal(keys[2], (D, kv_dim)) * 0.02,
        'o_proj': jax.random.normal(keys[3], (q_dim, D)) * 0.02,
        'q_norm': jnp.zeros(cfg.gqa_head_dim),
        'k_norm': jnp.zeros(cfg.gqa_head_dim),
    }


def test_gqa_shape_no_cache():
    """GQA without cache should produce correct output shape."""
    cfg = Qwen35Config.mini()
    params = _make_gqa_params(cfg, jax.random.key(0))
    B, T = 2, 16
    freqs = precompute_rope_freqs(cfg.gqa_rope_dim, T, cfg.gqa_rope_theta)

    x = jax.random.normal(jax.random.key(1), (B, T, cfg.d_model))
    out, ck, cv = gqa_attention(
        x, params,
        cfg.gqa_n_q_heads, cfg.gqa_n_kv_heads, cfg.gqa_head_dim,
        freqs, cfg.gqa_rope_dim,
    )
    assert out.shape == (B, T, cfg.d_model)
    assert ck is None
    assert cv is None


def test_gqa_shape_with_cache():
    """GQA with cache should produce correct output and updated cache."""
    cfg = Qwen35Config.mini()
    params = _make_gqa_params(cfg, jax.random.key(2))
    B, T = 2, 1
    max_len = 64
    freqs = precompute_rope_freqs(cfg.gqa_rope_dim, max_len, cfg.gqa_rope_theta)

    x = jax.random.normal(jax.random.key(3), (B, T, cfg.d_model))
    cache_k = jnp.zeros((B, cfg.gqa_n_kv_heads, max_len, cfg.gqa_head_dim))
    cache_v = jnp.zeros((B, cfg.gqa_n_kv_heads, max_len, cfg.gqa_head_dim))
    pos = jnp.array(0, dtype=jnp.int32)

    out, new_ck, new_cv = gqa_attention(
        x, params,
        cfg.gqa_n_q_heads, cfg.gqa_n_kv_heads, cfg.gqa_head_dim,
        freqs, cfg.gqa_rope_dim,
        cache_k=cache_k, cache_v=cache_v, cache_pos=pos,
    )
    assert out.shape == (B, T, cfg.d_model)
    assert new_ck.shape == cache_k.shape
    assert new_cv.shape == cache_v.shape


def test_gqa_cache_incremental():
    """Incremental decode with cache should produce same result as full pass."""
    cfg = Qwen35Config.mini()
    params = _make_gqa_params(cfg, jax.random.key(4))
    B, T = 1, 8
    max_len = 32
    freqs = precompute_rope_freqs(cfg.gqa_rope_dim, max_len, cfg.gqa_rope_theta)

    x = jax.random.normal(jax.random.key(5), (B, T, cfg.d_model))

    # Full forward pass (no cache)
    full_out, _, _ = gqa_attention(
        x, params,
        cfg.gqa_n_q_heads, cfg.gqa_n_kv_heads, cfg.gqa_head_dim,
        freqs, cfg.gqa_rope_dim,
    )

    # Incremental with cache: prefill all T tokens at once
    cache_k = jnp.zeros((B, cfg.gqa_n_kv_heads, max_len, cfg.gqa_head_dim))
    cache_v = jnp.zeros((B, cfg.gqa_n_kv_heads, max_len, cfg.gqa_head_dim))
    pos = jnp.array(0, dtype=jnp.int32)

    cached_out, _, _ = gqa_attention(
        x, params,
        cfg.gqa_n_q_heads, cfg.gqa_n_kv_heads, cfg.gqa_head_dim,
        freqs, cfg.gqa_rope_dim,
        cache_k=cache_k, cache_v=cache_v, cache_pos=pos,
    )

    # Should match (same input, just with cache tracking)
    assert jnp.allclose(full_out, cached_out, atol=1e-5), \
        f"Max diff: {jnp.max(jnp.abs(full_out - cached_out))}"


def test_gqa_no_nan():
    """GQA should not produce NaN."""
    cfg = Qwen35Config.mini()
    params = _make_gqa_params(cfg, jax.random.key(6))
    B, T = 2, 16
    freqs = precompute_rope_freqs(cfg.gqa_rope_dim, T, cfg.gqa_rope_theta)

    x = jax.random.normal(jax.random.key(7), (B, T, cfg.d_model))
    out, _, _ = gqa_attention(
        x, params,
        cfg.gqa_n_q_heads, cfg.gqa_n_kv_heads, cfg.gqa_head_dim,
        freqs, cfg.gqa_rope_dim,
    )
    assert not jnp.any(jnp.isnan(out))


def test_gqa_causal():
    """GQA should be causal — changing future tokens shouldn't affect past outputs."""
    cfg = Qwen35Config.mini()
    params = _make_gqa_params(cfg, jax.random.key(8))
    B, T = 1, 8
    freqs = precompute_rope_freqs(cfg.gqa_rope_dim, T, cfg.gqa_rope_theta)

    x1 = jax.random.normal(jax.random.key(9), (B, T, cfg.d_model))
    x2 = x1.at[:, T//2:].set(jax.random.normal(jax.random.key(10), (B, T - T//2, cfg.d_model)))

    out1, _, _ = gqa_attention(
        x1, params,
        cfg.gqa_n_q_heads, cfg.gqa_n_kv_heads, cfg.gqa_head_dim,
        freqs, cfg.gqa_rope_dim,
    )
    out2, _, _ = gqa_attention(
        x2, params,
        cfg.gqa_n_q_heads, cfg.gqa_n_kv_heads, cfg.gqa_head_dim,
        freqs, cfg.gqa_rope_dim,
    )

    # First half of outputs should be identical
    assert jnp.allclose(out1[:, :T//2], out2[:, :T//2], atol=1e-5), \
        f"Causality violated: max diff = {jnp.max(jnp.abs(out1[:, :T//2] - out2[:, :T//2]))}"
