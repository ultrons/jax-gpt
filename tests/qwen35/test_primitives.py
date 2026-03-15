"""Tests for RMSNorm, RoPE, and SwiGLU primitives."""

import jax
import jax.numpy as jnp
import pytest

from jax_gpt.models.qwen35.primitives import (
    apply_rotary_emb,
    precompute_rope_freqs,
    rms_norm,
    swiglu,
)


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------

def test_rms_norm_shape():
    x = jax.random.normal(jax.random.key(0), (2, 8, 64))
    w = jnp.ones(64)
    out = rms_norm(x, w)
    assert out.shape == x.shape


def test_rms_norm_unit_scale():
    """With weight=0 (meaning 1+0=1), RMSNorm should produce vectors with RMS ≈ 1."""
    x = jax.random.normal(jax.random.key(1), (4, 32))
    w = jnp.zeros(32)  # (1+0) = 1
    out = rms_norm(x, w)
    rms = jnp.sqrt(jnp.mean(out ** 2, axis=-1))
    assert jnp.allclose(rms, 1.0, atol=1e-5)


def test_rms_norm_learnable_scale():
    """Weight=1 means scale of 2 (1+1), so output should be 2x the w=0 case."""
    x = jax.random.normal(jax.random.key(2), (4, 16))
    w0 = jnp.zeros(16)  # scale = 1+0 = 1
    w1 = jnp.ones(16)   # scale = 1+1 = 2
    out0 = rms_norm(x, w0)
    out1 = rms_norm(x, w1)
    assert jnp.allclose(out1, out0 * 2.0, atol=1e-5)


# ---------------------------------------------------------------------------
# RoPE
# ---------------------------------------------------------------------------

def test_rope_freqs_shape():
    freqs = precompute_rope_freqs(dim=64, max_seq_len=128)
    assert freqs.shape == (128, 32, 2)  # (seq, dim//2, cos/sin)


def test_rope_preserves_norm():
    """RoPE is a rotation — it should preserve vector norms."""
    key = jax.random.key(3)
    x = jax.random.normal(key, (2, 4, 16, 64))  # (B, heads, T, head_dim)
    freqs = precompute_rope_freqs(dim=64, max_seq_len=16)
    out = apply_rotary_emb(x, freqs, rope_dim=64)
    x_norms = jnp.linalg.norm(x, axis=-1)
    out_norms = jnp.linalg.norm(out, axis=-1)
    assert jnp.allclose(x_norms, out_norms, atol=1e-5)


def test_rope_partial_rotary():
    """With partial rotary, the un-rotated portion should be unchanged."""
    key = jax.random.key(4)
    head_dim = 256
    rope_dim = 64  # 25% of 256
    x = jax.random.normal(key, (2, 8, 32, head_dim))  # (B, heads, T, head_dim)
    freqs = precompute_rope_freqs(dim=rope_dim, max_seq_len=32)
    out = apply_rotary_emb(x, freqs, rope_dim=rope_dim)

    # The last (256-64)=192 dims should be identical
    assert jnp.allclose(x[..., rope_dim:], out[..., rope_dim:], atol=1e-7)
    # The first 64 dims should generally differ (unless x was zero)
    assert not jnp.allclose(x[..., :rope_dim], out[..., :rope_dim], atol=1e-3)


def test_rope_different_positions_differ():
    """Different positions should produce different rotations."""
    freqs = precompute_rope_freqs(dim=64, max_seq_len=128)
    assert not jnp.allclose(freqs[0], freqs[1])


# ---------------------------------------------------------------------------
# SwiGLU
# ---------------------------------------------------------------------------

def test_swiglu_shape():
    key = jax.random.key(5)
    k1, k2, k3 = jax.random.split(key, 3)
    D, I = 64, 128
    x = jax.random.normal(k1, (2, 8, D))
    gate_w = jax.random.normal(k1, (D, I)) * 0.02
    up_w = jax.random.normal(k2, (D, I)) * 0.02
    down_w = jax.random.normal(k3, (I, D)) * 0.02
    out = swiglu(x, gate_w, up_w, down_w)
    assert out.shape == x.shape


def test_swiglu_nonzero():
    """SwiGLU with random weights should produce non-zero output."""
    key = jax.random.key(6)
    k1, k2, k3, k4 = jax.random.split(key, 4)
    D, I = 32, 64
    x = jax.random.normal(k1, (4, D))
    gate_w = jax.random.normal(k2, (D, I)) * 0.1
    up_w = jax.random.normal(k3, (D, I)) * 0.1
    down_w = jax.random.normal(k4, (I, D)) * 0.1
    out = swiglu(x, gate_w, up_w, down_w)
    assert jnp.any(out != 0)


def test_swiglu_no_nan():
    """SwiGLU should not produce NaN even with larger inputs."""
    key = jax.random.key(7)
    k1, k2, k3, k4 = jax.random.split(key, 4)
    D, I = 32, 64
    x = jax.random.normal(k1, (4, D)) * 10.0
    gate_w = jax.random.normal(k2, (D, I)) * 0.02
    up_w = jax.random.normal(k3, (D, I)) * 0.02
    down_w = jax.random.normal(k4, (I, D)) * 0.02
    out = swiglu(x, gate_w, up_w, down_w)
    assert not jnp.any(jnp.isnan(out))
