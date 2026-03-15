"""Tests for FP8 matmul utilities."""

import jax
import jax.numpy as jnp
import pytest

from jax_gpt.models.qwen35.fp8 import (
    FP8_DTYPE,
    dynamic_quantize_fp8,
    fp8_linear,
    fp8_matmul,
    fp8_ragged_dot,
)


def test_dynamic_quantize_shape():
    x = jax.random.normal(jax.random.key(0), (4, 64))
    x_fp8, scale = dynamic_quantize_fp8(x)
    assert x_fp8.shape == x.shape
    assert x_fp8.dtype == FP8_DTYPE
    assert scale.shape == (4, 1)


def test_dynamic_quantize_roundtrip():
    """Quantize then dequantize should be close to original."""
    x = jax.random.normal(jax.random.key(1), (8, 32))
    x_fp8, scale = dynamic_quantize_fp8(x)
    x_restored = x_fp8.astype(jnp.float32) * scale
    # FP8 e4m3fn has ~3 bits of mantissa, so expect ~10% relative error
    rel_error = jnp.max(jnp.abs(x - x_restored)) / jnp.max(jnp.abs(x))
    assert rel_error < 0.15, f"Round-trip relative error too high: {rel_error}"


def test_fp8_matmul_shape():
    """fp8_matmul should produce correct output shape."""
    x = jax.random.normal(jax.random.key(2), (2, 8, 64))
    # Weight in HF convention: (out, in) as fp8
    w = jax.random.normal(jax.random.key(3), (128, 64)).astype(FP8_DTYPE)
    scale_inv = jnp.ones((128, 1))
    out = fp8_matmul(x, w, scale_inv)
    assert out.shape == (2, 8, 128)
    assert out.dtype == jnp.float32


def test_fp8_matmul_accuracy():
    """fp8_matmul should approximate the float32 matmul."""
    key = jax.random.key(4)
    k1, k2 = jax.random.split(key)
    x = jax.random.normal(k1, (4, 32)) * 0.1
    w_f32 = jax.random.normal(k2, (64, 32)) * 0.1

    # Reference: float32 matmul
    ref = x @ w_f32.T

    # FP8 matmul with scale_inv=1 (identity scale)
    w_fp8 = w_f32.astype(FP8_DTYPE)
    scale_inv = jnp.ones((64, 1))
    out = fp8_matmul(x, w_fp8, scale_inv)

    # Should be close — error from fp8 quantization of both x and w
    max_diff = jnp.max(jnp.abs(ref - out))
    assert max_diff < 0.5, f"FP8 matmul too inaccurate: max_diff={max_diff}"


def test_fp8_matmul_with_scale():
    """fp8_matmul with non-trivial weight scale should rescale correctly."""
    x = jnp.ones((2, 4), dtype=jnp.float32)
    w_fp8 = jnp.ones((3, 4), dtype=FP8_DTYPE)
    # scale_inv = 2 means w_real = w_fp8 * (1/2) = 0.5
    scale_inv = jnp.full((3, 1), 2.0)

    out = fp8_matmul(x, w_fp8, scale_inv)
    # x @ (w * 0.5)^T = [1,1,1,1] @ [0.5,0.5,0.5,0.5]^T = 2.0
    # But activation is also fp8-quantized, so there's some error
    assert jnp.allclose(out, 2.0, atol=0.1), f"Scaled result: {out}"


def test_fp8_linear_fp8_path():
    """fp8_linear should use fp8_matmul when weight is fp8."""
    x = jax.random.normal(jax.random.key(5), (2, 4, 32))
    w_fp8 = jax.random.normal(jax.random.key(6), (64, 32)).astype(FP8_DTYPE)
    params = {
        'weight': w_fp8,
        'weight_scale_inv': jnp.ones((64, 1)),
    }
    out = fp8_linear(x, params)
    assert out.shape == (2, 4, 64)
    assert out.dtype == jnp.float32


def test_fp8_linear_float_fallback():
    """fp8_linear should fall back to standard matmul for float weights."""
    x = jax.random.normal(jax.random.key(7), (2, 4, 32))
    w_f32 = jax.random.normal(jax.random.key(8), (32, 64)) * 0.02
    params = {'weight': w_f32}
    out = fp8_linear(x, params)
    assert out.shape == (2, 4, 64)
    expected = x @ w_f32
    assert jnp.allclose(out, expected)


def test_fp8_ragged_dot_shape():
    """fp8_ragged_dot should produce correct output shape."""
    M, K, N, E = 8, 32, 16, 2
    x = jax.random.normal(jax.random.key(9), (M, K))
    w_fp8 = jax.random.normal(jax.random.key(10), (E, K, N)).astype(FP8_DTYPE)
    group_sizes = jnp.array([4, 4])
    out = fp8_ragged_dot(x, w_fp8, group_sizes, w_scale_inv=None)
    assert out.shape == (M, N)
    assert out.dtype == jnp.float32


def test_fp8_no_nan():
    """All fp8 operations should produce no NaN."""
    x = jax.random.normal(jax.random.key(11), (4, 64))
    w_fp8 = jax.random.normal(jax.random.key(12), (32, 64)).astype(FP8_DTYPE)
    scale_inv = jnp.ones((32, 1))

    out = fp8_matmul(x, w_fp8, scale_inv)
    assert not jnp.any(jnp.isnan(out))

    x_fp8, scale = dynamic_quantize_fp8(x)
    assert not jnp.any(jnp.isnan(x_fp8.astype(jnp.float32)))
    assert not jnp.any(jnp.isnan(scale))
