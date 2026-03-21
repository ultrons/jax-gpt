"""Tests for Mixture of Experts layer."""

import jax
import jax.numpy as jnp
import pytest

from jax_gpt.models.qwen35.config import Qwen35Config
from jax_gpt.models.qwen35.moe import (
    expert_forward_single,
    moe_layer,
    moe_routing,
    shared_expert_forward,
)
from jax_gpt.models.qwen35.quantize import quantize_params_fp8


def _make_moe_params(cfg: Qwen35Config, key: jax.Array) -> dict:
    """Create random MoE params for testing."""
    keys = jax.random.split(key, 8)
    D = cfg.d_model
    E = cfg.n_routed_experts
    I = cfg.moe_intermediate_size
    SI = cfg.shared_expert_intermediate_size

    return {
        'gate_weight': jax.random.normal(keys[0], (D, E)) * 0.02,
        'gate_proj': jax.random.normal(keys[1], (E, D, I)) * 0.02,
        'up_proj': jax.random.normal(keys[2], (E, D, I)) * 0.02,
        'down_proj': jax.random.normal(keys[3], (E, I, D)) * 0.02,
        'shared_gate_proj': jax.random.normal(keys[4], (D, SI)) * 0.02,
        'shared_up_proj': jax.random.normal(keys[5], (D, SI)) * 0.02,
        'shared_down_proj': jax.random.normal(keys[6], (SI, D)) * 0.02,
        'shared_expert_gate_weight': jax.random.normal(keys[7], (D, 1)) * 0.02,
    }


def test_routing_shapes():
    """Routing should produce correct shapes."""
    cfg = Qwen35Config.mini()
    M, D = 8, cfg.d_model
    k = cfg.n_experts_per_token

    key = jax.random.key(0)
    x = jax.random.normal(key, (M, D))
    gate_w = jax.random.normal(jax.random.key(1), (D, cfg.n_routed_experts)) * 0.02

    indices, weights = moe_routing(x, gate_w, k)

    assert indices.shape == (M, k)
    assert weights.shape == (M, k)


def test_routing_weights_sum_to_one():
    """Expert weights per token should sum to 1 (softmax)."""
    M, D, E, k = 16, 64, 4, 2
    x = jax.random.normal(jax.random.key(2), (M, D))
    gate_w = jax.random.normal(jax.random.key(3), (D, E)) * 0.1

    _, weights = moe_routing(x, gate_w, k)
    sums = jnp.sum(weights, axis=-1)
    assert jnp.allclose(sums, 1.0, atol=1e-5)


def test_shared_expert_shape():
    """Shared expert should produce correct output shape."""
    cfg = Qwen35Config.mini()
    M, D = 8, cfg.d_model
    SI = cfg.shared_expert_intermediate_size

    x = jax.random.normal(jax.random.key(6), (M, D))
    gate_proj = jax.random.normal(jax.random.key(7), (D, SI)) * 0.02
    up_proj = jax.random.normal(jax.random.key(8), (D, SI)) * 0.02
    down_proj = jax.random.normal(jax.random.key(9), (SI, D)) * 0.02

    out = shared_expert_forward(x, gate_proj, up_proj, down_proj)
    assert out.shape == (M, D)
    assert not jnp.any(jnp.isnan(out))


def test_moe_layer_shape():
    """Full MoE layer should produce correct output shape."""
    cfg = Qwen35Config.mini()
    params = _make_moe_params(cfg, jax.random.key(10))
    B, T = 2, 8

    x = jax.random.normal(jax.random.key(11), (B, T, cfg.d_model))
    out = moe_layer(x, params, cfg.n_experts_per_token)
    assert out.shape == (B, T, cfg.d_model)


def test_moe_layer_no_nan():
    """MoE layer should not produce NaN."""
    cfg = Qwen35Config.mini()
    params = _make_moe_params(cfg, jax.random.key(12))
    B, T = 2, 8

    x = jax.random.normal(jax.random.key(13), (B, T, cfg.d_model))
    out = moe_layer(x, params, cfg.n_experts_per_token)
    assert not jnp.any(jnp.isnan(out))


def test_moe_nonzero_output():
    """MoE with random weights should produce non-zero output."""
    cfg = Qwen35Config.mini()
    params = _make_moe_params(cfg, jax.random.key(14))
    B, T = 1, 4

    x = jax.random.normal(jax.random.key(15), (B, T, cfg.d_model))
    out = moe_layer(x, params, cfg.n_experts_per_token)
    assert jnp.any(out != 0)


def test_moe_fp8_close_to_bf16():
    """FP8 native ragged_dot should produce similar output to dequantized path."""
    cfg = Qwen35Config.mini()
    params = _make_moe_params(cfg, jax.random.key(20))
    B, T = 2, 4

    x = jax.random.normal(jax.random.key(21), (B, T, cfg.d_model))
    out_bf16 = moe_layer(x, params, cfg.n_experts_per_token)

    params_fp8 = quantize_params_fp8(params)
    out_fp8 = moe_layer(x, params_fp8, cfg.n_experts_per_token)

    assert out_fp8.shape == out_bf16.shape
    assert not jnp.any(jnp.isnan(out_fp8))
    # fp8 quantization introduces some error — allow ~10% relative tolerance
    assert jnp.allclose(out_fp8, out_bf16, atol=0.5, rtol=0.15), (
        f"max diff: {jnp.max(jnp.abs(out_fp8 - out_bf16)):.4f}"
    )
