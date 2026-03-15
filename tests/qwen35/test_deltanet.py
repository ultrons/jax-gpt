"""Tests for Gated DeltaNet linear attention."""

import jax
import jax.numpy as jnp
import pytest

from jax_gpt.models.qwen35.config import Qwen35Config
from jax_gpt.models.qwen35.deltanet import (
    deltanet_prefill,
    deltanet_recurrent_step,
)


def _make_deltanet_params(cfg: Qwen35Config, key: jax.Array) -> dict:
    """Create random DeltaNet params for testing."""
    keys = jax.random.split(key, 10)
    key_dim = cfg.delta_n_qk_heads * cfg.delta_qk_head_dim
    value_dim = cfg.delta_n_v_heads * cfg.delta_v_head_dim
    conv_dim = key_dim * 2 + value_dim

    return {
        'in_proj_qkv': jax.random.normal(keys[0], (cfg.d_model, conv_dim)) * 0.02,
        'in_proj_z': jax.random.normal(keys[1], (cfg.d_model, value_dim)) * 0.02,
        'in_proj_b': jax.random.normal(keys[2], (cfg.d_model, cfg.delta_n_v_heads)) * 0.02,
        'in_proj_a': jax.random.normal(keys[3], (cfg.d_model, cfg.delta_n_v_heads)) * 0.02,
        'conv_weight': jax.random.normal(keys[4], (conv_dim, cfg.delta_conv_kernel)) * 0.02,
        'A_log': jnp.log(jax.random.uniform(keys[5], (cfg.delta_n_v_heads,), minval=0.1, maxval=16.0)),
        'dt_bias': jnp.ones(cfg.delta_n_v_heads),
        'norm_weight': jnp.zeros(cfg.delta_v_head_dim),
        'out_proj': jax.random.normal(keys[6], (value_dim, cfg.d_model)) * 0.02,
    }


def test_recurrent_step_shape():
    """Recurrent step should produce correct output and state shapes."""
    cfg = Qwen35Config.mini()
    params = _make_deltanet_params(cfg, jax.random.key(0))
    B = 2
    key_dim = cfg.delta_n_qk_heads * cfg.delta_qk_head_dim
    value_dim = cfg.delta_n_v_heads * cfg.delta_v_head_dim
    conv_dim = key_dim * 2 + value_dim

    x = jax.random.normal(jax.random.key(1), (B, 1, cfg.d_model))
    state = jnp.zeros((B, cfg.delta_n_v_heads, cfg.delta_qk_head_dim, cfg.delta_v_head_dim))
    conv_state = jnp.zeros((B, conv_dim, cfg.delta_conv_kernel))

    out, new_state, new_conv = deltanet_recurrent_step(
        x, params, state, conv_state,
        cfg.delta_n_qk_heads, cfg.delta_n_v_heads,
        cfg.delta_qk_head_dim, cfg.delta_v_head_dim,
    )
    assert out.shape == (B, 1, cfg.d_model)
    assert new_state.shape == state.shape
    assert new_conv.shape == conv_state.shape


def test_recurrent_step_no_nan():
    """Recurrent step should not produce NaN."""
    cfg = Qwen35Config.mini()
    params = _make_deltanet_params(cfg, jax.random.key(2))
    B = 2
    key_dim = cfg.delta_n_qk_heads * cfg.delta_qk_head_dim
    value_dim = cfg.delta_n_v_heads * cfg.delta_v_head_dim
    conv_dim = key_dim * 2 + value_dim

    x = jax.random.normal(jax.random.key(3), (B, 1, cfg.d_model))
    state = jnp.zeros((B, cfg.delta_n_v_heads, cfg.delta_qk_head_dim, cfg.delta_v_head_dim))
    conv_state = jnp.zeros((B, conv_dim, cfg.delta_conv_kernel))

    out, new_state, _ = deltanet_recurrent_step(
        x, params, state, conv_state,
        cfg.delta_n_qk_heads, cfg.delta_n_v_heads,
        cfg.delta_qk_head_dim, cfg.delta_v_head_dim,
    )
    assert not jnp.any(jnp.isnan(out))
    assert not jnp.any(jnp.isnan(new_state))


def test_prefill_shape():
    """Prefill should produce correct output and state shapes."""
    cfg = Qwen35Config.mini()
    params = _make_deltanet_params(cfg, jax.random.key(4))
    B, T = 2, 16

    x = jax.random.normal(jax.random.key(5), (B, T, cfg.d_model))
    out, final_state, final_conv = deltanet_prefill(
        x, params,
        cfg.delta_n_qk_heads, cfg.delta_n_v_heads,
        cfg.delta_qk_head_dim, cfg.delta_v_head_dim,
        cfg.delta_conv_kernel,
    )
    assert out.shape == (B, T, cfg.d_model)
    assert final_state.shape == (B, cfg.delta_n_v_heads, cfg.delta_qk_head_dim, cfg.delta_v_head_dim)


def test_prefill_matches_recurrent():
    """Prefill final state should match sequential recurrent application."""
    cfg = Qwen35Config.mini()
    params = _make_deltanet_params(cfg, jax.random.key(6))
    B, T = 1, 8
    key_dim = cfg.delta_n_qk_heads * cfg.delta_qk_head_dim
    value_dim = cfg.delta_n_v_heads * cfg.delta_v_head_dim
    conv_dim = key_dim * 2 + value_dim

    x = jax.random.normal(jax.random.key(7), (B, T, cfg.d_model))

    # Run prefill
    prefill_out, prefill_state, _ = deltanet_prefill(
        x, params,
        cfg.delta_n_qk_heads, cfg.delta_n_v_heads,
        cfg.delta_qk_head_dim, cfg.delta_v_head_dim,
        cfg.delta_conv_kernel,
    )

    # Run recurrent step-by-step
    state = jnp.zeros((B, cfg.delta_n_v_heads, cfg.delta_qk_head_dim, cfg.delta_v_head_dim))
    conv_state = jnp.zeros((B, conv_dim, cfg.delta_conv_kernel))
    recurrent_outs = []
    for t in range(T):
        x_t = x[:, t:t+1, :]  # (B, 1, D)
        out_t, state, conv_state = deltanet_recurrent_step(
            x_t, params, state, conv_state,
            cfg.delta_n_qk_heads, cfg.delta_n_v_heads,
            cfg.delta_qk_head_dim, cfg.delta_v_head_dim,
        )
        recurrent_outs.append(out_t)
    recurrent_out = jnp.concatenate(recurrent_outs, axis=1)

    # States should match
    assert jnp.allclose(prefill_state, state, atol=1e-4), \
        f"State mismatch: max diff = {jnp.max(jnp.abs(prefill_state - state))}"

    # Outputs should match
    assert jnp.allclose(prefill_out, recurrent_out, atol=1e-4), \
        f"Output mismatch: max diff = {jnp.max(jnp.abs(prefill_out - recurrent_out))}"


def test_state_decays():
    """With no new input contribution, state should decay over steps."""
    cfg = Qwen35Config.mini()
    params = _make_deltanet_params(cfg, jax.random.key(8))
    B = 1
    key_dim = cfg.delta_n_qk_heads * cfg.delta_qk_head_dim
    value_dim = cfg.delta_n_v_heads * cfg.delta_v_head_dim
    conv_dim = key_dim * 2 + value_dim

    # Initialize with some state
    state = jax.random.normal(jax.random.key(9), (B, cfg.delta_n_v_heads, cfg.delta_qk_head_dim, cfg.delta_v_head_dim))
    conv_state = jnp.zeros((B, conv_dim, cfg.delta_conv_kernel))
    initial_norm = jnp.linalg.norm(state)

    # Run several steps with zero input
    x = jnp.zeros((B, 1, cfg.d_model))
    for _ in range(5):
        _, state, conv_state = deltanet_recurrent_step(
            x, params, state, conv_state,
            cfg.delta_n_qk_heads, cfg.delta_n_v_heads,
            cfg.delta_qk_head_dim, cfg.delta_v_head_dim,
        )

    # State norm should decrease (decay)
    final_norm = jnp.linalg.norm(state)
    assert final_norm < initial_norm, \
        f"State should decay: initial={initial_norm}, final={final_norm}"
