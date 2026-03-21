"""Tests for fused DeltaNet Pallas kernel correctness."""

import jax
import jax.numpy as jnp
import pytest

from jax_gpt.models.qwen35.pallas_deltanet import (
    fused_deltanet_step,
    fused_deltanet_step_ref,
)


def _random_inputs(B, H, dk, dv, key=None):
    """Generate random inputs matching Qwen3.5 DeltaNet shapes."""
    if key is None:
        key = jax.random.PRNGKey(42)
    keys = jax.random.split(key, 6)
    state = jax.random.normal(keys[0], (B, H, dk, dv), dtype=jnp.float32) * 0.1
    q = jax.random.normal(keys[1], (B, H, dk), dtype=jnp.float32)
    k = jax.random.normal(keys[2], (B, H, dk), dtype=jnp.float32)
    v = jax.random.normal(keys[3], (B, H, dv), dtype=jnp.float32)
    # g_factor = exp(g) where g is negative → g_factor in (0, 1)
    g_factor = jax.random.uniform(keys[4], (B, H), dtype=jnp.float32, minval=0.5, maxval=0.99)
    beta = jax.random.uniform(keys[5], (B, H), dtype=jnp.float32, minval=0.0, maxval=1.0)
    return state, q, k, v, g_factor, beta


class TestFusedDeltaNetStep:
    """Test fused Pallas kernel against reference implementation."""

    @pytest.mark.parametrize("B,H,dk,dv", [
        (1, 1, 128, 128),     # minimal
        (2, 4, 128, 128),     # small batch
        (1, 8, 128, 128),     # single batch, TP=8 heads
        (16, 8, 128, 128),    # B=128/8 per dp shard, 8 heads per TP
    ])
    def test_correctness(self, B, H, dk, dv):
        """Pallas kernel output matches reference JAX implementation."""
        inputs = _random_inputs(B, H, dk, dv)

        ref_state, ref_output = fused_deltanet_step_ref(*inputs)
        new_state, output = fused_deltanet_step(*inputs)

        # State should match closely (all f32 computation)
        jnp.testing.assert_allclose(new_state, ref_state, rtol=1e-5, atol=1e-5)
        jnp.testing.assert_allclose(output, ref_output, rtol=1e-5, atol=1e-5)

    def test_decay_only(self):
        """When k=0 and beta=0, state just decays, output=0."""
        B, H, dk, dv = 1, 1, 128, 128
        key = jax.random.PRNGKey(0)
        state = jax.random.normal(key, (B, H, dk, dv), dtype=jnp.float32)
        q = jnp.zeros((B, H, dk), dtype=jnp.float32)
        k = jnp.zeros((B, H, dk), dtype=jnp.float32)
        v = jnp.zeros((B, H, dv), dtype=jnp.float32)
        g_factor = jnp.full((B, H), 0.9, dtype=jnp.float32)
        beta = jnp.zeros((B, H), dtype=jnp.float32)

        new_state, output = fused_deltanet_step(state, q, k, v, g_factor, beta)

        expected_state = state * 0.9
        jnp.testing.assert_allclose(new_state, expected_state, rtol=1e-6)
        jnp.testing.assert_allclose(output, jnp.zeros_like(output), atol=1e-6)

    def test_rank1_update(self):
        """Verify rank-1 update: state += outer(k, delta)."""
        B, H, dk, dv = 1, 1, 4, 4  # small for manual verification
        state = jnp.zeros((B, H, dk, dv), dtype=jnp.float32)
        q = jnp.zeros((B, H, dk), dtype=jnp.float32)
        k = jnp.ones((B, H, dk), dtype=jnp.float32) / 2  # unit-ish k
        v = jnp.ones((B, H, dv), dtype=jnp.float32)       # v = ones
        g_factor = jnp.ones((B, H), dtype=jnp.float32)     # no decay
        beta = jnp.ones((B, H), dtype=jnp.float32)         # full gate

        new_state, output = fused_deltanet_step(state, q, k, v, g_factor, beta)
        ref_state, ref_output = fused_deltanet_step_ref(state, q, k, v, g_factor, beta)

        jnp.testing.assert_allclose(new_state, ref_state, rtol=1e-6)
        jnp.testing.assert_allclose(output, ref_output, rtol=1e-6)

    def test_multiple_steps(self):
        """Run multiple recurrent steps and verify state accumulation."""
        B, H, dk, dv = 2, 8, 128, 128
        state = jnp.zeros((B, H, dk, dv), dtype=jnp.float32)
        ref_state = jnp.zeros_like(state)

        key = jax.random.PRNGKey(123)
        for step in range(10):
            key, subkey = jax.random.split(key)
            _, q, k, v, g_factor, beta = _random_inputs(B, H, dk, dv, subkey)

            state, output = fused_deltanet_step(state, q, k, v, g_factor, beta)
            ref_state, ref_output = fused_deltanet_step_ref(ref_state, q, k, v, g_factor, beta)

            jnp.testing.assert_allclose(state, ref_state, rtol=1e-4, atol=1e-4)
            jnp.testing.assert_allclose(output, ref_output, rtol=1e-4, atol=1e-4)

    def test_output_shapes(self):
        """Verify output shapes match expected."""
        B, H, dk, dv = 4, 8, 128, 128
        inputs = _random_inputs(B, H, dk, dv)
        new_state, output = fused_deltanet_step(*inputs)

        assert new_state.shape == (B, H, dk, dv)
        assert output.shape == (B, H, dv)
        assert new_state.dtype == jnp.float32
        assert output.dtype == jnp.float32
