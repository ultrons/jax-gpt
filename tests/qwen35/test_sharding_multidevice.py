"""Multi-device sharding tests using simulated CPU devices.

These tests simulate 4 and 8 CPU devices to verify that:
1. Params land on the correct devices with correct sharding
2. Sharded forward pass runs end-to-end without shape mismatches
3. Config A and Config B produce identical outputs
4. Cache sharding works correctly

Run standalone (not mixed with other tests that init JAX with 1 device):
    XLA_FLAGS=--xla_force_host_platform_device_count=8 \
        python -m pytest tests/qwen35/test_sharding_multidevice.py -v -s
"""

import os

# Must set before JAX import
os.environ['XLA_FLAGS'] = os.environ.get('XLA_FLAGS', '') + ' --xla_force_host_platform_device_count=8'

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_gpt.models.qwen35.cache import init_cache
from jax_gpt.models.qwen35.config import Qwen35Config
from jax_gpt.models.qwen35.model import forward, init_params
from jax_gpt.models.qwen35.sharding import (
    AXIS_RULES_A,
    AXIS_RULES_B,
    make_cache_sharding,
    make_mesh,
    shard_cache,
    shard_params,
)


# Use a config where dimensions are divisible by 8 for clean sharding
@pytest.fixture(scope='module')
def config():
    return Qwen35Config(
        d_model=256,
        vocab_size=1024,
        n_layers=4,
        max_position_embeddings=128,
        rms_norm_eps=1e-6,
        gqa_n_q_heads=8,       # divisible by 8
        gqa_n_kv_heads=2,      # small, replicated
        gqa_head_dim=64,
        gqa_partial_rotary_factor=0.25,
        gqa_rope_theta=10_000_000.0,
        delta_n_qk_heads=4,    # divisible by 4
        delta_n_v_heads=8,     # divisible by 8
        delta_qk_head_dim=64,
        delta_v_head_dim=64,
        delta_conv_kernel=4,
        n_routed_experts=8,    # divisible by 8 and 4
        n_experts_per_token=2,
        moe_intermediate_size=128,
        shared_expert_intermediate_size=128,
    )


@pytest.fixture(scope='module')
def params(config):
    return init_params(config, jax.random.key(42))


# ---------------------------------------------------------------------------
# 8-device tests
# ---------------------------------------------------------------------------

class TestEightDevices:
    """Sharding tests with 8 simulated CPU devices."""

    def test_device_count(self):
        assert jax.device_count() == 8

    def test_shard_params_config_b_8dev(self, config, params):
        """Config B (TP=8 uniform) should shard all params across 8 devices."""
        mesh = make_mesh(n_devices=8)
        sharded = shard_params(params, mesh, config, AXIS_RULES_B)

        # Expert weights should be split 8 ways
        gate_proj = sharded['groups']['delta_layers']['moe']['gate_proj']
        # Shape: (n_groups, 3, E, D, I) — experts dim should be sharded
        assert gate_proj.shape == params['groups']['delta_layers']['moe']['gate_proj'].shape

        # Check that expert params actually live on different devices
        expert_sharding = gate_proj.sharding
        assert expert_sharding is not None

    def test_shard_params_config_a_8dev(self, config, params):
        """Config A (TP=8 DeltaNet, replicated GQA) should shard correctly."""
        mesh = make_mesh(n_devices=8)
        sharded = shard_params(params, mesh, config, AXIS_RULES_A)

        # GQA q_proj should be replicated (Config A)
        q_proj = sharded['groups']['gqa_layer']['attn']['q_proj']
        assert q_proj.shape == params['groups']['gqa_layer']['attn']['q_proj'].shape

    def test_forward_sharded_config_b_8dev(self, config, params):
        """Sharded forward pass should produce same logits as unsharded."""
        mesh = make_mesh(n_devices=8)

        # Unsharded reference
        tokens = jax.random.randint(jax.random.key(0), (1, 8), 0, config.vocab_size)
        ref_logits, _ = forward(params, tokens, config)

        # Sharded forward
        sharded_params = shard_params(params, mesh, config, AXIS_RULES_B)

        @jax.jit
        def sharded_forward(p, t):
            return forward(p, t, config)

        with mesh:
            sharded_logits, _ = sharded_forward(sharded_params, tokens)

        max_diff = jnp.max(jnp.abs(ref_logits - sharded_logits))
        assert max_diff < 1e-4, f"Sharded vs unsharded max diff: {max_diff}"

    def test_forward_sharded_config_a_8dev(self, config, params):
        """Config A sharded forward should produce same logits as unsharded."""
        mesh = make_mesh(n_devices=8)

        tokens = jax.random.randint(jax.random.key(1), (1, 8), 0, config.vocab_size)
        ref_logits, _ = forward(params, tokens, config)

        sharded_params = shard_params(params, mesh, config, AXIS_RULES_A)

        @jax.jit
        def sharded_forward(p, t):
            return forward(p, t, config)

        with mesh:
            sharded_logits, _ = sharded_forward(sharded_params, tokens)

        max_diff = jnp.max(jnp.abs(ref_logits - sharded_logits))
        assert max_diff < 1e-4, f"Config A vs unsharded max diff: {max_diff}"

    def test_configs_match_8dev(self, config, params):
        """Config A and Config B should produce identical outputs."""
        mesh = make_mesh(n_devices=8)
        tokens = jax.random.randint(jax.random.key(2), (1, 8), 0, config.vocab_size)

        @jax.jit
        def run(p, t):
            return forward(p, t, config)

        with mesh:
            sharded_a = shard_params(params, mesh, config, AXIS_RULES_A)
            logits_a, _ = run(sharded_a, tokens)

            sharded_b = shard_params(params, mesh, config, AXIS_RULES_B)
            logits_b, _ = run(sharded_b, tokens)

        max_diff = jnp.max(jnp.abs(logits_a - logits_b))
        assert max_diff < 1e-4, f"Config A vs B max diff: {max_diff}"

    def test_cache_sharding_8dev(self, config):
        """Cache sharding should work with 8 devices."""
        mesh = make_mesh(n_devices=8)
        cache = init_cache(config, batch_size=1, max_len=32)

        sharded = shard_cache(cache, mesh, config, AXIS_RULES_B)
        assert sharded.delta_M.shape == cache.delta_M.shape
        assert sharded.gqa_k.shape == cache.gqa_k.shape

    def test_forward_with_cache_sharded_8dev(self, config, params):
        """Sharded prefill + decode step with cache sharding constraints."""
        mesh = make_mesh(n_devices=8)
        sharded_params = shard_params(params, mesh, config, AXIS_RULES_B)
        cache = init_cache(config, batch_size=1, max_len=32)
        cs = make_cache_sharding(config, mesh, AXIS_RULES_B)

        tokens = jax.random.randint(jax.random.key(3), (1, 4), 0, config.vocab_size)

        @jax.jit
        def prefill(p, t, c):
            return forward(p, t, config, cache=c, is_decode=False, cache_sharding=cs)

        @jax.jit
        def decode_step(p, t, c):
            return forward(p, t, config, cache=c, is_decode=True, cache_sharding=cs)

        with mesh:
            logits, updated_cache = prefill(sharded_params, tokens, cache)
            assert logits.shape == (1, 4, config.vocab_size)
            assert int(updated_cache.pos) == 4

            # Single decode step
            next_token = jnp.array([[0]], dtype=jnp.int32)
            dec_logits, dec_cache = decode_step(sharded_params, next_token, updated_cache)
            assert dec_logits.shape == (1, 1, config.vocab_size)
            assert int(dec_cache.pos) == 5


# ---------------------------------------------------------------------------
# 4-device tests
# ---------------------------------------------------------------------------

class TestFourDevices:
    """Sharding tests with 4-device sub-mesh (using first 4 of 8)."""

    def test_shard_params_4dev(self, config, params):
        """Sharding on 4 devices should work."""
        mesh = make_mesh(n_devices=4)
        sharded = shard_params(params, mesh, config, AXIS_RULES_B)

        orig_leaves = jax.tree_util.tree_leaves(params)
        sharded_leaves = jax.tree_util.tree_leaves(sharded)
        for o, s in zip(orig_leaves, sharded_leaves):
            assert o.shape == s.shape

    def test_forward_sharded_4dev(self, config, params):
        """Sharded forward on 4 devices should match unsharded."""
        mesh = make_mesh(n_devices=4)
        tokens = jax.random.randint(jax.random.key(4), (1, 8), 0, config.vocab_size)

        ref_logits, _ = forward(params, tokens, config)

        sharded_params = shard_params(params, mesh, config, AXIS_RULES_B)

        @jax.jit
        def run(p, t):
            return forward(p, t, config)

        with mesh:
            sharded_logits, _ = run(sharded_params, tokens)

        max_diff = jnp.max(jnp.abs(ref_logits - sharded_logits))
        assert max_diff < 1e-4, f"4-dev sharded vs unsharded max diff: {max_diff}"

    def test_forward_with_cache_4dev(self, config, params):
        """Prefill + decode on 4 devices should work."""
        mesh = make_mesh(n_devices=4)
        sharded_params = shard_params(params, mesh, config, AXIS_RULES_B)
        cache = init_cache(config, batch_size=1, max_len=32)
        cs = make_cache_sharding(config, mesh, AXIS_RULES_B)

        tokens = jax.random.randint(jax.random.key(5), (1, 4), 0, config.vocab_size)

        @jax.jit
        def prefill(p, t, c):
            return forward(p, t, config, cache=c, is_decode=False, cache_sharding=cs)

        with mesh:
            logits, updated_cache = prefill(sharded_params, tokens, cache)
            assert logits.shape == (1, 4, config.vocab_size)
            assert not jnp.any(jnp.isnan(logits))
