"""Tests for logical axis sharding."""

import jax
import jax.numpy as jnp
import pytest
from jax.sharding import PartitionSpec as P

from jax_gpt.models.qwen35.cache import init_cache
from jax_gpt.models.qwen35.config import Qwen35Config
from jax_gpt.models.qwen35.model import init_params
from jax_gpt.models.qwen35.sharding import (
    AXIS_RULES_A,
    AXIS_RULES_B,
    _param_logical_axes,
    _resolve_spec,
    make_mesh,
    shard_cache,
    shard_params,
)


@pytest.fixture
def mini_config():
    return Qwen35Config.mini()


@pytest.fixture
def mini_params(mini_config):
    return init_params(mini_config, jax.random.key(0))


def test_logical_axes_cover_all_params(mini_config, mini_params):
    """Every parameter should match at least one logical axis spec."""
    logical_axes = _param_logical_axes(mini_config)

    unmatched = []
    flat_params = jax.tree_util.tree_leaves_with_path(mini_params)
    for path, leaf in flat_params:
        path_str = '.'.join(
            str(k).strip("[]'.\"") for k in path
            if not str(k).strip("[]'.\"").isdigit()
        )
        best_len = -1
        for key in logical_axes:
            if path_str.endswith(key) and len(key) > best_len:
                best_len = len(key)
        if best_len < 0:
            unmatched.append(path_str)

    assert not unmatched, f"Params with no logical axis match: {unmatched}"


def test_resolve_spec_deltanet_attn():
    """DeltaNet attn weights should resolve to TP-sharded specs."""
    cfg = Qwen35Config.mini()
    logical_axes = _param_logical_axes(cfg)

    spec = _resolve_spec('groups.delta_layers.attn.in_proj_qkv', logical_axes, AXIS_RULES_B)
    assert spec == P(None, 'tp'), f"Expected P(None, 'tp'), got {spec}"

    spec = _resolve_spec('groups.delta_layers.attn.out_proj', logical_axes, AXIS_RULES_B)
    assert spec == P('tp', None)


def test_resolve_spec_gqa_config_a():
    """Config A: GQA Q heads should be replicated."""
    cfg = Qwen35Config.mini()
    logical_axes = _param_logical_axes(cfg)

    spec = _resolve_spec('groups.gqa_layer.attn.q_proj', logical_axes, AXIS_RULES_A)
    assert spec == P(None, None), f"Config A GQA q_proj should be replicated, got {spec}"


def test_resolve_spec_gqa_config_b():
    """Config B: GQA Q heads should be TP-sharded."""
    cfg = Qwen35Config.mini()
    logical_axes = _param_logical_axes(cfg)

    spec = _resolve_spec('groups.gqa_layer.attn.q_proj', logical_axes, AXIS_RULES_B)
    assert spec == P(None, 'tp'), f"Config B GQA q_proj should be P(None, 'tp'), got {spec}"


def test_resolve_spec_experts():
    """Expert weights should be sharded along expert dim."""
    cfg = Qwen35Config.mini()
    logical_axes = _param_logical_axes(cfg)

    spec = _resolve_spec('moe.gate_proj', logical_axes, AXIS_RULES_B)
    assert spec == P('tp', None, None)


def test_resolve_spec_shared_expert():
    """Shared expert weights should be replicated."""
    cfg = Qwen35Config.mini()
    logical_axes = _param_logical_axes(cfg)

    spec = _resolve_spec('moe.shared_gate_proj', logical_axes, AXIS_RULES_B)
    assert spec == P(None, None)


def test_make_mesh():
    """make_mesh should create a mesh with available devices."""
    mesh = make_mesh()
    assert len(mesh.devices.flatten()) == jax.device_count()
    assert mesh.axis_names == ('tp',)


def test_shard_params_runs(mini_config, mini_params):
    """shard_params should run without errors on single device."""
    mesh = make_mesh()
    sharded = shard_params(mini_params, mesh, mini_config, AXIS_RULES_B)
    # Verify structure preserved
    assert set(sharded.keys()) == set(mini_params.keys())
    # Verify shapes preserved
    orig_leaves = jax.tree_util.tree_leaves(mini_params)
    sharded_leaves = jax.tree_util.tree_leaves(sharded)
    for o, s in zip(orig_leaves, sharded_leaves):
        assert o.shape == s.shape


def test_shard_cache_runs(mini_config):
    """shard_cache should run without errors on single device."""
    cache = init_cache(mini_config, batch_size=1, max_len=32)
    mesh = make_mesh()
    sharded = shard_cache(cache, mesh, mini_config, AXIS_RULES_B)
    assert sharded.delta_M.shape == cache.delta_M.shape
    assert sharded.gqa_k.shape == cache.gqa_k.shape


def test_both_configs_run(mini_config, mini_params):
    """Both Config A and Config B should apply without errors."""
    mesh = make_mesh()
    for name, rules in [('A', AXIS_RULES_A), ('B', AXIS_RULES_B)]:
        sharded = shard_params(mini_params, mesh, mini_config, rules)
        leaves = jax.tree_util.tree_leaves(sharded)
        assert all(not jnp.any(jnp.isnan(l)) for l in leaves), \
            f"Config {name} produced NaN"
