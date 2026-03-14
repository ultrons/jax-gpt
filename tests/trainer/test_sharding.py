import os
import pytest
import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx

from jax_gpt.models.gpt2.config import GPTConfig
from jax_gpt.models.gpt2.model import GPT
from jax_gpt.trainer.config import TrainConfig
from jax_gpt.trainer.train_state import create_train_state
from jax_gpt.trainer.sharding import make_mesh, shard_train_state, logical_to_physical
from jax.sharding import PartitionSpec as P
import optax


def tiny_config():
    return GPTConfig(d_model=64, d_head=16, d_ff=256, n_head=4, n_kv_head=4, n_layers=2,
                     d_context=32, vocab_size=100)


def test_make_mesh_trivial():
    """1-device mesh should work on any machine."""
    config = TrainConfig(dp=1, fsdp=1, tp=1, sp=1)
    mesh = make_mesh(config)
    assert mesh.shape == {'dp': 1, 'fsdp': 1, 'tp': 1, 'sp': 1}


def test_unsharded_equals_trivially_sharded():
    """
    Trivial mesh (all dims=1): sharded forward pass must be bitwise identical
    to unsharded. This is Tier 1 correctness — always runs on M1 CPU.
    """
    cfg = tiny_config()
    model = GPT(cfg, rngs=nnx.Rngs(0))
    x = jnp.ones((2, 8), dtype=jnp.int32)

    # Unsharded reference
    logits_ref, _ = model(x, deterministic=True)

    # Trivial sharding
    train_cfg = TrainConfig(dp=1, fsdp=1, tp=1, sp=1)
    mesh = make_mesh(train_cfg)
    tx = optax.adam(1e-3)
    state = create_train_state(model, tx)
    state = shard_train_state(state, mesh, cfg)

    logits_sharded, _ = model(x, deterministic=True)
    np.testing.assert_allclose(
        np.array(logits_ref), np.array(logits_sharded), rtol=1e-5
    )


def test_logical_specs_extracted_correctly():
    """Verify that nnx.with_partitioning annotations produce expected logical specs."""
    cfg = tiny_config()
    model = GPT(cfg, rngs=nnx.Rngs(0))

    params = nnx.state(model, nnx.Param)
    logical_specs = nnx.get_partition_spec(params)

    # Collect (name, logical_spec, ndim) tuples
    results = {}
    def collect(path, leaf, spec):
        name = '.'.join(
            str(k).strip("[].'\"") for k in path
            if str(k).strip("[].'\"") not in ('raw_value', 'value')
        )
        if isinstance(spec, P):
            results[name] = (spec, leaf.ndim)
        return leaf
    jax.tree_util.tree_map_with_path(collect, params, logical_specs)

    # Check logical annotations
    assert results['wte.embedding'][0] == P('vocab', 'embed')
    assert results['wpe.embedding'][0] == P('context', 'embed')
    assert results['h.attn.c_attn.kernel'][0] == P('embed', 'joined_heads')
    assert results['h.attn.c_attn.bias'][0] == P('joined_heads',)
    assert results['h.attn.c_proj.kernel'][0] == P('heads', 'embed')
    assert results['h.attn.c_proj.bias'][0] == P('embed',)
    assert results['h.mlp.c_fc.kernel'][0] == P('embed', 'mlp')
    assert results['h.mlp.c_fc.bias'][0] == P('mlp',)
    assert results['h.mlp.c_proj.kernel'][0] == P('mlp', 'embed')
    assert results['h.mlp.c_proj.bias'][0] == P('embed',)
    assert results['h.ln_1.layer_norm.scale'][0] == P('embed',)
    assert results['h.ln_2.layer_norm.scale'][0] == P('embed',)
    assert results['ln_f.layer_norm.scale'][0] == P('embed',)
    assert results['lm_head.kernel'][0] == P('embed', 'vocab')

    # Check logical→physical mapping for vmapped params (ndim=3, spec len=2)
    spec, ndim = results['h.attn.c_attn.kernel']
    assert logical_to_physical(spec, ndim) == P(None, None, 'tp')

    spec, ndim = results['h.attn.c_proj.kernel']
    assert logical_to_physical(spec, ndim) == P(None, 'tp', None)

    spec, ndim = results['h.mlp.c_fc.kernel']
    assert logical_to_physical(spec, ndim) == P(None, None, 'tp')

    # Check non-vmapped params
    spec, ndim = results['wte.embedding']
    assert logical_to_physical(spec, ndim) == P('tp', None)

    spec, ndim = results['ln_f.layer_norm.scale']
    assert logical_to_physical(spec, ndim) == P(None,)


@pytest.mark.skipif(
    jax.device_count() < 8,
    reason="Needs 8 devices; set XLA_FLAGS=--xla_force_host_platform_device_count=8"
)
def test_tp8_forward_matches_reference():
    """
    TP=8 on 8 fake CPU devices: sharded output must match unsharded FP32
    within tolerance. Tier 2 — requires XLA_FLAGS.
    """
    cfg = tiny_config()
    model_ref = GPT(cfg, rngs=nnx.Rngs(42))
    x = jnp.ones((2, 8), dtype=jnp.int32)
    logits_ref, _ = model_ref(x, deterministic=True)

    model_sharded = GPT(cfg, rngs=nnx.Rngs(42))
    train_cfg = TrainConfig(tp=8)
    mesh = make_mesh(train_cfg)
    tx = optax.adam(1e-3)
    state = create_train_state(model_sharded, tx)
    state = shard_train_state(state, mesh, cfg)

    logits_sharded, _ = model_sharded(x, deterministic=True, mesh=mesh)
    np.testing.assert_allclose(
        np.array(logits_ref), np.array(logits_sharded), rtol=1e-4
    )
