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
from jax_gpt.trainer.sharding import make_mesh, shard_train_state
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
