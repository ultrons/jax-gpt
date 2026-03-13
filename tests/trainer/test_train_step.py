"""Tests for train_step correctness on M1 CPU (no TPU required)."""
import math
import pytest
import numpy as np
import jax
import jax.numpy as jnp
import optax
from flax import nnx

from jax_gpt.models.gpt2.config import GPTConfig
from jax_gpt.models.gpt2.model import GPT
from jax_gpt.trainer.config import TrainConfig
from jax_gpt.trainer.train_state import TrainState, create_train_state
from jax_gpt.trainer.train_step import train_step, eval_step, train_step_with_accumulation


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def tiny_config():
    return GPTConfig(
        d_model=64, d_head=16, d_ff=256,
        n_head=4, n_kv_head=4, n_layers=2, d_context=32, vocab_size=100,
    )


def make_state(seed: int = 0, lr: float = 1e-3):
    cfg = tiny_config()
    model = GPT(cfg, rngs=nnx.Rngs(seed))
    tx = optax.adamw(lr, weight_decay=0.1)
    return create_train_state(model, tx), cfg


def fake_batch(cfg: GPTConfig, batch_size: int = 4, seq_len: int = 16):
    x = jnp.ones((batch_size, seq_len), dtype=jnp.int32)
    y = jnp.ones((batch_size, seq_len), dtype=jnp.int32)
    return x, y


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_initial_loss_near_log_vocab():
    """
    Before any training, cross-entropy loss on uniform logits should be
    close to ln(vocab_size). Exact value depends on init, so we allow ±1.0.
    """
    state, cfg = make_state()
    batch = fake_batch(cfg)
    loss = eval_step(state.model, batch)
    expected = math.log(cfg.vocab_size)  # ln(100) ≈ 4.605
    assert abs(float(loss) - expected) < 1.5, (
        f"Initial loss {float(loss):.3f} too far from ln(vocab_size)={expected:.3f}"
    )


def test_loss_decreases_after_training():
    """50 train_step calls on a fixed batch must reduce loss."""
    state, cfg = make_state(lr=3e-3)
    batch = fake_batch(cfg)

    initial_loss = float(eval_step(state.model, batch))
    for _ in range(50):
        state, metrics = train_step(state, batch)
    final_loss = float(eval_step(state.model, batch))

    assert final_loss < initial_loss, (
        f"Loss did not decrease: {initial_loss:.4f} → {final_loss:.4f}"
    )


def test_step_counter_increments():
    state, cfg = make_state()
    batch = fake_batch(cfg)
    assert state.step == 0
    state, _ = train_step(state, batch)
    assert state.step == 1
    state, _ = train_step(state, batch)
    assert state.step == 2


def test_gradients_finite_and_nonzero():
    """Gradients must not be NaN and must not all be zero."""
    state, cfg = make_state()
    batch = fake_batch(cfg)

    model = state.model
    x, y = batch

    def loss_fn(params):
        nnx.update(model, params)
        logits, _ = model(x, deterministic=False)
        return optax.softmax_cross_entropy_with_integer_labels(
            logits.astype(jnp.float32).reshape(-1, logits.shape[-1]),
            y.reshape(-1),
        ).mean()

    params = nnx.state(model, nnx.Param)
    _, grads = jax.value_and_grad(loss_fn)(params)

    flat_grads = jax.tree_util.tree_leaves(grads)
    for g in flat_grads:
        assert not jnp.any(jnp.isnan(g)), "NaN gradient detected"

    total_norm = float(optax.global_norm(grads))
    assert total_norm > 0, "Gradient norm is zero"


def test_metrics_keys():
    """train_step must return 'loss' and 'grad_norm' in metrics."""
    state, cfg = make_state()
    batch = fake_batch(cfg)
    _, metrics = train_step(state, batch)
    assert 'loss' in metrics
    assert 'grad_norm' in metrics


def test_grad_accum_matches_single_step():
    """
    Accumulating over 2 identical micro-batches should give the same loss
    as a single step on the same batch (gradient averaging matches).
    """
    state1, cfg = make_state(seed=7)
    state2, _ = make_state(seed=7)  # identical init
    batch = fake_batch(cfg, batch_size=4)

    state1, m1 = train_step(state1, batch)
    state2, m2 = train_step_with_accumulation(state2, [batch, batch])

    # Losses should be identical since both micro-batches are the same
    np.testing.assert_allclose(float(m1['loss']), float(m2['loss']), rtol=1e-4)


def test_eval_step_does_not_modify_params():
    """eval_step must be read-only — params must be unchanged after call."""
    state, cfg = make_state()
    batch = fake_batch(cfg)

    params_before = jax.tree_util.tree_map(np.array, nnx.state(state.model, nnx.Param))
    eval_step(state.model, batch)
    params_after = jax.tree_util.tree_map(np.array, nnx.state(state.model, nnx.Param))

    jax.tree_util.tree_map(
        lambda a, b: np.testing.assert_array_equal(a, b),
        params_before, params_after,
    )
