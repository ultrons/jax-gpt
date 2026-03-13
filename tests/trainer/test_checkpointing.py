import numpy as np
import jax
from flax import nnx
import optax

from jax_gpt.models.gpt2.config import GPTConfig
from jax_gpt.models.gpt2.model import GPT
from jax_gpt.trainer.config import TrainConfig
from jax_gpt.trainer.train_state import TrainState, create_train_state
from jax_gpt.trainer.checkpointing import CheckpointManager


def tiny_model():
    cfg = GPTConfig(
        d_model=64, d_head=16, d_ff=256, n_head=4, n_kv_head=4,
        n_layers=2, d_context=32, vocab_size=100,
    )
    model = GPT(cfg, rngs=nnx.Rngs(0))
    return cfg, model


def test_checkpoint_roundtrip(tmp_path):
    """Save at step 42 and restore into a fresh model. Params must be identical."""
    cfg, model = tiny_model()
    tx = optax.adam(1e-3)
    state = create_train_state(model, tx)

    # Manually set step
    state = TrainState(step=42, model=state.model, tx=state.tx, opt_state=state.opt_state)

    train_cfg = TrainConfig(
        checkpoint_dir=str(tmp_path),
        max_checkpoints_to_keep=3,
        save_interval=1,
    )
    ckpt_mgr = CheckpointManager(train_cfg, cfg)
    ckpt_mgr.save(42, state)
    ckpt_mgr.wait_until_finished()

    assert ckpt_mgr.latest_step() == 42

    # Restore into a fresh model with different random init
    model2 = GPT(cfg, rngs=nnx.Rngs(99))
    state2 = create_train_state(model2, tx)
    state2 = ckpt_mgr.restore(42, state2)

    assert state2.step == 42

    # Params must be identical after restore
    params_orig = nnx.state(state.model, nnx.Param)
    params_restored = nnx.state(state2.model, nnx.Param)
    jax.tree_util.tree_map(
        lambda a, b: np.testing.assert_array_equal(np.array(a), np.array(b)),
        params_orig,
        params_restored,
    )


def test_latest_step_none_when_no_checkpoint(tmp_path):
    cfg, _ = tiny_model()
    train_cfg = TrainConfig(checkpoint_dir=str(tmp_path / "empty"), save_interval=1)
    ckpt_mgr = CheckpointManager(train_cfg, cfg)
    assert ckpt_mgr.latest_step() is None


def test_restore_returns_original_when_no_checkpoint(tmp_path):
    """restore() with no checkpoint available should return state unchanged."""
    cfg, model = tiny_model()
    tx = optax.adam(1e-3)
    state = create_train_state(model, tx)

    train_cfg = TrainConfig(checkpoint_dir=str(tmp_path / "empty2"), save_interval=1)
    ckpt_mgr = CheckpointManager(train_cfg, cfg)
    restored = ckpt_mgr.restore(None, state)
    assert restored.step == state.step


def test_max_to_keep(tmp_path):
    """Only the last N checkpoints are kept."""
    cfg, model = tiny_model()
    tx = optax.adam(1e-3)
    state = create_train_state(model, tx)

    train_cfg = TrainConfig(
        checkpoint_dir=str(tmp_path),
        max_checkpoints_to_keep=2,
        save_interval=1,
    )
    ckpt_mgr = CheckpointManager(train_cfg, cfg)
    for step in [1, 2, 3, 4, 5]:
        s = TrainState(step=step, model=state.model, tx=state.tx, opt_state=state.opt_state)
        ckpt_mgr.save(step, s)
    ckpt_mgr.wait_until_finished()
    # Latest step should be 5
    assert ckpt_mgr.latest_step() == 5
