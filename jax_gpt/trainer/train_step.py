"""
train_step.py — Training and evaluation step functions.

Design note on JIT in flax NNX 0.8.x
--------------------------------------
jax.jit cannot trace NNX modules directly (they are not JAX pytrees).
nnx.jit *can*, but it requires the NNX module to be a direct positional arg.

We therefore expose two layers:
  - train_step / eval_step: plain Python functions (no JIT).
    Call these from scripts/train.py wrapped with nnx.jit if desired.
  - jitted_train_step / jitted_eval_step: nnx.jit-wrapped variants,
    ready to use directly when performance matters.

The NNX functional split pattern used here:
  1. nnx.state(model, nnx.Param)   → extract params as a JAX pytree
  2. jax.value_and_grad(loss_fn)   → differentiate w.r.t. params only
  3. nnx.update(model, new_params) → write updated params back into model
"""
from typing import Any

import jax
import jax.numpy as jnp
import optax
from flax import nnx

from jax_gpt.trainer.train_state import TrainState


# ---------------------------------------------------------------------------
# Core step — no JIT (wrap with nnx.jit externally for performance)
# ---------------------------------------------------------------------------

def train_step(
    state: TrainState,
    batch: tuple[jax.Array, jax.Array],
    mesh=None,
) -> tuple[TrainState, dict[str, jax.Array]]:
    """
    Single gradient update.

    Args:
        state: current TrainState
        batch: (x, y) token index arrays, each shape (B, T)
        mesh: optional jax.sharding.Mesh; None = single device

    Returns:
        (new_state, metrics) where metrics contains 'loss' and 'grad_norm'
    """
    model = state.model
    x, y = batch

    def loss_fn(params: Any) -> jax.Array:
        nnx.update(model, params)
        logits, _ = model(x, deterministic=False, mesh=mesh)
        logits_f32 = logits.astype(jnp.float32)
        return optax.softmax_cross_entropy_with_integer_labels(
            logits_f32.reshape(-1, logits_f32.shape[-1]),
            y.reshape(-1),
        ).mean()

    params = nnx.state(model, nnx.Param)
    loss, grads = jax.value_and_grad(loss_fn)(params)
    grad_norm = optax.global_norm(grads)

    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)
    nnx.update(model, new_params)

    new_state = TrainState(
        step=state.step + 1,
        model=model,
        tx=state.tx,
        opt_state=new_opt_state,
    )
    return new_state, {'loss': loss, 'grad_norm': grad_norm}


def eval_step(
    model: nnx.Module,
    batch: tuple[jax.Array, jax.Array],
) -> jax.Array:
    """Compute cross-entropy loss without updating any state."""
    x, y = batch
    logits, _ = model(x, deterministic=True)
    logits_f32 = logits.astype(jnp.float32)
    return optax.softmax_cross_entropy_with_integer_labels(
        logits_f32.reshape(-1, logits_f32.shape[-1]),
        y.reshape(-1),
    ).mean()


# ---------------------------------------------------------------------------
# JIT-wrapped variants for production use
# ---------------------------------------------------------------------------

@nnx.jit
def jitted_eval_step(
    model: nnx.Module,
    batch: tuple[jax.Array, jax.Array],
) -> jax.Array:
    """JIT-compiled eval step. model must be an NNX module (direct arg for nnx.jit)."""
    return eval_step(model, batch)


# ---------------------------------------------------------------------------
# Gradient accumulation wrapper
# ---------------------------------------------------------------------------

def train_step_with_accumulation(
    state: TrainState,
    micro_batches: list[tuple[jax.Array, jax.Array]],
    mesh=None,
) -> tuple[TrainState, dict[str, jax.Array]]:
    """
    Accumulate gradients over multiple micro-batches before applying.

    The accumulation loop runs in Python (outside any JIT) to avoid
    recompilation when grad_accum_steps changes.

    For grad_accum_steps=1 this is equivalent to calling train_step directly.
    """
    if len(micro_batches) == 1:
        return train_step(state, micro_batches[0], mesh)

    model = state.model

    def loss_fn_for_batch(params: Any, batch: tuple) -> jax.Array:
        nnx.update(model, params)
        x, y = batch
        logits, _ = model(x, deterministic=False, mesh=mesh)
        logits_f32 = logits.astype(jnp.float32)
        return optax.softmax_cross_entropy_with_integer_labels(
            logits_f32.reshape(-1, logits_f32.shape[-1]),
            y.reshape(-1),
        ).mean()

    params = nnx.state(model, nnx.Param)
    zero_grads = jax.tree_util.tree_map(jnp.zeros_like, params)

    total_loss = jnp.zeros(())
    accumulated_grads = zero_grads
    for batch in micro_batches:
        loss, grads = jax.value_and_grad(loss_fn_for_batch)(params, batch)
        accumulated_grads = jax.tree_util.tree_map(
            lambda a, g: a + g, accumulated_grads, grads
        )
        total_loss = total_loss + loss

    scale = 1.0 / len(micro_batches)
    averaged_grads = jax.tree_util.tree_map(lambda g: g * scale, accumulated_grads)
    avg_loss = total_loss * scale
    grad_norm = optax.global_norm(averaged_grads)

    updates, new_opt_state = state.tx.update(averaged_grads, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)
    nnx.update(model, new_params)

    new_state = TrainState(
        step=state.step + 1,
        model=model,
        tx=state.tx,
        opt_state=new_opt_state,
    )
    return new_state, {'loss': avg_loss, 'grad_norm': grad_norm}
