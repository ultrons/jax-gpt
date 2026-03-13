from dataclasses import dataclass

import optax
from flax import nnx


@dataclass
class TrainState:
    """
    Holds all mutable training state.

    The model is a live NNX module (mutable Python object). For jax.jit
    compatibility, train_step extracts params as a pure pytree via
    nnx.state(model, nnx.Param), computes gradients, applies updates, then
    writes them back with nnx.update(model, new_params).

    opt_state is a pure pytree and can be sharded / donated directly.
    """
    step: int
    model: nnx.Module
    tx: optax.GradientTransformation
    opt_state: optax.OptState


def create_train_state(model: nnx.Module, tx: optax.GradientTransformation) -> TrainState:
    """
    Initialise a TrainState from a model and an optax optimizer.

    Only nnx.Param variables are handed to the optimizer. Other variable
    types (BatchStat, dropout RNG state, etc.) are excluded automatically.
    """
    params = nnx.state(model, nnx.Param)
    opt_state = tx.init(params)
    return TrainState(step=0, model=model, tx=tx, opt_state=opt_state)
