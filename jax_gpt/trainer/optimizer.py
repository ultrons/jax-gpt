"""Optimizer construction for GPT-2 training."""

import jax
import optax
from flax import nnx

from .config import TrainConfig


def _path_to_str(path) -> list[str]:
    """Convert a jax key path tuple to a list of lowercase string components."""
    parts = []
    for key in path:
        # DictKey, GetAttrKey, SequenceKey all have a str representation
        # Extract just the name/key portion
        key_str = str(key)
        # jax key types render as e.g. "[name]" or ".name" — strip punctuation
        key_str = key_str.strip("[].'\"")
        parts.append(key_str.lower())
    return parts


def _should_decay(path) -> bool:
    """Return True if the parameter at this path should have weight decay applied."""
    parts = _path_to_str(path)
    for part in parts:
        if 'bias' in part:
            return False
        if 'embedding' in part:
            return False
        if 'scale' in part:
            return False
    return True


def build_decay_mask(model: nnx.Module) -> any:
    """Build a pytree of bools (same structure as nnx.Param state) for weight decay."""
    params = nnx.state(model, nnx.Param)
    mask = jax.tree_util.tree_map_with_path(
        lambda path, _: _should_decay(path),
        params,
    )
    return mask


def create_optimizer(model: nnx.Module, config: TrainConfig) -> optax.GradientTransformation:
    """Create the AdamW optimizer with cosine LR schedule and per-param weight decay mask."""
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=config.learning_rate,
        warmup_steps=config.warmup_steps,
        decay_steps=config.lr_decay_steps,
        end_value=config.min_lr,
    )

    decay_mask = build_decay_mask(model)

    tx = optax.chain(
        optax.clip_by_global_norm(config.grad_clip),
        optax.scale_by_adam(b1=config.beta1, b2=config.beta2, eps=1e-8),
        optax.add_decayed_weights(config.weight_decay, mask=decay_mask),
        optax.scale_by_learning_rate(schedule),
    )
    return tx
