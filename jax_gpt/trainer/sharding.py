"""Sharding infrastructure for multi-device training."""

import numpy as np

import jax
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from flax import nnx

from jax_gpt.models.gpt2.config import GPTConfig
from jax_gpt.trainer.config import TrainConfig
from jax_gpt.trainer.train_state import TrainState


def make_mesh(config: TrainConfig) -> Mesh:
    """Create a 4-axis ('dp','fsdp','tp','sp') device mesh."""
    total = config.dp * config.fsdp * config.tp * config.sp
    assert total == jax.device_count(), (
        f"dp*fsdp*tp*sp={total} != jax.device_count()={jax.device_count()}"
    )
    devices = np.array(jax.devices()).reshape(
        config.dp, config.fsdp, config.tp, config.sp
    )
    # Phase 3 hook: if config.pp > 1 this would become a 5D mesh ('pp','dp','fsdp','tp','sp')
    assert config.pp == 1, "Pipeline parallelism (pp>1) is not yet implemented (Phase 3)"
    return Mesh(devices, ('dp', 'fsdp', 'tp', 'sp'))


def get_param_sharding_specs(model_config: GPTConfig) -> dict:
    """Return a dict mapping dotted path substrings to PartitionSpec.

    All layer weights have a leading n_layers axis from nnx.vmap, so a kernel
    of shape (n_layers, in, out) has axes [layers, in_features, out_features].
    """
    specs = {
        # Embeddings
        'wte.embedding':            PartitionSpec('tp', None),
        'wpe.embedding':            PartitionSpec(None, None),

        # Attention — layers axis is first (from vmap)
        'h.attn.c_attn.kernel':     PartitionSpec(None, None, 'tp'),
        'h.attn.c_attn.bias':       PartitionSpec(None, 'tp'),
        'h.attn.c_proj.kernel':     PartitionSpec(None, 'tp', None),
        'h.attn.c_proj.bias':       PartitionSpec(None, None),

        # MLP
        'h.mlp.c_fc.kernel':        PartitionSpec(None, None, 'tp'),
        'h.mlp.c_fc.bias':          PartitionSpec(None, 'tp'),
        'h.mlp.c_proj.kernel':      PartitionSpec(None, 'tp', None),
        'h.mlp.c_proj.bias':        PartitionSpec(None, None),

        # LayerNorm (replicated — small)
        'h.ln_1.layer_norm.scale':  PartitionSpec(None, None),
        'h.ln_1.layer_norm.bias':   PartitionSpec(None, None),
        'h.ln_2.layer_norm.scale':  PartitionSpec(None, None),
        'h.ln_2.layer_norm.bias':   PartitionSpec(None, None),

        # Final LayerNorm
        'ln_f.layer_norm.scale':    PartitionSpec(None),
        'ln_f.layer_norm.bias':     PartitionSpec(None),

        # lm_head
        'lm_head.kernel':           PartitionSpec('tp', None),
    }
    return specs


def _path_to_dotted(path) -> str:
    """Convert a jax key path tuple to a dotted lowercase string, stripping 'value' leaves."""
    parts = []
    for key in path:
        key_str = str(key).strip("[].'\"")
        if key_str.lower() == 'value':
            continue
        parts.append(key_str.lower())
    return '.'.join(parts)


def _match_spec(path, specs: dict):
    """Match a jax key path against the spec dict via substring matching."""
    dotted = _path_to_dotted(path)
    # Use longest matching key to avoid ambiguity (e.g. 'bias' vs 'c_attn.bias')
    best_key = None
    best_len = -1
    for key in specs:
        if key in dotted and len(key) > best_len:
            best_key = key
            best_len = len(key)
    if best_key is not None:
        return specs[best_key]
    return None


def _shard_opt_leaf(path, leaf, specs: dict, mesh: Mesh):
    """Apply sharding to an optimizer state leaf using same spec as corresponding param."""
    if not isinstance(leaf, jax.Array):
        return leaf
    spec = _match_spec(path, specs)
    if spec is None:
        return leaf
    # Optimizer state tensors may have an extra leading dimension (e.g. Adam count scalar).
    # Only apply when the ndim matches; fall back to full replication otherwise.
    if leaf.ndim == len(spec):
        return jax.device_put(leaf, NamedSharding(mesh, spec))
    return leaf


def shard_train_state(state: TrainState, mesh: Mesh, model_config: GPTConfig) -> TrainState:
    """Apply NamedSharding to all params and optimizer state in the train state."""
    specs = get_param_sharding_specs(model_config)

    # Extract params pytree
    params = nnx.state(state.model, nnx.Param)

    # Apply sharding: walk pytree with paths, look up matching spec
    def apply_sharding(path, leaf):
        spec = _match_spec(path, specs)
        if spec is not None:
            return jax.device_put(leaf, NamedSharding(mesh, spec))
        return leaf  # replicate by default

    sharded_params = jax.tree_util.tree_map_with_path(apply_sharding, params)
    nnx.update(state.model, sharded_params)

    # Shard optimizer state with same layout
    sharded_opt_state = jax.tree_util.tree_map_with_path(
        lambda path, leaf: _shard_opt_leaf(path, leaf, specs, mesh),
        state.opt_state,
    )

    return TrainState(
        step=state.step,
        model=state.model,
        tx=state.tx,
        opt_state=sharded_opt_state,
    )


def activation_sharding_constraint(x: jax.Array, spec: PartitionSpec, mesh) -> jax.Array:
    """No-op if mesh is None. Applies sharding constraint for SP/TP activations."""
    if mesh is None:
        return x
    return jax.lax.with_sharding_constraint(x, NamedSharding(mesh, spec))
