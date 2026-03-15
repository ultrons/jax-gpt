"""Sharding infrastructure for multi-device training.

Uses Flax NNX logical axis annotations (nnx.with_partitioning) on model
parameters, then maps logical axis names to physical mesh axes at shard time.

When annotations aren't present (e.g. model created without mesh on newer
Flax), falls back to path-based spec matching.
"""

from __future__ import annotations

import numpy as np

import jax
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from flax import nnx

from jax_gpt.trainer.config import TrainConfig
from jax_gpt.trainer.train_state import TrainState


# Logical axis name → physical mesh axis.
# Change this mapping to rearrange parallelism without touching model code.
LOGICAL_AXIS_RULES = {
    'vocab':        'tp',
    'embed':        None,
    'joined_heads': 'tp',
    'heads':        'tp',
    'mlp':          'tp',
    'context':      None,
    'layers':       None,   # reserved for PP (Phase 3)
}

# Path-based fallback specs, used when nnx.with_partitioning annotations
# aren't available (e.g. newer Flax without mesh at model creation time).
# All layer weights have a leading n_layers axis from nnx.vmap.
_FALLBACK_SPECS = {
    'wte.embedding':            PartitionSpec('tp', None),
    'wpe.embedding':            PartitionSpec(None, None),
    'h.attn.c_attn.kernel':     PartitionSpec(None, None, 'tp'),
    'h.attn.c_attn.bias':       PartitionSpec(None, 'tp'),
    'h.attn.c_proj.kernel':     PartitionSpec(None, 'tp', None),
    'h.attn.c_proj.bias':       PartitionSpec(None, None),
    'h.mlp.c_fc.kernel':        PartitionSpec(None, None, 'tp'),
    'h.mlp.c_fc.bias':          PartitionSpec(None, 'tp'),
    'h.mlp.c_proj.kernel':      PartitionSpec(None, 'tp', None),
    'h.mlp.c_proj.bias':        PartitionSpec(None, None),
    'h.ln_1.layer_norm.scale':  PartitionSpec(None, None),
    'h.ln_1.layer_norm.bias':   PartitionSpec(None, None),
    'h.ln_2.layer_norm.scale':  PartitionSpec(None, None),
    'h.ln_2.layer_norm.bias':   PartitionSpec(None, None),
    'ln_f.layer_norm.scale':    PartitionSpec(None),
    'ln_f.layer_norm.bias':     PartitionSpec(None),
    'lm_head.kernel':           PartitionSpec(None, 'tp'),
}


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


def logical_to_physical(logical_spec: PartitionSpec, ndim: int) -> PartitionSpec:
    """Map a logical PartitionSpec to physical mesh axes using LOGICAL_AXIS_RULES.

    If the array has more dimensions than the logical spec (e.g. vmap added a
    leading layers axis), prepend None for each extra dimension.
    """
    logical_axes = list(logical_spec)
    extra = ndim - len(logical_axes)
    if extra > 0:
        logical_axes = [None] * extra + logical_axes

    physical = tuple(
        LOGICAL_AXIS_RULES.get(ax, None) if ax is not None else None
        for ax in logical_axes
    )
    return PartitionSpec(*physical)


def _has_annotations(logical_specs) -> bool:
    """Check if any parameter has a non-empty PartitionSpec annotation."""
    has_any = [False]
    def check(spec):
        if isinstance(spec, PartitionSpec) and len(spec) > 0:
            has_any[0] = True
        return spec
    jax.tree_util.tree_map(check, logical_specs)
    return has_any[0]


def _path_to_dotted(path) -> str:
    """Convert a jax key path to a dotted string, stripping 'value'/'raw_value'."""
    parts = []
    for key in path:
        key_str = str(key).strip("[].'\"")
        if key_str.lower() in ('value', 'raw_value'):
            continue
        parts.append(key_str.lower())
    return '.'.join(parts)


def _match_fallback_spec(path) -> PartitionSpec | None:
    """Match a jax key path against _FALLBACK_SPECS via longest substring."""
    dotted = _path_to_dotted(path)
    best_key, best_len = None, -1
    for key in _FALLBACK_SPECS:
        if key in dotted and len(key) > best_len:
            best_key, best_len = key, len(key)
    return _FALLBACK_SPECS[best_key] if best_key is not None else None


def shard_train_state(state: TrainState, mesh: Mesh, model_config=None) -> TrainState:
    """Apply NamedSharding to all params and optimizer state.

    First tries logical axis annotations (from nnx.with_partitioning).
    Falls back to path-based spec matching if annotations aren't present
    (e.g. model created on newer Flax without a mesh).

    model_config is accepted for backward compatibility but no longer used.
    """
    params = nnx.state(state.model, nnx.Param)
    logical_specs = nnx.get_partition_spec(params)
    use_annotations = _has_annotations(logical_specs)

    if use_annotations:
        # Primary path: use logical annotations from nnx.with_partitioning
        def apply_param_sharding(leaf, spec):
            if isinstance(spec, PartitionSpec) and len(spec) > 0:
                physical_spec = logical_to_physical(spec, leaf.ndim)
                return jax.device_put(leaf, NamedSharding(mesh, physical_spec))
            return leaf

        sharded_params = jax.tree_util.tree_map(apply_param_sharding, params, logical_specs)

        param_physical_specs = {}
        def collect_specs(leaf, spec):
            if isinstance(spec, PartitionSpec) and len(spec) > 0:
                param_physical_specs[leaf.shape] = logical_to_physical(spec, leaf.ndim)
            return leaf
        jax.tree_util.tree_map(collect_specs, params, logical_specs)
    else:
        # Fallback: path-based matching (no annotations available)
        def apply_fallback_sharding(path, leaf):
            spec = _match_fallback_spec(path)
            if spec is not None:
                return jax.device_put(leaf, NamedSharding(mesh, spec))
            return leaf

        sharded_params = jax.tree_util.tree_map_with_path(apply_fallback_sharding, params)

        param_physical_specs = {}
        def collect_fallback_specs(path, leaf):
            spec = _match_fallback_spec(path)
            if spec is not None and leaf.ndim == len(spec):
                param_physical_specs[leaf.shape] = spec
            return leaf
        jax.tree_util.tree_map_with_path(collect_fallback_specs, params)

    nnx.update(state.model, sharded_params)

    # Shard optimizer state using the collected specs
    def shard_opt_leaf(leaf):
        if not isinstance(leaf, jax.Array):
            return leaf
        spec = param_physical_specs.get(leaf.shape)
        if spec is not None:
            return jax.device_put(leaf, NamedSharding(mesh, spec))
        return leaf

    sharded_opt_state = jax.tree_util.tree_map(shard_opt_leaf, state.opt_state)

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
