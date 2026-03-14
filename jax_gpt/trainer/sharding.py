"""Sharding infrastructure for multi-device training.

Uses Flax NNX logical axis annotations (nnx.with_partitioning) on model
parameters, then maps logical axis names to physical mesh axes at shard time.
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


def shard_train_state(state: TrainState, mesh: Mesh, model_config=None) -> TrainState:
    """Apply NamedSharding to all params and optimizer state.

    Uses logical axis annotations set via nnx.with_partitioning on parameters,
    maps them to physical mesh axes, and applies via jax.device_put.

    model_config is accepted for backward compatibility but no longer used.
    """
    params = nnx.state(state.model, nnx.Param)
    logical_specs = nnx.get_partition_spec(params)

    # Map logical → physical and apply sharding to each param
    def apply_param_sharding(leaf, spec):
        if isinstance(spec, PartitionSpec) and len(spec) > 0:
            physical_spec = logical_to_physical(spec, leaf.ndim)
            return jax.device_put(leaf, NamedSharding(mesh, physical_spec))
        return leaf  # no annotation or empty spec → replicate

    sharded_params = jax.tree_util.tree_map(apply_param_sharding, params, logical_specs)
    nnx.update(state.model, sharded_params)

    # Shard optimizer state: build physical spec lookup by shape from params,
    # then match optimizer leaves by shape.
    param_physical_specs = {}
    def collect_specs(leaf, spec):
        if isinstance(spec, PartitionSpec) and len(spec) > 0:
            param_physical_specs[leaf.shape] = logical_to_physical(spec, leaf.ndim)
        return leaf
    jax.tree_util.tree_map(collect_specs, params, logical_specs)

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
