"""Logical axis sharding for Qwen3.5 on TPU.

Every parameter is annotated with a PartitionSpec based on its logical role.
Physical mesh axes are assigned via AXIS_RULES dicts — swap the rules dict
to change parallelism strategy without touching model code.

Two configs provided:
  CONFIG_A: EP=8 for MoE, TP=8 for DeltaNet, TP=2 for GQA (sub-mesh)
  CONFIG_B: EP=8 for MoE, TP=8 for everything (uniform)
"""

from __future__ import annotations

from typing import Any

import numpy as np

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from jax_gpt.models.qwen35.cache import HybridCache
from jax_gpt.models.qwen35.config import Qwen35Config


# ---------------------------------------------------------------------------
# Logical axis specs per parameter path pattern
# ---------------------------------------------------------------------------
# These define what each axis of a parameter tensor *means* logically.
# The physical mapping is decided separately by the AXIS_RULES.
#
# Convention: param shapes use our (in, out) layout for standard matmuls,
# and (E, in, out) for expert weights. HF FP8 weights use (out, in).

def _param_logical_axes(config: Qwen35Config) -> dict[str, P]:
    """Return a dict mapping param path suffixes to logical PartitionSpecs.

    The keys are dotted path suffixes that uniquely identify a parameter
    in the nested pytree. Matching is by longest suffix match.
    """
    return {
        # Embeddings
        'embed':                    P('vocab', 'embed'),
        'lm_head':                  P('embed', 'vocab'),
        'final_norm':               P(None),

        # --- DeltaNet attention ---
        # in_proj_qkv: (D, key_dim*2 + value_dim) — shard output along heads
        'attn.in_proj_qkv':        P(None, 'delta_v_heads'),
        # in_proj_z: (D, value_dim) — shard along v_heads
        'attn.in_proj_z':          P(None, 'delta_v_heads'),
        # out_proj: (value_dim, D) — shard input along v_heads
        'attn.out_proj':           P('delta_v_heads', None),
        # in_proj_b, in_proj_a: (D, n_v_heads) — shard along v_heads
        'attn.in_proj_b':          P(None, 'delta_v_heads'),
        'attn.in_proj_a':          P(None, 'delta_v_heads'),
        # conv_weight: (conv_dim, kernel) — shard along conv_dim (= heads)
        'attn.conv_weight':        P('delta_v_heads', None),
        # A_log, dt_bias: (n_v_heads,) — shard along v_heads
        'attn.A_log':              P('delta_v_heads'),
        'attn.dt_bias':            P('delta_v_heads'),
        # norm_weight: (v_head_dim,) — replicated (per-head norm)
        'attn.norm_weight':        P(None),

        # --- GQA attention ---
        'attn.q_proj':             P(None, 'gqa_q_heads'),
        'attn.k_proj':             P(None, 'gqa_kv_heads'),
        'attn.v_proj':             P(None, 'gqa_kv_heads'),
        'attn.o_proj':             P('gqa_q_heads', None),
        'attn.q_norm':             P(None),
        'attn.k_norm':             P(None),

        # --- Layer norms (replicated) ---
        'attn_norm':               P(None),
        'moe_norm':                P(None),

        # --- MoE ---
        # Router: (D, n_experts) — shard along experts
        'moe.gate_weight':         P(None, 'experts'),
        # Expert weights: (E, D, I) — shard leading dim along experts
        'moe.gate_proj':           P('experts', None, None),
        'moe.up_proj':             P('experts', None, None),
        # Expert down: (E, I, D) — shard leading dim along experts
        'moe.down_proj':           P('experts', None, None),
        # Shared expert (replicated — runs on all devices)
        'moe.shared_gate_proj':    P(None, None),
        'moe.shared_up_proj':      P(None, None),
        'moe.shared_down_proj':    P(None, None),
        'moe.shared_expert_gate_weight': P(None, None),
    }


# ---------------------------------------------------------------------------
# Physical axis rule configs
# ---------------------------------------------------------------------------

# Config A: EP=8, TP=8 for DeltaNet, TP=2 for GQA
# GQA uses only 2-way sharding (pairs of devices).
# This works with a 1D mesh of 8 devices — GQA Q heads split 4-per-device
# when TP=8 is used, or we can use a 2D mesh (gqa_tp=2, rest=4).
# Simplest: use TP=8 mesh but map gqa_q_heads to None (replicate GQA Q),
# then only the output projection is sharded.
# Actually for config A, we use a 1D mesh but GQA heads map differently.
AXIS_RULES_A = {
    'vocab':           'tp',
    'embed':           None,
    'delta_v_heads':   'tp',      # 64 / 8 = 8 per device
    'delta_qk_heads':  'tp',      # 16 / 8 = 2 per device
    'gqa_q_heads':     None,      # replicate GQA Q (small — 32 heads × 64 head_dim × 2)
    'gqa_kv_heads':    None,      # replicate KV (only 2 heads)
    'experts':         'tp',      # 512 / 8 = 64 per device
}

# Config B: EP=8, TP=8 for everything (uniform)
AXIS_RULES_B = {
    'vocab':           'tp',
    'embed':           None,
    'delta_v_heads':   'tp',
    'delta_qk_heads':  'tp',
    'gqa_q_heads':     'tp',      # 32 / 8 = 4 per device
    'gqa_kv_heads':    None,      # replicate (only 2 heads, can't split 8 ways)
    'experts':         'tp',
}


def make_mesh(n_devices: int | None = None, axis_name: str = 'tp',
              dp: int = 1) -> Mesh:
    """Create a mesh over all available devices.

    Args:
        n_devices: number of devices (default: all available).
        axis_name: tensor-parallel axis name.
        dp: data-parallel factor. When dp > 1, creates a 2D mesh
            with shape (dp, n_devices // dp) and axes ('dp', axis_name).

    Returns:
        Mesh with shape (n_devices,) if dp=1, or (dp, tp) if dp>1.
    """
    devices = jax.devices()
    if n_devices is not None:
        devices = devices[:n_devices]
    if dp > 1:
        assert len(devices) % dp == 0, f"dp={dp} must divide n_devices={len(devices)}"
        tp = len(devices) // dp
        return Mesh(np.array(devices).reshape(dp, tp), ('dp', axis_name))
    return Mesh(np.array(devices), (axis_name,))


# ---------------------------------------------------------------------------
# Apply sharding to params
# ---------------------------------------------------------------------------

def _resolve_spec(
    path: str,
    logical_axes: dict[str, P],
    axis_rules: dict[str, str | None],
) -> P:
    """Resolve a param path to a physical PartitionSpec.

    1. Find the longest matching suffix in logical_axes → get logical spec
    2. Map each logical axis name through axis_rules → physical spec
    """
    # Strip FP8 quantized param suffixes so that e.g.
    # 'moe.gate_proj.w' matches logical key 'moe.gate_proj'.
    match_path = path
    is_fp8_w = False
    for suffix in ('.w', '.scale_inv'):
        if match_path.endswith(suffix):
            if suffix == '.w':
                is_fp8_w = True
            match_path = match_path[:-len(suffix)]
            break

    # Find best matching logical spec
    best_key = None
    best_len = -1
    for key in logical_axes:
        if match_path.endswith(key) and len(key) > best_len:
            best_key = key
            best_len = len(key)

    if best_key is None:
        # No match — replicate
        return P()

    logical_spec = logical_axes[best_key]

    # Map logical → physical
    physical = []
    for axis in logical_spec:
        if axis is None:
            physical.append(None)
        else:
            physical.append(axis_rules.get(axis))

    # FP8 .w params are stored in (out, in) layout while logical specs
    # assume (in, out).  Swap the last two axes so sharding targets the
    # correct dimension.  Only applies to 2-axis specs (linear weights);
    # expert weights have E as dim-0 which stays unchanged.
    if is_fp8_w and len(physical) >= 2:
        physical[-1], physical[-2] = physical[-2], physical[-1]

    return P(*physical)


def _flatten_path(path_tuple: tuple) -> str:
    """Convert a JAX pytree key path to a dotted string."""
    parts = []
    for key in path_tuple:
        s = str(key)
        # Strip JAX key wrapper chars
        for ch in "[]'.\"":
            s = s.replace(ch, '')
        if s.isdigit():
            continue  # skip numeric indices (scan axes)
        parts.append(s)
    return '.'.join(parts)


def _pad_spec_to_ndim(spec: P, ndim: int) -> P:
    """Pad a PartitionSpec with leading None dims to match array ndim.

    Stacked params (from _stack_tree for lax.scan) have leading
    (n_groups,) or (n_groups, n_delta_per_group) axes that should
    not be sharded. The logical spec describes only the trailing
    "per-param" axes.
    """
    spec_len = len(spec)
    if spec_len >= ndim:
        return spec
    padding = (None,) * (ndim - spec_len)
    return P(*padding, *spec)


def shard_params(
    params: dict,
    mesh: Mesh,
    config: Qwen35Config,
    axis_rules: dict[str, str | None] | None = None,
) -> dict:
    """Apply NamedSharding to all parameters.

    Handles stacked params (from lax.scan) by padding PartitionSpecs
    with leading None dims for the scan axes.

    Args:
        params: nested dict from init_params().
        mesh: device mesh.
        config: model config (for building logical axis map).
        axis_rules: logical→physical mapping. Defaults to AXIS_RULES_B.

    Returns:
        params with arrays placed on devices via jax.device_put.
    """
    if axis_rules is None:
        axis_rules = AXIS_RULES_B

    logical_axes = _param_logical_axes(config)
    mesh_size = mesh.devices.size

    def _shard_leaf(path, leaf):
        path_str = _flatten_path(path)
        spec = _resolve_spec(path_str, logical_axes, axis_rules)
        # Pad spec to match array ndim (leading scan axes get None)
        spec = _pad_spec_to_ndim(spec, leaf.ndim)
        # Check divisibility: if any sharded dim isn't divisible by
        # the mesh size, fall back to replicated for that dim
        safe_axes = []
        for i, axis in enumerate(spec):
            if axis is not None:
                axis_size = mesh.shape[axis]
                if leaf.shape[i] % axis_size != 0:
                    safe_axes.append(None)  # can't shard, replicate
                else:
                    safe_axes.append(axis)
            else:
                safe_axes.append(None)
        spec = P(*safe_axes)
        sharding = NamedSharding(mesh, spec)
        return jax.device_put(leaf, sharding)

    return jax.tree_util.tree_map_with_path(_shard_leaf, params)


def make_param_shardings(
    params_structure: dict,
    mesh: Mesh,
    config: Qwen35Config,
    axis_rules: dict[str, str | None] | None = None,
) -> dict:
    """Build a pytree of NamedSharding matching `params_structure`.

    Unlike `shard_params`, this does NOT call `jax.device_put`.
    It returns a pure sharding tree suitable for `jax.jit(..., out_shardings=...)`.

    Args:
        params_structure: nested dict with the same tree structure as params.
            Leaves can be real arrays or `jax.ShapeDtypeStruct`.
        mesh: device mesh.
        config: model config.
        axis_rules: logical→physical mapping. Defaults to AXIS_RULES_B.
    """
    if axis_rules is None:
        axis_rules = AXIS_RULES_B

    logical_axes = _param_logical_axes(config)

    def _make_sharding(path, leaf):
        path_str = _flatten_path(path)
        spec = _resolve_spec(path_str, logical_axes, axis_rules)
        ndim = leaf.ndim if hasattr(leaf, 'ndim') else len(leaf.shape)
        spec = _pad_spec_to_ndim(spec, ndim)
        shape = leaf.shape
        safe_axes = []
        for i, axis in enumerate(spec):
            if axis is not None:
                axis_size = mesh.shape[axis]
                if shape[i] % axis_size != 0:
                    safe_axes.append(None)
                else:
                    safe_axes.append(axis)
            else:
                safe_axes.append(None)
        return NamedSharding(mesh, P(*safe_axes))

    return jax.tree_util.tree_map_with_path(_make_sharding, params_structure)


def _safe_spec(spec: P, shape: tuple, mesh: Mesh) -> P:
    """Replace sharding axes that don't evenly divide the dim with None."""
    safe = []
    for i, axis in enumerate(spec):
        if axis is not None and shape[i] % mesh.shape[axis] != 0:
            safe.append(None)
        else:
            safe.append(axis)
    return P(*safe)


def shard_cache(
    cache: HybridCache,
    mesh: Mesh,
    config: Qwen35Config,
    axis_rules: dict[str, str | None] | None = None,
) -> HybridCache:
    """Apply NamedSharding to cache arrays.

    DeltaNet state M is sharded along heads (matches attn weight sharding).
    GQA KV cache is sharded along KV heads (or replicated if KV heads < TP).
    Dims that aren't divisible by the mesh axis size are replicated.
    """
    if axis_rules is None:
        axis_rules = AXIS_RULES_B

    tp_axis = axis_rules.get('delta_v_heads')
    gqa_kv_axis = axis_rules.get('gqa_kv_heads')
    dp_axis = 'dp' if 'dp' in mesh.axis_names else None

    # delta_M: (n_groups, 3, B, n_v_heads, qk_head_dim, v_head_dim)
    delta_M_spec = _safe_spec(
        P(None, None, dp_axis, tp_axis, None, None), cache.delta_M.shape, mesh)
    # delta_conv: (n_groups, 3, B, conv_dim, kernel)
    delta_conv_spec = _safe_spec(
        P(None, None, dp_axis, tp_axis, None), cache.delta_conv.shape, mesh)
    # gqa_k/v: (n_groups, B, n_kv_heads, max_len, head_dim)
    gqa_spec = _safe_spec(
        P(None, dp_axis, gqa_kv_axis, None, None), cache.gqa_k.shape, mesh)
    # pos: scalar
    pos_spec = P()

    return HybridCache(
        delta_M=jax.device_put(cache.delta_M, NamedSharding(mesh, delta_M_spec)),
        delta_conv=jax.device_put(cache.delta_conv, NamedSharding(mesh, delta_conv_spec)),
        gqa_k=jax.device_put(cache.gqa_k, NamedSharding(mesh, gqa_spec)),
        gqa_v=jax.device_put(cache.gqa_v, NamedSharding(mesh, gqa_spec)),
        pos=jax.device_put(cache.pos, NamedSharding(mesh, pos_spec)),
    )


def make_cache_sharding(
    config: Qwen35Config,
    mesh: Mesh,
    axis_rules: dict[str, str | None] | None = None,
    batch_size: int = 128,
) -> dict:
    """Build cache_sharding dict for forward() sharding constraints.

    Returns PartitionSpecs for per-group cache slices (without the
    leading n_groups axis, since they're used inside the lax.scan body).

    Dims that aren't divisible by the mesh axis are set to None (replicated).

    Args:
        batch_size: actual batch size, used for divisibility checks on the
            batch dimension (dp axis). Must be divisible by dp.
    """
    if axis_rules is None:
        axis_rules = AXIS_RULES_B

    tp_axis = axis_rules.get('delta_v_heads')
    gqa_kv_axis = axis_rules.get('gqa_kv_heads')
    dp_axis = 'dp' if 'dp' in mesh.axis_names else None

    # Per-group shapes (no leading n_groups dim — inside scan body):
    # delta_M: (3, B, n_v_heads, qk_head_dim, v_head_dim)
    delta_M_spec = P(None, dp_axis, tp_axis, None, None)
    # delta_conv: (3, B, conv_dim, kernel)
    delta_conv_spec = P(None, dp_axis, tp_axis, None)
    # gqa_k/v: (B, n_kv_heads, max_len, head_dim)
    gqa_spec = P(dp_axis, gqa_kv_axis, None, None)

    # Safe-check divisibility — use actual batch_size (not placeholder B=1)
    # so that dp_axis isn't incorrectly dropped when B=1 % dp != 0.
    n_delta = config.full_attention_interval - 1
    delta_M_shape = (n_delta, batch_size, config.delta_n_v_heads, config.delta_qk_head_dim, config.delta_v_head_dim)
    delta_M_spec = _safe_spec(delta_M_spec, delta_M_shape, mesh)

    key_dim = config.delta_n_qk_heads * config.delta_qk_head_dim
    value_dim = config.delta_n_v_heads * config.delta_v_head_dim
    conv_dim = key_dim * 2 + value_dim
    delta_conv_shape = (n_delta, batch_size, conv_dim, config.delta_conv_kernel)
    delta_conv_spec = _safe_spec(delta_conv_spec, delta_conv_shape, mesh)

    gqa_shape = (batch_size, config.gqa_n_kv_heads, 1, config.gqa_head_dim)
    gqa_spec = _safe_spec(gqa_spec, gqa_shape, mesh)

    # paged_kv per-group: (total_pages, page_size, kv_dim, pk, head_dim)
    # Shard total_pages on dp (pages are laid out contiguously per sequence).
    paged_kv_spec = P(dp_axis, None, None, None, None)

    return {
        'delta_M': delta_M_spec,
        'delta_conv': delta_conv_spec,
        'gqa_kv': gqa_spec,
        'paged_kv': paged_kv_spec,
    }


def get_output_shardings(
    mesh: Mesh,
    config: Qwen35Config,
    axis_rules: dict[str, str | None] | None = None,
) -> dict:
    """Get output shardings for jit-compiled forward pass.

    Returns a dict of NamedShardings for logits and cache outputs,
    useful for `jax.jit(..., out_shardings=...)`.
    """
    if axis_rules is None:
        axis_rules = AXIS_RULES_B

    vocab_axis = axis_rules.get('vocab')
    logits_spec = P(None, None, vocab_axis)  # (B, T, vocab)
    return {
        'logits': NamedSharding(mesh, logits_spec),
    }
