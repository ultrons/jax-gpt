"""Load HuggingFace Qwen3.5 weights into our JAX pytree.

Handles:
- HF param name → our pytree path mapping
- Weight transpositions (HF Linear stores (out, in), we use (in, out))
- Fused gate_up_proj splitting
- Layer index → group/position mapping
- Streaming layer-by-layer load + immediate sharding to avoid OOM on
  large models: each layer is loaded as numpy, sharded to devices via
  shard_params, then stacked on-device with jnp.stack.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from jax_gpt.models.qwen35.config import Qwen35Config


# ---------------------------------------------------------------------------
# Numpy helpers (no JAX allocation — keep data on CPU)
# ---------------------------------------------------------------------------

def _to_np(x) -> np.ndarray:
    """Convert torch tensor or array to float32 numpy (CPU)."""
    if hasattr(x, 'numpy'):
        x = x.detach().float().numpy()
    return np.asarray(x, dtype=np.float32)


def _to_np_T(x) -> np.ndarray:
    """Convert and transpose last two dims: (out, in) -> (in, out)."""
    return np.transpose(_to_np(x))


# ---------------------------------------------------------------------------
# Per-layer numpy loaders
# ---------------------------------------------------------------------------

def _load_moe_params_np(sd: dict, prefix: str, config: Qwen35Config) -> dict:
    """Load MoE params for one layer as numpy arrays (CPU only)."""
    gate_weight = _to_np_T(sd[f'{prefix}.mlp.gate.weight'])

    gate_up = _to_np(sd[f'{prefix}.mlp.experts.gate_up_proj'])  # (E, 2*I, D)
    I = config.moe_intermediate_size
    gate_proj = np.transpose(gate_up[:, :I, :], (0, 2, 1))   # (E, D, I)
    up_proj   = np.transpose(gate_up[:, I:, :], (0, 2, 1))   # (E, D, I)

    down_proj = np.transpose(
        _to_np(sd[f'{prefix}.mlp.experts.down_proj']), (0, 2, 1))  # (E, I, D)

    return {
        'gate_weight':              gate_weight,
        'gate_proj':                gate_proj,
        'up_proj':                  up_proj,
        'down_proj':                down_proj,
        'shared_gate_proj':         _to_np_T(sd[f'{prefix}.mlp.shared_expert.gate_proj.weight']),
        'shared_up_proj':           _to_np_T(sd[f'{prefix}.mlp.shared_expert.up_proj.weight']),
        'shared_down_proj':         _to_np_T(sd[f'{prefix}.mlp.shared_expert.down_proj.weight']),
        'shared_expert_gate_weight':_to_np_T(sd[f'{prefix}.mlp.shared_expert_gate.weight']),
    }


def _load_delta_layer_np(sd: dict, prefix: str, config: Qwen35Config) -> dict:
    """Load one DeltaNet layer as numpy dicts (CPU only)."""
    attn = {
        'in_proj_qkv': _to_np_T(sd[f'{prefix}.linear_attn.in_proj_qkv.weight']),
        'in_proj_z':   _to_np_T(sd[f'{prefix}.linear_attn.in_proj_z.weight']),
        'in_proj_b':   _to_np_T(sd[f'{prefix}.linear_attn.in_proj_b.weight']),
        'in_proj_a':   _to_np_T(sd[f'{prefix}.linear_attn.in_proj_a.weight']),
        'conv_weight': _to_np(sd[f'{prefix}.linear_attn.conv1d.weight']).squeeze(1),
        'A_log':       _to_np(sd[f'{prefix}.linear_attn.A_log']),
        'dt_bias':     _to_np(sd[f'{prefix}.linear_attn.dt_bias']),
        'norm_weight': _to_np(sd[f'{prefix}.linear_attn.norm.weight']),
        'out_proj':    _to_np_T(sd[f'{prefix}.linear_attn.out_proj.weight']),
    }
    return {
        'attn_norm': _to_np(sd[f'{prefix}.input_layernorm.weight']),
        'attn':      attn,
        'moe_norm':  _to_np(sd[f'{prefix}.post_attention_layernorm.weight']),
        'moe':       _load_moe_params_np(sd, prefix, config),
    }


def _load_gqa_layer_np(sd: dict, prefix: str, config: Qwen35Config) -> dict:
    """Load one GQA layer as numpy dicts (CPU only)."""
    attn = {
        'q_proj': _to_np_T(sd[f'{prefix}.self_attn.q_proj.weight']),
        'k_proj': _to_np_T(sd[f'{prefix}.self_attn.k_proj.weight']),
        'v_proj': _to_np_T(sd[f'{prefix}.self_attn.v_proj.weight']),
        'o_proj': _to_np_T(sd[f'{prefix}.self_attn.o_proj.weight']),
        'q_norm': _to_np(sd[f'{prefix}.self_attn.q_norm.weight']),
        'k_norm': _to_np(sd[f'{prefix}.self_attn.k_norm.weight']),
    }
    return {
        'attn_norm': _to_np(sd[f'{prefix}.input_layernorm.weight']),
        'attn':      attn,
        'moe_norm':  _to_np(sd[f'{prefix}.post_attention_layernorm.weight']),
        'moe':       _load_moe_params_np(sd, prefix, config),
    }


# ---------------------------------------------------------------------------
# Main loader
# ---------------------------------------------------------------------------

def load_from_hf_state_dict(
    sd: dict,
    config: Qwen35Config,
    mesh=None,
    axis_rules=None,
    dtype: jnp.dtype = jnp.bfloat16,
) -> dict:
    """Convert a HuggingFace state dict to our JAX param pytree.

    Streaming layer-by-layer strategy (when mesh is provided):
      1. Load each layer as float32 numpy (CPU).
      2. Cast to dtype and call shard_params immediately → weights land on
         the correct devices; the numpy buffer is freed.
      3. Stack the already-sharded per-layer device arrays with jnp.stack.
         Peak host RAM ≈ one MoE layer (~8 GB f32) rather than the full model.

    When mesh is None (e.g. unit tests), falls back to loading everything
    as JAX arrays on the default device (original behaviour).

    Args:
        sd: HuggingFace state dict (str → torch.Tensor or numpy array).
        config: Qwen35Config matching the HF model.
        mesh: optional JAX device Mesh for immediate sharding.
        axis_rules: optional axis-rules dict (passed to shard_params).
        dtype: target dtype for all weights (default bfloat16).

    Returns:
        Nested dict matching init_params() structure, with arrays on devices.
    """
    # ---- detect layer-name prefix (multimodal vs. pure-LM checkpoint) ----
    _sample_key = next(k for k in sd if 'layers.0.' in k)
    if 'language_model' in _sample_key:
        _layer_prefix = 'model.language_model.layers'
        _embed_key    = 'model.language_model.embed_tokens.weight'
        _norm_key     = 'model.language_model.norm.weight'
    else:
        _layer_prefix = 'model.layers'
        _embed_key    = 'model.embed_tokens.weight'
        _norm_key     = 'model.norm.weight'

    interval = config.full_attention_interval

    def _place(np_params: dict) -> dict:
        """Distribute numpy params to local device shards.

        In single-process mode: puts arrays on the default device.

        In multi-process mode: uses make_array_from_process_local_inputs so
        each process only transfers its LOCAL shard — avoids the global
        allgather that device_put(NamedSharding) triggers, which OOMs when
        the full tensor exceeds HBM.
        """
        if mesh is None:
            params = jax.tree.map(jnp.asarray, np_params)
            return jax.tree.map(
                lambda x: x.astype(dtype) if jnp.issubdtype(x.dtype, jnp.floating) else x,
                params,
            )

        from jax_gpt.models.qwen35.sharding import (
            _param_logical_axes, _resolve_spec, _pad_spec_to_ndim,
            _safe_spec, AXIS_RULES_B,
        )
        from jax.sharding import NamedSharding

        rules = axis_rules if axis_rules is not None else AXIS_RULES_B
        logical_axes = _param_logical_axes(config)

        def _path_str(path_tuple):
            parts = [s for k in path_tuple
                     if not (s := str(k).strip("[]'.\"")).isdigit()]
            return '.'.join(parts)

        def _place_leaf(path_tuple, np_leaf):
            path_str = _path_str(path_tuple)
            spec = _resolve_spec(path_str, logical_axes, rules)
            spec = _pad_spec_to_ndim(spec, np_leaf.ndim)
            spec = _safe_spec(spec, np_leaf.shape, mesh)
            sharding = NamedSharding(mesh, spec)

            np_leaf_f32 = np.asarray(np_leaf, dtype=np.float32)

            def _cb(index):
                return np_leaf_f32[index].astype(dtype)

            return jax.make_array_from_callback(np_leaf.shape, sharding, _cb)

        return jax.tree_util.tree_map_with_path(_place_leaf, np_params)

    # ---- load group by group -----------------------------------------------
    group_params_list = []
    for g in range(config.n_groups):
        print(f'  Loading group {g + 1}/{config.n_groups}...', flush=True)

        # 3 DeltaNet layers — load, shard, collect
        delta_sharded = []
        for d in range(interval - 1):
            layer_idx = g * interval + d
            prefix    = f'{_layer_prefix}.{layer_idx}'
            layer_np  = _load_delta_layer_np(sd, prefix, config)
            delta_sharded.append(_place(layer_np))
            del layer_np  # free float32 RAM before next layer

        # Stack the 3 already-sharded delta arrays (on device)
        delta_stacked = jax.tree.map(
            lambda *arrs: jnp.stack(arrs, axis=0), *delta_sharded
        )
        del delta_sharded

        # 1 GQA layer
        gqa_idx    = g * interval + (interval - 1)
        gqa_prefix = f'{_layer_prefix}.{gqa_idx}'
        gqa_np     = _load_gqa_layer_np(sd, gqa_prefix, config)
        gqa_placed = _place(gqa_np)
        del gqa_np

        group_params_list.append({
            'delta_layers': delta_stacked,
            'gqa_layer':    gqa_placed,
        })

    # Stack groups on device
    print('  Stacking groups...', flush=True)
    groups = jax.tree.map(
        lambda *arrs: jnp.stack(arrs, axis=0), *group_params_list
    )
    del group_params_list

    # ---- embed / head -------------------------------------------------------
    embed      = _place({'embed': _to_np(sd[_embed_key])})['embed']
    final_norm = _place({'final_norm': _to_np(sd[_norm_key])})['final_norm']
    lm_head    = _place({'lm_head': _to_np_T(sd['lm_head.weight'])})['lm_head']

    return {
        'embed':      embed,
        'groups':     groups,
        'final_norm': final_norm,
        'lm_head':    lm_head,
    }


def _stack_tree(trees: list[dict]) -> dict:
    """Stack a list of identical-structure dicts into one dict with leading axis."""
    return jax.tree.map(lambda *arrs: jnp.stack(arrs, axis=0), *trees)
