"""Load HuggingFace Qwen3.5 weights into our JAX pytree.

Handles:
- HF param name → our pytree path mapping
- Weight transpositions (HF Linear stores (out, in), we use (in, out))
- Fused gate_up_proj splitting
- Layer index → group/position mapping
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from jax_gpt.models.qwen35.config import Qwen35Config


def load_from_hf_state_dict(
    sd: dict,
    config: Qwen35Config,
) -> dict:
    """Convert a HuggingFace state dict to our JAX param pytree.

    Args:
        sd: HuggingFace state dict (str -> torch.Tensor or numpy array).
            Can come from model.state_dict() or safetensors.
        config: our Qwen35Config matching the HF model dimensions.

    Returns:
        Nested dict matching init_params() structure.
    """

    def _t(x):
        """Convert to JAX array, handling torch tensors."""
        if hasattr(x, 'numpy'):
            x = x.detach().float().numpy()
        return jnp.asarray(x)

    def _t_transpose(x):
        """Convert and transpose (out, in) -> (in, out)."""
        arr = _t(x)
        return jnp.transpose(arr)

    n_layers = config.n_layers
    n_groups = config.n_groups
    interval = config.full_attention_interval

    # Build per-group params
    group_params_list = []

    for g in range(n_groups):
        # 3 DeltaNet layers (indices 0,1,2 within group)
        delta_layers_list = []
        for d in range(interval - 1):
            layer_idx = g * interval + d
            prefix = f'model.layers.{layer_idx}'

            attn_params = {
                'in_proj_qkv': _t_transpose(sd[f'{prefix}.linear_attn.in_proj_qkv.weight']),
                'in_proj_z': _t_transpose(sd[f'{prefix}.linear_attn.in_proj_z.weight']),
                'in_proj_b': _t_transpose(sd[f'{prefix}.linear_attn.in_proj_b.weight']),
                'in_proj_a': _t_transpose(sd[f'{prefix}.linear_attn.in_proj_a.weight']),
                'conv_weight': _t(sd[f'{prefix}.linear_attn.conv1d.weight']).squeeze(1),  # (conv_dim, 1, K) -> (conv_dim, K)
                'A_log': _t(sd[f'{prefix}.linear_attn.A_log']),
                'dt_bias': _t(sd[f'{prefix}.linear_attn.dt_bias']),
                'norm_weight': _t(sd[f'{prefix}.linear_attn.norm.weight']),
                'out_proj': _t_transpose(sd[f'{prefix}.linear_attn.out_proj.weight']),
            }

            moe_params = _load_moe_params(sd, prefix, config)

            delta_layers_list.append({
                'attn_norm': _t(sd[f'{prefix}.input_layernorm.weight']),
                'attn': attn_params,
                'moe_norm': _t(sd[f'{prefix}.post_attention_layernorm.weight']),
                'moe': moe_params,
            })

        # 1 GQA layer (last in group)
        gqa_idx = g * interval + (interval - 1)
        gqa_prefix = f'model.layers.{gqa_idx}'

        gqa_attn_params = {
            'q_proj': _t_transpose(sd[f'{gqa_prefix}.self_attn.q_proj.weight']),
            'k_proj': _t_transpose(sd[f'{gqa_prefix}.self_attn.k_proj.weight']),
            'v_proj': _t_transpose(sd[f'{gqa_prefix}.self_attn.v_proj.weight']),
            'o_proj': _t_transpose(sd[f'{gqa_prefix}.self_attn.o_proj.weight']),
            'q_norm': _t(sd[f'{gqa_prefix}.self_attn.q_norm.weight']),
            'k_norm': _t(sd[f'{gqa_prefix}.self_attn.k_norm.weight']),
        }

        gqa_moe_params = _load_moe_params(sd, gqa_prefix, config)

        gqa_layer = {
            'attn_norm': _t(sd[f'{gqa_prefix}.input_layernorm.weight']),
            'attn': gqa_attn_params,
            'moe_norm': _t(sd[f'{gqa_prefix}.post_attention_layernorm.weight']),
            'moe': gqa_moe_params,
        }

        # Stack delta layers
        delta_layers_stacked = _stack_tree(delta_layers_list)

        group_params_list.append({
            'delta_layers': delta_layers_stacked,
            'gqa_layer': gqa_layer,
        })

    groups = _stack_tree(group_params_list)

    return {
        'embed': _t(sd['model.embed_tokens.weight']),
        'groups': groups,
        'final_norm': _t(sd['model.norm.weight']),
        'lm_head': _t_transpose(sd['lm_head.weight']),
    }


def _load_moe_params(sd: dict, prefix: str, config: Qwen35Config) -> dict:
    """Load MoE params for one layer from HF state dict."""

    def _t(x):
        if hasattr(x, 'numpy'):
            x = x.detach().float().numpy()
        return jnp.asarray(x)

    def _t_transpose(x):
        arr = _t(x)
        return jnp.transpose(arr)

    # Router: HF stores (E, D), we need (D, E)
    gate_weight = _t_transpose(sd[f'{prefix}.mlp.gate.weight'])

    # Experts: HF fuses gate+up into gate_up_proj (E, 2*I, D)
    gate_up = _t(sd[f'{prefix}.mlp.experts.gate_up_proj'])  # (E, 2*I, D)
    I = config.moe_intermediate_size
    gate_proj_hf = gate_up[:, :I, :]    # (E, I, D)
    up_proj_hf = gate_up[:, I:, :]      # (E, I, D)
    # Transpose last two dims: (E, I, D) -> (E, D, I) for ragged_dot
    gate_proj = jnp.transpose(gate_proj_hf, (0, 2, 1))
    up_proj = jnp.transpose(up_proj_hf, (0, 2, 1))

    # down_proj: HF (E, D, I) -> transpose to (E, I, D) for ragged_dot
    down_proj_hf = _t(sd[f'{prefix}.mlp.experts.down_proj'])  # (E, D, I)
    down_proj = jnp.transpose(down_proj_hf, (0, 2, 1))

    # Shared expert
    shared_gate_proj = _t_transpose(sd[f'{prefix}.mlp.shared_expert.gate_proj.weight'])
    shared_up_proj = _t_transpose(sd[f'{prefix}.mlp.shared_expert.up_proj.weight'])
    shared_down_proj = _t_transpose(sd[f'{prefix}.mlp.shared_expert.down_proj.weight'])

    # Shared expert gate
    shared_expert_gate_weight = _t_transpose(sd[f'{prefix}.mlp.shared_expert_gate.weight'])

    return {
        'gate_weight': gate_weight,
        'gate_proj': gate_proj,
        'up_proj': up_proj,
        'down_proj': down_proj,
        'shared_gate_proj': shared_gate_proj,
        'shared_up_proj': shared_up_proj,
        'shared_down_proj': shared_down_proj,
        'shared_expert_gate_weight': shared_expert_gate_weight,
    }


def _stack_tree(trees: list[dict]) -> dict:
    """Stack a list of identical-structure dicts into one dict with leading axis."""
    import jax
    return jax.tree.map(lambda *arrs: jnp.stack(arrs, axis=0), *trees)
