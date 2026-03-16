"""Qwen3.5-style model: pure JAX implementation.

Full model: embedding -> lax.scan over groups -> final norm -> lm_head.
All functions are pure — params are nested dicts, cache is a pytree.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from jax.sharding import PartitionSpec as P

from jax_gpt.models.qwen35.fp8 import matmul_maybe_fp8
from jax_gpt.models.qwen35.block import group_forward
from jax_gpt.models.qwen35.cache import HybridCache, init_cache
from jax_gpt.models.qwen35.config import Qwen35Config
from jax_gpt.models.qwen35.primitives import precompute_rope_freqs, rms_norm


# ---------------------------------------------------------------------------
# Parameter initialization
# ---------------------------------------------------------------------------

def _init_linear(key: jax.Array, in_dim: int, out_dim: int, scale: float = 0.02,
                  dtype: jnp.dtype = jnp.float32, fp8: bool = False):
    """Init a linear weight. If fp8=True, returns {'w': fp8, 'scale_inv': f32} dict."""
    if fp8:
        from jax_gpt.models.qwen35.fp8 import FP8_DTYPE
        # Generate in bf16, transpose to (out, in) for fp8_matmul, cast to fp8
        w = jax.random.normal(key, (out_dim, in_dim), dtype=jnp.bfloat16) * scale
        w_fp8 = w.astype(FP8_DTYPE)
        # scale_inv = 1.0 for random weights (values already in fp8 range)
        scale_inv = jnp.ones((out_dim, 1), dtype=jnp.float32)
        return {'w': w_fp8, 'scale_inv': scale_inv}
    return jax.random.normal(key, (in_dim, out_dim), dtype=dtype) * scale


def _init_deltanet_attn_params(key: jax.Array, config: Qwen35Config,
                               dtype: jnp.dtype = jnp.float32, fp8: bool = False) -> dict:
    """Initialize one DeltaNet attention sub-layer's params."""
    keys = jax.random.split(key, 10)
    D = config.d_model
    key_dim = config.delta_n_qk_heads * config.delta_qk_head_dim
    value_dim = config.delta_n_v_heads * config.delta_v_head_dim
    conv_dim = key_dim * 2 + value_dim

    return {
        'in_proj_qkv': _init_linear(keys[0], D, conv_dim, dtype=dtype, fp8=fp8),
        'in_proj_z': _init_linear(keys[1], D, value_dim, dtype=dtype, fp8=fp8),
        'in_proj_b': _init_linear(keys[2], D, config.delta_n_v_heads, dtype=dtype, fp8=fp8),
        'in_proj_a': _init_linear(keys[3], D, config.delta_n_v_heads, dtype=dtype, fp8=fp8),
        'conv_weight': (jax.random.normal(keys[4], (conv_dim, config.delta_conv_kernel), dtype=dtype) * 0.02),
        'A_log': jnp.log(jax.random.uniform(keys[5], (config.delta_n_v_heads,), minval=0.1, maxval=16.0)),
        'dt_bias': jnp.ones(config.delta_n_v_heads, dtype=dtype),
        'norm_weight': jnp.zeros(config.delta_v_head_dim, dtype=dtype),
        'out_proj': _init_linear(keys[6], value_dim, D, dtype=dtype, fp8=fp8),
    }


def _init_gqa_attn_params(key: jax.Array, config: Qwen35Config,
                          dtype: jnp.dtype = jnp.float32, fp8: bool = False) -> dict:
    """Initialize one GQA attention sub-layer's params."""
    keys = jax.random.split(key, 5)
    D = config.d_model
    q_dim = config.gqa_n_q_heads * config.gqa_head_dim
    kv_dim = config.gqa_n_kv_heads * config.gqa_head_dim

    return {
        'q_proj': _init_linear(keys[0], D, q_dim * 2, dtype=dtype, fp8=fp8),
        'k_proj': _init_linear(keys[1], D, kv_dim, dtype=dtype, fp8=fp8),
        'v_proj': _init_linear(keys[2], D, kv_dim, dtype=dtype, fp8=fp8),
        'o_proj': _init_linear(keys[3], q_dim, D, dtype=dtype, fp8=fp8),
        'q_norm': jnp.zeros(config.gqa_head_dim, dtype=dtype),
        'k_norm': jnp.zeros(config.gqa_head_dim, dtype=dtype),
    }


def _init_moe_params(key: jax.Array, config: Qwen35Config,
                     dtype: jnp.dtype = jnp.float32, fp8: bool = False) -> dict:
    """Initialize one MoE layer's params."""
    keys = jax.random.split(key, 8)
    D = config.d_model
    E = config.n_routed_experts
    I = config.moe_intermediate_size
    SI = config.shared_expert_intermediate_size

    # Expert weights: 3D (E, D, I) — for fp8, quantize with transposed last two dims
    if fp8:
        from jax_gpt.models.qwen35.fp8 import FP8_DTYPE
        def _init_expert_fp8(k, shape):
            # Generate in (E, out, in) layout for fp8, with scale_inv
            E_, out_, in_ = shape[0], shape[2], shape[1]  # transpose D,I -> I,D or D,I -> I,D
            w = jax.random.normal(k, (E_, out_, in_), dtype=jnp.bfloat16) * 0.02
            return {'w': w.astype(FP8_DTYPE),
                    'scale_inv': jnp.ones((E_, out_, 1), dtype=jnp.float32)}
        gate_proj = _init_expert_fp8(keys[1], (E, D, I))
        up_proj = _init_expert_fp8(keys[2], (E, D, I))
        down_proj = _init_expert_fp8(keys[3], (E, I, D))
    else:
        gate_proj = jax.random.normal(keys[1], (E, D, I), dtype=dtype) * 0.02
        up_proj = jax.random.normal(keys[2], (E, D, I), dtype=dtype) * 0.02
        down_proj = jax.random.normal(keys[3], (E, I, D), dtype=dtype) * 0.02

    return {
        'gate_weight': _init_linear(keys[0], D, E, dtype=dtype, fp8=fp8),
        'gate_proj': gate_proj,
        'up_proj': up_proj,
        'down_proj': down_proj,
        'shared_gate_proj': _init_linear(keys[4], D, SI, dtype=dtype, fp8=fp8),
        'shared_up_proj': _init_linear(keys[5], D, SI, dtype=dtype, fp8=fp8),
        'shared_down_proj': _init_linear(keys[6], SI, D, dtype=dtype, fp8=fp8),
        'shared_expert_gate_weight': _init_linear(keys[7], D, 1, dtype=dtype, fp8=fp8),
    }


def _init_delta_layer_params(key: jax.Array, config: Qwen35Config,
                             dtype: jnp.dtype = jnp.float32, fp8: bool = False) -> dict:
    """Initialize one DeltaNet layer (attn_norm + attn + moe_norm + moe)."""
    k1, k2 = jax.random.split(key)
    return {
        'attn_norm': jnp.zeros(config.d_model, dtype=dtype),
        'attn': _init_deltanet_attn_params(k1, config, dtype, fp8),
        'moe_norm': jnp.zeros(config.d_model, dtype=dtype),
        'moe': _init_moe_params(k2, config, dtype, fp8),
    }


def _init_gqa_layer_params(key: jax.Array, config: Qwen35Config,
                           dtype: jnp.dtype = jnp.float32, fp8: bool = False) -> dict:
    """Initialize one GQA layer (attn_norm + attn + moe_norm + moe)."""
    k1, k2 = jax.random.split(key)
    return {
        'attn_norm': jnp.zeros(config.d_model, dtype=dtype),
        'attn': _init_gqa_attn_params(k1, config, dtype, fp8),
        'moe_norm': jnp.zeros(config.d_model, dtype=dtype),
        'moe': _init_moe_params(k2, config, dtype, fp8),
    }


def _stack_tree(trees: list[dict]) -> dict:
    """Stack a list of identical-structure param dicts into one dict with
    leading axis. E.g. [{a: (D,), b: (D, D)}, ...] -> {a: (N, D), b: (N, D, D)}."""
    return jax.tree.map(lambda *arrs: jnp.stack(arrs, axis=0), *trees)


def init_params(config: Qwen35Config, key: jax.Array, dtype: jnp.dtype = jnp.float32, fp8: bool = False) -> dict:
    """Initialize all model parameters as a nested dict pytree.

    Args:
        config: model config.
        key: PRNG key.
        dtype: parameter dtype (use jnp.bfloat16 for large models to save memory).

    Structure:
        embed: (vocab_size, d_model)
        groups: stacked group params with leading n_groups axis
            delta_layers: stacked (3 per group) DeltaNet layer params
            gqa_layer: GQA layer params (1 per group)
        final_norm: (d_model,)
        lm_head: (d_model, vocab_size)
    """
    keys = jax.random.split(key, 3 + config.n_groups * 2)
    key_idx = 0

    # Embedding
    embed = jax.random.normal(keys[key_idx], (config.vocab_size, config.d_model), dtype=dtype) * 0.02
    key_idx += 1

    # Groups
    group_params_list = []
    for g in range(config.n_groups):
        # 3 DeltaNet layers
        delta_keys = jax.random.split(keys[key_idx], 3)
        key_idx += 1
        delta_layers = _stack_tree([
            _init_delta_layer_params(delta_keys[i], config, dtype, fp8) for i in range(3)
        ])

        # 1 GQA layer
        gqa_layer = _init_gqa_layer_params(keys[key_idx], config, dtype, fp8)
        key_idx += 1

        group_params_list.append({
            'delta_layers': delta_layers,
            'gqa_layer': gqa_layer,
        })

    groups = _stack_tree(group_params_list)

    # Final norm + lm_head (embed stays as regular array for lookup; lm_head can be fp8)
    final_norm = jnp.zeros(config.d_model, dtype=dtype)
    lm_head = _init_linear(keys[key_idx], config.d_model, config.vocab_size, dtype=dtype, fp8=fp8)

    return {
        'embed': embed,
        'groups': groups,
        'final_norm': final_norm,
        'lm_head': lm_head,
    }


# ---------------------------------------------------------------------------
# Forward pass
# ---------------------------------------------------------------------------

def forward(
    params: dict,
    tokens: jax.Array,
    config: Qwen35Config,
    cache: HybridCache | None = None,
    is_decode: bool = False,
    cache_sharding: dict | None = None,
) -> tuple[jax.Array, HybridCache | None]:
    """Full model forward pass.

    Args:
        params: nested dict pytree from init_params.
        tokens: (B, T) int32 token ids.
        config: model config.
        cache: HybridCache or None.
        is_decode: True for single-token decode mode.
        cache_sharding: optional dict with PartitionSpecs for cache arrays.
            Keys: 'delta_M', 'delta_conv', 'gqa_kv'. When provided,
            jax.lax.with_sharding_constraint is applied to cache outputs
            inside the scan to prevent XLA from inferring incompatible sharding.

    Returns:
        (logits, updated_cache)
        logits: (B, T, vocab_size)
    """
    B, T = tokens.shape

    with jax.named_scope('embedding'):
        x = params['embed'][tokens]  # (B, T, D)

    # Precompute RoPE frequencies
    rope_freqs = precompute_rope_freqs(
        config.gqa_rope_dim,
        config.max_position_embeddings,
        config.gqa_rope_theta,
    )

    # Prepare cache slices for scan
    if cache is not None:
        cache_pos = cache.pos
        delta_Ms = cache.delta_M        # (n_groups, 3, B, ...)
        delta_convs = cache.delta_conv   # (n_groups, 3, B, ...)
        gqa_ks = cache.gqa_k             # (n_groups, B, ...)
        gqa_vs = cache.gqa_v             # (n_groups, B, ...)
    else:
        cache_pos = None
        delta_Ms = None
        delta_convs = None
        gqa_ks = None
        gqa_vs = None

    # Scan over groups
    def _group_step(carry, group_inputs):
        x_carry = carry
        g_params, g_delta_M, g_delta_conv, g_gqa_k, g_gqa_v = group_inputs

        x_carry, new_dM, new_dC, new_gk, new_gv = group_forward(
            x_carry, g_params,
            g_delta_M, g_delta_conv,
            g_gqa_k, g_gqa_v,
            cache_pos, config, rope_freqs, is_decode,
        )

        # Apply sharding constraints on cache outputs to prevent XLA
        # from inferring incompatible sharding (e.g. n_kv_heads=2
        # can't be split across TP=8).
        if cache_sharding is not None:
            new_dM = jax.lax.with_sharding_constraint(new_dM, cache_sharding['delta_M'])
            new_dC = jax.lax.with_sharding_constraint(new_dC, cache_sharding['delta_conv'])
            new_gk = jax.lax.with_sharding_constraint(new_gk, cache_sharding['gqa_kv'])
            new_gv = jax.lax.with_sharding_constraint(new_gv, cache_sharding['gqa_kv'])

        return x_carry, (new_dM, new_dC, new_gk, new_gv)

    if cache is not None:
        scan_inputs = (
            params['groups'], delta_Ms, delta_convs, gqa_ks, gqa_vs,
        )
        x, (new_dMs, new_dConvs, new_gKs, new_gVs) = jax.lax.scan(
            _group_step, x, scan_inputs,
        )
        new_cache = HybridCache(
            delta_M=new_dMs,
            delta_conv=new_dConvs,
            gqa_k=new_gKs,
            gqa_v=new_gVs,
            pos=cache_pos + T,
        )
    else:
        # No cache: still need dummy arrays for scan signature consistency
        n_groups = config.n_groups
        n_delta = config.full_attention_interval - 1
        key_dim = config.delta_n_qk_heads * config.delta_qk_head_dim
        value_dim = config.delta_n_v_heads * config.delta_v_head_dim
        conv_dim = key_dim * 2 + value_dim

        dummy_dM = jnp.zeros((n_groups, n_delta, B,
                              config.delta_n_v_heads, config.delta_qk_head_dim, config.delta_v_head_dim))
        dummy_dC = jnp.zeros((n_groups, n_delta, B, conv_dim, config.delta_conv_kernel))
        dummy_gK = jnp.zeros((n_groups, B, config.gqa_n_kv_heads, T, config.gqa_head_dim))
        dummy_gV = jnp.zeros((n_groups, B, config.gqa_n_kv_heads, T, config.gqa_head_dim))

        scan_inputs = (
            params['groups'], dummy_dM, dummy_dC, dummy_gK, dummy_gV,
        )
        x, _ = jax.lax.scan(_group_step, x, scan_inputs)
        new_cache = None

    with jax.named_scope('output_head'):
        x = rms_norm(x, params['final_norm'], config.rms_norm_eps)
        logits = matmul_maybe_fp8(x, params['lm_head'])  # (B, T, vocab_size)

    return logits, new_cache


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate(
    params: dict,
    prompt_tokens: jax.Array,
    config: Qwen35Config,
    max_new_tokens: int,
    max_seq_len: int | None = None,
    temperature: float = 1.0,
    top_k: int | None = None,
    key: jax.Array | None = None,
) -> jax.Array:
    """Autoregressive generation with prefill + lax.scan decode loop.

    Args:
        params: model parameters.
        prompt_tokens: (B, T) prompt token ids.
        config: model config.
        max_new_tokens: number of new tokens to generate.
        max_seq_len: max sequence length for KV cache.
        temperature: sampling temperature.
        top_k: if set, sample from top-k logits.
        key: PRNG key for sampling.

    Returns:
        (B, max_new_tokens) generated token ids.
    """
    if key is None:
        key = jax.random.key(0)
    if max_seq_len is None:
        max_seq_len = prompt_tokens.shape[1] + max_new_tokens

    B = prompt_tokens.shape[0]

    # Initialize cache
    cache = init_cache(config, B, max_seq_len)

    # Prefill
    logits, cache = forward(params, prompt_tokens, config, cache=cache, is_decode=False)

    # Sample first token from last position
    first_logits = logits[:, -1, :]  # (B, vocab)
    key, subkey = jax.random.split(key)
    first_token = _sample(first_logits, temperature, top_k, subkey)  # (B,)

    # Decode loop via lax.scan
    def _decode_step(carry, _):
        token, cache_carry, rng = carry
        token_input = token[:, None]  # (B, 1)
        logits, new_cache = forward(params, token_input, config, cache=cache_carry, is_decode=True)
        next_logits = logits[:, 0, :]  # (B, vocab)
        rng, subkey = jax.random.split(rng)
        next_token = _sample(next_logits, temperature, top_k, subkey)
        return (next_token, new_cache, rng), next_token

    init_carry = (first_token, cache, key)
    _, generated = jax.lax.scan(_decode_step, init_carry, None, length=max_new_tokens - 1)

    # generated: (max_new_tokens-1, B) -> (B, max_new_tokens)
    all_tokens = jnp.concatenate([first_token[:, None], generated.T], axis=1)
    return all_tokens


def _sample(
    logits: jax.Array,
    temperature: float,
    top_k: int | None,
    key: jax.Array,
) -> jax.Array:
    """Sample from logits with temperature and optional top-k."""
    if temperature <= 0:
        return jnp.argmax(logits, axis=-1)

    logits = logits / temperature

    if top_k is not None:
        top_k_vals, _ = jax.lax.top_k(logits, top_k)
        threshold = top_k_vals[:, -1:]
        logits = jnp.where(logits >= threshold, logits, jnp.finfo(logits.dtype).min)

    return jax.random.categorical(key, logits, axis=-1)
