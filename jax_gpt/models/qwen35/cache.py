"""Cache structures for hybrid attention (DeltaNet + GQA).

DeltaNet layers use a fixed-size recurrent state (constant memory).
GQA layers use a traditional KV cache (grows with sequence length).

All arrays have a leading n_groups axis for compatibility with lax.scan
over the outer group dimension.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from jax_gpt.models.qwen35.config import Qwen35Config


@dataclass
class HybridCache:
    """Full model cache for all groups.

    DeltaNet state:
        delta_M:     (n_groups, 3, B, n_qk_heads, qk_head_dim, v_head_dim)
            Recurrent state matrix per DeltaNet layer.
        delta_conv:  (n_groups, 3, B, d_model, conv_kernel)
            Causal conv1d state per DeltaNet layer.

    GQA cache (contiguous — used for prefill and non-RPA decode):
        gqa_k:       (n_groups, B, n_kv_heads, max_len, head_dim)
        gqa_v:       (n_groups, B, n_kv_heads, max_len, head_dim)

    GQA cache (paged — used for RPA decode):
        paged_kv:    (n_groups, total_pages, page_size, kv_packed_dim, packing, hd)
            Stacked paged KV caches for all groups. None when not using RPA.
        kv_lens:     i32[B] — current KV length per sequence. None when not using RPA.
        page_indices: i32[B * pages_per_seq] — page lookup table. None when not using RPA.

    pos: scalar int — current sequence position.
    """
    delta_M: jax.Array
    delta_conv: jax.Array
    gqa_k: jax.Array
    gqa_v: jax.Array
    pos: jax.Array  # scalar int32
    paged_kv: jax.Array | None = None
    kv_lens: jax.Array | None = None
    page_indices: jax.Array | None = None


# Register as pytree so lax.scan can carry it
jax.tree_util.register_dataclass(
    HybridCache,
    data_fields=['delta_M', 'delta_conv', 'gqa_k', 'gqa_v', 'pos',
                 'paged_kv', 'kv_lens', 'page_indices'],
    meta_fields=[],
)


def init_cache(
    config: Qwen35Config,
    batch_size: int,
    max_len: int | None = None,
    dtype: jnp.dtype = jnp.float32,
) -> HybridCache:
    """Allocate an empty HybridCache.

    Args:
        config: model config.
        batch_size: batch dimension.
        max_len: max sequence length for GQA KV cache. Defaults to
            config.max_position_embeddings.
        dtype: array dtype.
    Returns:
        Zeroed HybridCache ready for inference.
    """
    if max_len is None:
        max_len = config.max_position_embeddings
    n_groups = config.n_groups
    n_delta_per_group = config.full_attention_interval - 1  # 3
    key_dim = config.delta_n_qk_heads * config.delta_qk_head_dim
    value_dim = config.delta_n_v_heads * config.delta_v_head_dim
    conv_dim = key_dim * 2 + value_dim

    delta_M = jnp.zeros(
        (n_groups, n_delta_per_group, batch_size,
         config.delta_n_v_heads, config.delta_qk_head_dim, config.delta_v_head_dim),
        dtype=dtype,
    )
    delta_conv = jnp.zeros(
        (n_groups, n_delta_per_group, batch_size,
         conv_dim, config.delta_conv_kernel),
        dtype=dtype,
    )
    gqa_k = jnp.zeros(
        (n_groups, batch_size, config.gqa_n_kv_heads, max_len, config.gqa_head_dim),
        dtype=dtype,
    )
    gqa_v = jnp.zeros(
        (n_groups, batch_size, config.gqa_n_kv_heads, max_len, config.gqa_head_dim),
        dtype=dtype,
    )
    pos = jnp.array(0, dtype=jnp.int32)

    return HybridCache(
        delta_M=delta_M,
        delta_conv=delta_conv,
        gqa_k=gqa_k,
        gqa_v=gqa_v,
        pos=pos,
    )
