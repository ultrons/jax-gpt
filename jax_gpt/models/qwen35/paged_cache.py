"""Paged KV cache for RPA v3 integration.

Converts the contiguous GQA KV cache to a paged layout compatible with
the Ragged Paged Attention kernel from tpu-inference.

Paged layout per GQA layer:
    (total_num_pages, page_size, kv_packed_dim, packing, head_dim)

where kv_packed_dim = align_to(n_kv_heads * 2, packing) // packing
and   packing = 32 // dtype_bits  (bf16: 2, fp8: 4)

For Qwen3.5 GQA (n_kv_heads=2, head_dim=256, bf16):
    packing = 2
    kv_packed_dim = align_to(4, 2) // 2 = 2
    → (total_pages, page_size, 2, 2, 256)
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp

from jax_gpt.models.qwen35.config import Qwen35Config

# Import RPA utilities
from tpu_inference.kernels.ragged_paged_attention.v3.kernel import (
    get_kv_cache_shape,
    merge_kv,
)
from tpu_inference.kernels.ragged_paged_attention.v3.util import (
    align_to,
    cdiv,
    get_dtype_packing,
)


@dataclass
class PagedGQACache:
    """Paged KV cache for all GQA layers + decode metadata.

    Attributes:
        kv_pages: list of per-layer paged caches, each
            (total_num_pages, page_size, kv_packed_dim, packing, head_dim)
        kv_lens: i32[max_num_seqs] — current KV length per sequence
        page_indices: i32[max_num_seqs * pages_per_seq] — page lookup table
        page_size: int — tokens per page
        pages_per_seq: int — max pages per sequence
    """
    kv_pages: list[jax.Array]
    kv_lens: jax.Array
    page_indices: jax.Array
    page_size: int
    pages_per_seq: int


def compute_page_params(
    max_len: int,
    page_size: int = 64,
) -> tuple[int, int]:
    """Compute paging parameters.

    Returns:
        (pages_per_seq, total_target_pages_per_seq)
    """
    pages_per_seq = cdiv(max_len, page_size)
    return pages_per_seq, pages_per_seq


def init_paged_cache(
    config: Qwen35Config,
    batch_size: int,
    max_len: int,
    page_size: int = 64,
    dtype: jnp.dtype = jnp.bfloat16,
) -> PagedGQACache:
    """Allocate empty paged KV caches for all GQA layers.

    Args:
        config: model config.
        batch_size: number of sequences.
        max_len: maximum sequence length.
        page_size: tokens per page.
        dtype: KV cache dtype.

    Returns:
        PagedGQACache ready for decode.
    """
    n_groups = config.n_groups
    n_kv_heads = config.gqa_n_kv_heads
    head_dim = config.gqa_head_dim
    pages_per_seq = cdiv(max_len, page_size)
    total_num_pages = batch_size * pages_per_seq

    cache_shape = get_kv_cache_shape(
        total_num_pages, page_size, n_kv_heads, head_dim, dtype,
    )

    # One paged cache per GQA layer (n_groups layers)
    kv_pages = [jnp.zeros(cache_shape, dtype=dtype) for _ in range(n_groups)]

    # Contiguous page allocation: seq i owns pages [i*pps, (i+1)*pps)
    page_indices = jnp.arange(
        batch_size * pages_per_seq, dtype=jnp.int32
    )  # already contiguous

    kv_lens = jnp.zeros((batch_size,), dtype=jnp.int32)

    return PagedGQACache(
        kv_pages=kv_pages,
        kv_lens=kv_lens,
        page_indices=page_indices,
        page_size=page_size,
        pages_per_seq=pages_per_seq,
    )


def contiguous_to_paged(
    gqa_k: jax.Array,
    gqa_v: jax.Array,
    prefill_len: int,
    page_size: int = 64,
    dtype: jnp.dtype = jnp.bfloat16,
) -> list[jax.Array]:
    """Convert contiguous GQA KV cache to paged format.

    Args:
        gqa_k: (n_groups, B, n_kv_heads, max_len, head_dim) — contiguous K cache.
        gqa_v: (n_groups, B, n_kv_heads, max_len, head_dim) — contiguous V cache.
        prefill_len: number of tokens that have been prefilled.
        page_size: tokens per page.
        dtype: target dtype for paged cache.

    Returns:
        List of n_groups paged KV caches, each:
            (total_num_pages, page_size, kv_packed_dim, packing, head_dim)
    """
    n_groups, B, n_kv_heads, max_len, head_dim = gqa_k.shape
    pages_per_seq = cdiv(max_len, page_size)
    total_num_pages = B * pages_per_seq

    cache_shape = get_kv_cache_shape(
        total_num_pages, page_size, n_kv_heads, head_dim, dtype,
    )
    packing = get_dtype_packing(dtype)
    kv_packed_dim = cache_shape[2]

    paged_caches = []
    for g in range(n_groups):
        # k_g, v_g: (B, n_kv_heads, max_len, head_dim)
        k_g = gqa_k[g]
        v_g = gqa_v[g]

        # Pad max_len to multiple of page_size
        padded_len = pages_per_seq * page_size
        if padded_len > max_len:
            pad_width = padded_len - max_len
            k_g = jnp.pad(k_g, ((0, 0), (0, 0), (0, pad_width), (0, 0)))
            v_g = jnp.pad(v_g, ((0, 0), (0, 0), (0, pad_width), (0, 0)))

        # Reshape to (B, n_kv_heads, pages_per_seq, page_size, head_dim)
        k_g = k_g.reshape(B, n_kv_heads, pages_per_seq, page_size, head_dim)
        v_g = v_g.reshape(B, n_kv_heads, pages_per_seq, page_size, head_dim)

        # Transpose to (B, pages_per_seq, page_size, n_kv_heads, head_dim)
        k_g = jnp.transpose(k_g, (0, 2, 3, 1, 4))
        v_g = jnp.transpose(v_g, (0, 2, 3, 1, 4))

        # Merge KV: interleave K and V heads
        # k_g, v_g: (B*pages_per_seq, page_size, n_kv_heads, head_dim)
        k_flat = k_g.reshape(B * pages_per_seq * page_size, n_kv_heads, head_dim)
        v_flat = v_g.reshape(B * pages_per_seq * page_size, n_kv_heads, head_dim)
        kv_merged = merge_kv(k_flat, v_flat)
        # kv_merged: (B*pps*ps, kv_packed_dim, packing, aligned_head_dim)

        # Reshape to (total_num_pages, page_size, kv_packed_dim, packing, head_dim)
        aligned_head_dim = align_to(head_dim, 128)
        paged = kv_merged.reshape(
            total_num_pages, page_size, kv_packed_dim, packing, aligned_head_dim
        )

        paged_caches.append(paged.astype(dtype))

    return paged_caches


def make_decode_metadata(
    batch_size: int,
    kv_lens: jax.Array,
    pages_per_seq: int,
) -> tuple[jax.Array, jax.Array]:
    """Build RPA metadata arrays for a pure-decode step.

    All sequences are decode-only (q_len=1).

    Args:
        batch_size: number of sequences.
        kv_lens: i32[batch_size] — current KV length per sequence.
        pages_per_seq: max pages per sequence.

    Returns:
        (cu_q_lens, distribution)
        cu_q_lens: i32[batch_size + 1] — cumulative query lengths.
        distribution: i32[3] — (decode_end, prefill_end, mixed_end).
    """
    # All decode: each sequence has q_len=1
    cu_q_lens = jnp.arange(batch_size + 1, dtype=jnp.int32)
    # All sequences are decode-only
    distribution = jnp.array([batch_size, batch_size, batch_size], dtype=jnp.int32)
    return cu_q_lens, distribution
