"""Lightning-style indexer: select top-1024 positions per query (CSA path only).

Per HF blog (huggingface.co/blog/deepseekv4) and LMSYS day-0 writeup:
  - 64 heads, 128 dim per head — small dedicated attention head.
  - Query side: x @ W_iq -> (B, T, 64*128).
  - Key side: comes from the C4 (4:1-compressed) KV pool, FP4.
  - Score: ReLU(q @ k.T) summed over heads (multi-head additive, not softmax).
  - Output: top-1024 positions per query.

Compute cost at 1M context, decode B=1, M=1, one CSA layer:
  S/4 = 250K positions
  K read = 250K * 64 * 128 * 0.5 bytes (FP4) ≈ 1 GB / token / layer
  Score compute = 2 * 64 * 128 * 250K = 4 GFLOPs / token / layer
  Arithmetic intensity ≈ 4 GFLOPs / 1 GB = 4 FLOPs/byte → HBM-bound on v7x
    (HBM peak ~3.65 TB/s/TC, compute peak ~3.7 PFLOPs/s/TC bf16)
  HBM-bound time = 1 GB / 3.65 TB/s = 0.27 ms / token / layer
  Across 30 CSA layers: ~8 ms / token decode floor from the indexer alone.

This file holds the JAX-level reference. The Pallas tile-and-score kernel
lives in pallas_indexer.py.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from jax_gpt.models.dsv4.config import DSv4Config


def indexer_topk_reference(
    q: jax.Array,            # (B, T, n_heads, head_dim) — index_n_heads=64
    k_compressed: jax.Array, # (B, S_comp, n_heads, head_dim) — full C4 pool
    causal_mask: jax.Array | None,
    *,
    topk: int,
) -> tuple[jax.Array, jax.Array]:
    """Pure-JAX reference: ReLU multi-head dot product, top-k.

    Returns (top_indices, top_scores).
      top_indices: (B, T, topk) int32
      top_scores:  (B, T, topk) f32
    """
    # Per-head scores then ReLU then sum (per HF blog "ReLU-scored multi-head dot product").
    # einsum('bthd,bshd->bths', q, k_compressed)
    scores = jnp.einsum('bthd,bshd->bths', q, k_compressed)  # (B,T,n_h,S_comp)
    scores = jax.nn.relu(scores).sum(axis=2)                  # (B,T,S_comp)
    if causal_mask is not None:
        scores = jnp.where(causal_mask, scores, jnp.full_like(scores, -jnp.inf))
    # top-k along the S_comp axis.
    top_scores, top_indices = jax.lax.top_k(scores, topk)
    return top_indices.astype(jnp.int32), top_scores


def select_topk(
    q_input: jax.Array,
    k_compressed: jax.Array,
    params: dict,
    cfg: DSv4Config,
) -> tuple[jax.Array, jax.Array]:
    """Top-level indexer entrypoint — projects, RoPEs (optional), scores, top-k.

    STUB. Will swap in the Pallas tile-and-score kernel once landed.
    """
    raise NotImplementedError('See indexer_topk_reference + pallas_indexer.py')
