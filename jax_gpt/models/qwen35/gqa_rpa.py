"""GQA attention using the RPA v3 (Ragged Paged Attention) kernel.

Drop-in replacement for gqa_attention() during decode, using the
tpu-inference RPA v3 Pallas kernel instead of jax.nn.dot_product_attention.

The RPA kernel fuses KV cache read + Q@K + softmax + @V + cache write into
a single Pallas kernel with async DMA double buffering, eliminating the
dynamic_slice/dynamic_update_slice bottleneck that dominates decode time.

For prefill, use the original gqa_attention() from gqa.py.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp

from jax_gpt.models.qwen35.fp8 import matmul_maybe_fp8
from jax_gpt.models.qwen35.primitives import apply_rotary_emb, rms_norm

from tpu_inference.kernels.ragged_paged_attention.v3.kernel import (
    ragged_paged_attention,
)


def gqa_attention_rpa(
    x: jax.Array,
    params: dict,
    n_q_heads: int,
    n_kv_heads: int,
    head_dim: int,
    rope_freqs: jax.Array,
    rope_dim: int,
    kv_cache: jax.Array,
    kv_lens: jax.Array,
    page_indices: jax.Array,
    cu_q_lens: jax.Array,
    distribution: jax.Array,
    cache_pos: jax.Array,
    mesh=None,
    axis_name: str = 'tp',
) -> tuple[jax.Array, jax.Array]:
    """GQA forward pass using the RPA v3 kernel for decode.

    This function handles QKV projection, QK norm, RoPE, output gate,
    and output projection — same as gqa_attention() — but replaces the
    cache update + SDPA with the RPA kernel.

    Args:
        x: (B, 1, D) input (decode: single token per sequence).
        params: dict with q_proj, k_proj, v_proj, o_proj, q_norm, k_norm.
        n_q_heads, n_kv_heads, head_dim: attention geometry.
        rope_freqs: precomputed RoPE frequencies.
        rope_dim: number of head dims to apply RoPE to.
        kv_cache: paged KV cache for this layer,
            (total_num_pages, page_size, kv_packed_dim, packing, head_dim).
        kv_lens: i32[B] — current KV length per sequence (before this step).
        page_indices: i32[B * pages_per_seq] — page lookup table.
        cu_q_lens: i32[B+1] — cumulative query lengths.
        distribution: i32[3] — (decode_end, prefill_end, mixed_end).
        cache_pos: scalar int32 — current position (for RoPE).

    Returns:
        (output, updated_kv_cache)
        output: (B, 1, D)
        updated_kv_cache: updated paged cache.
    """
    B, T, D = x.shape
    assert T == 1, f"RPA decode requires T=1, got T={T}"

    # ----- QKV projection -----
    with jax.named_scope('qkv_proj'):
        q_gate = matmul_maybe_fp8(x, params['q_proj']).reshape(B, T, n_q_heads, head_dim * 2)
    q = q_gate[..., :head_dim]      # (B, 1, n_q_heads, head_dim)
    gate = q_gate[..., head_dim:]    # (B, 1, n_q_heads, head_dim)
    gate = gate.reshape(B, T, -1)   # (B, 1, n_q_heads * head_dim)

    k = matmul_maybe_fp8(x, params['k_proj']).reshape(B, T, n_kv_heads, head_dim)
    v = matmul_maybe_fp8(x, params['v_proj']).reshape(B, T, n_kv_heads, head_dim)

    # ----- QK normalization -----
    if 'q_norm' in params:
        q = rms_norm(q, params['q_norm'])
        k = rms_norm(k, params['k_norm'])

    # ----- RoPE -----
    # Transpose to (B, heads, T, head_dim) for RoPE
    q = jnp.transpose(q, (0, 2, 1, 3))  # (B, n_q_heads, 1, head_dim)
    k = jnp.transpose(k, (0, 2, 1, 3))  # (B, n_kv_heads, 1, head_dim)

    freqs = jax.lax.dynamic_slice(
        rope_freqs, (cache_pos, 0, 0), (T, rope_dim // 2, 2)
    )
    q = apply_rotary_emb(q, freqs, rope_dim)
    k = apply_rotary_emb(k, freqs, rope_dim)

    # ----- Reshape for RPA: [max_num_tokens, heads, head_dim] -----
    # Squeeze T=1 and transpose: (B, heads, 1, hd) → (B, heads, hd)
    q_rpa = q[:, :, 0, :]  # (B, n_q_heads, head_dim)
    k_rpa = k[:, :, 0, :]  # (B, n_kv_heads, head_dim)
    v_rpa = v[:, 0, :, :]  # (B, n_kv_heads, head_dim) — v was (B, 1, kv_heads, hd)

    # Cast Q/K/V to match KV cache dtype (RPA requires all dtypes to match)
    cache_dtype = kv_cache.dtype
    q_rpa = q_rpa.astype(cache_dtype)
    k_rpa = k_rpa.astype(cache_dtype)
    v_rpa = v_rpa.astype(cache_dtype)

    # ----- RPA kernel call -----
    sm_scale = 1.0 / math.sqrt(head_dim)

    def _rpa_call(q, k, v, cache, lens, pages, cu_q, dist):
        return ragged_paged_attention(
            q, k, v, cache, lens, pages, cu_q, dist,
            use_causal_mask=False,
            sm_scale=sm_scale,
            disable_semaphore_checks=False,
        )

    with jax.named_scope('rpa_attn'):
        if mesh is not None:
            # Pallas kernels need shard_map for explicit partitioning.
            # Q heads are sharded on TP; batch/pages on DP.
            from jax.experimental.shard_map import shard_map
            from jax.sharding import PartitionSpec as P

            tp_axis = axis_name  # e.g. 'tp'
            dp_axis = 'dp' if 'dp' in mesh.axis_names else None
            attn_out, updated_kv_cache = shard_map(
                _rpa_call,
                mesh=mesh,
                in_specs=(
                    P(dp_axis, tp_axis, None),        # q: (B, n_q_heads, hd)
                    P(dp_axis, None, None),            # k: (B, n_kv_heads, hd)
                    P(dp_axis, None, None),            # v: (B, n_kv_heads, hd)
                    P(dp_axis, None, None, None, None),  # kv_cache: (total_pages, ps, kv_dim, pk, hd)
                    P(dp_axis,),                       # kv_lens: (B,)
                    P(None,),                          # page_indices: (B_local*pps,) replicated
                    P(None,),                          # cu_q_lens: (B_local+1,) replicated
                    P(None,),                          # distribution: (3,) replicated
                ),
                out_specs=(
                    P(dp_axis, tp_axis, None),         # attn_out: (B, n_q_heads, hd)
                    P(dp_axis, None, None, None, None),  # updated_cache: dp-sharded
                ),
                check_rep=False,
            )(q_rpa, k_rpa, v_rpa, kv_cache, kv_lens,
              page_indices, cu_q_lens, distribution)
        else:
            attn_out, updated_kv_cache = _rpa_call(
                q_rpa, k_rpa, v_rpa, kv_cache, kv_lens,
                page_indices, cu_q_lens, distribution,
            )
        # attn_out: (B, n_q_heads, head_dim)

    # Reshape to (B, 1, n_q_heads * head_dim)
    out = attn_out.reshape(B, 1, n_q_heads * head_dim)

    # ----- Output gate (sigmoid) -----
    out = out * jax.nn.sigmoid(gate).astype(out.dtype)

    # ----- Output projection -----
    out = matmul_maybe_fp8(out, params['o_proj'])

    return out.astype(x.dtype), updated_kv_cache
