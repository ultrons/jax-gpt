"""Grouped Query Attention (GQA) with RoPE, QK norm, output gate, and KV cache.

Pure-function implementation matching HuggingFace Qwen3.5.

Key differences from standard GQA:
- q_proj outputs 2x: half is query, half is sigmoid output gate
- QK normalization (RMSNorm per-head) applied to Q and K
- Output gate: attn_output *= sigmoid(gate)
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from jax_gpt.models.qwen35.fp8 import matmul_maybe_fp8
from jax_gpt.models.qwen35.primitives import apply_rotary_emb, rms_norm


def gqa_attention(
    x: jax.Array,
    params: dict,
    n_q_heads: int,
    n_kv_heads: int,
    head_dim: int,
    rope_freqs: jax.Array,
    rope_dim: int,
    cache_k: jax.Array | None = None,
    cache_v: jax.Array | None = None,
    cache_pos: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array | None, jax.Array | None]:
    """Grouped Query Attention forward pass.

    Args:
        x: (B, T, D) input.
        params: dict with keys:
            q_proj: (D, n_q_heads * head_dim * 2)  — query + output gate
            k_proj: (D, n_kv_heads * head_dim)
            v_proj: (D, n_kv_heads * head_dim)
            o_proj: (n_q_heads * head_dim, D)
            q_norm: (head_dim,) — QK norm weight for queries
            k_norm: (head_dim,) — QK norm weight for keys
        n_q_heads, n_kv_heads, head_dim: attention geometry.
        rope_freqs: (max_seq_len, rope_dim//2, 2) precomputed RoPE freqs.
        rope_dim: number of head dims to apply RoPE to.
        cache_k, cache_v: (B, n_kv_heads, max_len, head_dim) or None.
        cache_pos: scalar int32 or None.

    Returns:
        (output, new_cache_k, new_cache_v)
        output: (B, T, D)
        new_cache_k/v: updated caches if caching, else None.
    """
    B, T, D = x.shape
    groups = n_q_heads // n_kv_heads

    with jax.named_scope('qkv_proj'):
        q_gate = matmul_maybe_fp8(x, params['q_proj']).reshape(B, T, n_q_heads, head_dim * 2)
    q = q_gate[..., :head_dim]      # (B, T, n_q_heads, head_dim)
    gate = q_gate[..., head_dim:]    # (B, T, n_q_heads, head_dim)
    gate = gate.reshape(B, T, -1)   # (B, T, n_q_heads * head_dim)

    k = matmul_maybe_fp8(x, params['k_proj']).reshape(B, T, n_kv_heads, head_dim)
    v = matmul_maybe_fp8(x, params['v_proj']).reshape(B, T, n_kv_heads, head_dim)

    # QK normalization (per-head RMSNorm)
    if 'q_norm' in params:
        q = rms_norm(q, params['q_norm'])
        k = rms_norm(k, params['k_norm'])

    # Transpose to (B, heads, T, head_dim)
    q = jnp.transpose(q, (0, 2, 1, 3))
    k = jnp.transpose(k, (0, 2, 1, 3))
    v = jnp.transpose(v, (0, 2, 1, 3))

    # Apply RoPE
    if cache_pos is not None:
        freqs = jax.lax.dynamic_slice(
            rope_freqs, (cache_pos, 0, 0), (T, rope_dim // 2, 2)
        )
    else:
        freqs = rope_freqs[:T]

    q = apply_rotary_emb(q, freqs, rope_dim)
    k = apply_rotary_emb(k, freqs, rope_dim)

    # KV cache update
    if cache_pos is not None:
        k_update_idx = (0, 0, cache_pos, 0)
        cache_k = jax.lax.dynamic_update_slice(cache_k, k.astype(cache_k.dtype), k_update_idx)
        cache_v = jax.lax.dynamic_update_slice(cache_v, v.astype(cache_v.dtype), k_update_idx)
        k_full = cache_k
        v_full = cache_v
    else:
        k_full = k
        v_full = v

    with jax.named_scope('sdpa'):
        # Ensure Q, K, V share the same dtype (QK norm may promote Q/K to f32)
        compute_dtype = q.dtype
        k_full = k_full.astype(compute_dtype)
        v_full = v_full.astype(compute_dtype)

        # Transpose to (B, T, heads, head_dim) for jax.nn.dot_product_attention
        q_btnh = jnp.transpose(q, (0, 2, 1, 3))       # (B, T, n_q_heads, head_dim)
        k_bskh = jnp.transpose(k_full, (0, 2, 1, 3))  # (B, S, n_kv_heads, head_dim)
        v_bskh = jnp.transpose(v_full, (0, 2, 1, 3))  # (B, S, n_kv_heads, head_dim)

        if cache_pos is not None and T == 1:
            # Decode (single token): attend to all filled cache positions
            kv_len = cache_pos + 1
            kv_seq_lengths = jnp.broadcast_to(kv_len, (B,))
            out = jax.nn.dot_product_attention(
                q_btnh, k_bskh, v_bskh,
                is_causal=False,
                key_value_seq_lengths=kv_seq_lengths,
                implementation='xla',
            )
        elif cache_pos is not None:
            # Prefill with cache: need causal mask + restrict to filled positions
            # Build position-based causal mask: q_pos >= k_pos
            q_positions = cache_pos + jnp.arange(T)
            S = k_bskh.shape[1]
            k_positions = jnp.arange(S)
            mask = q_positions[:, None] >= k_positions[None, :]  # (T, S)
            # Expand for broadcast: (1, 1, T, S)
            bias = jnp.where(mask[None, None, :, :], 0.0, jnp.finfo(q_btnh.dtype).min)
            out = jax.nn.dot_product_attention(
                q_btnh, k_bskh, v_bskh,
                bias=bias,
                implementation='xla',
            )
        else:
            # Prefill (no cache): standard causal attention
            out = jax.nn.dot_product_attention(
                q_btnh, k_bskh, v_bskh,
                is_causal=True,
                implementation='xla',
            )
        # out: (B, T, n_q_heads, head_dim)
        out = out.reshape(B, T, n_q_heads * head_dim)

    # Output gate (sigmoid)
    out = out * jax.nn.sigmoid(gate).astype(out.dtype)

    # Output projection
    out = matmul_maybe_fp8(out, params['o_proj'])

    return out.astype(x.dtype), cache_k, cache_v
