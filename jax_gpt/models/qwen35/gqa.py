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

    # Project Q (with gate), K, V
    q_gate = (x @ params['q_proj']).reshape(B, T, n_q_heads, head_dim * 2)
    q = q_gate[..., :head_dim]      # (B, T, n_q_heads, head_dim)
    gate = q_gate[..., head_dim:]    # (B, T, n_q_heads, head_dim)
    gate = gate.reshape(B, T, -1)   # (B, T, n_q_heads * head_dim)

    k = (x @ params['k_proj']).reshape(B, T, n_kv_heads, head_dim)
    v = (x @ params['v_proj']).reshape(B, T, n_kv_heads, head_dim)

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
        cache_k = jax.lax.dynamic_update_slice(cache_k, k, k_update_idx)
        cache_v = jax.lax.dynamic_update_slice(cache_v, v, k_update_idx)
        k_full = cache_k
        v_full = cache_v
    else:
        k_full = k
        v_full = v

    # Expand KV heads for GQA
    if groups > 1:
        k_full = jnp.repeat(k_full, groups, axis=1)
        v_full = jnp.repeat(v_full, groups, axis=1)

    # Scaled dot-product attention
    scale = head_dim ** -0.5
    attn = jnp.matmul(q, jnp.swapaxes(k_full, -2, -1)) * scale

    # Causal mask
    if cache_pos is not None:
        q_positions = cache_pos + jnp.arange(T)
        k_positions = jnp.arange(attn.shape[-1])
        mask = q_positions[:, None] >= k_positions[None, :]
    else:
        mask = jnp.tril(jnp.ones((T, T), dtype=jnp.bool_))

    attn = jnp.where(mask, attn, jnp.finfo(attn.dtype).min)
    attn = jax.nn.softmax(attn, axis=-1)

    # Attend
    out = jnp.matmul(attn, v_full)  # (B, n_q_heads, T, head_dim)

    # Transpose back
    out = jnp.transpose(out, (0, 2, 1, 3)).reshape(B, T, n_q_heads * head_dim)

    # Output gate (sigmoid)
    out = out * jax.nn.sigmoid(gate)

    # Output projection
    out = out @ params['o_proj']

    return out, cache_k, cache_v
