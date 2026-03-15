"""Pure-function primitives: RMSNorm, RoPE, SwiGLU.

No classes — just functions that operate on JAX arrays.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------

def rms_norm(x: jax.Array, weight: jax.Array, eps: float = 1e-6) -> jax.Array:
    """RMSNorm: norm(x) * (1 + weight).

    Qwen3.5 convention: weight is initialized to zeros and applied as (1 + weight).
    This is equivalent to standard RMSNorm with weight initialized to ones.

    Args:
        x: (..., D)
        weight: (D,)  learnable scale (initialized to 0 in HF, 1 means no change)
        eps: numerical stability
    Returns:
        normalized x, same shape as input.
    """
    x_f32 = x.astype(jnp.float32)
    rms = jnp.sqrt(jnp.mean(jnp.square(x_f32), axis=-1, keepdims=True) + eps)
    normed = x_f32 / rms
    return (normed * (1.0 + weight.astype(jnp.float32))).astype(x.dtype)


# ---------------------------------------------------------------------------
# Rotary Position Embeddings (RoPE)
# ---------------------------------------------------------------------------

def precompute_rope_freqs(
    dim: int,
    max_seq_len: int,
    theta: float = 10_000_000.0,
) -> jax.Array:
    """Precompute RoPE complex frequency tensor.

    Args:
        dim: number of dimensions to apply RoPE to (must be even).
        max_seq_len: maximum sequence length to precompute.
        theta: base frequency.
    Returns:
        freqs_cis: (max_seq_len, dim//2, 2) — cos and sin interleaved.
    """
    freqs = 1.0 / (theta ** (jnp.arange(0, dim, 2, dtype=jnp.float32) / dim))
    t = jnp.arange(max_seq_len, dtype=jnp.float32)
    angles = jnp.outer(t, freqs)  # (max_seq_len, dim//2)
    return jnp.stack([jnp.cos(angles), jnp.sin(angles)], axis=-1)  # (seq, dim//2, 2)


def apply_rotary_emb(
    x: jax.Array,
    freqs_cis: jax.Array,
    rope_dim: int,
) -> jax.Array:
    """Apply RoPE to the first `rope_dim` dimensions of x.

    Partial rotary: only the first `rope_dim` dims of each head are rotated,
    the rest pass through unchanged.

    Args:
        x: (..., head_dim)
        freqs_cis: (seq_len, rope_dim//2, 2) from precompute_rope_freqs.
        rope_dim: how many dims to rotate (must be even, <= head_dim).
    Returns:
        x with RoPE applied, same shape.
    """
    head_dim = x.shape[-1]
    x_rot = x[..., :rope_dim]
    x_pass = x[..., rope_dim:]

    # Reshape for rotation: (..., rope_dim) -> (..., rope_dim//2, 2)
    x_rot = x_rot.reshape(*x_rot.shape[:-1], rope_dim // 2, 2)
    cos = freqs_cis[..., 0]  # (seq, rope_dim//2)
    sin = freqs_cis[..., 1]

    # Broadcast cos/sin to match x_rot batch dims
    # x_rot: (B, n_heads, T, rope_dim//2, 2)  or  (B, T, n_heads, rope_dim//2, 2)
    # freqs_cis: (T, rope_dim//2, 2)
    # We need cos/sin shaped for broadcasting. They come in as (T, rope_dim//2).
    # Expand to match the head dims of x.
    ndim = x_rot.ndim
    # freqs are (T, rope_dim//2), we need to insert dims for batch and heads
    for _ in range(ndim - 3):  # -3 because freqs already have (T, rope_dim//2) and we add one for the pair dim
        cos = jnp.expand_dims(cos, 0)
        sin = jnp.expand_dims(sin, 0)

    x0 = x_rot[..., 0]
    x1 = x_rot[..., 1]
    out0 = x0 * cos - x1 * sin
    out1 = x1 * cos + x0 * sin
    # Stack pairs back: (..., rope_dim//2, 2) -> (..., rope_dim)
    x_rotated = jnp.stack([out0, out1], axis=-1).reshape(*x_rot.shape[:-2], rope_dim)

    if rope_dim < head_dim:
        return jnp.concatenate([x_rotated, x_pass], axis=-1)
    return x_rotated


# ---------------------------------------------------------------------------
# SwiGLU
# ---------------------------------------------------------------------------

def swiglu(
    x: jax.Array,
    gate_weight: jax.Array,
    up_weight: jax.Array,
    down_weight: jax.Array,
) -> jax.Array:
    """SwiGLU FFN: down @ (silu(x @ gate) * (x @ up)).

    Args:
        x: (..., D)
        gate_weight: (D, intermediate_dim)
        up_weight: (D, intermediate_dim)
        down_weight: (intermediate_dim, D)
    Returns:
        (..., D)
    """
    gate = jax.nn.silu(x @ gate_weight)
    up = x @ up_weight
    return (gate * up) @ down_weight
