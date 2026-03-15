"""Gated DeltaNet linear attention.

Implements both recurrent (decode) and chunk-parallel (prefill) modes.
Follows the HuggingFace Qwen3.5 reference implementation exactly.

Key formula (recurrent step):
    g_t = exp(-A * softplus(a_t + dt_bias))   # per-head decay
    beta_t = sigmoid(b_t)                      # per-head gate
    state = state * g_t                        # decay old state
    kv_mem = einsum(state, k_t, 'b h dk dv, b h dk -> b h dv')
    delta = (v_t - kv_mem) * beta_t            # gated correction
    state = state + outer(k_t, delta)          # rank-1 update
    output = einsum(state, q_t, 'b h dk dv, b h dk -> b h dv')

All functions are pure — params are plain dicts of arrays.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def _l2_norm(x: jax.Array, eps: float = 1e-6) -> jax.Array:
    """L2 normalize along the last axis."""
    return x / jnp.maximum(jnp.linalg.norm(x, axis=-1, keepdims=True), eps)


def _causal_conv1d_update(
    x: jax.Array,
    conv_state: jax.Array,
    conv_weight: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Single-step causal conv1d update (decode mode).

    Args:
        x: (B, conv_dim, 1) — new input (single token, channels-first).
        conv_state: (B, conv_dim, kernel_size) — sliding window state.
        conv_weight: (conv_dim, kernel_size) — depthwise conv weights.

    Returns:
        (output, new_conv_state)
        output: (B, conv_dim, 1) after depthwise conv + silu.
        new_conv_state: updated sliding window.
    """
    # Shift state left and append new input
    new_state = jnp.concatenate([conv_state[..., 1:], x], axis=-1)
    # Depthwise conv: sum(state * weight) per channel
    out = jnp.sum(new_state * conv_weight[None, :, :], axis=-1, keepdims=True)
    out = jax.nn.silu(out)
    return out, new_state


def _causal_conv1d_prefill(
    x: jax.Array,
    conv_weight: jax.Array,
    kernel_size: int,
) -> tuple[jax.Array, jax.Array]:
    """Causal conv1d for prefill (full sequence).

    Args:
        x: (B, conv_dim, T) — channels-first input.
        conv_weight: (conv_dim, kernel_size) — depthwise weights.
        kernel_size: conv kernel size.

    Returns:
        (output, final_conv_state)
        output: (B, conv_dim, T) after causal depthwise conv + silu.
        final_conv_state: (B, conv_dim, kernel_size) for subsequent decode.
    """
    # Pad left for causal conv
    x_padded = jnp.pad(x, ((0, 0), (0, 0), (kernel_size - 1, 0)))
    # Depthwise conv via lax.conv_general_dilated
    # weight: (conv_dim, kernel_size) -> (conv_dim, 1, kernel_size) for group conv
    w = conv_weight[:, None, :]  # (conv_dim, 1, kernel_size)
    out = jax.lax.conv_general_dilated(
        x_padded, w,
        window_strides=(1,),
        padding='VALID',
        feature_group_count=x.shape[1],  # depthwise
    )
    out = jax.nn.silu(out)
    # Final conv state: last kernel_size values of x
    final_state = x[..., -kernel_size:]
    # If T < kernel_size, pad on the left
    if x.shape[-1] < kernel_size:
        pad_len = kernel_size - x.shape[-1]
        final_state = jnp.pad(final_state, ((0, 0), (0, 0), (pad_len, 0)))
    return out, final_state


def deltanet_recurrent_step(
    x: jax.Array,
    params: dict,
    state: jax.Array,
    conv_state: jax.Array,
    config_n_qk_heads: int,
    config_n_v_heads: int,
    config_qk_head_dim: int,
    config_v_head_dim: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Single-step DeltaNet recurrent update (decode mode).

    Args:
        x: (B, 1, D) single token input.
        params: dict with keys:
            in_proj_qkv: (D, key_dim*2 + value_dim)
            in_proj_z: (D, value_dim)
            in_proj_b: (D, n_v_heads)
            in_proj_a: (D, n_v_heads)
            conv_weight: (conv_dim, kernel_size)
            A_log: (n_v_heads,)
            dt_bias: (n_v_heads,)
            norm_weight: (v_head_dim,)
            out_proj: (value_dim, D)
        state: (B, n_v_heads, qk_head_dim, v_head_dim) recurrent state M.
        conv_state: (B, conv_dim, kernel_size) conv1d sliding window.
        config_*: static config values.

    Returns:
        (output, new_state, new_conv_state)
    """
    B = x.shape[0]
    key_dim = config_n_qk_heads * config_qk_head_dim
    value_dim = config_n_v_heads * config_v_head_dim
    groups = config_n_v_heads // config_n_qk_heads

    # Project QKV and gate/decay
    with jax.named_scope('qkv_proj'):
        mixed_qkv = x @ params['in_proj_qkv']  # (B, 1, key_dim*2 + value_dim)
    z = x @ params['in_proj_z']             # (B, 1, value_dim)
    b = x @ params['in_proj_b']             # (B, 1, n_v_heads)
    a = x @ params['in_proj_a']             # (B, 1, n_v_heads)

    # Causal conv1d update (channels-first)
    mixed_qkv_cf = jnp.transpose(mixed_qkv, (0, 2, 1))  # (B, conv_dim, 1)
    mixed_qkv_cf, new_conv_state = _causal_conv1d_update(
        mixed_qkv_cf, conv_state, params['conv_weight']
    )
    mixed_qkv = jnp.transpose(mixed_qkv_cf, (0, 2, 1))  # (B, 1, conv_dim)

    # Split into Q, K, V
    q = mixed_qkv[..., :key_dim]
    k = mixed_qkv[..., key_dim:key_dim * 2]
    v = mixed_qkv[..., key_dim * 2:]

    # Reshape to heads
    q = q.reshape(B, 1, config_n_qk_heads, config_qk_head_dim)
    k = k.reshape(B, 1, config_n_qk_heads, config_qk_head_dim)
    v = v.reshape(B, 1, config_n_v_heads, config_v_head_dim)

    # Compute decay and gate
    beta = jax.nn.sigmoid(b)  # (B, 1, n_v_heads)
    # g = -exp(A_log) * softplus(a + dt_bias)  (matches HF exactly)
    g = -jnp.exp(params['A_log']) * jax.nn.softplus(a.squeeze(1) + params['dt_bias'])  # (B, n_v_heads)

    # L2 normalize Q, K
    scale = config_qk_head_dim ** -0.5
    q = _l2_norm(q) * scale
    k = _l2_norm(k)

    # Repeat Q, K for grouped heads (n_qk_heads -> n_v_heads)
    if groups > 1:
        q = jnp.repeat(q, groups, axis=2)
        k = jnp.repeat(k, groups, axis=2)

    # Squeeze sequence dim (single token)
    q = q[:, 0]  # (B, n_v_heads, qk_head_dim)
    k = k[:, 0]
    v = v[:, 0]  # (B, n_v_heads, v_head_dim)
    beta = beta[:, 0]  # (B, n_v_heads)

    # Recurrent step (compute in float32, cast state back to input dtype for scan)
    input_dtype = x.dtype
    # state: (B, n_v_heads, qk_head_dim, v_head_dim)
    state_f32 = state.astype(jnp.float32)
    g_factor = jnp.exp(g.astype(jnp.float32))[..., None, None]
    new_state = state_f32 * g_factor

    kv_mem = jnp.einsum('bhkv,bhk->bhv', new_state, k.astype(jnp.float32))
    delta = (v.astype(jnp.float32) - kv_mem) * beta.astype(jnp.float32)[..., None]

    new_state = new_state + jnp.einsum('bhk,bhv->bhkv', k.astype(jnp.float32), delta)
    new_state = new_state.astype(input_dtype)

    output = jnp.einsum('bhkv,bhk->bhv', new_state.astype(jnp.float32), q.astype(jnp.float32))

    # Gated RMSNorm (applied per-head, weight is per v_head_dim)
    output = output.reshape(B * config_n_v_heads, config_v_head_dim)
    z_reshaped = z[:, 0].reshape(B, config_n_v_heads, config_v_head_dim)
    z_reshaped = z_reshaped.reshape(B * config_n_v_heads, config_v_head_dim)
    output = _gated_rms_norm(output, z_reshaped, params['norm_weight'], eps=1e-6)
    output = output.reshape(B, value_dim)
    output = (output @ params['out_proj'])[:, None, :]  # (B, 1, D)

    return output.astype(x.dtype), new_state, new_conv_state


def deltanet_prefill(
    x: jax.Array,
    params: dict,
    config_n_qk_heads: int,
    config_n_v_heads: int,
    config_qk_head_dim: int,
    config_v_head_dim: int,
    config_conv_kernel: int,
    chunk_size: int = 64,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Chunk-parallel DeltaNet prefill.

    Processes the full sequence in chunks for efficiency, while maintaining
    the same recurrent state as sequential processing.

    Args:
        x: (B, T, D) input sequence.
        params: same as deltanet_recurrent_step.
        config_*: static config values.
        chunk_size: chunk size for parallel processing.

    Returns:
        (output, final_state, final_conv_state)
        output: (B, T, D)
        final_state: (B, n_v_heads, qk_head_dim, v_head_dim)
        final_conv_state: (B, conv_dim, kernel_size)
    """
    B, T, D = x.shape
    key_dim = config_n_qk_heads * config_qk_head_dim
    value_dim = config_n_v_heads * config_v_head_dim
    groups = config_n_v_heads // config_n_qk_heads

    # Project
    mixed_qkv = x @ params['in_proj_qkv']  # (B, T, conv_dim)
    z = x @ params['in_proj_z']             # (B, T, value_dim)
    b = x @ params['in_proj_b']             # (B, T, n_v_heads)
    a = x @ params['in_proj_a']             # (B, T, n_v_heads)

    # Causal conv1d (full sequence)
    mixed_qkv_cf = jnp.transpose(mixed_qkv, (0, 2, 1))  # (B, conv_dim, T)
    mixed_qkv_cf, final_conv_state = _causal_conv1d_prefill(
        mixed_qkv_cf, params['conv_weight'], config_conv_kernel
    )
    mixed_qkv = jnp.transpose(mixed_qkv_cf, (0, 2, 1))  # (B, T, conv_dim)

    # Split Q, K, V
    q = mixed_qkv[..., :key_dim].reshape(B, T, config_n_qk_heads, config_qk_head_dim)
    k = mixed_qkv[..., key_dim:key_dim * 2].reshape(B, T, config_n_qk_heads, config_qk_head_dim)
    v = mixed_qkv[..., key_dim * 2:].reshape(B, T, config_n_v_heads, config_v_head_dim)

    # Compute decay and gate
    beta = jax.nn.sigmoid(b)  # (B, T, n_v_heads)
    # g = -exp(A_log) * softplus(a + dt_bias)  (matches HF exactly)
    g = -jnp.exp(params['A_log']) * jax.nn.softplus(a + params['dt_bias'])  # (B, T, n_v_heads)

    # L2 normalize Q, K
    scale = config_qk_head_dim ** -0.5
    q = _l2_norm(q) * scale
    k = _l2_norm(k)

    # Repeat Q, K for grouped heads
    if groups > 1:
        q = jnp.repeat(q, groups, axis=2)
        k = jnp.repeat(k, groups, axis=2)

    # Transpose to (B, heads, T, dim)
    q = jnp.transpose(q, (0, 2, 1, 3))  # (B, n_v_heads, T, qk_head_dim)
    k = jnp.transpose(k, (0, 2, 1, 3))
    v = jnp.transpose(v, (0, 2, 1, 3))  # (B, n_v_heads, T, v_head_dim)
    beta = jnp.transpose(beta, (0, 2, 1))  # (B, n_v_heads, T)
    g = jnp.transpose(g, (0, 2, 1))        # (B, n_v_heads, T)

    # Run recurrent computation via lax.scan for correctness
    # (chunk-parallel optimization can be added later as a Pallas kernel)
    initial_state = jnp.zeros(
        (B, config_n_v_heads, config_qk_head_dim, config_v_head_dim),
        dtype=q.dtype,
    )

    carry_dtype = q.dtype  # preserve dtype for scan carry

    def _step(state, inputs):
        q_t, k_t, v_t, beta_t, g_t = inputs
        # Compute in float32 for numerical stability, cast back for scan carry
        state_f32 = state.astype(jnp.float32)
        g_factor = jnp.exp(g_t.astype(jnp.float32))[..., None, None]
        state_f32 = state_f32 * g_factor

        kv_mem = jnp.einsum('bhkv,bhk->bhv', state_f32, k_t.astype(jnp.float32))
        delta = (v_t.astype(jnp.float32) - kv_mem) * beta_t.astype(jnp.float32)[..., None]
        state_f32 = state_f32 + jnp.einsum('bhk,bhv->bhkv', k_t.astype(jnp.float32), delta)

        o_t = jnp.einsum('bhkv,bhk->bhv', state_f32, q_t.astype(jnp.float32))
        return state_f32.astype(carry_dtype), o_t.astype(carry_dtype)

    # Prepare scan inputs: transpose T to leading axis
    scan_inputs = (
        jnp.transpose(q, (2, 0, 1, 3)),     # (T, B, H, dk)
        jnp.transpose(k, (2, 0, 1, 3)),     # (T, B, H, dk)
        jnp.transpose(v, (2, 0, 1, 3)),     # (T, B, H, dv)
        jnp.transpose(beta, (2, 0, 1)),     # (T, B, H)
        jnp.transpose(g, (2, 0, 1)),        # (T, B, H)
    )

    final_state, outputs = jax.lax.scan(_step, initial_state, scan_inputs)
    # outputs: (T, B, H, dv) -> (B, T, H, dv)
    core_attn_out = jnp.transpose(outputs, (1, 0, 2, 3))

    # Gated RMSNorm (applied per-head, weight is per v_head_dim)
    # core_attn_out: (B, T, n_v_heads, v_head_dim) -> (B*T*n_v_heads, v_head_dim)
    core_attn_out = core_attn_out.reshape(B * T * config_n_v_heads, config_v_head_dim)
    z_flat = z.reshape(B, T, config_n_v_heads, config_v_head_dim)
    z_flat = z_flat.reshape(B * T * config_n_v_heads, config_v_head_dim)
    core_attn_out = _gated_rms_norm(core_attn_out, z_flat, params['norm_weight'], eps=1e-6)
    core_attn_out = core_attn_out.reshape(B, T, value_dim)
    output = core_attn_out @ params['out_proj']  # (B, T, D)

    return output.astype(x.dtype), final_state, final_conv_state


def _gated_rms_norm(
    x: jax.Array,
    gate: jax.Array,
    weight: jax.Array,
    eps: float = 1e-6,
) -> jax.Array:
    """Gated RMSNorm: rms_norm(x) * weight * silu(gate).

    Args:
        x: (..., D)
        gate: (..., D) — gate input.
        weight: (D,) — learnable scale (per v_head_dim, broadcast across heads).
        eps: numerical stability.
    """
    orig_dtype = x.dtype
    x_f32 = x.astype(jnp.float32)
    variance = jnp.mean(jnp.square(x_f32), axis=-1, keepdims=True)
    x_normed = x_f32 * jax.lax.rsqrt(variance + eps)
    x_normed = (1.0 + weight.astype(jnp.float32)) * x_normed
    result = x_normed * jax.nn.silu(gate.astype(jnp.float32))
    return result.astype(orig_dtype)
