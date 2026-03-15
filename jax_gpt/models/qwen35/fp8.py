"""FP8 matmul utilities for native fp8 inference on TPU.

Weights are stored as float8_e4m3fn with per-row inverse scale factors.
At matmul time, activations are dynamically quantized to fp8, the dot
runs in fp8 with float32 accumulation, and the output is rescaled.

On TPU v5p+, XLA emits native fp8 MXU ops (2x FLOPS of bf16).
On CPU, the fp8 dot still works (JAX supports the dtype) but without
hardware acceleration.

Usage:
    # Standard linear: out = x @ weight
    # FP8 linear:      out = fp8_linear(x, w_fp8, scale_inv)

Weight format (from HuggingFace FP8 model):
    weight:           (out_features, in_features) as float8_e4m3fn
    weight_scale_inv: (out_features, 1) as float32
    Dequantized value: weight.astype(float32) * (1 / weight_scale_inv)
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

FP8_DTYPE = jnp.float8_e4m3fn
FP8_MAX = float(jnp.finfo(FP8_DTYPE).max)  # 448.0


def dynamic_quantize_fp8(x: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Dynamically quantize activation tensor to fp8 per-row.

    Args:
        x: (..., K) float32/bfloat16 activation tensor.

    Returns:
        (x_fp8, scale)
        x_fp8: same shape as x, dtype float8_e4m3fn.
        scale: (..., 1) float32 scale factor such that x ≈ x_fp8 * scale.
    """
    amax = jnp.max(jnp.abs(x), axis=-1, keepdims=True)
    # scale = amax / fp8_max, clamped to avoid division by zero
    scale = jnp.maximum(amax / FP8_MAX, jnp.finfo(jnp.float32).tiny)
    x_fp8 = (x / scale).astype(FP8_DTYPE)
    return x_fp8, scale


def fp8_matmul(
    x: jax.Array,
    w_fp8: jax.Array,
    w_scale_inv: jax.Array,
) -> jax.Array:
    """FP8 matrix multiply: x @ w^T with rescaling.

    Performs: output = (x_fp8 @ w_fp8^T) * x_scale * (1/w_scale_inv)

    This matches HuggingFace's fbgemm fp8 rowwise matmul pattern.
    On TPU, XLA fuses this into native fp8 MXU ops.

    Args:
        x: (..., in_features) float32/bfloat16 activations.
        w_fp8: (out_features, in_features) float8_e4m3fn weights.
        w_scale_inv: (out_features, 1) float32 inverse scale for weights.
            The dequantized weight is: w_fp8 * (1 / w_scale_inv)

    Returns:
        (..., out_features) float32 output.
    """
    # Dynamically quantize activation
    x_fp8, x_scale = dynamic_quantize_fp8(x)

    # FP8 matmul: x_fp8 @ w_fp8^T -> float32
    # x_fp8: (..., K), w_fp8: (N, K) -> (..., N)
    leading_shape = x_fp8.shape[:-1]
    K = x_fp8.shape[-1]
    N = w_fp8.shape[0]

    x_2d = x_fp8.reshape(-1, K)
    out = jax.lax.dot_general(
        x_2d, w_fp8,
        dimension_numbers=(((1,), (1,)), ((), ())),
        preferred_element_type=jnp.float32,
    )  # (M, N) in float32

    # Rescale: out * x_scale * w_scale
    # w_scale = 1 / w_scale_inv, shape (N, 1) -> (1, N) for broadcast
    w_scale = (1.0 / w_scale_inv).reshape(1, N)
    x_scale_2d = x_scale.reshape(-1, 1)
    out = out * x_scale_2d * w_scale

    return out.reshape(*leading_shape, N)


def fp8_linear(
    x: jax.Array,
    params: dict,
    key: str = 'weight',
) -> jax.Array:
    """FP8 linear layer using params dict.

    Convenience wrapper that handles both fp8 and regular float params.
    If params[key] is fp8 dtype, uses fp8_matmul. Otherwise falls back
    to standard matmul.

    For our convention (weight stored as (in, out) for standard matmul),
    FP8 weights from HF are stored as (out, in) — matching PyTorch Linear.

    Args:
        x: (..., in_features) input.
        params: dict containing either:
            - key: (in, out) float32 weight → standard x @ w
            - key: (out, in) fp8 weight + key_scale_inv: (out, 1) → fp8_matmul
    """
    w = params[key]
    if w.dtype == FP8_DTYPE:
        scale_inv = params[f'{key}_scale_inv']
        return fp8_matmul(x, w, scale_inv)
    else:
        # Standard float matmul — weight is (in, out) in our convention
        return x @ w


def fp8_ragged_dot(
    x: jax.Array,
    w_fp8: jax.Array,
    group_sizes: jax.Array,
    w_scale_inv: jax.Array,
) -> jax.Array:
    """FP8 ragged dot for MoE expert computation.

    Args:
        x: (M, K) float32/bfloat16 tokens sorted by expert.
        w_fp8: (E, K, N) float8_e4m3fn expert weights.
        group_sizes: (E,) int32 tokens per expert.
        w_scale_inv: (E, 1, K) or (E, K, 1) float32 inverse scales.
            Exact shape depends on quantization granularity.

    Returns:
        (M, N) float32 output.
    """
    x_fp8, x_scale = dynamic_quantize_fp8(x)

    out = jax.lax.ragged_dot(
        x_fp8, w_fp8, group_sizes,
        preferred_element_type=jnp.float32,
    )  # (M, N)

    # Rescale by activation scale
    out = out * x_scale

    # Rescale by weight scale — this is approximate since ragged_dot
    # uses different expert weights per token group. For per-tensor
    # expert scales, we'd need to scatter the scale per token.
    # For now, weight_scale_inv is handled at load time by baking
    # the scale into the weight (dequantize experts to bf16 or
    # use per-expert scalar scale).
    # TODO: implement proper per-expert rescaling when needed
    if w_scale_inv is not None:
        # If w_scale_inv is a per-expert scalar (E, 1, 1), we need to
        # expand and scatter it per token based on group_sizes
        pass  # handled externally or baked into weights

    return out
