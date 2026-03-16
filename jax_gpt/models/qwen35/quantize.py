"""Post-init quantization of model params to FP8.

Converts large weight matrices (linear projections, expert weights) to
float8_e4m3fn with per-row scale factors. Small params (norms, biases,
A_log, dt_bias) stay in their original dtype.

The forward pass uses fp8_matmul when it detects fp8 weights.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from jax_gpt.models.qwen35.fp8 import FP8_DTYPE, FP8_MAX


def _quantize_weight(w: jax.Array) -> dict:
    """Quantize a weight matrix to fp8 with per-row scale.

    Our weights are stored as (in, out). We transpose to (out, in) for
    fp8_matmul which does x @ w^T, then quantize per-row of (out, in).

    For 3D+ expert weights (E, in, out), we transpose last two dims
    to (E, out, in) and quantize per-row.

    Returns:
        {'w': fp8 in (out, in) or (E, out, in) layout,
         'scale_inv': per-row inverse scale}
    """
    # Transpose last two dims: (..., in, out) -> (..., out, in) for fp8_matmul
    # Works for any number of leading (stacked/scan) dimensions
    axes = list(range(w.ndim))
    axes[-2], axes[-1] = axes[-1], axes[-2]
    w_t = jnp.transpose(w, axes).astype(jnp.float32)

    # Per-row quantization
    amax = jnp.max(jnp.abs(w_t), axis=-1, keepdims=True)
    scale = jnp.maximum(amax / FP8_MAX, jnp.finfo(jnp.float32).tiny)
    w_fp8 = (w_t / scale).astype(FP8_DTYPE)
    scale_inv = (1.0 / scale).astype(jnp.float32)
    return {'w': w_fp8, 'scale_inv': scale_inv}


def _should_quantize(path: str, shape: tuple) -> bool:
    """Decide if a param should be quantized to fp8.

    Quantize: large weight matrices (linear projections, expert weights).
    Keep original: norms, biases, scalars, A_log, dt_bias, gate scalars.
    """
    # Skip small params
    if len(shape) < 2:
        return False
    # Skip tiny matrices (norms, etc)
    if shape[-1] < 32 or shape[-2] < 32:
        return False
    # Skip specific params that should stay in higher precision
    skip_patterns = ['norm', 'A_log', 'dt_bias', 'scale_inv', 'embed', 'conv_weight']
    for pattern in skip_patterns:
        if pattern in path:
            return False
    return True


def quantize_params_fp8(params: dict) -> dict:
    """Quantize model params to FP8 in-place.

    Large weight matrices become {'w': fp8, 'scale_inv': f32}.
    Small params (norms, biases) stay unchanged.

    The forward pass matmul functions need to handle both:
    - Regular: x @ params['weight']  (when weight is a plain array)
    - FP8: fp8_matmul(x, params['weight']['w'], params['weight']['scale_inv'])
    """
    def _maybe_quantize(path, leaf):
        path_str = '.'.join(
            str(k).strip("[]'.\"") for k in path
            if not str(k).strip("[]'.\"").isdigit()
        )
        if _should_quantize(path_str, leaf.shape):
            return _quantize_weight(leaf)
        return leaf

    return jax.tree_util.tree_map_with_path(_maybe_quantize, params)


def dequantize_params(params: dict) -> dict:
    """Dequantize FP8 params back to float for debugging/comparison.

    Converts {'w': fp8, 'scale_inv': f32} back to a single float array.
    """
    def _maybe_dequant(leaf):
        if isinstance(leaf, dict) and 'w' in leaf and 'scale_inv' in leaf:
            return (leaf['w'].astype(jnp.float32) / leaf['scale_inv']).astype(jnp.bfloat16)
        return leaf

    return jax.tree.map(_maybe_dequant, params)


def count_fp8_params(params: dict) -> tuple[int, int]:
    """Count total params and fp8-quantized params."""
    total = 0
    fp8_count = 0
    for leaf in jax.tree_util.tree_leaves(params):
        if isinstance(leaf, jax.Array):
            total += leaf.size
            if leaf.dtype == FP8_DTYPE:
                fp8_count += leaf.size
    return total, fp8_count
