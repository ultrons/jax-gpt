#!/usr/bin/env python3
"""Test that on-device FP8 dequant matches the old CPU implementation.

Run locally:
    python mini_dsv3/test_fp8_dequant.py
"""

import numpy as np
import jax
import jax.numpy as jnp

# --- old CPU reference (kept only for testing) ---
def _dequant_fp8_block_cpu(w_uint8, scale_inv, block_size=128):
    sign = ((w_uint8 >> 7) & 1).astype(np.float32)
    exp_bits = ((w_uint8 >> 3) & 0xF).astype(np.int32)
    mant_bits = (w_uint8 & 0x7).astype(np.float32)
    is_subnormal = (exp_bits == 0)
    mantissa = np.where(is_subnormal, mant_bits / 8.0, 1.0 + mant_bits / 8.0)
    exponent = np.where(is_subnormal, -6, exp_bits - 7).astype(np.float32)
    value = np.where(w_uint8 == 0, 0.0, mantissa * (2.0 ** exponent))
    value = np.where(sign > 0, -value, value)
    rows, cols = w_uint8.shape
    scale_tiled = np.repeat(np.repeat(scale_inv, block_size, axis=0), block_size, axis=1)
    return (value * scale_tiled[:rows, :cols]).astype(np.float32)

# --- new on-device path ---
from .load_weights import _dequant_on_device, _dequant_stacked_on_device


def test_2d(rows=256, cols=512, seed=42):
    rng = np.random.default_rng(seed)
    # Only use valid FP8 E4M3 bit patterns (avoid NaN/inf: 0x7F, 0xFF)
    w_u8 = rng.integers(0, 127, size=(rows, cols), dtype=np.uint8)
    br, bc = (rows + 127) // 128, (cols + 127) // 128
    scale_inv = rng.uniform(0.5, 2.0, size=(br, bc)).astype(np.float32)

    cpu_out = _dequant_fp8_block_cpu(w_u8, scale_inv).astype(np.float32)
    tpu_out = np.array(_dequant_on_device(jnp.array(w_u8), jnp.array(scale_inv)).astype(jnp.float32))

    max_err = np.max(np.abs(cpu_out - tpu_out))
    rel_err = max_err / (np.max(np.abs(cpu_out)) + 1e-8)
    print(f"2D [{rows}x{cols}]: max_abs_err={max_err:.2e}  rel_err={rel_err:.2e}", end="  ")
    # bf16 has ~0.4% relative precision; allow up to 1%
    assert rel_err < 0.01, f"FAIL: rel_err {rel_err:.2e} > 0.01"
    print("OK")


def test_stacked(E=4, rows=128, cols=256, seed=7):
    rng = np.random.default_rng(seed)
    w_u8 = rng.integers(0, 127, size=(E, rows, cols), dtype=np.uint8)
    br, bc = (rows + 127) // 128, (cols + 127) // 128
    scale_inv = rng.uniform(0.5, 2.0, size=(E, br, bc)).astype(np.float32)

    # CPU reference: dequant each expert independently
    cpu_out = np.stack([
        _dequant_fp8_block_cpu(w_u8[e], scale_inv[e]) for e in range(E)
    ])
    tpu_out = np.array(_dequant_stacked_on_device(
        jnp.array(w_u8), jnp.array(scale_inv)).astype(jnp.float32))

    max_err = np.max(np.abs(cpu_out - tpu_out))
    rel_err = max_err / (np.max(np.abs(cpu_out)) + 1e-8)
    print(f"Stacked [{E}x{rows}x{cols}]: max_abs_err={max_err:.2e}  rel_err={rel_err:.2e}", end="  ")
    assert rel_err < 0.01, f"FAIL: rel_err {rel_err:.2e} > 0.01"
    print("OK")


def test_non_multiple_of_block(seed=3):
    """Shapes not divisible by 128 — tests the [:rows, :cols] trim."""
    test_2d(rows=200, cols=300, seed=seed)
    test_stacked(E=3, rows=200, cols=300, seed=seed)


if __name__ == "__main__":
    print(f"Devices: {jax.device_count()} x {jax.devices()[0].device_kind}")
    test_2d(rows=256, cols=512)
    test_2d(rows=512, cols=1024)
    test_stacked(E=4, rows=128, cols=256)
    test_stacked(E=8, rows=256, cols=512)
    test_non_multiple_of_block()
    print("\nAll tests passed.")
