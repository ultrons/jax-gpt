"""Correctness test for sc_gather_rows (TC fallback path).

sc_gather_rows currently uses source[row_indices] (TC) unconditionally.
SC IndexedLoad was attempted but plsc.BlockSpec mis-lowers 2D indexed specs
to the Get register instruction in current JAX; see sc_gather_rows docstring.

For each case:
  out  = sc_gather_rows(source, indices)   # TC gather
  ref  = source[indices]                   # TC reference (should be identical)
  assert out == ref  (bit-exact)

Usage:
  python test_sc_gather_rows.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import jax
import jax.numpy as jnp

from .backward_kernel import sc_gather_rows


def check(label, source, indices):
    ref = source[indices]
    out = sc_gather_rows(source, indices)
    ok = bool(jnp.all(out == ref))
    status = "PASS" if ok else "FAIL"
    if not ok:
        diff = jnp.abs(out.astype(jnp.float32) - ref.astype(jnp.float32))
        worst = jnp.unravel_index(jnp.argmax(diff), diff.shape)
        print(f"  [{status}] {label}")
        print(f"         max_diff={float(diff.max()):.2e} at {worst}")
        print(f"         out={float(out[worst]):.6f}  ref={float(ref[worst]):.6f}")
    else:
        print(f"  [{status}] {label}")
    return ok


def main():
    print(f"JAX devices : {jax.devices()}")
    print()

    rng = np.random.default_rng(42)
    results = []

    bf16_cases = [
        ( 1024,   512,  128, "bf16 T=1024  D=512   n=128"),
        ( 4096,  2048,  512, "bf16 T=4096  D=2048  n=512"),
        (16384,  7168, 4096, "bf16 T=16384 D=7168  n=4096"),
        (  256,  2048,   64, "bf16 T=256   D=2048  n=64  (last-row safety)"),
        ( 1024,  2048,  512, "bf16 T=1024  D=2048  n=512 (with repeats, n>T)"),
    ]

    print("--- bfloat16 ---")
    for T, D, n, label in bf16_cases:
        src = jnp.array(rng.standard_normal((T, D)).astype(np.float32), dtype=jnp.bfloat16)
        idx = jnp.array(rng.integers(0, T, size=n, dtype=np.int32))
        results.append(check(label, src, idx))

    print()
    print("--- bfloat16 last-row edge case (all indices = T-1) ---")
    T, D = 256, 2048
    src = jnp.array(rng.standard_normal((T, D)).astype(np.float32), dtype=jnp.bfloat16)
    idx = jnp.full((64,), T - 1, dtype=jnp.int32)
    results.append(check("bf16 all indices = T-1", src, idx))

    f32_cases = [
        ( 4096,  2048,  512, "f32  T=4096  D=2048  n=512"),
        (16384,  7168, 4096, "f32  T=16384 D=7168  n=4096"),
    ]

    print()
    print("--- float32 ---")
    for T, D, n, label in f32_cases:
        src = jnp.array(rng.standard_normal((T, D)).astype(np.float32))
        idx = jnp.array(rng.integers(0, T, size=n, dtype=np.int32))
        results.append(check(label, src, idx))

    print()
    n_pass = sum(results)
    n_fail = len(results) - n_pass
    if n_fail == 0:
        print(f"ALL PASS ({n_pass}/{len(results)})")
        return 0
    else:
        print(f"FAILED {n_fail}/{len(results)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
