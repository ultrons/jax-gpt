"""Local execution test for gmm_v2_train wrappers.

Runs forward + backward on 4 v4 cores at SMALL shapes, validates against
jax.lax.ragged_dot reference. Catches NaN / wrong-grad bugs before
incurring a 15-min cluster cycle.

Run:
  source ~/xdb/.xprof/bin/activate
  python exec_gmm_v2_train.py
"""
import sys
sys.path.insert(0, "/home/sivaibhav_google_com/ml-experiments/dsv3/mini_dsv3")

import jax
import jax.numpy as jnp
import numpy as np

from kernels.gmm_v2_train import gmm_v2_train, gmm_v2_fused_silu_train


# Small shapes (fit on v4 cores).
M, K, N, E = 1024, 512, 256, 8


def make_inputs(seed=0, dtype=jnp.bfloat16):
    rng = np.random.default_rng(seed)
    lhs = jnp.asarray(rng.standard_normal((M, K)).astype(np.float32) * 0.1, dtype=dtype)
    wi_0 = jnp.asarray(rng.standard_normal((E, K, N)).astype(np.float32) * 0.02, dtype=dtype)
    wi_1 = jnp.asarray(rng.standard_normal((E, K, N)).astype(np.float32) * 0.02, dtype=dtype)
    # Group sizes that sum to M.
    gs = np.full(E, M // E, dtype=np.int32)
    gs = jnp.asarray(gs)
    return lhs, wi_0, wi_1, gs


def test_plain():
    lhs, wi_0, _, gs = make_inputs(0)
    rhs = wi_0  # use just one weight as rhs

    # Reference via jax.lax.ragged_dot
    def ref(l, r):
        return jax.lax.ragged_dot(l, r, gs).sum()
    # Wrapper
    def kern(l, r):
        return gmm_v2_train(l, r, gs).sum()

    out_ref, (d_lhs_ref, d_rhs_ref) = jax.value_and_grad(ref, argnums=(0, 1))(lhs, rhs)
    out_kern, (d_lhs_kern, d_rhs_kern) = jax.value_and_grad(kern, argnums=(0, 1))(lhs, rhs)

    print(f"  ref  out: {float(out_ref):.4f}, kern out: {float(out_kern):.4f}, "
          f"diff: {abs(float(out_ref - out_kern)):.4f}")
    print(f"  d_lhs max abs diff: {float(jnp.max(jnp.abs(d_lhs_ref - d_lhs_kern))):.6f} "
          f"(any NaN: {bool(jnp.any(jnp.isnan(d_lhs_kern)))})")
    print(f"  d_rhs max abs diff: {float(jnp.max(jnp.abs(d_rhs_ref - d_rhs_kern))):.6f} "
          f"(any NaN: {bool(jnp.any(jnp.isnan(d_rhs_kern)))})")


def test_fused_silu():
    lhs, wi_0, wi_1, gs = make_inputs(1)

    def ref(l, w0, w1):
        gate = jax.lax.ragged_dot(l, w0, gs)
        up = jax.lax.ragged_dot(l, w1, gs)
        return (jax.nn.silu(gate) * up).sum()

    def kern(l, w0, w1):
        return gmm_v2_fused_silu_train(l, w0, w1, gs).sum()

    out_ref, (d_l_ref, d_w0_ref, d_w1_ref) = jax.value_and_grad(ref, argnums=(0, 1, 2))(lhs, wi_0, wi_1)
    out_k, (d_l_k, d_w0_k, d_w1_k) = jax.value_and_grad(kern, argnums=(0, 1, 2))(lhs, wi_0, wi_1)

    print(f"  ref  out: {float(out_ref):.4f}, kern out: {float(out_k):.4f}, "
          f"diff: {abs(float(out_ref - out_k)):.4f}")
    for name, ref_g, kern_g in [
        ("d_lhs",  d_l_ref,  d_l_k),
        ("d_wi_0", d_w0_ref, d_w0_k),
        ("d_wi_1", d_w1_ref, d_w1_k),
    ]:
        nan = bool(jnp.any(jnp.isnan(kern_g)))
        finf = bool(jnp.any(jnp.isinf(kern_g)))
        max_diff = float(jnp.max(jnp.abs(ref_g - kern_g)))
        ref_max = float(jnp.max(jnp.abs(ref_g)))
        print(f"  {name}: max_abs_diff={max_diff:.6f}, ref_max={ref_max:.4f}, "
              f"NaN={nan}, Inf={finf}")


if __name__ == "__main__":
    print("devices:", jax.devices())
    print(f"M={M}, K={K}, N={N}, E={E}")
    print("\n=== plain ===")
    test_plain()
    print("\n=== fused_silu ===")
    test_fused_silu()
