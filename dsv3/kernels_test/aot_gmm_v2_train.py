"""Gate-1 AOT test for gmm_v2_train wrappers (forward + backward via jax.vjp).

Validates that the custom_vjp wrapper compiles cleanly at DSv3 shapes,
covering both the plain matmul wrapper and the fused-silu wrapper, plus
a full forward+backward pass through jax.vjp.
"""
import importlib.util
import sys
import time

import jax
import jax.numpy as jnp
from jax.experimental import topologies

# Bypass tpu_inference's vllm-dependent __init__.
sys.path.insert(0, "/home/sivaibhav_google_com/ml-experiments/dsv3/mini_dsv3")

from kernels.gmm_v2_train import gmm_v2_train, gmm_v2_fused_silu_train  # noqa

# DSv3 671B shapes per chunk.
M = 131072      # max_local_c
K = 7168        # D
N = 2048        # F_moe
E = 64          # E_local


def aot(name, fn, *args):
    print(f"[{name}] AOT compiling ...", flush=True)
    t0 = time.perf_counter()
    topo = topologies.get_topology_desc("tpu7x:4x4x4", platform="tpu")
    with jax.default_device(topo.devices[0]):
        compiled = jax.jit(fn).lower(*args).compile()
    dt = time.perf_counter() - t0
    print(f"[{name}] OK in {dt:.1f}s", flush=True)
    try:
        cost = compiled.cost_analysis()
        if cost is not None:
            f = cost.get('flops', 0)
            print(f"  flops={f:,.0f}", flush=True)
    except Exception:
        pass
    return dt


def case_plain_fwd():
    lhs = jax.ShapeDtypeStruct((M, K), jnp.bfloat16)
    rhs = jax.ShapeDtypeStruct((E, K, N), jnp.bfloat16)
    gs = jax.ShapeDtypeStruct((E,), jnp.int32)
    return aot("plain_fwd", lambda l, r, g: gmm_v2_train(l, r, g, 0), lhs, rhs, gs)


def case_plain_fwd_bwd():
    """Forward + backward via jax.value_and_grad."""
    lhs = jax.ShapeDtypeStruct((M, K), jnp.bfloat16)
    rhs = jax.ShapeDtypeStruct((E, K, N), jnp.bfloat16)
    gs = jax.ShapeDtypeStruct((E,), jnp.int32)

    def loss(l, r, g):
        return gmm_v2_train(l, r, g, 0).sum()

    return aot("plain_fwd_bwd",
               lambda l, r, g: jax.value_and_grad(loss, argnums=(0, 1))(l, r, g),
               lhs, rhs, gs)


def case_fused_silu_fwd():
    lhs = jax.ShapeDtypeStruct((M, K), jnp.bfloat16)
    wi_0 = jax.ShapeDtypeStruct((E, K, N), jnp.bfloat16)
    wi_1 = jax.ShapeDtypeStruct((E, K, N), jnp.bfloat16)
    gs = jax.ShapeDtypeStruct((E,), jnp.int32)
    # Fused needs explicit smaller VMEM (default tile picker OOMs).
    return aot("fused_silu_fwd",
               lambda l, w0, w1, g: gmm_v2_fused_silu_train(l, w0, w1, g,
                                                              48 * 1024 * 1024),
               lhs, wi_0, wi_1, gs)


def case_fused_silu_fwd_bwd():
    lhs = jax.ShapeDtypeStruct((M, K), jnp.bfloat16)
    wi_0 = jax.ShapeDtypeStruct((E, K, N), jnp.bfloat16)
    wi_1 = jax.ShapeDtypeStruct((E, K, N), jnp.bfloat16)
    gs = jax.ShapeDtypeStruct((E,), jnp.int32)

    def loss(l, w0, w1, g):
        return gmm_v2_fused_silu_train(l, w0, w1, g, 48 * 1024 * 1024).sum()

    return aot("fused_silu_fwd_bwd",
               lambda l, w0, w1, g: jax.value_and_grad(loss, argnums=(0, 1, 2))(l, w0, w1, g),
               lhs, wi_0, wi_1, gs)


if __name__ == "__main__":
    cases = [
        ("plain_fwd",          case_plain_fwd),
        ("plain_fwd_bwd",      case_plain_fwd_bwd),
        ("fused_silu_fwd",     case_fused_silu_fwd),
        ("fused_silu_fwd_bwd", case_fused_silu_fwd_bwd),
    ]
    failed = []
    for label, fn in cases:
        print(f"\n=== {label} ===")
        try:
            fn()
        except Exception as e:
            print(f"FAILED: {type(e).__name__}: {str(e)[:300]}", flush=True)
            failed.append(label)
    print("\n" + "=" * 60)
    if failed:
        print(f"FAILED ({len(failed)}/{len(cases)}): {failed}")
        sys.exit(1)
    print(f"PASSED all {len(cases)} cases")
