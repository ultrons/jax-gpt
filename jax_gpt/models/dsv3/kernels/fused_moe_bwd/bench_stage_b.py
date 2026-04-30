"""Performance benchmark for fused_ep_moe forward/backward.

Measures:
  1. Forward only:  fused_ep_moe (Pallas) vs ref_moe_with_residuals (JAX vmap)
  2. Training step: Stage A (JAX fwd+bwd) vs Stage B (Pallas fwd + JAX bwd)

Run with:
  /home/sivaibhav_google_com/xdb/.xprof/bin/python3 bench_stage_b.py

Dimensions are chosen to fit on 4× local TPU v4 (32 GB each).
"""

import time
import sys
import numpy as np
import env  # noqa: F401

import jax
import jax.numpy as jnp

from tpu_inference.kernels.fused_moe.v1.kernel import ref_moe, fused_ep_moe
from .backward import (
    make_fused_ep_moe_train,
    make_fused_ep_moe_train_v2,
    ref_moe_with_residuals,
)


# ---------------------------------------------------------------------------
# Timing helper
# ---------------------------------------------------------------------------

def timeit(fn, n_warmup=3, n_iter=10):
    """Run fn() n_warmup times (ignored), then n_iter times, return (mean_ms, std_ms)."""
    for _ in range(n_warmup):
        jax.block_until_ready(fn())
    times = []
    for _ in range(n_iter):
        t0 = time.perf_counter()
        jax.block_until_ready(fn())
        times.append((time.perf_counter() - t0) * 1e3)
    return float(np.mean(times)), float(np.std(times))


def make_ep_mesh(ep_size: int) -> jax.sharding.Mesh:
    devices = jax.devices()
    assert len(devices) >= ep_size
    return jax.sharding.Mesh(
        np.array(devices[:ep_size]).reshape(1, ep_size), ("data", "model"))


# ---------------------------------------------------------------------------
# Benchmark configs
# ---------------------------------------------------------------------------

CONFIGS = [
    # (label,        T,    D,    F,   E,  K,  EP)
    ("tiny  EP=1", 64,  128,  128,  8,  2,  1),
    ("tiny  EP=4", 64,  128,  128,  8,  2,  4),
    ("small EP=1", 256, 512,  512, 16,  2,  1),
    ("small EP=4", 256, 512,  512, 16,  2,  4),
    ("med   EP=1", 512, 1024,1024, 16,  8,  1),
    ("med   EP=4", 512, 1024,1024, 16,  8,  4),
]

MOE_CFG = dict(scoring_fn="sigmoid", renormalize_topk_logits=True)


def run_config(label, T, D, F, E, K, EP):
    print(f"\n{'─'*60}")
    print(f"  {label:12s}  T={T} D={D} F={F} E={E} K={K} EP={EP}")
    print(f"{'─'*60}")

    key = jax.random.PRNGKey(42)
    k1, k2, k3, k4 = jax.random.split(key, 4)
    tokens  = jax.random.normal(k1, (T, D),        dtype=jnp.float32)
    w1      = jax.random.normal(k2, (E, 2, D, F),  dtype=jnp.float32) * 0.02
    w2      = jax.random.normal(k3, (E, F, D),     dtype=jnp.float32) * 0.02
    gating  = jax.random.normal(k4, (T, E),        dtype=jnp.float32) * 0.1

    mesh = make_ep_mesh(EP)
    cfg  = dict(top_k=K, **MOE_CFG)

    results = {}

    # ------------------------------------------------------------------
    # 1. Forward-only: Pallas kernel
    # ------------------------------------------------------------------
    try:
        pallas_fwd = jax.jit(lambda: fused_ep_moe(
            mesh, tokens, w1, w2, gating, top_k=K, **MOE_CFG,
            ep_axis_name="model"))
        mean, std = timeit(pallas_fwd)
        results["pallas_fwd"] = mean
        print(f"  pallas_fwd          {mean:7.2f} ± {std:.2f} ms")
    except Exception as e:
        print(f"  pallas_fwd          ERROR: {e}")
        results["pallas_fwd"] = None

    # ------------------------------------------------------------------
    # 2. Forward-only: JAX vmap (ref_moe_with_residuals)
    # ------------------------------------------------------------------
    jax_fwd = jax.jit(lambda: ref_moe_with_residuals(
        tokens, w1, w2, gating, **cfg))
    mean, std = timeit(jax_fwd)
    results["jax_vmap_fwd"] = mean
    print(f"  jax_vmap_fwd        {mean:7.2f} ± {std:.2f} ms")

    if results["pallas_fwd"] is not None:
        speedup = results["jax_vmap_fwd"] / results["pallas_fwd"]
        print(f"  pallas speedup (fwd only): {speedup:.1f}×")

    # ------------------------------------------------------------------
    # 3. Training step: Stage A (JAX autograd through ref_moe_with_residuals)
    # ------------------------------------------------------------------
    fn_a = make_fused_ep_moe_train(**cfg)
    grad_a = jax.jit(jax.grad(
        lambda t, w1, w2, g: fn_a(t, w1, w2, g).sum(),
        argnums=(0, 1, 2, 3)))
    mean, std = timeit(lambda: grad_a(tokens, w1, w2, gating))
    results["stage_a_step"] = mean
    print(f"  stage_A training    {mean:7.2f} ± {std:.2f} ms  (1× JAX fwd + 1× JAX bwd)")

    # ------------------------------------------------------------------
    # 4. Training step: Stage B (Pallas fwd + JAX backward)
    # ------------------------------------------------------------------
    try:
        fn_b = make_fused_ep_moe_train_v2(mesh, **cfg)
        grad_b = jax.jit(jax.grad(
            lambda t, w1, w2, g: fn_b(t, w1, w2, g).sum(),
            argnums=(0, 1, 2, 3)))
        mean, std = timeit(lambda: grad_b(tokens, w1, w2, gating))
        results["stage_b_step"] = mean
        print(f"  stage_B training    {mean:7.2f} ± {std:.2f} ms  "
              f"(1× Pallas fwd + 1× JAX fwd + 1× JAX bwd)")
        overhead = results["stage_b_step"] / results["stage_a_step"]
        print(f"  stage_B overhead vs stage_A: {overhead:.2f}×")
    except Exception as e:
        print(f"  stage_B training    ERROR: {e}")
        results["stage_b_step"] = None

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    if results["pallas_fwd"] and results["stage_a_step"]:
        ratio = results["stage_a_step"] / results["pallas_fwd"]
        print(f"\n  JAX training step is {ratio:.1f}× the Pallas fwd kernel alone")
        print(f"  → Stage C target (Pallas fwd + Pallas bwd) should be ~{2*results['pallas_fwd']:.1f} ms "
              f"(assuming bwd ≈ fwd)")
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    n_devices = len(jax.devices())
    print(f"JAX devices ({n_devices}): {jax.devices()}")
    print(f"Backend: {jax.default_backend()}")

    all_results = {}
    for config in CONFIGS:
        label, T, D, F, E, K, EP = config
        if EP > n_devices:
            print(f"\nSkipping {label} (needs {EP} devices, have {n_devices})")
            continue
        try:
            all_results[label] = run_config(label, T, D, F, E, K, EP)
        except Exception as e:
            print(f"\nConfig {label} FAILED: {e}")

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------
    print(f"\n{'='*70}")
    print(f"{'Config':<16} {'pallas_fwd':>12} {'jax_fwd':>10} {'stageA_step':>13} {'stageB_step':>13}")
    print(f"{'─'*70}")
    for label, r in all_results.items():
        pf = f"{r['pallas_fwd']:.1f}" if r.get("pallas_fwd") else "N/A"
        jf = f"{r['jax_vmap_fwd']:.1f}" if r.get("jax_vmap_fwd") else "N/A"
        sa = f"{r['stage_a_step']:.1f}" if r.get("stage_a_step") else "N/A"
        sb = f"{r['stage_b_step']:.1f}" if r.get("stage_b_step") else "N/A"
        print(f"{label:<16} {pf:>12} ms {jf:>7} ms {sa:>10} ms {sb:>10} ms")
    print(f"{'='*70}")
    print("\nNote: Stage B = Pallas fwd (output) + JAX vmap fwd (residuals) + JAX bwd.")
    print("      Stage B is slower than Stage A — it only proves Pallas wires correctly.")
    print("      Stage C target: Pallas fwd-only + Pallas bwd ≈ 2–3× pallas_fwd latency.")
