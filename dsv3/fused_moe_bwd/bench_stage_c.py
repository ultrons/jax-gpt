"""Stage C performance benchmark: Pallas fwd + Pallas bwd vs Stage A (JAX fwd+bwd).

Measures wall-clock time for a full training step (forward + backward) and
breaks down the Pallas forward kernel time for comparison.

EP=1 only — Stage C backward does not yet support EP>1 (no reverse A2A).

VMEM constraint: D×F×6×4 B < 16 MB → max D=F=512 without intra-expert tiling.

Run with:
  /home/sivaibhav_google_com/xdb/.xprof/bin/python3 bench_stage_c.py

Works on local 4× TPU v4 and on 4×4×4 v7x (use the v7x configs at the bottom).
"""

import time
import sys
import numpy as np
import env  # noqa: F401

import jax
import jax.numpy as jnp

from tpu_inference.kernels.fused_moe.v1.kernel import fused_ep_moe
from backward import make_fused_ep_moe_train, make_fused_ep_moe_train_v3


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def timeit(fn, n_warmup=3, n_iter=10):
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


MOE_CFG = dict(scoring_fn="sigmoid", renormalize_topk_logits=True)


# ---------------------------------------------------------------------------
# Per-config benchmark
# ---------------------------------------------------------------------------

def run_config(label, T, D, F, E, K):
    """EP=1 only: Stage A (JAX) vs Stage C (Pallas fwd + Pallas bwd)."""
    print(f"\n{'─'*62}")
    print(f"  {label}")
    print(f"  T={T}  D={D}  F={F}  E={E}  K={K}  EP=1")
    vmem_mb = 6 * D * F * 4 / 1024 / 1024 + 3 * 128 * D * 4 / 1024 / 1024
    print(f"  VMEM weight+tok buffers ≈ {vmem_mb:.1f} MB  (limit 16 MB)")
    print(f"{'─'*62}")

    key = jax.random.PRNGKey(0)
    k1, k2, k3, k4 = jax.random.split(key, 4)
    tokens = jax.random.normal(k1, (T, D),       dtype=jnp.float32)
    w1     = jax.random.normal(k2, (E, 2, D, F), dtype=jnp.float32) * 0.02
    w2     = jax.random.normal(k3, (E, F, D),    dtype=jnp.float32) * 0.02
    gating = jax.random.normal(k4, (T, E),        dtype=jnp.float32) * 0.1

    mesh = make_ep_mesh(1)
    cfg  = dict(top_k=K, **MOE_CFG)
    results = {}

    # ---- Pallas forward-only (baseline reference) ----
    try:
        pallas_fwd_fn = jax.jit(lambda: fused_ep_moe(
            mesh, tokens, w1, w2, gating, top_k=K, **MOE_CFG, ep_axis_name="model"))
        mean, std = timeit(pallas_fwd_fn)
        results["pallas_fwd"] = mean
        print(f"  pallas_fwd (fwd only)   {mean:7.2f} ± {std:.2f} ms")
    except Exception as e:
        print(f"  pallas_fwd              ERROR: {e}")
        results["pallas_fwd"] = None

    # ---- Stage A: JAX fwd + JAX bwd ----
    fn_a   = make_fused_ep_moe_train(**cfg)
    grad_a = jax.jit(jax.grad(
        lambda t, w1, w2, g: fn_a(t, w1, w2, g).sum(), argnums=(0, 1, 2, 3)))
    mean, std = timeit(lambda: grad_a(tokens, w1, w2, gating))
    results["stage_a"] = mean
    print(f"  stage_A (JAX fwd+bwd)   {mean:7.2f} ± {std:.2f} ms")

    # ---- Stage C: Pallas fwd + Pallas bwd ----
    try:
        fn_c   = make_fused_ep_moe_train_v3(mesh, **cfg)
        grad_c = jax.jit(jax.grad(
            lambda t, w1, w2, g: fn_c(t, w1, w2, g).sum(), argnums=(0, 1, 2, 3)))
        mean, std = timeit(lambda: grad_c(tokens, w1, w2, gating))
        results["stage_c"] = mean
        print(f"  stage_C (Pallas fwd+bwd){mean:7.2f} ± {std:.2f} ms")
    except Exception as e:
        print(f"  stage_C                 ERROR: {e}")
        results["stage_c"] = None

    # ---- Ratios ----
    if results.get("pallas_fwd") and results.get("stage_a"):
        print(f"\n  stage_A / pallas_fwd  = {results['stage_a'] / results['pallas_fwd']:.1f}×  "
              f"(JAX overhead)")
    if results.get("pallas_fwd") and results.get("stage_c"):
        print(f"  stage_C / pallas_fwd  = {results['stage_c'] / results['pallas_fwd']:.1f}×  "
              f"(ideal target: ~2×)")
    if results.get("stage_a") and results.get("stage_c"):
        speedup = results["stage_a"] / results["stage_c"]
        print(f"  stage_A / stage_C     = {speedup:.1f}×  ← training step speedup")

    return results


# ---------------------------------------------------------------------------
# Config tables
# ---------------------------------------------------------------------------

# Configs that run on local 4× v4 (D≤512, fits VMEM)
LOCAL_CONFIGS = [
    # label                          T     D    F    E   K
    ("tiny   D=128",                 64,  128,  128,  8,  2),
    ("small  D=512 K=2",            256,  512,  512, 16,  2),
    ("medium D=512 K=8",            512,  512,  512, 64,  8),
    ("large  D=512 K=8",           1024,  512,  512, 64,  8),
]

# Additional configs for 4×4×4 v7x (more tokens, same D/F limit)
V7X_CONFIGS = [
    # label                          T     D    F    E   K
    ("v7x  D=512 T=2048 K=8",      2048,  512,  512, 64,  8),
    ("v7x  D=512 T=4096 K=8",      4096,  512,  512, 64,  8),
]


if __name__ == "__main__":
    n_dev = len(jax.devices())
    print(f"JAX devices ({n_dev}): {jax.devices()}")
    print(f"Backend: {jax.default_backend()}")
    print()
    print("Stage C backward is EP=1 only (Stage D will add EP>1 reverse A2A).")
    print("VMEM limit: D×F×6×4 B < 16 MB  →  max D=F=512 today.")

    configs = LOCAL_CONFIGS[:]
    if n_dev >= 64:
        configs += V7X_CONFIGS
    else:
        print(f"\n(v7x configs need 64 devices; have {n_dev} — skipping those)")

    all_results = {}
    for args in configs:
        label, T, D, F, E, K = args
        try:
            all_results[label] = run_config(label, T, D, F, E, K)
        except Exception as e:
            print(f"\n{label} FAILED: {e}")
            import traceback; traceback.print_exc()

    # ---- Summary table ----
    print(f"\n{'='*72}")
    print(f"{'Config':<26} {'pallas_fwd':>11} {'stage_A':>10} {'stage_C':>10} {'speedup':>9}")
    print(f"{'─'*72}")
    for label, r in all_results.items():
        pf = f"{r['pallas_fwd']:.1f}" if r.get("pallas_fwd") else "N/A"
        sa = f"{r['stage_a']:.1f}"    if r.get("stage_a")    else "N/A"
        sc = f"{r['stage_c']:.1f}"    if r.get("stage_c")    else "N/A"
        if r.get("stage_a") and r.get("stage_c"):
            sp = f"{r['stage_a'] / r['stage_c']:.1f}×"
        else:
            sp = "N/A"
        print(f"{label:<26} {pf:>8} ms {sa:>7} ms {sc:>7} ms {sp:>9}")
    print(f"{'='*72}")
    print("\nstage_A = JAX fwd+bwd (1× JAX fwd + 1× JAX bwd)")
    print("stage_C = Pallas fwd + Pallas bwd  (EP=1)")
    print("speedup = stage_A / stage_C")
