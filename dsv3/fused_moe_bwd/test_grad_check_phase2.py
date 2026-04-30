"""Phase 2 gradient check: Pallas forward + JAX backward (make_fused_ep_moe_train_v2).

Tests that the v2 training wrapper (fused_ep_moe Pallas forward + JAX autodiff
backward) produces correct gradients across DSv3 mini/medium/full dims, EP=1.

Phase 2 is the correctness baseline that validates the integration plumbing
(custom_vjp wiring, residual computation, EP=1 shard_map) before committing
to the full Pallas backward (Phase 3 / Stage C kernel).

Configs under test:
  tiny:   D=512,  D_moe=256,  E=8,  K=2  — fast sanity check
  mini:   D=2048, D_moe=1024, E=16, K=4  — DS-v3 mini dims
  medium: D=4096, D_moe=2048, E=64, K=8  — DS-v3 medium dims

Run with:
  /home/sivaibhav_google_com/xdb/.xprof/bin/python3 test_grad_check_phase2.py
"""

import sys
import types
import logging
import time
import numpy as np

# Bootstrap: mock vllm (needed by tpu_inference/logger.py) and add tpu-inference
# to sys.path for local dev.  In container, tpu_inference is already at /app/.
_vllm = types.ModuleType("vllm")
_vllm_logger = types.ModuleType("vllm.logger")

class _VllmLogger(logging.Logger):
    def warning_once(self, msg, *args, **kwargs): self.warning(msg, *args, **kwargs)
    def info_once(self, msg, *args, **kwargs): self.info(msg, *args, **kwargs)
    def debug_once(self, msg, *args, **kwargs): self.debug(msg, *args, **kwargs)

logging.setLoggerClass(_VllmLogger)
_vllm_logger.init_logger = lambda name: logging.getLogger(name)
_vllm_logger._VllmLogger = _VllmLogger
_vllm_logger.init_vllm_logger = lambda name: logging.getLogger(name)
sys.modules.setdefault("vllm", _vllm)
sys.modules.setdefault("vllm.logger", _vllm_logger)

# Local dev only: add tpu-inference path (container has /app/tpu_inference).
_TPU_INFERENCE = "/home/sivaibhav_google_com/tpu-inference"
if _TPU_INFERENCE not in sys.path:
    sys.path.insert(0, _TPU_INFERENCE)

import jax
import jax.numpy as jnp

from tpu_inference.kernels.fused_moe.v1.kernel import ref_moe
from backward import make_fused_ep_moe_train_v2


def norm_parity(a, b, name="", tol=0.02):
    na = float(jnp.linalg.norm(jnp.asarray(a)))
    nb = float(jnp.linalg.norm(jnp.asarray(b)))
    ratio = na / (nb + 1e-12)
    ok = abs(ratio - 1.0) < tol
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}  norm_ratio={ratio:.6f}"
          f"  (v2={na:.4f}, ref={nb:.4f})")
    return ok


def make_ep1_mesh():
    # Use the first LOCAL device so each pod in a multi-host job runs
    # independently on its own chip (no cross-pod collectives needed for EP=1).
    device = jax.local_devices()[0]
    return jax.sharding.Mesh(
        np.array([[device]]), ("data", "model"))


def check_grads(mesh, T, D, F, E, K, scoring_fn, renormalize_topk_logits):
    """Compare v2 gradients against ref_moe on EP=1."""
    key = jax.random.PRNGKey(42)
    k1, k2, k3, k4 = jax.random.split(key, 4)
    tokens  = jax.random.normal(k1, (T, D),        dtype=jnp.float32)
    w1      = jax.random.normal(k2, (E, 2, D, F),  dtype=jnp.float32) * 0.02
    w2      = jax.random.normal(k3, (E, F, D),     dtype=jnp.float32) * 0.02
    gating  = jax.random.normal(k4, (T, E),         dtype=jnp.float32) * 0.1

    cfg = dict(top_k=K, scoring_fn=scoring_fn,
               renormalize_topk_logits=renormalize_topk_logits)

    # Reference gradients via full JAX autodiff.
    ref_grads = jax.grad(
        lambda t, w1, w2, g: ref_moe(t, w1, w2, g, **cfg).sum(),
        argnums=(0, 1, 2, 3))(tokens, w1, w2, gating)

    # Phase 2: Pallas forward + JAX backward.
    fn = make_fused_ep_moe_train_v2(
        mesh, top_k=K,
        scoring_fn=scoring_fn,
        renormalize_topk_logits=renormalize_topk_logits,
        act_fn="silu",
        ep_axis_name="model",
    )
    v2_grads = jax.grad(
        lambda t, w1, w2, g: fn(t, w1, w2, g).sum(),
        argnums=(0, 1, 2, 3))(tokens, w1, w2, gating)

    ok_tok  = norm_parity(v2_grads[0], ref_grads[0], "d_tokens")
    ok_gate = norm_parity(v2_grads[3], ref_grads[3], "d_gating")
    ok_w1   = norm_parity(v2_grads[1], ref_grads[1], "d_w1   ")
    ok_w2   = norm_parity(v2_grads[2], ref_grads[2], "d_w2   ")
    return all([ok_tok, ok_gate, ok_w1, ok_w2])


def bench_fwd(mesh, T, D, F, E, K, n_warmup=3, n_iters=10):
    """Compare fused_ep_moe (Pallas) vs ref_moe (JAX) forward latency."""
    from tpu_inference.kernels.fused_moe.v1.kernel import fused_ep_moe

    key = jax.random.PRNGKey(7)
    k1, k2, k3, k4 = jax.random.split(key, 4)
    tokens = jax.random.normal(k1, (T, D), dtype=jnp.float32)
    w1     = jax.random.normal(k2, (E, 2, D, F), dtype=jnp.float32) * 0.02
    w2     = jax.random.normal(k3, (E, F, D),    dtype=jnp.float32) * 0.02
    gating = jax.random.normal(k4, (T, E),        dtype=jnp.float32) * 0.1

    cfg = dict(top_k=K, scoring_fn="sigmoid", renormalize_topk_logits=True)

    ref_fn  = jax.jit(lambda t, w1, w2, g: ref_moe(t, w1, w2, g, **cfg))
    pallas_fn = jax.jit(lambda t, w1, w2, g: fused_ep_moe(
        mesh, t, w1, w2, g, act_fn="silu", ep_axis_name="model", **cfg))

    for _ in range(n_warmup):
        jax.block_until_ready(ref_fn(tokens, w1, w2, gating))
        jax.block_until_ready(pallas_fn(tokens, w1, w2, gating))

    t0 = time.perf_counter()
    for _ in range(n_iters):
        jax.block_until_ready(ref_fn(tokens, w1, w2, gating))
    ref_ms = (time.perf_counter() - t0) / n_iters * 1e3

    t0 = time.perf_counter()
    for _ in range(n_iters):
        jax.block_until_ready(pallas_fn(tokens, w1, w2, gating))
    pallas_ms = (time.perf_counter() - t0) / n_iters * 1e3

    print(f"  fwd bench: ref={ref_ms:.1f}ms  pallas={pallas_ms:.1f}ms"
          f"  speedup={ref_ms/pallas_ms:.2f}x")


def run_test(name, mesh, **kwargs):
    print(f"\n=== {name} ===")
    return check_grads(mesh, **kwargs)


if __name__ == "__main__":
    n_dev = len(jax.local_devices())
    proc = jax.process_index()
    # In a 16-pod 4x4x4 slice each pod runs independently on its local devices.
    # Only pod 0 prints the header to avoid 16× duplicate output.
    if proc == 0:
        print(f"JAX local devices ({n_dev}): {jax.local_devices()}")
        print(f"Total devices across all processes: {jax.device_count()}")
        print(f"Backend: {jax.default_backend()}")

    mesh = make_ep1_mesh()
    results = {}

    # In multi-host: each pod runs the test independently on its local device.
    # Results are identical across pods (same random seeds, no communication).
    # Suppress output from non-zero processes to keep logs clean.

    # ---- Tiny: D=512, D_moe=256, E=8, K=2 ----
    results["tiny_sigmoid_renorm"] = run_test(
        "Tiny  D=512  F=256 T=64  E=8  K=2  sigmoid+renorm",
        mesh, T=64, D=512, F=256, E=8, K=2,
        scoring_fn="sigmoid", renormalize_topk_logits=True,
    )
    results["tiny_softmax"] = run_test(
        "Tiny  D=512  F=256 T=64  E=8  K=2  softmax",
        mesh, T=64, D=512, F=256, E=8, K=2,
        scoring_fn="softmax", renormalize_topk_logits=False,
    )

    # ---- Mini: D=2048, D_moe=1024, E=16, K=4 — DS-v3 mini dims ----
    results["mini_sigmoid_renorm"] = run_test(
        "Mini  D=2048 F=1024 T=64  E=16 K=4  sigmoid+renorm",
        mesh, T=64, D=2048, F=1024, E=16, K=4,
        scoring_fn="sigmoid", renormalize_topk_logits=True,
    )
    results["mini_sigmoid_renorm_t128"] = run_test(
        "Mini  D=2048 F=1024 T=128 E=16 K=4  sigmoid+renorm",
        mesh, T=128, D=2048, F=1024, E=16, K=4,
        scoring_fn="sigmoid", renormalize_topk_logits=True,
    )

    # ---- Medium: D=4096, D_moe=2048, E=64, K=8 — DS-v3 medium dims ----
    # T=64 keeps vmap weight gather at (512, 4096, 2048) = 16 GB, feasible.
    results["medium_sigmoid_renorm"] = run_test(
        "Medium D=4096 F=2048 T=64  E=64 K=8  sigmoid+renorm",
        mesh, T=64, D=4096, F=2048, E=64, K=8,
        scoring_fn="sigmoid", renormalize_topk_logits=True,
    )

    # ---- Forward speedup benchmark (pod 0 only, tiny config) ----
    if proc == 0:
        print("\n=== Forward speedup benchmark (tiny config) ===")
        bench_fwd(mesh, T=256, D=512, F=256, E=8, K=2)

    # ---- Summary (pod 0 only) ----
    all_pass = all(results.values())
    if proc == 0:
        print("\n" + "=" * 60)
        for name, ok in results.items():
            print(f"  {'PASS' if ok else 'FAIL'}  {name}")
        print("=" * 60)
        print(f"Overall: {'ALL PASSED' if all_pass else 'SOME FAILED'}")
    sys.exit(0 if all_pass else 1)
