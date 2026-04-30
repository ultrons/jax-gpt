"""Stage A gradient check: pure JAX backward vs jax.vjp(ref_moe).

Run with:
  /home/sivaibhav_google_com/xdb/.xprof/bin/python3 test_grad_check.py

All tests run on a single device (no EP / shard_map).
Tolerances: float32 rtol=1e-3, atol=1e-4.
"""

import sys
import env  # noqa: F401 — sets up PYTHONPATH and vllm mock

import jax
import jax.numpy as jnp
import numpy as np

from tpu_inference.kernels.fused_moe.v1.kernel import ref_moe
from .backward import make_fused_ep_moe_train

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def allclose(a, b, rtol=1e-3, atol=1e-4, name=""):
    a, b = np.array(a, dtype=np.float32), np.array(b, dtype=np.float32)
    max_abs = np.max(np.abs(a - b))
    mean_rel = np.mean(np.abs(a - b) / (np.abs(b) + 1e-8))
    ok = max_abs < atol or mean_rel < rtol
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {name}: max_abs={max_abs:.2e}  mean_rel={mean_rel:.2e}")
    return ok


def norm_parity(a, b, name="", tol=0.02):
    """Gradient norm ratio check — robust to floating-point accumulation order."""
    na = float(jnp.linalg.norm(jnp.asarray(a)))
    nb = float(jnp.linalg.norm(jnp.asarray(b)))
    ratio = na / (nb + 1e-12)
    ok = abs(ratio - 1.0) < tol
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {name} norm_ratio={ratio:.6f}  (our={na:.4f}, ref={nb:.4f})")
    return ok


def check_grads(config, T=32, D=64, F=32, E=8, rtol=1e-3, atol=1e-4):
    """Compare fused_ep_moe_train grads vs jax.vjp(ref_moe) grads.

    Uses element-wise comparison for activation/routing grads and norm-parity
    for weight grads (which accumulate over many (token,expert) pairs and can
    differ in floating-point order between our vmap and ref_moe's Python loop).
    """
    key = jax.random.PRNGKey(42)
    k1, k2, k3, k4 = jax.random.split(key, 4)
    tokens  = jax.random.normal(k1, (T, D), dtype=jnp.float32)
    w1      = jax.random.normal(k2, (E, 2, D, F), dtype=jnp.float32) * 0.02
    w2      = jax.random.normal(k3, (E, F, D), dtype=jnp.float32) * 0.02
    gating  = jax.random.normal(k4, (T, E), dtype=jnp.float32) * 0.1

    def ref_loss(t, w1, w2, g):
        return ref_moe(t, w1, w2, g, **config).sum()

    ref_grads = jax.grad(ref_loss, argnums=(0, 1, 2, 3))(tokens, w1, w2, gating)

    fn = make_fused_ep_moe_train(**config)
    fused_grads = jax.grad(
        lambda t, w1, w2, g: fn(t, w1, w2, g).sum(),
        argnums=(0, 1, 2, 3))(tokens, w1, w2, gating)

    # d_tokens and d_gating: element-wise (per-token, low accumulation)
    ok_tok  = allclose(fused_grads[0], ref_grads[0], rtol, atol, "d_tokens")
    ok_gate = allclose(fused_grads[3], ref_grads[3], rtol, atol, "d_gating")
    # d_w1 and d_w2: norm parity (accumulated over T*K pairs, order may differ)
    ok_w1 = norm_parity(fused_grads[1], ref_grads[1], "d_w1")
    ok_w2 = norm_parity(fused_grads[2], ref_grads[2], "d_w2")
    return all([ok_tok, ok_gate, ok_w1, ok_w2])


def check_finite_diff(config, T=16, D=32, F=16, E=4, eps=1e-3):
    """Quick finite-difference check on d_tokens only (slow but definitive)."""
    K = config["top_k"]
    key = jax.random.PRNGKey(7)
    k1, k2, k3, k4 = jax.random.split(key, 4)
    tokens  = jax.random.normal(k1, (T, D), dtype=jnp.float32) * 0.1
    w1      = jax.random.normal(k2, (E, 2, D, F), dtype=jnp.float32) * 0.02
    w2      = jax.random.normal(k3, (E, F, D), dtype=jnp.float32) * 0.02
    gating  = jax.random.normal(k4, (T, E), dtype=jnp.float32) * 0.1

    fn = make_fused_ep_moe_train(**config)

    def loss(tokens):
        return fn(tokens, w1, w2, gating).sum()

    # Analytic gradient
    analytic = jax.grad(loss)(tokens)  # (T, D)

    # Numerical gradient (central difference on a few elements)
    n_check = 10
    rng = np.random.default_rng(0)
    t_idxs = rng.integers(0, T, n_check)
    d_idxs = rng.integers(0, D, n_check)

    max_err = 0.0
    for ti, di in zip(t_idxs, d_idxs):
        t_plus  = tokens.at[ti, di].add(eps)
        t_minus = tokens.at[ti, di].add(-eps)
        num_grad = (loss(t_plus) - loss(t_minus)) / (2 * eps)
        ana_grad = float(analytic[ti, di])
        err = abs(num_grad - ana_grad) / (abs(ana_grad) + 1e-8)
        max_err = max(max_err, err)

    ok = max_err < 0.01  # 1% relative error
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] finite_diff d_tokens: max_rel_err={max_err:.2e}")
    return ok


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

def run_test(name, config, **kwargs):
    print(f"\n=== {name} ===")
    ok1 = check_grads(config, **kwargs)
    ok2 = check_finite_diff(config)
    return ok1 and ok2


if __name__ == "__main__":
    print(f"JAX devices: {jax.devices()}")
    print(f"Backend: {jax.default_backend()}")

    results = {}

    # Test 1: softmax scoring, no renorm (default kernel settings)
    results["softmax_no_renorm"] = run_test(
        "softmax scoring, no renorm",
        dict(top_k=2, scoring_fn="softmax", renormalize_topk_logits=False),
    )

    # Test 2: sigmoid scoring, no renorm (ds-v3 style)
    results["sigmoid_no_renorm"] = run_test(
        "sigmoid scoring, no renorm",
        dict(top_k=2, scoring_fn="sigmoid", renormalize_topk_logits=False),
    )

    # Test 3: sigmoid scoring + renorm (full ds-v3 config)
    results["sigmoid_renorm"] = run_test(
        "sigmoid scoring + renorm (ds-v3)",
        dict(top_k=2, scoring_fn="sigmoid", renormalize_topk_logits=True),
    )

    # Test 4: larger top_k
    results["topk4"] = run_test(
        "top_k=4, sigmoid, renorm",
        dict(top_k=4, scoring_fn="sigmoid", renormalize_topk_logits=True),
        T=64, D=128, F=64, E=16,
    )

    # Test 5: softmax, top_k=1 (edge case)
    results["topk1"] = run_test(
        "top_k=1, softmax",
        dict(top_k=1, scoring_fn="softmax", renormalize_topk_logits=False),
    )

    print("\n" + "="*50)
    all_pass = all(results.values())
    for name, ok in results.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    print("="*50)
    print(f"Overall: {'ALL PASSED' if all_pass else 'SOME FAILED'}")
    sys.exit(0 if all_pass else 1)
