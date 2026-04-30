"""Stage C gradient check: Pallas forward + Pallas backward vs jax.grad(ref_moe).

Run with:
  /home/sivaibhav_google_com/xdb/.xprof/bin/python3 test_grad_check_stage_c.py

Constraints:
  hidden_size % 128 == 0, intermediate_size % 128 == 0
  num_tokens % ep_size == 0, num_experts % ep_size == 0
  D * F * 12 * 4 bytes < 16 MB VMEM  →  D=F=512 uses ~6 MB (fits)
"""

import sys
import numpy as np
import env  # noqa: F401

import jax
import jax.numpy as jnp

from tpu_inference.kernels.fused_moe.v1.kernel import ref_moe
from .backward import make_fused_ep_moe_train_v3


def allclose(a, b, rtol=1e-3, atol=1e-4, name=""):
    a, b = np.array(a, dtype=np.float32), np.array(b, dtype=np.float32)
    max_abs = np.max(np.abs(a - b))
    mean_rel = np.mean(np.abs(a - b) / (np.abs(b) + 1e-8))
    ok = max_abs < atol or mean_rel < rtol
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: max_abs={max_abs:.2e}  mean_rel={mean_rel:.2e}")
    return ok


def norm_parity(a, b, name="", tol=0.02):
    na = float(jnp.linalg.norm(jnp.asarray(a)))
    nb = float(jnp.linalg.norm(jnp.asarray(b)))
    ratio = na / (nb + 1e-12)
    ok = abs(ratio - 1.0) < tol
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} norm_ratio={ratio:.6f}  (ours={na:.4f}, ref={nb:.4f})")
    return ok


def make_ep_mesh(ep_size):
    devices = jax.devices()
    assert len(devices) >= ep_size
    return jax.sharding.Mesh(
        np.array(devices[:ep_size]).reshape(1, ep_size), ("data", "model"))


def check_grads(mesh, config, T=64, D=128, F=128, E=8):
    key = jax.random.PRNGKey(42)
    k1, k2, k3, k4 = jax.random.split(key, 4)
    tokens  = jax.random.normal(k1, (T, D), dtype=jnp.float32)
    w1      = jax.random.normal(k2, (E, 2, D, F), dtype=jnp.float32) * 0.02
    w2      = jax.random.normal(k3, (E, F, D), dtype=jnp.float32) * 0.02
    gating  = jax.random.normal(k4, (T, E), dtype=jnp.float32) * 0.1

    def ref_loss(t, w1, w2, g):
        return ref_moe(t, w1, w2, g, **config).sum()

    ref_grads = jax.grad(ref_loss, argnums=(0, 1, 2, 3))(tokens, w1, w2, gating)

    fn = make_fused_ep_moe_train_v3(mesh, **config)
    v3_grads = jax.grad(
        lambda t, w1, w2, g: fn(t, w1, w2, g).sum(),
        argnums=(0, 1, 2, 3))(tokens, w1, w2, gating)

    ok_tok  = norm_parity(v3_grads[0], ref_grads[0], "d_tokens")
    ok_gate = norm_parity(v3_grads[3], ref_grads[3], "d_gating")
    ok_w1   = norm_parity(v3_grads[1], ref_grads[1], "d_w1")
    ok_w2   = norm_parity(v3_grads[2], ref_grads[2], "d_w2")
    return all([ok_tok, ok_gate, ok_w1, ok_w2])


def run_test(name, mesh, config, **kwargs):
    print(f"\n=== {name} ===")
    return check_grads(mesh, config, **kwargs)


if __name__ == "__main__":
    print(f"JAX devices: {jax.devices()}")
    print(f"Backend: {jax.default_backend()}")

    results = {}
    mesh_ep1 = make_ep_mesh(1)
    print(f"\nMesh EP=1: {mesh_ep1}")

    results["softmax_no_renorm"] = run_test(
        "EP=1 softmax, no renorm",
        mesh_ep1,
        dict(top_k=2, scoring_fn="softmax", renormalize_topk_logits=False),
        T=64, D=128, F=128, E=8,
    )

    results["sigmoid_no_renorm"] = run_test(
        "EP=1 sigmoid, no renorm",
        mesh_ep1,
        dict(top_k=2, scoring_fn="sigmoid", renormalize_topk_logits=False),
        T=64, D=128, F=128, E=8,
    )

    results["sigmoid_renorm"] = run_test(
        "EP=1 sigmoid + renorm (ds-v3 style)",
        mesh_ep1,
        dict(top_k=2, scoring_fn="sigmoid", renormalize_topk_logits=True),
        T=64, D=128, F=128, E=8,
    )

    results["topk4"] = run_test(
        "EP=1 top_k=4, sigmoid, renorm",
        mesh_ep1,
        dict(top_k=4, scoring_fn="sigmoid", renormalize_topk_logits=True),
        T=128, D=128, F=128, E=16,
    )

    print("\n" + "=" * 50)
    all_pass = all(results.values())
    for name, ok in results.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    print("=" * 50)
    print(f"Overall: {'ALL PASSED' if all_pass else 'SOME FAILED'}")
    sys.exit(0 if all_pass else 1)
