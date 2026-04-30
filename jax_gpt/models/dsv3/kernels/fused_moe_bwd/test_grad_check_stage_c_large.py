"""Stage C gradient check at larger dims — validates VMEM budget and correctness
before benchmarking on a 4x4x4 v7x slice.

Tests EP=1 only at D=F=512 (the max that fits in 16 MB VMEM with the current
single-block-per-expert kernel design):

  6 weight VMEM buffers × 512×512×4 B  = 6.0 MB
  3 token  VMEM buffers × 128×512×4 B  = 0.75 MB (bte=128)
  Total                                 ≈ 6.75 MB  < 16 MB  ✓

EP>1 backward is NOT yet implemented (Stage D).  Those configs are skipped with
an explicit message rather than giving silently wrong gradients.

Run with:
  /home/sivaibhav_google_com/xdb/.xprof/bin/python3 test_grad_check_stage_c_large.py
"""

import sys
import numpy as np
import env  # noqa: F401

import jax
import jax.numpy as jnp

from tpu_inference.kernels.fused_moe.v1.kernel import ref_moe
from .backward import make_fused_ep_moe_train_v3


def norm_parity(a, b, name="", tol=0.02):
    na = float(jnp.linalg.norm(jnp.asarray(a)))
    nb = float(jnp.linalg.norm(jnp.asarray(b)))
    ratio = na / (nb + 1e-12)
    ok = abs(ratio - 1.0) < tol
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}  norm_ratio={ratio:.6f}"
          f"  (ours={na:.4f}, ref={nb:.4f})")
    return ok


def make_ep_mesh(ep_size):
    devices = jax.devices()
    assert len(devices) >= ep_size
    return jax.sharding.Mesh(
        np.array(devices[:ep_size]).reshape(1, ep_size), ("data", "model"))


def check_grads(mesh, config, T, D, F, E):
    key = jax.random.PRNGKey(42)
    k1, k2, k3, k4 = jax.random.split(key, 4)
    tokens  = jax.random.normal(k1, (T, D), dtype=jnp.float32)
    w1      = jax.random.normal(k2, (E, 2, D, F), dtype=jnp.float32) * 0.02
    w2      = jax.random.normal(k3, (E, F, D),    dtype=jnp.float32) * 0.02
    gating  = jax.random.normal(k4, (T, E),        dtype=jnp.float32) * 0.1

    ref_grads = jax.grad(
        lambda t, w1, w2, g: ref_moe(t, w1, w2, g, **config).sum(),
        argnums=(0, 1, 2, 3))(tokens, w1, w2, gating)

    fn = make_fused_ep_moe_train_v3(mesh, **config)
    v3_grads = jax.grad(
        lambda t, w1, w2, g: fn(t, w1, w2, g).sum(),
        argnums=(0, 1, 2, 3))(tokens, w1, w2, gating)

    ok_tok  = norm_parity(v3_grads[0], ref_grads[0], "d_tokens")
    ok_gate = norm_parity(v3_grads[3], ref_grads[3], "d_gating")
    ok_w1   = norm_parity(v3_grads[1], ref_grads[1], "d_w1   ")
    ok_w2   = norm_parity(v3_grads[2], ref_grads[2], "d_w2   ")
    return all([ok_tok, ok_gate, ok_w1, ok_w2])


def run_test(name, mesh, config, T, D, F, E):
    print(f"\n=== {name} ===")
    return check_grads(mesh, config, T=T, D=D, F=F, E=E)


if __name__ == "__main__":
    n_dev = len(jax.devices())
    print(f"JAX devices ({n_dev}): {jax.devices()}")
    print(f"Backend: {jax.default_backend()}")
    print()
    print("Stage C backward is EP=1 only.  EP>1 configs are skipped.")
    print("VMEM budget at D=F=512: ~6.75 MB (limit 16 MB).")

    results = {}
    mesh_ep1 = make_ep_mesh(1)

    # ------------------------------------------------------------------
    # D=128 smoke tests (same as test_grad_check_stage_c.py) — fast sanity
    # ------------------------------------------------------------------
    results["d128_sigmoid_renorm"] = run_test(
        "EP=1  D=128 F=128 T=64  E=8  K=2  sigmoid+renorm",
        mesh_ep1,
        dict(top_k=2, scoring_fn="sigmoid", renormalize_topk_logits=True),
        T=64, D=128, F=128, E=8,
    )

    # ------------------------------------------------------------------
    # D=512 tests — validates VMEM budget
    # ------------------------------------------------------------------
    results["d512_k2"] = run_test(
        "EP=1  D=512 F=512 T=256  E=16 K=2  sigmoid+renorm",
        mesh_ep1,
        dict(top_k=2, scoring_fn="sigmoid", renormalize_topk_logits=True),
        T=256, D=512, F=512, E=16,
    )

    results["d512_k8"] = run_test(
        "EP=1  D=512 F=512 T=512  E=64 K=8  sigmoid+renorm",
        mesh_ep1,
        dict(top_k=8, scoring_fn="sigmoid", renormalize_topk_logits=True),
        T=512, D=512, F=512, E=64,
    )

    results["d512_k8_large_t"] = run_test(
        "EP=1  D=512 F=512 T=1024 E=64 K=8  sigmoid+renorm",
        mesh_ep1,
        dict(top_k=8, scoring_fn="sigmoid", renormalize_topk_logits=True),
        T=1024, D=512, F=512, E=64,
    )

    # ------------------------------------------------------------------
    # EP>1 — skip, document why
    # ------------------------------------------------------------------
    print()
    print("=== EP>1 backward: SKIPPED (Stage D — reverse A2A not yet implemented) ===")
    print("  With EP>1, each device only sees its local experts' token gradients.")
    print("  d_tokens would be missing contributions from remote experts.")
    print("  Do NOT benchmark EP>1 Stage C backward until Stage D is complete.")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    all_pass = all(results.values())
    for name, ok in results.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    print("=" * 60)
    print(f"Overall: {'ALL PASSED' if all_pass else 'SOME FAILED'}")
    sys.exit(0 if all_pass else 1)
