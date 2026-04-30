"""Stage D gradient check: EP>1 backward (Pallas fwd + Pallas bwd with shard_map).

Verifies that make_fused_ep_moe_train_v3 with EP>1 produces correct gradients
by comparing against jax.grad(ref_moe).

EP>1 backward design:
  - shard_map in _bwd distributes w1/w2 by expert axis
  - fused_ep_moe_bwd detects E_local < E_global, uses lax.axis_index to get
    expert_offset, zeros non-local expert slots
  - lax.psum aggregates partial d_tokens and d_gating across EP devices

Run with:
  /home/sivaibhav_google_com/xdb/.xprof/bin/python3 test_grad_check_stage_d.py
"""

import sys
import numpy as np
import env  # noqa: F401

import jax
import jax.numpy as jnp

from tpu_inference.kernels.fused_moe.v1.kernel import ref_moe
from backward import make_fused_ep_moe_train_v3


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
    assert len(devices) >= ep_size, f"Need {ep_size} devices, have {len(devices)}"
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


def run_test(name, mesh, config, **kwargs):
    print(f"\n=== {name} ===")
    return check_grads(mesh, config, **kwargs)


if __name__ == "__main__":
    n_dev = len(jax.devices())
    print(f"JAX devices ({n_dev}): {jax.devices()}")
    print(f"Backend: {jax.default_backend()}")

    results = {}

    # ---- EP=1 regression: make sure EP=1 path still works ----
    mesh_ep1 = make_ep_mesh(1)
    results["ep1_sigmoid_renorm"] = run_test(
        "EP=1  D=128 T=64  E=8  K=2  sigmoid+renorm (regression)",
        mesh_ep1,
        dict(top_k=2, scoring_fn="sigmoid", renormalize_topk_logits=True),
        T=64, D=128, F=128, E=8,
    )

    # ---- EP=2 tests ----
    if n_dev >= 2:
        mesh_ep2 = make_ep_mesh(2)
        results["ep2_sigmoid_renorm_k2"] = run_test(
            "EP=2  D=128 T=64  E=8  K=2  sigmoid+renorm",
            mesh_ep2,
            dict(top_k=2, scoring_fn="sigmoid", renormalize_topk_logits=True),
            T=64, D=128, F=128, E=8,
        )
        results["ep2_sigmoid_renorm_k4"] = run_test(
            "EP=2  D=128 T=128 E=16 K=4  sigmoid+renorm",
            mesh_ep2,
            dict(top_k=4, scoring_fn="sigmoid", renormalize_topk_logits=True),
            T=128, D=128, F=128, E=16,
        )
        results["ep2_softmax_k2"] = run_test(
            "EP=2  D=128 T=64  E=8  K=2  softmax",
            mesh_ep2,
            dict(top_k=2, scoring_fn="softmax", renormalize_topk_logits=False),
            T=64, D=128, F=128, E=8,
        )
    else:
        print(f"\nSkipping EP=2 tests (need 2 devices, have {n_dev})")

    # ---- EP=4 tests ----
    if n_dev >= 4:
        mesh_ep4 = make_ep_mesh(4)
        results["ep4_sigmoid_renorm_k2"] = run_test(
            "EP=4  D=128 T=64  E=8  K=2  sigmoid+renorm",
            mesh_ep4,
            dict(top_k=2, scoring_fn="sigmoid", renormalize_topk_logits=True),
            T=64, D=128, F=128, E=8,
        )
        results["ep4_sigmoid_renorm_k8"] = run_test(
            "EP=4  D=128 T=512 E=64 K=8  sigmoid+renorm",
            mesh_ep4,
            dict(top_k=8, scoring_fn="sigmoid", renormalize_topk_logits=True),
            T=512, D=128, F=128, E=64,
        )
    else:
        print(f"\nSkipping EP=4 tests (need 4 devices, have {n_dev})")

    # ---- Full 671B config: EP=8, D=7168, F=2048, E=256, K=8 ----
    # Tests D-tiling (tile_D=1024 on v7x with 64 MB VMEM; auto-selected on v4).
    # T=64 keeps compilation fast; E_local = E/EP = 32 experts per device.
    # vmem_limit_bytes=64MB for v7x; 16MB for v4 → tile_D auto-selects smaller tile.
    if n_dev >= 8:
        mesh_ep8 = make_ep_mesh(8)
        # Detect VMEM limit from device kind
        device_kind = jax.devices()[0].device_kind.lower()
        vmem_limit = 64 * 1024 * 1024 if "7x" in device_kind else 16 * 1024 * 1024

        def make_fn_full(mesh):
            from backward import make_fused_ep_moe_train_v3
            return make_fused_ep_moe_train_v3(
                mesh, top_k=8,
                scoring_fn="sigmoid", renormalize_topk_logits=True, act_fn="silu",
                ep_axis_name="model",
                vmem_limit_bytes=vmem_limit,
            )

        print(f"\n=== EP=8  D=7168 F=2048 T=64 E=256 K=8  (full 671B config, vmem={vmem_limit//2**20}MB) ===")
        key = jax.random.PRNGKey(99)
        k1, k2, k3, k4 = jax.random.split(key, 4)
        T, D, F_full, E_full, K_full = 64, 7168, 2048, 256, 8
        tokens_full  = jax.random.normal(k1, (T, D), dtype=jnp.float32)
        w1_full      = jax.random.normal(k2, (E_full, 2, D, F_full), dtype=jnp.float32) * 0.02
        w2_full      = jax.random.normal(k3, (E_full, F_full, D), dtype=jnp.float32) * 0.02
        gating_full  = jax.random.normal(k4, (T, E_full), dtype=jnp.float32) * 0.1

        from tpu_inference.kernels.fused_moe.v1.kernel import ref_moe
        ref_grads_full = jax.grad(
            lambda t, w1, w2, g: ref_moe(
                t, w1, w2, g,
                top_k=K_full, scoring_fn="sigmoid", renormalize_topk_logits=True,
            ).sum(),
            argnums=(0, 1, 2, 3))(tokens_full, w1_full, w2_full, gating_full)

        fn_full = make_fn_full(mesh_ep8)
        v3_grads_full = jax.grad(
            lambda t, w1, w2, g: fn_full(t, w1, w2, g).sum(),
            argnums=(0, 1, 2, 3))(tokens_full, w1_full, w2_full, gating_full)

        ok_tok  = norm_parity(v3_grads_full[0], ref_grads_full[0], "d_tokens")
        ok_gate = norm_parity(v3_grads_full[3], ref_grads_full[3], "d_gating")
        ok_w1   = norm_parity(v3_grads_full[1], ref_grads_full[1], "d_w1   ")
        ok_w2   = norm_parity(v3_grads_full[2], ref_grads_full[2], "d_w2   ")
        results["ep8_full_671b"] = all([ok_tok, ok_gate, ok_w1, ok_w2])
    else:
        print(f"\nSkipping EP=8 full-671B test (need 8 devices, have {n_dev})")

    # ---- Summary ----
    print("\n" + "=" * 60)
    all_pass = all(results.values())
    for name, ok in results.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    print("=" * 60)
    print(f"Overall: {'ALL PASSED' if all_pass else 'SOME FAILED'}")
    sys.exit(0 if all_pass else 1)
