"""Local gradient-correctness test for streaming_bwd v1 with EP=2, FSDP=2.

Exercises _streaming_bwd_fn in model.py end-to-end via jax.grad.
Requires exactly 4 JAX devices (e.g. 4 local v4 TPU chips).

Strategy:
  - Reference: streaming_bwd_version=0 (jax.vjp — always correct AD)
  - Test:      streaming_bwd_version=1 (streaming backward, fixed FSDP psum)
  Both run on the same EP=2 × FSDP=2 mesh so any mesh-handling bug shows up.

Run:
  python fused_moe_bwd/test_streaming_bwd_ep_fsdp_local.py
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "mini_dsv3"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from jax_gpt.models.dsv3.model import _moe_jax_ep_fn


def norm_parity(a, b, name="", tol=0.05):
    a = jnp.asarray(a).reshape(-1)
    b = jnp.asarray(b).reshape(-1)
    na = float(jnp.linalg.norm(a))
    nb = float(jnp.linalg.norm(b))
    ratio = na / (nb + 1e-12)
    ok = abs(ratio - 1.0) < tol
    # Also check elementwise agreement (cosine sim) to catch sign/scale bugs
    cos = float(jnp.dot(a, b) / (na * nb + 1e-12))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name:30s}  "
          f"norm_ratio={ratio:.4f}  cos={cos:.4f}  "
          f"(v1={na:.4f}, ref={nb:.4f})")
    return ok


def make_sharded(x, mesh, spec):
    sharding = NamedSharding(mesh, spec)
    return jax.device_put(x, sharding)


def run(EP=2, FSDP=2, T=32, D=64, F=32, E=4, K=2, seed=42):
    n_devices = EP * FSDP
    available = jax.local_device_count()
    if available < n_devices:
        print(f"  [SKIP] need {n_devices} devices, have {available}")
        return None

    # Mesh: (FSDP, EP) → axes ("fsdp", "ep")
    devices = np.array(jax.local_devices()[:n_devices]).reshape(FSDP, EP)
    mesh = Mesh(devices, ("fsdp", "ep"))

    act_spec    = P(("ep", "fsdp"), None)
    w_gate_spec = P("ep", None, "fsdp")   # (E, D, F)
    wout_spec   = P("ep", "fsdp", None)   # (E, F, D)
    ep_axis_name = "ep"
    T_fsdp = T // FSDP
    max_tpe = max(1, 2 * T_fsdp * K // E)

    rng = np.random.default_rng(seed)

    # Inputs (float32 throughout — streaming bwd kernel uses f32 internally)
    fx   = jnp.array(rng.standard_normal((T, D)).astype(np.float32))
    w0   = jnp.array(rng.standard_normal((E, D, F)).astype(np.float32) * 0.02)
    w1   = jnp.array(rng.standard_normal((E, D, F)).astype(np.float32) * 0.02)
    wout = jnp.array(rng.standard_normal((E, F, D)).astype(np.float32) * 0.02)

    # Routing: each token picks K experts; ensure no expert is completely empty.
    # Use a structured assignment (round-robin) so every expert gets T*K/E tokens.
    fi_flat = np.tile(np.arange(E), T * K // E + 1)[:T * K].reshape(T, K)
    rng.shuffle(fi_flat.ravel())
    fi = jnp.array(fi_flat.astype(np.int32))
    # Routing weights: uniform (simplifies reference comparison)
    fw = jnp.ones((T, K), dtype=jnp.float32) / K

    # Shard
    fx_s   = make_sharded(fx,   mesh, act_spec)
    fi_s   = make_sharded(fi,   mesh, act_spec)
    fw_s   = make_sharded(fw,   mesh, act_spec)
    w0_s   = make_sharded(w0,   mesh, w_gate_spec)
    w1_s   = make_sharded(w1,   mesh, w_gate_spec)
    wout_s = make_sharded(wout, mesh, wout_spec)

    # Loss: sum of all outputs (gradient = all-ones tensor, uniform across devices)
    def loss(fx_, fw_, w0_, w1_, wout_, version):
        out = _moe_jax_ep_fn(
            fx_, fi_s, fw_, w0_, w1_, wout_,
            mesh, K, act_spec, ep_axis_name, max_tpe, version)
        return out.sum()

    grad_fn_ref = jax.grad(loss, argnums=(0, 1, 2, 3, 4))
    grad_fn_v1  = jax.grad(loss, argnums=(0, 1, 2, 3, 4))

    print("  Computing reference (version=0, jax.vjp) ...")
    ref = grad_fn_ref(fx_s, fw_s, w0_s, w1_s, wout_s, 0)
    d_fx_ref, d_fw_ref, d_w0_ref, d_w1_ref, d_wout_ref = ref

    print("  Computing v1 (version=1, streaming_bwd, fixed FSDP psum) ...")
    v1 = grad_fn_v1(fx_s, fw_s, w0_s, w1_s, wout_s, 1)
    d_fx_v1, d_fw_v1, d_w0_v1, d_w1_v1, d_wout_v1 = v1

    print()
    ok_fx   = norm_parity(d_fx_v1,   d_fx_ref,   "d_tokens")
    ok_fw   = norm_parity(d_fw_v1,   d_fw_ref,   "d_router_weights")
    ok_w0   = norm_parity(d_w0_v1,   d_w0_ref,   "d_w0 (gate proj)")
    ok_w1   = norm_parity(d_w1_v1,   d_w1_ref,   "d_w1 (up proj)")
    ok_wout = norm_parity(d_wout_v1, d_wout_ref, "d_wout (down proj)")

    return all([ok_fx, ok_fw, ok_w0, ok_w1, ok_wout])


if __name__ == "__main__":
    print(f"JAX: {jax.__version__}  devices: {jax.local_device_count()}")
    print(f"Backend: {jax.default_backend()}")
    print()

    if jax.local_device_count() < 4:
        print("SKIP: need ≥4 local devices for EP=2 × FSDP=2 test")
        sys.exit(0)

    results = {}

    print("=== EP=2 FSDP=2 | T=32 D=64 F=32 E=4 K=2 ===")
    results["ep2_fsdp2_tiny"] = run(EP=2, FSDP=2, T=32, D=64, F=32, E=4, K=2)

    print()
    print("=== EP=2 FSDP=2 | T=64 D=256 F=128 E=8 K=4 ===")
    results["ep2_fsdp2_medium"] = run(EP=2, FSDP=2, T=64, D=256, F=128, E=8, K=4)

    if jax.local_device_count() >= 8:
        print()
        print("=== EP=4 FSDP=2 | T=64 D=256 F=128 E=8 K=4 ===")
        results["ep4_fsdp2"] = run(EP=4, FSDP=2, T=64, D=256, F=128, E=8, K=4)

        print()
        print("=== EP=2 FSDP=4 | T=64 D=256 F=128 E=8 K=4 ===")
        results["ep2_fsdp4"] = run(EP=2, FSDP=4, T=64, D=256, F=128, E=8, K=4)

    print()
    all_pass = all(v for v in results.values() if v is not None)
    print("=" * 60)
    for name, ok in results.items():
        label = "PASS" if ok else ("SKIP" if ok is None else "FAIL")
        print(f"  {label}  {name}")
    print("=" * 60)
    print(f"Overall: {'ALL PASSED' if all_pass else 'SOME FAILED'}")

    if not all_pass:
        print()
        print("DIAGNOSIS: if d_tokens FAIL with norm_ratio >> 1, FSDP psum still wrong.")
        print("If d_tokens FAIL with norm_ratio << 1, missing psum somewhere.")

    sys.exit(0 if all_pass else 1)
