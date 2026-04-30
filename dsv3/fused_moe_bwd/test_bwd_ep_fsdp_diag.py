"""Diagnostic: isolate EP vs FSDP contribution to gradient errors.

Runs four cases with 4 local devices:
  A. EP=2 FSDP=1  (EP-only, no FSDP sharding)
  B. EP=1 FSDP=2  (FSDP-only, no EP sharding)  — needs EP=1 forward path
  C. EP=2 FSDP=2  (both, 4 devices)
  D. EP=1 FSDP=1  (single-device baseline)

For each: compare version=1 (streaming bwd) vs version=0 (jax.vjp).
Expected: all should have norm_ratio≈1.0 if the backward is correct.

Run: python fused_moe_bwd/test_bwd_ep_fsdp_diag.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "mini_dsv3"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from model import _moe_jax_ep_fn


def make_sharded(x, mesh, spec):
    return jax.device_put(x, NamedSharding(mesh, spec))


def norm_parity(a, b, name="", tol=0.05):
    a = jnp.asarray(a).reshape(-1)
    b = jnp.asarray(b).reshape(-1)
    na, nb = float(jnp.linalg.norm(a)), float(jnp.linalg.norm(b))
    ratio = na / (nb + 1e-12)
    cos = float(jnp.dot(a, b) / (na * nb + 1e-12))
    ok = abs(ratio - 1.0) < tol
    print(f"  [{'OK' if ok else 'FAIL'}] {name:26s}  ratio={ratio:.4f}  cos={cos:.4f}")
    return ok


def run_case(label, EP, FSDP, T=32, D=64, F=32, E=4, K=2, seed=42):
    n = EP * FSDP
    if jax.local_device_count() < n:
        print(f"  [SKIP] {label}: need {n} devices")
        return None

    # Mesh: (FSDP, EP) named ("fsdp", "ep")
    devs = np.array(jax.local_devices()[:n]).reshape(FSDP, EP)
    mesh = Mesh(devs, ("fsdp", "ep"))

    act_spec = P(("ep", "fsdp"), None)
    ep_ax = "ep"
    T_fsdp = T // FSDP
    max_tpe = max(1, 2 * T_fsdp * K // E)

    rng = np.random.default_rng(seed)
    fx   = jnp.array(rng.standard_normal((T, D)).astype(np.float32))
    w0   = jnp.array(rng.standard_normal((E, D, F)).astype(np.float32) * 0.02)
    w1   = jnp.array(rng.standard_normal((E, D, F)).astype(np.float32) * 0.02)
    wout = jnp.array(rng.standard_normal((E, F, D)).astype(np.float32) * 0.02)
    # Uniform round-robin routing so every expert gets exactly T*K/E slots
    fi_flat = np.tile(np.arange(E), T * K // E + 1)[:T * K].reshape(T, K)
    rng.shuffle(fi_flat.ravel())
    fi = jnp.array(fi_flat.astype(np.int32))
    fw = jnp.ones((T, K), jnp.float32) / K

    fx_s   = make_sharded(fx,   mesh, act_spec)
    fi_s   = make_sharded(fi,   mesh, act_spec)
    fw_s   = make_sharded(fw,   mesh, act_spec)
    w0_s   = make_sharded(w0,   mesh, P("ep", None, "fsdp"))
    w1_s   = make_sharded(w1,   mesh, P("ep", None, "fsdp"))
    wout_s = make_sharded(wout, mesh, P("ep", "fsdp", None))

    def loss(fx_, fw_, w0_, w1_, wout_, ver):
        return _moe_jax_ep_fn(fx_, fi_s, fw_, w0_, w1_, wout_,
                               mesh, K, act_spec, ep_ax, max_tpe, ver).sum()

    ref = jax.grad(loss, argnums=(0, 1, 2, 3, 4))(fx_s, fw_s, w0_s, w1_s, wout_s, 0)
    v1  = jax.grad(loss, argnums=(0, 1, 2, 3, 4))(fx_s, fw_s, w0_s, w1_s, wout_s, 1)

    print(f"\n  {label} (EP={EP} FSDP={FSDP}):")
    ok = [
        norm_parity(v1[0], ref[0], "d_tokens"),
        norm_parity(v1[1], ref[1], "d_fw (routing)"),
        norm_parity(v1[2], ref[2], "d_w0"),
        norm_parity(v1[3], ref[3], "d_w1"),
        norm_parity(v1[4], ref[4], "d_wout"),
    ]
    return all(ok)


if __name__ == "__main__":
    print(f"JAX {jax.__version__}  devices={jax.local_device_count()}")

    results = {}

    # EP=1 FSDP=1 — sanity check (no sharding at all)
    results["ep1_fsdp1"] = run_case("EP=1 FSDP=1 (baseline)", EP=1, FSDP=1)

    if jax.local_device_count() >= 2:
        # EP=2 FSDP=1 — only EP sharding
        results["ep2_fsdp1"] = run_case("EP=2 FSDP=1", EP=2, FSDP=1)

        # EP=1 FSDP=2 — only FSDP sharding
        results["ep1_fsdp2"] = run_case("EP=1 FSDP=2", EP=1, FSDP=2)

    if jax.local_device_count() >= 4:
        # EP=2 FSDP=2 — full case
        results["ep2_fsdp2"] = run_case("EP=2 FSDP=2", EP=2, FSDP=2)

    print("\n" + "=" * 50)
    for k, v in results.items():
        lbl = "PASS" if v else ("SKIP" if v is None else "FAIL")
        print(f"  {lbl}  {k}")
    sys.exit(0 if all(v for v in results.values() if v is not None) else 1)
