"""Local repro for streaming_bwd v1 gradient bug + v2 fix validation.

Tests three configurations on 4 JAX devices:
  Case A: FSDP=2, EP=2, streaming_bwd v1  → FAIL (root cause: partial D_moe d_tok)
  Case B: FSDP=1, EP=4, streaming_bwd v1  → PASS (no FSDP → no partiality)
  Case C: FSDP=2, EP=2, streaming_bwd v2  → PASS (v2 all_gathers full D_moe weights)

Run:
    /home/sivaibhav_google_com/xdb/.venv/bin/python test_streaming_bwd_local.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "mini_dsv3"))

import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

# ── Config ─────────────────────────────────────────────────────────────────────
D     = 64
D_moe = 32
E     = 4      # total experts
K     = 2      # top-K routing
T     = 16     # total tokens
SEED  = 0

# ── Mesh helpers ───────────────────────────────────────────────────────────────

def make_mesh(fsdp, ep):
    devs = jax.devices()
    n = len(devs)
    assert n >= fsdp * ep, f"Need {fsdp*ep} devices, got {n}"
    arr = np.array(devs[:fsdp * ep]).reshape(fsdp, ep)
    return Mesh(arr, ("fsdp", "ep"))

def make_sharded(data, mesh, spec):
    sharding = NamedSharding(mesh, spec)
    idx_map = sharding.addressable_devices_indices_map(data.shape)
    # Use only devices in the mesh (idx_map keys), not all jax.local_devices()
    shards = [jax.device_put(data[idx_map[d]], d) for d in idx_map]
    return jax.make_array_from_single_device_arrays(data.shape, sharding, shards)

# ── Run one (mesh, version) config ─────────────────────────────────────────────

def run_version(mesh, ep, fsdp, streaming_bwd_version, fx, fi, fw, w0, w1, wout):
    from jax_gpt.models.dsv3.model import _moe_jax_ep_fn
    act_spec  = P(("ep", "fsdp"), None)
    ep_axis   = "ep"
    T_per_dev = T // fsdp           # tokens visible per device after EP all_gather
    E_local   = E // ep
    avg_tpe   = T_per_dev * K // E_local
    max_tpe   = min(T_per_dev * K, max(4, 2 * avg_tpe))

    def loss_fn(fx_, fw_, w0_, w1_, wout_):
        out = _moe_jax_ep_fn(fx_, fi, fw_, w0_, w1_, wout_,
                              mesh, K, act_spec, ep_axis, max_tpe,
                              streaming_bwd_version)
        return jnp.sum(out)

    val, grads = jax.value_and_grad(loss_fn, argnums=(0, 1, 2, 3, 4))(
        fx, fw, w0, w1, wout
    )
    return float(val), grads

# ── Comparison ─────────────────────────────────────────────────────────────────

def compare(ref, test, name):
    r = jnp.asarray(ref).astype(jnp.float32)
    t = jnp.asarray(test).astype(jnp.float32)
    max_err = float(jnp.max(jnp.abs(r - t)))
    norm_r  = float(jnp.linalg.norm(r))
    norm_t  = float(jnp.linalg.norm(t))
    ratio   = norm_t / (norm_r + 1e-12)
    ok      = max_err < 1e-3 and abs(ratio - 1.0) < 0.02
    status  = "PASS" if ok else "FAIL"
    print(f"  [{status}] {name:8s}  max_abs_err={max_err:.2e}  "
          f"norm_ratio={ratio:.4f}  (ref={norm_r:.4f}, test={norm_t:.4f})")
    return ok

# ── Build sharded inputs for a given (fsdp, ep) mesh ──────────────────────────

def build_inputs(mesh, fsdp, ep, fx_np, fi_np, fw_np, w0_np, w1_np, wo_np):
    act_spec = P(("ep", "fsdp"), None)
    w0_spec  = P("ep", None, "fsdp")
    w1_spec  = P("ep", None, "fsdp")
    wo_spec  = P("ep", "fsdp", None)

    fx   = make_sharded(fx_np, mesh, act_spec)
    w0   = make_sharded(w0_np, mesh, w0_spec)
    w1   = make_sharded(w1_np, mesh, w1_spec)
    wout = make_sharded(wo_np, mesh, wo_spec)
    fw   = make_sharded(fw_np, mesh, act_spec)
    fi   = make_sharded(fi_np, mesh, act_spec)
    return fx, fi, fw, w0, w1, wout

# ── Main ───────────────────────────────────────────────────────────────────────

def run_case(label, fsdp, ep, bwd_version, fx_np, fi_np, fw_np, w0_np, w1_np, wo_np,
             ref_grads):
    print(f"\n=== {label}: FSDP={fsdp} EP={ep} streaming_bwd_version={bwd_version} ===")
    mesh = make_mesh(fsdp, ep)
    fx, fi, fw, w0, w1, wout = build_inputs(mesh, fsdp, ep, fx_np, fi_np, fw_np, w0_np, w1_np, wo_np)

    # jax.vjp reference (version=0) for this mesh config
    ref_val, this_ref_grads = run_version(mesh, ep, fsdp, 0, fx, fi, fw, w0, w1, wout)
    test_val, test_grads    = run_version(mesh, ep, fsdp, bwd_version, fx, fi, fw, w0, w1, wout)

    print(f"  ref_loss={ref_val:.6f}  test_loss={test_val:.6f}")
    names = ["d_fx", "d_fw", "d_w0", "d_w1", "d_wout"]
    results = [compare(this_ref_grads[i], test_grads[i], names[i]) for i in range(5)]

    if all(results):
        print(f"  --> ALL PASS")
    else:
        failed = [n for n, ok in zip(names, results) if not ok]
        print(f"  --> FAIL: {failed}")
    return all(results)


def main():
    print(f"JAX {jax.__version__}  devices={jax.device_count()}  backend={jax.default_backend()}")
    print(f"Config: D={D} D_moe={D_moe} E={E} K={K} T={T}")

    key = jax.random.PRNGKey(SEED)
    k0, k1, k2, k3 = jax.random.split(key, 4)

    fx_np = np.array(jax.random.normal(k0, (T, D),        dtype=jnp.bfloat16))
    w0_np = np.array(jax.random.normal(k1, (E, D, D_moe), dtype=jnp.bfloat16) * 0.02)
    w1_np = np.array(jax.random.normal(k2, (E, D, D_moe), dtype=jnp.bfloat16) * 0.02)
    wo_np = np.array(jax.random.normal(k3, (E, D_moe, D), dtype=jnp.bfloat16) * 0.02)

    rng   = np.random.default_rng(SEED)
    fi_np = np.argsort(rng.random((T, E)), axis=-1)[:, :K].astype(np.int32)
    raw   = rng.random((T, K)).astype(np.float32)
    fw_np = (raw / raw.sum(axis=-1, keepdims=True)).astype(np.float32)

    results = {}

    # Case A: FSDP=2 EP=2, v1 — expected FAIL (partial D_moe d_tok)
    results["A: FSDP=2 EP=2 v1"] = run_case(
        "Case A (should FAIL — root cause)", 2, 2, 1,
        fx_np, fi_np, fw_np, w0_np, w1_np, wo_np, None)

    # Case B: FSDP=1 EP=4, v1 — expected PASS (no FSDP partiality)
    results["B: FSDP=1 EP=4 v1"] = run_case(
        "Case B (should PASS — FSDP=1 baseline)", 1, 4, 1,
        fx_np, fi_np, fw_np, w0_np, w1_np, wo_np, None)

    # Case C: FSDP=2 EP=2, v2 — testing
    results["C: FSDP=2 EP=2 v2"] = run_case(
        "Case C (v2 with EP=2)", 2, 2, 2,
        fx_np, fi_np, fw_np, w0_np, w1_np, wo_np, None)

    # Case D: FSDP=2 EP=1, v2 — EP=1 isolates FSDP-only path
    results["D: FSDP=2 EP=1 v2"] = run_case(
        "Case D (v2, EP=1 FSDP=2 — no EP complication)", 2, 1, 2,
        fx_np, fi_np, fw_np, w0_np, w1_np, wo_np, None)

    # Case E: FSDP=1 EP=4, v2 — no FSDP, should behave like v1
    results["E: FSDP=1 EP=4 v2"] = run_case(
        "Case E (v2, FSDP=1 — should match v1 PASS)", 1, 4, 2,
        fx_np, fi_np, fw_np, w0_np, w1_np, wo_np, None)

    print("\n" + "="*60)
    print("Summary:")
    for name, ok in results.items():
        print(f"  {name:30s}  {'PASS' if ok else 'FAIL'}")


if __name__ == "__main__":
    main()
