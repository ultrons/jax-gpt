"""Repro: max_tpe formula bug at EP=4, FSDP=1.

The bug: model.py expert_mlp_jax computes
    max_tpe = max(1, 2 * T_fsdp * K // cfg.E)   # uses global E
but should use
    max_tpe = max(1, 2 * T_fsdp * K // E_local)  # E_local = E / EP

At EP=4, E=16, K=4, GBS/FSDP=16:
  avg tokens per local expert = T_fsdp * K / E_local = 16 * 4 / 4 = 16
  max_tpe_buggy  = 2 * 16 * 4 / 16 = 8    ← HALF the average: overflow guaranteed
  max_tpe_correct = 2 * 16 * 4 /  4 = 32   ← 2× the average: correct

This test runs streaming_bwd v1 with BOTH values and shows:
  - buggy  max_tpe → wrong gradient (ratio << 1 or NaN)
  - correct max_tpe → matches jax.vjp (ratio ≈ 1.0)

Run:
    /home/sivaibhav_google_com/xdb/.venv/bin/python test_maxtpe_ep4.py
Needs 4 JAX devices (EP=4, FSDP=1).
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "mini_dsv3"))

import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

# ── Config: matches cluster mini config (E=16, K=4) ─────────────────────────
# Use small D/D_moe so it runs fast locally, but same E/K/EP as the cluster.
D     = 64
D_moe = 32
E     = 16     # total experts — same as mini config
K     = 4      # top-K — same as mini config
T     = 32     # total tokens (enough to show overflow at EP=4)
SEED  = 0

EP    = 4
FSDP  = 1

def make_mesh():
    devs = jax.devices()
    need = EP * FSDP
    if len(devs) < need:
        print(f"SKIP: need {need} devices, got {len(devs)}")
        sys.exit(0)
    arr = np.array(devs[:need]).reshape(FSDP, EP)
    return Mesh(arr, ("fsdp", "ep"))

def make_sharded(data, mesh, spec):
    sharding = NamedSharding(mesh, spec)
    idx_map  = sharding.addressable_devices_indices_map(data.shape)
    shards   = [jax.device_put(data[idx_map[d]], d) for d in idx_map]
    return jax.make_array_from_single_device_arrays(data.shape, sharding, shards)

def run(mesh, max_tpe, bwd_version, fx, fi, fw, w0, w1, wout, label):
    from jax_gpt.models.dsv3.model import _moe_jax_ep_fn
    act_spec = P(("ep", "fsdp"), None)

    def loss_fn(fx_, fw_, w0_, w1_, wout_):
        out = _moe_jax_ep_fn(fx_, fi, fw_, w0_, w1_, wout_,
                              mesh, K, act_spec, "ep", max_tpe, bwd_version)
        return jnp.sum(out)

    val, grads = jax.value_and_grad(loss_fn, argnums=(0, 1, 2, 3, 4))(fx, fw, w0, w1, wout)
    return float(val), grads

def norm(g):
    return float(jnp.linalg.norm(jnp.asarray(g).astype(jnp.float32)))

def ratio(ref_g, test_g):
    r = norm(ref_g)
    t = norm(test_g)
    return t / (r + 1e-12)

def main():
    print(f"JAX {jax.__version__}  devices={jax.device_count()}  backend={jax.default_backend()}")
    print(f"Config: D={D} D_moe={D_moe} E={E} K={K} T={T}  EP={EP} FSDP={FSDP}")

    # ── max_tpe values ───────────────────────────────────────────────────────
    T_fsdp  = T // FSDP          # tokens per FSDP stripe = T (FSDP=1)
    E_local = E // EP            # = 16/4 = 4
    avg_tpe = T_fsdp * K // E_local  # = 32 * 4 / 4 = 32  (T=32)

    max_tpe_buggy   = max(1, 2 * T_fsdp * K // E)        # uses global E: 2*32*4/16 = 16
    max_tpe_correct = max(1, 2 * T_fsdp * K // E_local)  # uses E_local:  2*32*4/4  = 64

    print(f"\nT_fsdp={T_fsdp}  E_local={E_local}  avg_tpe={avg_tpe}")
    print(f"max_tpe_buggy   = {max_tpe_buggy}  (should be ≥ avg={avg_tpe} for 1× safety, "
          f"{'OVERFLOW!' if max_tpe_buggy < avg_tpe else 'OK'})")
    print(f"max_tpe_correct = {max_tpe_correct}  (2× avg = {2*avg_tpe})")

    # ── Random inputs ────────────────────────────────────────────────────────
    key = jax.random.PRNGKey(SEED)
    k0, k1, k2, k3 = jax.random.split(key, 4)
    fx_np = np.array(jax.random.normal(k0, (T, D),        dtype=jnp.float32))
    w0_np = np.array(jax.random.normal(k1, (E, D, D_moe), dtype=jnp.float32) * 0.02)
    w1_np = np.array(jax.random.normal(k2, (E, D, D_moe), dtype=jnp.float32) * 0.02)
    wo_np = np.array(jax.random.normal(k3, (E, D_moe, D), dtype=jnp.float32) * 0.02)

    rng   = np.random.default_rng(SEED)
    fi_np = np.argsort(rng.random((T, E)), axis=-1)[:, :K].astype(np.int32)
    raw   = rng.random((T, K)).astype(np.float32)
    fw_np = (raw / raw.sum(axis=-1, keepdims=True)).astype(np.float32)

    # Show actual routing distribution to confirm overflow
    E_local_val = E // EP
    expert_offset = 0  # EP device 0's experts are 0..E_local-1
    fi_flat = fi_np.reshape(-1)
    for ep_dev in range(EP):
        offset = ep_dev * E_local_val
        counts = [(fi_flat == (offset + e)).sum() * T_fsdp // T
                  for e in range(E_local_val)]
        print(f"EP device {ep_dev} expert token counts (approx): {counts}  "
              f"max_tpe_buggy={max_tpe_buggy}")

    # ── Shard inputs ─────────────────────────────────────────────────────────
    mesh = make_mesh()
    act_spec = P(("ep", "fsdp"), None)
    fx   = make_sharded(fx_np, mesh, act_spec)
    w0   = make_sharded(w0_np, mesh, P("ep", None, "fsdp"))
    w1   = make_sharded(w1_np, mesh, P("ep", None, "fsdp"))
    wout = make_sharded(wo_np, mesh, P("ep", "fsdp", None))
    fw   = make_sharded(fw_np, mesh, act_spec)
    fi   = make_sharded(fi_np, mesh, act_spec)

    # ── Run jax.vjp reference (version=0) with correct max_tpe ──────────────
    print(f"\n--- Reference: jax.vjp, max_tpe={max_tpe_correct} ---")
    ref_val, ref_grads = run(mesh, max_tpe_correct, 0, fx, fi, fw, w0, w1, wout, "ref")
    print(f"  loss = {ref_val:.6f}")
    print(f"  d_fx norm = {norm(ref_grads[0]):.6f}")

    # ── streaming_bwd v1 with BUGGY max_tpe ─────────────────────────────────
    print(f"\n--- streaming_bwd v1, max_tpe_BUGGY={max_tpe_buggy} "
          f"(avg={avg_tpe}, overflow={'YES' if max_tpe_buggy < avg_tpe else 'NO'}) ---")
    bug_val, bug_grads = run(mesh, max_tpe_buggy, 1, fx, fi, fw, w0, w1, wout, "bug")
    d_fx_ratio_bug = ratio(ref_grads[0], bug_grads[0])
    d_w0_ratio_bug = ratio(ref_grads[2], bug_grads[2])
    print(f"  loss = {bug_val:.6f}")
    print(f"  d_fx norm = {norm(bug_grads[0]):.6f}  ratio vs ref = {d_fx_ratio_bug:.4f}")
    print(f"  d_w0 norm = {norm(bug_grads[2]):.6f}  ratio vs ref = {d_w0_ratio_bug:.4f}")
    has_nan_bug = any(
        np.any(np.isnan(np.array(jnp.asarray(g).astype(jnp.float32))))
        for g in bug_grads
    )
    if has_nan_bug:
        print(f"  *** NaN detected in gradients! ***")

    # ── streaming_bwd v1 with CORRECT max_tpe ───────────────────────────────
    print(f"\n--- streaming_bwd v1, max_tpe_CORRECT={max_tpe_correct} ---")
    fix_val, fix_grads = run(mesh, max_tpe_correct, 1, fx, fi, fw, w0, w1, wout, "fix")
    d_fx_ratio_fix = ratio(ref_grads[0], fix_grads[0])
    d_w0_ratio_fix = ratio(ref_grads[2], fix_grads[2])
    print(f"  loss = {fix_val:.6f}")
    print(f"  d_fx norm = {norm(fix_grads[0]):.6f}  ratio vs ref = {d_fx_ratio_fix:.4f}")
    print(f"  d_w0 norm = {norm(fix_grads[2]):.6f}  ratio vs ref = {d_w0_ratio_fix:.4f}")
    has_nan_fix = any(
        np.any(np.isnan(np.array(jnp.asarray(g).astype(jnp.float32))))
        for g in fix_grads
    )
    if has_nan_fix:
        print(f"  *** NaN detected in gradients! ***")

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("Summary:")
    bug_ok = not has_nan_bug and abs(d_fx_ratio_bug - 1.0) < 0.02
    fix_ok = not has_nan_fix and abs(d_fx_ratio_fix - 1.0) < 0.02
    print(f"  buggy  max_tpe={max_tpe_buggy}: d_fx ratio={d_fx_ratio_bug:.4f}  "
          f"{'PASS' if bug_ok else 'FAIL (expected)'}")
    print(f"  correct max_tpe={max_tpe_correct}: d_fx ratio={d_fx_ratio_fix:.4f}  "
          f"{'PASS' if fix_ok else 'FAIL'}")

    if not bug_ok and fix_ok:
        print("\nBug reproduced: buggy formula causes wrong gradients; correct formula fixes it.")
    elif bug_ok:
        print("\nBug NOT reproduced (routing may not overflow for this seed — try larger T).")
    else:
        print("\nBoth fail — check other issues.")


if __name__ == "__main__":
    main()
