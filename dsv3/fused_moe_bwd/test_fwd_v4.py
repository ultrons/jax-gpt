#!/usr/bin/env python3
"""
test_fwd_v4.py — local unit test for the Pallas v4 MoE forward kernel.

Why this test exists
--------------------
Errors v164–v171 were Mosaic compilation failures that only fired at cluster
scale (EP=16, bt=16, E=128) because there was NO local forward kernel test.
The backward tests (test_bwd_v3.py) don't exercise the forward path at all.

Root cause of the local blind spot
-----------------------------------
The auto-computed bt in forward_kernel.py scales inversely with EP:

    _a2a_bytes_per_bt = 2 * num_devices * t_packing * (D // t_packing) * 2
    _bt_max = int(30MB * 0.60) // (2 * _a2a_bytes_per_bt)

At cluster (EP=16, D=7168): bt_max = 20 → bt = 16.
Locally  (EP=1,  D=7168): bt_max = 328 → bt = 64.
Locally  (EP=1,  D=256 ): bt_max = 9215 → bt = 64.

All values bt > 8 with padded_E=128 or padded_top_k=128 trigger the Mosaic
relayout error ("Non-singleton logical dimension is replicated in destination
but not in source for vector<BtxExi1>"). So even EP=1 locally catches the
error when using bt=BT=16 explicitly, or auto-bt which is typically 64.

This test forces bt=BT=16 explicitly so it exactly reproduces the cluster
shapes. The critical constraint is bt × max(padded_E, padded_top_k) > 8*128,
i.e. bt > 8 with E=128.

Test matrix
-----------
  aot:  CPU-only Mosaic compile check via abstract mesh + cross-AOT.
        Uses jax.experimental.topologies to get virtual tpu7x:4x4x4 devices.
        No TPU hardware needed — Mosaic runs for real on virtual chips.
        Catches all shape-cast/relayout errors before touching any hardware.
        Primary regression gate; runs anywhere with libtpu in the venv.

  ep1:  EP=1 on 1 chip — execution test, shape + NaN checks.
        No ICI comms; runs on any single device.

  ep4:  EP=4 on 4 chips — full ICI A2A path + same shapes.
        Skipped if < 4 chips available.

Run
---
  source ~/xdb/.xprof/bin/activate
  cd ~/ml-experiments/dsv3
  python fused_moe_bwd/test_fwd_v4.py

Pass: all [PASS] lines, exit 0
Fail: [FAIL] line with exception, exit 1
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
try:
    from jax.shard_map import shard_map
except ImportError:
    from jax.experimental.shard_map import shard_map  # pre-0.8.0 fallback

from fused_moe_bwd.forward_kernel import fused_ep_moe_fwd_streaming_v1

jax.config.update("jax_traceback_filtering", "off")

# ── Shapes that match the 4×4×16 cluster config ──────────────────────────────
#
# Cluster: EP=16, FSDP=16, GBS=256, mini D=7168, E=128, top_k=8
#   T_local = 16384
#   bt_max  = 20 → bt = 16   (first divisor of T_local that is ≤ 20)
#   padded_E      = align(128, 128) = 128
#   padded_top_k  = align(8,   128) = 128
#
# Any (bt, 128) bool in compute_gating triggers the relayout error.
# D and F_shard are small here (local speed); the tile shapes are bt × E, not D.
BT       = 16    # << MUST be > 8 to reproduce the Mosaic relayout errors
E_GLOBAL = 128   # mini config num_experts (padded_E = 128)
TOP_K    = 8     # DSv3 top_k (padded_top_k = 128)
D        = 512   # hidden dim — smaller than cluster for speed; shapes that
                 # matter for Mosaic are bt×E, not D
F_SHARD  = 128   # intermediate per FSDP shard; must be 128-aligned
T_LOCAL  = 64    # tokens per EP device; must be divisible by BT (64 = 4×16)

_PASSES = 0
_FAILS  = 0


def _ok(msg):
    global _PASSES
    _PASSES += 1
    print(f"  [PASS] {msg}")


def _fail(msg, exc=None):
    global _FAILS
    _FAILS += 1
    suffix = f": {type(exc).__name__}: {exc}" if exc else ""
    print(f"  [FAIL] {msg}{suffix}")


# ── Forward function (body executed inside shard_map) ────────────────────────

def _fwd(tokens, w1, w2, gating):
    """Called per-device inside shard_map(ep, fsdp).

    tokens  (T_LOCAL, D)         — replicated across ep; EP-gathered by caller
    w1      (E_local, 2, D, F)   — this device's expert weights
    w2      (E_local, F, D)
    gating  (T_LOCAL, E_GLOBAL)  — full routing scores; replicated across ep
    """
    return fused_ep_moe_fwd_streaming_v1(
        tokens, w1, w2, gating, TOP_K,
        ep_axis_name="ep",
        bt=BT,  # force cluster-scale bt — this is what triggers the Mosaic errors
    )


# ── AOT compile-only check (no TPU hardware needed) ──────────────────────────

def run_aot_compile_check(ep_size: int = 4):
    """Mosaic compile check via abstract mesh + cross-AOT.

    Uses jax.experimental.topologies to create virtual tpu7x devices — Mosaic
    runs for real against them but no physical TPU is required.  Catches every
    shape-cast/relayout error (the v164–v171 bug series) on any machine that has
    libtpu in the venv.  No data is allocated; jax.ShapeDtypeStruct is used for
    all arguments.

    This is the *fastest* gate: run it before touching any hardware.
    """
    try:
        from jax.experimental import topologies
    except ImportError:
        print("\n--- aot_compile: SKIP (jax.experimental.topologies not available) ---")
        return

    tag = f"aot_ep{ep_size}"
    try:
        # tpu7x:4x4x4 = 64 chips × 2 cores = 128 virtual devices.
        # We only use the first ep_size for the EP axis (fsdp=tp=dp=1).
        topo = topologies.get_topology_desc("tpu7x:4x4x4", platform="tpu")
    except Exception as e:
        print(f"\n--- {tag}: SKIP (topology unavailable: {type(e).__name__}: {e}) ---")
        return

    E_local = E_GLOBAL // ep_size
    devs = np.array(topo.devices[:ep_size]).reshape(1, ep_size, 1, 1)
    mesh = Mesh(devs, axis_names=("dp", "ep", "fsdp", "tp"))

    fn = shard_map(
        _fwd, mesh=mesh,
        in_specs=(
            P(None, None),
            P("ep", None, None, None),
            P("ep", None, None),
            P(None, None),
        ),
        out_specs=P(None, None),
        check_rep=False,
    )

    tokens_abs = jax.ShapeDtypeStruct((T_LOCAL, D),             jnp.bfloat16)
    w1_abs     = jax.ShapeDtypeStruct((E_local, 2, D, F_SHARD), jnp.bfloat16)
    w2_abs     = jax.ShapeDtypeStruct((E_local, F_SHARD, D),    jnp.bfloat16)
    gating_abs = jax.ShapeDtypeStruct((T_LOCAL, E_GLOBAL),      jnp.float32)

    print(f"\n--- {tag} (virtual tpu7x, no hardware, EP={ep_size}, bt={BT}) ---")
    try:
        with jax.default_device(topo.devices[0]):
            lowered = jax.jit(fn).lower(tokens_abs, w1_abs, w2_abs, gating_abs)
            lowered.compile()
        _ok(f"{tag} Mosaic compile (no hardware)")
    except Exception as e:
        _fail(f"{tag} Mosaic compile", e)
        import traceback; traceback.print_exc()


# ── Test runner ───────────────────────────────────────────────────────────────

def run_test(ep_size: int):
    tag = f"ep{ep_size}"
    n_avail = jax.device_count()
    if n_avail < ep_size:
        print(f"\n--- {tag}: SKIP (need {ep_size} chips, have {n_avail}) ---")
        return

    E_local = E_GLOBAL // ep_size
    devs = np.array(jax.devices()[:ep_size])

    # 4-D mesh (dp=1, ep=ep_size, fsdp=1, tp=1).
    # forward_kernel.py hardcodes:
    #   extra_device_id_prefix=(0,)  — dp rank 0
    #   extra_device_id_suffix=(0,)  — tp rank 0
    #   non_ep_axis_name="fsdp"
    # So device IDs are (dp=0, ep_rank, fsdp_rank, tp=0).
    # With dp=fsdp=tp=1: flat index = ep_rank → physical device ep_rank. ✓
    mesh = Mesh(devs.reshape(1, ep_size, 1, 1),
                axis_names=("dp", "ep", "fsdp", "tp"))

    key = jax.random.PRNGKey(0)
    k1, k2, k3, k4 = jax.random.split(key, 4)
    # Global shapes: w1/w2 will be split on ep by shard_map.
    tokens = (jax.random.normal(k1, (T_LOCAL, D))           * 0.1 ).astype(jnp.bfloat16)
    w1_g   = (jax.random.normal(k2, (E_GLOBAL, 2, D, F_SHARD)) * 0.02).astype(jnp.bfloat16)
    w2_g   = (jax.random.normal(k3, (E_GLOBAL, F_SHARD, D)) * 0.02).astype(jnp.bfloat16)
    gating = (jax.random.normal(k4, (T_LOCAL, E_GLOBAL))         ).astype(jnp.float32)

    # Distribute: tokens/gating replicated; weights sharded on ep.
    tokens_d = jax.device_put(tokens, NamedSharding(mesh, P(None, None)))
    w1_d     = jax.device_put(w1_g,   NamedSharding(mesh, P("ep", None, None, None)))
    w2_d     = jax.device_put(w2_g,   NamedSharding(mesh, P("ep", None, None)))
    gating_d = jax.device_put(gating, NamedSharding(mesh, P(None, None)))

    # shard_map: each ep-device sees local weights slice, full tokens/gating.
    fn = shard_map(
        _fwd, mesh=mesh,
        in_specs=(
            P(None, None),              # tokens:  not split on ep
            P("ep", None, None, None),  # w1:      split on ep (E_local per device)
            P("ep", None, None),        # w2:      split on ep
            P(None, None),              # gating:  not split on ep
        ),
        out_specs=P(None, None),        # partial output — caller would psum_scatter+psum
        check_rep=False,
    )

    print(f"\n--- {tag} (EP={ep_size}, bt={BT}, E={E_GLOBAL}, padded_top_k=128) ---")

    # 1. Compile + run ---------------------------------------------------------
    try:
        out = jax.jit(fn)(tokens_d, w1_d, w2_d, gating_d)
        out.block_until_ready()
        _ok(f"{tag} compile + run")
    except Exception as e:
        _fail(f"{tag} compile + run", e)
        import traceback; traceback.print_exc()
        return

    # 2. Output shape ----------------------------------------------------------
    try:
        assert out.shape == (T_LOCAL, D), f"got {out.shape}"
        _ok(f"{tag} output shape = {out.shape}")
    except AssertionError as e:
        _fail(f"{tag} output shape", e)

    # 3. No NaN ----------------------------------------------------------------
    try:
        n_nan = int(jnp.sum(jnp.isnan(out.astype(jnp.float32))))
        assert n_nan == 0, f"{n_nan}/{out.size} values are NaN"
        _ok(f"{tag} no NaN  (output min={float(jnp.min(out)):.4f}, max={float(jnp.max(out)):.4f})")
    except AssertionError as e:
        _fail(f"{tag} NaN check", e)


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"JAX devices  : {jax.devices()}")
    print(f"Test shapes  : BT={BT}, E={E_GLOBAL}, TOP_K={TOP_K}")
    print(f"             : padded_E=align({E_GLOBAL},128)={((E_GLOBAL+127)//128)*128}")
    print(f"             : padded_top_k=align({TOP_K},128)={((TOP_K+127)//128)*128}")
    print(f"             : D={D}, F_SHARD={F_SHARD}, T_LOCAL={T_LOCAL}")
    print(f"Cluster bt   : EP=16,D=7168 → bt_max=20 → bt=16 (= BT={BT} ✓)")

    # Gate 0: AOT compile-only (no hardware) — catches Mosaic shape/relayout errors
    # fast, runs anywhere with libtpu in venv. ep_size=4 exercises the A2A ICI path.
    run_aot_compile_check(ep_size=4)

    # Gate 1/2: execution tests — need real TPU devices
    run_test(ep_size=1)   # single-device; shape + NaN checks
    run_test(ep_size=4)   # multi-device; tests ICI A2A + same shapes

    print(f"\n{'='*50}")
    status = "ALL PASS" if _FAILS == 0 else f"{_FAILS} FAILED"
    print(f"{status}  ({_PASSES} passed, {_FAILS} failed)")
    sys.exit(0 if _FAILS == 0 else 1)
