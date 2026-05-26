"""AOT compile probe for kernel-agent fused MoE @ jax-gpt production-class shapes.

Tries the v_outside.moe_block_ep kernel at three shape points:

1. validated@D5  — E=64  D=2048 F=128  K=4  EP=4 FSDP=32  (rbq 4x4x4 = 128 cores)
   This is what kernel-agent 945964d cluster-validated.

2. mid           — E=64  D=2048 F=128  K=8  EP=4 FSDP=32
   Same D as (1) but K=8 to test the K parameter is just a config flip.

3. prod@dsv3     — E=256 D=7168 F=2048 K=8 EP=4 FSDP=128 (4x8x8 = 512 cores)
   The shape jax-gpt actually trains at. Per kernel-agent SPEC:
   "Full DSv3 production (D=7168, K=8) requires D.6 (true D-tiling)."

For each shape we report: AOT-PASS or AOT-FAIL (with the Mosaic error trunc'd).

Run:  source ~/xdb/.xprof/bin/activate; \
      PYTHONPATH=research/dsv3/kernel-agent-snapshot-945964d/dsv3-fused-ep-moe/build \
      python research/dsv3/kernel_agent_aot_check.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import topologies
from jax.sharding import Mesh, PartitionSpec as P

# Snapshot of kernel-agent 945964d.
_SNAP = Path(__file__).parent / "kernel-agent-snapshot-945964d" / "dsv3-fused-ep-moe"
sys.path.insert(0, str(_SNAP / "build"))

try:
    from jax import shard_map
    SM_KWARG = {"check_vma": False}
except ImportError:
    from jax.experimental.shard_map import shard_map  # type: ignore
    SM_KWARG = {"check_rep": False}

from v_outside.moe_block_ep import MoEBlockEPConfig, moe_block_ep_fwd  # noqa: E402


def _build_mesh(topo_str: str, axes: tuple[int, int, int, int]):
    topo = topologies.get_topology_desc(topo_str, platform="tpu")
    n = int(np.prod(axes))
    assert len(topo.devices) == n, f"{topo_str}: expected {n} devs, got {len(topo.devices)}"
    devs = np.array(topo.devices).reshape(*axes)
    return Mesh(devs, ("dp", "ep", "fsdp", "tp")), topo


def _try_compile(label: str, topo_str: str, mesh_axes: tuple[int, int, int, int], *,
                 E: int, D: int, F: int, K: int, T_global: int, bt_ffn: int):
    EP = mesh_axes[1]
    FSDP = mesh_axes[2]
    assert E % EP == 0 and T_global % FSDP == 0
    mesh, topo = _build_mesh(topo_str, mesh_axes)
    cfg = MoEBlockEPConfig(E=E, D=D, F=F, K=K, EP=EP, bt_router=16, bt_ffn=bt_ffn)

    x_abs   = jax.ShapeDtypeStruct((T_global, D),       jnp.bfloat16)
    Wg_abs  = jax.ShapeDtypeStruct((E, D),              jnp.bfloat16)
    W1_abs  = jax.ShapeDtypeStruct((E, D, 2 * F),       jnp.bfloat16)
    Wd_abs  = jax.ShapeDtypeStruct((E, F, D),           jnp.bfloat16)
    in_specs  = (P("fsdp", None), P(None, None),
                 P("ep", None, None), P("ep", None, None))
    out_specs = P("fsdp", None)

    def fn(x, Wg, W1, Wd):
        return moe_block_ep_fwd(x, Wg, W1, Wd, cfg)

    fn_sm = shard_map(fn, mesh=mesh, in_specs=in_specs, out_specs=out_specs, **SM_KWARG)

    print(f"\n----- {label}  topo={topo_str} mesh={mesh_axes} -----")
    print(f"  cfg: E={E} D={D} F={F} K={K}  EP={EP} FSDP={FSDP}  T_global={T_global} bt_ffn={bt_ffn}")
    print(f"  E_local = E/EP = {E//EP},  T_local = T_global/FSDP = {T_global//FSDP}")
    sys.stdout.flush()

    t0 = time.perf_counter()
    try:
        with jax.default_device(topo.devices[0]):
            lowered  = jax.jit(fn_sm).lower(x_abs, Wg_abs, W1_abs, Wd_abs)
            compiled = lowered.compile()
        dt = time.perf_counter() - t0
        print(f"  -> AOT PASS  (compile_time={dt:.1f}s)")
        try:
            ca = compiled.cost_analysis()
            if isinstance(ca, list) and ca:
                ca = ca[0]
            if isinstance(ca, dict):
                keep = {k: v for k, v in ca.items() if k in (
                    "flops", "bytes accessed", "transcendentals", "optimal_seconds")}
                print(f"  -> cost_analysis (subset): {keep}")
        except Exception:
            pass
        return True, dt, None
    except Exception as e:
        dt = time.perf_counter() - t0
        msg = str(e)
        if len(msg) > 1600:
            msg = msg[:800] + "\n  ... (truncated) ...\n" + msg[-800:]
        print(f"  -> AOT FAIL ({type(e).__name__}) after {dt:.1f}s")
        print(f"  -> {msg}")
        return False, dt, msg


def main():
    print(f"jax {jax.__version__}  jaxlib backend {jax.default_backend()}")
    results = []

    # 1) Validated D.5 shape (paper-reported PASS at cluster).
    ok, dt, _ = _try_compile(
        "validated@D5", "tpu7x:2x2x1", (1, 4, 2, 1),
        E=64, D=2048, F=128, K=4, T_global=4096, bt_ffn=128)
    results.append(("validated@D5 (E=64 D=2048 K=4 EP=4)", ok, dt))

    # 2) Same as (1) but K=8 to test the K parameter.
    ok, dt, _ = _try_compile(
        "mid K=8",       "tpu7x:2x2x1", (1, 4, 2, 1),
        E=64, D=2048, F=128, K=8, T_global=4096, bt_ffn=128)
    results.append(("mid (E=64 D=2048 K=8 EP=4)", ok, dt))

    # 3) jax-gpt production-class shape (D=7168 K=8 E=256 on 4x8x8).
    # NOTE: this exercises kernel-agent's open D.6 ("true D-tiling") gap.
    # We probe at tpu7x:4x8x8 = 512 devices, mesh (dp=1, ep=4, fsdp=128, tp=1).
    # T_global=4096 keeps memory bounded; the kernel is shape-parametric so
    # the compile-time error class is the same regardless of T_global.
    ok, dt, _ = _try_compile(
        "prod@dsv3", "tpu7x:4x8x8", (1, 4, 128, 1),
        E=256, D=7168, F=2048, K=8, T_global=4096, bt_ffn=128)
    results.append(("prod@dsv3 (E=256 D=7168 K=8 EP=4 FSDP=128)", ok, dt))

    print("\n========== SUMMARY ==========")
    for label, ok, dt in results:
        verdict = "PASS" if ok else "FAIL"
        print(f"  [{verdict}] {label}  ({dt:.1f}s)")


if __name__ == "__main__":
    main()
