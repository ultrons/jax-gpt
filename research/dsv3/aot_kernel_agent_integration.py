"""AOT compile probe — kernel-agent FFN swap inside _expert_mlp_gmm_ag_body.

Tests whether the gated `cfg.moe_use_kernel_agent_ffn` path compiles
cleanly when called through the full custom_vjp + shard_map scaffold
that production training uses.

Two shape points:

  small        E=32  D=2048 F=128  K=4 | tpu7x:2x2x1 mesh (1,2,4,1)
                 production-proxy at the kernel-agent's cluster-validated
                 shape. Catches our integration wiring against the
                 surrounding _moe_gmm_ag scaffold.

  prod@dsv3    E=256 D=7168 F=2048 K=8 | tpu7x:4x8x8 mesh (1,4,128,1)
                 the actual jax-gpt training shape. Exercises D.6
                 D-tiling (~3.7 GB W1 per device) inside our scaffold.

For each shape we report PASS / FAIL and (on PASS) compile time +
cost-analysis subset.

Run:
  source ~/xdb/.xprof/bin/activate
  PYTHONPATH=. python research/dsv3/aot_kernel_agent_integration.py
"""
from __future__ import annotations

import os
import sys
import time

# Make jax_gpt importable when run as a script.
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import topologies
from jax.sharding import Mesh, PartitionSpec as P

from jax_gpt.models.dsv3.model import ModelConfig, expert_mlp_gmm_ag


def _build_mesh(topo_str: str, axes: tuple[int, int, int, int]):
    topo = topologies.get_topology_desc(topo_str, platform="tpu")
    n = int(np.prod(axes))
    assert len(topo.devices) == n, (
        f"{topo_str}: expected {n} devs, got {len(topo.devices)}")
    devs = np.array(topo.devices).reshape(*axes)
    return Mesh(devs, ("dp", "ep", "fsdp", "tp")), topo


def _try_compile(label: str, topo_str: str, axes: tuple[int, int, int, int],
                 *, E: int, D: int, F: int, K: int, B: int, S: int,
                 use_kernel_agent_ffn: bool = True,
                 n_chunks: int = 2):
    mesh, topo = _build_mesh(topo_str, axes)
    cfg = ModelConfig(name="aot_probe")
    cfg.D = D
    cfg.F = F
    cfg.E = E
    cfg.K = K
    cfg.L = 1
    cfg.L_dense = 0
    cfg.mesh = mesh
    cfg.moe_use_kernel_agent_ffn = use_kernel_agent_ffn
    cfg.moe_use_gmm_v2 = False
    cfg.moe_use_sc_scatter = False
    cfg.moe_fp8_weights = False
    cfg.moe_debug_nans = False
    cfg.moe_n_chunks = n_chunks

    x_abs   = jax.ShapeDtypeStruct((B, S, D),             jnp.bfloat16)
    wi0_abs = jax.ShapeDtypeStruct((E, F, D),             jnp.bfloat16)
    wi1_abs = jax.ShapeDtypeStruct((E, F, D),             jnp.bfloat16)
    wo_abs  = jax.ShapeDtypeStruct((E, F, D),             jnp.bfloat16)
    tkw_abs = jax.ShapeDtypeStruct((B, S, K),             jnp.bfloat16)
    tki_abs = jax.ShapeDtypeStruct((B, S, K),             jnp.int32)

    print(f"\n----- {label}  topo={topo_str}  mesh={axes} -----")
    print(f"  cfg: E={E} D={D} F={F} K={K}  B={B} S={S}  n_chunks={n_chunks}")
    print(f"  use_kernel_agent_ffn={use_kernel_agent_ffn}")
    sys.stdout.flush()

    def fn(x, w0, w1, wo, tkw, tki):
        return expert_mlp_gmm_ag(x, w0, w1, wo, tkw, tki, cfg)

    t0 = time.perf_counter()
    try:
        with jax.default_device(topo.devices[0]):
            lowered  = jax.jit(fn).lower(x_abs, wi0_abs, wi1_abs, wo_abs,
                                          tkw_abs, tki_abs)
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
        return True, dt
    except Exception as e:
        dt = time.perf_counter() - t0
        msg = str(e)
        if len(msg) > 1600:
            msg = msg[:800] + "\n  ... (truncated) ...\n" + msg[-800:]
        print(f"  -> AOT FAIL ({type(e).__name__}) after {dt:.1f}s")
        print(f"  -> {msg}")
        return False, dt


def main():
    print(f"jax {jax.__version__}  backend {jax.default_backend()}")
    results = []

    ok, dt = _try_compile(
        "small@2x2x1 (production-proxy, kernel_agent on)",
        "tpu7x:2x2x1", (1, 2, 4, 1),
        E=32, D=2048, F=128, K=4, B=1, S=512,
        use_kernel_agent_ffn=True, n_chunks=2)
    results.append(("small@2x2x1  E=32 D=2048 K=4  kernel_agent=on", ok, dt))

    ok, dt = _try_compile(
        "small@2x2x1 (same shape, kernel_agent OFF baseline)",
        "tpu7x:2x2x1", (1, 2, 4, 1),
        E=32, D=2048, F=128, K=4, B=1, S=512,
        use_kernel_agent_ffn=False, n_chunks=2)
    results.append(("small@2x2x1  E=32 D=2048 K=4  kernel_agent=off", ok, dt))

    # Production AOT requires realistic B*S to satisfy the inner kernel's
    # bt=128 divisibility. v304 production = (BS=4096, seq=4096); per-device
    # T_local = 4096*4096/(128*4) = 32,768; max_local_c = T_local*K/n_chunks
    # = 32768*8/2 = 131,072 = 1024*128. AOT shape is abstract so memory
    # cost is not a concern.
    ok, dt = _try_compile(
        "prod@dsv3 (kernel_agent on, full DSv3 shape)",
        "tpu7x:4x8x8", (1, 4, 128, 1),
        E=256, D=7168, F=2048, K=8, B=4096, S=4096,
        use_kernel_agent_ffn=True, n_chunks=2)
    results.append(("prod@dsv3   E=256 D=7168 K=8 BS=4096 seq=4096  kernel_agent=on", ok, dt))

    # Same shape, kernel_agent OFF — comparator for compile-time + scaffold.
    ok, dt = _try_compile(
        "prod@dsv3 (baseline ragged_dot, same shape)",
        "tpu7x:4x8x8", (1, 4, 128, 1),
        E=256, D=7168, F=2048, K=8, B=4096, S=4096,
        use_kernel_agent_ffn=False, n_chunks=2)
    results.append(("prod@dsv3   E=256 D=7168 K=8 BS=4096 seq=4096  kernel_agent=off", ok, dt))

    print("\n========== SUMMARY ==========")
    for label, ok, dt in results:
        verdict = "PASS" if ok else "FAIL"
        print(f"  [{verdict}] {label}  ({dt:.1f}s)")


if __name__ == "__main__":
    main()
