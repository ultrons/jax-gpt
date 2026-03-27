"""Collective bandwidth benchmark on v7x.

Tests all-reduce, all-gather, and reduce-scatter on TP axis,
comparing two mesh layouts:
  - 'reshape': plain np.reshape (current make_mesh behaviour, no topo awareness)
  - 'topo':    mesh_utils.create_device_mesh (topology-aware, uses wrap-around)

Purpose: verify whether topology-aware placement recovers the missing 2x
bandwidth from torus wrap-around links.

Usage:
  python scripts/collective_bench.py
"""

import sys
import time
import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from jax.experimental.shard_map import shard_map
from jax.experimental import mesh_utils

sys.path.insert(0, '/app')


# Message sizes to sweep (bytes) — total data across all tp devices
SIZES_BYTES = [
    64 * 1024,           # 64 KB
    512 * 1024,          # 512 KB
    2 * 1024**2,         # 2 MB
    16 * 1024**2,        # 16 MB
    64 * 1024**2,        # 64 MB
    256 * 1024**2,       # 256 MB
    1 * 1024**3,         # 1 GB
]

N_WARMUP = 3
N_RUNS = 10
DTYPE = jnp.bfloat16
BYTES_PER_EL = 2


def make_mesh_reshape(dp: int = 8) -> Mesh:
    """Current make_mesh behaviour: plain np.reshape, no topology awareness."""
    devices = jax.devices()
    n = len(devices)
    tp = n // dp
    return Mesh(np.array(devices).reshape(dp, tp), ('dp', 'tp'))


def make_mesh_topo(dp: int = 8) -> Mesh:
    """Topology-aware: mesh_utils.create_device_mesh places tp on fast ICI links."""
    devices = jax.devices()
    n = len(devices)
    tp = n // dp
    # create_device_mesh assigns the last (innermost) axis to the fastest links.
    # With shape (dp, tp), tp gets the fast inter-chip torus axis with wrap-around.
    device_arr = mesh_utils.create_device_mesh((dp, tp), devices=devices)
    return Mesh(device_arr, ('dp', 'tp'))


def bench_collective(fn, x, n_warmup=N_WARMUP, n_runs=N_RUNS):
    """Time a collective fn. Returns median latency in ms."""
    for _ in range(n_warmup):
        y = fn(x)
        jax.block_until_ready(y)

    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        y = fn(x)
        jax.block_until_ready(y)
        times.append((time.perf_counter() - t0) * 1e3)

    return float(np.median(times))


def algbw(size_bytes, time_ms):
    return size_bytes / (time_ms * 1e-3) / 1e9


def busbw_allreduce(size_bytes, tp, time_ms):
    return 2 * (tp - 1) / tp * size_bytes / (time_ms * 1e-3) / 1e9


def busbw_allgather(size_bytes, tp, time_ms):
    return (tp - 1) / tp * size_bytes / (time_ms * 1e-3) / 1e9


def busbw_reducescatter(size_bytes, tp, time_ms):
    return (tp - 1) / tp * size_bytes / (time_ms * 1e-3) / 1e9


def _fmt(n):
    if n >= 1024**3:
        return f"{n/1024**3:.0f} GB"
    if n >= 1024**2:
        return f"{n/1024**2:.0f} MB"
    if n >= 1024:
        return f"{n/1024:.0f} KB"
    return f"{n} B"


def run_bench_for_mesh(mesh, label):
    tp = mesh.shape['tp']
    print(f"\n{'#'*70}")
    print(f"# Mesh layout: {label}  |  shape={dict(mesh.shape)}")
    print(f"# Device layout (dp x tp):")
    print(f"#   {mesh.devices}")
    print(f"{'#'*70}")

    ax = 'tp'

    # --- ALL-REDUCE ---
    print(f"\n{'─'*70}")
    print(f"ALL-REDUCE on 'tp'  (psum)")
    print(f"{'─'*70}")
    print(f"{'Size':>10}  {'Time(ms)':>10}  {'AlgBW(GB/s)':>14}  {'BusBW(GB/s)':>14}")
    for size in SIZES_BYTES:
        local_n = size // BYTES_PER_EL // tp
        global_n = local_n * tp
        x = jax.device_put(jnp.ones(global_n, dtype=DTYPE),
                           NamedSharding(mesh, P(ax)))
        fn = jax.jit(shard_map(
            lambda x: jax.lax.psum(x, axis_name=ax),
            mesh=mesh, in_specs=P(ax), out_specs=P(ax), check_rep=False))
        t_ms = bench_collective(fn, x)
        print(f"{_fmt(size):>10}  {t_ms:>10.3f}  {algbw(size, t_ms):>14.2f}"
              f"  {busbw_allreduce(size, tp, t_ms):>14.2f}")

    # --- ALL-GATHER ---
    print(f"\n{'─'*70}")
    print(f"ALL-GATHER on 'tp'  (all_gather)")
    print(f"{'─'*70}")
    print(f"{'Size':>10}  {'Time(ms)':>10}  {'AlgBW(GB/s)':>14}  {'BusBW(GB/s)':>14}")
    for size in SIZES_BYTES:
        local_n = size // BYTES_PER_EL // tp
        global_input_n = local_n * tp
        x = jax.device_put(jnp.ones(global_input_n, dtype=DTYPE),
                           NamedSharding(mesh, P(ax)))
        fn = jax.jit(shard_map(
            lambda x: jax.lax.all_gather(x, axis_name=ax, tiled=True),
            mesh=mesh, in_specs=P(ax), out_specs=P(), check_rep=False))
        t_ms = bench_collective(fn, x)
        print(f"{_fmt(size):>10}  {t_ms:>10.3f}  {algbw(size, t_ms):>14.2f}"
              f"  {busbw_allgather(size, tp, t_ms):>14.2f}")

    # --- REDUCE-SCATTER ---
    print(f"\n{'─'*70}")
    print(f"REDUCE-SCATTER on 'tp'  (psum_scatter)")
    print(f"{'─'*70}")
    print(f"{'Size':>10}  {'Time(ms)':>10}  {'AlgBW(GB/s)':>14}  {'BusBW(GB/s)':>14}")
    for size in SIZES_BYTES:
        # Use in_specs=P('tp') here so each device has size/tp (not replicated 1GB)
        # This matches the model's actual RS input (sharded matmul output).
        local_n = size // BYTES_PER_EL // tp
        global_n = local_n * tp
        x = jax.device_put(jnp.ones(global_n, dtype=DTYPE),
                           NamedSharding(mesh, P(ax)))
        fn = jax.jit(shard_map(
            lambda x: jax.lax.psum_scatter(x, axis_name=ax,
                                           scatter_dimension=0, tiled=True),
            mesh=mesh, in_specs=P(ax), out_specs=P(), check_rep=False))
        t_ms = bench_collective(fn, x)
        print(f"{_fmt(size):>10}  {t_ms:>10.3f}  {algbw(size, t_ms):>14.2f}"
              f"  {busbw_reducescatter(size, tp, t_ms):>14.2f}")


def main():
    print(f"JAX devices: {jax.device_count()}")
    print(f"Device kind: {jax.devices()[0].device_kind}")

    mesh_reshape = make_mesh_reshape(dp=8)
    mesh_topo = make_mesh_topo(dp=8)

    print(f"\nReshape mesh devices:\n{mesh_reshape.devices}")
    print(f"\nTopo-aware mesh devices:\n{mesh_topo.devices}")

    run_bench_for_mesh(mesh_reshape, label='reshape (current)')
    run_bench_for_mesh(mesh_topo,    label='topo-aware (create_device_mesh)')

    print("\n\nDone.")


if __name__ == '__main__':
    main()
