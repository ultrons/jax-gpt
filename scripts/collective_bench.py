"""Collective bandwidth benchmark on v7x.

Tests all-reduce, all-gather, and reduce-scatter on the same logical/physical
mesh as the Qwen3.5 decode benchmark (dp=8, tp=8, np.reshape layout).

Usage:
  python scripts/collective_bench.py
"""

import sys
import time
import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

sys.path.insert(0, '/app')

from jax_gpt.models.qwen35.sharding import make_mesh


# Message sizes to sweep (bytes)
SIZES_BYTES = [
    4 * 1024,           # 4 KB
    64 * 1024,          # 64 KB
    512 * 1024,         # 512 KB
    2 * 1024**2,        # 2 MB
    16 * 1024**2,       # 16 MB
    64 * 1024**2,       # 64 MB
    256 * 1024**2,      # 256 MB
    1 * 1024**3,        # 1 GB
]

N_WARMUP = 3
N_RUNS = 10
DTYPE = jnp.bfloat16
BYTES_PER_EL = 2


def bench_collective(fn, x_sharded, n_warmup=N_WARMUP, n_runs=N_RUNS):
    """Time a collective fn. Returns median latency in ms."""
    # Warmup
    for _ in range(n_warmup):
        y = fn(x_sharded)
        jax.effects_barrier()

    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        y = fn(x_sharded)
        jax.effects_barrier()
        times.append((time.perf_counter() - t0) * 1e3)

    return float(np.median(times))


def make_sharded(mesh, size_bytes, axis_name):
    """Create a bf16 array sharded on axis_name."""
    tp = mesh.shape[axis_name]
    n_el = size_bytes // BYTES_PER_EL
    # Local elements per device = n_el / tp; global = n_el
    x = jnp.ones(n_el, dtype=DTYPE)
    sharding = NamedSharding(mesh, P(axis_name))
    return jax.device_put(x, sharding)


def algbw(size_bytes, time_ms):
    """Algorithm bandwidth: size / time."""
    return size_bytes / (time_ms * 1e-3) / 1e9  # GB/s


def busbw_allreduce(size_bytes, tp, time_ms):
    """Bus bandwidth for ring allreduce: 2*(tp-1)/tp * size / time."""
    return 2 * (tp - 1) / tp * size_bytes / (time_ms * 1e-3) / 1e9


def busbw_allgather(size_bytes, tp, time_ms):
    """Bus bandwidth for ring all-gather: (tp-1)/tp * size / time."""
    return (tp - 1) / tp * size_bytes / (time_ms * 1e-3) / 1e9


def busbw_reducescatter(size_bytes, tp, time_ms):
    """Bus bandwidth for ring reduce-scatter: (tp-1)/tp * size / time."""
    return (tp - 1) / tp * size_bytes / (time_ms * 1e-3) / 1e9


def _fmt(n):
    if n >= 1024**3:
        return f"{n/1024**3:.0f} GB"
    if n >= 1024**2:
        return f"{n/1024**2:.0f} MB"
    if n >= 1024:
        return f"{n/1024:.0f} KB"
    return f"{n} B"


def run_collective_bench(mesh, axis_name):
    tp = mesh.shape[axis_name]
    print(f"\n{'='*70}")
    print(f"Axis: '{axis_name}'  (size={tp})  mesh={dict(mesh.shape)}")
    print(f"{'='*70}")

    # --- ALL-REDUCE ---
    print(f"\n{'─'*70}")
    print(f"ALL-REDUCE on '{axis_name}'  (psum)")
    print(f"{'─'*70}")
    print(f"{'Size':>10}  {'Time(ms)':>10}  {'AlgBW(GB/s)':>14}  {'BusBW(GB/s)':>14}")
    for size in SIZES_BYTES:
        x = make_sharded(mesh, size // tp, axis_name)  # local=size/tp, global=size/tp (replicated after psum)
        # Actually for allreduce, each device has a local shard and we reduce across axis
        # Let's use global=size (each device has full size, psum reduces in-place)
        x_full = jax.device_put(jnp.ones(size // BYTES_PER_EL, dtype=DTYPE),
                                NamedSharding(mesh, P(None)))  # replicated on all
        # Shard on tp for a more realistic test: each device has size/tp locally
        x_sharded = jax.device_put(
            jnp.ones(size // BYTES_PER_EL // tp, dtype=DTYPE),
            NamedSharding(mesh, P(axis_name))
        )
        # allreduce: psum — each device sends its shard and gets the total
        fn = jax.jit(lambda x: jax.lax.psum(x, axis_name=axis_name),
                     in_shardings=NamedSharding(mesh, P(axis_name)),
                     out_shardings=NamedSharding(mesh, P(axis_name)))
        t_ms = bench_collective(fn, x_sharded)
        abw = algbw(size, t_ms)
        bbw = busbw_allreduce(size, tp, t_ms)
        print(f"{_fmt(size):>10}  {t_ms:>10.3f}  {abw:>14.2f}  {bbw:>14.2f}")

    # --- ALL-GATHER ---
    print(f"\n{'─'*70}")
    print(f"ALL-GATHER on '{axis_name}'  (all_gather)")
    print(f"{'─'*70}")
    print(f"{'Size':>10}  {'Time(ms)':>10}  {'AlgBW(GB/s)':>14}  {'BusBW(GB/s)':>14}")
    for size in SIZES_BYTES:
        # Input: each device has size/tp, output: each device has size
        x_sharded = jax.device_put(
            jnp.ones(size // BYTES_PER_EL // tp, dtype=DTYPE),
            NamedSharding(mesh, P(axis_name))
        )
        fn = jax.jit(lambda x: jax.lax.all_gather(x, axis_name=axis_name, tiled=True),
                     in_shardings=NamedSharding(mesh, P(axis_name)),
                     out_shardings=NamedSharding(mesh, P(None)))
        t_ms = bench_collective(fn, x_sharded)
        abw = algbw(size, t_ms)
        bbw = busbw_allgather(size, tp, t_ms)
        print(f"{_fmt(size):>10}  {t_ms:>10.3f}  {abw:>14.2f}  {bbw:>14.2f}")

    # --- REDUCE-SCATTER ---
    print(f"\n{'─'*70}")
    print(f"REDUCE-SCATTER on '{axis_name}'  (psum_scatter)")
    print(f"{'─'*70}")
    print(f"{'Size':>10}  {'Time(ms)':>10}  {'AlgBW(GB/s)':>14}  {'BusBW(GB/s)':>14}")
    for size in SIZES_BYTES:
        # Input: each device has size, output: each device has size/tp
        x_full = jax.device_put(
            jnp.ones(size // BYTES_PER_EL, dtype=DTYPE),
            NamedSharding(mesh, P(None))
        )
        fn = jax.jit(lambda x: jax.lax.psum_scatter(x, axis_name=axis_name,
                                                      scatter_dimension=0, tiled=True),
                     in_shardings=NamedSharding(mesh, P(None)),
                     out_shardings=NamedSharding(mesh, P(axis_name)))
        t_ms = bench_collective(fn, x_full)
        abw = algbw(size, t_ms)
        bbw = busbw_reducescatter(size, tp, t_ms)
        print(f"{_fmt(size):>10}  {t_ms:>10.3f}  {abw:>14.2f}  {bbw:>14.2f}")


def main():
    print(f"JAX devices: {jax.device_count()}")
    print(f"Device kind: {jax.devices()[0].device_kind}")

    # Same mesh as the benchmark: dp=8, tp=8, no topology-aware placement
    mesh = make_mesh(dp=8)
    print(f"Mesh shape: {dict(mesh.shape)}")
    print(f"Device layout: {mesh.devices.shape}")

    # Benchmark TP axis (the one used for model collectives)
    run_collective_bench(mesh, axis_name='tp')

    # Also benchmark DP axis for comparison
    run_collective_bench(mesh, axis_name='dp')

    print("\n\nDone.")


if __name__ == '__main__':
    main()
