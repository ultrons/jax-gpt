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
from jax.sharding import NamedSharding, PartitionSpec as P
from jax.experimental.shard_map import shard_map

sys.path.insert(0, '/app')

from jax_gpt.models.qwen35.sharding import make_mesh


# Message sizes to sweep (bytes) — total data across all devices per collective
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
    # 'size' = total message = per-device * tp. Each device contributes size/tp bytes.
    # psum: each device starts with size/tp elements, result = sum across axis devices.
    print(f"\n{'─'*70}")
    print(f"ALL-REDUCE on '{axis_name}'  (psum)")
    print(f"{'─'*70}")
    print(f"{'Size':>10}  {'Time(ms)':>10}  {'AlgBW(GB/s)':>14}  {'BusBW(GB/s)':>14}")
    for size in SIZES_BYTES:
        local_n = size // BYTES_PER_EL // tp  # elements per device
        global_n = local_n * tp
        x = jax.device_put(
            jnp.ones(global_n, dtype=DTYPE),
            NamedSharding(mesh, P(axis_name)),
        )
        ax = axis_name  # capture for closure
        fn = jax.jit(shard_map(
            lambda x: jax.lax.psum(x, axis_name=ax),
            mesh=mesh,
            in_specs=P(axis_name),
            out_specs=P(axis_name),
            check_rep=False,
        ))
        t_ms = bench_collective(fn, x)
        abw = algbw(size, t_ms)
        bbw = busbw_allreduce(size, tp, t_ms)
        print(f"{_fmt(size):>10}  {t_ms:>10.3f}  {abw:>14.2f}  {bbw:>14.2f}")

    # --- ALL-GATHER ---
    # 'size' = total output = per-device-input * tp. Each device has size/tp input, size output.
    print(f"\n{'─'*70}")
    print(f"ALL-GATHER on '{axis_name}'  (all_gather)")
    print(f"{'─'*70}")
    print(f"{'Size':>10}  {'Time(ms)':>10}  {'AlgBW(GB/s)':>14}  {'BusBW(GB/s)':>14}")
    for size in SIZES_BYTES:
        local_n = size // BYTES_PER_EL // tp  # input elements per device
        global_input_n = local_n * tp
        x = jax.device_put(
            jnp.ones(global_input_n, dtype=DTYPE),
            NamedSharding(mesh, P(axis_name)),
        )
        ax = axis_name
        fn = jax.jit(shard_map(
            lambda x: jax.lax.all_gather(x, axis_name=ax, tiled=True),
            mesh=mesh,
            in_specs=P(axis_name),
            out_specs=P(),          # each device holds the full gathered array
            check_rep=False,
        ))
        t_ms = bench_collective(fn, x)
        abw = algbw(size, t_ms)
        bbw = busbw_allgather(size, tp, t_ms)
        print(f"{_fmt(size):>10}  {t_ms:>10.3f}  {abw:>14.2f}  {bbw:>14.2f}")

    # --- REDUCE-SCATTER ---
    # 'size' = total input = per-device-output * tp. Each device has size input (replicated),
    # size/tp output.
    print(f"\n{'─'*70}")
    print(f"REDUCE-SCATTER on '{axis_name}'  (psum_scatter)")
    print(f"{'─'*70}")
    print(f"{'Size':>10}  {'Time(ms)':>10}  {'AlgBW(GB/s)':>14}  {'BusBW(GB/s)':>14}")
    for size in SIZES_BYTES:
        global_n = size // BYTES_PER_EL  # elements per device (input is replicated)
        x = jax.device_put(
            jnp.ones(global_n, dtype=DTYPE),
            NamedSharding(mesh, P()),   # replicated on all devices
        )
        ax = axis_name
        fn = jax.jit(shard_map(
            lambda x: jax.lax.psum_scatter(x, axis_name=ax,
                                           scatter_dimension=0, tiled=True),
            mesh=mesh,
            in_specs=P(),               # each device has the full replicated array
            out_specs=P(axis_name),     # output sharded
            check_rep=False,
        ))
        t_ms = bench_collective(fn, x)
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
