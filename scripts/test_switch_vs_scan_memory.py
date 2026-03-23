"""Test: does jax.lax.switch inside while_loop avoid the scan copy?

Compares peak HBM for three approaches (run one at a time for accurate measurement):
  1. scan    — jax.lax.scan with stacked (N, ...) array as xs
  2. switch  — jax.lax.while_loop + jax.lax.switch with N unstacked branch consts
  3. pyloop  — Python for-loop with per-call JIT

Usage:
  python scripts/test_switch_vs_scan_memory.py --approach scan    [--n-groups 6] [--dim 4096]
  python scripts/test_switch_vs_scan_memory.py --approach switch  [--n-groups 6] [--dim 4096]
  python scripts/test_switch_vs_scan_memory.py --approach pyloop  [--n-groups 6] [--dim 4096]
  python scripts/test_switch_vs_scan_memory.py --approach all     [--n-groups 6] [--dim 4096]
"""

import argparse
import subprocess
import sys
import time
import os

import jax
import jax.numpy as jnp


def get_memory_stats():
    device = jax.local_devices()[0]
    stats = device.memory_stats()
    if not stats:
        return {}
    return {
        'current_gb': stats.get('bytes_in_use', 0) / 1e9,
        'peak_gb': stats.get('peak_bytes_in_use', 0) / 1e9,
        'limit_gb': stats.get('bytes_limit', 0) / 1e9,
    }


def test_scan(n_groups, dim):
    """jax.lax.scan with stacked (n_groups, dim, dim) weights as xs."""
    w = jnp.ones((n_groups, dim, dim), dtype=jnp.bfloat16) * 0.001
    w.block_until_ready()
    weight_gb = w.nbytes / 1e9

    def scan_body(carry, w_slice):
        x = carry
        x = x + jnp.dot(x, w_slice)
        return x, None

    @jax.jit
    def run(x, weights):
        x, _ = jax.lax.scan(scan_body, x, weights)
        return x

    x = jnp.ones((dim, dim), dtype=jnp.bfloat16) * 0.01
    x.block_until_ready()

    mem_before = get_memory_stats()
    print(f"  Before compile: current={mem_before['current_gb']:.3f} GB, peak={mem_before['peak_gb']:.3f} GB")

    t0 = time.time()
    result = run(x, w)
    result.block_until_ready()
    compile_time = time.time() - t0

    mem_after_compile = get_memory_stats()
    print(f"  After compile:  current={mem_after_compile['current_gb']:.3f} GB, peak={mem_after_compile['peak_gb']:.3f} GB")
    print(f"  Compile time: {compile_time:.2f}s")

    # Second run (cached)
    t0 = time.time()
    result = run(x, w)
    result.block_until_ready()
    run_time = time.time() - t0

    mem_final = get_memory_stats()
    print(f"  After run:      current={mem_final['current_gb']:.3f} GB, peak={mem_final['peak_gb']:.3f} GB")
    print(f"  Run time: {run_time*1000:.2f}ms")

    return weight_gb, mem_after_compile['peak_gb'], compile_time, run_time


def test_switch(n_groups, dim):
    """while_loop + switch with unstacked weights as branch consts."""
    ws = [jnp.ones((dim, dim), dtype=jnp.bfloat16) * 0.001 * (g + 1)
          for g in range(n_groups)]
    for w in ws:
        w.block_until_ready()
    weight_gb = sum(w.nbytes for w in ws) / 1e9

    def make_branch(w):
        def branch_fn(x):
            return x + jnp.dot(x, w)
        return branch_fn

    branches = [make_branch(w) for w in ws]

    def while_cond(state):
        i, _ = state
        return i < n_groups

    def while_body(state):
        i, x = state
        x = jax.lax.switch(i, branches, x)
        return (i + 1, x)

    @jax.jit
    def run(x):
        _, result = jax.lax.while_loop(while_cond, while_body, (jnp.int32(0), x))
        return result

    x = jnp.ones((dim, dim), dtype=jnp.bfloat16) * 0.01
    x.block_until_ready()

    mem_before = get_memory_stats()
    print(f"  Before compile: current={mem_before['current_gb']:.3f} GB, peak={mem_before['peak_gb']:.3f} GB")

    t0 = time.time()
    result = run(x)
    result.block_until_ready()
    compile_time = time.time() - t0

    mem_after_compile = get_memory_stats()
    print(f"  After compile:  current={mem_after_compile['current_gb']:.3f} GB, peak={mem_after_compile['peak_gb']:.3f} GB")
    print(f"  Compile time: {compile_time:.2f}s")

    t0 = time.time()
    result = run(x)
    result.block_until_ready()
    run_time = time.time() - t0

    mem_final = get_memory_stats()
    print(f"  After run:      current={mem_final['current_gb']:.3f} GB, peak={mem_final['peak_gb']:.3f} GB")
    print(f"  Run time: {run_time*1000:.2f}ms")

    return weight_gb, mem_after_compile['peak_gb'], compile_time, run_time


def test_pyloop(n_groups, dim):
    """Python for-loop, each step JIT'd."""
    ws = [jnp.ones((dim, dim), dtype=jnp.bfloat16) * 0.001 * (g + 1)
          for g in range(n_groups)]
    for w in ws:
        w.block_until_ready()
    weight_gb = sum(w.nbytes for w in ws) / 1e9

    @jax.jit
    def step(x, w):
        return x + jnp.dot(x, w)

    x = jnp.ones((dim, dim), dtype=jnp.bfloat16) * 0.01
    x.block_until_ready()

    mem_before = get_memory_stats()
    print(f"  Before compile: current={mem_before['current_gb']:.3f} GB, peak={mem_before['peak_gb']:.3f} GB")

    t0 = time.time()
    for w in ws:
        x = step(x, w)
    x.block_until_ready()
    compile_time = time.time() - t0

    mem_after_compile = get_memory_stats()
    print(f"  After compile:  current={mem_after_compile['current_gb']:.3f} GB, peak={mem_after_compile['peak_gb']:.3f} GB")
    print(f"  Compile time: {compile_time:.2f}s")

    x = jnp.ones((dim, dim), dtype=jnp.bfloat16) * 0.01
    t0 = time.time()
    for w in ws:
        x = step(x, w)
    x.block_until_ready()
    run_time = time.time() - t0

    mem_final = get_memory_stats()
    print(f"  After run:      current={mem_final['current_gb']:.3f} GB, peak={mem_final['peak_gb']:.3f} GB")
    print(f"  Run time: {run_time*1000:.2f}ms")

    return weight_gb, mem_after_compile['peak_gb'], compile_time, run_time


def run_single(approach, n_groups, dim):
    """Run a single approach and print results."""
    weight_gb = n_groups * dim * dim * 2 / 1e9
    print(f"\nJAX {jax.__version__} | {jax.device_count()} x {jax.devices()[0].platform}")
    mem = get_memory_stats()
    print(f"HBM: {mem.get('limit_gb', 0):.2f} GB limit, {mem.get('current_gb', 0):.3f} GB in use")
    print(f"Config: n_groups={n_groups}, dim={dim}, weight_size={weight_gb:.3f} GB")

    print(f"\n── {approach.upper()} ──")
    if approach == 'scan':
        return test_scan(n_groups, dim)
    elif approach == 'switch':
        return test_switch(n_groups, dim)
    elif approach == 'pyloop':
        return test_pyloop(n_groups, dim)
    else:
        raise ValueError(f"Unknown approach: {approach}")


def run_all(n_groups, dim):
    """Run each approach in a separate subprocess for clean peak measurement."""
    print(f"Running all approaches in separate processes for clean peak measurement...")
    print(f"Config: n_groups={n_groups}, dim={dim}")
    weight_gb = n_groups * dim * dim * 2 / 1e9
    print(f"Total weight size: {weight_gb:.3f} GB")
    print()

    results = {}
    for approach in ['scan', 'switch', 'pyloop']:
        print(f"{'='*60}")
        print(f"Running: {approach}")
        print(f"{'='*60}")
        cmd = [
            sys.executable, __file__,
            '--approach', approach,
            '--n-groups', str(n_groups),
            '--dim', str(dim),
        ]
        env = os.environ.copy()
        # Suppress the large constants warning for switch
        env['JAX_CAPTURED_CONSTANTS_WARN_BYTES'] = '-1'
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300, env=env)
        print(proc.stdout)
        if proc.stderr:
            # Filter out common noise
            for line in proc.stderr.split('\n'):
                if line.strip() and 'WARNING' not in line and 'UserWarning' not in line:
                    print(f"  STDERR: {line}")

        # Parse peak from output
        for line in proc.stdout.split('\n'):
            if 'After compile:' in line:
                parts = line.split('peak=')[1].split(' GB')[0]
                peak = float(parts)
                results[approach] = peak
            if 'Compile time:' in line:
                ct = float(line.split(':')[1].strip().rstrip('s'))
                results[f'{approach}_compile'] = ct
            if 'Run time:' in line:
                rt = float(line.split(':')[1].strip().rstrip('ms'))
                results[f'{approach}_runtime'] = rt

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"Weight size: {weight_gb:.3f} GB | n_groups={n_groups}, dim={dim}")
    print()
    print(f"{'Approach':<12} {'Peak HBM (GB)':>14} {'HBM overhead':>13} {'Compile (s)':>12} {'Run (ms)':>10}")
    print("-" * 65)
    for approach in ['scan', 'switch', 'pyloop']:
        peak = results.get(approach, 0)
        overhead = peak - weight_gb
        ct = results.get(f'{approach}_compile', 0)
        rt = results.get(f'{approach}_runtime', 0)
        ratio = overhead / weight_gb if weight_gb > 0 else 0
        print(f"{approach:<12} {peak:>14.3f} {overhead:>10.3f} ({ratio:.1f}x) {ct:>11.2f} {rt:>10.2f}")

    scan_peak = results.get('scan', 0)
    switch_peak = results.get('switch', 0)
    print()
    if scan_peak > 0 and switch_peak > 0:
        diff = scan_peak - switch_peak
        print(f"Scan peak - Switch peak = {diff:.3f} GB")
        if diff > weight_gb * 0.5:
            print(f">>> Switch saves ~{diff:.1f} GB — scan copies, switch doesn't!")
        elif diff > weight_gb * 0.1:
            print(f">>> Switch saves some memory but less than expected.")
        elif abs(diff) < weight_gb * 0.1:
            print(f">>> Similar peak — both copy or neither copies.")
        else:
            print(f">>> Switch uses MORE memory than scan!")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--approach', choices=['scan', 'switch', 'pyloop', 'all'],
                        default='all')
    parser.add_argument('--n-groups', type=int, default=6)
    parser.add_argument('--dim', type=int, default=4096,
                        help='Matrix dimension (dim x dim weight per group)')
    args = parser.parse_args()

    if args.approach == 'all':
        run_all(args.n_groups, args.dim)
    else:
        run_single(args.approach, args.n_groups, args.dim)


if __name__ == '__main__':
    main()
