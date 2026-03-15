#!/usr/bin/env python3
"""Qwen3.5 inference benchmark script for TPU profiling.

Runs prefill and decode benchmarks with configurable model size and
sharding strategy. Designed for profiling on TPU v5p.

Usage:
    # Mini config, no sharding (Mac / single device):
    python scripts/qwen35_benchmark.py --config mini

    # Mid config on 4x TPU v5p with Config B sharding:
    python scripts/qwen35_benchmark.py --config mid --sharding B --devices 4

    # Mid config with Config A sharding + profiling:
    PROFILE_DIR=/tmp/qwen35 python scripts/qwen35_benchmark.py \
        --config mid --sharding A --devices 4 --profile

    # Custom sequence lengths:
    python scripts/qwen35_benchmark.py --config mid --sharding B \
        --prompt-len 512 --decode-steps 128 --batch-size 1

Options:
    --config        mini | mid | full (default: mini)
    --sharding      A | B | none (default: none)
    --devices       Number of devices to use (default: all)
    --batch-size    Batch size (default: 1)
    --prompt-len    Prompt length for prefill (default: 128)
    --decode-steps  Number of decode steps (default: 32)
    --max-seq-len   Max sequence length for KV cache (default: prompt-len + decode-steps + 64)
    --n-runs        Number of timed runs (default: 5)
    --profile       Enable JAX profiler trace (writes to PROFILE_DIR)
    --skip-prefill  Skip prefill benchmark
    --skip-decode   Skip decode benchmark
    --dtype         float32 | bfloat16 (default: float32)
"""

from __future__ import annotations

import argparse
import contextlib
import os
import sys
import time
from contextlib import contextmanager

import jax
import jax.numpy as jnp
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jax_gpt.models.qwen35.cache import init_cache
from jax_gpt.models.qwen35.config import Qwen35Config
from jax_gpt.models.qwen35.model import forward, init_params
from jax_gpt.models.qwen35.sharding import (
    AXIS_RULES_A,
    AXIS_RULES_B,
    make_cache_sharding,
    make_mesh,
    shard_cache,
    shard_params,
)


@contextmanager
def maybe_profile(name: str, enabled: bool):
    profile_dir = os.environ.get("PROFILE_DIR", "/tmp/qwen35_profiles")
    if enabled:
        trace_dir = os.path.join(profile_dir, name)
        os.makedirs(trace_dir, exist_ok=True)
        print(f"  Profiling to: {trace_dir}")
        with jax.profiler.trace(trace_dir):
            yield
    else:
        yield


def count_params(params) -> int:
    return sum(x.size for x in jax.tree_util.tree_leaves(params))


def get_config(name: str) -> Qwen35Config:
    configs = {
        'mini': Qwen35Config.mini,
        'mid': Qwen35Config.mid,
        'full': Qwen35Config.full,
    }
    if name not in configs:
        raise ValueError(f"Unknown config: {name}. Choose from {list(configs.keys())}")
    return configs[name]()


def get_axis_rules(name: str) -> dict | None:
    rules = {
        'none': None,
        'A': AXIS_RULES_A,
        'B': AXIS_RULES_B,
    }
    if name not in rules:
        raise ValueError(f"Unknown sharding config: {name}. Choose from {list(rules.keys())}")
    return rules[name]


def run_prefill_benchmark(
    params, cfg, cache, tokens, cache_sharding,
    n_runs: int, profile: bool, mesh,
):
    """Benchmark prefill latency."""
    B, T = tokens.shape

    @jax.jit
    def prefill(p, t, c):
        return forward(p, t, cfg, cache=c, is_decode=False, cache_sharding=cache_sharding)

    # Warm-up
    print("  Compiling prefill...")
    t_compile = time.perf_counter()
    logits, _ = prefill(params, tokens, cache)
    logits.block_until_ready()
    compile_ms = (time.perf_counter() - t_compile) * 1000
    print(f"  Compilation: {compile_ms:.0f} ms")

    # Timed runs
    with maybe_profile("prefill", profile):
        t0 = time.perf_counter()
        for _ in range(n_runs):
            logits, new_cache = prefill(params, tokens, cache)
        logits.block_until_ready()
        elapsed = time.perf_counter() - t0

    avg_ms = (elapsed / n_runs) * 1000
    tokens_per_sec = (B * T * n_runs) / elapsed

    print(f"\n  PREFILL RESULTS")
    print(f"  {'Avg latency:':<20s} {avg_ms:.2f} ms")
    print(f"  {'Throughput:':<20s} {tokens_per_sec:,.0f} tokens/sec")
    print(f"  {'Tokens:':<20s} {B} × {T} = {B*T}")
    return avg_ms, tokens_per_sec


def run_decode_benchmark(
    params, cfg, cache, prompt_tokens, n_decode_steps, cache_sharding,
    n_runs: int, profile: bool, mesh,
):
    """Benchmark decode latency using lax.scan (single HLO)."""
    B = prompt_tokens.shape[0]

    @jax.jit
    def generate_scan(p, prompt, initial_cache, rng_key):
        # Prefill
        logits, cache_after = forward(
            p, prompt, cfg, cache=initial_cache, is_decode=False,
            cache_sharding=cache_sharding,
        )
        first_token = jnp.argmax(logits[:, -1, :], axis=-1)

        # Decode via lax.scan
        def _step(carry, _):
            token, c, key = carry
            step_logits, new_c = forward(
                p, token[:, None], cfg, cache=c, is_decode=True,
                cache_sharding=cache_sharding,
            )
            key, subkey = jax.random.split(key)
            next_token = jnp.argmax(step_logits[:, 0, :], axis=-1)
            return (next_token, new_c, key), next_token

        _, generated = jax.lax.scan(
            _step, (first_token, cache_after, rng_key), None,
            length=n_decode_steps,
        )
        return generated  # (n_decode_steps, B)

    rng = jax.random.key(42)

    # Warm-up
    print("  Compiling decode (lax.scan)...")
    t_compile = time.perf_counter()
    generated = generate_scan(params, prompt_tokens, cache, rng)
    generated.block_until_ready()
    compile_ms = (time.perf_counter() - t_compile) * 1000
    print(f"  Compilation: {compile_ms:.0f} ms")

    # Timed runs
    with maybe_profile("decode_scan", profile):
        t0 = time.perf_counter()
        for _ in range(n_runs):
            generated = generate_scan(params, prompt_tokens, cache, rng)
        generated.block_until_ready()
        elapsed = time.perf_counter() - t0

    total_new_tokens = n_decode_steps + 1  # +1 for first from prefill
    avg_ms = (elapsed / n_runs) * 1000
    per_step_ms = avg_ms / total_new_tokens
    tokens_per_sec = (B * total_new_tokens * n_runs) / elapsed

    print(f"\n  DECODE RESULTS (lax.scan — single HLO)")
    print(f"  {'Total:':<20s} {avg_ms:.2f} ms")
    print(f"  {'Per step:':<20s} {per_step_ms:.2f} ms/token")
    print(f"  {'Throughput:':<20s} {tokens_per_sec:.0f} tok/s")
    print(f"  {'Decode steps:':<20s} {n_decode_steps}")
    return avg_ms, per_step_ms, tokens_per_sec


def main():
    parser = argparse.ArgumentParser(description="Qwen3.5 inference benchmark")
    parser.add_argument('--config', default='mini', choices=['mini', 'mid', 'full'])
    parser.add_argument('--sharding', default='none', choices=['none', 'A', 'B'])
    parser.add_argument('--devices', type=int, default=None)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--prompt-len', type=int, default=128)
    parser.add_argument('--decode-steps', type=int, default=32)
    parser.add_argument('--max-seq-len', type=int, default=None)
    parser.add_argument('--n-runs', type=int, default=5)
    parser.add_argument('--profile', action='store_true')
    parser.add_argument('--skip-prefill', action='store_true')
    parser.add_argument('--skip-decode', action='store_true')
    parser.add_argument('--dtype', default='float32', choices=['float32', 'bfloat16'])
    args = parser.parse_args()

    if args.max_seq_len is None:
        args.max_seq_len = args.prompt_len + args.decode_steps + 64

    # Config
    cfg = get_config(args.config)
    axis_rules = get_axis_rules(args.sharding)

    print("=" * 70)
    print("QWEN3.5 INFERENCE BENCHMARK")
    print("=" * 70)
    print(f"  Config:         {args.config} ({cfg.n_layers}L, {cfg.d_model}D, "
          f"{cfg.n_routed_experts}E top-{cfg.n_experts_per_token})")
    print(f"  Sharding:       Config {args.sharding}")
    print(f"  Devices:        {args.devices or jax.device_count()}x "
          f"{jax.devices()[0].platform}")
    print(f"  Batch size:     {args.batch_size}")
    print(f"  Prompt len:     {args.prompt_len}")
    print(f"  Decode steps:   {args.decode_steps}")
    print(f"  Max seq len:    {args.max_seq_len}")
    print(f"  Dtype:          {args.dtype}")
    print(f"  Runs:           {args.n_runs}")
    print(f"  Profile:        {args.profile}")

    # Initialize model
    param_dtype = jnp.bfloat16 if args.dtype == 'bfloat16' else jnp.float32
    print(f"\nInitializing model ({args.dtype})...")
    t0 = time.perf_counter()
    params = init_params(cfg, jax.random.key(0), dtype=param_dtype)
    n_params = count_params(params)
    bytes_per_param = 2 if args.dtype == 'bfloat16' else 4
    init_ms = (time.perf_counter() - t0) * 1000
    print(f"  Params:         {n_params:,} ({n_params * bytes_per_param / 1e9:.2f} GB {args.dtype})")
    print(f"  Init time:      {init_ms:.0f} ms")

    # Setup mesh and sharding
    mesh = None
    cache_sharding = None
    if axis_rules is not None:
        n_dev = args.devices or jax.device_count()
        mesh = make_mesh(n_devices=n_dev)
        print(f"\nSharding params across {n_dev} devices...")
        t0 = time.perf_counter()
        params = shard_params(params, mesh, cfg, axis_rules)
        shard_ms = (time.perf_counter() - t0) * 1000
        print(f"  Shard time:     {shard_ms:.0f} ms")
        cache_sharding = make_cache_sharding(cfg, mesh, axis_rules)

    # Inputs
    tokens = jnp.ones((args.batch_size, args.prompt_len), dtype=jnp.int32)
    cache = init_cache(cfg, args.batch_size, args.max_seq_len,
                       dtype=jnp.bfloat16 if args.dtype == 'bfloat16' else jnp.float32)

    if mesh is not None:
        cache = shard_cache(cache, mesh, cfg, axis_rules)

    # Run benchmarks
    ctx = mesh if mesh is not None else contextlib.nullcontext()

    with ctx:
        if not args.skip_prefill:
            print(f"\n{'─'*70}")
            print(f"PREFILL BENCHMARK")
            print(f"{'─'*70}")
            run_prefill_benchmark(
                params, cfg, cache, tokens, cache_sharding,
                args.n_runs, args.profile, mesh,
            )

        if not args.skip_decode:
            print(f"\n{'─'*70}")
            print(f"DECODE BENCHMARK")
            print(f"{'─'*70}")
            prompt = jnp.ones((args.batch_size, args.prompt_len), dtype=jnp.int32)
            run_decode_benchmark(
                params, cfg, cache, prompt, args.decode_steps, cache_sharding,
                args.n_runs, args.profile, mesh,
            )

    print(f"\n{'='*70}")
    print("DONE")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
