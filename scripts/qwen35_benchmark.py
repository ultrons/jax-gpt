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


def _fmt_flops(n):
    if n >= 1e12: return f'{n/1e12:.2f} TFLOP'
    if n >= 1e9: return f'{n/1e9:.2f} GFLOP'
    if n >= 1e6: return f'{n/1e6:.1f} MFLOP'
    return f'{n:,}'

def _fmt_bytes(n):
    if n >= 1e9: return f'{n/1e9:.2f} GB'
    if n >= 1e6: return f'{n/1e6:.1f} MB'
    return f'{n/1e3:.1f} KB'


def run_roofline_analysis(params, cfg, cache, prompt_len, batch_size):
    """Print roofline analysis for prefill and decode."""
    from jax.experimental.roofline import roofline
    from jax_gpt.models.qwen35.deltanet import deltanet_prefill, deltanet_recurrent_step
    from jax_gpt.models.qwen35.gqa import gqa_attention
    from jax_gpt.models.qwen35.moe import moe_layer
    from jax_gpt.models.qwen35.primitives import precompute_rope_freqs

    B = batch_size
    T = prompt_len
    param_dtype = params['embed'].dtype
    param_shapes = jax.tree.map(lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype), params)
    cache_shapes = jax.tree.map(lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype), cache)

    def _try_roofline(name, fn, *args):
        try:
            _, r = roofline(fn)(*args)
            return r
        except Exception as e:
            print(f"  [roofline] {name} failed: {e}")
            return None

    # Overall prefill
    tok_pre = jax.ShapeDtypeStruct((B, T), jnp.int32)
    def fwd_pre(p, t, c):
        return forward(p, t, cfg, cache=c, is_decode=False)
    pre_r = _try_roofline('prefill', fwd_pre, param_shapes, tok_pre, cache_shapes)

    # Overall decode
    tok_dec = jax.ShapeDtypeStruct((B, 1), jnp.int32)
    def fwd_dec(p, t, c):
        return forward(p, t, cfg, cache=c, is_decode=True)
    dec_r = _try_roofline('decode', fwd_dec, param_shapes, tok_dec, cache_shapes)

    # Per-module shapes
    delta_params = jax.tree.map(lambda x: x[0, 0], params['groups']['delta_layers'])
    delta_attn_s = jax.tree.map(lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype), delta_params['attn'])
    delta_moe_s = jax.tree.map(lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype), delta_params['moe'])
    gqa_params = jax.tree.map(lambda x: x[0], params['groups']['gqa_layer'])
    gqa_attn_s = jax.tree.map(lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype), gqa_params['attn'])

    x_T = jax.ShapeDtypeStruct((B, T, cfg.d_model), param_dtype)
    x_1 = jax.ShapeDtypeStruct((B, 1, cfg.d_model), param_dtype)

    key_dim = cfg.delta_n_qk_heads * cfg.delta_qk_head_dim
    value_dim = cfg.delta_n_v_heads * cfg.delta_v_head_dim
    conv_dim = key_dim * 2 + value_dim
    state_s = jax.ShapeDtypeStruct((B, cfg.delta_n_v_heads, cfg.delta_qk_head_dim, cfg.delta_v_head_dim), param_dtype)
    conv_s = jax.ShapeDtypeStruct((B, conv_dim, cfg.delta_conv_kernel), param_dtype)

    rope = precompute_rope_freqs(cfg.gqa_rope_dim, cfg.max_position_embeddings, cfg.gqa_rope_theta)

    modules = []

    # DeltaNet prefill
    def dn_pre(p, x):
        return deltanet_prefill(x, p, cfg.delta_n_qk_heads, cfg.delta_n_v_heads,
                                cfg.delta_qk_head_dim, cfg.delta_v_head_dim,
                                cfg.delta_conv_kernel, chunk_size=cfg.delta_chunk_size)
    r = _try_roofline('DeltaNet prefill', dn_pre, delta_attn_s, x_T)
    if r: modules.append((f'DeltaNet attn (T={T})', r))

    # DeltaNet decode
    def dn_dec(p, x, s, c):
        return deltanet_recurrent_step(x, p, s, c, cfg.delta_n_qk_heads, cfg.delta_n_v_heads,
                                        cfg.delta_qk_head_dim, cfg.delta_v_head_dim)
    r = _try_roofline('DeltaNet decode', dn_dec, delta_attn_s, x_1, state_s, conv_s)
    if r: modules.append(('DeltaNet attn (T=1)', r))

    # GQA prefill
    def gqa_pre(p, x):
        return gqa_attention(x, p, cfg.gqa_n_q_heads, cfg.gqa_n_kv_heads, cfg.gqa_head_dim,
                             rope, cfg.gqa_rope_dim)
    r = _try_roofline('GQA prefill', gqa_pre, gqa_attn_s, x_T)
    if r: modules.append((f'GQA attn (T={T})', r))

    # MoE
    def moe_fwd(p, x):
        return moe_layer(x, p, cfg.n_experts_per_token)
    r = _try_roofline('MoE prefill', moe_fwd, delta_moe_s, x_T)
    if r: modules.append((f'MoE (T={T})', r))
    r = _try_roofline('MoE decode', moe_fwd, delta_moe_s, x_1)
    if r: modules.append(('MoE (T=1)', r))

    # TPU v5p: 459 TFLOPS bf16, 2.8 TB/s HBM bandwidth
    tpu_flops = 459e12
    tpu_bw = 2.8e12

    print(f"\n{'='*78}")
    print(f"ROOFLINE ANALYSIS (B={B})")
    print(f"{'='*78}")
    print(f"\n  {'Overall':<28s} {'FLOPs':>12s} {'HBM':>10s} {'AI':>10s} {'Bound':>10s}")
    print(f"  {'-'*72}")
    for name, r in [('Prefill (full model)', pre_r), ('Decode (full model)', dec_r)]:
        if r is None:
            print(f"  {name:<28s} {'(failed)':>12s}")
            continue
        ai = r.flops / max(r.hbm_bytes, 1)
        ridge = tpu_flops / tpu_bw  # ~164 FLOPs/byte for v5p
        bound = 'COMPUTE' if ai > ridge else 'MEMORY'
        t_compute = r.flops / tpu_flops * 1000  # ms
        t_memory = r.hbm_bytes / tpu_bw * 1000  # ms
        t_roof = max(t_compute, t_memory)
        print(f"  {name:<28s} {_fmt_flops(r.flops):>12s} {_fmt_bytes(r.hbm_bytes):>10s} {ai:>8.1f}x {bound:>10s}")
        print(f"  {'':28s} {'roofline:':>12s} {t_roof:>9.2f}ms (compute={t_compute:.2f}, mem={t_memory:.2f})")

    if modules:
        print(f"\n  {'Per module (1 layer)':<28s} {'FLOPs':>12s} {'HBM':>10s} {'AI':>10s} {'Bound':>10s}")
        print(f"  {'-'*72}")
        for name, r in modules:
            ai = r.flops / max(r.hbm_bytes, 1)
            ridge = tpu_flops / tpu_bw
            bound = 'COMPUTE' if ai > ridge else 'MEMORY'
            print(f"  {name:<28s} {_fmt_flops(r.flops):>12s} {_fmt_bytes(r.hbm_bytes):>10s} {ai:>8.1f}x {bound:>10s}")

    print(f"\n  TPU v5p reference: {tpu_flops/1e12:.0f} TFLOPS bf16, {tpu_bw/1e12:.1f} TB/s HBM, ridge={tpu_flops/tpu_bw:.0f} FLOPs/byte")
    print(f"{'='*78}\n")


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
        'mid_large': Qwen35Config.mid_large,
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
    parser.add_argument('--config', default='mini', choices=['mini', 'mid', 'mid_large', 'full'])
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
    parser.add_argument('--chunk-size', type=int, default=None,
                        help='DeltaNet prefill chunk size (default: from config)')
    parser.add_argument('--roofline', action='store_true',
                        help='Print roofline analysis (FLOPs, HBM, arithmetic intensity). '
                             'Tracing can be slow for large models — use --chunk-size 32 for faster analysis.')
    args = parser.parse_args()

    if args.max_seq_len is None:
        args.max_seq_len = args.prompt_len + args.decode_steps + 64

    # Config
    cfg = get_config(args.config)
    if args.chunk_size is not None:
        from dataclasses import replace
        cfg = replace(cfg, delta_chunk_size=args.chunk_size)
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
    print(f"  Chunk size:     {cfg.delta_chunk_size}")
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

    # Roofline analysis (before benchmarks — no actual computation needed)
    if args.roofline:
        run_roofline_analysis(params, cfg, cache, args.prompt_len, args.batch_size)

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
