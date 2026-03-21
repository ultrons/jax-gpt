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

from jax.sharding import NamedSharding, PartitionSpec as P

from jax_gpt.models.qwen35.cache import HybridCache, init_cache
from jax_gpt.models.qwen35.config import Qwen35Config
from jax_gpt.models.qwen35.model import forward, init_params

try:
    from jax_gpt.models.qwen35.model import forward_rpa_decode
    HAS_RPA = True
except ImportError:
    HAS_RPA = False
from jax_gpt.models.qwen35.sharding import (
    AXIS_RULES_A,
    AXIS_RULES_B,
    make_cache_sharding,
    make_mesh,
    make_param_shardings,
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
    n_runs: int, profile: bool, mesh, n_devices: int = 1,
):
    """Benchmark prefill latency."""
    B, T = tokens.shape

    @jax.jit
    def prefill(p, t, c):
        return forward(p, t, cfg, cache=c, is_decode=False,
                       cache_sharding=cache_sharding, n_devices=n_devices, mesh=mesh,
                       last_logit_only=True)

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

    tps_per_chip = tokens_per_sec / n_devices

    print(f"\n  PREFILL RESULTS")
    print(f"  {'Avg latency:':<20s} {avg_ms:.2f} ms")
    print(f"  {'Throughput:':<20s} {tokens_per_sec:,.0f} tokens/sec")
    print(f"  {'TPS/chip:':<20s} {tps_per_chip:,.1f} tokens/sec/chip")
    print(f"  {'Tokens:':<20s} {B} × {T} = {B*T}")
    return avg_ms, tokens_per_sec


def run_decode_benchmark(
    params, cfg, cache, prompt_tokens, n_decode_steps, cache_sharding,
    n_runs: int, profile: bool, mesh, n_devices: int = 1,
    use_rpa: bool = False,
):
    """Benchmark decode latency.

    Uses separate JITs for prefill and single-step decode, with a Python
    for-loop over decode steps.  This avoids XLA's while-loop semantics
    which copy all captured params into the loop body (doubling HBM for
    large MoE models).
    """
    B = prompt_tokens.shape[0]

    @jax.jit
    def prefill_fn(p, t, c):
        return forward(p, t, cfg, cache=c, is_decode=False,
                       cache_sharding=cache_sharding, n_devices=n_devices,
                       mesh=mesh, last_logit_only=True)

    if use_rpa and HAS_RPA:
        # RPA decode: use forward() with use_rpa=True inside a single JIT.
        # For small configs (few groups) the scan+Pallas program fits in HBM.
        # For large configs (60L/15 groups), the program may be >3 GB and OOM —
        # in that case fall back to forward_rpa_decode (per-group JIT).
        @jax.jit
        def decode_step_rpa(p, tok, c):
            logits, new_c = forward(
                p, tok[:, None], cfg, cache=c, is_decode=True,
                cache_sharding=cache_sharding, n_devices=n_devices,
                mesh=mesh, use_rpa=True,
            )
            next_token = jnp.argmax(logits[:, 0, :], axis=-1)
            return next_token, new_c

        decode_step = decode_step_rpa
    else:
        @jax.jit
        def decode_step(p, tok, c):
            logits, new_c = forward(
                p, tok[:, None], cfg, cache=c, is_decode=True,
                cache_sharding=cache_sharding, n_devices=n_devices, mesh=mesh,
            )
            next_token = jnp.argmax(logits[:, 0, :], axis=-1)
            return next_token, new_c

    # Warm-up: compile prefill
    print("  Compiling prefill (for decode)...")
    t_compile = time.perf_counter()
    logits, cache_after = prefill_fn(params, prompt_tokens, cache)
    logits.block_until_ready()
    first_token = jnp.argmax(logits[:, 0, :], axis=-1)
    compile_prefill_ms = (time.perf_counter() - t_compile) * 1000
    print(f"  Prefill compilation: {compile_prefill_ms:.0f} ms")

    # Convert contiguous cache to paged for RPA decode
    if use_rpa:
        from jax_gpt.models.qwen35.paged_cache import contiguous_to_paged
        from tpu_inference.kernels.ragged_paged_attention.v3.util import cdiv

        page_size = 64
        prefill_len = int(cache_after.pos)
        max_len = cache_after.gqa_k.shape[3]  # may be global shape
        pages_per_seq = cdiv(max_len, page_size)

        print(f"  Converting to paged KV cache (page_size={page_size}, "
              f"pages_per_seq={pages_per_seq})...")
        t_convert = time.perf_counter()
        paged_list = contiguous_to_paged(
            cache_after.gqa_k, cache_after.gqa_v,
            prefill_len=prefill_len,
            page_size=page_size,
        )
        # Stack per-group paged caches: (n_groups, total_pages, ps, kv_dim, pk, hd)
        paged_kv = jnp.stack(paged_list, axis=0)
        kv_lens = jnp.full((B,), prefill_len, dtype=jnp.int32)
        page_indices = jnp.arange(B * pages_per_seq, dtype=jnp.int32)

        # Apply DP sharding to paged arrays: pages dim holds B sequences
        # contiguously (seq_i owns pages [i*pps, (i+1)*pps)), so we shard
        # on the pages axis which splits by batch.
        if mesh is not None:
            from jax.sharding import NamedSharding, PartitionSpec as P
            dp_axis = 'dp' if 'dp' in mesh.axis_names else None
            # paged_kv: (n_groups, total_pages, ps, kv_dim, pk, hd) — shard total_pages on dp
            paged_kv = jax.device_put(paged_kv, NamedSharding(mesh, P(None, dp_axis, None, None, None, None)))
            kv_lens = jax.device_put(kv_lens, NamedSharding(mesh, P(dp_axis)))
            page_indices = jax.device_put(page_indices, NamedSharding(mesh, P(dp_axis)))

        # Replace contiguous GQA cache with minimal dummy to free HBM
        dummy_gqa_k = jnp.zeros((cache_after.gqa_k.shape[0], 1, 1, 1, 1),
                                dtype=cache_after.gqa_k.dtype)
        dummy_gqa_v = jnp.zeros_like(dummy_gqa_k)

        cache_after = HybridCache(
            delta_M=cache_after.delta_M,
            delta_conv=cache_after.delta_conv,
            gqa_k=dummy_gqa_k,
            gqa_v=dummy_gqa_v,
            pos=cache_after.pos,
            paged_kv=paged_kv,
            kv_lens=kv_lens,
            page_indices=page_indices,
        )
        convert_ms = (time.perf_counter() - t_convert) * 1000
        print(f"  Paged cache conversion: {convert_ms:.0f} ms")
        print(f"  Paged KV shape: {paged_kv.shape}")

    # Warm-up: compile single decode step
    print("  Compiling decode step...")
    t_compile = time.perf_counter()
    next_tok, cache_step = decode_step(params, first_token, cache_after)
    next_tok.block_until_ready()
    compile_ms = (time.perf_counter() - t_compile) * 1000
    print(f"  Decode compilation: {compile_ms:.0f} ms")

    # Free warm-up results to reclaim HBM before timed loop
    del cache_step, next_tok
    jax.effects_barrier()

    # Timed runs — measure decode steps separately from prefill
    with maybe_profile("decode", profile):
        step_times = []  # per-step wall times across all runs
        t0_total = time.perf_counter()
        for _ in range(n_runs):
            logits, c = prefill_fn(params, prompt_tokens, cache)
            tok = jnp.argmax(logits[:, 0, :], axis=-1)

            if use_rpa:
                # Re-attach paged cache after prefill
                c = HybridCache(
                    delta_M=c.delta_M, delta_conv=c.delta_conv,
                    gqa_k=c.gqa_k, gqa_v=c.gqa_v, pos=c.pos,
                    paged_kv=cache_after.paged_kv,
                    kv_lens=cache_after.kv_lens,
                    page_indices=cache_after.page_indices,
                )

            for _step in range(n_decode_steps):
                t_step = time.perf_counter()
                tok, c = decode_step(params, tok, c)
                tok.block_until_ready()
                step_times.append((time.perf_counter() - t_step) * 1000)
        elapsed_total = time.perf_counter() - t0_total

    # Separate first step (includes data transfer overhead) from steady-state
    steps_per_run = n_decode_steps
    first_steps = [step_times[i * steps_per_run] for i in range(n_runs)]
    steady_steps = [t for i, t in enumerate(step_times) if i % steps_per_run != 0]

    avg_first_ms = sum(first_steps) / len(first_steps)
    avg_steady_ms = sum(steady_steps) / len(steady_steps) if steady_steps else avg_first_ms
    avg_all_ms = sum(step_times) / len(step_times)

    tokens_per_sec = B / (avg_steady_ms / 1000)
    tps_per_chip = tokens_per_sec / n_devices

    print(f"\n  DECODE RESULTS {'(RPA)' if use_rpa else '(contiguous)'}")
    print(f"  {'First step:':<20s} {avg_first_ms:.2f} ms  (includes data transfer)")
    print(f"  {'Steady-state:':<20s} {avg_steady_ms:.2f} ms/step")
    print(f"  {'All steps avg:':<20s} {avg_all_ms:.2f} ms/step")
    print(f"  {'Throughput:':<20s} {tokens_per_sec:,.0f} tok/s  (steady-state)")
    print(f"  {'TPS/chip:':<20s} {tps_per_chip:,.1f} tok/s/chip")
    print(f"  {'Decode steps:':<20s} {n_decode_steps}")
    print(f"  {'Total wall time:':<20s} {elapsed_total*1000:.0f} ms ({n_runs} runs)")
    return avg_all_ms, avg_steady_ms, tokens_per_sec


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
    parser.add_argument('--dtype', default='float32', choices=['float32', 'bfloat16', 'fp8'],
                        help='float32/bfloat16: uniform dtype. fp8: weights in fp8, activations/cache in bf16.')
    parser.add_argument('--chunk-size', type=int, default=None,
                        help='DeltaNet prefill chunk size (default: from config)')
    parser.add_argument('--n-layers', type=int, default=None,
                        help='Override number of layers (must be divisible by 4)')
    parser.add_argument('--n-experts', type=int, default=None,
                        help='Override number of routed experts')
    parser.add_argument('--dp', type=int, default=1,
                        help='Data-parallel factor. Creates 2D mesh (dp, tp) when > 1.')
    parser.add_argument('--roofline', action='store_true',
                        help='Print roofline analysis (FLOPs, HBM, arithmetic intensity). '
                             'Tracing can be slow for large models — use --chunk-size 32 for faster analysis.')
    parser.add_argument('--use-rpa', action='store_true',
                        help='Use RPA v3 kernel for GQA decode (paged KV cache).')
    args = parser.parse_args()

    if args.max_seq_len is None:
        args.max_seq_len = args.prompt_len + args.decode_steps + 64

    # Config
    from dataclasses import replace
    cfg = get_config(args.config)
    overrides = {}
    if args.chunk_size is not None:
        overrides['delta_chunk_size'] = args.chunk_size
    if args.n_layers is not None:
        assert args.n_layers % cfg.full_attention_interval == 0, \
            f"--n-layers must be divisible by {cfg.full_attention_interval}"
        overrides['n_layers'] = args.n_layers
    if args.n_experts is not None:
        overrides['n_routed_experts'] = args.n_experts
        # Clamp top-k to not exceed number of experts
        current_top_k = overrides.get('n_experts_per_token', cfg.n_experts_per_token)
        if current_top_k > args.n_experts:
            overrides['n_experts_per_token'] = min(current_top_k, args.n_experts)
    if overrides:
        cfg = replace(cfg, **overrides)
    axis_rules = get_axis_rules(args.sharding)

    print("=" * 70)
    print("QWEN3.5 INFERENCE BENCHMARK")
    print("=" * 70)
    print(f"  Config:         {args.config} ({cfg.n_layers}L, {cfg.d_model}D, "
          f"{cfg.n_routed_experts}E top-{cfg.n_experts_per_token})")
    print(f"  Sharding:       Config {args.sharding} (dp={args.dp})")
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
    print(f"  RPA decode:     {args.use_rpa}")

    # Initialize model
    use_fp8 = args.dtype == 'fp8'
    init_dtype = jnp.bfloat16 if args.dtype in ('bfloat16', 'fp8') else jnp.float32
    cache_dtype = init_dtype  # activations and cache always match init dtype

    # Setup mesh and sharding
    mesh = None
    cache_sharding = None
    n_dev = args.devices or jax.device_count()
    multihost = jax.process_count() > 1

    if axis_rules is not None:
        mesh = make_mesh(n_devices=n_dev, dp=args.dp)
        cache_sharding = make_cache_sharding(cfg, mesh, axis_rules,
                                             batch_size=args.batch_size)

    print(f"\nInitializing model ({args.dtype})...")
    t0 = time.perf_counter()

    if multihost and axis_rules is not None:
        # Multi-host: create sharded zero arrays directly on device.
        # We can't use init_params + shard_params because jax.device_put
        # does an all-gather verification that OOMs (expert weights > chip HBM).
        # For benchmarking, zeros give identical latency to random values.
        print(f"  Multi-host init: {jax.process_count()} hosts, {n_dev} devices")
        # 1) Get abstract shapes
        _init_fn = lambda key: init_params(cfg, key, dtype=init_dtype, fp8=use_fp8)
        abstract_params = jax.eval_shape(_init_fn, jax.random.key(0))
        # 2) Build matching sharding tree
        param_shardings = make_param_shardings(abstract_params, mesh, cfg, axis_rules)
        # Debug: print per-param sharding and memory breakdown
        if jax.process_index() == 0:
            print(f"\n  {'PARAM MEMORY BREAKDOWN':=^70}")
            print(f"  {'Path':<50s} {'Global Shape':<28s} {'Shard Shape':<20s} {'Dtype':<8s} {'Per-Dev MB':>10s}")
            print(f"  {'-'*118}")
            total_per_dev = 0
            def _print_leaf(path, aval, sharding):
                nonlocal total_per_dev
                path_str = '.'.join(
                    str(k).strip("[]'.\"") for k in path
                    if not str(k).strip("[]'.\"").isdigit()
                )
                spec = sharding.spec
                # Compute shard shape
                shard_shape = list(aval.shape)
                if hasattr(spec, '__iter__'):
                    for i, axis in enumerate(spec):
                        if axis is not None and i < len(shard_shape):
                            shard_shape[i] = shard_shape[i] // mesh.shape[axis]
                shard_shape = tuple(shard_shape)
                bytes_per = 1 if 'float8' in str(aval.dtype) else (4 if aval.dtype == jnp.float32 else 2)
                shard_bytes = 1
                for s in shard_shape:
                    shard_bytes *= s
                shard_bytes *= bytes_per
                shard_mb = shard_bytes / 1e6
                total_per_dev += shard_mb
                # Only print params > 1 MB
                if shard_mb >= 1.0:
                    shape_str = str(aval.shape)
                    shard_str = str(shard_shape)
                    print(f"  {path_str:<50s} {shape_str:<28s} {shard_str:<20s} {str(aval.dtype):<8s} {shard_mb:>10.1f}")
            jax.tree_util.tree_map_with_path(_print_leaf, abstract_params, param_shardings)
            print(f"  {'-'*118}")
            print(f"  {'TOTAL PER DEVICE:':<100s} {total_per_dev:>10.1f} MB")
            print(f"  {'':=^70}\n")

        # 3) Create zeros with correct sharding — each device gets only its shard.
        #    jax.make_array_from_callback creates per-shard data locally,
        #    no all-gather needed.  Use np.zeros (host RAM) not jnp.zeros
        #    to avoid staging temporaries on the default TPU device.
        cpu = jax.devices('cpu')[0]
        def _make_zeros(aval, sharding):
            is_exotic = 'float8' in str(aval.dtype)
            def _cb(idx):
                shard_shape = tuple(
                    (s.stop - s.start) if s.start is not None else dim
                    for s, dim in zip(idx, aval.shape)
                )
                if is_exotic:
                    # fp8 dtypes not in numpy — create on CPU device
                    with jax.default_device(cpu):
                        return jnp.zeros(shard_shape, dtype=aval.dtype)
                return np.zeros(shard_shape, dtype=np.dtype(str(aval.dtype)))
            if aval.shape == ():
                def _scalar_cb(idx):
                    if is_exotic:
                        with jax.default_device(cpu):
                            return jnp.zeros((), dtype=aval.dtype)
                    return np.zeros((), dtype=np.dtype(str(aval.dtype)))
                return jax.make_array_from_callback((), sharding, _scalar_cb)
            return jax.make_array_from_callback(aval.shape, sharding, _cb)
        with mesh:
            params = jax.tree.map(_make_zeros, abstract_params, param_shardings)
        jax.tree.map(lambda x: x.block_until_ready(), params)
    else:
        # Single-host: init on CPU then optionally shard
        with jax.default_device(jax.devices('cpu')[0]):
            params = init_params(cfg, jax.random.key(0), dtype=init_dtype, fp8=use_fp8)
        if axis_rules is not None:
            print(f"\nSharding params across {n_dev} devices...")
            params = shard_params(params, mesh, cfg, axis_rules)

    n_params = count_params(params)
    init_ms = (time.perf_counter() - t0) * 1000

    if use_fp8:
        from jax_gpt.models.qwen35.quantize import count_fp8_params
        total, fp8_count = count_fp8_params(params)
        non_fp8 = total - fp8_count
        est_gb = (fp8_count * 1 + non_fp8 * 2) / 1e9  # fp8=1byte, rest=bf16=2bytes
        print(f"  Params:         {n_params:,} (fp8 weights + bf16 norms/cache)")
        print(f"  FP8 weights:    {fp8_count:,} / {total:,} ({100*fp8_count/max(total,1):.0f}%)")
        print(f"  Est. memory:    {est_gb:.2f} GB")
    else:
        bytes_per = 2 if init_dtype == jnp.bfloat16 else 4
        print(f"  Params:         {n_params:,} ({n_params * bytes_per / 1e9:.2f} GB {args.dtype})")
    print(f"  Init time:      {init_ms:.0f} ms")

    # Inputs — shard batch dim on dp when using 2D mesh
    tokens = jnp.ones((args.batch_size, args.prompt_len), dtype=jnp.int32)
    if mesh is not None and args.dp > 1:
        dp_axis = 'dp' if 'dp' in mesh.axis_names else None
        tok_sharding = NamedSharding(mesh, P(dp_axis, None))
        if multihost:
            def _tok_cb(idx):
                shard_shape = tuple(
                    (s.stop - s.start) if s.start is not None else dim
                    for s, dim in zip(idx, tokens.shape)
                )
                return np.ones(shard_shape, dtype=np.int32)
            tokens = jax.make_array_from_callback(tokens.shape, tok_sharding, _tok_cb)
        else:
            tokens = jax.device_put(tokens, tok_sharding)

    if multihost and mesh is not None:
        # Multi-host: init cache directly on device with make_array_from_callback
        from jax_gpt.models.qwen35.sharding import _safe_spec
        tp_axis = axis_rules.get('delta_v_heads')
        gqa_kv_axis = axis_rules.get('gqa_kv_heads')
        dp_axis = 'dp' if 'dp' in mesh.axis_names else None

        n_groups = cfg.n_groups
        n_delta = cfg.full_attention_interval - 1
        B = args.batch_size
        max_len = args.max_seq_len
        conv_dim = cfg.delta_n_qk_heads * cfg.delta_qk_head_dim * 2 + cfg.delta_n_v_heads * cfg.delta_v_head_dim

        cache_fields = {
            'delta_M': ((n_groups, n_delta, B, cfg.delta_n_v_heads, cfg.delta_qk_head_dim, cfg.delta_v_head_dim),
                        P(None, None, dp_axis, tp_axis, None, None)),
            'delta_conv': ((n_groups, n_delta, B, conv_dim, cfg.delta_conv_kernel),
                           P(None, None, dp_axis, tp_axis, None)),
            'gqa_k': ((n_groups, B, cfg.gqa_n_kv_heads, max_len, cfg.gqa_head_dim),
                       P(None, dp_axis, gqa_kv_axis, None, None)),
            'gqa_v': ((n_groups, B, cfg.gqa_n_kv_heads, max_len, cfg.gqa_head_dim),
                       P(None, dp_axis, gqa_kv_axis, None, None)),
        }

        cache_arrays = {}
        with mesh:
            for name, (shape, spec) in cache_fields.items():
                safe = _safe_spec(spec, shape, mesh)
                sharding = NamedSharding(mesh, safe)
                def _cb(idx, dt=cache_dtype, sh=shape):
                    shard_shape = tuple(
                        (s.stop - s.start) if s.start is not None else dim
                        for s, dim in zip(idx, sh)
                    )
                    with jax.default_device(cpu):
                        return jnp.zeros(shard_shape, dtype=dt)
                cache_arrays[name] = jax.make_array_from_callback(shape, sharding, _cb)
            pos_sharding = NamedSharding(mesh, P())
            cache_arrays['pos'] = jax.make_array_from_callback((), pos_sharding,
                                                                lambda idx: np.array(0, dtype=np.int32))
        cache = HybridCache(**cache_arrays)
    else:
        with jax.default_device(jax.devices('cpu')[0]):
            cache = init_cache(cfg, args.batch_size, args.max_seq_len, dtype=cache_dtype)
        if mesh is not None:
            cache = shard_cache(cache, mesh, cfg, axis_rules)

    # ── Memory report: enumerate all HBM buffers ──────────────────────
    def _report_memory(label, tree, indent=4):
        """Print per-buffer sizes for a pytree of arrays."""
        pad = ' ' * indent
        flat = jax.tree.leaves(tree)
        total = 0
        rows = []
        for leaf in flat:
            if hasattr(leaf, 'shape'):
                nbytes = leaf.size * leaf.dtype.itemsize
                total += nbytes
                # Per-device size (addressable shard)
                try:
                    shard_shape = leaf.addressable_shards[0].data.shape
                    dev_bytes = 1
                    for s in shard_shape:
                        dev_bytes *= s
                    dev_bytes *= leaf.dtype.itemsize
                except Exception:
                    dev_bytes = nbytes
                rows.append((str(leaf.shape), str(leaf.dtype),
                             f"{nbytes/1e9:.3f}", f"{dev_bytes/1e9:.6f}"))
        print(f"\n{pad}{label} ({len(flat)} arrays, {total/1e9:.3f} GB total)")
        print(f"{pad}{'Shape':<45s} {'Dtype':<15s} {'Global GB':<12s} {'Per-TC GB'}")
        print(f"{pad}{'-'*90}")
        # Group by shape for compactness
        from collections import Counter
        shape_counts = Counter()
        shape_info = {}
        for shape, dtype, gb, dev_gb in rows:
            key = (shape, dtype)
            shape_counts[key] += 1
            shape_info[key] = (gb, dev_gb)
        for (shape, dtype), count in sorted(shape_counts.items(),
                                             key=lambda x: -float(shape_info[x[0]][1])):
            gb, dev_gb = shape_info[(shape, dtype)]
            cnt = f" x{count}" if count > 1 else ""
            print(f"{pad}{shape:<45s} {dtype:<15s} {gb:<12s} {dev_gb}{cnt}")

    print("\n" + "=" * 70)
    print("HBM MEMORY REPORT")
    print("=" * 70)
    _report_memory("PARAMS", params)
    _report_memory("CACHE", cache)
    _report_memory("TOKENS", tokens)

    # Device memory stats
    try:
        dev = jax.local_devices()[0]
        stats = dev.memory_stats()
        if stats:
            print(f"\n    Device {dev.id} memory stats:")
            for k, v in sorted(stats.items()):
                if isinstance(v, (int, float)) and v > 0:
                    print(f"      {k:<40s} {v/1e9:.3f} GB" if v > 1e6
                          else f"      {k:<40s} {v}")
    except Exception as e:
        print(f"    (memory_stats unavailable: {e})")

    print("=" * 70 + "\n")

    # Roofline analysis (before benchmarks — no actual computation needed)
    if args.roofline:
        if use_fp8:
            print("\n  [roofline] Skipped — not supported with fp8 params. "
                  "Run with --dtype bfloat16 for roofline analysis.")
        else:
            run_roofline_analysis(params, cfg, cache, args.prompt_len, args.batch_size)

    # Run benchmarks
    ctx = mesh if mesh is not None else contextlib.nullcontext()

    n_ep_devices = (args.devices or jax.device_count()) if axis_rules is not None else 1

    with ctx:
        if not args.skip_prefill:
            print(f"\n{'─'*70}")
            print(f"PREFILL BENCHMARK")
            print(f"{'─'*70}")
            run_prefill_benchmark(
                params, cfg, cache, tokens, cache_sharding,
                args.n_runs, args.profile, mesh, n_devices=n_ep_devices,
            )

        if not args.skip_decode:
            print(f"\n{'─'*70}")
            print(f"DECODE BENCHMARK {'(RPA)' if args.use_rpa else ''}")
            print(f"{'─'*70}")
            prompt = jnp.ones((args.batch_size, args.prompt_len), dtype=jnp.int32)
            run_decode_benchmark(
                params, cfg, cache, prompt, args.decode_steps, cache_sharding,
                args.n_runs, args.profile, mesh, n_devices=n_ep_devices,
                use_rpa=args.use_rpa,
            )

    print(f"\n{'='*70}")
    print("DONE")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
