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
    scan_mode: str = 'scan',
    moe_backend: str = 'ragged_dot',
):
    """Benchmark prefill latency."""
    B, T = tokens.shape

    @jax.jit
    def prefill(p, t, c):
        return forward(p, t, cfg, cache=c, is_decode=False,
                       cache_sharding=cache_sharding, n_devices=n_devices, mesh=mesh,
                       last_logit_only=True, scan_mode=scan_mode,
                       moe_backend=moe_backend)

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

    # P2: Fix TPS/chip — divide by physical chips, not devices (TCs)
    n_chips = n_devices
    try:
        from third_party.tpu_inference.tpu_info import get_num_cores_per_chip
        n_chips = n_devices // get_num_cores_per_chip()
    except (ImportError, Exception):
        pass
    tps_per_chip = tokens_per_sec / n_chips

    print(f"\n  PREFILL RESULTS")
    print(f"  {'Avg latency:':<20s} {avg_ms:.2f} ms")
    print(f"  {'Throughput:':<20s} {tokens_per_sec:,.0f} tokens/sec")
    print(f"  {'TPS/chip:':<20s} {tps_per_chip:,.1f} tokens/sec/chip ({n_chips} chips)")
    print(f"  {'Tokens:':<20s} {B} × {T} = {B*T}")
    return avg_ms, tokens_per_sec


def run_decode_benchmark(
    params, cfg, cache, prompt_tokens, n_decode_steps, cache_sharding,
    n_runs: int, profile: bool, mesh, n_devices: int = 1,
    use_rpa: bool = False,
    scan_mode: str = 'scan',
    moe_backend: str = 'ragged_dot',
    micro_batches: int = 1,
    max_seq_len: int | None = None,
    prefill_micro_batches: int = 0,
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
                       mesh=mesh, last_logit_only=True, scan_mode=scan_mode,
                       moe_backend=moe_backend)

    # Dispatch optimization: AOT compile and call with pre-flattened args.
    # The standard jax.jit dispatch path takes ~84ms/step on multi-host TPU
    # due to buffer binding overhead: 74 pytree leaves × 64 devices = 4,736
    # buffer handles per dispatch.
    #
    # Strategy: define the function normally (params as arg), compile once via
    # jax.jit().lower().compile(), then in the hot loop call via the Compiled
    # object which has a faster dispatch path.
    n_param_leaves = len(jax.tree.leaves(params))
    n_cache_leaves = len(jax.tree.leaves(cache))
    print(f"  Dispatch pytree: {n_param_leaves} param + {n_cache_leaves} cache"
          f" + 1 tok = {n_param_leaves + n_cache_leaves + 1} leaves")

    if use_rpa and HAS_RPA and scan_mode == 'unrolled':
        # Unrolled RPA: single JIT with Python for-loop over groups.
        # NOTE: donate_argnums=(2,) would save ~6 GB HBM but breaks the benchmark
        # loop: the warmup call donates cache_after, then the run loop tries to
        # reuse it. Needs a per-run cache copy strategy to fix properly.
        @jax.jit
        def decode_step(p, tok, c):
            logits, new_c = forward(
                p, tok[:, None], cfg, cache=c, is_decode=True,
                cache_sharding=cache_sharding, n_devices=n_devices,
                mesh=mesh, use_rpa=True, scan_mode='unrolled',
                moe_backend=moe_backend,
            )
            next_token = jnp.argmax(logits[:, 0, :], axis=-1)
            return next_token, new_c

    elif use_rpa and HAS_RPA:
        # Single-JIT scan path (scan_mode='scan' + use_rpa).
        # Uses XLA while_loop over layers → only 1 layer's buffers live at once.
        # Replaces the legacy per-group JIT which had high Python dispatch overhead.
        # With einsum-based deltanet this is the only path that fits in HBM:
        # unrolled at BS=2048 needs ~11.8 GB HLO temps, scan needs ~537 MB.
        @jax.jit
        def decode_step(p, tok, c):
            logits, new_c = forward(
                p, tok[:, None], cfg, cache=c, is_decode=True,
                cache_sharding=cache_sharding, n_devices=n_devices,
                mesh=mesh, use_rpa=True, scan_mode=scan_mode,
                moe_backend=moe_backend,
            )
            next_token = jnp.argmax(logits[:, 0, :], axis=-1)
            return next_token, new_c

    else:
        @jax.jit
        def decode_step(p, tok, c):
            logits, new_c = forward(
                p, tok[:, None], cfg, cache=c, is_decode=True,
                cache_sharding=cache_sharding, n_devices=n_devices,
                mesh=mesh, scan_mode=scan_mode,
                moe_backend=moe_backend,
            )
            next_token = jnp.argmax(logits[:, 0, :], axis=-1)
            return next_token, new_c

    # For RPA decode, we can skip the full-model prefill (which may OOM at
    # 60 layers due to XLA copying all expert weights) and create the paged
    # cache directly from zeros.  Decode latency is identical regardless of
    # cache contents — it's a pure shape/sharding benchmark.
    _paged = False

    if use_rpa:
        from tpu_inference.kernels.ragged_paged_attention.v3.util import cdiv

        page_size = 64
        prefill_len = prompt_tokens.shape[1]  # pretend we prefilled this many tokens
        # When use_rpa=True, gqa_k is a tiny stub (shape[3]=1). Use max_seq_len
        # if provided; otherwise fall back to gqa_k shape (non-rpa path).
        max_len = max_seq_len if (use_rpa and max_seq_len is not None) else cache.gqa_k.shape[3]
        pages_per_seq = cdiv(max_len, page_size)
        n_groups = cache.delta_M.shape[0]
        total_pages = B * pages_per_seq

        print(f"  Skipping full prefill (decode-only benchmark).")
        print(f"  Creating paged KV cache directly (page_size={page_size}, "
              f"pages_per_seq={pages_per_seq})...")

        cache_dtype = cache.delta_M.dtype
        kv_packed = 2  # K and V packed together
        n_kv_heads = cfg.gqa_n_kv_heads
        head_dim = cfg.gqa_head_dim

        if mesh is not None:
            from jax.sharding import NamedSharding, PartitionSpec as P
            dp_axis = 'dp' if 'dp' in mesh.axis_names else None
            cpu = jax.devices('cpu')[0]

            paged_kv_shape = (n_groups, total_pages, page_size, kv_packed, n_kv_heads, head_dim)
            paged_kv_sharding = NamedSharding(mesh, P(None, dp_axis, None, None, None, None))
            def _paged_cb(idx):
                shard_shape = tuple(
                    (s.stop - s.start) if s.start is not None else dim
                    for s, dim in zip(idx, paged_kv_shape)
                )
                with jax.default_device(cpu):
                    return jnp.zeros(shard_shape, dtype=cache_dtype)
            paged_kv = jax.make_array_from_callback(paged_kv_shape, paged_kv_sharding, _paged_cb)

            kv_lens_sharding = NamedSharding(mesh, P(dp_axis))
            kv_lens = jax.make_array_from_callback(
                (B,), kv_lens_sharding,
                lambda idx: np.full(((idx[0].stop - idx[0].start),), prefill_len, dtype=np.int32))
            page_indices_sharding = NamedSharding(mesh, P(dp_axis))
            page_indices = jax.make_array_from_callback(
                (total_pages,), page_indices_sharding,
                lambda idx: np.arange(idx[0].start, idx[0].stop, dtype=np.int32))
        else:
            paged_kv = jnp.zeros((n_groups, total_pages, page_size, kv_packed, n_kv_heads, head_dim),
                                  dtype=cache_dtype)
            kv_lens = jnp.full((B,), prefill_len, dtype=jnp.int32)
            page_indices = jnp.arange(total_pages, dtype=jnp.int32)

        dummy_gqa_k = jnp.zeros((n_groups, 1, 1, 1, 1), dtype=cache_dtype)
        dummy_gqa_v = jnp.zeros_like(dummy_gqa_k)

        if prefill_micro_batches > 0:
            # ── Chunked prefill → paged KV cache (no scatter) ────────────────
            # Run prefill in N micro-batches. Collect raw gqa_k/v from each chunk,
            # concatenate along the batch axis after the loop, then convert to paged
            # format once via contiguous_to_paged. This avoids a 50+ GB jit_scatter
            # program that cannot fit in HBM alongside model weights.
            import gc
            from jax_gpt.models.qwen35.paged_cache import contiguous_to_paged
            from jax_gpt.models.qwen35.sharding import _safe_spec

            assert B % prefill_micro_batches == 0, (
                f"--prefill-micro-batches ({prefill_micro_batches}) must divide "
                f"batch_size ({B})"
            )
            pfill_bs = B // prefill_micro_batches

            n_delta = cache.delta_M.shape[1]
            conv_dim_size = cache.delta_conv.shape[3]

            @jax.jit
            def prefill_chunk_fn(p, t, c):
                return forward(p, t, cfg, cache=c, is_decode=False,
                               cache_sharding=cache_sharding, n_devices=n_devices,
                               mesh=mesh, last_logit_only=True, scan_mode=scan_mode,
                               moe_backend=moe_backend)

            # gqa_len must match pages_per_seq*page_size so contiguous_to_paged
            # produces exactly pages_per_seq pages/seq after concatenation.
            chunk_gqa_len = pages_per_seq * page_size

            def _make_chunk_cache(bs_local):
                """Zero-initialized HybridCache for bs_local seqs with gqa_len=chunk_gqa_len."""
                if mesh is not None:
                    # Reuse sharding specs from the existing cache (same axes, smaller batch)
                    chunk_fields = {
                        'delta_M': (
                            (n_groups, n_delta, bs_local, cfg.delta_n_v_heads,
                             cfg.delta_qk_head_dim, cfg.delta_v_head_dim),
                            cache.delta_M.sharding.spec,
                        ),
                        'delta_conv': (
                            (n_groups, n_delta, bs_local, conv_dim_size, cfg.delta_conv_kernel),
                            cache.delta_conv.sharding.spec,
                        ),
                        'gqa_k': (
                            (n_groups, bs_local, cfg.gqa_n_kv_heads, chunk_gqa_len, cfg.gqa_head_dim),
                            P(None, dp_axis, cache.gqa_k.sharding.spec[2], None, None),
                        ),
                        'gqa_v': (
                            (n_groups, bs_local, cfg.gqa_n_kv_heads, chunk_gqa_len, cfg.gqa_head_dim),
                            P(None, dp_axis, cache.gqa_k.sharding.spec[2], None, None),
                        ),
                    }
                    chunk_arrays = {}
                    with mesh:
                        for name, (shape, spec) in chunk_fields.items():
                            safe = _safe_spec(spec, shape, mesh)
                            sharding = NamedSharding(mesh, safe)
                            def _cb(idx, sh=shape, dt=cache_dtype):
                                shard_shape = tuple(
                                    (s.stop - s.start) if s.start is not None else dim
                                    for s, dim in zip(idx, sh)
                                )
                                with jax.default_device(cpu):
                                    return jnp.zeros(shard_shape, dtype=dt)
                            chunk_arrays[name] = jax.make_array_from_callback(
                                shape, sharding, _cb)
                        chunk_arrays['pos'] = jax.make_array_from_callback(
                            (), NamedSharding(mesh, P()),
                            lambda idx: np.array(0, dtype=np.int32))
                    return HybridCache(**chunk_arrays)
                else:
                    from jax_gpt.models.qwen35.cache import init_cache
                    return init_cache(cfg, bs_local, chunk_gqa_len, dtype=cache_dtype)

            print(f"  Chunked prefill: {prefill_micro_batches} micro-batches × BS={pfill_bs}")

            # Compile once on the first chunk
            print("  Compiling prefill chunk...")
            t_pfill_compile = time.perf_counter()
            _cc = _make_chunk_cache(pfill_bs)
            _ = prefill_chunk_fn(params, prompt_tokens[:pfill_bs], _cc)
            del _cc
            gc.collect()
            print(f"  Prefill chunk compilation: {(time.perf_counter()-t_pfill_compile)*1000:.0f} ms")

            # Run N chunks. Convert each chunk's gqa_k/v to paged format immediately
            # so we never hold all contiguous gqa_k/v tensors simultaneously.
            # Avoids: (1) accumulating B-wide contiguous tensors, (2) 50+ GB jit_scatter.
            all_paged_chunks = []  # each entry: list of n_groups paged tensors for that chunk
            all_delta_M, all_delta_conv, all_first_tokens = [], [], []

            for c in range(prefill_micro_batches):
                start, end = c * pfill_bs, (c + 1) * pfill_bs
                print(f"  Prefill chunk {c+1}/{prefill_micro_batches} (seqs {start}–{end-1})...")
                chunk_cache = _make_chunk_cache(pfill_bs)
                logits, c_out = prefill_chunk_fn(params, prompt_tokens[start:end], chunk_cache)
                logits.block_until_ready()

                # Immediately convert to paged (reshape+transpose — tiny XLA program).
                # Free contiguous gqa_k/v before moving to the next chunk.
                chunk_paged = contiguous_to_paged(c_out.gqa_k, c_out.gqa_v, prefill_len, page_size, cache_dtype)
                chunk_paged[0].block_until_ready()
                all_paged_chunks.append(chunk_paged)

                all_delta_M.append(c_out.delta_M)
                all_delta_conv.append(c_out.delta_conv)
                all_first_tokens.append(jnp.argmax(logits[:, 0, :], axis=-1))
                del c_out, chunk_cache, logits, chunk_paged
                gc.collect()

            # Concatenate paged chunks per group (axis=0 = pages axis), then stack groups.
            # Each chunk: (chunk_pages, page_size, kv_packed, packing, head_dim) per group.
            print("  Assembling paged KV cache...")
            n_groups_paged = len(all_paged_chunks[0])
            merged = [
                jnp.concatenate([all_paged_chunks[c][g] for c in range(prefill_micro_batches)], axis=0)
                for g in range(n_groups_paged)
            ]
            paged_kv = jnp.stack(merged, axis=0)  # (n_groups, total_pages, page_size, ...)
            del all_paged_chunks, merged
            gc.collect()

            prefill_delta_M = jnp.concatenate(all_delta_M, axis=2)
            prefill_delta_conv = jnp.concatenate(all_delta_conv, axis=2)
            first_token = jnp.concatenate(all_first_tokens, axis=0)
            print(f"  Chunked prefill complete. KV cache populated.")
        else:
            # Skip prefill: zero KV cache, zero delta states (decode-only benchmark).
            prefill_delta_M = cache.delta_M
            prefill_delta_conv = cache.delta_conv
            first_token = jnp.ones((B,), dtype=jnp.int32)
            if mesh is not None and 'dp' in mesh.axis_names:
                first_token = jax.device_put(first_token, NamedSharding(mesh, P(dp_axis)))

        cache_after = HybridCache(
            delta_M=prefill_delta_M,
            delta_conv=prefill_delta_conv,
            gqa_k=dummy_gqa_k,
            gqa_v=dummy_gqa_v,
            pos=jnp.array(prefill_len, dtype=jnp.int32),
            paged_kv=paged_kv,
            kv_lens=kv_lens,
            page_indices=page_indices,
        )
        print(f"  Paged KV shape: {paged_kv.shape}")

        # ── Paged decode path (micro_batches >= 1) ────────────────────────
        # Use this path whenever use_rpa=True and scan_mode=unrolled.
        # donate_argnums=(2,) aliases the input/output cache buffer so the
        # update is in-place — avoids doubling peak memory for the cache.
        # micro_batches=1 means one JIT call for the full batch (useful for
        # large BS like 4096 where HLO temps are constant but cache is big).
        # micro_batches>1 splits the batch into smaller JIT calls to reduce
        # per-call HLO temps if needed.
        if micro_batches >= 1 and scan_mode == 'unrolled':
            import functools, gc
            _paged = True
            assert B % micro_batches == 0, (
                f"decode_micro_batches ({micro_batches}) must divide "
                f"batch_size ({B})"
            )
            page_B = B // micro_batches
            page_total_pages = page_B * pages_per_seq
            print(f"  Paged decode: {micro_batches} micro-batches × BS={page_B}")

            # Free the full-batch paged_kv (cache_after) and the original init_cache
            # arrays before creating per-micro-batch page_caches.
            #
            # cache_after.paged_kv is ~17 GB (full BS=4096) but is never used in the
            # micro-batch path — each micro-batch creates its own page_cache instead.
            # cache.delta_M/delta_conv are 6 GB at BS=4096 and not passed to any JIT.
            # Holding all three simultaneously leaves only ~2 GB free for the 15 GB
            # program binary, causing RuntimeProgramAllocationFailure at v100.
            #
            # Extract the shape/sharding metadata we need, then delete.
            dm_tail = cache.delta_M.shape[3:]
            dc_tail = cache.delta_conv.shape[3:]
            dm_sharding = cache.delta_M.sharding
            dc_sharding = cache.delta_conv.sharding
            del cache_after, paged_kv, kv_lens, page_indices, dummy_gqa_k, dummy_gqa_v
            del cache
            gc.collect()

            # Build sharded zero arrays directly via make_array_from_callback
            # to avoid init_cache + shard_cache which would try to replicate
            # large gqa_k/v tensors on device (90 GB OOM with params already loaded).
            # Reuse sharding specs from the existing input cache (extracted above).
            cpu = jax.devices('cpu')[0]
            dm_global = (n_groups, 3, page_B) + dm_tail
            dc_global = (n_groups, 3, page_B) + dc_tail

            def _sharded_zeros(global_shape, sharding, dtype):
                def cb(idx):
                    shard_shape = tuple(
                        (s.stop - s.start) if s.start is not None else d
                        for s, d in zip(idx, global_shape)
                    )
                    with jax.default_device(cpu):
                        return jnp.zeros(shard_shape, dtype=dtype)
                return jax.make_array_from_callback(global_shape, sharding, cb)

            page_caches = []
            for _mi in range(micro_batches):
                page_delta_M = _sharded_zeros(dm_global, dm_sharding, cache_dtype)
                page_delta_conv = _sharded_zeros(dc_global, dc_sharding, cache_dtype)

                if mesh is not None:
                    pkv_shape = (n_groups, page_total_pages, page_size,
                                 kv_packed, n_kv_heads, head_dim)
                    pkv_sharding = NamedSharding(mesh, P(None, dp_axis, None, None, None, None))
                    def _pkv_cb(idx, _sh=pkv_shape):
                        shard_shape = tuple(
                            (s.stop - s.start) if s.start is not None else d
                            for s, d in zip(idx, _sh)
                        )
                        with jax.default_device(cpu):
                            return jnp.zeros(shard_shape, dtype=cache_dtype)
                    page_pkv = jax.make_array_from_callback(pkv_shape, pkv_sharding, _pkv_cb)
                    page_kv_lens = jax.make_array_from_callback(
                        (page_B,), NamedSharding(mesh, P(dp_axis)),
                        lambda idx: np.full(
                            (idx[0].stop - idx[0].start,), prefill_len, dtype=np.int32))
                    page_pi = jax.make_array_from_callback(
                        (page_total_pages,), NamedSharding(mesh, P(dp_axis)),
                        lambda idx: np.arange(idx[0].start, idx[0].stop, dtype=np.int32))
                else:
                    page_pkv = jnp.zeros(
                        (n_groups, page_total_pages, page_size,
                         kv_packed, n_kv_heads, head_dim),
                        dtype=cache_dtype)
                    page_kv_lens = jnp.full((page_B,), prefill_len, dtype=jnp.int32)
                    page_pi = jnp.arange(page_total_pages, dtype=jnp.int32)

                dummy_k = jnp.zeros((n_groups, 1, 1, 1, 1), dtype=cache_dtype)
                page_caches.append(HybridCache(
                    delta_M=page_delta_M,
                    delta_conv=page_delta_conv,
                    gqa_k=dummy_k,
                    gqa_v=jnp.zeros_like(dummy_k),
                    pos=jnp.array(prefill_len, dtype=jnp.int32),
                    paged_kv=page_pkv,
                    kv_lens=page_kv_lens,
                    page_indices=page_pi,
                ))
            print(f"  Paged KV shape per micro-batch: {page_caches[0].paged_kv.shape}")

            @functools.partial(jax.jit, donate_argnums=(2,))
            def decode_step_micro(p, tok, c):
                logits, new_c = forward(
                    p, tok[:, None], cfg, cache=c, is_decode=True,
                    cache_sharding=cache_sharding, n_devices=n_devices,
                    mesh=mesh, use_rpa=True, scan_mode='unrolled',
                    moe_backend=moe_backend,
                )
                return jnp.argmax(logits[:, 0, :], axis=-1), new_c

            # Warmup = first real decode step on page 0 (cache donated in-place)
            tok_page0 = first_token[:page_B]
            if mesh is not None and dp_axis is not None:
                tok_page0 = jax.device_put(tok_page0, NamedSharding(mesh, P(dp_axis)))
            print("  Compiling paged decode step...")
            t_compile = time.perf_counter()
            _, page_caches[0] = decode_step_micro(params, tok_page0, page_caches[0])
            jax.effects_barrier()
            compile_ms = (time.perf_counter() - t_compile) * 1000
            print(f"  Paged decode compilation: {compile_ms:.0f} ms")

            with maybe_profile("decode", profile):
                async_run_times = []
                for run_idx in range(n_runs):
                    tok = first_token
                    t0_run = time.perf_counter()
                    for _step in range(n_decode_steps):
                        page_toks = []
                        for mi in range(micro_batches):
                            s = mi * page_B
                            tok_mi = tok[s:s + page_B]
                            if mesh is not None and dp_axis is not None:
                                tok_mi = jax.device_put(
                                    tok_mi, NamedSharding(mesh, P(dp_axis)))
                            next_tok_mi, page_caches[mi] = decode_step_micro(
                                params, tok_mi, page_caches[mi])
                            page_toks.append(next_tok_mi)
                        tok = jnp.concatenate(page_toks, axis=0)
                    tok.block_until_ready()
                    async_run_times.append((time.perf_counter() - t0_run) * 1000)

            tok = first_token
            sync_times = []
            dispatch_times = []
            for _step in range(n_decode_steps):
                t_step = time.perf_counter()
                page_toks = []
                for mi in range(micro_batches):
                    s = mi * page_B
                    tok_mi = tok[s:s + page_B]
                    if mesh is not None and dp_axis is not None:
                        tok_mi = jax.device_put(
                            tok_mi, NamedSharding(mesh, P(dp_axis)))
                    next_tok_mi, page_caches[mi] = decode_step_micro(
                        params, tok_mi, page_caches[mi])
                    page_toks.append(next_tok_mi)
                tok = jnp.concatenate(page_toks, axis=0)
                dispatch_times.append((time.perf_counter() - t_step) * 1000)
                tok.block_until_ready()
                sync_times.append((time.perf_counter() - t_step) * 1000)
            elapsed_total = sum(async_run_times) / 1000

            # Compute summary metrics (same formulas as non-paged path below)
            avg_async_total_ms = sum(async_run_times) / len(async_run_times)
            avg_async_step_ms = avg_async_total_ms / n_decode_steps
            if len(async_run_times) > 1:
                avg_steady_async_ms = sum(async_run_times[1:]) / len(async_run_times[1:]) / n_decode_steps
            else:
                avg_steady_async_ms = avg_async_step_ms
            avg_sync_ms = sum(sync_times[1:]) / len(sync_times[1:]) if len(sync_times) > 1 else sync_times[0]
            avg_dispatch_ms = sum(dispatch_times[1:]) / len(dispatch_times[1:]) if len(dispatch_times) > 1 else dispatch_times[0]
            avg_steady_ms = avg_steady_async_ms
            avg_all_ms = avg_async_step_ms

    else:
        # Non-RPA: run prefill to fill contiguous KV cache
        print("  Compiling prefill (for decode)...")
        t_compile = time.perf_counter()
        logits, cache_after = prefill_fn(params, prompt_tokens, cache)
        logits.block_until_ready()
        first_token = jnp.argmax(logits[:, 0, :], axis=-1)
        compile_prefill_ms = (time.perf_counter() - t_compile) * 1000
        print(f"  Prefill compilation: {compile_prefill_ms:.0f} ms")

    use_compiled_loop = False  # while_loop OOMs (copies closed-over params)

    if not _paged:
        # Standard path: per-step Python loop
        print("  Compiling decode step...")
        t_compile = time.perf_counter()
        next_tok, cache_step = decode_step(params, first_token, cache_after)
        next_tok.block_until_ready()
        compile_ms = (time.perf_counter() - t_compile) * 1000
        print(f"  Decode compilation: {compile_ms:.0f} ms")

        del cache_step, next_tok
        jax.effects_barrier()

        # Async dispatch: dispatch all steps without per-step blocking.
        # This pipelines host dispatch with device execution, hiding the
        # ~2s per-step dispatch overhead (24.5x speedup measured in v65).
        with maybe_profile("decode", profile):
            async_run_times = []
            for run_idx in range(n_runs):
                if use_rpa:
                    c = cache_after
                    tok = first_token
                else:
                    logits, c = prefill_fn(params, prompt_tokens, cache)
                    tok = jnp.argmax(logits[:, 0, :], axis=-1)
                tok.block_until_ready()

                t0_run = time.perf_counter()
                for _step in range(n_decode_steps):
                    tok, c = decode_step(params, tok, c)
                tok.block_until_ready()
                run_ms = (time.perf_counter() - t0_run) * 1000
                async_run_times.append(run_ms)

            # Also run one sync measurement for comparison
            if use_rpa:
                c = cache_after
                tok = first_token
            else:
                logits, c = prefill_fn(params, prompt_tokens, cache)
                tok = jnp.argmax(logits[:, 0, :], axis=-1)

            sync_times = []
            dispatch_times = []
            for _step in range(n_decode_steps):
                t_step = time.perf_counter()
                tok, c = decode_step(params, tok, c)
                t_dispatched = time.perf_counter()
                dispatch_times.append((t_dispatched - t_step) * 1000)
                tok.block_until_ready()
                sync_times.append((time.perf_counter() - t_step) * 1000)
            elapsed_total = sum(async_run_times) / 1000

        # Async metrics (primary)
        avg_async_total_ms = sum(async_run_times) / len(async_run_times)
        avg_async_step_ms = avg_async_total_ms / n_decode_steps
        # Use first run as warmup if multiple runs
        if len(async_run_times) > 1:
            steady_async = async_run_times[1:]
            avg_steady_async_ms = sum(steady_async) / len(steady_async) / n_decode_steps
        else:
            avg_steady_async_ms = avg_async_step_ms

        # Sync metrics (comparison)
        avg_sync_ms = sum(sync_times[1:]) / len(sync_times[1:]) if len(sync_times) > 1 else sync_times[0]
        avg_dispatch_ms = sum(dispatch_times[1:]) / len(dispatch_times[1:]) if len(dispatch_times) > 1 else dispatch_times[0]

        # Use async step time as the primary metric
        avg_steady_ms = avg_steady_async_ms
        avg_all_ms = avg_async_step_ms

    # P2: Fix TPS/chip — divide by physical chips, not devices (TCs)
    # v7x has 2 TCs per chip; n_devices reports TCs (64), should be chips (32).
    n_chips = n_devices  # default: 1 TC per chip
    try:
        from third_party.tpu_inference.tpu_info import get_num_cores_per_chip
        cores_per_chip = get_num_cores_per_chip()
        n_chips = n_devices // cores_per_chip
    except (ImportError, Exception):
        pass
    tokens_per_sec = B / (avg_steady_ms / 1000)
    tps_per_chip = tokens_per_sec / n_chips

    print(f"\n  DECODE RESULTS {'(RPA)' if use_rpa else '(contiguous)'}")
    print(f"  {'Async dispatch:':<25s} {avg_steady_ms:.2f} ms/step  (pipelined)")
    print(f"  {'Throughput:':<25s} {tokens_per_sec:,.0f} tok/s")
    print(f"  {'TPS/chip:':<25s} {tps_per_chip:,.1f} tok/s/chip  ({n_chips} chips)")
    print(f"  {'Decode steps:':<25s} {n_decode_steps}")
    print(f"  {'Total wall time:':<25s} {elapsed_total*1000:.0f} ms ({n_runs} runs)")
    for i, rt in enumerate(async_run_times):
        print(f"    Run {i}: {rt:.0f} ms ({rt/n_decode_steps:.1f} ms/step)")
    print(f"\n  SYNC COMPARISON (1 run, per-step block_until_ready)")
    print(f"  {'Sync step time:':<25s} {avg_sync_ms:.2f} ms/step")
    print(f"  {'  dispatch:':<25s} {avg_dispatch_ms:.2f} ms")
    print(f"  {'  block:':<25s} {avg_sync_ms - avg_dispatch_ms:.2f} ms")
    print(f"  {'Async speedup:':<25s} {avg_sync_ms / avg_steady_ms:.1f}x")
    return avg_all_ms, avg_steady_ms, tokens_per_sec


def _export_measurements(args, cfg, mesh, params, cache, decode_results, use_fp8):
    """Write roofline measurement JSON after a decode benchmark run.

    Captures:
      - decode step time and TPS/chip from the benchmark
      - JAX roofline per-module flops/bytes (bf16 only)
      - XLA profile extraction via xla_shell (if PROFILE_DIR env var is set)

    If args.export_measurements already points to a predictions.json (from
    roofline_bridge.exporter), measurements are merged into it. Otherwise a
    measurement-only file is written.
    """
    import json
    from pathlib import Path
    from datetime import datetime, timezone

    bridge_path = Path(__file__).parent.parent / "ml-experiments" / "inference" / "roofline_bridge"
    if not bridge_path.exists():
        # Try relative to repo root
        for candidate in [
            Path(__file__).parent.parent / "ml-experiments" / "inference",
            Path("/home/sivaibhav_google_com/ml-experiments/inference"),
        ]:
            if (candidate / "roofline_bridge").exists():
                bridge_path = candidate / "roofline_bridge"
                break
    sys.path.insert(0, str(bridge_path.parent))

    try:
        from roofline_bridge.schema import (
            RooflineDataModel, RunConfig, HardwareConstants, SummaryStats,
            OpRecord, OpPrediction, HARDWARE_PRESETS
        )
        from roofline_bridge.op_mapper import aggregate_fusions, validate_profile
    except ImportError as e:
        print(f"  [export] Cannot import roofline_bridge: {e} — skipping export")
        return

    # Only rank 0 writes — all hosts have identical measurements.
    if jax.process_index() != 0:
        return

    out_str = args.export_measurements
    is_gcs = out_str.startswith("gs://")
    if is_gcs:
        import tempfile
        _local_tmp = Path(tempfile.mktemp(suffix="_roofline_meas.json"))
        out_path = _local_tmp
    else:
        out_path = Path(out_str)
    avg_all_ms, avg_steady_ms, tokens_per_sec = decode_results

    # Determine TPS/chip (benchmark already computed this internally; re-derive here)
    try:
        from third_party.tpu_inference.tpu_info import get_num_cores_per_chip
        cores_per_chip = get_num_cores_per_chip()
        n_chips = (args.devices or jax.device_count()) // cores_per_chip
    except Exception:
        n_chips = args.devices or jax.device_count()
    tps_per_chip = tokens_per_sec / n_chips

    print(f"\n  [export] Writing measurements to {out_path}")

    # --- JAX roofline (bf16 only) ---
    jax_modules: dict[str, object] = {}
    if not use_fp8:
        try:
            from jax.experimental.roofline import roofline
            B = args.batch_size
            param_shapes = jax.tree.map(lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype), params)
            cache_shapes = jax.tree.map(lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype), cache)
            tok_dec = jax.ShapeDtypeStruct((B, 1), jnp.int32)
            from jax_gpt.models.qwen35.model import forward
            def _fwd_dec(p, t, c):
                return forward(p, t, cfg, cache=c, is_decode=True)
            _, r = roofline(_fwd_dec)(param_shapes, tok_dec, cache_shapes)
            jax_modules["full_decode"] = r
            print(f"  [export] JAX roofline: decode FLOPs={r.flops:.3e}  HBM={r.hbm_bytes:.3e}")
        except Exception as e:
            print(f"  [export] JAX roofline failed: {e}")
    else:
        print("  [export] JAX roofline skipped (fp8 params)")

    # --- XLA profile extraction via xla_shell ---
    xla_measurements: dict = {}
    profile_dir = os.environ.get("PROFILE_DIR", "")
    if profile_dir and args.profile:
        try:
            import subprocess
            result = subprocess.run(
                [sys.executable, "-m", "xla_shell", "-c",
                 f"read_xplane {profile_dir}; list_fusions --json"],
                capture_output=True, text=True,
                cwd=str(Path(__file__).parent.parent / "ml-experiments" / "timing"),
                timeout=120,
            )
            if result.returncode == 0 and result.stdout.strip():
                fusions = json.loads(result.stdout)
                validate_profile(fusions)
                xla_measurements = aggregate_fusions(fusions)
                print(f"  [export] xla_shell: {len(xla_measurements)} ops mapped "
                      f"from {len(fusions)} fusions")
            else:
                print(f"  [export] xla_shell failed: {result.stderr[:200]}")
        except Exception as e:
            print(f"  [export] xla_shell extraction failed: {e}")

    # --- Build measurement-only model or merge into existing predictions ---
    hw_name = "v7x" if "v7x" in str(getattr(args, 'config', '')).lower() else "v5p"
    hw = HARDWARE_PRESETS.get(hw_name, HARDWARE_PRESETS["v7x"])

    # For GCS paths: try to download an existing predictions file to merge into.
    if is_gcs:
        try:
            from google.cloud import storage as _gcs
            _parts = out_str[5:].split("/", 1)
            _gcs.Client().bucket(_parts[0]).blob(_parts[1]).download_to_filename(str(out_path))
        except Exception:
            pass  # file doesn't exist yet — will create fresh

    if out_path.exists():
        try:
            meas_model = RooflineDataModel.from_json(str(out_path))
            print(f"  [export] Merging into existing {out_str if is_gcs else out_path}")
        except Exception:
            meas_model = None
    else:
        meas_model = None

    if meas_model is None:
        # Build measurement-only model
        from roofline_bridge.schema import OpPrediction
        n_devices = args.devices or jax.device_count()
        tp = n_devices // max(args.dp, 1)
        dp = args.dp
        run_config = RunConfig(
            model="qwen35-397b",
            hardware=hw_name,
            n_chips=n_chips,
            mesh=f"({dp},{tp})",
            dtype="W8A16" if use_fp8 else "bfloat16",
            batch_size=args.batch_size,
            prompt_len=args.prompt_len,
            decode_len=args.decode_steps,
            n_layers=None,
            tp=tp,
            dp=dp,
            ep=1,
            profile_gcs_path=profile_dir or None,
            image_tag=None,
        )
        meas_model = RooflineDataModel.new(run_config, hw, phase="decode")

    # Overlay measured step time and TPS/chip
    meas_model.summary.measured_step_ms = avg_steady_ms
    meas_model.summary.measured_tps_per_chip = tps_per_chip
    if "first_principles" not in meas_model.summary.sources:
        pass  # predictions file already has it
    if "benchmark" not in meas_model.summary.sources:
        meas_model.summary.sources.append("benchmark")

    # Overlay JAX roofline on full_decode op (store on summary)
    if "full_decode" in jax_modules:
        r = jax_modules["full_decode"]
        meas_model.summary.jax_roofline_flops = r.flops
        meas_model.summary.jax_roofline_hbm_bytes = r.hbm_bytes
        if "jax_roofline" not in meas_model.summary.sources:
            meas_model.summary.sources.append("jax_roofline")

    # Overlay xla_shell measurements on matching op records
    if xla_measurements:
        from roofline_bridge.schema import OpMeasurement
        op_map = {op.op_name: op for op in meas_model.ops}
        for op_name, agg in xla_measurements.items():
            if op_name in op_map:
                ai = agg["flops"] / agg["bytes_accessed"] if agg["bytes_accessed"] > 0 else 0.0
                ridge = hw.peak_flops_bf16_tflops * 1e12 / (hw.hbm_bw_tbps * 1e12)
                op_map[op_name].measurement = OpMeasurement(
                    source="xla_shell",
                    self_time_us=agg["self_time_us"],
                    flops=agg["flops"],
                    bytes_accessed=agg["bytes_accessed"],
                    arithmetic_intensity=ai,
                    bound_by="compute" if ai > ridge else "memory",
                    n_fusions_matched=agg["n_fusions_matched"],
                    tf_op_names=agg["tf_op_names"],
                    fusion_names=agg["fusion_names"],
                )
        if "xla_shell" not in meas_model.summary.sources:
            meas_model.summary.sources.append("xla_shell")

    meas_model.generated_at = datetime.now(timezone.utc).isoformat()
    meas_model.to_json(str(out_path))

    if is_gcs:
        from google.cloud import storage as _gcs
        _parts = out_str[5:].split("/", 1)
        _gcs.Client().bucket(_parts[0]).blob(_parts[1]).upload_from_filename(str(out_path))
        out_path.unlink(missing_ok=True)
        print(f"  [export] Uploaded to {out_str}  (sources: {meas_model.summary.sources})")
    else:
        print(f"  [export] Wrote {out_path}  (sources: {meas_model.summary.sources})")
    print(f"  [export] Measured step: {avg_steady_ms:.2f} ms  "
          f"TPS/chip: {tps_per_chip:.1f}")


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
    parser.add_argument('--scan-mode', default='scan', choices=['scan', 'unrolled'],
                        help='Group iteration strategy: scan (default) uses lax.scan; '
                             'unrolled uses Python for-loop inside JIT to avoid XLA copy insertion OOM.')
    parser.add_argument('--moe-backend', default='ragged_dot', choices=['ragged_dot', 'gmm'],
                        help='MoE expert matmul backend: ragged_dot (XLA) or gmm (Pallas kernel).')
    parser.add_argument('--decode-micro-batches', type=int, default=1,
                        help='Split decode into N micro-batches (each BS/N) to reduce JIT '
                             'compilation HBM footprint. Use 2 for BS=2048 if '
                             'RuntimeProgramAllocationFailure occurs. '
                             'Requires --use-rpa --scan-mode=unrolled.')
    parser.add_argument('--prefill-micro-batches', type=int, default=0,
                        help='Run real chunked prefill into paged KV cache using N micro-batches '
                             '(each BS/N sequences). 0 = skip prefill (decode-only benchmark). '
                             'Enables correctness testing at full decode BS. '
                             'Requires --use-rpa --scan-mode=unrolled.')
    parser.add_argument('--export-measurements', metavar='PATH', default=None,
                        help='Write roofline measurement JSON to PATH after decode benchmark. '
                             'If PATH already contains a predictions.json from exporter.py, '
                             'measurements are merged into it. Captures: decode step time, '
                             'TPS/chip, and JAX roofline (bf16 only). '
                             'XLA profile extraction requires --profile + PROFILE_DIR.')
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
    print(f"  Scan mode:      {args.scan_mode}")

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
        # When using RPA paged decode, the contiguous gqa_k/v are never used.
        # Allocate them as size-1 stubs to avoid wasting ~17 GB per TC at BS>=4096.
        gqa_len = 1 if args.use_rpa else max_len

        cache_fields = {
            'delta_M': ((n_groups, n_delta, B, cfg.delta_n_v_heads, cfg.delta_qk_head_dim, cfg.delta_v_head_dim),
                        P(None, None, dp_axis, tp_axis, None, None)),
            'delta_conv': ((n_groups, n_delta, B, conv_dim, cfg.delta_conv_kernel),
                           P(None, None, dp_axis, tp_axis, None)),
            'gqa_k': ((n_groups, B, cfg.gqa_n_kv_heads, gqa_len, cfg.gqa_head_dim),
                       P(None, dp_axis, gqa_kv_axis, None, None)),
            'gqa_v': ((n_groups, B, cfg.gqa_n_kv_heads, gqa_len, cfg.gqa_head_dim),
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
                scan_mode=args.scan_mode,
                moe_backend=args.moe_backend,
            )

        _decode_results = None
        if not args.skip_decode:
            print(f"\n{'─'*70}")
            print(f"DECODE BENCHMARK {'(RPA)' if args.use_rpa else ''}")
            print(f"{'─'*70}")
            prompt = jnp.ones((args.batch_size, args.prompt_len), dtype=jnp.int32)
            _decode_results = run_decode_benchmark(
                params, cfg, cache, prompt, args.decode_steps, cache_sharding,
                args.n_runs, args.profile, mesh, n_devices=n_ep_devices,
                use_rpa=args.use_rpa,
                scan_mode=args.scan_mode,
                moe_backend=args.moe_backend,
                micro_batches=args.decode_micro_batches,
                max_seq_len=args.max_seq_len,
                prefill_micro_batches=args.prefill_micro_batches,
            )

    # Post-benchmark peak memory report
    try:
        dev = jax.local_devices()[0]
        stats = dev.memory_stats()
        if stats:
            print(f"\n{'='*70}")
            print("PEAK HBM AFTER BENCHMARKS")
            print(f"{'='*70}")
            print(f"    Device {dev.id}:")
            peak = stats.get('peak_bytes_in_use', 0)
            current = stats.get('bytes_in_use', 0)
            limit = stats.get('bytes_limit', 0)
            print(f"      peak_bytes_in_use:  {peak/1e9:.3f} GB")
            print(f"      bytes_in_use:       {current/1e9:.3f} GB")
            print(f"      bytes_limit:        {limit/1e9:.3f} GB")
            print(f"      peak utilization:   {100*peak/limit:.1f}%" if limit else "")
            hlo_temps = peak - current
            print(f"      HLO temps (peak - current): {hlo_temps/1e9:.3f} GB")
    except Exception as e:
        print(f"    (peak memory stats unavailable: {e})")

    # ------------------------------------------------------------------
    # Export measurements (--export-measurements PATH)
    # ------------------------------------------------------------------
    if args.export_measurements and _decode_results is not None:
        _export_measurements(args, cfg, mesh, params, cache, _decode_results, use_fp8)

    print(f"\n{'='*70}")
    print("DONE")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
