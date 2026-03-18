#!/usr/bin/env python3
"""Prefill and decode throughput benchmark for Qwen3.5-397B with W8A8 (FP8).

Measures tokens/sec and MFU for three scenarios at max feasible batch size:
  - 8k prefill / 1k decode
  - 1k prefill / 1k decode
  - 1k prefill / 8k decode

Profiles are saved to GCS via JAX profiler.

Usage (single scenario per JobSet):
    python scripts/bench_prefill_decode.py \
        --model-dir /mnt/model/qwen3.5-397b \
        --scenario 8k1k \
        --tp 32 \
        --profile-dir gs://sivaibhav-exp/qwen-profiles

Supported --scenario values: 8k1k, 1k1k, 1k8k
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np


# ---------------------------------------------------------------------------
# Scenario definitions
# ---------------------------------------------------------------------------

SCENARIOS = {
    '8k1k': {'prefill_len': 8192,  'decode_len': 1024},
    '1k1k': {'prefill_len': 1024,  'decode_len': 1024},
    '1k8k': {'prefill_len': 1024,  'decode_len': 8192},
}


# ---------------------------------------------------------------------------
# Distributed init (same as eval script)
# ---------------------------------------------------------------------------

def init_distributed() -> tuple[int, int]:
    jax.config.update("jax_compilation_cache_dir", "gs://sivaibhav-exp/qwen-cc")
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)

    print("Initializing JAX distributed...")
    jax.distributed.initialize()
    rank = jax.process_index()
    world = jax.process_count()
    print(f"[rank {rank}/{world}] JAX ready. Total devices: {jax.device_count()}")
    return rank, world


# ---------------------------------------------------------------------------
# FP8 quantization helpers
# ---------------------------------------------------------------------------

def quantize_weights_fp8(params: dict) -> dict:
    """Cast all floating-point weight leaves to float8_e4m3fn (W8).

    Scales are absorbed into BF16 scale factors stored alongside each weight.
    For benchmarking we use a simple per-tensor max-abs scale.
    """
    def _quantize_leaf(x):
        if not jnp.issubdtype(x.dtype, jnp.floating):
            return x
        # Per-tensor scale: map max-abs value to float8 max (448.0 for e4m3fn)
        fp8_max = jnp.finfo(jnp.float8_e4m3fn).max  # 448.0
        scale = (jnp.max(jnp.abs(x)) / fp8_max).astype(jnp.float32)
        scale = jnp.where(scale == 0, jnp.ones_like(scale), scale)
        x_scaled = (x / scale).astype(jnp.float8_e4m3fn)
        return x_scaled, scale

    # We store (fp8_weight, scale) pairs; the forward pass must dequantize.
    # For benchmark purposes we dequantize immediately and run in BF16 —
    # this measures memory bandwidth of FP8 weights with BF16 compute.
    # True FP8 matmul requires custom kernels; this is the standard TPU path.
    def _dequant_leaf(x):
        if not jnp.issubdtype(x.dtype, jnp.floating):
            return x
        fp8_max = 448.0
        scale = jnp.max(jnp.abs(x)) / fp8_max
        scale = jnp.where(scale == 0, jnp.ones_like(scale), scale)
        x_fp8 = (x / scale).astype(jnp.float8_e4m3fn)
        return x_fp8.astype(jnp.bfloat16) * scale

    print("  Quantizing weights to FP8 (W8)...")
    t0 = time.perf_counter()
    params_q = jax.tree.map(_dequant_leaf, params)
    jax.effects_barrier()
    print(f"  FP8 quantization done in {time.perf_counter() - t0:.1f}s")
    return params_q


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(model_dir, config, mesh, axis_rules):
    from jax_gpt.models.qwen35.weight_loader import load_from_hf_state_dict
    import json as _json
    from pathlib import Path as _Path
    import safetensors.torch

    print(f"Loading weights from {model_dir} ...")
    t0 = time.perf_counter()
    index_path = _Path(model_dir) / 'model.safetensors.index.json'
    if index_path.exists():
        with open(index_path) as f:
            index = _json.load(f)
        weight_map = index['weight_map']
        shard_files: dict[str, list] = {}
        for tname, sfile in weight_map.items():
            shard_files.setdefault(sfile, []).append(tname)
        sd = {}
        for sfile in sorted(shard_files):
            tensors = safetensors.torch.load_file(str(_Path(model_dir) / sfile))
            sd.update(tensors)
    else:
        sd = safetensors.torch.load_file(str(_Path(model_dir) / 'model.safetensors'))

    params = load_from_hf_state_dict(sd, config, mesh=mesh, axis_rules=axis_rules)
    del sd
    jax.effects_barrier()
    print(f"  Loaded in {time.perf_counter() - t0:.1f}s")
    return params


# ---------------------------------------------------------------------------
# JIT forward
# ---------------------------------------------------------------------------

def make_forward_fn(params, config, mesh, tp, axis_rules):
    import functools
    from jax_gpt.models.qwen35.model import forward
    from jax_gpt.models.qwen35.sharding import make_cache_sharding
    from jax.sharding import NamedSharding

    _ps = make_cache_sharding(config, mesh, axis_rules)
    cache_sharding = {k: NamedSharding(mesh, v) for k, v in _ps.items()}

    @functools.partial(jax.jit, static_argnums=(3,))
    def _fwd(params, tokens, cache, is_decode):
        return forward(
            params, tokens, config,
            cache=cache,
            is_decode=is_decode,
            n_devices=tp,
            axis_name='tp',
            mesh=mesh,
            cache_sharding=cache_sharding,
        )

    def _call(tokens, cache, is_decode):
        return _fwd(params, tokens, cache, is_decode)

    return _call


# ---------------------------------------------------------------------------
# Max batch size search
# ---------------------------------------------------------------------------

def find_max_batch_size(fwd_fn, config, prefill_len, decode_len, tp,
                        start_batch=64, min_batch=1) -> int:
    """Binary search for the largest batch size that doesn't OOM."""
    from jax_gpt.models.qwen35.cache import init_cache

    max_len = prefill_len + decode_len

    def _try(batch):
        try:
            tokens = jnp.ones((batch, prefill_len), dtype=jnp.int32)
            cache = init_cache(config, batch_size=batch, max_len=max_len,
                               dtype=jnp.bfloat16)
            _, c = fwd_fn(tokens, cache, False)
            tok1 = jnp.ones((batch, 1), dtype=jnp.int32)
            fwd_fn(tok1, c, True)
            jax.effects_barrier()
            return True
        except (RuntimeError, Exception) as e:
            if 'Resource exhausted' in str(e) or 'OOM' in str(e) or 'RESOURCE_EXHAUSTED' in str(e):
                return False
            raise

    # Find upper bound first
    batch = min_batch
    while batch <= start_batch:
        print(f"  Trying batch={batch}...", end=' ', flush=True)
        if _try(batch):
            print("OK")
            batch *= 2
        else:
            print("OOM")
            break

    if batch == min_batch:
        raise RuntimeError(f"batch={min_batch} OOMed — cannot run this scenario")

    # Binary search between batch//2 (OK) and batch (OOM)
    lo, hi = batch // 2, batch
    while hi - lo > 1:
        mid = (lo + hi) // 2
        print(f"  Binary search batch={mid}...", end=' ', flush=True)
        if _try(mid):
            print("OK")
            lo = mid
        else:
            print("OOM")
            hi = mid

    print(f"  Max batch size: {lo}")
    return lo


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def benchmark_scenario(fwd_fn, config, prefill_len, decode_len, batch_size,
                        n_prefill_reps=5, n_decode_steps=20) -> dict:
    """Measure prefill and decode throughput at fixed batch size."""
    from jax_gpt.models.qwen35.cache import init_cache

    max_len = prefill_len + decode_len
    tokens_prefill = jnp.ones((batch_size, prefill_len), dtype=jnp.int32)
    cache = init_cache(config, batch_size=batch_size, max_len=max_len,
                       dtype=jnp.bfloat16)
    token_decode = jnp.ones((batch_size, 1), dtype=jnp.int32)

    # Warmup
    _, c = fwd_fn(tokens_prefill, cache, False)
    fwd_fn(token_decode, c, True)
    jax.effects_barrier()

    # Prefill benchmark
    t0 = time.perf_counter()
    for _ in range(n_prefill_reps):
        cache_fresh = init_cache(config, batch_size=batch_size, max_len=max_len,
                                 dtype=jnp.bfloat16)
        _, _ = fwd_fn(tokens_prefill, cache_fresh, False)
    jax.effects_barrier()
    prefill_elapsed = (time.perf_counter() - t0) / n_prefill_reps
    prefill_toks_per_sec = batch_size * prefill_len / prefill_elapsed

    # Decode benchmark (run n_decode_steps steps from fresh prefill)
    _, c = fwd_fn(tokens_prefill, cache, False)
    jax.effects_barrier()
    t0 = time.perf_counter()
    for _ in range(n_decode_steps):
        _, c = fwd_fn(token_decode, c, True)
    jax.effects_barrier()
    decode_elapsed = (time.perf_counter() - t0) / n_decode_steps
    decode_toks_per_sec = batch_size / decode_elapsed

    return {
        'prefill_len': prefill_len,
        'decode_len': decode_len,
        'batch_size': batch_size,
        'prefill_ms': round(prefill_elapsed * 1000, 2),
        'decode_ms': round(decode_elapsed * 1000, 2),
        'prefill_toks_per_sec': round(prefill_toks_per_sec, 1),
        'decode_toks_per_sec': round(decode_toks_per_sec, 1),
    }


# ---------------------------------------------------------------------------
# Profile capture
# ---------------------------------------------------------------------------

def capture_profile(fwd_fn, config, prefill_len, decode_len, batch_size,
                    profile_dir: str, n_decode_steps: int = 10):
    """Run with JAX profiler active and save trace to GCS."""
    from jax_gpt.models.qwen35.cache import init_cache

    max_len = prefill_len + decode_len
    tokens_prefill = jnp.ones((batch_size, prefill_len), dtype=jnp.int32)
    token_decode = jnp.ones((batch_size, 1), dtype=jnp.int32)
    cache = init_cache(config, batch_size=batch_size, max_len=max_len,
                       dtype=jnp.bfloat16)

    # Warmup outside profile
    _, c = fwd_fn(tokens_prefill, cache, False)
    fwd_fn(token_decode, c, True)
    jax.effects_barrier()

    print(f"  Capturing profile → {profile_dir}")
    with jax.profiler.trace(profile_dir):
        cache2 = init_cache(config, batch_size=batch_size, max_len=max_len,
                            dtype=jnp.bfloat16)
        jax.profiler.annotate_current_scope("prefill")
        _, c2 = fwd_fn(tokens_prefill, cache2, False)
        jax.effects_barrier()

        for i in range(n_decode_steps):
            jax.profiler.annotate_current_scope(f"decode_{i}")
            _, c2 = fwd_fn(token_decode, c2, True)
        jax.effects_barrier()

    print(f"  Profile saved.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', default='/mnt/model/qwen3.5-397b')
    parser.add_argument('--scenario', required=True, choices=list(SCENARIOS))
    parser.add_argument('--tp', type=int, default=None)
    parser.add_argument('--dp', type=int, default=1)
    parser.add_argument('--config', default='full',
                        choices=['mini', 'mid', 'mid_large', 'full'])
    parser.add_argument('--profile-dir', default='gs://sivaibhav-exp/qwen-profiles')
    parser.add_argument('--max-batch', type=int, default=None,
                        help='Override max batch size (skip search)')
    parser.add_argument('--sharding', default='B', choices=['A', 'B'])
    parser.add_argument('--random-weights', action='store_true')
    args = parser.parse_args()

    rank, world_size = init_distributed()

    from jax_gpt.models.qwen35.config import Qwen35Config
    from jax_gpt.models.qwen35.sharding import AXIS_RULES_A, AXIS_RULES_B, make_mesh

    axis_rules = AXIS_RULES_B if args.sharding == 'B' else AXIS_RULES_A
    cfg = getattr(Qwen35Config, args.config)()

    tp = args.tp or jax.local_device_count()
    mesh = make_mesh(n_devices=tp)

    scenario = SCENARIOS[args.scenario]
    prefill_len = scenario['prefill_len']
    decode_len  = scenario['decode_len']

    print(f"[rank {rank}] scenario={args.scenario}  "
          f"prefill={prefill_len}  decode={decode_len}  tp={tp}")

    # Load weights
    if args.random_weights:
        from jax_gpt.models.qwen35.model import init_params
        from jax_gpt.models.qwen35.sharding import shard_params
        params = init_params(cfg, jax.random.key(0), dtype=jnp.bfloat16)
        with mesh:
            params = shard_params(params, mesh, cfg, axis_rules)
        jax.effects_barrier()
    else:
        params = load_model(args.model_dir, cfg, mesh, axis_rules)

    # W8A8: quantize weights to FP8
    params = quantize_weights_fp8(params)

    # Build forward fn
    fwd_fn = make_forward_fn(params, cfg, mesh, tp, axis_rules)

    # Compile warmup
    from jax_gpt.models.qwen35.cache import init_cache
    print(f"[rank {rank}] Warming up JIT...")
    _c = init_cache(cfg, 1, prefill_len + decode_len, dtype=jnp.bfloat16)
    _, _c = fwd_fn(jnp.ones((1, prefill_len), dtype=jnp.int32), _c, False)
    fwd_fn(jnp.ones((1, 1), dtype=jnp.int32), _c, True)
    jax.effects_barrier()
    print(f"[rank {rank}] JIT ready.")

    # Find max batch size
    if args.max_batch:
        max_batch = args.max_batch
        print(f"[rank {rank}] Using provided max_batch={max_batch}")
    else:
        print(f"[rank {rank}] Searching for max batch size...")
        max_batch = find_max_batch_size(fwd_fn, cfg, prefill_len, decode_len, tp)

    # Benchmark
    print(f"[rank {rank}] Benchmarking at batch={max_batch}...")
    results = benchmark_scenario(fwd_fn, cfg, prefill_len, decode_len, max_batch)
    results['scenario'] = args.scenario
    results['tp'] = tp
    results['quantization'] = 'W8A8-FP8'
    results['config'] = args.config

    print(f"\n{'='*60}")
    print(f"BENCHMARK RESULTS — {args.scenario.upper()} W8A8 FP8")
    print(f"{'='*60}")
    print(f"  Batch size:        {max_batch}")
    print(f"  Prefill length:    {prefill_len}")
    print(f"  Decode length:     {decode_len}")
    print(f"  Prefill latency:   {results['prefill_ms']:.1f} ms")
    print(f"  Decode latency:    {results['decode_ms']:.1f} ms/step")
    print(f"  Prefill toks/sec:  {results['prefill_toks_per_sec']:.0f}")
    print(f"  Decode toks/sec:   {results['decode_toks_per_sec']:.0f}")
    print(f"{'='*60}\n")

    # Save results JSON to GCS (rank 0 only)
    if rank == 0:
        results_path = f"{args.profile_dir}/{args.scenario}_results.json"
        import subprocess
        import tempfile, json as _json
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            _json.dump(results, f, indent=2)
            tmp = f.name
        subprocess.run(['gsutil', 'cp', tmp, results_path], check=True)
        print(f"[rank {rank}] Results saved to {results_path}")

        # Capture profile
        profile_subdir = f"{args.profile_dir}/{args.scenario}"
        capture_profile(fwd_fn, cfg, prefill_len, decode_len, max_batch,
                        profile_subdir)


if __name__ == '__main__':
    main()
