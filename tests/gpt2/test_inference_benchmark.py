"""
Inference benchmarks: prefill and decode latency with clean device traces.

=== PERFORMANCE BEST PRACTICES DEMONSTRATED ===

1. SEPARATE COMPILATION FROM MEASUREMENT
   jax.jit traces and compiles on first call. Always do a warm-up call and
   block_until_ready() before starting the timer. Otherwise you measure
   XLA compilation, not device execution.

2. USE PURE jax.jit WITH NNX FUNCTIONAL PATTERN
   For stateless forward passes (inference), use nnx.split() once to extract
   (graphdef, state), capture graphdef in a closure, and pass only state
   through jax.jit. This avoids NNX overhead on every call and enables
   donate_argnums.

3. block_until_ready() ONLY AT MEASUREMENT BOUNDARIES
   JAX dispatches work asynchronously. Calling block_until_ready() inside the
   loop forces a host-device sync and kills pipelining. Call it exactly twice:
   once after warm-up (to ensure compilation is done), once after the timed
   run (to get true end time).

4. USE jnp.arange(T) + offset INSTEAD OF jnp.arange(start, stop)
   jnp.arange(start, stop) requires concrete values. When start/stop come
   from a lax.scan carry, they're traced and arange will fail. Use
   jnp.arange(T) + offset where T is a static Python int from x.shape.

5. USE jax.lax.dynamic_update_slice FOR KV CACHE UPDATES
   Static slicing (arr.at[:, :, pos:pos+T, :].set(v)) fails when pos is
   traced. dynamic_update_slice handles traced indices natively.

6. USE jax.profiler.trace FOR TPU TRACES
   Wrap the timed section in jax.profiler.trace() to capture a TPU profile.
   Inspect in TensorBoard to verify back-to-back kernel execution with no
   idle gaps. Set PROFILE_DIR env var to enable.

7. ON-DEVICE DECODE LOOPS WITH jax.lax.scan (WHEN POSSIBLE)
   A Python for-loop over decode steps causes a host->device roundtrip per
   step. jax.lax.scan compiles the entire loop into one HLO program — the TPU
   runs all steps back-to-back with zero host interaction.

   HOWEVER: our model uses nnx.scan internally for layer iteration, which
   performs NNX graph mutations (split/merge). These mutations cannot be nested
   inside jax.lax.scan due to trace level conflicts. For production TPU
   inference, you would write a purely functional forward pass (raw matmuls +
   jax.lax.scan for layers) to enable the outer lax.scan decode loop.
   This benchmark uses nnx.jit + Python loop, which is the correct pattern
   when using NNX models.

Usage:
    # CPU (correctness only — timings not meaningful):
    python -m pytest tests/gpt2/test_inference_benchmark.py -v -s

    # TPU with profiling:
    PROFILE_DIR=/tmp/jax_profiles python -m pytest tests/gpt2/test_inference_benchmark.py -v -s

    # Run only decode:
    python -m pytest tests/gpt2/test_inference_benchmark.py -v -s -k decode
"""

import os
import time
from contextlib import contextmanager

import jax
import jax.numpy as jnp
from flax import nnx

from jax_gpt.models.gpt2.config import GPTConfig
from jax_gpt.models.gpt2.model import GPT, KVCache


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_model(cfg: GPTConfig):
    return GPT(cfg, rngs=nnx.Rngs(0))


def _make_cache(cfg: GPTConfig, batch_size: int, dtype=jnp.float32):
    """Pre-allocate KV cache."""
    cache_shape = (cfg.n_layers, batch_size, cfg.n_head, cfg.d_context, cfg.d_head)
    return KVCache(
        key=jnp.zeros(cache_shape, dtype=dtype),
        value=jnp.zeros(cache_shape, dtype=dtype),
        pos=0,
    )


@contextmanager
def _maybe_profile(name: str):
    """Wrap a section in jax.profiler.trace if PROFILE_DIR is set."""
    profile_dir = os.environ.get("PROFILE_DIR")
    if profile_dir:
        trace_dir = os.path.join(profile_dir, name)
        with jax.profiler.trace(trace_dir):
            yield
    else:
        yield


# ---------------------------------------------------------------------------
# JIT-compiled functions
#
# PREFILL uses jax.jit with the NNX functional split pattern (graphdef
# captured in closure, state as pytree arg). This is the fastest option for
# a single forward pass.
#
# DECODE uses nnx.jit with model as a direct arg. This is required because
# our model uses nnx.scan internally (for layer iteration), which needs the
# NNX context that nnx.jit provides.
# ---------------------------------------------------------------------------

def _make_prefill_fn(graphdef):
    """Return a jitted prefill function with graphdef captured in closure.

    Why closure-capture graphdef?
    graphdef describes the model structure (a Python object, not a JAX array).
    By capturing it in a closure, only JAX pytrees (state, tokens, cache) go
    through jit — no static_argnums/hashability needed.
    """

    @jax.jit
    def prefill(state, tokens, cache_keys, cache_values):
        model = nnx.merge(graphdef, state)
        cache_in = KVCache(key=cache_keys, value=cache_values, pos=0)
        logits, cache_out = model(tokens, deterministic=True, cache=cache_in)
        return logits, cache_out.key, cache_out.value, cache_out.pos

    return prefill


@nnx.jit
def _decode_one_step(model, token, cache_keys, cache_values, cache_pos):
    """Single autoregressive decode step.

    Uses nnx.jit so the NNX model (with internal nnx.scan for layers) works
    correctly. On each call, nnx.jit handles split/merge of the NNX module.
    """
    cache_in = KVCache(key=cache_keys, value=cache_values, pos=cache_pos)
    logits, cache_out = model(token, deterministic=True, cache=cache_in)
    next_token = jnp.argmax(logits[:, -1, :], axis=-1, keepdims=True)
    return next_token, cache_out.key, cache_out.value, cache_out.pos


# ---------------------------------------------------------------------------
# BENCHMARK 1: Prefill
# ---------------------------------------------------------------------------

def test_prefill_benchmark():
    """
    Benchmark prefill (prompt processing) latency.

    What to look for in the TPU trace:
    - A single dense block of compute (matmuls, layer norms, attention)
    - No gaps between operations (no host roundtrips)
    - The HLO should show the full model as one fused program
    """
    cfg = GPTConfig(
        d_model=768, d_head=64, d_ff=3072,
        n_head=12, n_kv_head=12, n_layers=12,
        d_context=1024, vocab_size=50257,
    )
    batch_size = 4
    seq_len = 512

    model = _make_model(cfg)
    graphdef, state = nnx.split(model)
    cache = _make_cache(cfg, batch_size)
    tokens = jnp.ones((batch_size, seq_len), dtype=jnp.int32)

    prefill_fn = _make_prefill_fn(graphdef)

    # --- Warm-up: trigger compilation, then block until complete ---
    # Without this, the first call includes XLA compilation time (seconds).
    logits, k_out, v_out, pos_out = prefill_fn(state, tokens, cache.key, cache.value)
    logits.block_until_ready()  # sync point 1: compilation done

    # --- Timed runs ---
    # Dispatch n_runs calls asynchronously, then block once at the end.
    # JAX's async dispatch means these pipeline on the device — the host
    # enqueues work while the device is still executing previous calls.
    n_runs = 5
    with _maybe_profile("prefill"):
        t0 = time.perf_counter()
        for _ in range(n_runs):
            logits, k_out, v_out, pos_out = prefill_fn(
                state, tokens, cache.key, cache.value,
            )
        # sync point 2: all dispatched work is complete
        logits.block_until_ready()
        elapsed = time.perf_counter() - t0

    avg_ms = (elapsed / n_runs) * 1000
    tokens_per_sec = (batch_size * seq_len * n_runs) / elapsed

    print(f"\n{'='*60}")
    print(f"PREFILL BENCHMARK")
    print(f"  Config:         GPT-2 small (12L, 768D, 12H)")
    print(f"  Batch size:     {batch_size}")
    print(f"  Sequence len:   {seq_len}")
    print(f"  Devices:        {jax.device_count()}x {jax.devices()[0].platform}")
    print(f"  Avg latency:    {avg_ms:.2f} ms")
    print(f"  Throughput:     {tokens_per_sec:,.0f} tokens/sec")
    print(f"{'='*60}\n")

    # Correctness checks
    assert logits.shape == (batch_size, seq_len, cfg.vocab_size)
    assert pos_out == seq_len
    assert jnp.all(jnp.isfinite(logits))


# ---------------------------------------------------------------------------
# BENCHMARK 2: Decode (autoregressive generation)
#
# Each decode step is one jitted call. The host dispatches steps as fast
# as possible (async dispatch), and we block only at the end.
#
# On CPU/GPU this is reasonably efficient — dispatch overhead is small
# relative to compute. On TPU, the dispatch gap becomes significant.
# For maximum TPU decode throughput, you would:
#   1. Write a purely functional forward pass (no NNX modules)
#   2. Use jax.lax.scan for the layer loop (replacing nnx.scan)
#   3. Wrap the decode loop in jax.lax.scan
# This eliminates all host-device roundtrips during generation.
# ---------------------------------------------------------------------------

def test_decode_benchmark():
    """
    Benchmark autoregressive decode latency.

    Measures per-step decode time with proper warm-up and async dispatch.
    Each step generates one token using greedy argmax.

    Key timing pattern:
    - Warm up JIT (first call compiles)
    - For each timed run:
      1. Prefill → get fresh KV cache and first token
      2. block_until_ready() — ensure prefill is done (sync boundary)
      3. Dispatch all N decode steps WITHOUT blocking between them
         (JAX async dispatch — host enqueues, device executes)
      4. block_until_ready() on final output only (sync boundary)
    """
    cfg = GPTConfig(
        d_model=768, d_head=64, d_ff=3072,
        n_head=12, n_kv_head=12, n_layers=12,
        d_context=1024, vocab_size=50257,
    )
    batch_size = 4
    prompt_len = 128
    n_decode_steps = 64

    model = _make_model(cfg)
    graphdef, state = nnx.split(model)
    cache = _make_cache(cfg, batch_size)
    tokens = jnp.ones((batch_size, prompt_len), dtype=jnp.int32)

    prefill_fn = _make_prefill_fn(graphdef)

    # --- Prefill to populate KV cache ---
    logits, k_out, v_out, pos_out = prefill_fn(state, tokens, cache.key, cache.value)
    first_token = jnp.argmax(logits[:, -1, :], axis=-1, keepdims=True)

    # --- Warm-up decode step: compile once ---
    tok, k, v, pos = _decode_one_step(model, first_token, k_out, v_out, pos_out)
    tok.block_until_ready()

    # --- Timed decode runs ---
    n_runs = 3
    decode_times = []
    all_generated = []

    for _ in range(n_runs):
        # Get a fresh cache for this run
        _, k_fresh, v_fresh, pos_fresh = prefill_fn(
            state, tokens, cache.key, cache.value,
        )
        k_fresh.block_until_ready()  # sync: prefill done before timing decode

        generated = []
        t0 = time.perf_counter()

        # Dispatch all decode steps without blocking between them.
        # JAX enqueues each step asynchronously — the device executes them
        # back-to-back while the host races ahead dispatching the next one.
        # No block_until_ready() here — that would force a sync per step.
        with _maybe_profile("decode"):
            tok = first_token
            k, v, pos = k_fresh, v_fresh, pos_fresh
            for _ in range(n_decode_steps):
                tok, k, v, pos = _decode_one_step(model, tok, k, v, pos)
                generated.append(tok)

            # Sync point: all decode steps are done.
            tok.block_until_ready()

        decode_times.append(time.perf_counter() - t0)
        all_generated = generated  # keep last run for correctness check

    # --- Results ---
    avg_ms = (sum(decode_times) / n_runs) * 1000
    per_step_ms = avg_ms / n_decode_steps
    tok_per_sec = (batch_size * n_decode_steps * n_runs) / sum(decode_times)

    print(f"\n{'='*60}")
    print(f"DECODE BENCHMARK")
    print(f"  Config:         GPT-2 small (12L, 768D, 12H)")
    print(f"  Batch size:     {batch_size}")
    print(f"  Prompt len:     {prompt_len}")
    print(f"  Decode steps:   {n_decode_steps}")
    print(f"  Devices:        {jax.device_count()}x {jax.devices()[0].platform}")
    print(f"  ---")
    print(f"  Total:          {avg_ms:.2f} ms")
    print(f"  Per step:       {per_step_ms:.2f} ms/token")
    print(f"  Throughput:     {tok_per_sec:,.0f} tok/s")
    print(f"{'='*60}\n")

    # Correctness checks
    generated_stack = jnp.stack(all_generated, axis=0)
    assert generated_stack.shape == (n_decode_steps, batch_size, 1)
    assert jnp.all(generated_stack >= 0)
    assert jnp.all(generated_stack < cfg.vocab_size)
