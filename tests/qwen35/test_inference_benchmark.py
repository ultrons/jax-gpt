"""
Inference benchmarks for Qwen3.5 mini model: prefill and decode latency.

=== KEY ADVANTAGES OF PURE JAX OVER NNX ===

The GPT-2 benchmark had to use a Python loop for decode because NNX's
nnx.scan for layers conflicts with jax.lax.scan for the decode loop
(trace level conflict). Our pure JAX model has no such limitation:

- Prefill: jax.jit wrapping forward() — standard
- Decode: jax.lax.scan over decode steps — the ENTIRE decode loop
  compiles into a single HLO program. Zero host-device roundtrips.

This is the primary performance motivation for the pure JAX approach.

Usage:
    python -m pytest tests/qwen35/test_inference_benchmark.py -v -s

    # With profiling:
    PROFILE_DIR=/tmp/qwen35_profiles python -m pytest tests/qwen35/test_inference_benchmark.py -v -s
"""

import os
import time
from contextlib import contextmanager

import jax
import jax.numpy as jnp

from jax_gpt.models.qwen35.cache import init_cache
from jax_gpt.models.qwen35.config import Qwen35Config
from jax_gpt.models.qwen35.model import forward, init_params


@contextmanager
def _maybe_profile(name: str):
    profile_dir = os.environ.get("PROFILE_DIR")
    if profile_dir:
        with jax.profiler.trace(os.path.join(profile_dir, name)):
            yield
    else:
        yield


def _make_config():
    """Mini config for benchmarking. Larger than unit test config for
    more realistic timings, but still fits comfortably on CPU."""
    return Qwen35Config(
        d_model=512,
        vocab_size=4096,
        n_layers=8,       # 2 groups of 4
        max_position_embeddings=2048,
        rms_norm_eps=1e-6,
        gqa_n_q_heads=8,
        gqa_n_kv_heads=2,
        gqa_head_dim=64,
        gqa_partial_rotary_factor=0.25,
        gqa_rope_theta=10_000_000.0,
        delta_n_qk_heads=4,
        delta_n_v_heads=8,
        delta_qk_head_dim=64,
        delta_v_head_dim=64,
        delta_conv_kernel=4,
        n_routed_experts=4,
        n_experts_per_token=2,
        moe_intermediate_size=256,
        shared_expert_intermediate_size=256,
    )


# ---------------------------------------------------------------------------
# BENCHMARK 1: Prefill
# ---------------------------------------------------------------------------

def test_prefill_benchmark():
    """
    Benchmark prefill (prompt processing) latency.

    What to look for in the TPU trace:
    - Single dense compute block (matmuls, norms, attention, MoE routing)
    - No idle gaps between ops
    - The full model is one fused HLO program
    """
    cfg = _make_config()
    params = init_params(cfg, jax.random.key(0))
    batch_size = 2
    seq_len = 128

    tokens = jnp.ones((batch_size, seq_len), dtype=jnp.int32)
    cache = init_cache(cfg, batch_size, max_len=seq_len + 64)

    @jax.jit
    def prefill(p, t, c):
        return forward(p, t, cfg, cache=c, is_decode=False)

    # Warm-up: trigger compilation
    logits, new_cache = prefill(params, tokens, cache)
    logits.block_until_ready()

    # Timed runs
    n_runs = 5
    with _maybe_profile("prefill"):
        t0 = time.perf_counter()
        for _ in range(n_runs):
            logits, new_cache = prefill(params, tokens, cache)
        logits.block_until_ready()
        elapsed = time.perf_counter() - t0

    avg_ms = (elapsed / n_runs) * 1000
    tokens_per_sec = (batch_size * seq_len * n_runs) / elapsed

    print(f"\n{'='*60}")
    print(f"PREFILL BENCHMARK (Qwen3.5 mini)")
    print(f"  Config:         {cfg.n_layers}L, {cfg.d_model}D, "
          f"{cfg.delta_n_v_heads}+{cfg.gqa_n_q_heads}H, "
          f"{cfg.n_routed_experts}E")
    print(f"  Batch size:     {batch_size}")
    print(f"  Sequence len:   {seq_len}")
    print(f"  Devices:        {jax.device_count()}x {jax.devices()[0].platform}")
    print(f"  Avg latency:    {avg_ms:.2f} ms")
    print(f"  Throughput:     {tokens_per_sec:,.0f} tokens/sec")
    print(f"{'='*60}\n")

    assert logits.shape == (batch_size, seq_len, cfg.vocab_size)
    assert jnp.all(jnp.isfinite(logits))


# ---------------------------------------------------------------------------
# BENCHMARK 2: Decode with lax.scan
#
# This is the key advantage of pure JAX: the entire decode loop compiles
# into one HLO program. No host-device roundtrips during generation.
# Compare this with the GPT-2 NNX benchmark which required a Python loop.
# ---------------------------------------------------------------------------

def test_decode_lax_scan_benchmark():
    """
    Benchmark autoregressive decode using jax.lax.scan.

    The entire decode loop (prefill + N decode steps) is compiled into
    a single HLO program. This eliminates:
    - Host-device roundtrips per decode step
    - Python loop overhead
    - Re-dispatch overhead

    On TPU, this means the device runs all steps back-to-back with zero
    idle time between them.
    """
    cfg = _make_config()
    params = init_params(cfg, jax.random.key(1))
    batch_size = 2
    prompt_len = 32
    n_decode_steps = 32

    prompt = jnp.ones((batch_size, prompt_len), dtype=jnp.int32)
    max_len = prompt_len + n_decode_steps
    cache = init_cache(cfg, batch_size, max_len=max_len)

    @jax.jit
    def generate_scan(p, prompt_tokens, initial_cache, rng_key):
        """Prefill + lax.scan decode — single HLO program."""
        # Prefill
        logits, cache_after_prefill = forward(
            p, prompt_tokens, cfg, cache=initial_cache, is_decode=False,
        )
        first_token = jnp.argmax(logits[:, -1, :], axis=-1)  # (B,)

        # Decode loop via lax.scan
        def _step(carry, _):
            token, c, key = carry
            token_input = token[:, None]  # (B, 1)
            step_logits, new_c = forward(
                p, token_input, cfg, cache=c, is_decode=True,
            )
            key, subkey = jax.random.split(key)
            next_token = jnp.argmax(step_logits[:, 0, :], axis=-1)  # greedy
            return (next_token, new_c, key), next_token

        init_carry = (first_token, cache_after_prefill, rng_key)
        _, generated = jax.lax.scan(_step, init_carry, None, length=n_decode_steps)

        # generated: (n_decode_steps, B) -> (B, n_decode_steps)
        return jnp.concatenate([first_token[:, None], generated.T], axis=1)

    rng = jax.random.key(42)

    # Warm-up: trigger compilation (this is the expensive part)
    generated = generate_scan(params, prompt, cache, rng)
    generated.block_until_ready()

    # Timed runs
    n_runs = 3
    with _maybe_profile("decode_scan"):
        t0 = time.perf_counter()
        for _ in range(n_runs):
            generated = generate_scan(params, prompt, cache, rng)
        generated.block_until_ready()
        elapsed = time.perf_counter() - t0

    total_tokens = n_decode_steps + 1  # +1 for first token from prefill
    avg_ms = (elapsed / n_runs) * 1000
    per_step_ms = avg_ms / total_tokens
    tokens_per_sec = (batch_size * total_tokens * n_runs) / elapsed

    print(f"\n{'='*60}")
    print(f"DECODE BENCHMARK (lax.scan — single HLO)")
    print(f"  Config:         {cfg.n_layers}L, {cfg.d_model}D, "
          f"{cfg.delta_n_v_heads}+{cfg.gqa_n_q_heads}H, "
          f"{cfg.n_routed_experts}E")
    print(f"  Batch size:     {batch_size}")
    print(f"  Prompt len:     {prompt_len}")
    print(f"  Decode steps:   {n_decode_steps}")
    print(f"  Devices:        {jax.device_count()}x {jax.devices()[0].platform}")
    print(f"  ---")
    print(f"  Total:          {avg_ms:.2f} ms")
    print(f"  Per step:       {per_step_ms:.2f} ms/token")
    print(f"  Throughput:     {tokens_per_sec:.0f} tok/s")
    print(f"{'='*60}\n")

    assert generated.shape == (batch_size, total_tokens)
    assert jnp.all(generated >= 0)
    assert jnp.all(generated < cfg.vocab_size)


# ---------------------------------------------------------------------------
# BENCHMARK 3: Decode with Python loop (for comparison)
#
# This is the NNX-style approach — a Python loop dispatching one jitted
# step at a time. Faster to compile but slower to execute on TPU due to
# host-device roundtrips.
# ---------------------------------------------------------------------------

def test_decode_python_loop_benchmark():
    """
    Benchmark decode with Python loop for comparison with lax.scan.

    On CPU, the Python loop may actually be faster (no scan overhead).
    On TPU, the lax.scan version should be significantly faster due to
    eliminating host-device roundtrips.
    """
    cfg = _make_config()
    params = init_params(cfg, jax.random.key(2))
    batch_size = 2
    prompt_len = 32
    n_decode_steps = 32

    prompt = jnp.ones((batch_size, prompt_len), dtype=jnp.int32)
    max_len = prompt_len + n_decode_steps
    cache = init_cache(cfg, batch_size, max_len=max_len)

    @jax.jit
    def prefill(p, t, c):
        return forward(p, t, cfg, cache=c, is_decode=False)

    @jax.jit
    def decode_step(p, t, c):
        return forward(p, t, cfg, cache=c, is_decode=True)

    # Warm-up
    logits, warm_cache = prefill(params, prompt, cache)
    tok = jnp.argmax(logits[:, -1, :], axis=-1, keepdims=True)
    step_logits, _ = decode_step(params, tok, warm_cache)
    step_logits.block_until_ready()

    # Timed runs
    n_runs = 3
    decode_times = []

    for _ in range(n_runs):
        logits, run_cache = prefill(params, prompt, cache)
        tok = jnp.argmax(logits[:, -1, :], axis=-1, keepdims=True)
        tok.block_until_ready()

        t0 = time.perf_counter()
        with _maybe_profile("decode_loop"):
            for _ in range(n_decode_steps):
                step_logits, run_cache = decode_step(params, tok, run_cache)
                tok = jnp.argmax(step_logits[:, 0, :], axis=-1, keepdims=True)
            tok.block_until_ready()
        decode_times.append(time.perf_counter() - t0)

    avg_ms = (sum(decode_times) / n_runs) * 1000
    per_step_ms = avg_ms / n_decode_steps
    tokens_per_sec = (batch_size * n_decode_steps * n_runs) / sum(decode_times)

    print(f"\n{'='*60}")
    print(f"DECODE BENCHMARK (Python loop — per-step dispatch)")
    print(f"  Config:         {cfg.n_layers}L, {cfg.d_model}D, "
          f"{cfg.delta_n_v_heads}+{cfg.gqa_n_q_heads}H, "
          f"{cfg.n_routed_experts}E")
    print(f"  Batch size:     {batch_size}")
    print(f"  Prompt len:     {prompt_len}")
    print(f"  Decode steps:   {n_decode_steps}")
    print(f"  Devices:        {jax.device_count()}x {jax.devices()[0].platform}")
    print(f"  ---")
    print(f"  Total:          {avg_ms:.2f} ms")
    print(f"  Per step:       {per_step_ms:.2f} ms/token")
    print(f"  Throughput:     {tokens_per_sec:.0f} tok/s")
    print(f"{'='*60}\n")

    assert tok.shape == (batch_size, 1)
