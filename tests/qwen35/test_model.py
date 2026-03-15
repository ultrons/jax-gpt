"""Tests for the full Qwen3.5 model."""

import jax
import jax.numpy as jnp
import pytest

from jax_gpt.models.qwen35.cache import init_cache
from jax_gpt.models.qwen35.config import Qwen35Config
from jax_gpt.models.qwen35.model import forward, generate, init_params


def test_forward_no_cache_shape():
    """Forward without cache should produce correct logits shape."""
    cfg = Qwen35Config.mini()
    params = init_params(cfg, jax.random.key(0))
    B, T = 1, 8

    tokens = jax.random.randint(jax.random.key(1), (B, T), 0, cfg.vocab_size)
    logits, cache_out = forward(params, tokens, cfg)

    assert logits.shape == (B, T, cfg.vocab_size)
    assert cache_out is None


def test_forward_with_cache_shape():
    """Forward with cache should produce correct logits and updated cache."""
    cfg = Qwen35Config.mini()
    params = init_params(cfg, jax.random.key(2))
    B, T = 1, 4
    max_len = 32

    cache = init_cache(cfg, B, max_len)
    tokens = jax.random.randint(jax.random.key(3), (B, T), 0, cfg.vocab_size)

    logits, new_cache = forward(params, tokens, cfg, cache=cache, is_decode=False)

    assert logits.shape == (B, T, cfg.vocab_size)
    assert new_cache is not None
    assert int(new_cache.pos) == T


def test_forward_decode_step():
    """Single decode step should produce (B, 1, vocab) logits."""
    cfg = Qwen35Config.mini()
    params = init_params(cfg, jax.random.key(4))
    B = 1
    max_len = 32

    # Prefill with some tokens
    cache = init_cache(cfg, B, max_len)
    prompt = jax.random.randint(jax.random.key(5), (B, 4), 0, cfg.vocab_size)
    _, cache = forward(params, prompt, cfg, cache=cache, is_decode=False)

    # Decode one token
    token = jax.random.randint(jax.random.key(6), (B, 1), 0, cfg.vocab_size)
    logits, new_cache = forward(params, token, cfg, cache=cache, is_decode=True)

    assert logits.shape == (B, 1, cfg.vocab_size)
    assert int(new_cache.pos) == 5  # 4 prefill + 1 decode


def test_forward_no_nan():
    """Forward should not produce NaN."""
    cfg = Qwen35Config.mini()
    params = init_params(cfg, jax.random.key(7))
    B, T = 1, 8

    tokens = jax.random.randint(jax.random.key(8), (B, T), 0, cfg.vocab_size)
    logits, _ = forward(params, tokens, cfg)

    assert not jnp.any(jnp.isnan(logits)), "Forward produced NaN logits"


def test_generate_shape():
    """Generate should produce correct number of tokens."""
    cfg = Qwen35Config.mini()
    params = init_params(cfg, jax.random.key(9))
    B = 1
    prompt_len = 4
    new_tokens = 3

    prompt = jax.random.randint(jax.random.key(10), (B, prompt_len), 0, cfg.vocab_size)
    generated = generate(
        params, prompt, cfg,
        max_new_tokens=new_tokens,
        max_seq_len=prompt_len + new_tokens,
        temperature=1.0,
        key=jax.random.key(11),
    )

    assert generated.shape == (B, new_tokens)
    # All generated tokens should be valid vocab indices
    assert jnp.all(generated >= 0)
    assert jnp.all(generated < cfg.vocab_size)
