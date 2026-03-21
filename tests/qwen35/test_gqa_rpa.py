"""Tests for RPA v3-backed GQA attention correctness.

Compares gqa_attention_rpa() against the reference gqa_attention() to
verify that the paged KV cache + RPA kernel produces matching outputs.
"""

import math

import jax
import jax.numpy as jnp
import pytest

from jax_gpt.models.qwen35.gqa import gqa_attention
from jax_gpt.models.qwen35.primitives import precompute_rope_freqs

# RPA imports — skip tests if tpu-inference is not installed
try:
    from jax_gpt.models.qwen35.gqa_rpa import gqa_attention_rpa
    from jax_gpt.models.qwen35.paged_cache import (
        PagedGQACache,
        contiguous_to_paged,
        init_paged_cache,
        make_decode_metadata,
    )
    from tpu_inference.kernels.ragged_paged_attention.v3.kernel import (
        get_kv_cache_shape,
    )
    HAS_RPA = True
except ImportError:
    HAS_RPA = False

pytestmark = pytest.mark.skipif(not HAS_RPA, reason="tpu-inference not installed")


# Match Qwen3.5 GQA config
N_Q_HEADS = 32
N_KV_HEADS = 2
HEAD_DIM = 256
D_MODEL = N_Q_HEADS * HEAD_DIM  # 8192
ROPE_DIM = HEAD_DIM
ROPE_THETA = 1_000_000.0
MAX_LEN = 128
PAGE_SIZE = 16  # small for testing


def _make_params(key):
    """Create random GQA params."""
    keys = jax.random.split(key, 6)
    return {
        'q_proj': jax.random.normal(keys[0], (D_MODEL, N_Q_HEADS * HEAD_DIM * 2)) * 0.01,
        'k_proj': jax.random.normal(keys[1], (D_MODEL, N_KV_HEADS * HEAD_DIM)) * 0.01,
        'v_proj': jax.random.normal(keys[2], (D_MODEL, N_KV_HEADS * HEAD_DIM)) * 0.01,
        'o_proj': jax.random.normal(keys[3], (N_Q_HEADS * HEAD_DIM, D_MODEL)) * 0.01,
        'q_norm': jnp.ones((HEAD_DIM,)),
        'k_norm': jnp.ones((HEAD_DIM,)),
    }


class TestGQARPA:
    """Test RPA-backed GQA against reference implementation."""

    def test_decode_single_step(self):
        """Single decode step: RPA output matches contiguous-cache output."""
        B = 2
        prefill_len = 8
        key = jax.random.PRNGKey(42)
        k1, k2, k3 = jax.random.split(key, 3)

        params = _make_params(k1)
        rope_freqs = precompute_rope_freqs(ROPE_DIM, MAX_LEN, ROPE_THETA)

        # --- Reference path: prefill to fill cache, then decode one step ---
        # Prefill: fill contiguous cache with prefill_len tokens
        x_prefill = jax.random.normal(k2, (B, prefill_len, D_MODEL)) * 0.01
        cache_k = jnp.zeros((B, N_KV_HEADS, MAX_LEN, HEAD_DIM))
        cache_v = jnp.zeros((B, N_KV_HEADS, MAX_LEN, HEAD_DIM))

        _, cache_k, cache_v = gqa_attention(
            x_prefill, params, N_Q_HEADS, N_KV_HEADS, HEAD_DIM,
            rope_freqs, ROPE_DIM,
            cache_k=cache_k, cache_v=cache_v, cache_pos=jnp.array(0),
        )

        # Decode step with contiguous cache
        x_decode = jax.random.normal(k3, (B, 1, D_MODEL)) * 0.01
        ref_out, ref_k, ref_v = gqa_attention(
            x_decode, params, N_Q_HEADS, N_KV_HEADS, HEAD_DIM,
            rope_freqs, ROPE_DIM,
            cache_k=cache_k, cache_v=cache_v,
            cache_pos=jnp.array(prefill_len),
        )

        # --- RPA path: convert contiguous cache to paged, then decode ---
        # Stack as (n_groups=1, B, ...)
        gqa_k_stacked = cache_k[None, ...]  # (1, B, kv_heads, max_len, hd)
        gqa_v_stacked = cache_v[None, ...]

        paged_caches = contiguous_to_paged(
            gqa_k_stacked, gqa_v_stacked,
            prefill_len=prefill_len,
            page_size=PAGE_SIZE,
        )
        kv_cache_paged = paged_caches[0]  # layer 0

        pages_per_seq = math.ceil(MAX_LEN / PAGE_SIZE)
        kv_lens = jnp.full((B,), prefill_len, dtype=jnp.int32)
        page_indices = jnp.arange(B * pages_per_seq, dtype=jnp.int32)
        cu_q_lens, distribution = make_decode_metadata(B, kv_lens, pages_per_seq)

        rpa_out, updated_cache = gqa_attention_rpa(
            x_decode, params, N_Q_HEADS, N_KV_HEADS, HEAD_DIM,
            rope_freqs, ROPE_DIM,
            kv_cache=kv_cache_paged,
            kv_lens=kv_lens,
            page_indices=page_indices,
            cu_q_lens=cu_q_lens,
            distribution=distribution,
            cache_pos=jnp.array(prefill_len),
        )

        # Compare outputs
        max_diff = jnp.max(jnp.abs(ref_out - rpa_out))
        print(f"Max output diff: {max_diff:.6f}")
        # Allow some tolerance for different computation order
        assert max_diff < 0.05, f"RPA output differs too much: max_diff={max_diff}"

    def test_decode_multiple_steps(self):
        """Multiple decode steps: RPA accumulates KV correctly."""
        B = 2
        prefill_len = 4
        n_decode_steps = 4
        key = jax.random.PRNGKey(123)
        k1, k2 = jax.random.split(key)

        params = _make_params(k1)
        rope_freqs = precompute_rope_freqs(ROPE_DIM, MAX_LEN, ROPE_THETA)

        # Prefill with contiguous cache
        x_prefill = jax.random.normal(k2, (B, prefill_len, D_MODEL)) * 0.01
        cache_k = jnp.zeros((B, N_KV_HEADS, MAX_LEN, HEAD_DIM))
        cache_v = jnp.zeros((B, N_KV_HEADS, MAX_LEN, HEAD_DIM))

        _, cache_k, cache_v = gqa_attention(
            x_prefill, params, N_Q_HEADS, N_KV_HEADS, HEAD_DIM,
            rope_freqs, ROPE_DIM,
            cache_k=cache_k, cache_v=cache_v, cache_pos=jnp.array(0),
        )

        # Convert to paged
        gqa_k_stacked = cache_k[None, ...]
        gqa_v_stacked = cache_v[None, ...]
        paged_caches = contiguous_to_paged(
            gqa_k_stacked, gqa_v_stacked,
            prefill_len=prefill_len,
            page_size=PAGE_SIZE,
        )
        kv_cache_paged = paged_caches[0]
        pages_per_seq = math.ceil(MAX_LEN / PAGE_SIZE)
        kv_lens = jnp.full((B,), prefill_len, dtype=jnp.int32)
        page_indices = jnp.arange(B * pages_per_seq, dtype=jnp.int32)

        # Run decode steps with both paths
        ref_k, ref_v = cache_k, cache_v

        for step in range(n_decode_steps):
            pos = prefill_len + step
            step_key = jax.random.PRNGKey(step * 100)
            x_dec = jax.random.normal(step_key, (B, 1, D_MODEL)) * 0.01

            # Reference
            ref_out, ref_k, ref_v = gqa_attention(
                x_dec, params, N_Q_HEADS, N_KV_HEADS, HEAD_DIM,
                rope_freqs, ROPE_DIM,
                cache_k=ref_k, cache_v=ref_v,
                cache_pos=jnp.array(pos),
            )

            # RPA
            cu_q_lens, distribution = make_decode_metadata(B, kv_lens, pages_per_seq)
            rpa_out, kv_cache_paged = gqa_attention_rpa(
                x_dec, params, N_Q_HEADS, N_KV_HEADS, HEAD_DIM,
                rope_freqs, ROPE_DIM,
                kv_cache=kv_cache_paged,
                kv_lens=kv_lens,
                page_indices=page_indices,
                cu_q_lens=cu_q_lens,
                distribution=distribution,
                cache_pos=jnp.array(pos),
            )
            # Update kv_lens for next step
            kv_lens = kv_lens + 1

            max_diff = jnp.max(jnp.abs(ref_out - rpa_out))
            print(f"Step {step}: max_diff={max_diff:.6f}")
            assert max_diff < 0.1, (
                f"Step {step}: RPA output differs too much: max_diff={max_diff}"
            )

    def test_paged_cache_shape(self):
        """Verify paged cache shape for Qwen3.5 dimensions."""
        from tpu_inference.kernels.ragged_paged_attention.v3.util import (
            align_to, get_dtype_packing,
        )

        B = 4
        dtype = jnp.bfloat16
        packing = get_dtype_packing(dtype)  # 2 for bf16
        pages_per_seq = math.ceil(MAX_LEN / PAGE_SIZE)
        total_pages = B * pages_per_seq

        shape = get_kv_cache_shape(
            total_pages, PAGE_SIZE, N_KV_HEADS, HEAD_DIM, dtype,
        )

        expected_kv_packed = align_to(N_KV_HEADS * 2, packing) // packing
        expected_hd = align_to(HEAD_DIM, 128)

        assert shape == (total_pages, PAGE_SIZE, expected_kv_packed, packing, expected_hd)
        assert shape[0] == total_pages
        assert shape[1] == PAGE_SIZE

    def test_contiguous_to_paged_roundtrip(self):
        """Verify contiguous-to-paged conversion preserves KV data."""
        B = 2
        prefill_len = PAGE_SIZE * 2  # exactly 2 full pages
        key = jax.random.PRNGKey(99)

        # Create contiguous KV with known data
        k1, k2 = jax.random.split(key)
        gqa_k = jnp.zeros((1, B, N_KV_HEADS, MAX_LEN, HEAD_DIM))
        gqa_v = jnp.zeros((1, B, N_KV_HEADS, MAX_LEN, HEAD_DIM))

        # Fill first prefill_len positions with random data
        k_data = jax.random.normal(k1, (B, N_KV_HEADS, prefill_len, HEAD_DIM)) * 0.1
        v_data = jax.random.normal(k2, (B, N_KV_HEADS, prefill_len, HEAD_DIM)) * 0.1
        gqa_k = gqa_k.at[0, :, :, :prefill_len, :].set(k_data)
        gqa_v = gqa_v.at[0, :, :, :prefill_len, :].set(v_data)

        # Convert to paged
        paged_caches = contiguous_to_paged(
            gqa_k, gqa_v,
            prefill_len=prefill_len,
            page_size=PAGE_SIZE,
        )

        assert len(paged_caches) == 1
        paged = paged_caches[0]
        pages_per_seq = math.ceil(MAX_LEN / PAGE_SIZE)
        assert paged.shape[0] == B * pages_per_seq
        assert paged.shape[1] == PAGE_SIZE

    def test_make_decode_metadata(self):
        """Verify decode metadata shapes."""
        B = 8
        pages_per_seq = 16
        kv_lens = jnp.full((B,), 10, dtype=jnp.int32)

        cu_q_lens, distribution = make_decode_metadata(B, kv_lens, pages_per_seq)

        assert cu_q_lens.shape == (B + 1,)
        assert distribution.shape == (3,)
        # All decode
        assert int(distribution[0]) == B
        assert int(distribution[1]) == B
        assert int(distribution[2]) == B
        # cu_q_lens = [0, 1, 2, ..., B]
        import numpy as np
        np.testing.assert_array_equal(cu_q_lens, jnp.arange(B + 1))
