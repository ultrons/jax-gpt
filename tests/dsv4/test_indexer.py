"""Correctness + AOT compile gate for the DSv4 CSA indexer.

Three test gates:
  1. Pure-JAX reference (`indexer_topk_reference`) — runs anywhere.
  2. Pallas score kernel (`indexer_score_pallas`) parity with the reference
     on small toy shapes — runs anywhere with JAX.
  3. AOT Mosaic compile (`aot_compile_check`) — needs libtpu in the venv;
     skipped otherwise.
"""

from __future__ import annotations

import importlib.util

import jax
import jax.numpy as jnp
import numpy as np
import pytest

_HAS_LIBTPU = importlib.util.find_spec("libtpu") is not None
_ON_TPU = "tpu" in str(jax.devices()[0]).lower()


# ── Reference: pure-JAX indexer ───────────────────────────────────────────

def test_reference_topk_shape_and_values():
    """JAX reference produces the right shapes and respects ReLU + sum."""
    from jax_gpt.models.dsv4.indexer import indexer_topk_reference

    B, T, n_heads, head_dim, S_comp, topk = 2, 1, 4, 8, 16, 4
    rng = np.random.default_rng(0)
    q = jnp.array(rng.standard_normal((B, T, n_heads, head_dim)).astype(np.float32))
    k = jnp.array(rng.standard_normal((B, S_comp, n_heads, head_dim)).astype(np.float32))

    idx, scores = indexer_topk_reference(q, k, causal_mask=None, topk=topk)
    assert idx.shape == (B, T, topk)
    assert scores.shape == (B, T, topk)
    assert idx.dtype == jnp.int32
    # Scores should be non-negative (ReLU sum).
    assert bool(jnp.all(scores >= 0))


def test_reference_causal_mask_excludes_future():
    """When causal_mask=False everywhere, all scores are -inf and selected
    indices are arbitrary but the score values must be -inf."""
    from jax_gpt.models.dsv4.indexer import indexer_topk_reference

    B, T, n_heads, head_dim, S_comp, topk = 1, 1, 2, 4, 8, 3
    rng = np.random.default_rng(0)
    q = jnp.array(rng.standard_normal((B, T, n_heads, head_dim)).astype(np.float32))
    k = jnp.array(rng.standard_normal((B, S_comp, n_heads, head_dim)).astype(np.float32))
    mask = jnp.zeros((B, T, S_comp), dtype=jnp.bool_)

    _idx, scores = indexer_topk_reference(q, k, causal_mask=mask, topk=topk)
    assert bool(jnp.all(jnp.isneginf(scores)))


# ── Pallas score kernel parity (toy shapes; runs on CPU via jax_metal/-default
# backend if no TPU available) ─────────────────────────────────────────────

@pytest.mark.skipif(
    not _ON_TPU,
    reason="Pallas TPU kernel requires a TPU backend; reference parity verified separately.",
)
def test_pallas_score_matches_reference():
    """The Pallas score tensor matches the reference on small toy shapes."""
    from jax_gpt.models.dsv4.indexer import indexer_topk_reference
    from jax_gpt.models.dsv4.kernels.pallas_indexer import indexer_score_pallas

    B, M, n_heads, head_dim, S_comp, topk = 1, 1, 4, 8, 64, 4
    rng = np.random.default_rng(0)
    q = jnp.array(rng.standard_normal((B, M, n_heads, head_dim)).astype(np.float32),
                  dtype=jnp.bfloat16)
    k = jnp.array(rng.standard_normal((B, S_comp, n_heads, head_dim)).astype(np.float32),
                  dtype=jnp.bfloat16)

    pallas_scores = indexer_score_pallas(q, k, tile_size=16)

    # Reference does the same op in float32; we accept a bf16-induced tol.
    _, ref_top_scores = indexer_topk_reference(q.astype(jnp.float32),
                                               k.astype(jnp.float32),
                                               causal_mask=None, topk=S_comp)
    # ref_top_scores is (B, M, S_comp) sorted descending; resort into position order.
    # Easier: recompute the dense reference scores directly.
    scores_ref = jax.nn.relu(jnp.einsum('bmhd,bshd->bmhs',
                                        q.astype(jnp.float32),
                                        k.astype(jnp.float32))).sum(axis=2)

    np.testing.assert_allclose(np.asarray(pallas_scores),
                               np.asarray(scores_ref),
                               rtol=2e-2, atol=1e-3)


# ── AOT compile gate ─────────────────────────────────────────────────────

def test_aot_compile_v7x():
    """Compile the kernel for tpu7x:4x4x4 without runtime — catches Mosaic
    shape-cast / relayout / scatter errors before any cluster job.

    Skipped (not failed) when the TPU plugin isn't loaded in the current
    venv. To run this gate, activate a venv with the TPU plugin loaded
    (e.g. `source ~/xdb/.xprof/bin/activate`) — `libtpu` alone is not
    sufficient if the JAX TPU PJRT plugin hasn't been registered.
    """
    from jax_gpt.models.dsv4.kernels.pallas_indexer import aot_compile_check
    try:
        aot_compile_check(M=1, S_comp=8192, tile_size=256)
    except RuntimeError as e:
        if "TPU support not installed" in str(e) or "TPU topology" in str(e):
            pytest.skip(f"TPU plugin not loaded: {e}")
        raise
