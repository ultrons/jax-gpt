"""Pallas kernel for the DSv4 CSA indexer score computation.

The indexer scores every (query, compressed-KV-position) pair with a
ReLU-multi-head dot product, then a separate top-k stage selects the
top-1024 positions per query. This file implements the SCORE stage as a
tiled Pallas kernel; the top-k step uses jax.lax.top_k as a fallback
(true Lightning-TopK with cluster-of-8 radix-select is a follow-up).

Cost model (decode B=1, M=1, S_comp=250k at 1M ctx, FP8 K, 64×128 indexer):
  K read = 250k × 64 × 128 × 1 byte = 2 GB per layer per token
  FLOPs  = 2 × 64 × 128 × 250k = 4 GFLOPs per layer per token
  AI     = 4 GF / 2 GB = 2 FLOPs/byte → HBM-bound on v7x
  Time   = 2 GB / 3.65 TB/s ≈ 0.55 ms / layer / token (HBM ceiling)
  Across 30 CSA layers: ~16 ms / decode token from the indexer alone.

Mosaic constraints honored (per global CLAUDE.md):
  - No `[:, None]` reshapes (use full-shape buffers).
  - VMEM scratch has no trailing size-1 dimensions.
  - No scatter on JAX arrays — `.at[]` is on pl.Ref only.
  - No bool reshape — score masks computed as int32 if needed.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def _indexer_score_kernel(
    q_ref,          # VMEM read: (M, n_heads, head_dim)  bf16
    k_ref,          # VMEM read: (T_tile, n_heads, head_dim) bf16
    out_ref,        # VMEM write: (M, T_tile) f32
):
    """Inner per-tile score kernel.

    For each (m, t) in the tile, computes:
        score[m, t] = sum_h max(0, q[m, h, :] · k[t, h, :])

    Implementation: one big batched dot product (mhd, thd → mht), then
    ReLU and sum over the head axis. Stays in f32 on the accumulator path.
    """
    # Load full tiles into local registers.
    q = q_ref[...]          # (M, n_heads, head_dim) bf16
    k = k_ref[...]          # (T, n_heads, head_dim) bf16

    # Per-head dot product. dot_general contracts head_dim, batches n_heads.
    # q: (M, n_heads, head_dim) → swap to (n_heads, M, head_dim) for batched
    # k: (T, n_heads, head_dim) → swap to (n_heads, T, head_dim)
    # batched matmul: (n_heads, M, head_dim) @ (n_heads, head_dim, T) → (n_heads, M, T)
    q_h = jnp.transpose(q, (1, 0, 2))               # (n_heads, M, head_dim)
    k_h = jnp.transpose(k, (1, 0, 2))               # (n_heads, T, head_dim)
    scores = jax.lax.dot_general(
        q_h, k_h,
        dimension_numbers=(((2,), (2,)), ((0,), (0,))),  # contract head_dim, batch n_heads
        preferred_element_type=jnp.float32,
    )  # (n_heads, M, T) f32

    # ReLU per-(head, m, t), then sum across heads.
    scores = jax.nn.relu(scores)
    scores = jnp.sum(scores, axis=0)                # (M, T) f32

    out_ref[...] = scores


def indexer_score_pallas(
    q: jax.Array,            # (B, M, n_heads, head_dim) bf16
    k_compressed: jax.Array, # (B, S_comp, n_heads, head_dim) bf16
    *,
    tile_size: int = 256,
) -> jax.Array:
    """Score every query against every compressed-KV position via Pallas.

    Returns: (B, M, S_comp) f32 score tensor. Top-k is applied separately.

    Tile strategy: the kernel processes one (M × tile_size) score block per
    invocation, looping over the compressed-KV dimension via a Pallas grid.
    Tile size 256 is comfortable on v7x VMEM (~64 MB/core) for reasonable M
    and matches the v7x MXU width of 256.
    """
    B, M, n_heads, head_dim = q.shape
    Bk, S_comp, n_heads_k, head_dim_k = k_compressed.shape
    assert (B, n_heads, head_dim) == (Bk, n_heads_k, head_dim_k)
    assert S_comp % tile_size == 0, (
        f"S_comp={S_comp} must be a multiple of tile_size={tile_size}; "
        "pad the compressed-KV pool to the nearest multiple before calling."
    )

    n_tiles = S_comp // tile_size

    def _per_batch(q_b, k_b):
        # q_b: (M, n_heads, head_dim)
        # k_b: (S_comp, n_heads, head_dim)
        scores = pl.pallas_call(
            _indexer_score_kernel,
            out_shape=jax.ShapeDtypeStruct((M, S_comp), jnp.float32),
            grid=(n_tiles,),
            in_specs=[
                pl.BlockSpec((M, n_heads, head_dim), lambda i: (0, 0, 0)),
                pl.BlockSpec((tile_size, n_heads, head_dim), lambda i: (i, 0, 0)),
            ],
            out_specs=pl.BlockSpec((M, tile_size), lambda i: (0, i)),
            compiler_params=pltpu.CompilerParams(
                dimension_semantics=("parallel",),
            ),
            name="indexer_score",
        )(q_b, k_b)
        return scores

    return jax.vmap(_per_batch)(q, k_compressed)


def indexer_topk_pallas(
    q: jax.Array,            # (B, M, n_heads, head_dim) bf16
    k_compressed: jax.Array, # (B, S_comp, n_heads, head_dim) bf16
    causal_mask: jax.Array | None = None,  # (B, M, S_comp) bool, or None
    *,
    topk: int,
    tile_size: int = 256,
) -> tuple[jax.Array, jax.Array]:
    """Full indexer entrypoint: scores + top-k. Drop-in for indexer_topk_reference."""
    scores = indexer_score_pallas(q, k_compressed, tile_size=tile_size)
    if causal_mask is not None:
        scores = jnp.where(causal_mask, scores, jnp.full_like(scores, -jnp.inf))
    top_scores, top_indices = jax.lax.top_k(scores, topk)
    return top_indices.astype(jnp.int32), top_scores


# ── AOT compile check (no hardware required) ─────────────────────────────────

def aot_compile_check(
    M: int = 1,
    S_comp: int = 8192,
    n_heads: int = 64,
    head_dim: int = 128,
    tile_size: int = 256,
    topology: str = "tpu7x:4x4x4",
) -> None:
    """Compile the indexer Pallas kernel for v7x without any TPU runtime.

    Exercises the Mosaic compile path on the abstract topology so shape-cast
    / relayout / scatter-on-array errors surface in <2 minutes locally
    (vs the 10–30 min round trip of a cluster job).

    Per global CLAUDE.md: this is gate 1 of 3 (AOT compile, EP=1 exec,
    EP=N exec). Gates 2 and 3 require real hardware.

    Requires `libtpu` in the active venv (e.g. ~/xdb/.xprof/bin/activate).
    """
    from jax.experimental import topologies
    import numpy as np

    topo = topologies.get_topology_desc(topology, platform="tpu")
    B = 1  # single batch for AOT check; vmap composes over B at runtime

    q_abs = jax.ShapeDtypeStruct((B, M, n_heads, head_dim), jnp.bfloat16)
    k_abs = jax.ShapeDtypeStruct((B, S_comp, n_heads, head_dim), jnp.bfloat16)

    fn = functools.partial(indexer_score_pallas, tile_size=tile_size)

    with jax.default_device(topo.devices[0]):
        lowered = jax.jit(fn).lower(q_abs, k_abs)
        lowered.compile()


if __name__ == "__main__":
    # Smoke check from the command line:
    #   PYTHONPATH=. python -m jax_gpt.models.dsv4.kernels.pallas_indexer
    aot_compile_check()
    print("indexer Pallas AOT compile: OK")
