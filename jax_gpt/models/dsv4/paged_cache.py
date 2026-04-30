"""Three-pool paged KV cache for DSv4 (ShadowRadix-lite).

DSv4 has three KV pools with different sizes / lifetimes / dtypes:

  Pool       Per-seq size           Dtype       Lifetime
  ----       ------------           -----       --------
  swa        sliding_window=128     fp8 + bf16  freed when SWA window slides past
  c4         ceil(S / 4)            fp8 + fp4   alive while sequence alive (prefix-shareable)
  c128       ceil(S / 128)          fp8 + bf16  alive while sequence alive (prefix-shareable)

Plus two compression-state ring buffers (per-token state of the 4:1 / 128:1
softmax-gated pooling builder), only accessed when *appending* tokens — not
read at attend time.

ShadowRadix (full SGLang impl) layers these onto a radix tree for prefix
caching. Two-counter lock per node:
  - full_lock_ref: covers the source token + its C4/C128 shadows.
  - swa_lock_ref:  only tracks whether the node is still in someone's SWA window.
This lets SWA slots be tombstoned independently of the C4/C128 shadows, so a
10k-token request keeps only 128 SWA tokens + full compressed KV — and the
compressed KV can be shared across prefix-matching requests.

For a v1 implementation, three flat paged pools with independent free-lists
are enough — defer the radix-tree prefix sharing.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax


@dataclass
class DSv4PagedCache:
    # Three independent paged pools.
    # Each is (n_pages, page_size, n_kv_heads, head_dim) in its own dtype.
    swa_kv: jax.Array              # fp8 main + bf16 RoPE; page_size ~32
    c4_kv: jax.Array               # fp8 main + fp4 indexer; page_size ~64
    c128_kv: jax.Array             # fp8 main + bf16 RoPE; page_size ~128

    # Per-sequence page tables (B, max_pages_per_seq) int32.
    swa_page_table: jax.Array
    c4_page_table: jax.Array
    c128_page_table: jax.Array

    # Per-sequence current write positions (length actually used).
    swa_len: jax.Array             # (B,) int32
    c4_len: jax.Array              # (B,) int32
    c128_len: jax.Array            # (B,) int32

    # Compression-state ring buffers (write-side only).
    c4_compress_state: jax.Array
    c128_compress_state: jax.Array


def append_token(
    cache: DSv4PagedCache,
    new_kv: jax.Array,           # (B, n_kv_heads, head_dim) — raw token KV at current step
    seq_id: jax.Array,           # (B,) int32 — which sequence this token belongs to
) -> DSv4PagedCache:
    """Append one token to all three pools, updating compression state.

    The 4:1 pool advances on every 4th raw token; 128:1 on every 128th.
    SWA pool advances on every token but is a 128-entry ring.
    STUB.
    """
    raise NotImplementedError('Three-pool append — see module docstring.')


def gather_pool(
    cache: DSv4PagedCache,
    pool: str,                  # 'swa' | 'c4' | 'c128'
    seq_id: jax.Array,
    indices: jax.Array | None = None,  # for c4 only: top-1024 indexer output
) -> jax.Array:
    """Gather a contiguous slice (swa/c128) or top-k indexer-selected (c4) KV. STUB."""
    raise NotImplementedError('Pool-specific gather — see module docstring.')
