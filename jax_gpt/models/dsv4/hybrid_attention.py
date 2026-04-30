"""Hybrid CSA / HCA / SWA dispatcher for a single DSv4 layer.

Every layer runs three branches in parallel and sums the outputs:

  o = o_swa + o_compressed + (mtp branch optional)

  o_swa         := MLA over the last `sliding_window` raw tokens (always).
  o_compressed  := one of:
      'hca' (compress_ratio=128): MLA over the full 128:1-compressed KV pool.
      'csa' (compress_ratio=4):   MLA over indexer-selected top-1024 positions
                                  in the 4:1-compressed KV pool.
      'mtp' (compress_ratio=0):   uncompressed full attention (only the final
                                  MTP layer; trivial seq length there).

Three KV pools are read from `cache`:
  cache['swa']  : ring buffer length sliding_window
  cache['c4']   : positions = ceil(seqlen / 4)
  cache['c128'] : positions = ceil(seqlen / 128)

Compression-state ring buffers (the in-flight per-token state of the 4:1 and
128:1 pooling) live alongside but are not read at attend time — they only
participate when *appending* new tokens to the compressed pools. See
paged_cache.py for the lifetime / "ShadowRadix" details.
"""

from __future__ import annotations

import jax

from jax_gpt.models.dsv4.config import DSv4Config


def hybrid_attention(
    x: jax.Array,
    params: dict,
    cfg: DSv4Config,
    layer_idx: int,
    cache: dict,
    rope_freqs: jax.Array,
    compressed_rope_freqs: jax.Array,
) -> tuple[jax.Array, dict]:
    """Run SWA + (CSA | HCA | MTP) for one layer.

    Returns (output, updated_cache). STUB.
    """
    kind = cfg.layer_kind(layer_idx)
    if kind == 'csa':
        # Path: indexer.select_topk -> mla over selected -> add SWA.
        raise NotImplementedError('CSA layer dispatch — see indexer.py + mla.py.')
    if kind == 'hca':
        # Path: mla over c128 pool -> add SWA.
        raise NotImplementedError('HCA layer dispatch — see mla.py.')
    if kind == 'mtp':
        raise NotImplementedError('MTP layer — uncompressed MLA.')
    raise ValueError(f'Unknown layer kind {kind!r}')
