"""Multi-head Latent Attention (MLA) for DSv4 — q-LoRA + kv-LoRA + partial RoPE.

DSv4 attention head geometry (per V4-Pro config.json):
  q_lora_rank      = 1536    # LoRA rank for query
  o_lora_rank      = 1024    # LoRA rank for output
  head_dim         = 512     # latent KV head dim
  qk_rope_head_dim = 64      # head dims that get RoPE
  qk_nope_head_dim = 448     # = head_dim - qk_rope_head_dim
  num_attention_heads = 128
  num_key_value_heads = 1   (single shared latent KV)
  o_groups = 16             (grouped output projection)

Forward pass (decoupled-RoPE MLA, sketch):
  q_latent  = x @ W_dq                       # (B,T,q_lora_rank)
  q_latent  = rmsnorm(q_latent)
  q_full    = q_latent @ W_uq                # (B,T, n_h*(qk_nope+qk_rope))
  q_nope, q_rope = split(q_full)
  q_rope    = apply_rope(q_rope, ...)

  c_kv      = x @ W_dkv                      # (B,T, head_dim - qk_rope) latent KV (NOPE part)
  c_kv      = rmsnorm(c_kv)
  k_rope    = x @ W_kr                       # (B,T, qk_rope) shared RoPE-K
  k_rope    = apply_rope(k_rope, ...)

  Stored cache: (c_kv, k_rope) — small.
  At attend time, k_full[h] = [c_kv @ W_uk[h] ; broadcast(k_rope)]
                  v_full[h] = c_kv @ W_uv[h]

  o = softmax(q_full · k_full / sqrt(head_dim)) · v_full
  o = (o.reshape(B,T,o_groups,o_lora_rank/o_groups) @ W_uo)  # grouped output
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def mla_attention(
    x: jax.Array,
    params: dict,
    *,
    n_heads: int,
    head_dim: int,
    qk_rope_head_dim: int,
    q_lora_rank: int,
    o_lora_rank: int,
    o_groups: int,
    rope_freqs: jax.Array,
    cache: dict | None = None,
) -> tuple[jax.Array, dict | None]:
    """MLA forward — STUB. See module docstring for the formulation.

    params keys:
      W_dq, W_uq, W_dkv, W_uk, W_uv, W_kr, W_uo, q_norm, kv_norm
    cache keys (when present):
      c_kv: (B, max_len, head_dim - qk_rope_head_dim)
      k_rope: (B, max_len, qk_rope_head_dim)
      pos: scalar int32
    """
    raise NotImplementedError('MLA forward — to be implemented; see docstring.')
