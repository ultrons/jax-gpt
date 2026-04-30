#!/usr/bin/env python3
"""Layer-by-layer comparison between HF-convention reference (numpy) and our JAX model.

Runs on a single device with no sharding (mesh=None). Uses layer 3 (first MoE layer)
and a fixed 5-token input. Prints max-abs-diff at each sub-step.

Usage (inside container, single-device):
    python compare_layers.py --model_dir /mnt/model/DeepSeek-V3

The numpy reference uses HF weight conventions (weights are (out_features, in_features))
and does not transpose. The JAX path uses our load_weights.py (which transposes).
A mismatch in any step identifies the bug.
"""

from __future__ import annotations

import argparse
import math
import struct
import json
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp

from model import full_671b_config, ShardConfig, forward, mla_attention, rms_norm
from load_weights import ShardCache, _to_jax, _to_jax_fp8


# ============================================================================
# HF-convention numpy reference helpers
# ============================================================================

def _load_raw_f32(cache: ShardCache, name: str) -> np.ndarray:
    """Load weight as float32 (HF convention: no transpose). BF16 widened."""
    return cache.load_tensor(name).astype(np.float32)


def _fp8_dequant_block(w_u8: np.ndarray, s: np.ndarray | None) -> np.ndarray:
    """Block-wise FP8 dequant: w_u8 (rows, cols) uint8, s (br, bc) float32 → float32."""
    if s is None:
        return w_u8.astype(np.float32)
    sign = ((w_u8 >> 7) & 1).astype(np.float32)
    exp_bits = ((w_u8 >> 3) & 0xF).astype(np.int32)
    mant_bits = (w_u8 & 0x7).astype(np.float32)
    is_subnormal = (exp_bits == 0)
    mantissa = np.where(is_subnormal, mant_bits / 8.0, 1.0 + mant_bits / 8.0)
    exponent = np.where(is_subnormal, -6, exp_bits - 7).astype(np.float32)
    value = np.where(w_u8 == 0, 0.0, mantissa * (2.0 ** exponent))
    value = np.where(sign > 0, -value, value)
    rows, cols = value.shape
    scale_tiled = np.repeat(np.repeat(s, 128, axis=0), 128, axis=1)
    return (value * scale_tiled[:rows, :cols]).astype(np.float32)


def _load_raw_fp8_dequant(cache: ShardCache, name: str) -> np.ndarray:
    """Load FP8 weight, dequant on CPU, return float32 (HF convention: no transpose)."""
    w_u8, s = cache.load_fp8_raw(name)
    return _fp8_dequant_block(w_u8, s)


def np_rms_norm(x: np.ndarray, weight: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """RMSNorm in numpy."""
    variance = np.mean(x * x, axis=-1, keepdims=True)
    x = x * (1.0 / np.sqrt(variance + eps))
    return x * weight


def np_rope(x: np.ndarray, positions: np.ndarray, d_rope: int) -> np.ndarray:
    """Apply RoPE to x[..., d_rope]. x: (S, H, d_rope), positions: (S,)."""
    S, H, d = x.shape
    half = d // 2
    freqs = 1.0 / (10000.0 ** (np.arange(0, half, dtype=np.float32) / half))
    angles = positions[:, None] * freqs[None, :]  # (S, half)
    cos_val = np.cos(angles)[:, None, :]  # (S, 1, half)
    sin_val = np.sin(angles)[:, None, :]
    x1, x2 = x[..., :half], x[..., half:]
    out = np.concatenate([x1 * cos_val - x2 * sin_val,
                          x1 * sin_val + x2 * cos_val], axis=-1)
    return out


def np_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def diff(name: str, ref: np.ndarray, got: np.ndarray) -> float:
    ref_f = ref.astype(np.float32).ravel()
    got_f = got.astype(np.float32).ravel()
    mad = float(np.max(np.abs(ref_f - got_f)))
    rel = mad / (float(np.max(np.abs(ref_f))) + 1e-8)
    print(f"  {name:40s}  max_abs={mad:.4e}  rel={rel:.4e}  "
          f"ref_mean={float(np.mean(np.abs(ref_f))):.4e}  got_mean={float(np.mean(np.abs(got_f))):.4e}")
    return mad


# ============================================================================
# HF-convention numpy MLA forward (layer p_idx)
# ============================================================================

def hf_ref_mla(cache: ShardCache, p: str, x_np: np.ndarray,
               H: int, R_q: int, R_kv: int,
               d_nope: int, d_rope: int, d_v: int, norm_eps: float):
    """Run MLA attention using HF weight conventions (no transpose).

    x_np: (S, D) float32
    Returns attn_out: (S, D) float32, plus all intermediates.
    """
    S, D = x_np.shape
    qk_dim = d_nope + d_rope

    # Weights in HF convention: (out, in)
    pre_norm_w = _load_raw_f32(cache, f'{p}.input_layernorm.weight')  # (D,)
    x_normed = np_rms_norm(x_np, pre_norm_w, norm_eps)

    wq_a = _load_raw_fp8_dequant(cache, f'{p}.self_attn.q_a_proj.weight')   # (R_q, D)
    q_norm_w = _load_raw_f32(cache, f'{p}.self_attn.q_a_layernorm.weight')   # (R_q,)
    wq_b = _load_raw_fp8_dequant(cache, f'{p}.self_attn.q_b_proj.weight')    # (H*qk, R_q)
    wkv_a = _load_raw_fp8_dequant(cache, f'{p}.self_attn.kv_a_proj_with_mqa.weight')  # (R_kv+d_rope, D)
    kv_norm_w = _load_raw_f32(cache, f'{p}.self_attn.kv_a_layernorm.weight') # (R_kv,)
    wkv_b = _load_raw_fp8_dequant(cache, f'{p}.self_attn.kv_b_proj.weight')  # (H*(d_nope+d_v), R_kv)
    w_out = _load_raw_fp8_dequant(cache, f'{p}.self_attn.o_proj.weight')     # (D, H*d_v)

    # Q path: (S,D) @ (D,R_q) = (S,R_q)
    q_low = x_normed @ wq_a.T                               # (S, R_q)
    q_low_normed = np_rms_norm(q_low, q_norm_w, norm_eps)   # (S, R_q)
    q_full = q_low_normed @ wq_b.T                           # (S, H*qk)
    q = q_full.reshape(S, H, qk_dim)                         # (S, H, qk)
    q_nope = q[:, :, :d_nope]                                # (S, H, d_nope)
    q_rope_raw = q[:, :, d_nope:]                             # (S, H, d_rope)

    # KV path: (S,D) @ (D,R_kv+d_rope) = (S, R_kv+d_rope)
    kv_low = x_normed @ wkv_a.T                             # (S, R_kv+d_rope)
    kv_main = kv_low[:, :R_kv]                               # (S, R_kv)
    k_rope_raw_shared = kv_low[:, R_kv:]                     # (S, d_rope)  — 1 head (MQA)
    kv_main_normed = np_rms_norm(kv_main, kv_norm_w, norm_eps)
    kv_full = kv_main_normed @ wkv_b.T                       # (S, H*(d_nope+d_v))
    kv = kv_full.reshape(S, H, d_nope + d_v)
    k_nope = kv[:, :, :d_nope]                               # (S, H, d_nope)
    v = kv[:, :, d_nope:]                                    # (S, H, d_v)

    # RoPE
    positions = np.arange(S, dtype=np.float32)
    q_rope = np_rope(q_rope_raw, positions, d_rope)          # (S, H, d_rope)
    # k_rope_raw_shared is (S, d_rope) — broadcast to (S, H, d_rope) then rope
    k_rope_raw_h = np.broadcast_to(
        k_rope_raw_shared[:, None, :], (S, H, d_rope)).copy()
    k_rope = np_rope(k_rope_raw_h, positions, d_rope)         # (S, H, d_rope)

    # Full Q and K
    query = np.concatenate([q_nope, q_rope], axis=-1)        # (S, H, qk)
    key   = np.concatenate([k_nope, k_rope], axis=-1)        # (S, H, qk)
    scale = math.sqrt(qk_dim)

    # Attention: query(S,H,qk) @ key(S,H,qk)^T → (S,H,S) then softmax
    # attn[s,h,t] = sum_d query[s,h,d] * key[t,h,d] / scale
    attn_logits = np.einsum('shd,thd->sht', query, key) / scale  # (S, H, S)
    mask = np.tril(np.ones((S, S), dtype=bool))
    attn_logits = np.where(mask[:, None, :], attn_logits, -1e9)
    attn_weights = np_softmax(attn_logits.astype(np.float64), axis=-1).astype(np.float32)

    # Weighted sum of values
    attn_out_h = np.einsum('sht,thd->shd', attn_weights, v)  # (S, H, d_v)
    attn_flat = attn_out_h.reshape(S, H * d_v)               # (S, H*d_v)
    attn_out = attn_flat @ w_out.T                            # (S, D)

    return {
        "x_normed":      x_normed,
        "q_low":         q_low,
        "q_low_normed":  q_low_normed,
        "q":             q,
        "kv_low":        kv_low,
        "kv_main_normed":kv_main_normed,
        "k_nope":        k_nope,
        "v":             v,
        "q_rope":        q_rope,
        "k_rope":        k_rope,
        "query":         query,
        "key":           key,
        "attn_logits":   attn_logits,
        "attn_weights":  attn_weights,
        "attn_out":      attn_out,
    }


# ============================================================================
# JAX model intermediates (instrumented forward for one layer)
# ============================================================================

def jax_mla_intermediates(layer_p, x_jax, positions_jax, cfg):
    """Run MLA forward saving intermediate activations, return dict of numpy arrays."""
    from jax.sharding import NamedSharding, PartitionSpec as P
    import jax.numpy as jnp

    B, S, D = x_jax.shape
    H, R_q, R_kv = cfg.H, cfg.R_q, cfg.R_kv
    d_nope, d_rope, d_v = cfg.d_nope, cfg.d_rope, cfg.d_v
    qk_dim = d_nope + d_rope
    norm_eps = cfg.norm_eps

    x_normed = rms_norm(x_jax[0], layer_p["pre_attn_norm"], norm_eps)  # (S, D)

    # Q path
    q_low = jnp.einsum("sd,dr->sr", x_normed, layer_p["wq_a"])  # (S, R_q)
    q_low_normed = rms_norm(q_low, layer_p["q_norm_scale"], norm_eps)
    q_full = jnp.einsum("sr,rhd->shd", q_low_normed,
                         layer_p["wq_b"].reshape(R_q, H, qk_dim))  # (S, H, qk)
    q_nope = q_full[:, :, :d_nope]
    q_rope_raw = q_full[:, :, d_nope:]

    # KV path
    kv_low = jnp.einsum("sd,dr->sr", x_normed, layer_p["wkv_a"])  # (S, R_kv+d_rope)
    kv_main = kv_low[:, :R_kv]
    k_rope_raw_shared = kv_low[:, R_kv:]
    kv_main_normed = rms_norm(kv_main, layer_p["kv_norm_scale"], norm_eps)
    kv_full = jnp.einsum("sr,rhd->shd", kv_main_normed,
                          layer_p["wkv_b"].reshape(R_kv, H, d_nope + d_v))
    k_nope = kv_full[:, :, :d_nope]
    v = kv_full[:, :, d_nope:]

    # RoPE
    half = d_rope // 2
    freqs = 1.0 / (10000.0 ** (jnp.arange(0, half, dtype=jnp.float32) / half))
    angles = positions_jax[0, :, None] * freqs[None, :]  # (S, half)
    cos_val = jnp.cos(angles)[:, None, :]
    sin_val = jnp.sin(angles)[:, None, :]
    q1, q2 = q_rope_raw[:, :, :half], q_rope_raw[:, :, half:]
    q_rope = jnp.concatenate([q1 * cos_val - q2 * sin_val,
                               q1 * sin_val + q2 * cos_val], axis=-1)
    k_rope_h = jnp.broadcast_to(k_rope_raw_shared[:, None, :], (S, H, d_rope))
    k1, k2 = k_rope_h[:, :, :half], k_rope_h[:, :, half:]
    k_rope = jnp.concatenate([k1 * cos_val - k2 * sin_val,
                               k1 * sin_val + k2 * cos_val], axis=-1)

    query = jnp.concatenate([q_nope, q_rope], axis=-1)
    key   = jnp.concatenate([k_nope, k_rope], axis=-1)
    scale = math.sqrt(qk_dim)

    attn_logits = jnp.einsum("shd,thd->sht", query, key) / scale
    mask = jnp.tril(jnp.ones((S, S), dtype=jnp.bool_))
    attn_logits_masked = jnp.where(mask[:, None, :], attn_logits, -1e9)
    attn_weights = jax.nn.softmax(attn_logits_masked.astype(jnp.float32), axis=-1)

    attn_out_h = jnp.einsum("sht,thd->shd", attn_weights, v)
    attn_flat = attn_out_h.reshape(S, H * d_v)
    attn_out = jnp.einsum("sd,do->so", attn_flat, layer_p["w_out"])

    def np32(x):
        return np.array(x.astype(jnp.float32))

    return {
        "x_normed":      np32(x_normed),
        "q_low":         np32(q_low),
        "q_low_normed":  np32(q_low_normed),
        "q":             np32(q_full),
        "kv_low":        np32(kv_low),
        "kv_main_normed":np32(kv_main_normed),
        "k_nope":        np32(k_nope),
        "v":             np32(v),
        "q_rope":        np32(q_rope),
        "k_rope":        np32(k_rope),
        "query":         np32(query),
        "key":           np32(key),
        "attn_logits":   np32(attn_logits),
        "attn_weights":  np32(attn_weights),
        "attn_out":      np32(attn_out),
    }


# ============================================================================
# HF numpy MoE reference
# ============================================================================

def hf_ref_moe(cache: ShardCache, p: str, x_np: np.ndarray, cfg, with_bias: bool = True):
    """Run MoE layer using HF weight conventions (numpy).

    x_np: (S, D) float32 — input AFTER post-attention layernorm.
    Returns dict with routing and output intermediates.
    """
    S, D = x_np.shape
    E, K = cfg.E, cfg.K
    D_moe = cfg.D_moe

    # Gate weight: HF (E, D) → transposed to (D, E) for matmul
    gate_w = _load_raw_f32(cache, f'{p}.mlp.gate.weight')         # (E, D)
    bias   = _load_raw_f32(cache, f'{p}.mlp.gate.e_score_correction_bias')  # (E,)

    # Routing
    logits = x_np @ gate_w.T                                       # (S, E)
    scores = 1.0 / (1.0 + np.exp(-logits.astype(np.float64))).astype(np.float32)  # sigmoid

    # Top-k selection: HF uses biased scores for selection
    biased = scores + bias[None, :]                                # (S, E)
    top_k_indices = np.argsort(biased, axis=-1)[:, -K:][:, ::-1]  # (S, K) descending
    top_k_scores  = scores[np.arange(S)[:, None], top_k_indices]   # unbiased weights
    top_k_weights = top_k_scores / top_k_scores.sum(axis=-1, keepdims=True)
    top_k_weights = top_k_weights * cfg.routed_scaling_factor

    # Routing without bias (what our code does)
    top_k_indices_nobias = np.argsort(scores, axis=-1)[:, -K:][:, ::-1]
    top_k_scores_nobias  = scores[np.arange(S)[:, None], top_k_indices_nobias]
    top_k_weights_nobias = top_k_scores_nobias / top_k_scores_nobias.sum(axis=-1, keepdims=True)
    top_k_weights_nobias = top_k_weights_nobias * cfg.routed_scaling_factor

    # Expert MLP: load all E experts and compute output
    # HF: gate_proj (D_moe, D), up_proj (D_moe, D), down_proj (D, D_moe)
    expert_out = np.zeros((S, D), dtype=np.float32)
    expert_out_nobias = np.zeros((S, D), dtype=np.float32)

    # Collect unique experts needed
    experts_needed = set(top_k_indices.ravel()) | set(top_k_indices_nobias.ravel())
    expert_weights_cache = {}
    for e in experts_needed:
        wi_0 = _load_raw_fp8_dequant(cache, f'{p}.mlp.experts.{e}.gate_proj.weight')  # (D_moe,D)
        wi_1 = _load_raw_fp8_dequant(cache, f'{p}.mlp.experts.{e}.up_proj.weight')    # (D_moe,D)
        wo   = _load_raw_fp8_dequant(cache, f'{p}.mlp.experts.{e}.down_proj.weight')  # (D, D_moe)
        expert_weights_cache[e] = (wi_0, wi_1, wo)

    def _run_expert(e, x_tok):
        """x_tok: (D,) → (D,)"""
        wi_0, wi_1, wo = expert_weights_cache[e]
        gate = 1.0 / (1.0 + np.exp(-(x_tok @ wi_0.T)))  # silu approx via sigmoid won't work
        # SiLU: x * sigmoid(x)
        gate_pre = x_tok @ wi_0.T                         # (D_moe,)
        gate_act = gate_pre * (1.0 / (1.0 + np.exp(-gate_pre.astype(np.float64)))).astype(np.float32)
        up   = x_tok @ wi_1.T                             # (D_moe,)
        hidden = gate_act * up
        return hidden @ wo.T                              # (D,)

    for s in range(S):
        for k_idx in range(K):
            e = top_k_indices[s, k_idx]
            w = top_k_weights[s, k_idx]
            expert_out[s] += w * _run_expert(e, x_np[s])

        for k_idx in range(K):
            e = top_k_indices_nobias[s, k_idx]
            w = top_k_weights_nobias[s, k_idx]
            expert_out_nobias[s] += w * _run_expert(e, x_np[s])

    # Shared expert
    sp = f'{p}.mlp.shared_experts'
    swi_0 = _load_raw_fp8_dequant(cache, f'{sp}.gate_proj.weight')  # (D_moe, D)
    swi_1 = _load_raw_fp8_dequant(cache, f'{sp}.up_proj.weight')    # (D_moe, D)
    swo   = _load_raw_fp8_dequant(cache, f'{sp}.down_proj.weight')  # (D, D_moe)

    sg_pre = x_np @ swi_0.T                                         # (S, D_moe)
    sg_act = sg_pre * (1.0 / (1.0 + np.exp(-sg_pre.astype(np.float64)))).astype(np.float32)
    sup    = x_np @ swi_1.T
    shared_out = (sg_act * sup) @ swo.T

    moe_out_ref   = expert_out + shared_out         # with bias routing
    moe_out_nobias = expert_out_nobias + shared_out  # without bias routing

    # Check which tokens got different expert selection
    routing_match = (top_k_indices == top_k_indices_nobias)
    n_diff = int((~routing_match).any(axis=-1).sum())

    return {
        "top_k_indices":        top_k_indices,
        "top_k_indices_nobias": top_k_indices_nobias,
        "top_k_weights":        top_k_weights,
        "top_k_weights_nobias": top_k_weights_nobias,
        "routing_tokens_differ": n_diff,
        "moe_out":              moe_out_ref,
        "moe_out_nobias":       moe_out_nobias,
        "shared_out":           shared_out,
        "expert_out":           expert_out,
        "expert_out_nobias":    expert_out_nobias,
    }


def load_moe_layer_jax(cache: ShardCache, p: str, cfg):
    """Load one full MoE layer (MLA + MoE weights) via load_weights helpers, no sharding."""
    from load_weights import _load_mla
    lp = _load_mla(cache, p, mesh=None, cfg=cfg)

    # Gate weight: HF (E, D) → transpose to (D, E)
    gate_raw = cache.load_tensor(f'{p}.mlp.gate.weight').astype(np.float32)  # (E, D)
    lp["gate"] = jnp.array(gate_raw.T, dtype=jnp.bfloat16)  # (D, E)
    lp["gate_bias"] = jnp.array(
        cache.load_tensor(f'{p}.mlp.gate.e_score_correction_bias').astype(np.float32))

    # Expert weights: use load_experts_stacked_raw for batch loading
    # Returns (E_local, out, in) fp8 uint8 + optional scale
    E, D, D_moe = cfg.E, cfg.D, cfg.D_moe
    print(f"  Loading {E} experts (3 projections each)...")
    for proj, key in [('gate_proj', 'wi_0'), ('up_proj', 'wi_1'), ('down_proj', 'wo')]:
        w_u8, s = cache.load_experts_stacked_raw(p, proj, E, expert_range=(0, E))
        # HF shape: (E, out, in) → our shape: (E, in, out)
        w_u8 = w_u8.transpose(0, 2, 1)
        if s is not None:
            s = s.transpose(0, 2, 1)
        # Dequant each expert on CPU
        w_f32 = np.zeros(w_u8.shape, dtype=np.float32)
        for e in range(E):
            w_e = w_u8[e]  # (in, out)
            s_e = s[e] if s is not None else None
            # reuse _load_raw_fp8_dequant logic directly
            w_f32[e] = _fp8_dequant_block(w_e, s_e)
        lp[key] = jnp.array(w_f32, dtype=jnp.bfloat16)
        del w_u8, s, w_f32
        print(f"    {proj} done")

    # Shared expert
    sp = f'{p}.mlp.shared_experts'
    for name, key in [('gate_proj', 'shared_wi_0'), ('up_proj', 'shared_wi_1'),
                      ('down_proj', 'shared_wo')]:
        lp[key] = jnp.array(
            _load_raw_fp8_dequant(cache, f'{sp}.{name}.weight').T, dtype=jnp.bfloat16)

    return lp


# ============================================================================
# Load JAX params for a single layer, no sharding
# ============================================================================

def load_single_layer_jax(cache: ShardCache, p: str, cfg):
    """Load one layer's MLA weights via our load_weights helpers (mesh=None → no sharding)."""
    from load_weights import _load_mla
    return _load_mla(cache, p, mesh=None, cfg=cfg)


# ============================================================================
# Main
# ============================================================================

def main():
    import os
    # Multi-host v7x requires all pods to init JAX together.
    # All pods run the comparison; results from pod 0 are what we care about.
    if os.environ.get("MEGASCALE_COORDINATOR_ADDRESS"):
        jax.distributed.initialize(initialization_timeout=600)

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default="/mnt/model/DeepSeek-V3")
    parser.add_argument("--layer", type=int, default=3,
                        help="Which HF layer to compare (default: 3, first MoE layer)")
    parser.add_argument("--seq_len", type=int, default=5)
    parser.add_argument("--fsdp", type=int, default=8,
                        help="FSDP degree for sharded comparison (default: 8)")
    parser.add_argument("--ep", type=int, default=8,
                        help="EP degree for sharded comparison (default: 8)")
    args = parser.parse_args()

    print(f"JAX devices: {jax.device_count()}")

    with open(Path(args.model_dir) / 'model.safetensors.index.json') as f:
        weight_map = json.load(f)['weight_map']
    cache = ShardCache(args.model_dir, weight_map)

    cfg = full_671b_config()
    cfg.moe_backend = "jax"
    cfg.gradient_checkpoint = False
    cfg.mesh = None

    p = f'model.layers.{args.layer}'
    S = args.seq_len
    D = cfg.D
    H = cfg.H
    R_q = cfg.R_q
    R_kv = cfg.R_kv
    d_nope = cfg.d_nope
    d_rope = cfg.d_rope
    d_v = cfg.d_v

    print(f"\nComparing layer {args.layer} MLA, S={S}, D={D}")
    print(f"Config: H={H}, R_q={R_q}, R_kv={R_kv}, d_nope={d_nope}, d_rope={d_rope}, d_v={d_v}")

    # Fixed synthetic input (bf16 range, small values)
    rng = np.random.default_rng(42)
    x_np = rng.standard_normal((S, D)).astype(np.float32) * 0.1

    # ---- HF reference ----
    print("\n[HF numpy reference]")
    ref = hf_ref_mla(cache, p, x_np, H, R_q, R_kv, d_nope, d_rope, d_v, cfg.norm_eps)
    print(f"  attn_out norm: {float(np.sqrt(np.mean(ref['attn_out']**2))):.4f}")

    # ---- JAX path ----
    print("\n[Loading JAX weights (with transpositions)...]")
    layer_p = load_single_layer_jax(cache, p, cfg)

    x_jax = jnp.array(x_np[None, :, :])  # (1, S, D)
    positions_jax = jnp.broadcast_to(jnp.arange(S), (1, S))

    print("[JAX forward]")
    got = jax_mla_intermediates(layer_p, x_jax, positions_jax, cfg)
    print(f"  attn_out norm: {float(np.sqrt(np.mean(got['attn_out']**2))):.4f}")

    # ---- Step-by-step diff ----
    print("\n[Step-by-step diffs (max abs, relative)]")
    steps = [
        "x_normed", "q_low", "q_low_normed", "q",
        "kv_low", "kv_main_normed", "k_nope", "v",
        "q_rope", "k_rope", "query", "key",
        "attn_logits", "attn_weights", "attn_out",
    ]
    for step in steps:
        diff(step, ref[step], got[step])

    # ---- Sharded vs unsharded comparison ----
    # Load the same weights with the REAL mesh (FSDP+EP sharding) and compare
    # to the unsharded result. If they match, the sharding is correct.
    # If they diverge, the sharding is the bug.
    print(f"\n[Sharded vs unsharded: loading with fsdp={args.fsdp}, ep={args.ep}...]")
    shard_cfg = ShardConfig(fsdp=args.fsdp, ep=args.ep)
    mesh = shard_cfg.create_mesh()
    cfg_sharded = full_671b_config()
    cfg_sharded.moe_backend = "jax"
    cfg_sharded.gradient_checkpoint = False
    cfg_sharded.mesh = mesh

    layer_p_sharded = load_single_layer_jax(cache, p, cfg_sharded)

    print("[Sharded JAX forward (layer_p loaded with mesh)]")
    got_sharded = jax_mla_intermediates(layer_p_sharded, x_jax, positions_jax, cfg_sharded)
    print(f"  attn_out norm: {float(np.sqrt(np.mean(got_sharded['attn_out']**2))):.4f}")

    print("\n[Sharded vs unsharded diffs]")
    for step in steps:
        diff(f"sharded/{step}", got[step], got_sharded[step])

    # ---- MoE comparison ----
    # Compute x_post_attn = x_np (residual stream) + attn_out, then post_attn_norm
    # x_np is the pre-layer input; ref['attn_out'] is the MLA output.
    print("\n[MoE comparison: loading full MoE layer weights (E=256 experts)...]")
    x_residual = x_np + ref["attn_out"]  # (S, D) after attention residual
    post_attn_norm_w = _load_raw_f32(cache, f'{p}.post_attention_layernorm.weight')
    h_moe = np_rms_norm(x_residual, post_attn_norm_w, cfg.norm_eps)  # MoE input

    print("[HF numpy MoE reference (with bias routing)]")
    moe_ref = hf_ref_moe(cache, p, h_moe, cfg)
    print(f"  Routing tokens with different experts (bias vs no-bias): "
          f"{moe_ref['routing_tokens_differ']}/{S}")
    print(f"  moe_out (with bias) norm: "
          f"{float(np.sqrt(np.mean(moe_ref['moe_out']**2))):.4f}")
    print(f"  moe_out (no bias) norm:   "
          f"{float(np.sqrt(np.mean(moe_ref['moe_out_nobias']**2))):.4f}")
    diff("moe_out[bias_vs_nobias]", moe_ref["moe_out"], moe_ref["moe_out_nobias"])

    print("\n[Loading JAX MoE weights (batch-loading all 256 experts)...]")
    layer_p_moe = load_moe_layer_jax(cache, p, cfg)

    x_residual_jax = jnp.array(x_residual[None, :, :])  # (1, S, D)
    h_moe_jax = jnp.array(h_moe[None, :, :])            # (1, S, D)

    print("[JAX moe_layer forward]")
    from model import moe_layer as jax_moe_layer
    # Add gate and gate_bias as jax arrays
    layer_p_moe["gate"] = jnp.array(
        cache.load_tensor(f'{p}.mlp.gate.weight').astype(np.float32).T, dtype=jnp.bfloat16)
    layer_p_moe["gate_bias"] = jnp.array(
        cache.load_tensor(f'{p}.mlp.gate.e_score_correction_bias').astype(np.float32))
    jax_moe_out, _ = jax_moe_layer(h_moe_jax, layer_p_moe, cfg)
    jax_moe_out_np = np.array(jax_moe_out[0].astype(jnp.float32))  # (S, D)

    print(f"  JAX moe_out norm: {float(np.sqrt(np.mean(jax_moe_out_np**2))):.4f}")
    print("\n[MoE diffs: JAX vs HF reference (with bias routing)]")
    diff("moe_out",        moe_ref["moe_out"],       jax_moe_out_np)
    diff("moe_out_nobias", moe_ref["moe_out_nobias"], jax_moe_out_np)
    # shared_out from hf vs (jax_moe_out - expert contribution) not easily separable;
    # the full output diff above is the key comparison.

    # Full layer output: x + moe_out
    layer_out_ref = x_residual + moe_ref["moe_out"]
    layer_out_jax = np.array((x_residual_jax[0] + jax_moe_out[0]).astype(jnp.float32))
    diff("full_layer_out", layer_out_ref, layer_out_jax)

    # ---- Embedding layer check ----
    print("\n[Embedding sanity: load embed_tokens.weight, check lookup]")
    embed_np = _load_raw_f32(cache, 'model.embed_tokens.weight')  # (V, D)
    tok_ids = [1, 450, 7483, 310, 3444, 338]  # "The capital of France is" approx
    embed_ref = embed_np[tok_ids[:S]]          # (S, D)
    embed_jax_w = _to_jax(embed_np, mesh=None, spec=None)
    embed_jax = np.array(embed_jax_w[jnp.array(tok_ids[:S])].astype(jnp.float32))
    diff("embed_lookup", embed_ref, embed_jax)


if __name__ == "__main__":
    main()
