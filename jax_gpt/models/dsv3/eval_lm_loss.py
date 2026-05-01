"""Tier-2 correctness probe: LM cross-entropy on real text with real weights.

Loads the released HuggingFace DeepSeek-V3 base weights from the cluster's
PVC mount (`/mnt/model/DeepSeek-V3` by default), tokenizes a fixed English
snippet via the matching HF tokenizer, runs one forward pass, and reports
**LM cross-entropy only** (NOT including the MoE aux/load-balance term).

If the reported lm_loss is in the expected base-model range (~2-4 for
well-trained DSv3 on natural English), the (load_hf_weights → forward →
_vocab_ce) path is correct — and any apparent "high loss" we see in
random-init training runs is just the expected MoE aux penalty + random-init
cross-entropy, not a forward-pass bug.

Usage (cluster):
    python -m jax_gpt.models.dsv3.eval_lm_loss \\
        --model_dir /mnt/model/DeepSeek-V3 --fsdp 64 --ep 1

Designed to slot into the cde workflow:
    cde run --tag t2-hf-lm-loss --inherit t1-loss-split \\
        --set entrypoint_module=jax_gpt.models.dsv3.eval_lm_loss \\
        --set mount_weights=true \\
        --set fsdp=64 --set ep=1 ...
"""

from __future__ import annotations

import argparse
import time

import jax
import jax.numpy as jnp

from .model import (
    ModelConfig, full_671b_config, ShardConfig, forward,
    _vocab_ce,
)
from .load_weights import load_hf_weights


# A short English snippet — chosen to be unambiguous, well-formed prose.
# 60+ tokens at the DSv3 BPE granularity is enough to get a stable LM CE
# without inflating compile time / memory.
DEFAULT_TEXT = (
    "The capital of France is Paris. France is a country in Western Europe "
    "known for its art, culture, cuisine, and history. Paris, the capital, "
    "is home to many world-famous landmarks including the Eiffel Tower, "
    "the Louvre Museum, and the Notre-Dame Cathedral. The Seine River runs "
    "through the heart of the city, dividing it into two banks: the Right "
    "Bank and the Left Bank, each with its own distinct character."
)


def main():
    import os
    if os.environ.get("MEGASCALE_COORDINATOR_ADDRESS"):
        jax.distributed.initialize()

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default="/mnt/model/DeepSeek-V3")
    parser.add_argument("--fsdp", type=int, default=64)
    parser.add_argument("--ep", type=int, default=1)
    parser.add_argument("--n_layers", type=int, default=None,
                        help="Load only first N layers (memory-limited debug)")
    parser.add_argument("--text", default=DEFAULT_TEXT,
                        help="Override the test text snippet")
    # Inherited from t1 via cde --inherit; ignore safely.
    args, unknown = parser.parse_known_args()
    if unknown:
        print(f"(ignoring inherited training-only flags: {unknown})")

    # Config (reuse eval_logits.py conventions for consistency).
    cfg = full_671b_config()
    if args.n_layers is not None:
        cfg.L = args.n_layers
        cfg.L_dense = min(cfg.L_dense, args.n_layers)
    cfg.moe_backend = "jax"          # einsum backend — works under any FSDP/EP
    cfg.gradient_checkpoint = False  # inference only

    shard_cfg = ShardConfig(fsdp=args.fsdp, ep=args.ep)
    mesh = shard_cfg.create_mesh()
    cfg.mesh = mesh

    print(f"Devices: {jax.device_count()} x {jax.devices()[0].device_kind}")
    print(f"Mesh: dp={shard_cfg.dp}, fsdp={shard_cfg.fsdp}, ep={shard_cfg.ep}")
    print(f"Layers: {cfg.L} ({cfg.L_dense} dense + {cfg.L - cfg.L_dense} MoE)")

    # ── Load weights ──────────────────────────────────────────────────────
    print(f"\nLoading HF DeepSeek-V3 weights from {args.model_dir} ...")
    t0 = time.time()
    params = load_hf_weights(args.model_dir, cfg, mesh=mesh)
    jax.block_until_ready(params)
    print(f"Weights loaded in {time.time() - t0:.0f}s")

    # ── Tokenize ──────────────────────────────────────────────────────────
    print("\nTokenizing test text...")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    ids = tokenizer.encode(args.text)
    # Pad to the model's expected sequence length (cfg.S) so the existing
    # forward + _vocab_ce shapes work without modification.
    if len(ids) > cfg.S:
        ids = ids[:cfg.S]
    elif len(ids) < cfg.S:
        # Right-pad with the eos / pad token so the trailing positions
        # don't dominate the average loss; we'll mask them out below.
        pad_id = tokenizer.eos_token_id or tokenizer.pad_token_id or 0
        ids = ids + [pad_id] * (cfg.S - len(ids))
    real_len = min(len(tokenizer.encode(args.text)), cfg.S)
    print(f"Real-text tokens: {real_len} / {cfg.S} (padded the rest)")

    # Defensive: our cfg.V is hardcoded to 102400 but HF DSv3 vocab is 129280.
    # IDs ≥ cfg.V would index out of bounds in our chunked LM CE. Clamp + warn.
    n_oob = sum(1 for t in ids if t >= cfg.V)
    if n_oob:
        print(f"WARN: {n_oob} tokens have ID >= cfg.V={cfg.V} (HF vocab is 129280); "
              f"clamping to cfg.V-1. Fix cfg.V to 129280 for a clean number.")
        ids = [min(t, cfg.V - 1) for t in ids]

    tokens = jnp.array([ids], dtype=jnp.int32)  # (B=1, S)

    # ── Forward + LM CE ───────────────────────────────────────────────────
    print("\nRunning forward pass...")
    t0 = time.time()
    x_final, aux_loss = forward(params, tokens, cfg, return_final_x=True)
    x_pred = x_final[:, :-1]                      # (1, S-1, D)
    targets = tokens[:, 1:1 + x_pred.shape[1]].astype(jnp.int32)
    V_CHUNK = 4096
    n_chunks = cfg.V // V_CHUNK
    lm_loss_full = _vocab_ce(x_pred, params["output_head"],
                             targets, n_chunks, V_CHUNK)

    # ── LM CE on real-text positions only ─────────────────────────────────
    # _vocab_ce reduces over (B, S-1). Recompute on a sliced view that only
    # covers the actual text positions (not the pad).
    real_S = max(1, real_len - 1)  # number of next-token predictions in real text
    x_pred_real = x_pred[:, :real_S, :]
    targets_real = targets[:, :real_S]
    # Round real_S up to V_CHUNK-friendly sizing isn't required — _vocab_ce
    # handles arbitrary S; only the vocab dim is chunked.
    lm_loss_real = _vocab_ce(x_pred_real, params["output_head"],
                             targets_real, n_chunks, V_CHUNK)

    jax.block_until_ready(lm_loss_real)
    print(f"Forward + CE took {time.time() - t0:.1f}s")

    print("\n" + "=" * 70)
    print(f"  LM cross-entropy (real text, {real_S} positions): "
          f"{float(lm_loss_real):.4f}")
    print(f"  LM cross-entropy (full S incl. pad, {cfg.S - 1} positions): "
          f"{float(lm_loss_full):.4f}")
    print(f"  MoE aux_loss (small coefficient ~1e-4):           "
          f"{float(aux_loss):.6f}")
    print("=" * 70)
    print("Reference baselines:")
    print("  Random init + uniform vocab → CE ≈ log(vocab) = log(102400) ≈ 11.54")
    print("  Well-trained DSv3 base on natural English → CE ≈ 2-4")
    print("  MaxText reported 5.337 = mid-training resume, not directly comparable")


if __name__ == "__main__":
    main()
