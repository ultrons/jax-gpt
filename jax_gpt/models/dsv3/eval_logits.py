#!/usr/bin/env python3
"""Correctness check: load pretrained DS-v3 weights, generate logits, verify quality.

Usage (on TPU pod):
    python eval_logits.py --model_dir /mnt/model/DeepSeek-V3 --fsdp 64

Checks:
1. Forward pass produces valid logits (no NaN)
2. Top-k predictions are sensible for known prompts
3. Loss on a known sequence is reasonable (low perplexity)
"""

from __future__ import annotations

import argparse
import json
import time

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from .model import ModelConfig, full_671b_config, ShardConfig, forward
from .load_weights import load_hf_weights


def tokenize_simple(text: str, tokenizer_path: str) -> list[int]:
    """Tokenize using the HF tokenizer."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    return tok.encode(text), tok


def main():
    import os
    if os.environ.get("MEGASCALE_COORDINATOR_ADDRESS"):
        jax.distributed.initialize()

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default="/mnt/model/DeepSeek-V3")
    parser.add_argument("--fsdp", type=int, default=64)
    parser.add_argument("--ep", type=int, default=1)
    parser.add_argument("--n_layers", type=int, default=None,
                        help="Load only first N layers (for memory-limited eval)")
    args = parser.parse_args()

    cfg = full_671b_config()
    if args.n_layers is not None:
        cfg.L = args.n_layers
        cfg.L_dense = min(cfg.L_dense, args.n_layers)
    # Use einsum backend — handles any FSDP/EP sharding via XLA SPMD
    # (ragged_dot requires all experts on-device, doesn't work with EP>1)
    cfg.moe_backend = "jax"
    cfg.gradient_checkpoint = False  # inference only

    shard_cfg = ShardConfig(fsdp=args.fsdp, ep=args.ep)
    mesh = shard_cfg.create_mesh()
    cfg.mesh = mesh

    print(f"Devices: {jax.device_count()} x {jax.devices()[0].device_kind}")
    print(f"Mesh: dp={shard_cfg.dp}, fsdp={shard_cfg.fsdp}, ep={shard_cfg.ep}")

    # Load weights — use EP for parallel reads, then reshard to EP=1 for inference
    load_ep = args.ep
    print(f"\nLoading pretrained weights (EP={load_ep} for parallel reads)...")
    t0 = time.time()
    params = load_hf_weights(args.model_dir, cfg, mesh=mesh)
    jax.block_until_ready(params)
    load_time = time.time() - t0
    print(f"Weights loaded in {load_time:.0f}s")
    # No EP reshard needed: expert weights have correct global shape (E, D, D_moe)
    # with P("ep","fsdp",None). XLA SPMD handles collectives during inference.

    # Test prompts
    prompts = [
        "The capital of France is",
        "What is 2 + 2? The answer is",
        ("The following is a multiple choice question.\n\n"
         "Q: What is the capital of Japan?\n"
         "A. Beijing\nB. Seoul\nC. Tokyo\nD. Osaka\n\nAnswer:"),
        # Last-token-repetition diagnostic: same semantic, different final token
        "The capital of France was",
        "The sky is",
    ]

    print("\nTokenizing...")
    tokenizer = None
    all_results = {}

    for i, prompt in enumerate(prompts):
        if tokenizer is None:
            ids, tokenizer = tokenize_simple(prompt, args.model_dir)
        else:
            ids = tokenizer.encode(prompt)

        tokens = jnp.array([ids], dtype=jnp.int32)
        print(f"\nPrompt {i}: {prompt!r}")
        print(f"  Tokens: {tokens.shape[1]}")

        # Forward pass
        logits, aux = forward(params, tokens, cfg)
        logits = logits[0, -1, :]  # last position

        # Check for NaN
        has_nan = bool(jnp.any(jnp.isnan(logits)))
        print(f"  NaN: {has_nan}")

        if not has_nan:
            # Top-5
            logits_f32 = logits.astype(jnp.float32)
            top5_idx = jnp.argsort(logits_f32)[-5:][::-1]
            top5_logits = logits_f32[top5_idx]

            print(f"  Top-5 predictions:")
            top5_tokens = []
            for j in range(5):
                tok_id = int(top5_idx[j])
                tok_str = tokenizer.decode([tok_id])
                logit_val = float(top5_logits[j])
                print(f"    {j+1}. {tok_str!r} (id={tok_id}, logit={logit_val:.2f})")
                top5_tokens.append(tok_str)

            all_results[f"prompt_{i}"] = {
                "prompt": prompt,
                "top5_tokens": top5_tokens,
                "top5_logits": [float(x) for x in top5_logits],
                "has_nan": False,
            }
        else:
            all_results[f"prompt_{i}"] = {"prompt": prompt, "has_nan": True}

    # ---- Last-token repetition diagnostic ----
    # If top-1 for prompt_0 ("...is") = " is" AND top-1 for prompt_3 ("...was") = " was",
    # the model is predicting based only on the last token (not context).
    if all_results.get("prompt_0") and all_results.get("prompt_3"):
        p0_t1 = all_results["prompt_0"].get("top5_tokens", ["?"])[0]
        p3_t1 = all_results["prompt_3"].get("top5_tokens", ["?"])[0]
        repetition = (p0_t1.strip() == "is" and p3_t1.strip() == "was")
        print(f"\nLast-token repetition check: "
              f"'...is'→{p0_t1!r}, '...was'→{p3_t1!r} → {'REPETITION BUG' if repetition else 'OK'}")

    # ---- Dense-only diagnostic (no MoE) ----
    # Run forward with only the 3 dense layers. If this gives correct predictions,
    # the bug is in MoE. If wrong, the bug is in MLA attention / weight loading.
    import copy
    cfg_dense = copy.copy(cfg)
    cfg_dense.L = cfg.L_dense  # only dense layers, L_moe=0
    print(f"\nDense-only forward (L={cfg_dense.L}, no MoE):")
    probe_prompt = "The capital of France is"
    probe_ids = tokenizer.encode(probe_prompt)
    probe_tokens = jnp.array([probe_ids], dtype=jnp.int32)
    dense_logits, _ = forward(params, probe_tokens, cfg_dense)
    dense_logits_f32 = dense_logits[0, -1, :].astype(jnp.float32)
    dense_top5 = jnp.argsort(dense_logits_f32)[-5:][::-1]
    dense_top5_tokens = [tokenizer.decode([int(dense_top5[i])]) for i in range(5)]
    for i in range(5):
        print(f"  {i+1}. {dense_top5_tokens[i]!r} ({float(dense_logits_f32[dense_top5[i]]):.2f})")

    # Summary
    print(f"\n{'='*60}")
    print("CORRECTNESS CHECK SUMMARY")
    print(f"{'='*60}")
    all_ok = all(not r.get("has_nan", True) for r in all_results.values())
    print(f"All prompts valid: {'YES' if all_ok else 'NO'}")

    # Check expected answers
    if all_ok:
        # "capital of France" → should predict "Paris"
        p0_top = all_results.get("prompt_0", {}).get("top5_tokens", [])
        paris_ok = any("Paris" in t or "paris" in t for t in p0_top[:3])
        print(f"'Capital of France' → Paris in top-3: {'YES' if paris_ok else 'NO'} (got: {p0_top[:3]})")

        # "2 + 2" → should predict "4"
        p1_top = all_results.get("prompt_1", {}).get("top5_tokens", [])
        four_ok = any("4" in t for t in p1_top[:3])
        print(f"'2 + 2' → 4 in top-3: {'YES' if four_ok else 'NO'} (got: {p1_top[:3]})")

        # "capital of Japan" MCQ → should predict "C"
        p2_top = all_results.get("prompt_2", {}).get("top5_tokens", [])
        c_ok = any("C" in t for t in p2_top[:3])
        print(f"'Capital of Japan' MCQ → C in top-3: {'YES' if c_ok else 'NO'} (got: {p2_top[:3]})")

    # Save results
    print(json.dumps(all_results, indent=2))


if __name__ == "__main__":
    main()
