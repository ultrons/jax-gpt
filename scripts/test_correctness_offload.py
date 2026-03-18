#!/usr/bin/env python3
"""Correctness test: compare JAX offload forward pass against HuggingFace.

Loads the full Qwen3.5-397B checkpoint one group at a time, runs a short
forward pass, and compares logits against the HuggingFace PyTorch reference.

Usage (4 TPU v5p):
    python scripts/test_correctness_offload.py \
        --model-dir /mnt/disks/tpu_data/qwen3.5-397b \
        --prompt "The answer to the ultimate question" \
        --max-new-tokens 5

The script:
  1. Runs the JAX offload forward pass on your prompt.
  2. Runs the HF model forward pass on the same prompt on CPU.
  3. Compares logits at the last prompt token.
  4. Reports max/mean absolute diff and top-5 token agreement.
"""

from __future__ import annotations

import argparse
import time

import jax
import jax.numpy as jnp
import numpy as np

from jax_gpt.models.qwen35.config import Qwen35Config
from jax_gpt.models.qwen35.offload import forward_offload
from jax_gpt.models.qwen35.sharding import AXIS_RULES_B, make_mesh


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_hf_model(model_dir: str):
    """Load HuggingFace Qwen3.5 model for reference (runs on CPU)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
    print("Loading HF tokenizer...")
    tok = AutoTokenizer.from_pretrained(model_dir)
    print("Loading HF model (CPU, bfloat16)...")
    t0 = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.bfloat16,
        device_map='cpu',
        low_cpu_mem_usage=True,
    )
    model.eval()
    print(f"  HF model loaded in {time.perf_counter() - t0:.1f}s")
    return tok, model


def _hf_forward(model, input_ids):
    """Run HF forward pass, return logits as numpy float32."""
    import torch
    with torch.no_grad():
        out = model(torch.tensor(input_ids))
    return out.logits.float().numpy()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', default='/mnt/disks/tpu_data/qwen3.5-397b')
    parser.add_argument('--prompt', default='The answer to the ultimate question of life is')
    parser.add_argument('--max-new-tokens', type=int, default=3,
                        help='Number of greedy tokens to generate and compare')
    parser.add_argument('--sharding', default='B', choices=['A', 'B'])
    parser.add_argument('--n-devices', type=int, default=None,
                        help='Number of devices (default: all available)')
    parser.add_argument('--skip-hf', action='store_true',
                        help='Skip HF reference (only run JAX offload)')
    args = parser.parse_args()

    axis_rules = AXIS_RULES_B if args.sharding == 'B' else __import__(
        'jax_gpt.models.qwen35.sharding', fromlist=['AXIS_RULES_A']
    ).AXIS_RULES_A

    n_dev = args.n_devices or jax.device_count()
    mesh = make_mesh(n_devices=n_dev)

    print("=" * 70)
    print("QWEN3.5-397B CORRECTNESS TEST (offload mode)")
    print("=" * 70)
    print(f"  Devices:   {n_dev}x {jax.devices()[0].platform}")
    print(f"  Sharding:  Config {args.sharding}")
    print(f"  Model dir: {args.model_dir}")
    print(f"  Prompt:    {args.prompt!r}")
    print()

    # ------------------------------------------------------------------
    # Tokenize
    # ------------------------------------------------------------------
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model_dir)
    input_ids = tok(args.prompt, return_tensors='np').input_ids  # (1, T)
    T = input_ids.shape[1]
    print(f"  Prompt tokens: {T}")

    cfg = Qwen35Config.full()

    # ------------------------------------------------------------------
    # JAX offload forward pass
    # ------------------------------------------------------------------
    print("\n[JAX] Running offload forward pass...")
    tokens_jax = jnp.array(input_ids)

    t0 = time.perf_counter()
    with mesh:
        jax_logits, _ = forward_offload(
            tokens_jax, cfg, args.model_dir, mesh, axis_rules, verbose=True,
        )
    jax.effects_barrier()
    jax_elapsed = time.perf_counter() - t0

    jax_logits_np = np.array(jax_logits).astype(np.float32)
    print(f"  JAX forward done in {jax_elapsed:.1f}s")
    print(f"  Logits shape: {jax_logits_np.shape}")
    print(f"  Logits range: [{jax_logits_np.min():.3f}, {jax_logits_np.max():.3f}]")

    # Top-5 predicted next tokens (from last position)
    last_logits_jax = jax_logits_np[0, -1, :]
    top5_jax = np.argsort(last_logits_jax)[::-1][:5]
    print(f"  Top-5 JAX tokens: {top5_jax.tolist()}")
    print(f"  Top-5 JAX words:  {[tok.decode([t]) for t in top5_jax]}")

    if args.skip_hf:
        print("\n[Skipped HF reference]")
        return

    # ------------------------------------------------------------------
    # HuggingFace reference forward pass
    # ------------------------------------------------------------------
    print("\n[HF] Running reference forward pass (CPU)...")
    _, hf_model = _load_hf_model(args.model_dir)

    t0 = time.perf_counter()
    hf_logits_np = _hf_forward(hf_model, input_ids)
    hf_elapsed = time.perf_counter() - t0
    print(f"  HF forward done in {hf_elapsed:.1f}s")

    last_logits_hf = hf_logits_np[0, -1, :]
    top5_hf = np.argsort(last_logits_hf)[::-1][:5]
    print(f"  Top-5 HF tokens:  {top5_hf.tolist()}")
    print(f"  Top-5 HF words:   {[tok.decode([t]) for t in top5_hf]}")

    # ------------------------------------------------------------------
    # Comparison
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("COMPARISON (last token logits)")
    print("=" * 70)

    # Full vocab comparison
    diff = np.abs(last_logits_jax - last_logits_hf)
    max_diff  = diff.max()
    mean_diff = diff.mean()
    top1_match = top5_jax[0] == top5_hf[0]
    top5_overlap = len(set(top5_jax) & set(top5_hf))

    print(f"  Max  |logits_jax - logits_hf|: {max_diff:.4f}")
    print(f"  Mean |logits_jax - logits_hf|: {mean_diff:.4f}")
    print(f"  Top-1 token match:              {'✅ YES' if top1_match else '❌ NO'}")
    print(f"  Top-5 token overlap:            {top5_overlap}/5")

    # Per-position comparison (all prompt positions)
    all_diff = np.abs(jax_logits_np[0] - hf_logits_np[0])
    print(f"\n  Per-position max diff (all {T} positions):")
    for pos in range(T):
        print(f"    pos {pos:3d}: max={all_diff[pos].max():.4f}  mean={all_diff[pos].mean():.5f}")

    # Pass/fail
    print()
    if max_diff < 2.0 and top1_match:
        print("✅ CORRECTNESS CHECK PASSED")
        print("   (max diff < 2.0 and top-1 token matches — expected for BF16 vs BF16)")
    else:
        print("❌ CORRECTNESS CHECK FAILED")
        print(f"   max_diff={max_diff:.4f}, top1_match={top1_match}")
        print("   Check DeltaNet/GQA/MoE implementations and weight loading.")


if __name__ == '__main__':
    main()
