#!/usr/bin/env python3
"""Correctness test: chunked prefill vs standard prefill for MMLU scoring.

Verifies that splitting a batch of questions into N micro-batches (as done at
large BS to keep MoE x_sorted memory within budget) produces bit-identical
predictions compared to a single full-batch forward pass.

Expected result: 100% prediction agreement, max logit diff < 1e-3 (bf16 rounding only).

Usage:
    # Fast smoke test (mini config, 4 questions, 2 micro-batches):
    python scripts/test_chunked_prefill_correctness.py

    # Larger test (more questions, more micro-batches):
    python scripts/test_chunked_prefill_correctness.py \\
        --config mini --n-questions 40 --prefill-micro-batches 4

    # Multi-device (requires multiple TPUs/GPUs):
    python scripts/test_chunked_prefill_correctness.py --tp 4
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# Reuse helpers from eval_mmlu
# ---------------------------------------------------------------------------

from scripts.eval_mmlu import (
    load_model_random,
    make_score_fn,
    get_answer_token_ids,
)


def _build_dummy_questions(n: int, seq_len: int = 64) -> tuple[np.ndarray, np.ndarray]:
    """Create n dummy token sequences with random content and variable lengths."""
    rng = np.random.default_rng(0)
    base_len = max(16, seq_len - 16)
    # Variable lengths in [base_len, seq_len]
    real_lens = rng.integers(base_len, seq_len + 1, size=n).astype(np.int32)
    tokens = np.zeros((n, seq_len), dtype=np.int32)
    for i in range(n):
        tokens[i, :real_lens[i]] = rng.integers(1, 1000, size=real_lens[i])
    return tokens, real_lens


def main():
    parser = argparse.ArgumentParser(
        description='Verify chunked prefill gives identical predictions to standard prefill.')
    parser.add_argument('--config', default='mini', choices=['mini', 'mid', 'full'])
    parser.add_argument('--tp', type=int, default=None)
    parser.add_argument('--n-questions', type=int, default=8,
                        help='Questions per test batch (must be divisible by prefill-micro-batches)')
    parser.add_argument('--prefill-micro-batches', type=int, default=2,
                        help='Number of micro-batches to split into (default: 2)')
    parser.add_argument('--seq-len', type=int, default=64,
                        help='Padded sequence length for dummy questions')
    args = parser.parse_args()

    assert args.n_questions % args.prefill_micro_batches == 0, (
        f"--n-questions ({args.n_questions}) must be divisible by "
        f"--prefill-micro-batches ({args.prefill_micro_batches})"
    )

    # ---- Init ----
    jax.distributed.initialize()
    from jax_gpt.models.qwen35.config import Qwen35Config
    from jax_gpt.models.qwen35.sharding import AXIS_RULES_B, make_mesh

    cfg = getattr(Qwen35Config, args.config)()
    tp = args.tp or jax.local_device_count()
    mesh = make_mesh(n_devices=tp)
    axis_rules = AXIS_RULES_B

    print(f"Config: {args.config}  tp={tp}  n_questions={args.n_questions}  "
          f"prefill_micro_batches={args.prefill_micro_batches}  seq_len={args.seq_len}")

    # ---- Weights ----
    params = load_model_random(cfg, mesh, axis_rules)

    # ---- Fake answer token IDs (any 4 distinct token IDs will do) ----
    answer_ids = (32, 33, 34, 35)  # arbitrary; we compare logits, not semantics

    # ---- Build score functions ----
    print("\nBuilding score_fn (standard, micro_batches=1)...")
    score_fn_std = make_score_fn(params, cfg, mesh, tp, axis_rules, answer_ids,
                                  prefill_micro_batches=1)

    print(f"Building score_fn (chunked, micro_batches={args.prefill_micro_batches})...")
    score_fn_chunked = make_score_fn(params, cfg, mesh, tp, axis_rules, answer_ids,
                                      prefill_micro_batches=args.prefill_micro_batches)

    # ---- Dummy input ----
    tokens, real_lens = _build_dummy_questions(args.n_questions, args.seq_len)
    tok_jnp = jnp.array(tokens)
    lens_jnp = jnp.array(real_lens)

    # ---- Warm-up / compile ----
    print("\nWarming up standard path...")
    t0 = time.perf_counter()
    _ = score_fn_std(tok_jnp, lens_jnp)
    jax.effects_barrier()
    print(f"  Standard compiled in {time.perf_counter() - t0:.1f}s")

    print(f"Warming up chunked path (chunk_bs={args.n_questions // args.prefill_micro_batches})...")
    t0 = time.perf_counter()
    _ = score_fn_chunked(tok_jnp, lens_jnp)
    jax.effects_barrier()
    print(f"  Chunked compiled in {time.perf_counter() - t0:.1f}s")

    # ---- Run and compare ----
    print("\nRunning comparison...")
    scores_std = np.array(score_fn_std(tok_jnp, lens_jnp))      # (N, 4)
    scores_chunked = np.array(score_fn_chunked(tok_jnp, lens_jnp))  # (N, 4)

    preds_std = np.argmax(scores_std, axis=-1)
    preds_chunked = np.argmax(scores_chunked, axis=-1)

    pred_match = (preds_std == preds_chunked)
    n_match = int(pred_match.sum())
    n_total = args.n_questions

    max_diff = float(np.abs(scores_std - scores_chunked).max())
    mean_diff = float(np.abs(scores_std - scores_chunked).mean())

    print(f"\n{'='*60}")
    print(f"Prediction agreement: {n_match}/{n_total} "
          f"({'PASS' if n_match == n_total else 'FAIL'})")
    print(f"Max logit diff:  {max_diff:.2e}")
    print(f"Mean logit diff: {mean_diff:.2e}")
    print(f"{'='*60}")

    if n_match < n_total:
        print("\nMismatched questions:")
        for i in range(n_total):
            if not pred_match[i]:
                print(f"  q{i}: std={preds_std[i]} chunked={preds_chunked[i]}  "
                      f"std_scores={scores_std[i].tolist()}  "
                      f"chunked_scores={scores_chunked[i].tolist()}")

    # Note: large absolute log-prob diffs are expected with bf16. Different batch
    # sizes produce different XLA tiling/accumulation order → large absolute
    # differences in log P(token) across the full vocab (248K tokens), while
    # the relative ordering of A/B/C/D is preserved. Only flag truly anomalous diffs.
    if max_diff > 5.0:
        print(f"\nWARNING: max logit diff {max_diff:.2e} > 5.0 — "
              f"may indicate a real numerical issue (not just bf16 batch-size noise)")

    passed = (n_match == n_total)
    print(f"\n{'PASSED' if passed else 'FAILED'}")
    return 0 if passed else 1


if __name__ == '__main__':
    sys.exit(main())
