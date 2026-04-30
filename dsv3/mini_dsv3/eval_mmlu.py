#!/usr/bin/env python3
"""MMLU-5 evaluation for DeepSeek-V3.

Uses the prefill-only approach: no KV cache or autoregressive decode needed.
For each question, format with 5-shot examples, run forward pass, compare
log-probabilities of answer tokens (A/B/C/D).

Usage (on TPU pod):
    python eval_mmlu.py --model_dir /mnt/model/DeepSeek-V3 --fsdp 8 --ep 8 --n_subjects 5

Requires: datasets (pip install datasets)
"""

from __future__ import annotations

import argparse
import json
import os
import time

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from model import ModelConfig, full_671b_config, ShardConfig, forward
from load_weights import load_hf_weights


# ============================================================================
# MMLU formatting
# ============================================================================

CHOICES = ["A", "B", "C", "D"]

def format_example(question: str, choices: list[str], answer: str | None = None) -> str:
    """Format a single MMLU example using (A)/(B)/(C)/(D) format.

    Prompt ends with 'Correct answer: (' so the model predicts 'A'/'B'/'C'/'D'
    (no leading space) immediately after the open paren.
    """
    prompt = f"Question: {question}\nChoices:"
    for i, choice in enumerate(choices):
        prompt += f" ({CHOICES[i]}) {choice}"
    prompt += "\nCorrect answer: ("
    if answer is not None:
        prompt += f"{answer})"
    return prompt


def format_subject_prompt(subject: str, examples: list[dict], question: dict) -> str:
    """Format a 5-shot MMLU prompt for a subject."""
    prompt = f"The following are multiple choice questions (with answers) about {subject.replace('_', ' ')}.\n\n"
    for ex in examples:
        prompt += format_example(ex["question"], ex["choices"], CHOICES[ex["answer"]])
        prompt += "\n\n"
    prompt += format_example(question["question"], question["choices"])
    return prompt


# ============================================================================
# Evaluation
# ============================================================================

def evaluate_mmlu(params, cfg, tokenizer, subjects=None, n_subjects=None,
                  max_questions=None):
    """Run MMLU evaluation.

    For each question: compute forward pass, get log-prob of A/B/C/D at last position.
    Pick highest log-prob as prediction. Compare with ground truth.
    """
    from datasets import load_dataset

    # Get answer token IDs: prompt ends with "Correct answer: (" so model predicts "A"/"B"/"C"/"D"
    # without leading space (bare letter immediately after open paren).
    answer_ids = {}
    for ch in CHOICES:
        ids = tokenizer.encode(ch, add_special_tokens=False)
        answer_ids[ch] = ids[-1]  # Token for bare "A", "B", "C", "D"
    print(f"Answer token IDs: {answer_ids}")
    # Also try with space prefix as fallback mapping
    answer_ids_space = {}
    for ch in CHOICES:
        ids = tokenizer.encode(f" {ch}", add_special_tokens=False)
        answer_ids_space[ch] = ids[-1]
    print(f"Answer token IDs (space-prefixed): {answer_ids_space}")
    # Build reverse map: token_id -> choice for top-k scanning
    id_to_choice = {}
    for ch in CHOICES:
        id_to_choice[answer_ids[ch]] = ch
        id_to_choice[answer_ids_space[ch]] = ch  # catch both variants

    # Load MMLU dataset
    print("Loading MMLU dataset...")
    ds = load_dataset("cais/mmlu", "all")

    available_subjects = sorted(set(ds["test"]["subject"]))
    if subjects:
        eval_subjects = [s for s in subjects if s in available_subjects]
    elif n_subjects:
        eval_subjects = available_subjects[:n_subjects]
    else:
        eval_subjects = available_subjects

    print(f"Evaluating {len(eval_subjects)} subjects")

    # JIT the forward pass
    @jax.jit
    def get_logits(params, tokens):
        logits, _ = forward(params, tokens, cfg)
        return logits

    # Evaluate
    total_correct = 0
    total_count = 0
    subject_results = {}

    for subj_idx, subject in enumerate(eval_subjects):
        # Get 5-shot examples from dev split
        dev_data = [x for x in ds["dev"] if x["subject"] == subject]
        few_shot = dev_data[:5]

        # Get test questions
        test_data = [x for x in ds["test"] if x["subject"] == subject]
        if max_questions:
            test_data = test_data[:max_questions]

        correct = 0
        count = 0

        for q_idx, question in enumerate(test_data):
            prompt = format_subject_prompt(subject, few_shot, question)
            input_ids = tokenizer.encode(prompt, add_special_tokens=False)
            # Print prompt for first question of first subject to verify format
            if subj_idx == 0 and q_idx == 0:
                print(f"\n=== Sample prompt (subject={subject}, Q0) ===")
                print(repr(prompt[:800]))
                print("===")

            # Truncate from the start to keep the question + "Answer:" at the end.
            # This may drop some few-shot examples but preserves the test question.
            if len(input_ids) > cfg.S - 1:
                input_ids = input_ids[-(cfg.S - 1):]

            tokens = jnp.array([input_ids], dtype=jnp.int32)
            logits = get_logits(params, tokens)

            # Get log-probs at last position for each answer choice
            last_logits = logits[0, -1, :].astype(jnp.float32)
            log_probs = jax.nn.log_softmax(last_logits)

            choice_probs = {ch: float(log_probs[answer_ids[ch]]) for ch in CHOICES}
            pred_logprob = max(choice_probs, key=choice_probs.get)
            gold = CHOICES[question["answer"]]

            # Alternative: pick first letter in top-100 tokens
            top100_idx = jnp.argsort(last_logits)[-100:][::-1]
            pred_topk = None
            for tid in top100_idx:
                tid_int = int(tid)
                if tid_int in id_to_choice:
                    pred_topk = id_to_choice[tid_int]
                    break
            if pred_topk is None:
                pred_topk = pred_logprob  # fallback

            # Debug: print first 3 questions per subject
            if q_idx < 3:
                top5_idx = jnp.argsort(last_logits)[-5:][::-1]
                top5 = [(tokenizer.decode([int(top5_idx[i])]),
                         float(last_logits[top5_idx[i]])) for i in range(5)]
                print(f"    Q{q_idx}: seq_len={len(input_ids)} gold={gold} "
                      f"pred_logprob={pred_logprob} pred_topk={pred_topk} "
                      f"probs={choice_probs} top5={top5}")

            pred = pred_topk  # use top-k method

            if pred == gold:
                correct += 1
            count += 1

        acc = correct / count if count > 0 else 0
        subject_results[subject] = {"correct": correct, "total": count, "accuracy": acc}
        total_correct += correct
        total_count += count

        print(f"  [{subj_idx+1}/{len(eval_subjects)}] {subject}: "
              f"{correct}/{count} = {acc*100:.1f}%")

    overall_acc = total_correct / total_count if total_count > 0 else 0
    print(f"\n{'='*60}")
    print(f"MMLU Overall: {total_correct}/{total_count} = {overall_acc*100:.1f}%")
    print(f"{'='*60}")

    return {
        "overall_accuracy": overall_acc,
        "total_correct": total_correct,
        "total_count": total_count,
        "subjects": subject_results,
    }


def main():
    if os.environ.get("MEGASCALE_COORDINATOR_ADDRESS"):
        jax.distributed.initialize(initialization_timeout=600)

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default="/mnt/model/DeepSeek-V3")
    parser.add_argument("--fsdp", type=int, default=64)
    parser.add_argument("--ep", type=int, default=1)
    parser.add_argument("--n_layers", type=int, default=None)
    parser.add_argument("--n_subjects", type=int, default=None,
                        help="Number of subjects to eval (None=all 57)")
    parser.add_argument("--max_questions", type=int, default=None,
                        help="Max questions per subject")
    parser.add_argument("--subjects", nargs="*", default=None,
                        help="Specific subjects to eval")
    args = parser.parse_args()

    cfg = full_671b_config()
    if args.n_layers is not None:
        cfg.L = args.n_layers
        cfg.L_dense = min(cfg.L_dense, args.n_layers)
    cfg.moe_backend = "jax"  # einsum handles any sharding
    cfg.gradient_checkpoint = False

    shard_cfg = ShardConfig(fsdp=args.fsdp, ep=args.ep)
    mesh = shard_cfg.create_mesh()
    cfg.mesh = mesh

    print(f"Devices: {jax.device_count()} x {jax.devices()[0].device_kind}")
    print(f"Mesh: dp={shard_cfg.dp}, fsdp={shard_cfg.fsdp}, ep={shard_cfg.ep}")

    # Load weights
    print("\nLoading pretrained weights...")
    t0 = time.time()
    params = load_hf_weights(args.model_dir, cfg, mesh=mesh)
    jax.block_until_ready(params)
    load_time = time.time() - t0
    print(f"Weights loaded in {load_time:.0f}s")
    # No EP reshard needed: expert weights have correct global shape with P("ep","fsdp",None).
    # XLA SPMD inserts collectives automatically during inference.

    # Load tokenizer
    print("\nLoading tokenizer...")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)

    # First: quick logit sanity check
    print("\n=== Quick sanity check ===")
    test_prompt = "The capital of France is"
    input_ids = tokenizer.encode(test_prompt)
    tokens = jnp.array([input_ids], dtype=jnp.int32)
    logits, _ = forward(params, tokens, cfg)
    last_logits = logits[0, -1, :].astype(jnp.float32)
    has_nan = bool(jnp.any(jnp.isnan(last_logits)))
    top5_idx = jnp.argsort(last_logits)[-5:][::-1]
    print(f"Prompt: {test_prompt!r}")
    print(f"NaN: {has_nan}")
    if not has_nan:
        for i in range(5):
            tok = tokenizer.decode([int(top5_idx[i])])
            print(f"  {i+1}. {tok!r} ({float(last_logits[top5_idx[i]]):.2f})")

    # Run MMLU
    print("\n=== MMLU Evaluation ===")
    results = evaluate_mmlu(
        params, cfg, tokenizer,
        subjects=args.subjects,
        n_subjects=args.n_subjects,
        max_questions=args.max_questions,
    )

    # Save results
    results_json = json.dumps(results, indent=2)
    print(f"\nResults JSON:\n{results_json}")

    # Try to save to GCS
    try:
        from google.cloud import storage
        client = storage.Client()
        bucket = client.bucket("max-experiments")
        blob = bucket.blob("sivaibhav-dsv3/mmlu_results.json")
        blob.upload_from_string(results_json)
        print("Saved to gs://max-experiments/sivaibhav-dsv3/mmlu_results.json")
    except Exception as e:
        print(f"Could not save to GCS: {e}")


if __name__ == "__main__":
    main()
