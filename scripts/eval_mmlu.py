#!/usr/bin/env python3
"""MMLU 5-shot evaluation for Qwen3.5 on TPU.

Uses loglikelihood scoring: for each question build the standard 5-shot
"Direct Answer" prompt ending with "Answer:", run a single prefill forward
pass, then score log P(" A") / P(" B") / P(" C") / P(" D") at the last
token position and take the argmax.  No generation or decode loop needed.

Target: ~93.0% accuracy for Qwen3.5-397B.

Usage — full 397B on GKE 4×4×4 v5p pod (via k8s/qwen35_mmlu_jobset.yaml):
    python scripts/eval_mmlu.py \\
        --model-dir /mnt/model/qwen3.5-397b \\
        --tp=16 --dp=4

Usage — smoke test (mini config, random weights, 4 chips):
    python scripts/eval_mmlu.py \\
        --random-weights --config mini \\
        --n-questions 20 --tp 4
"""

from __future__ import annotations

import argparse
import functools
import json
import math
import os
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np


# ---------------------------------------------------------------------------
# Distributed init (same as eval_humaneval.py)
# ---------------------------------------------------------------------------

def init_distributed() -> tuple[int, int]:
    jax.config.update("jax_compilation_cache_dir", "gs://sivaibhav-exp/qwen-cc")
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)

    coordinator = os.environ.get('JAX_COORDINATOR_ADDRESS', '')
    num_processes = int(os.environ.get('JAX_NUM_PROCESSES', '1'))
    process_index = int(os.environ.get('JAX_PROCESS_INDEX', '0'))

    if num_processes > 1 and coordinator:
        print(f"[rank {process_index}/{num_processes}] Initializing JAX distributed "
              f"(coordinator={coordinator})...")
        jax.distributed.initialize(
            coordinator_address=coordinator,
            num_processes=num_processes,
            process_id=process_index,
        )
    else:
        print("Initializing JAX distributed (auto-detect from TPU env)...")
        jax.distributed.initialize()

    print(f"[rank {process_index}/{num_processes}] JAX distributed ready. "
          f"Total devices: {jax.device_count()}")
    return process_index, num_processes


# ---------------------------------------------------------------------------
# Model loading (same as eval_humaneval.py)
# ---------------------------------------------------------------------------

def load_model_random(config, mesh, axis_rules):
    from jax_gpt.models.qwen35.model import init_params
    from jax_gpt.models.qwen35.sharding import shard_params

    print(f"Initializing random weights ({config})")
    t0 = time.perf_counter()
    params = init_params(config, jax.random.key(42), dtype=jnp.bfloat16)
    print(f"  init_params done in {time.perf_counter() - t0:.1f}s, sharding...")
    t1 = time.perf_counter()
    with mesh:
        params = shard_params(params, mesh, config, axis_rules)
    jax.effects_barrier()
    print(f"  Sharding done in {time.perf_counter() - t1:.1f}s")
    return params


def load_model_from_checkpoint(model_dir, config, mesh, axis_rules):
    from jax_gpt.models.qwen35.weight_loader import load_from_hf_state_dict

    print(f"Loading weights from {model_dir} ...")
    t0 = time.perf_counter()

    index_path = Path(model_dir) / 'model.safetensors.index.json'
    if index_path.exists():
        sd = _load_safetensors_sharded(model_dir, index_path)
    else:
        import safetensors.torch
        sd = safetensors.torch.load_file(str(Path(model_dir) / 'model.safetensors'))

    params = load_from_hf_state_dict(sd, config, mesh=mesh, axis_rules=axis_rules)
    del sd
    jax.effects_barrier()
    print(f"  Weights loaded and sharded in {time.perf_counter() - t0:.1f}s")
    return params


def _load_safetensors_sharded(model_dir, index_path):
    import safetensors.torch

    with open(index_path) as f:
        index = json.load(f)

    weight_map = index['weight_map']
    shard_files: dict[str, list] = {}
    for tensor_name, shard_file in weight_map.items():
        shard_files.setdefault(shard_file, []).append(tensor_name)

    sd = {}
    for shard_file in sorted(shard_files):
        tensors = safetensors.torch.load_file(str(Path(model_dir) / shard_file))
        sd.update(tensors)
    return sd


# ---------------------------------------------------------------------------
# JIT-compiled loglikelihood scorer
# ---------------------------------------------------------------------------

def make_score_fn(params, config, mesh, tp, axis_rules, answer_ids: tuple[int, ...],
                  prefill_micro_batches: int = 1):
    """Return a JIT-compiled loglikelihood scoring function.

    The returned function scores log-probability of each answer token
    (A/B/C/D) at the last real token position of the prompt.

    Calling convention:
        scores = score_fn(tokens, real_lens)
        tokens:    (B, T) int32, right-padded to length T
        real_lens: (B,)   int32, actual (unpadded) sequence lengths
        scores:    (B, 4) float32, log-probs for A/B/C/D
    """
    from jax_gpt.models.qwen35.model import forward
    from jax_gpt.models.qwen35.sharding import make_cache_sharding
    from jax.sharding import NamedSharding

    _ps = make_cache_sharding(config, mesh, axis_rules)
    cache_sharding = {k: NamedSharding(mesh, v) for k, v in _ps.items()}

    # Bake answer token IDs into the trace as Python-level constants.
    # answer_ids is a 4-tuple of ints — closed over as static values.
    _aid = tuple(int(x) for x in answer_ids)

    @jax.jit
    def _score(params, tokens, real_lens):
        # tokens: (B, T) right-padded
        # real_lens: (B,) actual lengths (last real token at real_lens[i]-1)
        logits, _ = forward(
            params, tokens, config,
            cache=None,
            is_decode=False,
            n_devices=tp,
            axis_name='tp',
            mesh=mesh,
            cache_sharding=cache_sharding,
        )  # (B, T, V)

        B = tokens.shape[0]
        # Gather the logit at each sequence's last real token position.
        # With right-padding: real token at index real_lens[i]-1.
        last_logits = logits[jnp.arange(B), real_lens - 1, :]  # (B, V)

        # Log-softmax in float32 for numerical precision.
        log_probs = jax.nn.log_softmax(last_logits.astype(jnp.float32), axis=-1)

        # Extract scores for A/B/C/D — static indexing, no runtime overhead.
        return jnp.stack([log_probs[:, i] for i in _aid], axis=-1)  # (B, 4)

    if prefill_micro_batches <= 1:
        def _call(tokens, real_lens):
            return _score(params, tokens, real_lens)
    else:
        def _call(tokens, real_lens):
            B = tokens.shape[0]
            assert B % prefill_micro_batches == 0, (
                f"batch_size ({B}) must be divisible by "
                f"prefill_micro_batches ({prefill_micro_batches})"
            )
            chunk_bs = B // prefill_micro_batches
            chunks = []
            for mb in range(prefill_micro_batches):
                s = mb * chunk_bs
                chunks.append(_score(params, tokens[s:s + chunk_bs],
                                     real_lens[s:s + chunk_bs]))
            return jnp.concatenate(chunks, axis=0)

    return _call


# ---------------------------------------------------------------------------
# Answer token IDs
# ---------------------------------------------------------------------------

def get_answer_token_ids(tokenizer) -> list[int]:
    """Return single-token IDs for answer choices A, B, C, D.

    Tries ' A' (with leading space) first since that is the natural
    continuation after 'Answer:'. Falls back to 'A' without space.
    """
    ids = []
    for letter in 'ABCD':
        with_space = tokenizer.encode(f' {letter}', add_special_tokens=False)
        without_space = tokenizer.encode(letter, add_special_tokens=False)
        if len(with_space) == 1:
            ids.append(with_space[0])
        elif len(without_space) == 1:
            ids.append(without_space[0])
        else:
            raise ValueError(
                f"No single-token encoding for answer letter {letter!r}. "
                f"with_space={with_space}, without_space={without_space}")
    print(f"Answer token IDs: "
          f"A={ids[0]} B={ids[1]} C={ids[2]} D={ids[3]} "
          f"({[tokenizer.decode([i]) for i in ids]})")
    return ids


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_mmlu(n_questions: int | None = None) -> tuple[list[dict], dict[str, list[dict]]]:
    """Load MMLU test questions and per-subject 5-shot dev examples.

    Returns:
        questions: list of test dicts with keys
            question, choices (list[str]), answer (int 0-3), subject
        dev_by_subject: dict mapping subject → list of 5 dev dicts
    """
    from datasets import load_dataset

    print("Loading MMLU dataset (cais/mmlu all)...")
    t0 = time.perf_counter()
    ds_test = load_dataset('cais/mmlu', 'all', split='test')
    ds_dev  = load_dataset('cais/mmlu', 'all', split='dev')
    print(f"  Loaded in {time.perf_counter() - t0:.1f}s: "
          f"{len(ds_test)} test, {len(ds_dev)} dev examples")

    questions = [dict(q) for q in ds_test]
    if n_questions is not None:
        questions = questions[:n_questions]

    dev_by_subject: dict[str, list[dict]] = {}
    for ex in ds_dev:
        dev_by_subject.setdefault(ex['subject'], []).append(dict(ex))

    return questions, dev_by_subject


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

SUBJECT_DISPLAY = {s: s.replace('_', ' ') for s in [
    'abstract_algebra', 'anatomy', 'astronomy', 'business_ethics',
    'clinical_knowledge', 'college_biology', 'college_chemistry',
    'college_computer_science', 'college_mathematics', 'college_medicine',
    'college_physics', 'computer_security', 'conceptual_physics',
    'econometrics', 'electrical_engineering', 'elementary_mathematics',
    'formal_logic', 'global_facts', 'high_school_biology', 'high_school_chemistry',
    'high_school_computer_science', 'high_school_european_history',
    'high_school_geography', 'high_school_government_and_politics',
    'high_school_macroeconomics', 'high_school_mathematics',
    'high_school_microeconomics', 'high_school_physics', 'high_school_psychology',
    'high_school_statistics', 'high_school_us_history', 'high_school_world_history',
    'human_aging', 'human_sexuality', 'international_law', 'jurisprudence',
    'logical_fallacies', 'machine_learning', 'management', 'marketing',
    'medical_genetics', 'miscellaneous', 'moral_disputes', 'moral_scenarios',
    'nutrition', 'philosophy', 'prehistory', 'professional_accounting',
    'professional_law', 'professional_medicine', 'professional_psychology',
    'public_relations', 'security_studies', 'sociology', 'us_foreign_policy',
    'virology', 'world_religions',
]}


def _format_question(q: str, choices: list[str], answer: int | None = None) -> str:
    text = f"Question: {q}\n"
    for i, c in enumerate(choices):
        text += f"{chr(ord('A') + i)}. {c}\n"
    text += "Answer:"
    if answer is not None:
        text += f" {chr(ord('A') + answer)}"
    return text


def build_5shot_prompt(subject: str, dev_examples: list[dict], question: str,
                       choices: list[str]) -> str:
    """Build a standard MMLU 5-shot direct-answer prompt."""
    display = SUBJECT_DISPLAY.get(subject, subject.replace('_', ' '))
    header = f"The following are multiple choice questions (with answers) about {display}.\n\n"
    shots = ''.join(
        _format_question(ex['question'], ex['choices'], ex['answer']) + "\n\n"
        for ex in dev_examples[:5]
    )
    target = _format_question(question, choices, answer=None)
    return header + shots + target


# ---------------------------------------------------------------------------
# Batched evaluation
# ---------------------------------------------------------------------------

def evaluate_questions(
    score_fn,
    tokenizer,
    questions: list[dict],
    dev_by_subject: dict[str, list[dict]],
    batch_size: int,
    rank: int,
) -> list[dict]:
    """Score all questions and return result dicts."""

    # ---- Tokenize all prompts ----
    print(f"[rank {rank}] Tokenizing {len(questions)} prompts...")
    t0 = time.perf_counter()
    prompts = [
        build_5shot_prompt(q['subject'], dev_by_subject.get(q['subject'], []),
                           q['question'], q['choices'])
        for q in questions
    ]
    token_ids_list = [
        tokenizer.encode(p, add_special_tokens=False)
        for p in prompts
    ]
    real_lens = [len(ids) for ids in token_ids_list]
    max_len = max(real_lens)
    # Round up to next multiple of 128 for XLA tiling efficiency
    max_len_padded = math.ceil(max_len / 128) * 128
    print(f"[rank {rank}]   max_len={max_len} → padded to {max_len_padded}  "
          f"(in {time.perf_counter() - t0:.1f}s)")

    # Right-pad all sequences to max_len_padded with 0 (pad token)
    pad_id = tokenizer.pad_token_id or 0
    tokens_arr = np.full((len(questions), max_len_padded), pad_id, dtype=np.int32)
    for i, ids in enumerate(token_ids_list):
        tokens_arr[i, :len(ids)] = ids

    # ---- Warm-up (compile JIT) ----
    print(f"[rank {rank}] Warming up score JIT ({batch_size} × {max_len_padded})...")
    t_compile = time.perf_counter()
    _dummy_tok = jnp.zeros((batch_size, max_len_padded), dtype=jnp.int32)
    _dummy_lens = jnp.ones((batch_size,), dtype=jnp.int32)
    score_fn(_dummy_tok, _dummy_lens)
    jax.effects_barrier()
    print(f"[rank {rank}]   JIT compiled in {time.perf_counter() - t_compile:.1f}s")

    # ---- Batch evaluation ----
    n = len(questions)
    n_batches = math.ceil(n / batch_size)
    results = []
    n_correct = 0

    for b in range(n_batches):
        s = b * batch_size
        e = min(s + batch_size, n)
        actual_bs = e - s

        # Pad last batch to full batch_size (avoids JIT recompile)
        tok_batch = np.zeros((batch_size, max_len_padded), dtype=np.int32)
        len_batch = np.ones((batch_size,), dtype=np.int32)
        tok_batch[:actual_bs] = tokens_arr[s:e]
        len_batch[:actual_bs] = np.array(real_lens[s:e], dtype=np.int32)

        t_step = time.perf_counter()
        scores = score_fn(
            jnp.array(tok_batch, dtype=jnp.int32),
            jnp.array(len_batch, dtype=jnp.int32),
        )  # (batch_size, 4)
        scores_np = np.array(scores[:actual_bs])  # (actual_bs, 4)
        elapsed = time.perf_counter() - t_step

        for i in range(actual_bs):
            qi = questions[s + i]
            pred = int(np.argmax(scores_np[i]))
            correct = (pred == qi['answer'])
            n_correct += int(correct)

            results.append({
                'subject': qi['subject'],
                'question': qi['question'][:120],
                'answer': qi['answer'],
                'pred': pred,
                'correct': correct,
                'scores': scores_np[i].tolist(),
            })

        acc = n_correct / (s + actual_bs)
        status = '✓' if results[-1]['correct'] else '✗'
        print(f"[rank {rank}] [{s + actual_bs}/{n}] {status}  "
              f"batch={actual_bs}  {elapsed:.2f}s  acc={acc:.4f}", flush=True)

    return results


# ---------------------------------------------------------------------------
# Result merging
# ---------------------------------------------------------------------------

def merge_and_summarise(output_dir: Path, dp: int, total_questions: int) -> dict:
    all_results = []
    for r in range(dp):
        p = output_dir / f'results_rank{r:02d}.json'
        with open(p) as f:
            all_results.extend(json.load(f)['results'])

    # Sort by original order (results_rank00 then 01 etc. are already ordered)
    n_correct = sum(r['correct'] for r in all_results)
    accuracy = n_correct / len(all_results) if all_results else 0.0

    # Per-subject accuracy
    by_subject: dict[str, list] = {}
    for r in all_results:
        by_subject.setdefault(r['subject'], []).append(r['correct'])
    subject_acc = {s: sum(v) / len(v) for s, v in sorted(by_subject.items())}

    summary = {
        'n_total': len(all_results),
        'n_correct': n_correct,
        'accuracy': accuracy,
        'subject_accuracy': subject_acc,
    }
    summary_path = output_dir / 'summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"MMLU 5-shot accuracy: {accuracy:.4f}  ({n_correct}/{len(all_results)})")
    print(f"{'='*60}")
    print("Per-subject (bottom 10):")
    for s, a in sorted(subject_acc.items(), key=lambda x: x[1])[:10]:
        print(f"  {s:<45s} {a:.3f}")
    print("Per-subject (top 10):")
    for s, a in sorted(subject_acc.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {s:<45s} {a:.3f}")
    print(f"\nSummary written to {summary_path}")
    return summary


def _barrier_file_sync(output_dir: Path, rank: int, dp: int, timeout: int = 600):
    sentinel = output_dir / f'.done_rank{rank:02d}'
    sentinel.touch()
    deadline = time.time() + timeout
    while time.time() < deadline:
        if all((output_dir / f'.done_rank{r:02d}').exists() for r in range(dp)):
            return
        time.sleep(5)
    raise TimeoutError(f"Timed out waiting for all {dp} DP ranks to finish")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', default='/mnt/model',
                        help='Path to model dir (weights + tokenizer)')
    parser.add_argument('--tokenizer', default=None,
                        help='Tokenizer path (defaults to --model-dir)')
    parser.add_argument('--output-dir', default='/tmp/mmlu_out')
    parser.add_argument('--config', default='full', choices=['mini', 'mid', 'full'])
    parser.add_argument('--random-weights', action='store_true')
    parser.add_argument('--sharding', default='B', choices=['A', 'B'])
    parser.add_argument('--tp', type=int, default=None)
    parser.add_argument('--dp', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Questions per forward pass per DP rank')
    parser.add_argument('--n-questions', type=int, default=None,
                        help='Limit total questions (default: all ~14K)')
    parser.add_argument('--prefill-micro-batches', type=int, default=1,
                        help='Split each forward pass into N sub-batches (batch-size must be divisible). '
                             'Tests that chunked execution gives identical predictions. Default: 1 (no chunking).')
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Distributed init
    # ------------------------------------------------------------------
    rank, world_size = init_distributed()

    from jax_gpt.models.qwen35.config import Qwen35Config
    from jax_gpt.models.qwen35.sharding import AXIS_RULES_A, AXIS_RULES_B, make_mesh

    axis_rules = AXIS_RULES_B if args.sharding == 'B' else AXIS_RULES_A
    cfg = getattr(Qwen35Config, args.config)()

    # ------------------------------------------------------------------
    # Build device mesh
    # ------------------------------------------------------------------
    tp = args.tp or jax.local_device_count()
    dp = args.dp
    total_devices = tp * dp

    assert jax.device_count() >= total_devices, (
        f"Need {total_devices} devices, have {jax.device_count()}")

    devices = np.array(jax.devices()[:total_devices])
    if dp > 1:
        mesh = jax.sharding.Mesh(devices.reshape(dp, tp), ('dp', 'tp'))
    else:
        mesh = make_mesh(n_devices=tp)

    processes_per_dp_replica = tp // jax.local_device_count()
    dp_rank = rank // processes_per_dp_replica

    print(f"[rank {rank}] mesh={dict(mesh.shape)}  dp_rank={dp_rank}  config={args.config}")

    # ------------------------------------------------------------------
    # Tokenizer
    # ------------------------------------------------------------------
    from transformers import AutoTokenizer
    tok_src = args.tokenizer or args.model_dir
    tok = AutoTokenizer.from_pretrained(tok_src)
    answer_ids = get_answer_token_ids(tok)

    # ------------------------------------------------------------------
    # Load weights
    # ------------------------------------------------------------------
    if args.random_weights:
        params = load_model_random(cfg, mesh, axis_rules)
    else:
        params = load_model_from_checkpoint(args.model_dir, cfg, mesh, axis_rules)

    # ------------------------------------------------------------------
    # Load dataset and split across DP ranks
    # ------------------------------------------------------------------
    questions, dev_by_subject = load_mmlu(n_questions=args.n_questions)
    n_total = len(questions)

    questions_per_rank = math.ceil(n_total / dp)
    q_start = dp_rank * questions_per_rank
    q_end = min(q_start + questions_per_rank, n_total)
    my_questions = questions[q_start:q_end]

    print(f"[rank {rank}] dp_rank={dp_rank}  questions {q_start}..{q_end - 1} "
          f"({len(my_questions)} questions, batch_size={args.batch_size})")

    # ------------------------------------------------------------------
    # Build score function
    # ------------------------------------------------------------------
    score_fn = make_score_fn(params, cfg, mesh, tp, axis_rules, tuple(answer_ids),
                             prefill_micro_batches=args.prefill_micro_batches)

    # ------------------------------------------------------------------
    # Sanity check: verify " A"/" B"/" C"/" D" logit ordering on a dummy
    # prompt where the answer is known.  All TP ranks participate in lockstep.
    # ------------------------------------------------------------------
    _sanity_prompt = (
        "The following are multiple choice questions (with answers) about mathematics.\n\n"
        "Question: What is 1 + 1?\n"
        "A. 1\n"
        "B. 2\n"
        "C. 3\n"
        "D. 4\n"
        "Answer: B\n\n"
        "Question: What is 2 + 2?\n"
        "A. 2\n"
        "B. 3\n"
        "C. 4\n"
        "D. 5\n"
        "Answer:"
    )
    _sid = tok.encode(_sanity_prompt, add_special_tokens=False)
    _slen = len(_sid)
    _stok = np.zeros((args.batch_size, _slen), dtype=np.int32)
    _stok[0] = _sid
    _slens = np.ones((args.batch_size,), dtype=np.int32)
    _slens[0] = _slen
    _sanity_scores = np.array(score_fn(
        jnp.array(_stok, dtype=jnp.int32),
        jnp.array(_slens, dtype=jnp.int32),
    )[:1])  # (1, 4)
    _sanity_pred = int(np.argmax(_sanity_scores[0]))
    _sanity_letter = 'ABCD'[_sanity_pred]
    if rank == 0:
        print(f"[SANITY rank0] 2+2=4 → pred={_sanity_letter}  "
              f"scores={_sanity_scores[0].tolist()}  "
              f"({'PASS' if _sanity_pred == 2 else 'FAIL — model may be misconfigured'})",
              flush=True)

    # ------------------------------------------------------------------
    # Evaluate
    # ------------------------------------------------------------------
    results = evaluate_questions(
        score_fn, tok, my_questions, dev_by_subject,
        batch_size=args.batch_size, rank=rank,
    )

    # ------------------------------------------------------------------
    # Write per-DP-rank results
    # ------------------------------------------------------------------
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_correct = sum(r['correct'] for r in results)
    is_dp_leader = (rank % processes_per_dp_replica == 0)
    if is_dp_leader:
        rank_file = output_dir / f'results_rank{dp_rank:02d}.json'
        with open(rank_file, 'w') as f:
            json.dump({'dp_rank': dp_rank, 'results': results,
                       'n_correct': n_correct, 'n_total': len(results)}, f, indent=2)
        print(f"[rank {rank}] Wrote {rank_file}  "
              f"acc={n_correct}/{len(results)}={n_correct/len(results):.4f}")

    # ------------------------------------------------------------------
    # Merge (DP leader only, after barrier)
    # ------------------------------------------------------------------
    if dp > 1:
        _barrier_file_sync(output_dir, dp_rank, dp)

    if rank == 0:
        merge_and_summarise(output_dir, dp, n_total)


if __name__ == '__main__':
    main()
