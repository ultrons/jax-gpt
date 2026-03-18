#!/usr/bin/env python3
"""HumanEval evaluation for Qwen3.5 on TPU.

Supports:
  - Full Qwen3.5-397B on a 4×4×4 v5p pod (64 chips, via GKE JobSet)
  - Mini / mid configs with random weights for pipeline smoke-testing

Usage — smoke test on this VM (4 chips, mini config, random weights):
    python scripts/eval_humaneval.py \
        --model-dir /mnt/disks/tpu_data/qwen3.5-397b \
        --random-weights \
        --config mini \
        --n-problems 5 \
        --tp 4

Usage — full 397B on GKE 4×4×4 pod (launched via k8s/qwen35_eval_jobset.yaml):
    python scripts/eval_humaneval.py \
        --model-dir /mnt/model \
        --config full \
        --n-problems 164 \
        --tp 8 --dp 8

Output:
    <output-dir>/results_rank<N>.json  — per-dp-rank partial results
    <output-dir>/summary.json          — merged summary (dp-rank 0 only)
"""

from __future__ import annotations

import argparse
import functools
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np


# ---------------------------------------------------------------------------
# Distributed init
# ---------------------------------------------------------------------------

def init_distributed() -> tuple[int, int]:
    """Init JAX distributed, with auto-detection fallback.

    Prefers explicit JAX_COORDINATOR_ADDRESS env var (single-host testing).
    Falls back to jax.distributed.initialize() with no args, which auto-detects
    from TPU_WORKER_ID / TPU_WORKER_HOSTNAMES injected by the GKE TPU webhook.

    Returns (process_index, num_processes).
    """
    # Persistent GCS compilation cache — survives pod restarts.
    # Thresholds set to 0 so all compilations are cached (default min is 1s).
    jax.config.update("jax_compilation_cache_dir", "gs://sivaibhav-exp/qwen-cc")
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)

    coordinator = os.environ.get('JAX_COORDINATOR_ADDRESS', '')
    num_processes = int(os.environ.get('JAX_NUM_PROCESSES', '1'))
    process_index = int(os.environ.get('JAX_PROCESS_INDEX', '0'))

    if num_processes > 1 and coordinator:
        # Explicit mode: single-host testing or manual override
        print(f"[rank {process_index}/{num_processes}] Initializing JAX distributed "
              f"(coordinator={coordinator})...")
        jax.distributed.initialize(
            coordinator_address=coordinator,
            num_processes=num_processes,
            process_id=process_index,
        )
    else:
        # Auto-detect mode: GKE TPU webhook injects TPU_WORKER_ID / TPU_WORKER_HOSTNAMES
        print("Initializing JAX distributed (auto-detect from TPU env)...")
        jax.distributed.initialize()
        process_index = jax.process_index()
        num_processes = jax.process_count()

    print(f"[rank {process_index}/{num_processes}] JAX distributed ready. "
          f"Total devices: {jax.device_count()}")
    return process_index, num_processes


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model_random(config, mesh, axis_rules):
    """Initialize random weights and shard them (for smoke-testing)."""
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
    """Load full model weights from safetensors checkpoint and shard.

    Weights are loaded layer-by-layer as numpy (CPU) and sharded to devices
    immediately — peak host RAM is one MoE layer (~8 GB) rather than the
    full model.
    """
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


def _load_safetensors_sharded(model_dir: str, index_path: Path) -> dict:
    import safetensors.torch

    with open(index_path) as f:
        index = json.load(f)

    weight_map: dict[str, str] = index['weight_map']
    shard_files: dict[str, list[str]] = {}
    for tensor_name, shard_file in weight_map.items():
        shard_files.setdefault(shard_file, []).append(tensor_name)

    sd = {}
    for shard_file in sorted(shard_files):
        tensors = safetensors.torch.load_file(str(Path(model_dir) / shard_file))
        sd.update(tensors)
    return sd


# ---------------------------------------------------------------------------
# JIT-compiled forward / generate
# ---------------------------------------------------------------------------

def make_forward_fn(params, config, mesh, tp: int, axis_rules=None):
    """Return a JIT-compiled forward function.

    Calling convention:
        logits, cache = fwd(params, tokens, cache, is_decode)
    params is passed explicitly (not closed over) so JAX multi-process JIT
    can handle sharded arrays spanning non-addressable devices.
    is_decode is a Python bool (static).
    """
    from jax_gpt.models.qwen35.model import forward
    from jax_gpt.models.qwen35.sharding import make_cache_sharding

    # Build cache sharding constraints so GSPMD doesn't try to shard
    # KV heads (n_kv_heads=2) across TP devices — they must stay replicated.
    # Convert PartitionSpecs to NamedShardings so with_sharding_constraint
    # works without requiring a mesh context manager inside JIT.
    from jax.sharding import NamedSharding
    _ps = make_cache_sharding(config, mesh, axis_rules)
    cache_sharding = {k: NamedSharding(mesh, v) for k, v in _ps.items()}

    @functools.partial(jax.jit, static_argnums=(3,))
    def _fwd(params, tokens, cache, is_decode):
        return forward(
            params, tokens, config,
            cache=cache,
            is_decode=is_decode,
            n_devices=tp,
            axis_name='tp',
            mesh=mesh,
            cache_sharding=cache_sharding,
        )

    # Return a wrapper that binds params so call sites keep the same interface
    def _call(tokens, cache, is_decode):
        return _fwd(params, tokens, cache, is_decode)

    return _call


def _extract_function_body(text: str) -> str:
    """Extract Python function body from model output.

    Handles:
    - Qwen3 thinking blocks (<think>...</think>) — strip, take content after
    - Markdown code blocks (```python ... ```)
    - Thinking preamble (text before first indented line)
    - Repeated function signature (model re-states 'def ...')
    """
    # 0. Strip Qwen3 thinking block if present — take everything after </think>
    if '</think>' in text:
        text = text[text.find('</think>') + len('</think>'):].strip()

    # 1. Prefer content inside a markdown python code block
    if '```python' in text:
        start = text.find('```python') + 9
        end = text.find('```', start)
        text = text[start:end].strip() if end > start else text[start:].strip()
    elif '```' in text:
        start = text.find('```') + 3
        end = text.find('```', start)
        text = text[start:end].strip() if end > start else text[start:].strip()

    # 2. Strip any repeated 'def ...' signature the model may have emitted
    lines = text.split('\n')
    body_lines = []
    skip_def = False
    for line in lines:
        stripped = line.lstrip()
        if stripped.startswith('def ') and not body_lines:
            skip_def = True
            continue
        if skip_def and stripped.startswith('"""'):
            # Skip repeated docstring too
            in_doc = True
            while lines:
                l = lines.pop(0) if lines else ''
                if l.lstrip().endswith('"""') and in_doc and l.lstrip() != '"""':
                    break
                if l.count('"""') >= 2:
                    break
            skip_def = False
            continue
        skip_def = False
        body_lines.append(line)

    text = '\n'.join(body_lines)

    # 3. Drop leading non-code prose: find first indented line
    lines = text.split('\n')
    start_idx = 0
    for i, line in enumerate(lines):
        if line.startswith('    ') or line.startswith('\t'):
            start_idx = i
            break

    return '\n'.join(lines[start_idx:])


def make_sample_fn(temperature: float):
    """Return a JIT-compiled sampling function, defined once at startup.

    Defining @jax.jit inside generate_completion creates a new JIT object per
    call → recompilation on every problem. Defining it here and passing it in
    as sample_fn ensures the kernel is compiled once and reused.

    Logits are fully replicated P(None,None,None) — guaranteed by the
    with_sharding_constraint AllGather in model.py's output_head.
    """
    @jax.jit
    def _get_next(l, k):
        logit = l[0, -1, :].astype(jnp.float32)
        rng_new, subkey = jax.random.split(k)
        if temperature <= 0.0:
            next_tok = jnp.argmax(logit).astype(jnp.int32)
        else:
            next_tok = jax.random.categorical(
                subkey, logit / temperature).astype(jnp.int32)
        return next_tok, rng_new
    return _get_next


def generate_completion(
    fwd_fn,
    tok,
    config,
    prompt: str,
    sample_fn,
    max_new_tokens: int = 256,
    temperature: float = 0.2,
    fixed_max_len: int | None = None,
) -> str:
    """Generate a completion for `prompt` using greedy / temperature sampling.

    Args:
        fwd_fn: JIT-compiled forward function from make_forward_fn().
        tok: HuggingFace tokenizer.
        config: Qwen35Config.
        prompt: text prompt string.
        sample_fn: pre-compiled JIT from make_sample_fn() — passed in to avoid
            recompilation on every problem.
        max_new_tokens: max tokens to generate.
        temperature: sampling temperature (0 = greedy).
        fixed_max_len: if set, all caches use this length so the decode JIT
            compiles once regardless of prompt length variation.

    Returns:
        Generated text (decoded, without the prompt).
    """
    from jax_gpt.models.qwen35.cache import init_cache

    # Qwen3.5 is chat-tuned — wrap raw HumanEval prompt with chat template.
    # We do NOT force thinking mode: strip the trailing '<think>\n' that
    # add_generation_prompt=True appends, so the model sees a plain
    # '<|im_start|>assistant\n' prefix and generates freely.
    # (With thinking mode forced: model consistently generates <|im_end|> immediately.)
    rank = jax.process_index()
    chat = [{'role': 'user', 'content':
             'Complete the following Python function. '
             'Output ONLY the indented function body, no explanation, no markdown:\n\n'
             + prompt}]
    formatted = tok.apply_chat_template(
        chat, tokenize=False, add_generation_prompt=True)
    # Strip the <think>\n (or <think>) suffix added by the Qwen3 template.
    for _think_sfx in ('<think>\n', '<think>'):
        if formatted.endswith(_think_sfx):
            formatted = formatted[:-len(_think_sfx)]
            break
    input_ids = tok(formatted, return_tensors='np').input_ids[0]
    T = len(input_ids)
    # One-time debug: print last 200 chars of formatted prompt to confirm template
    if not hasattr(tok, '_debug_template_printed'):
        tok._debug_template_printed = True
        print(f"[DEBUG template tail] {repr(formatted[-200:])}", flush=True)

    # Stop tokens — use tokenizer to get correct IDs (extended vocab: 248044/248046)
    _stop_ids = {tok.eos_token_id,
                 tok.convert_tokens_to_ids('<|endoftext|>'),
                 tok.convert_tokens_to_ids('<|im_end|>')} - {None}

    # No right-padding — use exact prompt length to avoid EOS tokens polluting
    # the KV cache (right-padding with EOS causes model to predict EOS immediately
    # in decode since it attends back to the padded positions).
    # Per-length recompilation is handled by the GCS compilation cache.
    tokens = jnp.array(input_ids[None], dtype=jnp.int32)  # (1, T)

    # Allocate cache. Use a fixed max_len so the cache array shape is identical
    # across all problems → decode step compiles once regardless of prompt length.
    max_len = fixed_max_len if fixed_max_len is not None else T + max_new_tokens
    cache = init_cache(config, batch_size=1, max_len=max_len, dtype=jnp.bfloat16)

    # ----- Prefill -----
    logits, cache = fwd_fn(tokens, cache, False)  # (1, T, vocab)

    # ----- Decode loop -----
    generated_ids: list[int] = []
    rng = jax.random.key(0)

    # _get_next is passed in as a pre-compiled JIT to avoid recompilation
    # on every call to generate_completion (defining @jax.jit inside a
    # function creates a new JIT object per call → recompile each problem).

    for step in range(max_new_tokens):
        # All ranks must call the same JIT in lockstep (TP collective ops).
        next_id_jax, rng = sample_fn(logits, rng)
        next_id = int(next_id_jax)

        # All ranks print their first token to verify cross-rank agreement.
        if step == 0:
            print(f"  [DEBUG rank{rank} tok0] id={next_id} "
                  f"'{tok.decode([next_id])}'", flush=True)

        if next_id in _stop_ids:
            if not generated_ids:  # stopped on very first token
                print(f"  [DEBUG stop] first token is stop token id={next_id} "
                      f"'{tok.decode([next_id])}'", flush=True)
            break
        generated_ids.append(next_id)

        # Single-token decode step
        token_in = jnp.array([[next_id]], dtype=jnp.int32)  # (1, 1)
        logits, cache = fwd_fn(token_in, cache, True)

    raw = tok.decode(generated_ids, skip_special_tokens=False)  # keep special tokens for debug
    # </think> is token 151668 — a special token stripped by skip_special_tokens=True.
    # Find split point in token IDs directly, decode only the post-think suffix.
    _THINK_CLOSE_ID = tok.convert_tokens_to_ids('</think>')  # 248069 for Qwen3.5-397B
    if _THINK_CLOSE_ID in generated_ids:
        suffix_ids = generated_ids[generated_ids.index(_THINK_CLOSE_ID) + 1:]
        decoded_for_extraction = tok.decode(suffix_ids, skip_special_tokens=True)
    else:
        decoded_for_extraction = tok.decode(generated_ids, skip_special_tokens=True)
    # Warn if we hit the token limit before closing thinking
    hit_limit = len(generated_ids) == max_new_tokens
    if hit_limit and _THINK_CLOSE_ID not in generated_ids:
        print(f"  [WARN] hit max_new_tokens={max_new_tokens} without </think> — "
              f"thinking overflowed; increase --max-new-tokens", flush=True)
    return _extract_function_body(decoded_for_extraction), raw, generated_ids


# ---------------------------------------------------------------------------
# HumanEval problem loading and execution
# ---------------------------------------------------------------------------

def load_humaneval_problems() -> list[dict]:
    """Load HumanEval problems from the 'datasets' library."""
    try:
        from datasets import load_dataset
        ds = load_dataset('openai_humaneval', split='test')
        return list(ds)
    except Exception as e:
        print(f"Warning: could not load HumanEval from datasets ({e})")
        print("Falling back to a single dummy problem for pipeline testing.")
        return [_dummy_problem()]


def _dummy_problem() -> dict:
    """A trivially passing problem for smoke-testing without datasets."""
    return {
        'task_id': 'HumanEval/dummy',
        'prompt': (
            'def add(a: int, b: int) -> int:\n'
            '    """Return the sum of a and b.\n\n'
            '    >>> add(1, 2)\n    3\n    """\n'
        ),
        'test': (
            'def check(candidate):\n'
            '    assert candidate(1, 2) == 3\n'
            '    assert candidate(-1, 1) == 0\n'
        ),
        'entry_point': 'add',
    }


def evaluate_completion(problem: dict, completion: str) -> bool:
    """Run the HumanEval test harness; return True if all tests pass."""
    code = problem['prompt'] + completion + '\n' + problem['test']
    code += f'\ncheck({problem["entry_point"]})\n'

    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(code)
        tmp_path = f.name
    try:
        result = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True, text=True, timeout=10,
        )
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        return False
    finally:
        os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', default='/mnt/model',
                        help='Path to the model directory (for tokenizer + weights)')
    parser.add_argument('--output-dir', default='/tmp/humaneval_out')
    parser.add_argument('--n-problems', type=int, default=164)
    parser.add_argument('--max-new-tokens', type=int, default=256)
    parser.add_argument('--temperature', type=float, default=0.6)  # Qwen3 thinking mode: use 0.6
    parser.add_argument('--sharding', default='B', choices=['A', 'B'])
    parser.add_argument('--tp', type=int, default=None,
                        help='Tensor-parallel degree (default: all local chips)')
    parser.add_argument('--dp', type=int, default=1,
                        help='Data-parallel degree across JobSet replicas')
    # Config / weight flags
    parser.add_argument('--config', default='full',
                        choices=['mini', 'mid', 'mid_large', 'full'],
                        help='Model size config (default: full = 397B)')
    parser.add_argument('--tokenizer', default=None,
                        help='Tokenizer path or HF model name (defaults to --model-dir)')
    parser.add_argument('--random-weights', action='store_true',
                        help='Use random weights instead of loading from disk '
                             '(for pipeline smoke-testing)')
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Distributed init
    # ------------------------------------------------------------------
    rank, world_size = init_distributed()

    from jax_gpt.models.qwen35.config import Qwen35Config
    from jax_gpt.models.qwen35.sharding import (
        AXIS_RULES_A, AXIS_RULES_B, make_mesh,
    )

    axis_rules = AXIS_RULES_B if args.sharding == 'B' else AXIS_RULES_A
    cfg = getattr(Qwen35Config, args.config)()

    # ------------------------------------------------------------------
    # Build device mesh
    # ------------------------------------------------------------------
    tp = args.tp or jax.local_device_count()
    dp = args.dp
    total_devices = tp * dp

    assert jax.device_count() >= total_devices, (
        f"Need {total_devices} devices, have {jax.device_count()}"
    )

    devices = np.array(jax.devices()[:total_devices])
    if dp > 1:
        # 2D mesh: rows = dp replicas, cols = tp shards
        mesh = jax.sharding.Mesh(devices.reshape(dp, tp), ('dp', 'tp'))
    else:
        mesh = make_mesh(n_devices=tp)

    # DP rank: which data-parallel replica this process belongs to.
    # With tp=32 and 4 local devices per host, 8 processes form one TP group
    # → dp_rank=0 for all. With dp>1, each TP group is a separate replica.
    processes_per_dp_replica = tp // jax.local_device_count()
    dp_rank = rank // processes_per_dp_replica

    print(f"[rank {rank}] mesh={dict(mesh.shape)}  config={args.config}  "
          f"random_weights={args.random_weights}")

    # ------------------------------------------------------------------
    # Load tokenizer — from --tokenizer if given, else --model-dir
    # ------------------------------------------------------------------
    from transformers import AutoTokenizer
    tokenizer_src = args.tokenizer if args.tokenizer else args.model_dir
    tok = AutoTokenizer.from_pretrained(tokenizer_src)

    # ------------------------------------------------------------------
    # Load / init model weights
    # ------------------------------------------------------------------
    if args.random_weights:
        params = load_model_random(cfg, mesh, axis_rules)
    else:
        params = load_model_from_checkpoint(args.model_dir, cfg, mesh, axis_rules)

    # ------------------------------------------------------------------
    # Load problems and split across DP ranks (before warmup so we can
    # compute fixed_max_len for the cache shape)
    # ------------------------------------------------------------------
    all_problems = load_humaneval_problems()
    all_problems = all_problems[:args.n_problems]
    n_total = len(all_problems)

    problems_per_rank = math.ceil(n_total / dp)
    start = dp_rank * problems_per_rank
    end = min(start + problems_per_rank, n_total)
    my_problems = all_problems[start:end]

    print(f"[rank {rank}] dp_rank={dp_rank} Evaluating problems {start}..{end - 1} "
          f"({len(my_problems)} problems)")

    # Pre-tokenize with chat template to find max formatted prompt length.
    # fixed_max_len keeps cache shape identical across problems so decode
    # compiles only once. Prefill length varies per problem (no padding) —
    # GCS compilation cache handles per-length recompilation across runs.
    from jax_gpt.models.qwen35.cache import init_cache
    if my_problems:
        def _fmt_len(p):
            chat = [{'role': 'user', 'content':
                     'Complete the following Python function. '
                     'Output only the function body (the indented code), no explanation:\n\n'
                     + p['prompt']}]
            fmt = tok.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
            return len(tok(fmt, return_tensors='np').input_ids[0])
        max_prompt_len = max(_fmt_len(p) for p in my_problems)
    else:
        max_prompt_len = 256
    fixed_max_len = max_prompt_len + args.max_new_tokens
    print(f"[rank {rank}] max_formatted_prompt_len={max_prompt_len}  "
          f"fixed_max_len={fixed_max_len}")

    # ------------------------------------------------------------------
    # Build JIT'd forward function
    # ------------------------------------------------------------------
    print(f"[rank {rank}] Compiling forward pass (prefill + decode)...")
    fwd_fn = make_forward_fn(params, cfg, mesh, tp, axis_rules=axis_rules)

    # Warm-up with a short dummy prompt to trigger decode JIT compile.
    print(f"[rank {rank}] Warming up JIT (prefill + decode)...")
    t_compile = time.perf_counter()
    _dummy_ids = jnp.ones((1, 4), dtype=jnp.int32)
    _dummy_cache = init_cache(cfg, 1, fixed_max_len, dtype=jnp.bfloat16)
    _, _dc = fwd_fn(_dummy_ids, _dummy_cache, False)
    jax.effects_barrier()
    print(f"  Prefill JIT compile: {time.perf_counter() - t_compile:.1f}s")

    t_compile = time.perf_counter()
    _tok1 = jnp.ones((1, 1), dtype=jnp.int32)
    fwd_fn(_tok1, _dc, True)
    jax.effects_barrier()
    print(f"  Decode JIT compile:  {time.perf_counter() - t_compile:.1f}s")

    # Build sampling JIT once — reused across all problems.
    sample_fn = make_sample_fn(args.temperature)
    # Warm up sampling JIT with dummy logits.
    _dummy_logits = jnp.zeros((1, 1, cfg.vocab_size), dtype=jnp.bfloat16)
    _, _ = sample_fn(_dummy_logits, jax.random.key(0))
    jax.effects_barrier()
    print(f"  Sample JIT compiled.")

    # ------------------------------------------------------------------
    # Eval loop
    # ------------------------------------------------------------------
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    n_pass = 0

    for i, problem in enumerate(my_problems):
        global_idx = start + i
        t_start = time.perf_counter()

        completion, raw_text, token_ids = generate_completion(
            fwd_fn, tok, cfg,
            prompt=problem['prompt'],
            sample_fn=sample_fn,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            fixed_max_len=fixed_max_len,
        )
        elapsed = time.perf_counter() - t_start
        passed = evaluate_completion(problem, completion)
        n_pass += int(passed)

        result = {
            'task_id': problem['task_id'],
            'global_idx': global_idx,
            'passed': passed,
            'elapsed_s': round(elapsed, 2),
            'completion': completion[:200],  # truncate for readability
        }
        results.append(result)

        status = '✓' if passed else '✗'
        acc = n_pass / (i + 1)
        print(f"[rank {rank}] [{i + 1}/{len(my_problems)}] {status} "
              f"{problem['task_id']}  ({elapsed:.1f}s)  pass@1={acc:.3f}")
        # Debug: dump raw tokens + text for first 10 problems on rank 0 → GCS
        if i < 10 and rank == 0:
            print(f"  [DEBUG raw]        {repr(raw_text[:500])}")
            print(f"  [DEBUG completion] {repr(completion[:300])}")
            print(f"  [DEBUG token_ids] {token_ids[:50]}", flush=True)
        sys.stdout.flush()

    # ------------------------------------------------------------------
    # Write per-dp-rank results (only one process per DP replica writes)
    # ------------------------------------------------------------------
    # Use process rank 0 within each TP group as the writer for that dp_rank.
    is_dp_leader = (rank % processes_per_dp_replica == 0)
    if is_dp_leader:
        rank_file = output_dir / f'results_rank{dp_rank:02d}.json'
        with open(rank_file, 'w') as f:
            json.dump({'dp_rank': dp_rank, 'results': results,
                       'n_pass': n_pass, 'n_total': len(my_problems)}, f, indent=2)
        print(f"[rank {rank}] Wrote {rank_file}")

    # ------------------------------------------------------------------
    # DP rank 0 merges results (after barrier)
    # ------------------------------------------------------------------
    if dp > 1:
        jax.effects_barrier()
        _barrier_file_sync(output_dir, dp_rank, dp)

    if rank == 0:
        _merge_results(output_dir, dp, n_total)


# ---------------------------------------------------------------------------
# Barrier + merge helpers
# ---------------------------------------------------------------------------

def _barrier_file_sync(output_dir: Path, rank: int, dp: int, timeout: int = 600):
    """File-based barrier: rank 0 waits for all result files to appear.

    NOTE for GKE: all DP pods must mount a shared ReadWriteMany volume
    (e.g. Cloud Filestore) or use GCS FUSE so that each rank's result file
    is visible to rank 0. A ReadWriteOnce PVC won't work for dp > 1.
    """
    sentinel = output_dir / f'.done_rank{rank:02d}'
    sentinel.touch()

    if rank == 0:
        t0 = time.time()
        for r in range(dp):
            s = output_dir / f'.done_rank{r:02d}'
            while not s.exists():
                if time.time() - t0 > timeout:
                    print(f"WARNING: timed out waiting for rank {r}, merging anyway")
                    break
                time.sleep(2)


def _merge_results(output_dir: Path, dp: int, n_total: int):
    all_results = []
    total_pass = 0
    total_evaluated = 0

    for r in range(dp):
        rank_file = output_dir / f'results_rank{r:02d}.json'
        if not rank_file.exists():
            print(f"WARNING: missing {rank_file}")
            continue
        with open(rank_file) as f:
            data = json.load(f)
        all_results.extend(data['results'])
        total_pass += data['n_pass']
        total_evaluated += data['n_total']

    all_results.sort(key=lambda x: x['global_idx'])
    pass_at_1 = total_pass / total_evaluated if total_evaluated > 0 else 0.0

    summary = {
        'model': 'Qwen3.5-397B-A17B',
        'config': 'full',
        'benchmark': 'HumanEval',
        'n_problems': n_total,
        'n_evaluated': total_evaluated,
        'n_pass': total_pass,
        'pass_at_1': round(pass_at_1, 4),
        'results': all_results,
    }

    summary_file = output_dir / 'summary.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print('\n' + '=' * 60)
    print('HUMANEVAL RESULTS')
    print('=' * 60)
    print(f'  Problems:  {total_evaluated}/{n_total}')
    print(f'  Pass@1:    {total_pass}/{total_evaluated} = {pass_at_1:.3f}')
    print(f'  Summary:   {summary_file}')
    print('=' * 60)

    # Save to GCS so results survive pod termination
    gcs_dest = 'gs://sivaibhav-exp/qwen-profiles/humaneval/summary.json'
    try:
        import subprocess as _sp
        _sp.run(['gsutil', 'cp', str(summary_file), gcs_dest], check=True)
        print(f'  GCS copy:  {gcs_dest}')
    except Exception as e:
        print(f'  WARNING: GCS copy failed: {e}')


if __name__ == '__main__':
    main()
