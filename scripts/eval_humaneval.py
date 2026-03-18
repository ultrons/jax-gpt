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


def generate_completion(
    fwd_fn,
    tok,
    config,
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.2,
    fixed_max_len: int | None = None,
    prefill_bucket: int = 128,
) -> str:
    """Generate a completion for `prompt` using greedy / temperature sampling.

    Args:
        fwd_fn: JIT-compiled forward function from make_forward_fn().
        tok: HuggingFace tokenizer.
        config: Qwen35Config.
        prompt: text prompt string.
        max_new_tokens: max tokens to generate.
        temperature: sampling temperature (0 = greedy).
        fixed_max_len: if set, all caches use this length (avoids recompilation
            across problems with different prompt lengths).
        prefill_bucket: round prefill length up to nearest multiple of this value
            to reduce JIT recompilation (right-pad with eos token).

    Returns:
        Generated text (decoded, without the prompt).
    """
    from jax_gpt.models.qwen35.cache import init_cache

    input_ids = tok(prompt, return_tensors='np').input_ids[0]
    T = len(input_ids)

    # Bucket prefill length to limit recompilation across problems.
    # Right-pad with eos so we get the same shape for all prompts in the bucket.
    bucket_T = math.ceil(T / prefill_bucket) * prefill_bucket
    if bucket_T > T:
        pad_id = tok.eos_token_id if tok.eos_token_id is not None else 0
        input_ids = np.concatenate(
            [input_ids, np.full(bucket_T - T, pad_id, dtype=np.int32)])
    tokens = jnp.array(input_ids[None], dtype=jnp.int32)  # (1, bucket_T)

    # Allocate cache. Use a fixed max_len when provided so the cache array
    # shape is identical across all problems → decode step compiles once.
    max_len = fixed_max_len if fixed_max_len is not None else bucket_T + max_new_tokens
    cache = init_cache(config, batch_size=1, max_len=max_len, dtype=jnp.bfloat16)

    # ----- Prefill -----
    logits, cache = fwd_fn(tokens, cache, False)  # (1, bucket_T, vocab)

    # Grab the logit at the last *real* token position (not the padding tail).
    # We need to pass this position through JIT to avoid materializing the
    # full vocab-sharded logits on the host.
    last_real_pos = T - 1

    # ----- Decode loop -----
    generated_ids: list[int] = []
    rng = jax.random.key(0)

    # Sample next token inside JIT so sharded logits (vocab-dim sharded across
    # TP devices) are never materialized on host — avoids "non-addressable
    # device" error in multi-process JAX.
    # Use last_real_pos (not -1) because prefill output may have padding tail.
    if temperature <= 0.0:
        _get_next = jax.jit(
            lambda l, pos: jnp.argmax(l[0, pos, :]).astype(jnp.int32))
    else:
        @jax.jit
        def _get_next_with_temp(l, pos, k):
            rng_new, subkey = jax.random.split(k)
            next_tok = jax.random.categorical(
                subkey, (l[0, pos, :] / temperature).astype(jnp.float32))
            return next_tok.astype(jnp.int32), rng_new

    # Use -1 for decode steps (token_in is always length 1)
    _decode_pos = jnp.array(-1, dtype=jnp.int32)

    for step in range(max_new_tokens):
        pos = jnp.array(last_real_pos, dtype=jnp.int32) if step == 0 else _decode_pos
        if temperature <= 0.0:
            next_id = int(_get_next(logits, pos))
        else:
            next_id_jax, rng = _get_next_with_temp(logits, pos, rng)
            next_id = int(next_id_jax)

        if next_id == tok.eos_token_id:
            break
        generated_ids.append(next_id)

        # Single-token decode step
        token_in = jnp.array([[next_id]], dtype=jnp.int32)  # (1, 1)
        logits, cache = fwd_fn(token_in, cache, True)

    return tok.decode(generated_ids, skip_special_tokens=True)


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
    parser.add_argument('--temperature', type=float, default=0.2)
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

    # Pre-tokenize to find the max prompt length → fix cache shape so the
    # decode JIT only compiles once regardless of prompt length variation.
    PREFILL_BUCKET = 128
    from jax_gpt.models.qwen35.cache import init_cache
    if my_problems:
        prompt_lens = [
            len(tok(p['prompt'], return_tensors='np').input_ids[0])
            for p in my_problems
        ]
        max_prompt_len = max(prompt_lens)
    else:
        max_prompt_len = PREFILL_BUCKET
    max_prompt_bucketed = math.ceil(max_prompt_len / PREFILL_BUCKET) * PREFILL_BUCKET
    fixed_max_len = max_prompt_bucketed + args.max_new_tokens
    print(f"[rank {rank}] max_prompt_len={max_prompt_len}  "
          f"fixed_max_len={fixed_max_len}")

    # ------------------------------------------------------------------
    # Build JIT'd forward function
    # ------------------------------------------------------------------
    print(f"[rank {rank}] Compiling forward pass (prefill + decode)...")
    fwd_fn = make_forward_fn(params, cfg, mesh, tp, axis_rules=axis_rules)

    # Warm-up: use the real fixed_max_len so the cache shape matches inference.
    # Prefill with PREFILL_BUCKET tokens (smallest bucket) to trigger one compile.
    print(f"[rank {rank}] Warming up JIT (prefill)...")
    t_compile = time.perf_counter()
    _dummy_ids = jnp.ones((1, PREFILL_BUCKET), dtype=jnp.int32)
    _dummy_cache = init_cache(cfg, 1, fixed_max_len, dtype=jnp.bfloat16)
    _, _dc = fwd_fn(_dummy_ids, _dummy_cache, False)
    jax.effects_barrier()
    print(f"  Prefill JIT compile: {time.perf_counter() - t_compile:.1f}s")

    t_compile = time.perf_counter()
    _tok1 = jnp.ones((1, 1), dtype=jnp.int32)
    fwd_fn(_tok1, _dc, True)
    jax.effects_barrier()
    print(f"  Decode JIT compile:  {time.perf_counter() - t_compile:.1f}s")

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

        completion = generate_completion(
            fwd_fn, tok, cfg,
            prompt=problem['prompt'],
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            fixed_max_len=fixed_max_len,
            prefill_bucket=PREFILL_BUCKET,
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


if __name__ == '__main__':
    main()
