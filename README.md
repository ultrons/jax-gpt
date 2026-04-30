# jax-gpt

Minimal, optimized implementation of frontier language models in JAX, focused
on TPU inference.

## Models

| Model | Status | Highlights |
|---|---|---|
| **Qwen3.5-397B-A17B** (`jax_gpt/models/qwen35/`) | ✅ optimized inference | GQA + DeltaNet linear attention, 512-expert MoE with grouped matmul (`gmm_v2`), FP8 weights, paged KV cache, 728 TPS/chip on v7x at BS=4096 |
| **DeepSeek-V4-Pro** (`jax_gpt/models/dsv4/`) | 🚧 skeleton | MLA, hybrid CSA/HCA attention, 384-expert MoE with FP4 expert weights, 1M-context YaRN, mHC residuals. Indexer kernel + correctness wiring in progress. |
| **DeepSeek-V3** (`dsv3/`) | ✅ active training | MoE training stack with SparseCore Pallas kernels (`fused_moe_bwd/`), v337/v338 benchmarks on v7x 4×8×8. Top-level subdir (kept as a research workspace rather than a library module — owns its own `k8s/`, `docs/`, Dockerfiles). |
| **GPT-2** (`jax_gpt/models/gpt2/`) | ✅ reference impl | Reproduces HF activations to verify the JAX harness and trainer |

## Project layout

```
jax-gpt/
  jax_gpt/
    models/
      qwen35/      # model.py, moe.py, gqa.py, deltanet.py, fp8.py,
                   # paged_cache.py, pallas_deltanet.py, sharding.py,
                   # weight_loader.py, megablox/ (GMM kernel)
      dsv4/        # config.py, mla.py, hybrid_attention.py, indexer.py,
                   # moe.py, mhc.py, paged_cache.py (skeleton — see CLAUDE.md)
      gpt2/        # reference impl + lm-eval-harness adapter
    trainer/       # TrainConfig, TrainState, optimizer, sharding, metrics
    data/          # grain pipeline + offline tokenization
  tests/           # pytest tree, mirrors jax_gpt/ layout
  k8s/             # JobSet YAMLs for v7x / v5p clusters
  scripts/         # train.py, verify_generation.py, debug_pytorch.py
  Dockerfile.tpu   # TPU build (qwen35 / dsv4)
  dsv3/            # DeepSeek-V3 training workspace
    mini_dsv3/     # core model + training loop + eval scripts
    fused_moe_bwd/ # SparseCore Pallas backward kernels
    kernels_test/, mini_chunk_repro/, roofline/, specs/, tests/
    k8s/           # 431+ versioned JobSet YAMLs (v304b–v338)
    docs/          # engineering reports + slide decks
    Dockerfile.tpu, Dockerfile.test_k, Dockerfile.v322_fix
    LOGBOOK.md, dsv3_*_analysis.md
```

## Environment

```bash
source ~/xdb/.xprof/bin/activate    # always — base env lacks JAX
```

## Quick start — local

```bash
git clone git@github.com:<owner>/jax-gpt.git
cd jax-gpt
pip install -e .

# Run unit tests (no multi-host)
pytest tests/qwen35/ -k "not multidevice" -q
pytest tests/gpt2/                          -q
pytest tests/trainer/                       -q
```

## Test commands

```bash
pytest tests/qwen35/                              # all qwen35 tests
pytest tests/qwen35/test_moe.py                   # MoE only
pytest tests/qwen35/test_pallas_deltanet.py       # DeltaNet Pallas kernel
pytest tests/qwen35/test_fp8.py                   # FP8 correctness gate
```

Multi-device tests (`test_sharding_multidevice.py`, `test_inference_benchmark.py`)
need ≥4 chips and run on a real TPU cluster.

## Build & deploy

```bash
git commit -m "..."
sudo docker build -t gcr.io/tpu-vm-gke-testing/jax-gpt:<tag> .
sudo docker push gcr.io/tpu-vm-gke-testing/jax-gpt:<tag>
kubectl --context <cluster-context> apply -f k8s/<job>.yaml
```

GKE cluster contexts:

| Purpose | Context |
|---|---|
| Performance benchmarks (v7x) | `gke_cloud-tpu-multipod-dev_us-central1_ninja-v7x-64-spot` |
| Accuracy eval (v5p, 4×4×4) | `gke_cloud-tpu-multipod-dev_europe-west4_mlperf-rl-v5p-128` |

## Inference patterns specific to this repo

**Chunked prefill is mandatory for large-batch MoE.** MoE activations scale as
`x_sorted: (M/dp × k, D)` — at BS=4096, seqlen=1024 this is ~40 GB/device.
Always pass `--prefill-micro-batches N` (N≥16 for BS=4096).

**Decode micro-batches control HBM temp scaling.** HLO temporaries scale with
`page_B = batch_size / decode_micro_batches`, not total batch
(`page_B=1024` ≈ 21.5 GB temps, `page_B=2048` ≈ 42.6 GB). Increase
`--decode-micro-batches` if compile OOMs.

**Never use `donate_argnums` on the warmup call.** The buffer is consumed; a
subsequent timed call fails. Use a non-donating warmup variant followed by the
donating timed loop.

**FP8 correctness gate.** `tests/qwen35/test_fp8.py` is the canonical check
before any FP8 deployment.

## Pallas kernels

Custom Pallas kernels (Mosaic backend, TPU-only):

- `jax_gpt/models/qwen35/pallas_deltanet.py` — fused DeltaNet recurrent state.
- `jax_gpt/models/qwen35/megablox/` — `gmm_v2` grouped-matmul kernel for MoE.

All Mosaic constraints documented in `CLAUDE.md` apply (no scatter on JAX
arrays, no bool reshape, no VMEM trailing-1-dim, no `@jax.jit` inside
`shard_map`, etc.).

## Related repos

- **`perfsim`** — performance simulator (per-GEMM roofline + parallelism
  optimizer + measurement harness). Used to predict TPS / TPOT / HBM for
  jax-gpt's models before launching cluster jobs.
- **`xla-shell`** — interactive XLA / Pallas / Mosaic MLIR debugger and
  profile analyzer. Used to diagnose post-launch perf regressions.

## Status

- **Qwen3.5 inference**: 728 TPS/chip at BS=4096 on v7x (v101). Documented
  optimization history in `CLAUDE.md`.
- **DSv4-Pro**: skeleton in `jax_gpt/models/dsv4/`. mHC residual formula
  transcribed from arxiv 2512.24880; per-layer `compress_ratios` schedule
  from V4-Pro `config.json`. Indexer Pallas kernel + correctness wiring
  pending.
- **Training loop** (`scripts/train.py`): stub — Phase 2 of the project plan.
