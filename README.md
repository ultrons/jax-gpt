# jax-gpt

Minimal, optimized implementation of frontier language models in JAX, focused
on TPU inference.

> **Status:** open exploration / personal research repo. Not packaged for
> external use — APIs are unstable, performance numbers are
> hardware-specific, and several pieces (training loop, DSv4 wiring) are
> in-flight. Issues and PRs from outside collaborators are not actively
> solicited at this stage. Treat the code as reading material rather than
> a library.

## Models

| Model | Status | Highlights |
|---|---|---|
| **Qwen3.5-397B-A17B** (`jax_gpt/models/qwen35/`) | ✅ optimized inference | GQA + DeltaNet linear attention, 512-expert MoE with grouped matmul (`gmm_v2`), FP8 weights, paged KV cache, 728 TPS/chip on v7x at BS=4096 |
| **DeepSeek-V4-Pro** (`jax_gpt/models/dsv4/`) | 🚧 skeleton | MLA, hybrid CSA/HCA attention, 384-expert MoE with FP4 expert weights, 1M-context YaRN, mHC residuals. Indexer kernel + correctness wiring in progress. |
| **DeepSeek-V3** (`jax_gpt/models/dsv3/`) | ✅ active training | MoE training stack with SparseCore Pallas kernels (`kernels/fused_moe_bwd/`), v337/v338 benchmarks on v7x 4×8×8. Self-contained model dir — owns its kernels in `kernels/`. |
| **GPT-2** (`jax_gpt/models/gpt2/`) | ✅ reference impl | Reproduces HF activations to verify the JAX harness and trainer |

## Project layout

Each model lives under `jax_gpt/models/<name>/` and is **self-contained** —
its Pallas kernels live in a local `kernels/` subdirectory, even at the
cost of cross-model duplication. This matches the reality that kernels
co-evolve with their model (DSv3 and Qwen3.5 `gmm_v2` have already
diverged in practice). See the `## Pallas kernels` section below.

```
jax-gpt/
  jax_gpt/
    models/
      qwen35/      # model.py, moe.py, gqa.py, deltanet.py, fp8.py,
                   # paged_cache.py, sharding.py, weight_loader.py
        kernels/   # pallas_deltanet.py, megablox/ (gmm_v2)
      dsv3/        # model.py, train.py, load_weights.py, eval_*.py,
                   # compare_layers.py, test_fp8_dequant.py
        kernels/   # gmm_v2.py, gather_reduce_*.py, sort_activations.py,
                   # helpers.py, gather_reduce_standalone.py
                   # fused_moe_bwd/ (SparseCore backward kernel suite)
      dsv4/        # config.py, mla.py, hybrid_attention.py, indexer.py,
                   # moe.py, mhc.py, paged_cache.py, pallas_indexer.py
                   # (skeleton + indexer kernel; see CLAUDE.md)
      gpt2/        # reference impl + lm-eval-harness adapter
    trainer/       # TrainConfig, TrainState, optimizer, sharding, metrics
    data/          # grain pipeline + offline tokenization
  tests/
    qwen35/, dsv4/, gpt2/, trainer/
    dsv3/          # dsv3 unit + correctness tests + Pallas probe scripts
  k8s/
    qwen35-*.yaml  # qwen35 / dsv4 JobSet YAMLs
    dsv3/          # 431+ versioned JobSet YAMLs (v304b–v338)
  docs/
    dsv3/          # engineering reports, sharding analyses, *.png diagrams
  scripts/
    dsv3/          # dt.sh, watch-pools.sh
  notebooks/
    dsv3/          # pallas_moe_kernel_workshop.ipynb
  research/
    dsv3/          # mini_chunk_repro/, roofline/ — exploratory
  docker/
    Dockerfile.dsv3.{tpu,test_k,test_k_fix,tpu.nightly,v322_fix}
  Dockerfile.tpu   # qwen35 / dsv4 TPU build (root)
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

Each model owns its Pallas kernels in `jax_gpt/models/<model>/kernels/`.
Cross-model duplication is preferred over a shared kernels package — the
two `gmm_v2.py` files have already diverged because each model needs
slightly different shapes / dtypes / tiling.

| Kernel | Location | Notes |
|---|---|---|
| `gmm_v2` (grouped matmul, inference) | `qwen35/kernels/megablox/` | Used by qwen35 MoE decode |
| `pallas_deltanet` (linear attn) | `qwen35/kernels/` | DeltaNet recurrent step |
| `gmm_v2`, `gmm_v2_train` | `dsv3/kernels/` | DSv3 forward + training-specific variant |
| `gather_reduce_pallas`, `gather_reduce_sc`, `sort_activations` | `dsv3/kernels/` | DSv3 MoE dispatch helpers |
| `fused_moe_bwd/` (multi-version) | `dsv3/kernels/fused_moe_bwd/` | SparseCore backward suite (`backward_kernel_v3`, `_v4`, plus benches/tests) |
| `pallas_indexer` (CSA top-k score) | `dsv4/` | DSv4 indexer prototype |

All Mosaic constraints documented in `CLAUDE.md` apply (no scatter on JAX
arrays, no bool reshape, no VMEM trailing-1-dim, no `@jax.jit` inside
`shard_map`, etc.). Run the AOT compile gate (`jax.experimental.topologies`)
locally before any cluster job — see `dsv4/pallas_indexer.py:aot_compile_check`
for the canonical pattern.

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
  from V4-Pro `config.json`. Indexer Pallas score kernel landed
  (`pallas_indexer.py`); correctness wiring + C4-attend stage pending.
- **DSv3**: model lifted into `jax_gpt/models/dsv3/` from the imported
  research workspace. SparseCore backward kernel suite at
  `kernels/fused_moe_bwd/`. v337/v338 benchmarks pending re-run after
  the layout normalization.
- **Training loop** (`scripts/train.py`): stub — Phase 2 of the project plan.
