# jax-gpt — Multi-model JAX/TPU repo

> **Audience for this file:** AI coding assistants (Claude Code, etc.) and
> contributors who want the operating model of this repo in <100 lines.
> Code-level docstrings cover the rest.

See global `~/.claude/CLAUDE.md` for JAX/TPU/Pallas platform rules
(HBM management, Mosaic constraints, AOT compile gates, etc.). This file
covers jax-gpt-specific conventions only.

## Models in this repo

| Model | Path | State |
|---|---|---|
| Qwen3.5-397B-A17B | `jax_gpt/models/qwen35/` | Production inference. 728 TPS/chip on v7x at BS=4096 (v101). |
| DeepSeek-V4-Pro | `jax_gpt/models/dsv4/` | Skeleton + indexer Pallas score kernel. mla.py / hybrid_attention.py / moe.py / mhc.py / paged_cache.py are still `NotImplementedError` stubs. |
| DeepSeek-V3 | `jax_gpt/models/dsv3/` | Production training. `train.py` (507 lines) drives the v304b–v338 experiment series on v7x 4×8×8. |
| GPT-2 | `jax_gpt/models/gpt2/` | Reference impl + lm-eval-harness adapter. |

## Self-contained model layout

Each model owns its Pallas kernels in a local `kernels/` subdir, even at
the cost of cross-model duplication. The two `gmm_v2.py` files (qwen35:
1150 lines, dsv3: 1270 lines) have already diverged in practice; trying
to share them is a fight against research-code reality.

```
jax_gpt/models/<model>/
  *.py                          # model + train + eval scripts
  kernels/                      # Pallas kernels owned by this model
    *.py
    [subdirs e.g. megablox/, fused_moe_bwd/]
```

Tests mirror this structure under `tests/<model>/`.

Workspace artifacts go to canonical jax-gpt locations:
- `k8s/<model>/` for JobSet YAMLs (dsv3 has 431+ versioned YAMLs)
- `docs/<model>/` for engineering reports + diagrams
- `scripts/<model>/` for shell scripts (dt.sh, watch-pools.sh)
- `notebooks/<model>/` for Jupyter
- `docker/Dockerfile.<model>.<variant>` for per-model Docker variants
- `research/<model>/` for repros, exploratory probes, roofline analyses

## Test commands

```bash
# Tests that don't need TPU runtime
pytest tests/qwen35/test_moe.py tests/qwen35/test_gqa.py tests/dsv4/ -q

# All qwen35 tests (Pallas tests will skip without TPU)
pytest tests/qwen35/

# Specific kernel correctness gates
pytest tests/qwen35/test_pallas_deltanet.py    # DeltaNet recurrent kernel
pytest tests/qwen35/test_fp8.py                # FP8 dequant — canonical gate
pytest tests/dsv4/test_indexer.py              # DSv4 indexer score + AOT gate
```

Multi-device tests (`test_sharding_multidevice.py`,
`test_inference_benchmark.py`) need ≥4 chips and run on a real TPU
cluster. There is a known pre-existing flax/nnx ↔ jax version mismatch
that makes some `tests/{gpt2,trainer}/` files error during collection;
it's a deps issue, not a layout issue.

## Build & deploy

For qwen35 / dsv4 (root Dockerfile):
```bash
git commit -m "..."
sudo docker build -t gcr.io/tpu-vm-gke-testing/jax-gpt:<tag> -f Dockerfile.tpu .
sudo docker push gcr.io/tpu-vm-gke-testing/jax-gpt:<tag>
kubectl --context <ctx> apply -f k8s/<job>.yaml
```

For DSv3 (variant Dockerfiles under `docker/`):
```bash
sudo docker build -t gcr.io/tpu-vm-gke-testing/jax-gpt-dsv3:<tag> \
  -f docker/Dockerfile.dsv3.tpu .
sudo docker push gcr.io/tpu-vm-gke-testing/jax-gpt-dsv3:<tag>
kubectl --context <ctx> apply -f k8s/dsv3/<job>.yaml
```

After the layout normalization, train invocations inside k8s YAMLs now
use the package import path:
```
python -m jax_gpt.models.dsv3.train --config <yaml>
```
(Older YAMLs that ran `python mini_dsv3/train.py` need updating before
re-submission.)

GKE cluster contexts:

| Purpose | Context |
|---|---|
| Performance benchmarks (v7x) | `gke_cloud-tpu-multipod-dev_us-central1_ninja-v7x-64-spot` |
| Accuracy eval (v5p, 4×4×4) | `gke_cloud-tpu-multipod-dev_europe-west4_mlperf-rl-v5p-128` |

Always use `tee -a <log>` when streaming `kubectl logs` so multi-attempt
crash history is preserved.

## Inference patterns specific to this repo

**Chunked prefill is mandatory for large-batch MoE.** MoE activations
scale as `x_sorted: (M/dp × k, D)` — at BS=4096, seqlen=1024 this is
~40 GB/device. Always pass `--prefill-micro-batches N` (N ≥ 16 for
BS=4096).

**Decode micro-batches control HBM temp scaling.** HLO temporaries scale
with `page_B = batch_size / decode_micro_batches` (page_B=1024 ≈ 21.5 GB
temps; page_B=2048 ≈ 42.6 GB). Increase `--decode-micro-batches` if
compile OOMs.

**Never use `donate_argnums` on the warmup call.** The buffer is consumed;
a subsequent timed call fails. Use a non-donating warmup variant followed
by a donating timed loop.

**FP8 correctness gate.** `tests/qwen35/test_fp8.py` is the canonical
check before any FP8 deployment.

## Pallas kernels in this repo

| Kernel | Path |
|---|---|
| `pallas_deltanet` (DeltaNet recurrent step) | `qwen35/kernels/pallas_deltanet.py` |
| `gmm_v2` inference (grouped matmul) | `qwen35/kernels/megablox/gmm_v2.py` |
| `gmm_v2`, `gmm_v2_train` (DSv3 variants) | `dsv3/kernels/` |
| `gather_reduce_*`, `sort_activations` | `dsv3/kernels/` |
| `fused_moe_bwd/` (SparseCore backward, multi-version) | `dsv3/kernels/fused_moe_bwd/` |
| `pallas_indexer` (DSv4 CSA score kernel) | `dsv4/kernels/pallas_indexer.py` |

All Mosaic constraints from global CLAUDE.md apply: no scatter_p on JAX
arrays, no bool reshape, no VMEM trailing-1-dim, no `@jax.jit` inside
`shard_map`. Run the AOT compile gate
(`jax.experimental.topologies.get_topology_desc("tpu7x:4x4x4")`) locally
before any cluster job — see `dsv4/kernels/pallas_indexer.py:aot_compile_check`
for the canonical pattern.

## Known issues

- **flax/nnx ↔ jax version mismatch**: `from jax.core import MainTrace`
  fails — breaks collection of `tests/gpt2/`, `tests/trainer/`, and
  `tests/dsv3/test_correctness.py`. Bump or pin compatible versions.
- **`dsv3/tpu_inference/`** is gitignored / vendored on disk only. Some
  `dsv3/kernels/fused_moe_bwd/test_*.py` files import from it; they
  won't work on a fresh clone until tpu_inference is set up locally.
- **Pre-existing TPU-required Pallas tests in `tests/qwen35/`** fail on
  CPU-only environments instead of skipping cleanly.
