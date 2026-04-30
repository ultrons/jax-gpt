# dsv3 — DeepSeek-V3 Training on TPU v7x

## Repo layout

```
dsv3/
  mini_dsv3/         # model + training loop (model.py, train.py, eval_logits.py)
  fused_moe_bwd/     # Pallas backward kernels; current: backward_kernel_v4.py
  tpu_inference/kernels/fused_moe/v1/kernel.py  # Pallas forward kernel
  tests/             # model-level integration tests (test_correctness.py)
  k8s/               # JobSet YAMLs for cluster runs
  Dockerfile.tpu     # training image
```

## Environment

Always activate the xprof venv before any dev or test work:
```bash
source ~/xdb/.xprof/bin/activate
```
The base env lacks JAX; kernel tests fail without this.

## Test commands

### Kernel unit tests (local, no cluster)
```bash
cd ~/ml-experiments/dsv3/fused_moe_bwd
python test_bwd_v3.py          # current canonical backward kernel test — 3 stages
```
Each stage prints `[PASS]`/`[FAIL]` + max_diff. Exit 0 = all stages pass.

When adding a new kernel version `vN`: write `backward_kernel_vN.py` AND `test_bwd_vN.py`
together. The test must cover all 3 stages and run locally without a cluster.

### Model integration test
```bash
cd ~/ml-experiments/dsv3
python tests/test_correctness.py
```

### Test ladder (must pass in order — never skip levels)
1. EP=1 mini config — backward math only, no ICI DMA
2. EP=4 local — A2A scatter/gather paths
3. Full-scale cluster (EP=16, FSDP=16 on bodaborg 4x4x16)

**Mini configs hide FSDP bugs** because F_shard = F_full when FSDP=1. Always test at
scale before declaring correctness.

### What "passing" means
- Kernel: `max_diff < 1e-2` vs `jax.vjp` reference in bf16; all 3 stages exit 0
- Model: finite decreasing loss from step 1 (no NaN); loss within 0.1% of JAX baseline at step 100

## Current active kernel

`fused_moe_bwd/backward_kernel_v4.py` (backward) + `tpu_inference/kernels/fused_moe/v1/kernel.py` (forward).  
Activated via `--moe_backend=fused_ep_moe_v4`.

## Build & deploy

This IS a git repo. Always commit before building:
```bash
cd ~/ml-experiments
git add dsv3/ && git commit -m "..."
sudo docker build -f dsv3/Dockerfile.tpu -t gcr.io/tpu-vm-gke-testing/mini-dsv3:<tag> dsv3/
sudo docker push gcr.io/tpu-vm-gke-testing/mini-dsv3:<tag>
kubectl --context gke_cloud-tpu-multipod-dev_us-central1_bodaborg-super-alpha-cluster \
  -n poc-dev apply -f ~/ml-experiments/dsv3/k8s/<yaml>
```

**JobSet name limit:** ≤ 20 chars (poc-dev Kueue webhook adds 29-char prefix → 49 total).

## Working topology (4x4x16 on bodaborg)

Cluster context: `gke_cloud-tpu-multipod-dev_us-central1_bodaborg-super-alpha-cluster`, namespace `poc-dev`.
- FSDP=16, EP=16, TP=2
- `podset-slice-size: "16"` (four 4×4×4 sub-slices; do NOT add `exclusive-topology` annotation)
- F_shard = 2048/16 = 128 — kernel-aligned; no weight all-gather needed (`needs_gather=False`)
- GBS=256: T_fsdp=65536 (SC row limit, no chunking); GBS=512+: SC-free for Pallas v4

## Pallas / Mosaic constraints

See global `~/.claude/CLAUDE.md` for the full catalogue (zero-init VMEM, remat+A2A,
SMEM 1-D, SC row limit, no jit-in-shard_map, scatter_p, bool reshape, trailing-1-dim).

## Diagnosing common failures

| Symptom | First check |
|---|---|
| NaN from step 1 | VJP structure (conjugate collectives), not numerics |
| SC gather XLA error | Run `--moe_backend=jax` first — if same error, bug is in model.py JAX graph |
| Compilation hang (60+ min) | Nested `@jax.jit` inside `shard_map` |
| Wrong grads at full scale, correct at mini | F_shard vs F_full; FSDP=1 hides the bug |
| Wrong grads at EP>1 | Check SC BF16 row gather; T_fsdp may exceed 65536 |
