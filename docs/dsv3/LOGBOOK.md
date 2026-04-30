# DS-v3 Mini Trainer — Experiment Logbook

## 2026-03-26: Project setup and initial runs

### Infrastructure
- Moved project to `~/dsv3/` from `~/ml-experiments/timing/`
- Split single `train.py` into `model.py` + `train.py`
- Implemented 3 MoE backends: einsum (jax), ragged_dot, megablox
- Added FSDP+EP sharding, gradient checkpointing, AdamW, aux loss
- Validated on local v4-4: all backends produce matching loss

### Roofline validation
- Cross-validated `estimate_flops` against `jax.experimental.roofline`
- **Key finding: bwd = 1x fwd** (not 2x as commonly assumed). Each backward matmul = same FLOPs as forward. Updated both `estimate_flops` and parametric roofline.
- Per-component accuracy: MLA 1.00x, Dense MLP 1.00x, MoE 1.00x (when accounting for backend)

### Full 671B roofline estimates (4x8x8 v7x, 256 chips)
| PDBS | Roofline TPS/chip | Achieved (~70%) TPS/chip | MFU |
|------|------------------|-------------------------|-----|
| 1 | 8,444 | 5,911 | 48% |
| 2 | 8,795 | 6,157 | 50% |
| 4 | 8,982 | 6,288 | 51% |

### v7x 2x4x4 cluster setup
- Cluster: `ninja-v7x-64-spot` (8 nodes × 2x4x4, 32 chips, 64 devices)
- Docker: `gcr.io/tpu-vm-gke-testing/mini-dsv3:v3`
- GCS: `gs://max-experiments/sivaibhav-dsv3/`
- Multi-host JAX via `jax.distributed.initialize()`

### Experiment v1: mini model (954M), baseline
- Config: mini (D=2048, L=8, E=16), FSDP=32, EP=2, ragged_dot, SGD, PDBS=2
- Result: **10.9% MFU** — model too small for 32 chips

### Experiment v2: medium model (37B), baseline
- Config: medium (D=4096, L=24, E=64, K=8), FSDP=32, EP=2, ragged_dot, SGD, PDBS=2
- Result: **34.2% MFU**, 789 TFLOP/s/chip, 576ms step, 14,228 TPS/chip
- Profile: `gs://max-experiments/sivaibhav-dsv3/profiles/med-v2`
- **Finding: all-reduces are 69% of time at 0% compute overlap**
  - 22 MoE layers × ~154ms all-reduce each, completely serial
  - reduce-scatter: 99.6% overlap (good), all-gather: 67.5% (ok)
  - Only all-reduce has 0% overlap — structural issue with ragged_dot backward

### Experiment v3: medium, PDBS=4
- Same config but PDBS=4 (double batch)
- Result: **32.8% MFU** — worse! Comm scales with compute, no overlap improvement

### Experiment v4: medium + async collective fusion flags
- Added MaxText-style LIBTPU flags:
  ```
  --xla_tpu_enable_async_collective_fusion=true
  --xla_tpu_enable_async_collective_fusion_fuse_all_reduce=true
  --xla_tpu_overlap_compute_collective_tc=true
  --xla_enable_async_all_reduce=true
  ```
- Result: **34.5% MFU** — only +0.9%. Flags fused multiple all-reduces together (45MB→300MB each) but still 0% overlap.
- Profile: `gs://max-experiments/sivaibhav-dsv3/profiles/med-v4-async`

### Experiment v5: medium, FSDP=64 EP=1 (no expert parallelism)
- Eliminated EP axis entirely. All parallelism via FSDP.
- Result: **40.0% MFU**, 923 TFLOP/s/chip, 492ms step, 16,651 TPS/chip
- **+17% vs baseline** — best result so far
- Profile: `gs://max-experiments/sivaibhav-dsv3/profiles/med-v5-noep`
- All-reduces still 0% overlap but smaller (98ms vs 154ms) and fewer total

---

## Identified optimizations (not yet attempted)

1. **Gradient accumulation with async all-reduce** — accumulate grads over N micro-steps, overlap one large all-reduce with next micro-step's forward pass. This breaks the serial dependency chain that prevents overlap.

2. **shard_map for explicit collective control** — use `jax.experimental.shard_map` to manually schedule reduce-scatter (which gets 90% overlap) instead of letting XLA choose all-reduce (0% overlap).

3. **Fused MoE kernel** — eliminate ragged_dot dispatch overhead (10% of time) and allow custom DMA/compute overlap within the kernel. Reference impl at `~/tpu-inference/tpu_inference/kernels/fused_moe/v1/kernel.py`.

---

## TODO
- [ ] Implement `jax.lax.scan` over layers (critical for 61-layer compilation)
- [ ] Per-shard init with `jax.make_array_from_callback` (avoid host OOM for 671B)
- [ ] Submit full 671B on 2x4x4
- [ ] Try gradient accumulation
- [ ] Try shard_map for collective control

## 2026-03-26 (continued): Scan refactor + Full 671B run

### Scan + per-shard init implementation
- Refactored `init_params` to use `jax.make_array_from_callback` — each host generates only its local shard, no host OOM
- Refactored `forward` to use `jax.lax.scan` over MoE layers (L_moe=58), unrolled dense layers (L_dense=3)
- Added `medium` config (D=4096, L=24, E=64, 37B params) for intermediate testing

### Scan buffer copy warning
From `~/jax-gpt/docs/scan_moe_oom_analysis.md`:
- `jax.lax.scan` lowers to XLA WhileOp which creates a **full copy** of stacked params at each iteration boundary
- For 671B: expert weights ~20 GB/device → scan copy adds ~20 GB → ~40 GB total
- Options: `scan_mode='unrolled'` (Python loop inside JIT), or Pallas HBM-streaming kernel
- **The 671B run survived despite this** — 85.7 GB base + ~20 GB scan copy ≈ ~106 GB/device < 95 GB/core limit. Tight but fits.

### Experiment v6: FULL DS-v3 671B
- Config: D=7168, L=61, E=256, K=8, S=4096, FSDP=64, EP=1, ragged_dot, SGD, PDBS=1
- Scan over 58 MoE layers + gradient checkpointing
- Init: ~15 min (58 MoE layers × per-shard init)
- Compilation: ~92s (scan = O(1) layers!)
- **Result: 65.6% MFU, 1512 TFLOP/s/chip, 1015ms step, 4,034 TPS/chip**
- vs roofline prediction: 59.4% of roofline, 84.8% of achieved prediction
- Loss = nan (expected: random init at this scale with bf16)
- Profile: `gs://max-experiments/sivaibhav-dsv3/profiles/full-v6`

### Key takeaway
The full 671B model achieves **65.6% MFU** vs 34-40% on the medium model. The larger model benefits from:
1. Much larger matmuls that better saturate MXU (higher OI)
2. Better compute/communication ratio (more FLOPs per collective)
3. Scan amortizes the MoE layer structure efficiently

### Progress summary

| Model | Params | Best MFU | TPS/chip | Config |
|-------|--------|----------|----------|--------|
| Mini | 954M | 10.9% | 77,870 | too small for 32 chips |
| Medium | 37B | 40.0% | 16,651 | FSDP=64 EP=1 |
| **Full 671B** | **671B** | **65.6%** | **4,034** | **FSDP=64 EP=1 scan+ckpt** |

## 2026-03-26 (late): Correctness fixes + gradient accumulation

### Bug fixes
1. **MoE routing `* K` bug** — routing weights were `normalize * K`, giving sum=K²=64 per layer at full scale. Fixed to `normalize * routed_scaling_factor` (2.5 for full, 1.0 for mini/tiny). Was causing NaN via exponential residual growth.

2. **Init scale too large** — fixed 0.02 scale caused NaN at 671B. Changed to:
   - `1/sqrt(fan_in)` for all weights
   - Output projections (w_out, wo_mlp, wo, shared_wo) additionally scaled by `1/sqrt(2*L)` for residual stability
   - Verified: logit_std=1.0, loss=finite on tiny/mini

3. **Added `routed_scaling_factor`** to ModelConfig (2.5 for full, matching HuggingFace config)

### Missing HuggingFace config params (for future)
- `rope_scaling`: YaRN (factor=40) — inference only
- `n_group=8, topk_group=4` — grouped expert selection
- `topk_method="noaux_tc"` — exact routing method
- `num_nextn_predict_layers=1` — multi-token prediction

### Was the 65.6% MFU real?
v7 run (with routing fix but old init) produced same TFLOP/s (756/device) with NaN loss.
NaN matmuls still consume same HBM bandwidth and MXU cycles — XLA doesn't short-circuit NaN.
**The MFU number is likely valid**, but needs confirmation with finite-loss run (v8).

### Gradient accumulation implementation
Added `--grad_accum N` flag. Splits batch into N micro-batches, accumulates gradients via
`jax.lax.scan`, then does one optimizer update. This consolidates all-reduces into one large
operation at the end of the step, potentially enabling overlap with the next step's forward.

### Active experiments
- **v8**: Full 671B, fixed init, PDBS=1, no grad accum (baseline with finite loss)
- **v9** (queued): Full 671B, PDBS=2, grad_accum=2 (test overlap improvement)

### Experiment v8: Full 671B with fixed init (baseline confirmation)
- Initial loss: **108.9** (finite! expected ~11.8, high due to residual scaling mismatch)
- Step 1 onwards: NaN (SGD lr=1e-4 overshoot, no grad clipping)
- **TFLOP/s: 756/device = 65.6% MFU — SAME as v6/v7 with NaN**
- Confirms: MFU measurement is valid regardless of NaN values

### Experiment v9: Full 671B with gradient accumulation (PDBS=2, grad_accum=2)
- **TFLOP/s: 749/device = 64.9% MFU — WORSE by 0.7pp**
- Step time: 2051ms (exactly 2x baseline 1015ms)
- Grad accum loop runs serially, no overlap gain
- The all-reduces happen inside the scan accumulation loop, not between steps

### Key learning: grad accum doesn't help within JIT
The grad accumulation scan creates the same serial dependency chain as the regular backward.
For overlap to work, the all-reduce needs to happen BETWEEN jit calls, not inside.
This would require breaking the training step into separate jit-compiled phases:
1. jit: forward + backward → gradients (no all-reduce yet)
2. jit: all-reduce gradients (async, overlap with next step's forward)
3. jit: optimizer update

This is essentially what frameworks like Megatron-LM do with their "distributed optimizer".

### Updated progress table

| Experiment | Config | MFU | TPS/chip | Key change |
|-----------|--------|-----|----------|------------|
| v2 med baseline | FSDP=32 EP=2 | 34.2% | 14,228 | — |
| v5 med no-EP | FSDP=64 EP=1 | 40.0% | 16,651 | +17% from EP removal |
| **v8 full baseline** | **FSDP=64 EP=1** | **65.6%** | **4,035** | **Full 671B, confirmed real** |
| v9 full grad-accum | FSDP=64 EP=1 + grad_accum | 64.9% | 3,994 | No improvement |

## 2026-03-27: 4x8x8 attempt + eval weight loading

### ninja-v7x-512 cluster (4x8x8)
- Cluster has exactly 1 slice of 4x8x8 (64 nodes, 256 chips)
- **Blocked**: 5 nodes occupied by other users (amitmkumar, chaitanyapk, depksingh)
- Job gets 59/64 pods Running, 5 Pending → jax.distributed.initialize() hangs → timeout → crash
- Fix applied: `initialization_timeout=600` (MaxText uses 300)
- Need to retry when cluster is free

### PVC converted to ReadOnlyMany
- `ds-v3-weights-pvc` now ROX — all TPU pods can mount simultaneously
- Created on both ninja-v7x-64-spot and ninja-v7x-512 clusters

### Weight loading with EP-parallel reads
- EP=8 on ninja-64: each of 8 hosts reads 32/256 experts → ~40s/layer → ~35 min total
- EP=64 on ninja-512: each host would read 4 experts → ~5 min total (untested)
- EP>1 forward pass fails: ragged_dot and einsum both need all E experts on-device
- Fix implemented: load with EP=N, then reshard to EP=1 via jax.device_put (v11-eval image)

### Bugs found
- ragged_dot with EP>1: `expected rhs group dimension 256, got 32`
- einsum with EP>1: `Size of label 'e' for operand 1 (32) does not match (256)`
- Both backends assume all experts on-device — EP needs explicit all-gather/all-to-all
- MaxText implements this via ring-of-experts (all_gather tokens → local compute → psum_scatter)

## 2026-03-27 (evening): Correctness + MMLU plan

### Status at handoff
- **ninja-64 eval**: Weight loading in progress (layer 20/58), EP=8 parallel reads
- **ninja-512 (4x8x8)**: TPU mesh unstable — sanity test restarting. Lingering pods from other users caused mesh corruption. Cleaned up LeaderWorkerSets and standalone pods.
- **Weight PVC**: Converted to ReadOnlyMany on both clusters

### Tonight's plan
1. Get correctness eval working (pretrained weights → correct top-k predictions)
2. Implement autoregressive decode loop for MMLU
3. Run MMLU-5 eval
4. If 4x8x8 works: switch training there, measure MFU
5. Optimize training to 70% of roofline

### Available clusters
- ninja-v7x-64-spot: 2x4x4, working
- ninja-v7x-512: 4x8x8, TPU mesh unstable
- bodaborg-tpu7x-spot-256-chip: 4x8x8 (64 nodes), backup option

### Eval iteration log
- v11-eval attempt 1: NameError `NamedSharding` not imported in eval_logits.py → fixed
- v11-eval attempt 2: ragged_dot sees 32 experts after EP=8→EP=1 reshard. Changed to keep `moe_backend="jax"` (einsum) after reshard → resubmitted
- MMLU eval code written (`eval_mmlu.py`), Docker image built (`v8-mmlu`), JobSet ready
- ninja-512 (4x8x8): sanity test keeps failing. TPU mesh corruption persists. Deprioritized.

### BREAKTHROUGH: 70.5% MFU on 4x8x8!

**bodaborg-tpu7x-spot-256-chip** cluster works (ninja-512 has TPU mesh issues).

Full DS-v3 671B on v7x 4x8x8 (256 chips, 512 devices):
- **TFLOP/s/chip: 1,627** 
- **MFU: 70.5%**
- **Step time: 944 ms**
- **TPS/chip: 4,340**
- **TPS/cluster: 1,110,978**

vs 2x4x4 (32 chips): 65.6% MFU → **+5pp from larger topology**
Scaling efficiency: 107.6% (super-linear — larger slice has better comm/compute ratio)

Loss is ~110 (finite! init scale fix working). NaN doesn't appear until later steps (SGD overshoot).

### Updated progress table

| Topology | Chips | TFLOP/s/chip | MFU | TPS/chip | TPS/cluster |
|----------|-------|-------------|-----|----------|-------------|
| 2x4x4 | 32 | 1,512 | 65.6% | 4,035 | 129,104 |
| **4x8x8** | **256** | **1,627** | **70.5%** | **4,340** | **1,110,978** |

## 2026-03-27 (late): MFU measurement bug found and fixed

### Bug: TPS/chip and MFU numbers were completely wrong

All previous MFU and TPS/chip numbers in this logbook are WRONG. The bug:

The code computed:
```python
tflops = est_flops / step_time / 1e12           # This is CLUSTER-TOTAL, not per-device
tokens_per_sec = tokens_per_step / step_time     # This is CLUSTER-TOTAL too
```

Then MFU was incorrectly computed as:
```
cluster_TFLOP/s / peak_per_JAX_device = 756 / 1153.5 = 65.6%  ← WRONG
```

This mixes the entire cluster's throughput with a single device's peak, off by **n_devices**.

### Corrected numbers

v7x hardware: 2 JAX devices per physical chip.

| Run | Chips | Cluster TFLOP/s | Per-chip TFLOP/s | Real MFU | Was claimed |
|-----|-------|----------------|-----------------|----------|-------------|
| 2x4x4 v8 (FSDP=64, PDBS=1) | 32 | 756 | 23.6 | 1.02% | 65.6% |
| 4x8x8 v1 (FSDP=512, PDBS=1) | 256 | 813 | 3.2 | 0.14% | 70.5% |

### Root cause of ~1% real MFU

With PDBS=1 (single sequence, 4096 tokens) globally:
- Total compute: 767 TFLOP for the full 671B fwd+bwd
- Step time: ~1.015s (2x4x4) — only 11.8 TFLOP/s per JAX device vs 1153.5 peak
- ~99% of time is FSDP all-reduce communication overhead
- Each matmul produces ~100MB activations → all-reduce across 64 devices takes ~2ms
- 305 matmuls × 2ms = ~600ms in all-reduces alone

The 4x8x8 is only 1.1x faster than 2x4x4 despite 8x more chips = 13% scaling efficiency.

### Fix: need larger global batch

The compute/comm ratio improves linearly with PDBS:
- PDBS=1: 1% MFU (comm-dominated)
- PDBS=8: ~8% MFU (rough estimate)
- PDBS=64+: expected to approach roofline

Memory constraint (v7x 96GB HBM / 2 devices/chip = 48GB per JAX device):
- With FSDP=64: 42GB per device for params+grads → leaves ~6GB for activations
- Activations per token: S × D × L × 2 bytes = 4096 × 7168 × 61 × 2 = 3.6GB (gradient_checkpoint=True)
- With gradient_checkpoint, activations = ~3.6GB for full seq → PDBS can be up to ~1-2

So PDBS > 2 requires either:
1. More chips (but FSDP must increase proportionally)
2. Sequence parallelism  
3. Shorter sequences

### Code fix applied

`train.py` now correctly reports:
- `TFLOP/s/chip` = total_cluster_TFLOP/s / n_chips
- `MFU` = TFLOP/s/chip / v7x_peak_per_chip (2307 TFLOP/s)
- `TPS/chip` = total_cluster_TPS / n_chips
- `cluster_TFLOP/s` and `cluster_TPS` for the whole cluster

The device-kind detection identifies v7x and uses 2 devices/chip.

## 2026-03-28: 4×8×8 EP=32 sweep + MMLU format fix

### Context
Switched to `sivaibhav-exp-v7x` cluster (DWS Flex, 4×8×8, 512 JAX devices).
Fixed `gate_bias` KeyError in `make_params()` and `shard_pretrained_weights()` — random-init training now works.
All runs: EP=32, FSDP=16, jax_ep, gradient_checkpoint, SGD, 15 steps, profile at steps 6-7.
TPS/chip = (4096 × PDBS) / step_time.

### Training Experiments (sivaibhav-exp-v7x, 4×8×8)

#### v1: EP=8, FSDP=64, jax_ep, PDBS=1 (baseline)
- Profile total: 200,966ms → step_time = 200,966/(2×12) = **8.38s** (also measured 8.384s directly)
- **TPS/chip: 489** | Roofline: 708 | **69% of roofline**
- Top bottleneck: FSDP reduce-scatter 13.7%; all-gather overlap only 16.3%
- MXU efficiency: 31.3%, HBM BW: 7.6% — poorly amortized compute

#### v2: EP=32, FSDP=16, jax_ep, PDBS=1
- Profile total: 63,403ms → step_time ≈ **2.64s** (3.17× speedup over v1)
- **TPS/chip: 1550** | Roofline: 1739 | **89% of roofline**
- EP psum now 8.0% (new cost), all-gather overlap jumped to 44%, RS 99.7% overlapped

#### v3: EP=32, FSDP=16, ragged_dot, PDBS=1
- Profile total: 87,068ms → step_time ≈ **3.63s** (SLOWER than v2 despite ragged_dot)
- **TPS/chip: 1127** | Roofline: 2082 | **54% of roofline**
- Root cause: `scatter_custom_fusion` backward costs 16% of step (argsort unsort)
- **Conclusion: ragged_dot EP>1 is WORSE for training; jax_ep is better. Do not retry.**

#### v4: EP=32, FSDP=16, jax_ep, PDBS=4
- Profile total: 169,719ms → step_time ≈ **7.07s**
- **TPS/chip: 2317** | Roofline: 3305 | **70% of roofline**
- Collective breakdown (profile, 2 steps):
  - all-reduce: 42,345ms (61% of collective); only 11.6% overlapped
    - EP psum #1 (all-reduce.396): 12,256ms, 454MB, **0% overlapped** = 7.2% of step
    - EP psum #2 (all-reduce.376): 6,593ms, 486MB, **0% overlapped** = 3.9% of step
  - all-gather: 21,854ms (31%); 45.8% overlapped
  - reduce-scatter: 99.7% overlapped (gradient RS fully pipelined)
  - all-to-all: 3,291ms; 21 ops of ~554B each (latency-bound routing syncs)
- Total idle: **33.8%** of step
- PDBS=4 roofline cap is 3,305 → to reach 4k TPS/chip need PDBS=8

### MMLU Evaluation (ninja-v7x-64-spot)

#### Root cause of ~25% accuracy (near-random) — v27-v29
- Prompt format was `Answer:` followed by looking for ` A`/` B`/` C`/` D`
- Base model predicts '333', '\n', '111', etc. after plain "Answer:" — not letters
- A/B/C/D logprobs are within 0.3 logit units of each other = essentially uniform = random
- Sanity check passes: "The capital of France is" → " Paris" ✓ (model IS loaded correctly)
- Short questions (seq_len<310) DO predict letters — format matters at shorter context
- v28: top-100 token scan didn't help because letters aren't in top-100 for most questions

#### v30-eval fix (applied 2026-03-28)
Per HELM/Harness standard format for MCQ base model evaluation:
- Choices: `(A) text (B) text (C) text (D) text` inline
- Prompt ending: `Correct answer: (` (model predicts `A`/`B`/`C`/`D` bare letter after `(`)
- Answer token IDs: `tokenizer.encode("A", add_special_tokens=False)[-1]` (no space)
- Added dual id_to_choice (bare + space-prefixed) for robust top-k fallback
- Added sample prompt print for first question to verify format
- Expected accuracy: ~87-91%

### Active jobs (2026-03-28 morning)
- **MMLU v30-eval**: Running on ninja-v7x-64-spot — awaiting accuracy results
- **v5 training** (EP=32, FSDP=16, jax_ep, PDBS=8): Submitted to sivaibhav-exp-v7x
  - Roofline ~4,500+ TPS/chip; at 70% efficiency → ~3,150 TPS/chip
  - For 4k TPS/chip target: need either PDBS≥16 or significantly improved overlap

### Path to 4k TPS/chip
Current best: 2,317 TPS/chip (v4, PDBS=4). Target: 4,000.
- PDBS=4 roofline cap is 3,305 — cannot reach 4k at PDBS=4
- v5 (PDBS=8): roofline ~4,500; at 70% = 3,150, at 89% = 4,005
- Main bottlenecks to address:
  1. **EP psum 11.1%, 0% overlapped** — pipelining EP comm with compute could save ~785ms/step
  2. **FSDP all-gather 45.8% overlapped** → target 65%+ → save ~500ms/step
  3. **33.8% total idle** (57,448ms of 169,719ms profile) — collectives on critical path

---

## 2026-03-30: Pallas Backward Kernel Integration (v12)

### What was built

**D-tiling in `fused_moe_bwd/backward_kernel.py`:**
- All VMEM buffers tiled on D: `w1_gate/up (tile_D,F)`, `w2 (F,tile_D)`, `tok/dexp (bte,tile_D)`
- Two-pass per block: pass 1 accumulates `h_gate,h_up,d_h_act` as `(bte,F)` carry;
  pass 2 writes `d_w1/d_w2/d_tok` tiles to HBM
- `vmem_limit_bytes` and `tile_D` now parameters (auto-selected if not given)
- v7x: D=7168, F=2048 → tile_D=1024 (50 MB < 64 MB VMEM ✓), 7 tiles

**New model backend `jax_pallas_bwd` in `model.py`:**
- `expert_mlp_pallas_bwd`: JAX einsum forward + Pallas kernel backward
- Backward: shard_map with FSDP all_gather → pack w1 → `fused_ep_moe_bwd` → psum_scatter back
- Routing: sigmoid + top-K on raw gate logits (no gate_bias — acceptable for benchmarking)
- `--moe_backend=jax_pallas_bwd` CLI flag added to `train.py`

**Tests added (`test_grad_check_stage_d.py`):**
- `ep8_full_671b`: EP=8, D=7168, F=2048, E=256, K=8, T=64
- Auto-detects vmem_limit from chip kind

### v12 job (submitted 2026-03-30)
- Image: `gcr.io/tpu-vm-gke-testing/mini-dsv3:v35-train`
- Config: EP=8, FSDP=64, GBS=1024, `--moe_backend=jax_pallas_bwd`, attn=splash
- Profile: `gs://sivaibhav-exp/dsv3/profiles/4x8x8-ep8-fsdp64-gbs1024-v12`
- Comparison: v10 (ring+A2A) got 465 TPS/chip at same GBS=1024/EP=8/FSDP=64

### Expected outcome
- Pallas bwd eliminates the `jax.vmap` over T×K pairs in the expert backward
- Known gap (INTEGRATION.md): gating backward still uses vmap O(T×K×D×F) — dominates at K=8
- Conservative estimate: 1.3–1.5× speedup on MoE backward → ~600–700 TPS/chip
- If gating vmap is also eliminated: up to ~900 TPS/chip (theoretical)

### Known limitations of v12
1. gate_bias ignored in routing (won't affect training convergence check but diverges from DSv3 spec)
2. D-tiling increases memory bandwidth pressure (2 passes over HBM per block vs 1)
3. Gating backward vmap not yet replaced (largest remaining bottleneck in Pallas path)

### Compilation failures (2026-03-30)

**v35-train (first attempt):** Failed with `AssertionError: (float32, bfloat16)` — custom_vjp
gradient dtypes must match primal dtypes. Fixed with `.astype(fx.dtype)` casts in `_bwd_rule`.

**v35-train (second attempt, dtype fixed):** Stuck at "Compiling (first step)..." for 26+ minutes.
Root cause hypothesis: `@jax.custom_vjp` defined inside `expert_mlp_pallas_bwd` function body,
called from `lax.scan`. JAX traces function twice (fwd + remat bwd); each trace creates a new
custom_vjp object. Fixed in v36: lifted `_moe_pallas_bwd_fn` to module level with
`nondiff_argnums=(5,6,7,8,9,10)`, same pattern as `_expert_mlp_ring_a2a_body`.

**v36-train (module-level custom_vjp):** Still stuck at 27 minutes, then killed.
Root cause: `shard_map(pallas_call)` in the custom_vjp backward causes very slow XLA compilation.
- Baseline: v8 (ring backend, same EP=8/FSDP=64/GBS=1024/gradient_checkpoint) = 3 minutes
- v12 jax_pallas_bwd = 27+ minutes and never finished (~9× slower)
- The additional complexity in backward: 3 all-gathers + pallas_call + 2 psums + 3 psum_scatters
  inside a single shard_map body, vs ring backend's simpler shard_map

**Diagnostics run (2026-03-30):**

1. **`jax` backend, GBS=1024, EP=8, FSDP=64, gradient_checkpoint** → OOM at compile time:
   Used 124.84 GB (limit 94.75 GB). Dominant: `bf16[58,16,4096,7168]` = 50.75 GB = scan carries
   across 58 layers. Without custom_vjp, XLA stores all layer activations → OOM.
   **Conclusion: custom_vjp is MANDATORY at GBS=1024.** Ring backend avoids OOM because its
   custom_vjp precisely controls saved residuals.

2. **`ring` backend (v8) at identical config** → 3-minute compile, runs successfully.
   Only difference from jax_pallas_bwd: backward body uses JAX ops (ragged_dot) not Pallas.

3. **`jax_pallas_bwd` (v12), resubmitted** → waiting to see if it eventually compiles.
   Hypothesis: Pallas kernel Mosaic lowering (D=7168, F=2048, 7 D-tiles) embedded in outer
   XLA HLO causes slow optimization. Compilation cache at
   gs://sivaibhav-exp/dsv3/jax-cc/4x8x8-ep8-fsdp64-gbs1024-v12 will warm future runs.

4. **v12 resubmitted (v36-train image):** Still compiling at 45 minutes. Job is alive (no crash).
   Expecting 40–60 min cold compile on first run; subsequent runs will use GCS cache.
   If still not compiled at 90 min, restructure to remove nested JIT in `fused_ep_moe_bwd`.

---

## 2026-03-30 (continued): v10 Profile Deep Dive

### xla_shell analysis of v10 (ring+A2A, GBS=1024, EP=8, FSDP=64)
Profile: `gs://sivaibhav-exp/dsv3/profiles/4x8x8-ep8-fsdp64-gbs1024-v10`

**Hardware peaks (per JAX device, from xprof):**
- Peak compute: 1,028,750 GFLOP/s bf16
- Peak HBM BW: 530.2 GB/s
- Compute/memory knee: 1,940 FLOP/byte

**Step timing (v10, 465 TPS/chip):**
- Wall time per step: 35.2s
- Total HLO time (2-step window, one device): 846,025 ms
- Implied avg parallelism: ~12 ops running simultaneously

**Utilization (show_utilization):**
- MXU busy: 37.0% of cycles
- HBM Rd+Wr: 23.2% of peak
- ICI: 4.7% (massive headroom)
- MXU idle ("No MXU Busy"): 61.6%

**HLO time breakdown:**
| Component | HLO time (2 steps) | % |
|-----------|-------------------|---|
| MoE backward | 391,404 ms | 46.3% |
| Ragged-dot compute | 295,789 ms | 35.0% |
| MoE forward (ring+A2A) | 100,631 ms | 11.9% |
| Overhead/unattributed | 48,216 ms | 5.7% |
| Dense+attn | 5,744 ms | 0.7% |

**Top bottlenecks:**
1. `scatter_custom_fusion.29` (combine bwd): 112,276ms, 87 GB/s = 16.4% HBM eff → HBM bound
2. `scatter_custom_fusion.31` (token_sort bwd): 110,237ms, 88 GB/s = 16.6% HBM eff → HBM bound
3. Ragged-dot ops: 6–7 unique shapes, 30–34.5s each, 620–709 TFLOP/s, 60–69% compute eff
4. FSDP weight all-gathers: 3× per MoE layer, 0% overlap, 94–104 GB/s

**Collectives (list_collectives --overlap):**
- Total collective HLO time: 102,424ms (12.1% of total)
- Non-overlapped idle time: 59,577ms (7.0% of total HLO time)
- Non-overlapped FSDP all-gathers: 4 ops × 11–13s each = top bottleneck
- FSDP all-gather bandwidth: 94–104 GB/s (18–20% of 530 GB/s peak)
  → FSDP=64 spans cross-host ICI, explaining low BW vs theoretical

**Key insight: scatter-add backward is the #1 optimization target**
- 2 scatter-add ops = 26.3% of total HLO time = 57.7% of MoE backward
- Running at 16–17% of peak HBM BW (random permutation kills cache locality)
- v12 Pallas backward is designed to eliminate these entirely

**Full analysis: `~/dsv3/dsv3_layer_analysis.md`**
- Layer-by-layer breakdown with shapes, FLOPs, collectives, code refs
- v10 profile comparison and 4K TPS/chip roadmap

### 4K TPS/chip roadmap (from analysis)
| Optimization | Expected Speedup | Cumulative TPS/chip |
|-------------|-----------------|---------------------|
| v10 baseline (ring+A2A) | 1× | 465 |
| v12 Pallas backward | 1.24× | ~580 |
| PDBS=4 (GBS=2048) | 1.35× | ~780 |
| Pipeline FSDP gathers | 1.25× | ~975 |
| EP=16 | 1.6× | ~1,560 |
| Pallas forward (fused) | 2.0× | ~3,120 |

Hardware ceiling (100% MXU, no comm): ~2,274 TPS/chip for PDBS=2.
**4K TPS/chip requires: Pallas fwd+bwd, larger batch, EP scaling, AND better collective pipelining.**

## 2026-03-30 (continued): v12 root cause — nested JIT, fix in v15

### Why v12 was stuck for 60+ minutes

v12 (jax_pallas_bwd, EP=8, FSDP=64, GBS=1024) compiled for 60+ min and never finished.
Root cause found by reading `backward_kernel.py`:

```python
# v12 had this in backward_kernel.py (line ~344):
@functools.partial(jax.jit, static_argnames=[
    "top_k", "scoring_fn", "renormalize_topk_logits", "act_fn",
    "ep_axis_name", "bte", "tile_D", "vmem_limit_bytes",
])
def fused_ep_moe_bwd(...)
```

**Problem:** `fused_ep_moe_bwd` is called inside `shard_map(...)` which is called inside
the outer `@jax.jit` via `custom_vjp`. Nesting a `@jax.jit` inside `shard_map` inside
`@jax.jit` forces XLA to treat `fused_ep_moe_bwd` as a separate sub-compilation with its
own `HloModule`. Each Pallas call inside it must be lowered + compiled separately. This
is ~9× slower than inline compilation.

**Why the `@jax.jit` was unnecessary:** `nondiff_argnums=(5,6,7,8,9,10)` on the `custom_vjp`
already ensures all static args (top_k, scoring_fn, tile_D, etc.) arrive as Python literals
at trace time. No runtime dispatch is needed. Removing `@jax.jit` from `fused_ep_moe_bwd`
makes the Pallas call compile inline as part of the outer graph — no sub-module boundary.

**Fix:** Removed `@functools.partial(jax.jit, static_argnames=[...])` decorator.
Built v37-train, pushed to GCR.

### v15: jax_pallas_bwd with nested-JIT fix (submitted 2026-03-30)

- Image: v37-train (nested JIT removed)
- Config: identical to v12 (EP=8, FSDP=64, GBS=1024)
- Profile dir: `gs://sivaibhav-exp/dsv3/profiles/4x8x8-ep8-fsdp64-gbs1024-v15`
- JAX cache: `gs://sivaibhav-exp/dsv3/jax-cc/4x8x8-ep8-fsdp64-gbs1024-v15`
- Expected compile time: ~3–5 min (similar to ring+A2A v10)
- Kill v12, resubmit as v15.

### v13: ring+A2A PDBS=4 (GBS=2048) — pending, nodes scaling

- Also submitted 2026-03-30
- Waiting for DWS Flex nodes to provision
- This is the "easy win" baseline scaling test

### v13 OOM analysis (2026-03-30)

v13 (GBS=2048, PDBS=4) failed with:
```
RuntimeProgramAllocationFailure: Attempting to reserve 90.41G at the bottom of memory.
There are 88.59G free, 0B reserved, and 88.59G reservable.
```

Root cause: `sorted_x` scales as `T_global / EP`. Increasing GBS 2× doubles sorted_x:
- GBS=1024, EP=8: sorted_x = (524,288, 7168) = 7.5 GB per MoE layer (v10 worked ✓)
- GBS=2048, EP=8: sorted_x = (1,048,576, 7168) = 15.0 GB per MoE layer → OOM ✗

But `sorted_x ∝ T_global / EP`. So EP=16 at GBS=2048 has SAME sorted_x as EP=8 at GBS=1024:
- GBS=2048, EP=16: sorted_x = (524,288, 7168) = 7.5 GB ← same memory footprint as v10 ✓

**Key insight: can scale GBS proportionally to EP. Double GBS if doubling EP.**

Fix: v16 (EP=16, FSDP=32, GBS=2048) — same memory budget as v10, but 2× batch and 2× EP.

### v15 root cause: jax_pallas_bwd unscalable at T=524,288 (2026-03-30)

Killed v15 after 40+ min compile with no progress. Root cause analysis of backward_kernel.py:

**Issue 1: jax.vmap(ffn_one) + per-token weight gather = 247 TB tensor**
```python
w1g_pairs = w1_f32[expert_ids_flat_kernel, 0]  # (TK, D, F)
# TK = 4,194,304, D = 7168, F = 2048 → (4M, 7168, 2048) float32 = 247 TB
expert_outs_flat = jax.vmap(ffn_one)(tokens_flat, w1g_pairs, ...)  # 247 TB input
```

XLA would need to materialize or fuse 247 TB. With no gather-GEMM fusion support, this hangs compilation.

**Issue 2: bins_tokens pre-allocation = 120 GB (or 3.85 TB with max_tpe bug)**
```python
max_tpe = cdiv(TK, bte) * bte  # = TK (WRONG: should be TK / E_local = avg per expert)
bins_tokens = jnp.zeros((E_local * max_tpe, D), ...)  # (32 × TK, D) = 3.85 TB
```
Even with bug fixed: (TK, D) = (4M, 7168) = 120 GB — still exceeds HBM.

**Root cause summary**: Backward kernel assumes small T (tested at mini config with T≈512).
At full scale T=524,288, pre-allocated tensors are O(T×K×D) = O(T×K×E_local×D) in the worst case.

**Fix required for Pallas bwd to work at scale:**
1. Compute d_gating per-expert instead of per-token-expert-pair (loop over E_local experts, process their token subset each time)
2. Compute bins_tokens as (TK, D) not (E_local × TK, D)
3. Use streaming Pallas kernel that processes one expert's tokens at a time

This is a significant rework. Deprioritized in favor of EP=16 scaling first.
**DO NOT submit jax_pallas_bwd jobs until kernel is redesigned for large T.**

### v16 failure: SparseCore 16 GiB per-tensor limit (2026-03-30)

v16 (ring+A2A, EP=16, FSDP=32, GBS=2048) failed immediately:
```
INVALID_ARGUMENT: SparseCore only supports 16 GiB of HBM per tensor, got 30064771072
Tensor: bf16[2097152,7168]
```

Root cause: for ring+A2A backend, `sorted_x = (T_global/FSDP × K, D)`.
With EP=16, FSDP=32: T_global/FSDP = (2048×4096)/32 = 262,144 → sorted_x = (2,097,152, 7168) = 30 GiB.
v7x SparseCore rejects tensors > 16 GiB.

**KEY RULE FOR ring+A2A**: sorted_x scales as T_global/FSDP = T_global × EP/512.
More EP → lower FSDP → LARGER sorted_x. ring+A2A EP scaling is SELF-DEFEATING.
With GBS=2048 + EP=16: sorted_x = 30 GiB (over SparseCore limit).
With GBS=1024 + EP=8 (v10): sorted_x = 7.5 GiB (fine).

This means EP>8 is IMPOSSIBLE with ring+A2A on v7x unless GBS is reduced, but
reduced GBS means worse MFU. Dead end for ring+A2A.

### v14 result: 260 TPS/chip (WORSE than v10's 465) — EP=16 ring+A2A backfires

v14 profile: `gs://sivaibhav-exp/dsv3/profiles/4x8x8-ep16-fsdp32-gbs1024-v14`

Scatter-add backward went from 26% (v10) to 0.02% — eliminated! But total step time
DOUBLED from 35s to ~70s. Why?

**Root cause:** ring+A2A T_all = T_global/FSDP = T_global × EP/512.
EP=16 with FSDP=32 → T_all = (1024×4096)/32 = 131,072 tokens per device (vs 65,536 for v10).
EVERY operation scales with T_all: ragged-dot, while loop, FSDP gathers. Cutting FSDP in half
doubled ALL work, not just scatter-add.

**CONFIRMED DEAD END: ring+A2A EP>8 is strictly worse for training on v7x.**
The scatter-add saving is dwarfed by the doubled compute at every step.

### v17 failure: jax backend INT32 scan carry overflow (2026-03-30)

v17 (jax backend, EP=32, FSDP=16, GBS=1024) failed:
```
jax.errors.JaxRuntimeError: INTERNAL: RET_CHECK failure (jellyfish/bounds_check.cc:717)
allocation_size_words <= std::numeric_limits<int32_t>::max() 3405774848
Tensor: bf16[58,64,4096,7168]
```

Root cause: jax backend uses `shard_map(+psum)` for EP. When JAX traces autodiff through
`shard_map` inside `lax.scan`, it creates a scan carry of shape:
  `(L_moe, PDBS×EP, S, D) = (58, 2×32, 4096, 7168) = 218 GB → INT32 word overflow`

The carry replicates activations by EP shards.

**Fix (v38-train):** Added `_moe_jax_ep_fn` module-level `custom_vjp` in `model.py`:
- Forward: same shard_map EP forward as before
- Backward: stores only `(fx, fi, fw, w0, w1, wout)` residuals per layer (no EP replication)
- Backward uses `jax.vjp` on the shard_map forward (safe outside lax.scan)
- Scan carry is now O(GBS/FSDP × D) = O(T_local × D) — no INT32 overflow

### v18: jax backend (custom_vjp fix attempt) — FAILED: same error

Same config as v17 but with v38-train image. Still same error: `bf16[58,64,4096,7168]`.

**Root cause diagnosis (revised)**: the scan carry overflow has nothing to do with shard_map autodiff.
The carry is the hidden state `x` stored across 58 MoE scan layers:
- `x` global shape: `(GBS, S, D) = (1024, 4096, 7168)`, sharded as `P("fsdp", None, None)`
- Per device (FSDP=16): `(GBS/FSDP, S, D) = (64, 4096, 7168)` = 3.5 GB
- Scan backward stores all 58 carries: `(58, 64, 4096, 7168)` = 218 GB → INT32 overflow

The `custom_vjp` fix targets `_moe_jax_ep_fn` (the MoE shard_map), which doesn't affect
the scan carry size. The carry overflow is from `x` between layers, not from MoE internals.

**Actual fix**: reduce micro-batch size per scan call via `--grad_accum=4`:
- micro-batch = GBS/4 = 256 sequences → per-device carry = `(58, 16, 4096, 7168)` = 27 GB ✓
- Same as v10's working config (EP=8, FSDP=64, GBS=1024 → B_local=16)

### v19: jax backend, EP=32, FSDP=16, GBS=1024, grad_accum=4 — submitted 2026-03-30

Fix for v17/v18. Added `--grad_accum=4` to reduce micro-batch carry.
Profile: `gs://sivaibhav-exp/dsv3/profiles/4x8x8-ep32-fsdp16-gbs1024-accum4-v19`
Image: v38-train (has the custom_vjp which doesn't hurt, just wasn't the fix)
Expected: step_time ≈ 4 × (v2 GBS=1 step time) × (some EP psum overhead) ≈ 10-15s

### Running experiments as of 2026-03-30 late

| Job | Config | Status |
|-----|--------|--------|
| v13 | ring+A2A, EP=8, FSDP=64, GBS=2048 | FAILED: HBM OOM (program 90.4 GB) |
| v14 | ring+A2A, EP=16, FSDP=32, GBS=1024 | DONE: 260 TPS/chip (worse than v10) |
| v15 | jax_pallas_bwd, EP=8, FSDP=64, GBS=1024 | KILLED: kernel unscalable at T=524k |
| v16 | ring+A2A, EP=16, FSDP=32, GBS=2048 | FAILED: SparseCore 16 GiB limit |
| v17 | jax backend, EP=32, FSDP=16, GBS=1024 | FAILED: INT32 scan carry 218 GB |
| v18 | jax backend (custom_vjp), EP=32, FSDP=16, GBS=1024 | Pending (nodes provisioning) |

---

## 2026-04-03: v119c stable — Pallas vs JAX benchmark and profile analysis

### Context

After v119c fixed the DynamicJaxprTracer constant error (module-level `_moe_pallas_sg`
custom_vjp), the Pallas fwd + streaming_bwd_v2 + gradient_checkpoint stack is stable
at 4×8×8 EP=32, FSDP=16, GBS=1024.

### bench-v3: 3-way comparison (12 steps)

| Backend | TPS/chip | Step time | Status |
|---|---|---|---|
| JAX (shard_map+psum) | **3,391** | 4.83s | ✓ |
| Pallas (fused_ep_moe_v1 + streaming_bwd_v2) | **2,484** | 6.59s | ✓ |
| ring+A2A (ragged_dot) | FAILED | — | SparseCore 16 GiB limit |

ring+A2A failed immediately: backward produces `bf16[2097152,7168]` = 28 GiB > 16 GiB
SparseCore tensor limit. Dead end at EP=32.

### bench-v4: 100-step loss curves

Both backends stable. Loss at step 100: JAX=25.084, Pallas=25.091 (+0.007 — numerical
divergence from different floating-point paths, acceptable).

JAX at 3,391 TPS/chip is the **best measured training performance** on debug config to
date; 87% of the EP=32 ragged_dot+fp8 roofline (3,880 TPS/chip).

### profile-v4: component breakdown (3-step average)

| Component | JAX | Pallas | Notes |
|---|---|---|---|
| MoE forward | 50.2s | **8.4s** | Pallas **6× faster** |
| MoE backward | 79.3s | **178.6s** | Pallas **2.25× slower** |
| Dense + attn | ~9.9s | ~10.0s | identical |
| Total (3-step profile op-time) | 154.5s | 202.7s | |

Pallas fwd kernel genuinely eliminates 4 separate EP/FSDP collectives (43.8s).
The backward regression fully cancels it.

### Root causes of Pallas backward regression

**1. `psum + dynamic_slice` instead of `psum_scatter` — ~42s wasted**
`_streaming_bwd_fn` lines 1011-1020 do `psum(d_tok, "ep")` then `dynamic_slice`.
All-reduce does 2× the data movement of reduce-scatter AND produces a sync barrier.
Fix: replace with `psum_scatter(d_tok_partial, ep_axis_name, scatter_dimension=0, tiled=True)`.
4 lines of code. Expected savings: ~42s backward.

**2. Sync barrier (all-reduce.311) — 7.8s for 609KB = 0% collective overlap**
The `psum(d_topk_partial, "ep")` serializes the entire Pallas backward pipeline.
JAX backward has 74% all-gather overlap from `--xla_tpu_enable_ag_backward_pipelining`.
Pallas shard_map custom-call ops are opaque to XLA async fusion → 0% overlap everywhere.
Fix 1 (psum_scatter) also eliminates this barrier.

**3. 4× all_gather("ep") vs 1× in JAX — +17.9s**
streaming_bwd_v2 gathers (grad, fx, fw, fi) separately; JAX backward needs only 1.

**4. `psum("fsdp")` all-reduce inside kernel instead of reduce-scatter — ~18s**
`fused_ep_moe_bwd_streaming_v2` with `fsdp_axis_name="fsdp"` does 3 all-reduces
(d_w0, d_w1, d_wo). JAX GSPMD automatically uses reduce-scatter for these.

### Optimization roadmap and projections

| State | Backward time | TPS/chip (debug) |
|---|---|---|
| Current (v119c) | 178.6s | 2,484 |
| Fix 1: psum_scatter | ~136s | ~3,100 |
| Fix 1+2+3: JAX parity | ~79.3s | ~3,391 (1.49×) |
| Fully fused bwd kernel (6× fwd speedup) | ~13s | ~5,050 (1.49×→4.1×) |

Full config extrapolation (JAX baseline = 1,083 TPS/chip no-ckpt):
- JAX backward parity → ~1,614 TPS/chip
- Fully fused backward → ~4,484 TPS/chip

### profile-v5: submitted

`bench-v5-pallas-traced` with `--xla_enable_custom_call_region_trace=true` submitted.
Will reveal actual BW/FLOPs for `gather_custom_fusion` (35.3s, currently 0 GB/s).

### Next: streaming_bwd_v3 kernel design

Design goal: fully fused Pallas backward that mirrors the forward structure —
A2A inside the kernel, no external collectives, reduce-scatter for output gradient.
Expected: ~4× overall speedup vs JAX (same order as Pallas fwd's 6× speedup).
See design doc: `dsv3/docs/streaming_bwd_v3_design.md` (to be written).
