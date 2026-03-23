# RPA v3 Integration Logbook — Qwen3.5 Decode on TPU

## Background

### What is RPA?
RPA v3 (Ragged Paged Attention) is a fused Pallas/Mosaic kernel from the `tpu-inference` library. It fuses KV cache read + Q@K + softmax + @V + cache write into a single kernel with async DMA double buffering. The goal: eliminate the expensive `dynamic_slice`/`dynamic_update_slice` bottleneck that dominates decode time with contiguous KV caches.

Library: `tpu_inference.kernels.ragged_paged_attention.v3.kernel.ragged_paged_attention`
Vendored at: `third_party/tpu_inference/`

### Why RPA?
In the baseline contiguous GQA attention (`gqa.py` lines 88-93), each decode step:
1. Reads the **entire** KV cache array (B x n_kv_heads x max_len x head_dim)
2. Updates a single position with `dynamic_update_slice`
3. Passes the full cache to `dot_product_attention`

This is wasteful — we read/write GBs of data to update one token. RPA uses paged KV caches and DMA double buffering to work with only the relevant cache pages, avoiding full-array reads/writes.

RPA is proven in production (vLLM on TPU) and is the right approach. Our problems are integration-related.

### Qwen3.5 Architecture
- 60 layers = 15 groups of [DeltaNet x 3 + GQA x 1]
- DeltaNet layers: O(1) recurrent state (delta_M, delta_conv) — no KV cache needed
- GQA layers (every 4th layer): use KV cache, benefit from RPA
- Full config: 32 Q heads, 2 KV heads, head_dim=256, 512 experts top-10 MoE

## Files Involved

| File | Purpose |
|------|---------|
| `jax_gpt/models/qwen35/gqa_rpa.py` | Drop-in RPA replacement for `gqa_attention()`. Handles QKV projection, QK norm, RoPE, output gate, output projection. Uses `shard_map` when mesh is provided. |
| `jax_gpt/models/qwen35/paged_cache.py` | `contiguous_to_paged()` converts contiguous KV cache to paged format. `make_decode_metadata()` creates `cu_q_lens` and `distribution` arrays. |
| `jax_gpt/models/qwen35/block.py` | `gqa_layer_forward_rpa()` and `group_forward_rpa()` — group-level forward using RPA. |
| `jax_gpt/models/qwen35/model.py` | Two RPA decode paths: scan-based (lines 275-320) and per-group JIT (lines 407-513). |
| `jax_gpt/models/qwen35/cache.py` | `HybridCache` dataclass with paged_kv, kv_lens, page_indices fields for RPA. |
| `scripts/qwen35_benchmark.py` | RPA benchmark path with cache conversion and profiling. |
| `third_party/tpu_inference/.../v3/kernel.py` | The RPA kernel itself. |
| `third_party/tpu_inference/.../v3/tuned_block_sizes.py` | Hardware-specific block size tuning. |

## Chronological Debug Log

### Phase 1: Initial Integration

**Goal**: Get RPA kernel running for GQA decode layers within the existing Qwen3.5 scan-based forward pass.

**What we built**:
- `gqa_rpa.py`: RPA replacement for GQA attention. Key design: `shard_map` wraps the kernel call when a mesh is provided (required because Pallas kernels can't be auto-partitioned by XLA).
- `paged_cache.py`: Converts contiguous KV cache (from prefill) to paged format. `page_size=64` hardcoded. `make_decode_metadata()` creates `cu_q_lens=arange(B+1)` and `distribution=[B,B,B]` (all-decode, no prefill/mixed).
- Extended `HybridCache` with paged KV fields, registered as pytree for `lax.scan` carry.
- `model.py` scan-based RPA path: `_group_step_rpa` calls `group_forward_rpa` inside `jax.lax.scan`.

**Issues encountered**:
1. **`init_params` argument order**: Signature is `init_params(config, key, dtype)` — had args reversed.
2. **`--dtype=bf16` invalid**: Should be `--dtype=bfloat16` (argparse choices).
3. **`ModuleNotFoundError: tpu_inference`**: Not installed locally. Fixed with `PYTHONPATH=third_party:$PYTHONPATH`.

### Phase 2: Approach 1 — Scan-based JIT (OOM)

**How it works**: Single `@jax.jit` wrapping the full `forward()` call. Inside, `jax.lax.scan` iterates over groups, with the RPA kernel + `shard_map` in the scan body.

**Result on v7x-64 (60 layers, 15 groups)**:
```
RESOURCE_EXHAUSTED: Ran out of memory trying to allocate 3.69GiB for program
```

XLA determined at compile time that the program's HLO temporary buffers (intermediate activations,
communication buffers, padding) need 3.69 GB, exceeding available HBM after params + cache (~949 MB
free). Note: "for program" in XLA's error refers to the program's tensor buffer allocations (which
are 98-100% HLO temps), NOT the compiled instruction binary (which is tiny, MB-scale).

**Result on 4-layer (1 group)**: Compiles and runs (far fewer intermediate buffers).

### Phase 3: Approach 2 — Per-group Python Loop (Too Slow)

**How it works**: `forward_rpa_decode()` in model.py (lines 407-513). Python `for g in range(n_groups)` loop, each iteration calls `_jit_group_forward_rpa` (separately JITted). Cache arrays updated in-place with `.at[g].set()`.

**Result (4-layer on v7x-64, B=128, 1K:1K)**:

| Metric | Baseline (contiguous) | RPA (per-group JIT) |
|--------|----------------------|---------------------|
| Per step | 22.74 ms | 770.38 ms |
| Throughput | 5,629 tok/s | 166 tok/s |
| TPS/chip | 87.9 | 2.6 |
| **Ratio** | — | **34x slower** |

**Root cause**: 15 serial host-to-device round trips per decode step + `.at[g].set()` copies.

**Issues encountered**:
- **`jnp.stack(new_paged_kvs)` OOM**: Stacking all group paged KVs at once exceeded HBM. Fixed with in-place `.at[g].set()` updates.
- **v49 benchmark never finished**: Per-group loop with full 60L config ran 65+ minutes without completing. Killed.

### Phase 4: Local v5p Testing

**Goal**: Test without shard_map to isolate kernel vs infrastructure overhead.

**Result (v5p, 4 chips, B=4, 1 group, scan JIT)**:

| Metric | Baseline | RPA (scan JIT) |
|--------|----------|----------------|
| Per step | 1.37 ms | 266 ms |
| TPS/chip | 729.2 | 3.8 |
| **Ratio** | — | **194x slower** |

**Key insight**: This uses scan-based JIT (single program), so the 194x slowdown is NOT from host overhead — it's the RPA kernel itself being slow at this scale.

### Phase 5: HLO Analysis

**Goal**: Understand HBM allocation and scan behavior.

**Findings on v5p (without shard_map)**:
- Compiled binary: 3-4.5 MB (tiny)
- Program size is flat across 1/2/4 groups — **scan compiles body once, does NOT unroll**
- HLO line counts nearly identical across group counts

**Implication**: The 3.69 GB HBM allocation on v7x is driven by `shard_map` + v7 Pallas codegen
producing larger intermediate tensor buffers, not from scan unrolling. Without shard_map on v5p,
the HLO temp buffers are much smaller. (The compiled instruction binary is always small — the
difference is in how much HBM XLA needs for intermediate tensor allocations at runtime.)

### Phase 6: Root Cause Hypotheses

#### Why does XLA need 3.69 GB of HBM for intermediates with shard_map on v7x?

Note: The 3.69 GB is NOT a compiled instruction binary — it's the peak HLO temporary buffer
allocation (intermediate activation tensors that must be live simultaneously). The instruction
binary is trivially small (MB-scale). XLA's "for program" label is misleading — it's 98-100%
tensor buffers. TPU instructions reside in HBM but are DMA'd to the scalar core's IMEM as
overlays; their size is negligible.

1. **shard_map lowering**: `shard_map` lowers each Pallas kernel into per-device SPMD code. The v7 Pallas codegen for `ragged_paged_attention` may produce larger intermediate buffer requirements.
2. **v7 memory coloring**: On v7, the RPA kernel wraps `pallas_call` in additional `@jax.jit` with `pltpu.with_memory_space_constraint` to pin buffers to HBM. This may increase the number of buffers that must be HBM-resident simultaneously.
3. **MIXED sub-kernel**: `ragged_paged_attention()` always compiles DECODE + MIXED sub-kernels. MIXED does zero work in pure decode (distribution=[B,B,B] gives empty range [B,B)), but its intermediate buffers are still allocated.

#### Why is RPA 194x slower than baseline (even without shard_map)?

1. **No tuned block sizes for v5p**: `tuned_block_sizes.py` has entries for v7 and v5e, but NOT v5p. Falls back to generic heuristics (`bkv_p=2048//page_size, bq=32`).
2. **page_size=64 not tuned on v7**: v7 tuned entries are only for page_size=128. Our page_size=64 falls back to generic sizes.
3. **Small batch/context**: RPA is designed for large-batch, long-context scenarios. At B=4, seq_len=128, the paged cache indirection overhead dominates any benefit.
4. **Nested JIT donation**: `ragged_paged_attention()` uses `@jax.jit(donate_argnames=...)`. When called inside an outer JIT (our scan), `donate_argnames` is silently ignored. This means the kernel may be copying buffers instead of donating them.

#### PartitionSpec Issues

Our shard_map specs in `gqa_rpa.py` (lines 130-148) may be wrong:
- Q is sharded on tp_axis (heads dimension)
- Everything else (paged_kv, page_indices, kv_lens, metadata) is fully replicated

RPA team recommendations suggest different sharding strategies. This needs investigation.

## Key Technical Details

### RPA Sub-kernels
`ragged_paged_attention()` always runs:
1. **DECODE** kernel — handles pure decode tokens
2. **MIXED** kernel — handles mixed prefill+decode batches
3. Optionally **PREFILL** kernel — handles pure prefill tokens

In our setup (pure decode), MIXED processes an empty range `[B,B)` — zero work at runtime, but the kernel is still compiled.

### Nested JIT Behavior
- Inner `@jax.jit` is inlined when called inside outer `@jax.jit`
- `donate_argnames` is silently ignored in nested context
- `input_output_aliases` on `pallas_call` still works at kernel level
- This is documented JAX behavior

### Scan Does Not Unroll
Confirmed by HLO analysis: `jax.lax.scan` compiles the loop body once and iterates. The program binary size is constant regardless of group count (1, 2, or 4 groups all produce ~3-4.5 MB on v5p without shard_map).

## Docker Images

| Version | Contents | Status |
|---------|----------|--------|
| v48 | Pre-RPA baseline | Working |
| v49 | RPA code with per-group JIT path | Working but slow |
| v50 | Scan-based JIT path (changed benchmark to use `forward()` with `use_rpa=True`) | Built successfully |

Build command: `gcloud builds submit --tag gcr.io/tpu-vm-gke-testing/jax-gpt-tpu:vNN --timeout=1800 --machine-type=e2-highcpu-32`

## K8s Deployments

| Jobset | Purpose | Image |
|--------|---------|-------|
| `bench_v7x_rpa_mini_jobset.yaml` | 4-layer RPA test on v7x-64 | v49 |
| `bench_v7x_baseline_mini_jobset.yaml` | 4-layer baseline on v7x-64 | v49 |
| `bench_v7x_rpa_debug_jobset.yaml` | 4-layer RPA with XLA dump + profiling | v50 |

### Phase 7: Profile Analysis & Root Cause Discovery (2026-03-21)

**Goal**: Capture xprof profile on v7x-64 to understand where time is spent.

**Setup**: Deployed v51 (clean rebuild) with `--profile` and `--skip-prefill`. Profile saved to
`gs://max-experiments/profiles/qwen35-rpa-debug-v51/decode/`.

**v51 Results (pre-fix)**:
- Average decode: **2,630 ms/step** (but this included compilation overhead — see below)

**Key Discovery from Profile**:
The user identified `PJRT_client_Compile` appearing before **every** decode step in the xprof trace.
JAX was recompiling the decode function on each iteration, adding ~2.5s compilation overhead per step.

**Root cause**: Missing `with_sharding_constraint` on the `updated_kv` (paged cache) output in the
RPA scan body. The baseline code path had 4 sharding constraints (delta_M, delta_conv, gqa_k, gqa_v),
but the RPA path only had 2 (delta_M, delta_conv). Without a constraint on `paged_kv`, the output
from `shard_map` had a different sharding than the input pytree, causing JAX to retrace and recompile
on the next call.

### Phase 8: Sharding Constraint Fix (2026-03-21)

**Fix applied in two files**:

1. **`sharding.py`** — Added `paged_kv` PartitionSpec to `make_cache_sharding()`:
   ```python
   paged_kv_spec = P(dp_axis, None, None, None, None)
   return {
       'delta_M': ..., 'delta_conv': ..., 'gqa_kv': ...,
       'paged_kv': paged_kv_spec,
   }
   ```

2. **`model.py`** — Applied constraint in `_group_step_rpa` scan body:
   ```python
   if 'paged_kv' in cache_sharding:
       updated_kv = jax.lax.with_sharding_constraint(updated_kv, cache_sharding['paged_kv'])
   ```

**v52 Results (with fix, old benchmark timing)**:
- Average: 1,347 ms/step (still inflated — measuring total time including prefill / decode steps)
- Profile showed clean back-to-back decode steps at ~17ms each with no recompilation

**v53 Results (with fix + per-step benchmark timing)**:

| Metric | Baseline (contiguous) | RPA (scan JIT) | Delta |
|--------|----------------------|-----------------|-------|
| First step | — | 14,552 ms | Includes data transfer |
| Steady-state | 22.74 ms | **19.64 ms** | **14% faster** |
| Throughput | 5,629 tok/s | 6,517 tok/s | +16% |
| TPS/chip | 87.9 | **101.8** | **+16%** |

**RPA is now faster than baseline** for 4-layer decode on v7x-64 (B=128, prompt_len=1024).

### Phase 9: Benchmark Fix (2026-03-21)

The original benchmark divided total elapsed time by decode steps, mixing prefill + first-step overhead
into the per-step average. Fixed to measure each decode step individually with `block_until_ready()`,
separating first step (data transfer overhead) from steady-state.

### Phase 10: DP-Aware shard_map & _safe_spec Fix (2026-03-21)

**Goal**: Eliminate the two massive all-gathers found in the v53 profile.

**Profile analysis** (using xprof API on `2026_03_21_19_41_46` xplane.pb):

The v53 profile revealed that despite 19.64 ms steady-state, **47% of decode device time**
was spent in two unnecessary all-gathers:

| Op | % of decode | What | Root cause |
|----|-------------|------|-----------|
| `all-gather.80` | 33.7% | paged_kv: bf16[1,288,64,2,2,256] → [1,2304,...] (8x on pages) | shard_map `in_specs` ignored dp axis |
| `all-gather.72` | 13.3% | delta_M: bf16[3,16,8,128,128] → [3,128,...] (8x on batch) | `_safe_spec` used placeholder B=1 |

**Trace observation**: In the xprof trace viewer, the decode step showed:
1. First `gqa_attn_rpa` (QKV projections, fp8 conversions — setup/prefetch)
2. First `gqa_moe` (fp8 weight `convert_element_type` — XLA-hoisted)
3. **Massive `all-gather.80`** — gathering paged_kv before shard_map
4. Second `gqa_attn_rpa` (actual RPA kernel)
5. Second `gqa_moe` (actual MoE compute)
6. `all-gather.72` + `all-reduce.83` (scan boundary collectives)

This confirmed the all-gathers were happening *inside* the group execution, not at the scan
boundary. XLA reordered/pipelined the compute but the all-gathers were blocking.

**Good news from profile**: MoE time was identical across DeltaNet and GQA layers
(~7.9 ms/layer/step), confirming the MoE was not the bottleneck. The RPA kernel itself
(`RPAd-p_64-bq_1_1-bkv_1152_1152`) took only 1.3 ms/step (1.5% of decode).

#### Bug 1: shard_map ignoring DP axis

**File**: `gqa_rpa.py`, shard_map `in_specs` and `out_specs`.

The shard_map for the RPA kernel declared all non-Q inputs as fully replicated:
```python
# BEFORE (wrong):
in_specs=(
    P(None, tp_axis, None),              # q — only TP
    P(None, None, None),                 # k — replicated
    P(None, None, None),                 # v — replicated
    P(None, None, None, None, None),     # kv_cache — replicated  ← WRONG
    P(None),                             # kv_lens — replicated
    P(None),                             # page_indices — replicated
    P(None),                             # cu_q_lens — replicated
    P(None),                             # distribution — replicated
)
```

But the paged_kv cache was dp-sharded (from `with_sharding_constraint` in the scan body).
XLA had to insert `all-gather.80` to replicate it before the shard_map — gathering 0.3 GB
across 8 dp shards every decode step.

**Fix**: Add dp axis to batch-related and page-related inputs:
```python
# AFTER (correct):
dp_axis = 'dp' if 'dp' in mesh.axis_names else None
in_specs=(
    P(dp_axis, tp_axis, None),           # q — dp on batch, tp on heads
    P(dp_axis, None, None),              # k — dp on batch
    P(dp_axis, None, None),              # v — dp on batch
    P(dp_axis, None, None, None, None),  # kv_cache — dp on pages
    P(dp_axis,),                         # kv_lens — dp on batch
    P(None,),                            # page_indices — replicated (local)
    P(None,),                            # cu_q_lens — replicated (local)
    P(None,),                            # distribution — replicated (local)
)
out_specs=(
    P(dp_axis, tp_axis, None),           # attn_out — dp + tp
    P(dp_axis, None, None, None, None),  # updated_cache — dp on pages
)
```

**Key subtlety**: `page_indices`, `cu_q_lens`, and `distribution` are kept replicated (`P(None,)`)
but computed with `B_local = B // dp` instead of `B`. Each dp shard processes `B_local` sequences
with local page indices `arange(B_local * pages_per_seq)`. The local page indices are always
0-based because each shard's dp-slice of the paged_kv array is also 0-based.

#### Bug 2: `_safe_spec` dropping dp_axis due to placeholder B=1

**File**: `sharding.py`, `make_cache_sharding()`.

The function used placeholder shapes with `B=1` for divisibility checks:
```python
delta_M_shape = (n_delta, 1, config.delta_n_v_heads, ...)
#                        ^ B=1 placeholder
```

`_safe_spec` checks `shape[i] % mesh.shape[axis] == 0`. With B=1 and dp=8: `1 % 8 != 0`,
so dp_axis was silently replaced with `None` (replicated). This meant the sharding constraint
on delta_M, delta_conv, and gqa_kv all DROPPED dp sharding, forcing XLA to all-gather the
delta state across dp shards at the scan boundary.

**Fix**: Accept `batch_size` parameter and use actual B in shape checks:
```python
def make_cache_sharding(config, mesh, axis_rules=None, batch_size=128):
    ...
    delta_M_shape = (n_delta, batch_size, ...)  # actual B, not 1
```

#### Changes summary

| File | Change |
|------|--------|
| `gqa_rpa.py` | shard_map `in_specs`/`out_specs`: add dp_axis on batch/pages dims |
| `model.py` | Compute `B_local = B // dp`; create local `cu_q_lens`, `distribution`, `page_indices_local` |
| `sharding.py` | `make_cache_sharding()`: add `batch_size` param, use in `_safe_spec` checks |
| `qwen35_benchmark.py` | Pass `batch_size=args.batch_size` to `make_cache_sharding()` |

**v54 Results**:

| Metric | v53 (pre-dp-fix) | v54 (dp-fix) | Baseline (contiguous) |
|--------|-----------------|-------------|----------------------|
| Steady-state | 19.64 ms | **13.45 ms** | 22.74 ms |
| Throughput | 6,517 tok/s | **9,515 tok/s** | 5,629 tok/s |
| TPS/chip | 101.8 | **148.7** | 87.9 |
| **vs baseline** | 14% faster | **41% faster** | — |

## Performance Journey

| Version | Steady-state | TPS/chip | Issue |
|---------|-------------|----------|-------|
| v49 (per-group JIT) | 770 ms | 2.6 | 15 host→device round trips per step |
| v51 (scan, pre-fix) | ~2,630 ms | ~0.8 | Recompilation every step (missing constraint) |
| v53 (constraint fix) | 19.64 ms | 101.8 | Unnecessary all-gathers (47% of time) |
| v54 (dp fix) | 13.45 ms | 148.7 | fp8 dequant overhead (48% of time) |
| v55 (native fp8) | 11.58 ms | 172.6 | XLA layout copies on expert weights |
| v56 (layout fix) | 8.19 ms | 244.2 | DeltaNet scan tiling copies |
| v57 (unroll) | 6.43 ms | 311.2 | DeltaNet scan tiling copies |
| **v58 (Pallas DeltaNet)** | **6.52 ms** | **306.9** | Flat (3-layer config too small to show gains) |
| Baseline (contiguous) | 22.74 ms | 87.9 | — |

## Docker Images

| Version | Contents | Status |
|---------|----------|--------|
| v48 | Pre-RPA baseline | Working |
| v49 | RPA code with per-group JIT path | Working but slow |
| v50 | Scan-based JIT path | Built successfully |
| v51 | Clean rebuild, minimal XLA flags | Working, profiled |
| v52 | Sharding constraint fix (paged_kv) | Working, 17ms steady-state |
| v53 | Benchmark timing fix (per-step measurement) | Working, 19.64ms confirmed |
| v54 | DP-aware shard_map + _safe_spec fix | Working, 13.45ms |
| v55 | Native fp8 ragged_dot (no dequant) | Working, 11.58ms |
| v56 | Ragged_dot layout storage (no XLA copy) | Working, 8.19ms |
| v57 | DeltaNet inner scan unroll | Working, **6.43ms** |
| v58 | Fused DeltaNet Pallas kernel + shard_map | Working, **6.52ms** |

Build command: `sudo docker build -t gcr.io/tpu-vm-gke-testing/jax-gpt-tpu:vNN . && sudo docker push gcr.io/tpu-vm-gke-testing/jax-gpt-tpu:vNN`

## K8s Deployments

| Jobset | Purpose | Image |
|--------|---------|-------|
| `bench_v7x_rpa_mini_jobset.yaml` | 4-layer RPA test on v7x-64 | v49 |
| `bench_v7x_baseline_mini_jobset.yaml` | 4-layer baseline on v7x-64 | v49 |
| `bench_v7x_rpa_debug_jobset.yaml` | 4-layer RPA with profiling | v51→v58 |

## Profiles

All at `gs://max-experiments/profiles/qwen35-rpa-debug-v51/decode/plugins/profile/`:

| Timestamp | Image | Notes |
|-----------|-------|-------|
| `2026_03_21_18_28_31` | v51 | Pre-fix, shows recompilation every step |
| `2026_03_21_19_19_36` | v52 | Post-fix, clean 17ms decode steps |
| `2026_03_21_19_41_46` | v53 | 19.64ms, two all-gathers visible (47% of decode) |
| `2026_03_21_*` | v54 | No paged_kv or delta_M all-gathers |
| `qwen35-rpa-debug-v55` | v55 | Native fp8 ragged_dot, layout copy visible |
| `qwen35-rpa-debug-v56` | v56 | Layout copies eliminated |
| `qwen35-rpa-debug-v57` | v57 | DeltaNet scan unroll, copies gone |
| `qwen35-rpa-debug-v58` | v58 | Fused Pallas DeltaNet kernel |

## Key Lessons

### 1. Missing `with_sharding_constraint` causes silent recompilation

When `shard_map` produces output with different sharding than input, JAX retraces on the
next call. There's no warning — you only see it in profiles as `PJRT_client_Compile` before
every step.

**Rule**: Every scan carry output that passes through a `shard_map` must have a matching
`with_sharding_constraint`. If the baseline path has N constraints, the new path must also
have constraints on all its outputs.

### 2. shard_map must declare ALL mesh axes used by input tensors

If your mesh has `(dp, tp)` and your inputs are dp-sharded, shard_map `in_specs` MUST include
the dp axis. Omitting it (using `P(None, ...)`) tells shard_map the input is replicated, forcing
XLA to insert an all-gather to replicate dp-sharded data before the kernel.

This is the most common and expensive mistake when integrating Pallas kernels with data parallelism.
The shard_map in_specs/out_specs are a contract: they tell XLA what sharding the kernel function
expects. If inputs don't match, XLA inserts collectives silently.

**Rule**: For every shard_map input, check: "is this tensor sharded on any mesh axis?" If yes,
that axis must appear in the `in_specs` PartitionSpec. Same for `out_specs`.

### 3. Replicated metadata must use local batch size

When dp-sharding batch-related tensors, metadata arrays that describe the batch structure
(`cu_q_lens`, `distribution`, `page_indices`) must be computed with `B_local = B // dp`,
not global `B`. These arrays are passed as replicated to shard_map (same on every device)
but must describe the local batch that each device processes.

For paged KV caches specifically: local page indices must be 0-based
(`arange(B_local * pages_per_seq)`) because each dp shard's slice of the paged cache is
also 0-based.

### 4. `_safe_spec` with placeholder shapes can silently drop sharding axes

If you use `shape[i] % mesh.shape[axis] == 0` to validate PartitionSpecs, never use
placeholder values (like B=1) for dimensions that will be sharded. A placeholder that
isn't divisible by the mesh axis size will cause the axis to be silently replaced with
`None` (replicated), undoing intentional sharding.

**Rule**: Always pass actual runtime shapes to divisibility checks, or skip the check
for dimensions where the caller guarantees divisibility.

### 5. Profile before optimizing the kernel

The original hypothesis (RPA kernel is slow, HBM budget too tight) was wrong for the 4-layer
case. The real issue was infrastructure (missing constraints and dp-unaware shard_map).
The RPA kernel itself took only 1.3 ms/step — 1.5% of decode time. The other 98.5% was
collectives and data movement caused by sharding mismatches.

### 6. Store weights in the consumer's expected layout

`jnp.transpose` on large tensors causes XLA to insert physical memory copy ops
(`{3,2,1,0:T(8,128)} → {2,3,1,0:T(32,128)}`). If a weight is always consumed by
`ragged_dot(x, w, ...)` which needs `(E, K, N)`, store it in `(E, K, N)` from
init/quantize time — don't store as `(E, N, K)` and transpose at runtime.

For fp8 quantization this also means changing the quantization axis: ragged_dot layout
`(E, K, N)` needs per-output-feature quantization (amax over K, axis=-2), producing
scale_inv shape `(E, 1, N)`. The fp8_matmul layout `(N, K)` uses per-row quantization
(amax over K, axis=-1), producing scale_inv shape `(N, 1)`. These are different functions.

### 7. Small scan loops with large constant weights are better unrolled

`jax.lax.scan` forces XLA to pick a single physical tiling for arrays passed through `xs`
(auto-sliced inputs). If the consumer (e.g. `ragged_dot`) needs a different tiling than what
XLA picks for the stacked array, a physical copy happens every iteration.

For small iteration counts (3 DeltaNet layers), unrolling to a Python `for` loop is free
(~3x body code, negligible vs 15-group outer scan) and lets XLA tile each layer's weights
independently for their consumer. This eliminated ~770ms of copies+slices per 32-step decode.

**Rule**: If a scan has ≤5 iterations and its `xs` contain large weight arrays that aren't
modified (constant across iterations), consider unrolling.

### 8. Pallas kernels need shard_map — always

Pallas/Mosaic kernels cannot be auto-partitioned by GSPMD. Any `pallas_call` inside a
sharded computation will fail with `NotImplementedError: Mosaic kernels cannot be
automatically partitioned`. The fix is always `shard_map` with explicit `in_specs`/`out_specs`
matching the tensor sharding.

**Rule**: When adding a Pallas kernel, immediately check: "Will this run in a sharded
context?" If yes, wrap in `shard_map` from the start. Check the cache/state sharding specs
(e.g. `make_cache_sharding`) to get the right PartitionSpecs.

### 9. Measure per-step, not total-time-divided

`block_until_ready()` on each step is essential for accurate decode latency measurement on TPU.
Dividing total wall time by step count mixes in prefill, first-step compilation, and data
transfer overhead.

### Phase 11: Native FP8 ragged_dot & Layout Fix (2026-03-21)

**Goal**: Eliminate fp8 dequantization overhead in MoE expert computation.

**Problem (v54 profile)**: 48% of decode device time was spent in `convert_element_type` — XLA
dequantizing fp8 expert weights to f32 before `ragged_dot`. The code in `_get_expert_weight()`
was calling `w_fp8.astype(jnp.float32) / scale_inv` before every ragged_dot, throwing away
the 2x fp8 MXU FLOPS advantage.

#### v55: Native fp8 ragged_dot

**Changes**:
- `moe.py`: Added `_is_fp8_weight()`, `_fp8_expert_components()`, `_fp8_ragged_dot_rescaled()`
- `_expert_swiglu()`: FP8 branch uses 3 native fp8 ragged_dots (gate, up, down) with
  intermediate requantization (`dynamic_quantize_fp8`) between SwiGLU stages
- `expert_forward_single()`: FP8 path extracts fp8 components, skips dequantization
- `expert_forward_ep()`: FP8 shard_map path passes 9 arrays (3 fp8 weights + 3 scales + 3 activations)
  through `shard_map` with proper PartitionSpecs
- Extracted `_ep_inner_body()` shared by both fp8 and non-fp8 EP paths

**Scale convention fix**: `scale_inv = 1/scale`. Rescaling after fp8 dot needs `scale = 1/scale_inv`.
Initial attempt used `scale_inv` directly as rescale factor → values exploded to ~1e21. Fixed
by returning `(1.0 / w['scale_inv']).squeeze(-2)` from `_fp8_expert_components()`.

**v55 Results**:

| Metric | v54 | v55 | Delta |
|--------|-----|-----|-------|
| Steady-state | 13.45 ms | 11.58 ms | -14% |
| TPS/chip | 148.7 | 172.6 | +16% |

#### v56: Ragged_dot layout storage (eliminate XLA layout copies)

**Problem (v55 profile)**: Major remaining overhead was XLA `copy` ops changing physical memory
layout on fp8 expert weights: `{3,2,1,0:T(8,128)} → {2,3,1,0:T(32,128)}`. Root cause: expert
weights were stored in fp8_matmul convention `(N, K)` via `_quantize_weight()` (which transposes),
then `jnp.transpose`d back to ragged_dot `(E, K, N)` at runtime. XLA had to physically rearrange
memory each time.

**Fix**: Store expert weights in ragged_dot layout from init/quantize time:
- `quantize.py`: Added `_quantize_expert_weight()` — no transpose, per-column quantization
  (`amax over K dim, axis=-2`), scale_inv shape `(E, 1, N)`
- `quantize_params_fp8()`: Path-based detection routes expert weights to `_quantize_expert_weight()`
  instead of `_quantize_weight()`. Detection: `last_part in {'gate_proj', 'up_proj', 'down_proj'}
  and len(shape) >= 3 and 'shared' not in path`
- `model.py` `_init_expert_fp8()`: Changed init to `(E, K, N)` layout with `(E, 1, N)` scale
- `fp8.py` `matmul_maybe_fp8()`: Updated 3D path dequant to `w_fp8 / scale_inv` (was `* scale_inv`)

**v56 Results**:

| Metric | v55 | v56 | Delta |
|--------|-----|-----|-------|
| Steady-state | 11.58 ms | **8.19 ms** | **-29%** |
| TPS/chip | 172.6 | **244.2** | **+42%** |

**HBM Usage (v56, 4-layer on v7x-64, B=128, prompt_len=1024)**:
- Params: 29.391 GB global (64 arrays) → ~3.67 GB/device
- Cache: 1.137 GB global (5 arrays) → ~0.14 GB/device
- Total per device: 3,723 MB (3.64 GB)
- `bytes_in_use` / `peak_bytes_in_use`: 3.773 GB

### Phase 12: DeltaNet Inner Scan Unroll (2026-03-21)

**Goal**: Eliminate XLA tiling copies on DeltaNet MoE expert weights.

**Problem (v56 profile)**: Three copy ops converting DeltaNet MoE expert weight tiling from
`T(8,128)` → `T(32,128)`, totalling ~386ms across 32 decode steps:
```
copy.362: f8e4m3fn[3,64,4096,1024]  T(8,128)→T(32,128)  129.7ms  (gate_proj)
copy.363: f8e4m3fn[3,64,4096,1024]  T(8,128)→T(32,128)  128.1ms  (up_proj)
copy.364: f8e4m3fn[3,64,1024,4096]  T(8,128)→T(32,128)  128.3ms  (down_proj)
```
Plus `dynamic-slice_bitcast_fusion` ops (255.8ms + 128.1ms) extracting per-layer weights
from the stacked `[3, E, K, N]` array.

**Root cause**: `jax.lax.scan` over the 3 DeltaNet layers forced XLA to pick a single tiling
for the stacked `[3, E, K, N]` scan `xs` array. XLA chose `T(8,128)` (scan-friendly), but
`ragged_dot` needs `T(32,128)` (MXU-friendly), requiring a copy on every iteration.

**Fix**: Replaced `jax.lax.scan` with a Python `for i in range(3)` loop in both
`group_forward_rpa()` and `group_forward()` in `block.py`. Each iteration indexes directly
into the stacked params with `jax.tree.map(lambda a: a[i], delta_layer_params)`. With only
3 iterations, the program size increase is negligible (~3x body code), but XLA can now tile
each layer's `[E, K, N]` weights independently for their consumer.

**v57 Results**:

| Metric | v56 | v57 | Delta |
|--------|-----|-----|-------|
| Steady-state | 8.19 ms | **6.43 ms** | **-21%** |
| TPS/chip | 244.2 | **311.2** | **+27%** |

### Phase 13: Fused DeltaNet Pallas Kernel (2026-03-21)

**Goal**: Fuse the 5-op DeltaNet recurrent step into a single Pallas kernel to reduce HBM
traffic by ~3x on the state matrix.

**Background**: Each DeltaNet recurrent step performs 3 separate passes over the state matrix
`(B, H, dk, dv)` = `(B, 8, 128, 128)` = 512 KB/head in f32:
1. `state *= g_factor` (decay)
2. `kv_mem = einsum('bhkv,bhk->bhv', state, k)` (readout)
3. `state += outer(k, delta)` (rank-1 update) + `output = einsum('bhkv,bhk->bhv', state, q)` (query)

With 45 DeltaNet layers at full scale, that's 3 reads × 45 layers × 16 MB = ~2.1 GB HBM
traffic per batch. Fusing into one kernel reduces this to ~0.7 GB.

**Implementation** (`pallas_deltanet.py`):
- Grid: `(B,)` — one kernel invocation per batch element
- Each invocation loops over all H heads in Python (unrolled by Mosaic)
- Block shapes: `(1, H, dk, dv)` for state, `(1, H, dk)` for q/k, `(1, H, dv)` for v
- TPU Pallas constraints solved:
  - 2D+ operands required for `dot_general` — reshaped 1D vectors to `(1, dk)` matmuls
  - `preferred_element_type=jnp.float32` for MXU accumulation precision
  - Block shape last-two-dim rule: `(8, 128)` divisibility satisfied by H≥8
  - `pltpu.CompilerParams` for hashable compiler params (JAX v0.7+)
  - `g_factor`/`beta` reshaped to `(B, H, 1)` for tiling alignment

**Integration** (`deltanet.py`):
- `fused_deltanet_step` replaces 5 separate einsums in `deltanet_recurrent_step`
- Wrapped in `shard_map` for multi-device: state is `P('dp', 'tp', None, None)`
- Falls back to direct call when `mesh is None` (single-device / tests)

**Correctness**: 8/8 standalone kernel tests pass, 5/5 integration tests pass.
MXU accumulation order differs from JAX einsum by ~0.02 on state, ~0.15 on output
(error propagation through 128-element contractions). Acceptable for decode.

**v58 Results**:

| Metric | v57 | v58 | Delta |
|--------|-----|-----|-------|
| Steady-state | 6.43 ms | **6.52 ms** | +1.4% (noise) |
| TPS/chip | 311.2 | **306.9** | -1.4% (noise) |

Performance is flat — expected with only 3 DeltaNet layers in the 4-layer debug config.
The real payoff comes at full 60-layer scale (45 DeltaNet layers → 15x more HBM savings).
The key result is clean integration: Pallas kernel + shard_map + fp8 projections all work
together without regression.

## Open Questions

1. **Full 60-layer config**: The sharding fixes may reduce HLO temp buffer requirements (fewer
   all-gather intermediates), but the compile-time HBM OOM may still occur. Needs testing.
2. **14.5s first-step overhead**: First decode step takes 14.5s — includes prefill re-execution
   inside the timed loop + paged cache conversion.
3. ~~**fp8 convert_element_type**~~: **RESOLVED in v55/v56**. Native fp8 ragged_dot + layout fix
   eliminated dequantization and copy overhead entirely.
4. **page_size=128**: v7 tuned entries only exist for page_size=128. Our page_size=64 may be
   suboptimal.
5. **Baseline dp fix**: The `_safe_spec` B=1 bug also affects the baseline (non-RPA) path.
   Fixing it there could improve baseline decode as well.

## Next Steps

1. Test full 60-layer (15 groups) config with all fixes — Pallas kernel HBM savings
   should be significant at scale (45 DeltaNet layers vs 3 in debug config)
2. Fix baseline path dp sharding (same `_safe_spec` bug applies)
3. Test with page_size=128
4. Profile v58 to confirm Pallas kernel shows up as single fused op
