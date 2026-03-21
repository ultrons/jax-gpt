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

The compiled HLO program itself is 3.69 GB, exceeding available HBM (~949 MB free after params + cache).

**Result on 4-layer (1 group)**: Compiles and runs (program ~200 MB estimated).

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

**Goal**: Understand program size and scan behavior.

**Findings on v5p (without shard_map)**:
- Compiled binary: 3-4.5 MB (tiny)
- Program size is flat across 1/2/4 groups — **scan compiles body once, does NOT unroll**
- HLO line counts nearly identical across group counts

**Implication**: The 3.69 GB program on v7x comes from `shard_map` + v7 Pallas codegen, not from scan unrolling. Without shard_map on v5p, the program is 1000x smaller.

### Phase 6: Root Cause Hypotheses

#### Why is the program 3.69 GB with shard_map on v7x?

1. **shard_map lowering**: `shard_map` lowers each Pallas kernel into per-device SPMD code. The v7 Pallas codegen for `ragged_paged_attention` may produce massive per-shard programs.
2. **v7 memory coloring**: On v7, the RPA kernel wraps `pallas_call` in additional `@jax.jit` with `pltpu.with_memory_space_constraint` to pin buffers to HBM. This may increase program size.
3. **MIXED sub-kernel**: `ragged_paged_attention()` always compiles DECODE + MIXED sub-kernels. MIXED does zero work in pure decode (distribution=[B,B,B] gives empty range [B,B)), but the kernel code is still compiled, doubling the program size.

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
| **v54 (dp fix)** | **13.45 ms** | **148.7** | — |
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
| v54 | DP-aware shard_map + _safe_spec fix | Working, **13.45ms** |

Build command: `sudo docker build -t gcr.io/tpu-vm-gke-testing/jax-gpt-tpu:vNN . && sudo docker push gcr.io/tpu-vm-gke-testing/jax-gpt-tpu:vNN`

## K8s Deployments

| Jobset | Purpose | Image |
|--------|---------|-------|
| `bench_v7x_rpa_mini_jobset.yaml` | 4-layer RPA test on v7x-64 | v49 |
| `bench_v7x_baseline_mini_jobset.yaml` | 4-layer baseline on v7x-64 | v49 |
| `bench_v7x_rpa_debug_jobset.yaml` | 4-layer RPA with profiling | v51→v54 |

## Profiles

All at `gs://max-experiments/profiles/qwen35-rpa-debug-v51/decode/plugins/profile/`:

| Timestamp | Image | Notes |
|-----------|-------|-------|
| `2026_03_21_18_28_31` | v51 | Pre-fix, shows recompilation every step |
| `2026_03_21_19_19_36` | v52 | Post-fix, clean 17ms decode steps |
| `2026_03_21_19_41_46` | v53 | 19.64ms, two all-gathers visible (47% of decode) |
| (pending) | v54 | Expected: no paged_kv or delta_M all-gathers |

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

The original hypothesis (RPA kernel is slow, program is too big) was wrong for the 4-layer
case. The real issue was infrastructure (missing constraints and dp-unaware shard_map).
The RPA kernel itself took only 1.3 ms/step — 1.5% of decode time. The other 98.5% was
collectives and data movement caused by sharding mismatches.

### 6. Measure per-step, not total-time-divided

`block_until_ready()` on each step is essential for accurate decode latency measurement on TPU.
Dividing total wall time by step count mixes in prefill, first-step compilation, and data
transfer overhead.

## Open Questions

1. **Full 60-layer config**: The sharding fixes may reduce program size (less all-gather HLO),
   but the 3.69 GB program OOM may still occur. Needs testing.
2. **14.5s first-step overhead**: First decode step takes 14.5s — includes prefill re-execution
   inside the timed loop + paged cache conversion.
3. **fp8 convert_element_type**: Still a significant fraction of decode time. MoE weight
   conversions are the main remaining cost after eliminating the all-gathers.
4. **page_size=128**: v7 tuned entries only exist for page_size=128. Our page_size=64 may be
   suboptimal.
5. **Baseline dp fix**: The `_safe_spec` B=1 bug also affects the baseline (non-RPA) path.
   Fixing it there could improve baseline decode as well.

## Next Steps

1. Test full 60-layer (15 groups) config with all fixes
2. Fix baseline path dp sharding (same `_safe_spec` bug applies)
3. Investigate fp8 conversion overhead
4. Test with page_size=128
