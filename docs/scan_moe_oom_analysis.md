# Scan + MoE Expert Weight OOM: Root Cause Analysis

## Problem Statement

Qwen3.5 60-layer (15-group) decode and prefill OOM on v7x-64 (2x4x4, 64 TCs).
XLA reports `CompileTimeHbmOom (E1000)`:
- Decode: Used 96.15 GB of 94.75 GB budget (1.41 GB over)
- Prefill: Used 101.22 GB of 94.75 GB budget (6.47 GB over)

Root cause: `jax.lax.scan` creates full-size HLO temp copies of stacked MoE
expert weights passed as `xs`.

## The Stacking Mechanism

### Level 1: Within-group stacking (3 DeltaNet layers)

Each group has 3 DeltaNet layers, each with its own MoE. `_stack_tree` in
`model.py:185-187` stacks them with a leading axis of 3:

```python
delta_layers = _stack_tree([
    _init_delta_layer_params(delta_keys[i], config, dtype, fp8)
    for i in range(3)
])
# Expert weight shape: (3, E=512, D=4096, I=1024) per weight type
```

### Level 2: Cross-group stacking (15 groups)

`_stack_tree` at `model.py:198` stacks all 15 groups:

```python
groups = _stack_tree(group_params_list)
```

Final shapes in `params['groups']`:

| Weight | Path | Full Shape | Per-Device (TP=8) | Size |
|--------|------|-----------|-------------------|------|
| Delta gate_proj.w | groups.delta_layers.moe.gate_proj.w | (15, 3, 512, 4096, 1024) | (15, 3, 64, 4096, 1024) fp8 | 12.08 GB |
| Delta up_proj.w | groups.delta_layers.moe.up_proj.w | (15, 3, 512, 4096, 1024) | (15, 3, 64, 4096, 1024) fp8 | 12.08 GB |
| Delta down_proj.w | groups.delta_layers.moe.down_proj.w | (15, 3, 512, 1024, 4096) | (15, 3, 64, 1024, 4096) fp8 | 12.08 GB |
| GQA gate_proj.w | groups.gqa_layer.moe.gate_proj.w | (15, 512, 4096, 1024) | (15, 64, 4096, 1024) fp8 | 4.03 GB |
| GQA up_proj.w | groups.gqa_layer.moe.up_proj.w | (15, 512, 4096, 1024) | (15, 64, 4096, 1024) fp8 | 4.03 GB |
| GQA down_proj.w | groups.gqa_layer.moe.down_proj.w | (15, 512, 1024, 4096) | (15, 64, 1024, 4096) fp8 | 4.03 GB |
| **Total expert weights** | | | | **~48.3 GB** |

Plus fp8 scale_inv tensors (small, ~0.3 GB total).

## How Scan Creates the Copy

In `model.py:318-322`:

```python
scan_inputs = (params['groups'], delta_Ms, delta_convs, paged_kv)
x, (...) = jax.lax.scan(_group_step_rpa, x, scan_inputs)
```

### JAX → XLA lowering

`lax.scan` lowers to an XLA `WhileOp`. Both closed-over consts and `xs` become
`body_consts` — values passed through the while loop body **unchanged** every
iteration. The body does `dynamic_slice(xs, i)` to extract the per-iteration
slice, then returns the full `xs` unchanged:

```
# XLA WhileOp pseudocode:
while (i < 15):
    group_params_i = dynamic_slice(stacked_params, [i, 0, ...])  # small slice
    carry = body(carry, group_params_i)
    return (i+1, carry, stacked_params)  # ← full array passed through
```

### XLA copy insertion

At each iteration boundary, XLA's copy insertion pass decides whether the
output buffer can alias the input buffer. For the pass-through `stacked_params`,
this should be a no-op (it's read-only, never modified). But XLA's copy
insertion is **conservative** — it creates a full physical copy of the ~48 GB
expert weight arrays as HLO temp buffers.

This is a known XLA limitation documented in:
- [jax-ml/jax #16106](https://github.com/jax-ml/jax/discussions/16106): "scan requires a buffer swap (copy) at each iteration boundary"
- [jax-ml/jax #13356](https://github.com/jax-ml/jax/issues/13356): "D2D copies of broadcast constants" — acknowledged as "tracked internally, not trivial to fix"
- [jax-ml/jax #26618](https://github.com/jax-ml/jax/issues/26618): "rolled scan uses ~3x more memory than unrolled"

### Result: weights exist twice in HBM

```
Arguments (original params):     ~48.3 GB
HLO temps (scan copy):           ~48.3 GB
                                  --------
Expert weights alone:             ~96.6 GB  → exceeds 94.75 GB budget
```

## Options Considered

### 1. `donate_argnums` on outer `jit`

**Does not help.** `donate_argnums` operates at the `jax.jit` boundary only.
It tells XLA "the caller won't use this buffer after the call." There is no
equivalent inside `lax.scan`. Both `consts` (closed-over) and `xs` (passed as
arg) become `body_consts` in the WhileOp — same treatment.

### 2. `scan(unroll=True)` or `scan(unroll=N)`

Eliminates the while loop entirely (or reduces iterations). No while loop =
no buffer swap = no copy. But compilation time scales with N — 15 groups ×
MoE with ragged_dot would be very slow to compile, and the unrolled HLO would
be enormous.

### 3. `jax.checkpoint` on scan body

Reduces backward-pass memory by recomputing activations. **No effect on
forward-pass xs buffer copies.** Only useful for training.

### 4. Per-group JIT (Python loop)

`forward_rpa_decode()` in `model.py:420-526` uses a Python for-loop with
separately JITted `group_forward_rpa` calls. Avoids scan entirely. But:
- 15 separate kernel launches with no XLA-level overlap
- Was 34x slower at v49 (untested since)
- Each group is compiled independently — no cross-group optimization

### 5. Pallas HBM-streaming MoE kernel

Replace `moe_layer` with a Pallas kernel that:
- Declares expert weights with `BlockSpec(memory_space=HBM)` — they become
  `Ref` objects, not tensors in any while loop
- Uses `fori_loop` inside the kernel to iterate over experts
- DMAs one expert's weight tile at a time from HBM into VMEM
- Double-buffers DMA to overlap with compute

This is the approach used by vllm's `fused_ep_moe` kernel. Expert weights
exist exactly once in HBM — no scan, no copies, no temps.

### 6. 15 separate JIT-compiled functions with buffer donation

Instead of stacking params into `(15, ...)` and scanning, keep 15 separate
param dicts and JIT 15 function calls with `donate_argnums`. Each function
donates its params after use, freeing HBM for the next group. Peak usage
= 1 group's params (~3.2 GB) instead of all 15 (~48 GB).

**Status:** Untested. See "Unstacked Params with Buffer Donation" section below.

## Unstacked Params with Buffer Donation

### Concept

Instead of:
```python
params['groups'] = _stack_tree(group_params_list)  # (15, ...)
jax.lax.scan(f, init, params['groups'])            # XLA copies full array
```

Keep params as a list of 15 separate dicts:
```python
params['group_0'] = {...}  # single group's params, no leading axis
params['group_1'] = {...}
...
params['group_14'] = {...}
```

Then iterate with donation:
```python
@partial(jax.jit, donate_argnums=(1,))  # donate group params
def step_group(carry, group_params):
    return group_forward(carry, group_params, ...)

for g in range(15):
    carry = step_group(carry, params[f'group_{g}'])
    # group_params buffer is freed after donation
```

### What donation buys us

With stacked scan: all 48.3 GB of expert weights must be live simultaneously
(plus 48.3 GB of copies = 96.6 GB).

With unstacked donation: only the current group's expert weights need to be
live. Each group's expert weights = 48.3 / 15 = **3.22 GB**. After donation,
the buffer is freed and the next group can reuse the memory.

Peak expert weight HBM = ~3.22 GB instead of ~96.6 GB.

### Considerations

1. **Weight loading**: Weights would need to be loaded/materialized one group
   at a time, or all loaded then individually donated. If all 15 are in HBM
   at function start, donation doesn't reduce peak — it only helps if you
   can stream groups in.

2. **Reuse across decode steps**: Donation means the buffer is gone after one
   use. For autoregressive decode (many steps reusing same params), you'd need
   to re-materialize params each step — defeating the purpose.

3. **Compilation**: 15 separate JIT calls = 15 compilations (though identical
   shapes → should cache after first). Still no cross-group XLA optimization.

4. **HBM weight offloading**: Could combine with host→device streaming — keep
   params in host memory, DMA one group at a time to device, donate after use.
   This is essentially what the Pallas HBM-streaming kernel does at a finer
   granularity (per-expert instead of per-group).

### Verdict

Donation helps if params can be streamed rather than pre-loaded. For inference
with repeated decode steps, the Pallas HBM-streaming approach is strictly
better — it keeps weights in HBM permanently but only loads tiles into VMEM
on demand, with no scan/while-loop overhead.

## Recommended Path Forward

**Pallas HBM-streaming MoE kernel** (Option 5):
- Fixes both prefill and decode
- No scan copies, no per-group JIT overhead
- Expert weights exist once in HBM, streamed tile-by-tile to VMEM
- Double-buffered async DMA overlaps with compute
- Pattern proven in vllm's fused_ep_moe kernel

See `docs/rpa_integration_logbook.md` for integration history and
`docs/pallas_moe_kernel_design.md` (TBD) for kernel design.
