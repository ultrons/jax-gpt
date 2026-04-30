# Feedback: Pallas EP-MoE Backward Kernel — Integration Findings

**Context**: DSv3 671B on 4×8×8 v7x cluster (EP=8, FSDP=64, GBS=1024, 512 JAX devices).
Attempting to integrate `fused_ep_moe_bwd` (backward kernel from `~/dsv3/mini_dsv3/backward_kernel.py`)
into the training loop in `model.py`. Two separate experiments (v12, v15) were attempted.

---

## Experiment v12: Compile hang (60+ min) — nested JIT bug

### Symptom
`sudo docker logs` showed pods stuck indefinitely at "Compiling (first step)..." — never completing.
Manually killed after 60+ minutes.

### Root cause: `@jax.jit` decorator on `fused_ep_moe_bwd`

The backward wrapper function `fused_ep_moe_bwd` had a `@jax.jit` decorator:

```python
@jax.jit
def fused_ep_moe_bwd(tokens, ...):
    # calls Pallas backward kernel
```

This was placed inside `jax.custom_vjp`, which is itself inside `shard_map`, which is inside the
outer `jax.jit(train_step)`. This creates **nested JIT compilation**:

```
outer jax.jit(train_step)
  └── lax.scan (moe layers)
        └── jax.checkpoint(_moe_layer_body)
              └── moe_layer()
                    └── jax.custom_vjp (fused_ep_moe_bwd wrapper)
                          └── @jax.jit(fused_ep_moe_bwd)  ← PROBLEM
                                └── shard_map (calls Pallas kernel)
```

JAX evaluates `@jax.jit`-decorated functions eagerly when traced inside an outer `jit`. This triggers
a **sub-compilation of the Pallas module** inside the outer compilation, causing XLA to hang trying
to compile the Pallas kernel within the context of the outer compilation.

### Fix
Remove the `@jax.jit` decorator from `fused_ep_moe_bwd`. The function is already compiled as
part of the outer `jax.jit(train_step)` — the decorator is redundant AND harmful. All static
arguments were already handled via `nondiff_argnums`, so removal is safe.

```python
# WRONG: causes nested JIT compile hang
@jax.jit
def fused_ep_moe_bwd(tokens, w1, w2, g, K, ep_axis, ...):
    ...

# CORRECT: no decorator
def fused_ep_moe_bwd(tokens, w1, w2, g, K, ep_axis, ...):
    ...
```

**Rule: Never place `@jax.jit` on a function that will be called inside another `jax.jit` scope,
especially when Pallas kernels are involved.** Pallas kernels compile inline within the enclosing
`jax.jit` context.

---

## Experiment v15: Memory explosion at full 671B scale

After fixing the `@jax.jit` issue, the kernel compiled but generated illegal tensor sizes.

### Config at failure
- EP=8, FSDP=64, GBS=1024
- T_local (per device, inside shard_map) = GBS × S / (EP × FSDP) = 1024 × 4096 / 512 = 8192 tokens
- T_fsdp (after EP all_gather inside shard_map) = GBS × S / FSDP = 65,536 tokens
- max_tpe = 2 × T_fsdp × K / E = 2 × 65,536 × 8 / 256 = 4096 tokens per expert
- E_local = E / EP = 256 / 8 = 32 experts per device
- D = 7168, D_moe = 2048

### Bug 1: `jax.vmap(ffn_one)` creates (TK, D, F) tensor = 247 TB

Inside `backward_kernel.py`, there was a pattern:
```python
def ffn_one(token_idx):
    w_gathered = w1[assigned_expert[token_idx]]  # (D, D_moe) -- weight for this token
    x = tokens[token_idx]                         # (D,)
    return x @ w_gathered                         # (D_moe,)

out = jax.vmap(ffn_one)(jnp.arange(T * K))
```

With T=65,536 and K=8: T×K = 524,288 iterations. `jax.vmap` expands `w1[assigned_expert[token_idx]]`
into a **gather** over T×K indices: the vmapped weight tensor has shape `(T*K, D, D_moe)` =
(524,288, 7168, 2048) = **8.05 TB**. (It was even larger in practice due to the double-precision
or intermediate reshape.)

XLA reports: `bf16[524288, 7168, 2048]` = 7.57 TB → immediately OOMs.

**Root cause**: `vmap` over `w1[i]` where `i` is a dynamic index materializes the full
`(T*K, D, D_moe)` weight tensor even though only T×K ≪ E×D×D_moe values are needed.

**Fix**: The MoE backward must operate **per-expert**, not per-token. Sort the tokens by their
assigned expert, then run each expert's weight matmul on only its allocated tokens. This is
the same pattern as the forward kernel and avoids per-token weight gathers:

```python
for e in range(E_local):
    tids = tokens_assigned_to_expert_e  # (n_e,) — sparse subset
    x_e = tokens[tids]                  # (n_e, D) — only the relevant rows
    dw1_e += x_e.T @ d_gate_e          # (D, D_moe) — expert-specific gradient
    dx_e = d_gate_e @ w1_e.T           # (n_e, D) — scatter back to original positions
```

### Bug 2: `bins_tokens = zeros((E_local × max_tpe, D))` = 120 GB+

The backward pre-allocated a buffer for ALL tokens across ALL experts at static max size:
```python
bins_tokens = jnp.zeros((E_local * max_tpe, D))  # E_local=32, max_tpe=4096, D=7168
```

Shape: (32 × 4096, 7168) = (131,072, 7168) = 3.84 GB.

But there was a bug in max_tpe computation — it was set to `T_fsdp` instead of the correct
`2 × T_fsdp × K / E`. With T_fsdp=65,536 (incorrect max_tpe), the buffer became:
`(32 × 65,536, 7168)` = (2,097,152, 7168) = **30 GB** → OOM.

Even after fixing max_tpe to 4096, the buffer is 3.84 GB per call. Since this is traced inside
`jax.checkpoint` inside `lax.scan`, XLA may allocate it for every scan iteration simultaneously,
multiplying by 29 scan steps → ~111 GB.

**Fix**: This buffer pattern cannot work at training scale. Instead of materializing all experts'
tokens in one buffer, process each expert sequentially (32 iterations) with a static max-size
buffer per expert:
```python
buf = jnp.zeros((max_tpe, D))  # (4096, 7168) = 58.7 MB — per expert, reused
for e in range(E_local):
    buf = buf.at[:n_e].set(tokens[tids_e])  # fill only used slots
    # ... process buf ...
```

---

## Memory budget constraints at DSv3 671B scale

For any backward kernel implementation, these are the hard limits:

| Parameter | Value |
|-----------|-------|
| HBM per device | 102 GB (runtime); 94.75 GB (compile-time limit) |
| VMEM per device | 80 MB (`--xla_tpu_scoped_vmem_limit_kib=81920`) |
| Model weights + optimizer states | ~55 GB/device |
| Available for activations + kernel temps | ~40 GB |

### Key activation sizes (EP=8, FSDP=64, GBS=1024, S=4096)

| Tensor | Shape | Size |
|--------|-------|------|
| `flat_x_ep` (local tokens) | (8192, 7168) | 117 MB |
| `flat_x_fsdp` (after EP gather) | (65,536, 7168) | 940 MB |
| `sel_x_e` (per expert, forward) | (4096, 7168) | 58.7 MB |
| wi_0 or wi_1 per device | (32, 7168, 32) | 14.7 MB |
| wi_01 = concat(wi_0, wi_1) | (32, 7168, 64) | 29.4 MB |
| all expert weights combined | (3 × 32 × ...) | 44 MB → fits in VMEM |
| scan carry (x) | (2, 4096, 7168) | 117 MB |
| scan carry × 29 layers | (29, 2, 4096, 7168) | 3.4 GB |

**VMEM fits**: all expert weights (wi_0 + wi_1 + wo) = 44 MB < 80 MB VMEM limit.
The backward kernel should arrange memory so weights stay in VMEM (= cached) and
the dominant HBM traffic is `sel_x_e` reads and output scatter-adds.

---

## Current workaround: JAX custom_vjp with shard_map

After both v12 and v15 failures, the current production approach in `model.py` uses:
- **Forward**: JAX shard_map with sparse token routing (argsort, per-expert dynamic_slice)
- **Backward**: `jax.vjp` on the shard_map forward inside a `custom_vjp` boundary

This gives ~1037 TPS/chip at v24 (15.8s/step). The backward bottleneck is:
- `scatter_custom_fusion` ops: 1.18s/step at ~87 GB/s HBM BW (16% efficiency)
- These are the backward scatter-adds for token routing

Profile: `gs://sivaibhav-exp/dsv3/profiles/4x8x8-ep8-fsdp64-gbs1024-v24`

---

## Recommendations for backward kernel redesign

### Architecture

A viable Pallas backward kernel for training needs to:

1. **Process per-expert** (not per-token): 32 iterations of (max_tpe=4096, D) × (D, D_moe_local)
   - max_tpe × D is the VMEM working set per expert
   - At max_tpe=4096, D=7168: sel_x_e = 58.7 MB per expert → reads from HBM

2. **Keep weights in VMEM**: wi_0 + wi_1 + wo = 44 MB < 80 MB VMEM limit
   - Load weights ONCE into VMEM, then loop over token chunks

3. **Backward matmuls**: For each expert e:
   - d(sel_x_e) = (d_out_e @ wo_e^T) * d_silu = (max_tpe, D)
   - d(wi_01_e) = sel_x_e^T @ d(gate_up_e) = (D, 2×D_moe_local) — HBM write
   - d(wo_e) = (gate_e*up_e)^T @ d_out_e = (D_moe_local, D) — HBM write
   - Scatter d(sel_x_e) back to grad_flat_x at sorted_tids positions

4. **Avoid accumulating across experts in one buffer**: Don't pre-allocate `(E_local × max_tpe, D)`.
   Use a per-expert buffer that's reused each iteration.

### Memory estimate per kernel call (per shard_map device)
- Inputs: flat_x (940 MB), sel_x_e (58.7 MB per expert, streamed), weights (44 MB in VMEM)
- Outputs: grad_flat_x (940 MB), grad_w0 + grad_w1 + grad_wo (44 MB, accumulated in VMEM)
- Peak HBM temp: 2 × 58.7 MB (forward + backward sel_x_e) = 117 MB per expert
- This is well within budget.

### Integration notes

The current `_moe_jax_ep_fn` in `model.py` has a `custom_vjp` at module level that:
- Stores residuals: `(fx, fi, fw, w0, w1, wout)` per layer
- Backward: calls `jax.vjp` on the shard_map forward

A Pallas backward kernel should replace the `jax.vjp` in `_moe_jax_ep_fn_bwd` with a direct
Pallas backward call. The residuals are already correctly stored.

**Do NOT put `@jax.jit` on any function called from inside `custom_vjp` backward.**

---

## Reference files

| File | Description |
|------|-------------|
| `~/dsv3/mini_dsv3/model.py` | Current production implementation (`_moe_jax_ep_fn` custom_vjp, lines ~811-872) |
| `~/dsv3/specs/fused_ep_moe_backward_spec.md` | Full backward kernel spec (written 2026-03-29) |
| `~/tpu-inference/tpu_inference/kernels/fused_moe/v1/kernel.py` | Production forward kernel |
| Profile v24 | `gs://sivaibhav-exp/dsv3/profiles/4x8x8-ep8-fsdp64-gbs1024-v24` |
