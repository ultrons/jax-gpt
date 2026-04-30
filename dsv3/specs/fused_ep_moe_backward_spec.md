# Spec: fused_ep_moe Backward Pass (training support)

## Goal

Add a `jax.custom_vjp` backward pass to `fused_ep_moe` so it can be used inside
`jax.grad` / `jax.value_and_grad` for training in the ds-v3 codebase. Both forward
and backward are Pallas kernels for maximum throughput.

**Target integration:** Replace the `ragged_dot` / `jax` MoE backends in
`mini_dsv3/model.py` with `fused_pallas` for training performance on v7x.

---

## Files involved

| File | Role |
|------|------|
| `~/tpu-inference/tpu_inference/kernels/fused_moe/v1/kernel.py` | Forward Pallas kernel (`fused_ep_moe`) and `ref_moe` reference |
| `~/tpu-inference/tpu_inference/kernels/fused_moe/v1/backward.py` | **New file** — `custom_vjp` wrapper + Pallas backward kernels |
| `~/tpu-inference/tpu_inference/kernels/fused_moe/v1/__init__.py` | Export `fused_ep_moe_train` |
| `~/dsv3/mini_dsv3/model.py` | Integration point — add `fused_pallas` backend to `moe_layer()` |
| `~/dsv3/mini_dsv3/train.py` | Add `--moe_backend=fused_pallas` to CLI choices |
| `~/dsv3/fused_moe_bwd/` | **Self-contained dev directory** — all new code lives here during dev |

---

## Kernel signature corrections (actual vs spec)

The original spec had several mismatches with `kernel.py`. Actual signatures:

```python
# fused_ep_moe — mesh is first positional arg, not keyword
def fused_ep_moe(
    mesh: jax.sharding.Mesh,       # FIRST, positional
    tokens,                         # (T, D)
    w1,                             # (E_local, 2, D, D_moe//2)  ← 2-dim separates gate/up
    w2,                             # (E_local, D_moe//2, D)
    gating_output,                  # (T, E)
    top_k: int,
    *,
    renormalize_topk_logits: bool = False,   # ← not renormalize
    act_fn: str = "silu",
    scoring_fn: str = "softmax",    # ← default is softmax, not sigmoid
    ...
    ep_axis_name: str = "model",    # ← default is "model", not "ep"
) -> jax.Array                      # single tensor, (T, D)

# ref_moe — same signature, no mesh arg
def ref_moe(tokens, w1, w2, gating_output, top_k, *, ...)
```

**w1 shape convention:** `(E_local, 2, D, D_moe//2)` — the `2` dim is `[gate, up]`.
In ds-v3 `model.py`, `wi_0` (gate) and `wi_1` (up) are separate `(E, D, D_moe)` tensors
and must be stacked as `jnp.stack([wi_0, wi_1], axis=1)` before passing to `fused_ep_moe`.

---

## ds-v3 integration context

### Model config (671B full)
| Symbol | Value | Description |
|--------|-------|-------------|
| D | 7168 | Hidden dim |
| D_moe | 2048 | Expert intermediate dim (per EP device after gather) |
| E | 256 | Total experts |
| K | 8 | Top-K |
| L_moe | 58 | MoE layers |
| EP | 8 | Expert parallelism |
| FSDP | 64 | Weight sharding |

### Weight sharding in ds-v3
```python
wi_0: P("ep", "fsdp", None)   # (E, D, D_moe) → per device: (E/ep, D/fsdp, D_moe)
wi_1: P("ep", "fsdp", None)   # same
wo:   P("ep", None, "fsdp")   # (E, D_moe, D) → per device: (E/ep, D_moe, D/fsdp)
```

FSDP gather needed before passing to `fused_ep_moe`:
```python
w1_full = lax.all_gather(wi_0_shard, "fsdp", axis=1, tiled=True)  # restore D dim
w2_full = lax.all_gather(wo_shard,   "fsdp", axis=2, tiled=True)  # restore D dim
```

### Routing in ds-v3
ds-v3 uses **sigmoid** scoring + `gate_bias`, and pre-normalizes top-K weights:
```python
scores = sigmoid(x @ gate_weight + gate_bias)   # (T, E)
top_k_weights, top_k_indices = top_k(scores)
top_k_weights = softmax(top_k_weights) * routed_scaling_factor   # renormalized
```
`gating_output` passed to `fused_ep_moe` is the raw pre-sigmoid logit tensor `(T, E)`.
Use `scoring_fn="sigmoid"` and `renormalize_topk_logits=True`.

### Mesh axis names in ds-v3
```python
Mesh(devices, axis_names=("dp", "ep", "fsdp"))
# ep_axis_name for fused_ep_moe = "ep"  (not the default "model")
```

### Integration point in model.py
```python
# In moe_layer() after routing, add:
elif cfg.moe_backend == "fused_pallas":
    from tpu_inference.kernels.fused_moe.v1.backward import fused_ep_moe_train_fsdp
    w1 = jnp.stack([params["wi_0"], params["wi_1"]], axis=1)  # (E,2,D,D_moe)
    routed_out = fused_ep_moe_train_fsdp(
        cfg.mesh, x, w1, params["wo"],
        gate_logits,          # pre-sigmoid gating output (T, E)
        top_k=cfg.K,
        scoring_fn="sigmoid",
        renormalize_topk_logits=True,
        ep_axis_name="ep",
    )
```

---

## Forward pass data flow

```
tokens (T, D)  [EP-sharded: T/EP tokens per device]
  ↓  gating_output (T, E) → sigmoid → top_k → indices (T,K), weights (T,K)
  ↓  A2A scatter: route tokens to expert devices
  ↓  expert FFN per device:
       h_gate = token @ w1[e, 0]              # (T_expert, D_moe//2)
       h_up   = token @ w1[e, 1]              # (T_expert, D_moe//2)
       hidden = silu(h_gate) * h_up           # SwiGLU
       out    = hidden @ w2[e]                # (T_expert, D)
  ↓  A2A gather: return results to token-owner devices
  ↓  combine: output[t] = sum_k( weight[t,k] * expert_out[t,k] )
output (T, D)
```

---

## Gradient derivations

### 1. Gradient through combine
```
d_e[t,k]  = d_output[t] * w[t,k]       # grad w.r.t. expert outputs
d_w[t,k]  = dot(d_output[t], e[t,k])   # grad w.r.t. routing weights (scalar)
```

### 2. A2A directions reverse
- Forward scatter → Backward gather
- Forward gather  → Backward scatter

### 3. Gradient through expert FFN
```python
# Backward through W2 (down projection)
dW2      = hidden.T @ d_out                     # (D_moe//2, D)
d_hidden = d_out @ W2.T                         # (T_expert, D_moe//2)

# Backward through SwiGLU: hidden = silu(h_gate) * h_up
d_h_up   = d_hidden * silu(h_gate)
d_h_gate = d_hidden * h_up * silu_grad(h_gate)
# silu_grad(x) = silu(x) + x * sigmoid(x) * (1 - sigmoid(x))

# Backward through W1
dW1_gate = token_in.T @ d_h_gate               # (D, D_moe//2)
dW1_up   = token_in.T @ d_h_up                # (D, D_moe//2)
d_token  = d_h_gate @ W1_gate.T + d_h_up @ W1_up.T  # (T_expert, D)
```

### 4. Gradient through routing (gating logits)
```python
# d_w[t,k] → backward through renorm → backward through sigmoid
# Locally computable (no collective needed)
```

---

## Implementation: Full Pallas backward

### custom_vjp interface

```python
# backward.py

@jax.custom_vjp
def fused_ep_moe_train(mesh, tokens, w1, w2, gating_output, top_k, **kwargs):
    return fused_ep_moe(mesh, tokens, w1, w2, gating_output, top_k, **kwargs)

def _fwd(mesh, tokens, w1, w2, gating_output, top_k, **kwargs):
    # Run forward kernel, also returning residuals needed for backward
    out, residuals = fused_ep_moe_with_residuals(
        mesh, tokens, w1, w2, gating_output, top_k, **kwargs)
    return out, (tokens, w1, w2, gating_output, residuals, kwargs)

def _bwd(res, d_out):
    tokens, w1, w2, gating_output, residuals, kwargs = res
    token_in, h_gate, h_up, top_k_indices, top_k_weights, expert_starts = residuals
    # Run Pallas backward kernels
    d_tokens = _bwd_dX_kernel(d_out, w1, w2, token_in, h_gate, h_up,
                               top_k_indices, expert_starts, **kwargs)
    d_w1, d_w2 = _bwd_dW_kernel(d_out, token_in, h_gate, h_up,
                                  top_k_indices, expert_starts, **kwargs)
    d_gating = _bwd_routing(d_out, top_k_indices, top_k_weights,
                             gating_output, **kwargs)
    return (None, d_tokens, d_w1, d_w2, d_gating,
            None, None, None, None, None, None, None, None, None)

fused_ep_moe_train.defvjp(_fwd, _bwd)
```

### FSDP wrapper for ds-v3
```python
def fused_ep_moe_train_fsdp(mesh, tokens, w1, w2, gating_output, **kwargs):
    from jax.experimental.shard_map import shard_map
    @functools.partial(shard_map, mesh=mesh,
                       in_specs=(P(("ep","fsdp"), None),
                                 P("ep","fsdp",None,None),   # w1: (E,2,D,D_moe//2)
                                 P("ep",None,"fsdp"),         # w2: (E,D_moe//2,D)
                                 P(("ep","fsdp"),None)),
                       out_specs=P(("ep","fsdp"), None),
                       check_rep=False)
    def _inner(tok, w1_s, w2_s, gating):
        w1_full = lax.all_gather(w1_s, "fsdp", axis=2, tiled=True)
        w2_full = lax.all_gather(w2_s, "fsdp", axis=2, tiled=True)
        return fused_ep_moe_train(mesh, tok, w1_full, w2_full, gating, **kwargs)
    return _inner(tokens, w1, w2, gating_output)
```

### Residuals to expose from forward kernel

These are computed inside `_fused_ep_moe_kernel` but not currently returned:

| Residual | Shape | Location in kernel |
|----------|-------|-------------------|
| `token_in` | `(T_expert, D)` | `t_b32_vmem` after A2A scatter |
| `h_gate` | `(T_expert, D_moe//2)` | `acc1` in `dynamic_ffn1()` |
| `h_up` | `(T_expert, D_moe//2)` | `acc3` in `dynamic_ffn1()` |
| `top_k_indices` | `(T, K)` | output of `get_top_k()` |
| `top_k_weights` | `(T, K)` | post scoring+renorm in `get_top_k()` |
| `expert_starts` | `(E,)` | output of `all_reduce_metadata()` |

**Kernel modification needed:** add HBM output buffers for each residual and write
them out in the main loop. `fused_ep_moe` signature gains optional `return_residuals`
flag returning `(out, residuals_tuple)` when set.

### New Pallas kernels

**`_bwd_dX_kernel`** — gradient w.r.t. input tokens (~800–1000 lines):
1. A2A scatter `d_out` → expert devices (reverse of forward A2A gather)
2. Per expert: `d_token = d_out @ W2.T`; backward SwiGLU; `d_token += d_h_gate @ W1_gate.T + d_h_up @ W1_up.T`
3. A2A gather `d_token` → token-owner devices

**`_bwd_dW_kernel`** — gradient w.r.t. expert weights (~600–800 lines):
1. Reuse A2A scatter from `_bwd_dX_kernel` (share the pass to avoid duplicate comm)
2. Per expert (local, no cross-device comm): `dW2 = hidden.T @ d_out`, `dW1_gate = token_in.T @ d_h_gate`, `dW1_up = token_in.T @ d_h_up`

### Memory cost of residuals
At GBS=1024, EP=8, D=7168, D_moe=2048:
- `token_in`:  65536 × 7168 × 2 bytes = 939 MB
- `h_gate + h_up`: 65536 × 2048 × 2 × 2 = 537 MB
- Total: ~1.5 GB/layer → with gradient checkpointing keeping 2–3 layers live = ~4.5 GB

---

## Development setup

**Working directory:** `~/dsv3/fused_moe_bwd/` (self-contained, does not touch dsv3 work)

**Local dev hardware:** 4× TPU v4 (this machine). EP=4 for local tests.
All correctness verification runs locally; v7x perf tuning runs on the cluster.

**Python path setup:**
```bash
export PYTHONPATH=~/tpu-inference:$PYTHONPATH
```

---

## Correctness verification

### Stage A — float32, EP=1, no A2A (unit test pure math)
```python
# Single device, bypass EP/shard_map, test gradient math only
# Compare: Tier2 backward grads vs jax.vjp(ref_moe) grads
# Tolerance: rtol=1e-4 in float32
T, D, D_moe, E, K = 64, 256, 128, 4, 2
```

### Stage B — bfloat16, EP=4 (local v4 cluster)
```python
# Run inside shard_map with EP=4 on local 4-chip v4
# Tolerance: rtol=1e-3
```

### Stage C — production shapes on v7x 4x4x4
```python
# D=7168, D_moe=2048, E=256, K=8, T=4096, EP=16 (4x4x4 = 16 nodes)
```

### Gradient finite difference check
```python
jax.test_util.check_grads(fused_ep_moe_train, (mesh, tokens, w1, w2, gating),
                           order=1, modes=["rev"])
```

### Training convergence check
```python
# 10-step loop, confirm loss decreases monotonically
# Use "mini" config: D=2048, E=16, K=4
```

### Gradient norm parity (vs ref_moe vjp)
```python
norm_ratio = ||bwd_grad|| / ||ref_grad||
assert 0.95 < norm_ratio < 1.05
```

---

## Performance targets (v7x 4x4x4)

| Metric | Target |
|--------|--------|
| Backward wall time | ≤ 3× forward |
| A2A scatter+gather overlap | dX and dW share the scatter pass |
| HBM bandwidth utilization | ≥ 60% of peak (roofline) |

---

## Acceptance criteria

| Test | Required |
|------|----------|
| Stage A gradient check passes (rtol=1e-4, float32) | required |
| Stage B gradient check passes (rtol=1e-3, bf16, EP=4) | required |
| `verify_training_step` loss decreases | required |
| Backward wall time ≤ 3× forward on v7x | required |
| Works inside `jax.checkpoint` | required |
| `--moe_backend=fused_pallas` in train.py | required |

---

## Known risks

1. **`ref_moe` vs `fused_ep_moe` arg differences**: `mesh` is positional in `fused_ep_moe`
   but not in `ref_moe`. The `custom_vjp` wrapper must handle this correctly.

2. **EP axis name mismatch**: `fused_ep_moe` defaults to `ep_axis_name="model"`;
   ds-v3 mesh uses `"ep"`. Must pass explicitly.

3. **Residual exposure requires kernel modification**: `token_in`, `h_gate`, `h_up`
   are currently discarded inside the kernel. Modifying the forward to emit them
   as additional outputs changes the kernel's return type — must be opt-in
   (`return_residuals=True`) to avoid breaking inference.

4. **bfloat16 precision**: Run Stage A in float32 to validate math, then Stage B in bf16.

5. **`jax.checkpoint` re-runs forward**: The Pallas forward will be re-executed during
   backward recompute. Confirm Pallas kernel is idempotent in checkpoint scope.

6. **w1 shape stacking in ds-v3**: `wi_0` and `wi_1` are separate `(E,D,D_moe)` tensors.
   Must stack to `(E,2,D,D_moe//2)` — note the `D_moe//2` since each of gate/up is half.
   Verify shape conventions match before integration.
