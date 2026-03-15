# Qwen3.5 Pure JAX Implementation — Detailed Specification

**Target:** Implement the Qwen3.5-397B-A17B inference model in pure JAX, test on 4x TPU v5p with a 32B-param mid config, and be ready to scale to 8+ devices for the full 397B model with FP8 weights.

**Audience:** A developer who knows JAX and wants to build this from scratch.

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [HuggingFace Reference](#2-huggingface-reference)
3. [Pure JAX Design Principles](#3-pure-jax-design-principles)
4. [Module-by-Module Specification](#4-module-by-module-specification)
5. [Sharding Strategy](#5-sharding-strategy)
6. [FP8 Inference](#6-fp8-inference)
7. [Pitfalls and Bugs We Hit](#7-pitfalls-and-bugs-we-hit)
8. [Testing Strategy](#8-testing-strategy)
9. [TPU Deployment](#9-tpu-deployment)
10. [Memory Budget](#10-memory-budget)

---

## 1. Architecture Overview

Qwen3.5-397B-A17B is a **hybrid-attention sparse MoE** language model. "A17B" means 17B parameters are activated per token (out of 397B total) due to MoE sparsity.

### Layer Structure

60 layers organized in 15 groups of 4. Each group:

```
Layer 0: pre-norm → Gated DeltaNet (linear attention) → residual → pre-norm → MoE → residual
Layer 1: pre-norm → Gated DeltaNet (linear attention) → residual → pre-norm → MoE → residual
Layer 2: pre-norm → Gated DeltaNet (linear attention) → residual → pre-norm → MoE → residual
Layer 3: pre-norm → GQA (full softmax attention)      → residual → pre-norm → MoE → residual
```

The `full_attention_interval = 4` — every 4th layer uses full attention, the rest use linear attention.

### Full Model Dimensions

```
d_model:                    4096
vocab_size:                 248320
n_layers:                   60 (15 groups × 4)
max_position_embeddings:    262144

DeltaNet (45 layers):
  n_qk_heads:               16
  n_v_heads:                 64
  qk_head_dim:               128
  v_head_dim:                128
  conv_kernel:               4 (causal depthwise conv1d)
  QK grouping:               64 / 16 = 4 (each QK head maps to 4 V heads)

GQA (15 layers):
  n_q_heads:                 32
  n_kv_heads:                2
  head_dim:                  256
  partial_rotary_factor:     0.25 (RoPE on first 64 of 256 dims)
  rope_theta:                10,000,000
  output_gate:               True (q_proj outputs 2x, second half is sigmoid gate)
  QK norm:                   RMSNorm per-head on Q and K

MoE (all 60 layers):
  n_routed_experts:          512
  n_experts_per_token:       10
  moe_intermediate_size:     1024
  shared_expert_intermediate_size: 1024
  activation:                SwiGLU (silu(gate) * up)
  routing:                   softmax → top-k → renormalize
  shared_expert_gate:        sigmoid gate on shared expert output

Normalization:               RMSNorm with (1 + weight) convention (weight init to 0)
rms_norm_eps:                1e-6
```

---

## 2. HuggingFace Reference

The authoritative reference is the HF transformers implementation:

```
transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py
transformers/models/qwen3_5/modeling_qwen3_5.py  (DeltaNet functions)
```

### Key Classes and Functions

| HF Class/Function | What it does |
|---|---|
| `Qwen3_5GatedDeltaNet` | DeltaNet linear attention module |
| `torch_recurrent_gated_delta_rule()` | Recurrent step (decode) |
| `torch_chunk_gated_delta_rule()` | Chunked prefill |
| `torch_causal_conv1d_update()` | Conv1d single-step update |
| `Qwen3_5MoeAttention` | GQA with output gate and QK norm |
| `Qwen3_5MoeExperts` | Expert weights with fused gate_up_proj |
| `Qwen3_5MoeTopKRouter` | softmax → top-k → renormalize routing |
| `Qwen3_5MoeSparseMoeBlock` | Full MoE layer with shared expert gate |
| `Qwen3_5RMSNormGated` | RMSNorm * silu(gate) for DeltaNet output |
| `Qwen3_5MoeRMSNorm` | Standard RMSNorm with `(1 + weight)` |

### HF FP8 Model

`Qwen/Qwen3.5-397B-A17B-FP8` — 94 safetensors shards, ~406 GB.

Each weight has a paired `weight_scale_inv` parameter. The FP8 model stores experts
individually (`experts.0.gate_proj.weight`) not fused. The non-FP8 model fuses gate+up
into `experts.gate_up_proj`.

Weight param prefix in FP8 model: `model.language_model.layers.{i}.*`
Weight param prefix in bf16 model: `model.layers.{i}.*`

### HF Config

```python
from transformers import AutoConfig
cfg = AutoConfig.from_pretrained('Qwen/Qwen3.5-397B-A17B', trust_remote_code=True)
text_cfg = cfg.text_config  # Qwen3_5MoeTextConfig
```

You can instantiate a mini HF model for comparison testing:

```python
from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeTextConfig
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeForCausalLM

hf_cfg = Qwen3_5MoeTextConfig(
    hidden_size=256, num_hidden_layers=4,
    num_attention_heads=4, num_key_value_heads=1, head_dim=64,
    linear_num_key_heads=2, linear_num_value_heads=4,
    linear_key_head_dim=64, linear_value_head_dim=64,
    linear_conv_kernel_dim=4,
    num_experts=2, num_experts_per_tok=1,
    moe_intermediate_size=128, shared_expert_intermediate_size=128,
    vocab_size=1024, max_position_embeddings=128,
)
model = Qwen3_5MoeForCausalLM(hf_cfg)
```

---

## 3. Pure JAX Design Principles

### Why Not Flax NNX

NNX's `nnx.scan` for layer iteration creates trace-level conflicts with `jax.lax.scan`
for the decode loop. This makes it impossible to compile the entire decode into a single
HLO program. Pure JAX has no such limitation.

### Core Pattern

Every module is a **pure function**: `f(params, x, state) -> (y, new_state)`

- **Params**: nested dicts of JAX arrays. No Variable wrappers, no classes.
- **State**: dataclasses registered with `jax.tree_util.register_dataclass`.
- **Forward pass**: `forward(params, tokens, config, cache) -> (logits, new_cache)`

This makes `jax.jit`, `jax.lax.scan`, and `jax.grad` trivially composable.

### Param Pytree Structure

```python
{
    'embed': (vocab_size, d_model),
    'groups': {                              # leading axis: n_groups (for outer scan)
        'delta_layers': {                    # leading axes: n_groups, 3 (for inner scan)
            'attn_norm': (n_groups, 3, d_model),
            'attn': {
                'in_proj_qkv': (n_groups, 3, d_model, conv_dim),
                'in_proj_z': ...,
                'in_proj_b': ...,
                'in_proj_a': ...,
                'conv_weight': ...,
                'A_log': ...,
                'dt_bias': ...,
                'norm_weight': ...,
                'out_proj': ...,
            },
            'moe_norm': ...,
            'moe': { ... },
        },
        'gqa_layer': {                       # leading axis: n_groups only
            'attn_norm': (n_groups, d_model),
            'attn': {
                'q_proj': (n_groups, d_model, n_q_heads * head_dim * 2),  # query + gate
                'k_proj': ..., 'v_proj': ..., 'o_proj': ...,
                'q_norm': ..., 'k_norm': ...,
            },
            'moe_norm': ...,
            'moe': { ... },
        },
    },
    'final_norm': (d_model,),
    'lm_head': (d_model, vocab_size),
}
```

The leading axes come from `jnp.stack` during `init_params`. `lax.scan` slices along
these axes automatically inside the scan body.

### Two-Level Scan

```python
# Outer: scan over n_groups
x, (cache_outputs) = jax.lax.scan(_group_step, x, (group_params, cache_slices))

# Inner (inside group_forward): scan over 3 DeltaNet layers
x, (new_delta_states) = jax.lax.scan(_delta_step, x, (delta_params, delta_states))
# Then: 1 GQA layer call (not scanned)
x, new_gqa_k, new_gqa_v = gqa_layer_forward(x, gqa_params, ...)
```

---

## 4. Module-by-Module Specification

### 4.1 RMSNorm (`primitives.py`)

**Qwen3.5 convention**: weight initialized to **0**, applied as `(1 + weight)`.

```python
def rms_norm(x, weight, eps=1e-6):
    # Cast to float32 for numerical stability
    x_f32 = x.astype(float32)
    rms = sqrt(mean(x_f32 ** 2, axis=-1, keepdims=True) + eps)
    normed = x_f32 / rms
    return (normed * (1.0 + weight.astype(float32))).astype(x.dtype)
```

**Warning**: If you initialize weight to 1 and use `weight * normed` (standard convention),
your outputs will be 2x the HF reference.

### 4.2 RoPE (`primitives.py`)

Qwen3.5 uses **partial rotary**: only the first 25% of head_dim gets RoPE.

```python
rope_dim = int(head_dim * 0.25)  # 64 for head_dim=256
freqs = precompute_rope_freqs(rope_dim, max_seq_len, theta=10_000_000)
x_out = apply_rotary_emb(x, freqs, rope_dim)
# First rope_dim dims are rotated, the rest pass through unchanged
```

The rotation formula (for each pair of dims):
```
out[0] = x[0] * cos - x[1] * sin
out[1] = x[1] * cos + x[0] * sin
```

Reshape `(..., rope_dim)` → `(..., rope_dim//2, 2)`, apply rotation, reshape back.

**Pitfall**: When reshaping back from `(..., rope_dim//2, 2)` to `(..., rope_dim)`,
use `shape[:-2]` not `shape[:-1]` to avoid including the pair dimension.

### 4.3 Gated DeltaNet Linear Attention (`deltanet.py`)

This is the most complex module. Two modes:

#### Recurrent Step (decode — single token)

```
Inputs:
    x: (B, 1, D) — single token
    state: (B, n_v_heads, qk_head_dim, v_head_dim) — recurrent memory
    conv_state: (B, conv_dim, conv_kernel) — sliding window

Projections:
    mixed_qkv = x @ in_proj_qkv              # (B, 1, key_dim*2 + value_dim)
    z = x @ in_proj_z                         # (B, 1, value_dim) — output gate
    b = x @ in_proj_b                         # (B, 1, n_v_heads) — beta gate
    a = x @ in_proj_a                         # (B, 1, n_v_heads) — decay input

Causal Conv1d (depthwise, width=4):
    mixed_qkv → channels-first → shift conv_state left, append new → depthwise conv → silu
    Output: (B, conv_dim, 1), updated conv_state

Split into Q, K, V:
    q: (B, 1, n_qk_heads, qk_head_dim)
    k: (B, 1, n_qk_heads, qk_head_dim)
    v: (B, 1, n_v_heads, v_head_dim)

Compute decay and gate:
    beta = sigmoid(b)                          # (B, n_v_heads)
    g = -exp(A_log) * softplus(a + dt_bias)    # (B, n_v_heads) — NEGATIVE decay

    ⚠️  HF formula is: g = -A_log.exp() * softplus(a + dt_bias)
        NOT: g = -exp(-exp(A_log)) * softplus(...)
        This was a bug that took us a while to find.

L2 normalize Q, K (eps=1e-6), then scale Q:
    q = l2_norm(q) * (qk_head_dim ** -0.5)
    k = l2_norm(k)

Repeat Q, K for grouped heads (n_v_heads // n_qk_heads = 4):
    q = repeat(q, groups, axis=heads)  # n_qk_heads → n_v_heads
    k = repeat(k, groups, axis=heads)

Recurrent update (squeeze sequence dim, now per-token):
    g_factor = exp(g)[..., None, None]         # (B, H, 1, 1)
    state = state * g_factor                   # decay old state
    kv_mem = einsum('bhkv,bhk->bhv', state, k) # retrieve from memory
    delta = (v - kv_mem) * beta[..., None]     # gated correction
    state = state + einsum('bhk,bhv->bhkv', k, delta)  # rank-1 update
    output = einsum('bhkv,bhk->bhv', state, q) # query the state

Gated RMSNorm on output:
    output = rms_norm(output) * (1 + norm_weight) * silu(z)
    # Applied per-head: reshape to (B*n_v_heads, v_head_dim) first

Output projection:
    output = output @ out_proj                 # (B, 1, D)
```

#### Prefill (full sequence via lax.scan)

Same projections and conv1d (full causal conv), then the recurrent computation runs
via `jax.lax.scan` over the time dimension:

```python
def _step(state, inputs):
    q_t, k_t, v_t, beta_t, g_t = inputs
    state = state * exp(g_t)[..., None, None]
    kv_mem = einsum('bhkv,bhk->bhv', state, k_t)
    delta = (v_t - kv_mem) * beta_t[..., None]
    state = state + einsum('bhk,bhv->bhkv', k_t, delta)
    o_t = einsum('bhkv,bhk->bhv', state, q_t)
    return state, o_t

final_state, outputs = jax.lax.scan(_step, initial_state, scan_inputs)
```

**Verification**: Prefill final state must match sequential recurrent application.
This is a critical correctness test.

#### Causal Conv1d

Prefill: use `jax.lax.conv_general_dilated` with left-padding and `feature_group_count=conv_dim` (depthwise). Apply silu.

Decode: shift state left, append new input, dot with weights, silu.

```python
# Prefill
x_padded = pad(x, ((0,0), (0,0), (kernel-1, 0)))  # left-pad
out = conv_general_dilated(x_padded, w[:, None, :], strides=(1,),
                           padding='VALID', feature_group_count=conv_dim)
out = silu(out)
final_state = x[..., -kernel:]  # save last kernel values for subsequent decode

# Decode (single step)
new_state = concatenate([state[..., 1:], x], axis=-1)  # shift left + append
out = sum(new_state * weight[None, :, :], axis=-1, keepdims=True)  # depthwise
out = silu(out)
```

### 4.4 GQA Attention (`gqa.py`)

Qwen3.5 GQA has three non-standard features:

1. **q_proj outputs 2x** — second half is a sigmoid output gate
2. **QK normalization** — RMSNorm applied per-head to Q and K after projection
3. **Output gate** — `attn_output *= sigmoid(gate)`

```
q_gate = x @ q_proj                  # (B, T, n_q_heads * head_dim * 2)
q_gate = reshape(B, T, n_q_heads, head_dim * 2)
q = q_gate[..., :head_dim]
gate = q_gate[..., head_dim:].reshape(B, T, n_q_heads * head_dim)

k = x @ k_proj                       # (B, T, n_kv_heads * head_dim)
v = x @ v_proj

# QK norm (per-head RMSNorm)
q = rms_norm(q, q_norm_weight)        # q shape: (B, T, n_q_heads, head_dim)
k = rms_norm(k, k_norm_weight)        # k shape: (B, T, n_kv_heads, head_dim)

# Transpose to (B, heads, T, head_dim)
# Apply RoPE (partial: first 64 of 256 dims)
# KV cache update via dynamic_update_slice
# Expand KV heads for GQA (repeat n_q_heads // n_kv_heads times)
# Standard scaled dot-product attention with causal mask
# Transpose back to (B, T, n_q_heads * head_dim)

# Output gate
out = out * sigmoid(gate)

# Output projection
out = out @ o_proj
```

### 4.5 MoE Layer (`moe.py`)

#### Routing

```python
# HF routing: softmax FIRST, then top-k, then renormalize
logits = x @ gate_weight              # (M, n_experts)
probs = softmax(logits, axis=-1)      # full softmax over all experts
top_k_values, top_k_indices = top_k(probs, k)
expert_weights = top_k_values / sum(top_k_values, axis=-1, keepdims=True)
```

**Important**: HF does `softmax → top_k → renormalize`. NOT `top_k → softmax`.

#### Expert Computation with ragged_dot

```python
# Sort tokens by expert assignment
flat_token_ids = repeat(arange(M), k)        # (M*k,)
flat_expert_ids = top_k_indices.reshape(-1)   # (M*k,)
sort_order = argsort(flat_expert_ids)
sorted_token_ids = flat_token_ids[sort_order]
group_sizes = zeros(n_experts).at[flat_expert_ids].add(1)

# Gather sorted tokens
x_sorted = x[sorted_token_ids]               # (M*k, D)

# SwiGLU via three ragged_dots
gate_out = silu(ragged_dot(x_sorted, gate_proj, group_sizes))  # (M*k, I)
up_out = ragged_dot(x_sorted, up_proj, group_sizes)            # (M*k, I)
hidden = gate_out * up_out
expert_out = ragged_dot(hidden, down_proj, group_sizes)         # (M*k, D)

# Scatter back with routing weights
sorted_weights = flat_expert_slot_weights[sort_order]
weighted_out = expert_out * sorted_weights[:, None]
output = zeros(M, D).at[sorted_token_ids].add(weighted_out)
```

`jax.lax.ragged_dot(lhs, rhs, group_sizes)`:
- lhs: `(M, K)`, rhs: `(G, K, N)`, group_sizes: `(G,)`
- First `group_sizes[0]` rows of lhs multiplied by `rhs[0]`, etc.
- Returns `(M, N)`

#### Shared Expert

Standard SwiGLU MLP applied to all tokens, with a learned sigmoid gate:

```python
shared_out = down @ (silu(x @ gate) * (x @ up))
shared_out = sigmoid(x @ shared_expert_gate_weight) * shared_out
output = routed_out + shared_out
```

#### Weight Shapes for ragged_dot

Our convention (for `ragged_dot`): `gate_proj, up_proj: (E, D, I)`, `down_proj: (E, I, D)`

HF stores:
- `gate_up_proj`: `(E, 2*I, D)` — fused, PyTorch (out, in) convention
- `down_proj`: `(E, D, I)` — PyTorch (out, in) convention

Conversion:
```python
# gate_up_proj (E, 2*I, D) → split + transpose
gate_hf = gate_up_proj[:, :I, :]     # (E, I, D)
up_hf = gate_up_proj[:, I:, :]       # (E, I, D)
gate_proj = transpose(gate_hf, (0, 2, 1))  # (E, D, I) for ragged_dot
up_proj = transpose(up_hf, (0, 2, 1))

# down_proj (E, D, I) → transpose
down_proj = transpose(down_proj_hf, (0, 2, 1))  # (E, I, D) for ragged_dot
```

### 4.6 Cache (`cache.py`)

```python
@dataclass
class HybridCache:
    delta_M:     jax.Array   # (n_groups, 3, B, n_v_heads, qk_head_dim, v_head_dim)
    delta_conv:  jax.Array   # (n_groups, 3, B, conv_dim, conv_kernel)
    gqa_k:       jax.Array   # (n_groups, B, n_kv_heads, max_len, head_dim)
    gqa_v:       jax.Array   # (n_groups, B, n_kv_heads, max_len, head_dim)
    pos:         jax.Array   # scalar int32

# Register as pytree
jax.tree_util.register_dataclass(HybridCache, data_fields=[...], meta_fields=[])
```

All arrays have a leading `n_groups` axis so `lax.scan` can slice per-group.
DeltaNet layers additionally have a `3` axis (3 DeltaNet layers per group).

DeltaNet state M is **fixed-size** regardless of sequence length.
GQA KV cache grows with sequence length, updated via `dynamic_update_slice`.

### 4.7 Block (`block.py`)

```python
def group_forward(x, group_params, delta_Ms, delta_convs,
                  gqa_k, gqa_v, cache_pos, config, rope_freqs, is_decode):
    # Inner scan over 3 DeltaNet layers
    x, (new_Ms, new_convs) = lax.scan(
        _delta_step, x, (delta_layer_params, delta_Ms, delta_convs))

    # 1 GQA layer (not scanned)
    x, new_gqa_k, new_gqa_v = gqa_layer_forward(x, gqa_params, ...)

    return x, new_Ms, new_convs, new_gqa_k, new_gqa_v
```

Each layer (DeltaNet or GQA) follows the same pattern:
```
x = x + attn(rms_norm(x))    # pre-norm attention with residual
x = x + moe(rms_norm(x))     # pre-norm MoE with residual
```

### 4.8 Model (`model.py`)

```python
def forward(params, tokens, config, cache=None, is_decode=False, cache_sharding=None):
    x = params['embed'][tokens]   # (B, T, D) — no positional embedding (RoPE handles it)
    rope_freqs = precompute_rope_freqs(...)

    # Outer scan over groups
    def _group_step(carry, group_inputs):
        x, = carry  # carry is just the hidden state
        # ... call group_forward
        # Apply cache sharding constraints if provided
        if cache_sharding is not None:
            new_dM = lax.with_sharding_constraint(new_dM, cache_sharding['delta_M'])
            ...
        return x, (cache_outputs)

    x, cache_outputs = lax.scan(_group_step, x, scan_inputs)

    x = rms_norm(x, params['final_norm'])
    logits = x @ params['lm_head']
    return logits, new_cache
```

#### lax.scan Decode Loop

The key advantage of pure JAX:

```python
def generate(params, prompt, config, max_new_tokens, key):
    cache = init_cache(...)
    logits, cache = forward(params, prompt, config, cache=cache, is_decode=False)
    first_token = argmax(logits[:, -1, :], axis=-1)

    def _decode_step(carry, _):
        token, cache, key = carry
        logits, cache = forward(params, token[:, None], config, cache=cache, is_decode=True)
        key, subkey = jax.random.split(key)
        next_token = argmax(logits[:, 0, :], axis=-1)
        return (next_token, cache, key), next_token

    _, generated = lax.scan(_decode_step, (first_token, cache, key), None, length=max_new_tokens)
```

This compiles to a **single HLO program**. On TPU, the device runs all decode steps
back-to-back with zero host interaction.

---

## 5. Sharding Strategy

### Logical Axis Annotation

Every parameter gets a logical axis name. Physical mesh mapping is a separate dict.

```python
LOGICAL_AXES = {
    'embed':               P('vocab', 'embed'),
    'lm_head':             P('embed', 'vocab'),
    'attn.in_proj_qkv':   P(None, 'delta_v_heads'),
    'attn.out_proj':       P('delta_v_heads', None),
    'attn.q_proj':         P(None, 'gqa_q_heads'),
    'attn.o_proj':         P('gqa_q_heads', None),
    'moe.gate_proj':       P('experts', None, None),
    'moe.shared_gate_proj': P(None, None),  # replicated
    ...
}
```

### Two Configs

**Config A** (mixed): EP=8, TP=8 for DeltaNet, GQA replicated
```python
AXIS_RULES_A = {
    'delta_v_heads': 'tp', 'delta_qk_heads': 'tp',
    'gqa_q_heads': None,   'gqa_kv_heads': None,
    'experts': 'tp',       'vocab': 'tp',
}
```

**Config B** (uniform): TP=8 for everything
```python
AXIS_RULES_B = {
    'delta_v_heads': 'tp', 'delta_qk_heads': 'tp',
    'gqa_q_heads': 'tp',   'gqa_kv_heads': None,
    'experts': 'tp',       'vocab': 'tp',
}
```

### Handling Stacked Params

Params have leading scan axes `(n_groups, n_delta_per_group, ...)` from `jnp.stack`.
The logical spec describes only the trailing "per-param" axes. Pad with `None`:

```python
def pad_spec_to_ndim(spec, ndim):
    return P(*([None] * (ndim - len(spec))), *spec)
```

### Divisibility Fallback

If a sharded dim isn't divisible by the mesh axis size, fall back to replicated:
```python
if shape[i] % mesh.shape[axis] != 0:
    spec_axis = None  # replicate instead of shard
```

### Cache Sharding Constraint

XLA's GSPMD propagates sharding through the computation. When params are TP-sharded
and GQA has `n_kv_heads=2`, XLA may try to shard the output cache along that dim —
but 2 doesn't divide TP=8.

Fix: `jax.lax.with_sharding_constraint` on cache outputs inside the scan body:

```python
# Inside _group_step:
if cache_sharding is not None:
    new_dM = lax.with_sharding_constraint(new_dM, cache_sharding['delta_M'])
    new_gk = lax.with_sharding_constraint(new_gk, cache_sharding['gqa_kv'])
```

Build the specs with `make_cache_sharding(config, mesh, rules)` which applies
the same divisibility check.

---

## 6. FP8 Inference

### Approach

Load official HF FP8 weights as `float8_e4m3fn`. At matmul time:
1. Dynamically quantize activation to fp8 per-row
2. Run fp8 × fp8 dot with float32 accumulation
3. Rescale output by activation_scale × weight_scale

On TPU v5p+, XLA emits native fp8 MXU ops (2x FLOPS of bf16).

### Implementation

```python
FP8_MAX = 448.0  # jnp.finfo(float8_e4m3fn).max

def dynamic_quantize_fp8(x):
    amax = max(abs(x), axis=-1, keepdims=True)
    scale = maximum(amax / FP8_MAX, tiny)
    x_fp8 = (x / scale).astype(float8_e4m3fn)
    return x_fp8, scale

def fp8_matmul(x, w_fp8, w_scale_inv):
    x_fp8, x_scale = dynamic_quantize_fp8(x)
    out = lax.dot_general(x_fp8, w_fp8, ..., preferred_element_type=float32)
    w_scale = 1.0 / w_scale_inv
    return out * x_scale * w_scale
```

### HF FP8 Weight Format

- `weight`: `(out_features, in_features)` as `float8_e4m3fn`
- `weight_scale_inv`: `(out_features, 1)` as `float32`
- Experts stored individually (not fused): `experts.0.gate_proj.weight`

### ragged_dot with FP8

`jax.lax.ragged_dot` supports fp8 operands with `preferred_element_type=float32`.
Per-expert rescaling needs careful handling since different token groups use different
expert weights (and scales).

---

## 7. Pitfalls and Bugs We Hit

These are things that cost us debugging time. Save yourself the trouble.

### 7.1 DeltaNet Decay Formula

**Wrong**: `A = exp(-exp(A_log)); g = -A * softplus(a + dt_bias)`
**Right**: `g = -exp(A_log) * softplus(a + dt_bias)`

The HF code is `g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)`.
`A_log.exp()` is just `exp(A_log)`, NOT `exp(-exp(A_log))`.

### 7.2 RMSNorm (1+weight) Convention

Qwen3.5 initializes RMSNorm weight to **0** and applies as `(1 + weight)`:
```python
output = normed * (1.0 + self.weight.float())
```
If you use standard `weight * normed` with weight=1, your outputs will be correct
but you can't load HF weights (which are 0-initialized).

### 7.3 GQA q_proj is 2x Wide

`q_proj` output dim is `n_q_heads * head_dim * 2`, not `n_q_heads * head_dim`.
The second half is the sigmoid output gate. Split after projection:
```python
q_gate = (x @ q_proj).reshape(B, T, n_q_heads, head_dim * 2)
q = q_gate[..., :head_dim]
gate = q_gate[..., head_dim:]
```

### 7.4 Gated RMSNorm Applied Per-Head

The DeltaNet output gated norm has weight shape `(v_head_dim,)` — it's applied
per-head, not across all heads. Reshape to `(B * n_v_heads, v_head_dim)` before
applying, then reshape back.

**Wrong**: `output.reshape(B, value_dim)` with `norm_weight.shape = (v_head_dim,)`
**Right**: `output.reshape(B * n_v_heads, v_head_dim)` then norm then reshape back

### 7.5 jnp.rsqrt Removed in JAX 0.9.1

`jnp.rsqrt` is gone. Use `jax.lax.rsqrt` instead.

### 7.6 MoE Routing Order

HF: `softmax(logits) → top_k → renormalize`
NOT: `top_k(logits) → softmax`

### 7.7 RoPE Reshape Bug

When converting from `(..., rope_dim//2, 2)` back to `(..., rope_dim)`:
**Wrong**: `.reshape(*shape[:-1], rope_dim)` — includes the pair dim in shape
**Right**: `.reshape(*shape[:-2], rope_dim)` — excludes both pair dims

### 7.8 Cache Position Gating in GQA

Gate KV cache behavior on `cache_pos is not None`, NOT `cache_k is not None`.
When using `lax.scan`, dummy cache arrays are passed (not None) even in no-cache mode.

### 7.9 Numerical Diff: DeltaNet Recurrent Scan

`lax.scan` and PyTorch's explicit loop accumulate float32 rounding differently
through the recurrent state update. Expect ~0.02 per-layer diff, ~0.07 total
after 4 layers. This is normal and acceptable for inference — it's not a bug.

MoE matches exactly (0.0 diff) on identical inputs.

---

## 8. Testing Strategy

### Unit Tests (per module)

| Module | Key Tests |
|--------|-----------|
| primitives | RMSNorm shape/scale, RoPE norm-preservation/partial-rotary, SwiGLU shape/no-NaN |
| deltanet | Recurrent step shape/no-NaN, prefill shape, **prefill-matches-recurrent** (critical), state decay |
| gqa | Shape with/without cache, cache incremental consistency, no-NaN, causality |
| moe | Routing shapes/weights-sum-to-1/group-sizes-sum, shared expert, full layer shape/no-NaN |
| block | Group forward shape/no-NaN |
| model | Forward with/without cache, decode step, generate shape, no-NaN |
| fp8 | Dynamic quantize roundtrip, fp8_matmul accuracy/shape, ragged_dot fp8 |
| sharding | Logical axes cover all params, spec resolution for each component, shard_params runs |

### HF Comparison Test

Instantiate a mini HF model, load weights into JAX via `weight_loader`, compare
forward pass logits. Expected: max_diff < 1.0, mean_diff < 0.15 over 4 layers
(DeltaNet recurrent accumulation).

### Multi-Device Sharding Tests

Use `XLA_FLAGS=--xla_force_host_platform_device_count=8` to simulate 8 CPU devices.

Test:
- Params shard correctly on 4 and 8 devices
- Sharded forward (no cache) matches unsharded output
- Config A and Config B produce identical outputs
- Forward with cache + sharding constraint works

### Benchmarks

Three benchmarks:
1. **Prefill**: measure prompt processing latency
2. **Decode (lax.scan)**: single HLO decode loop — the key advantage
3. **Decode (Python loop)**: per-step dispatch for comparison

---

## 9. TPU Deployment

### Setup

```bash
pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
python -c "import jax; print(jax.device_count(), 'TPU devices')"
```

### Benchmark Script

```bash
# Mid config (32B params) on 4x TPU v5p
python scripts/qwen35_benchmark.py \
    --config mid --sharding B --devices 4 \
    --batch-size 1 --prompt-len 512 --decode-steps 128 \
    --dtype bfloat16

# With profiling
PROFILE_DIR=/tmp/qwen35 python scripts/qwen35_benchmark.py \
    --config mid --sharding B --devices 4 --profile

# Compare sharding configs
python scripts/qwen35_benchmark.py --config mid --sharding A --devices 4
python scripts/qwen35_benchmark.py --config mid --sharding B --devices 4
```

### Profile Analysis

Look for in TensorBoard (`tensorboard --logdir=/tmp/qwen35`):

**Prefill**: Single dense compute block, no idle gaps, high MXU utilization.

**Decode**: All steps fused, no host-device sync gaps. Compare lax.scan vs Python loop
to see the roundtrip elimination.

**Config A vs B**: Check ICI bandwidth — Config A avoids GQA allreduce (GQA is replicated),
Config B has allreduce on GQA output projection.

---

## 10. Memory Budget

### Full Model (397B) on 8x TPU v5p (8 × 95 GB)

| Component | Sharding | Per Device |
|-----------|----------|------------|
| MoE experts (EP=8) | 512/8 = 64 per device | 48.5 GB |
| DeltaNet attn (TP=8) | heads/8 | 0.66 GB |
| GQA attn (TP=2 or TP=8) | | 0.25–0.79 GB |
| Router + shared expert + norms | replicated | 0.88 GB |
| Embed + LM head (TP=8) | | 0.25 GB |
| Scales (fp8) | replicated | 0.5 GB |
| **Total weights** | | **~52 GB** |
| GQA KV cache (8K ctx) | | ~0.25 GB |
| DeltaNet state | | ~0.02 GB |
| Activations (prefill) | | ~0.08 GB |
| **Grand total** | | **~52 GB / 95 GB** |

### Mid Config (32B) on 4x TPU v5p (4 × 95 GB)

| Component | Sharding | Per Device |
|-----------|----------|------------|
| MoE experts (64, TP=4) | 16 per device | 0.8 GB |
| DeltaNet attn (TP=4) | | 1.3 GB |
| GQA attn (TP=4) | | 0.4 GB |
| Replicated | | 0.2 GB |
| Embed + LM head | | 0.5 GB |
| **Total (bf16)** | | **~16 GB** |
| **Headroom** | | **~80 GB** |

---

## File Inventory

```
jax_gpt/models/qwen35/
    config.py           — Qwen35Config dataclass (mini/mid/full)
    primitives.py       — rms_norm, precompute_rope_freqs, apply_rotary_emb, swiglu
    deltanet.py         — deltanet_recurrent_step, deltanet_prefill, _causal_conv1d_*, _gated_rms_norm
    gqa.py              — gqa_attention (with output gate, QK norm, KV cache)
    moe.py              — moe_routing, expert_forward, shared_expert_forward, moe_layer
    cache.py            — HybridCache dataclass + init_cache
    block.py            — deltanet_layer_forward, gqa_layer_forward, group_forward
    model.py            — init_params, forward, generate, _sample
    fp8.py              — dynamic_quantize_fp8, fp8_matmul, fp8_linear, fp8_ragged_dot
    sharding.py         — AXIS_RULES_A/B, make_mesh, shard_params, shard_cache, make_cache_sharding
    weight_loader.py    — load_from_hf_state_dict, _load_moe_params

tests/qwen35/          — 58 unit tests + 11 multi-device + 3 benchmarks
scripts/
    qwen35_benchmark.py — CLI benchmark runner (--config, --sharding, --profile, etc.)
```
