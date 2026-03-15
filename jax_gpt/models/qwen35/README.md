# Qwen3.5 — Pure JAX Implementation

A from-scratch implementation of the [Qwen3.5-397B-A17B](https://huggingface.co/Qwen/Qwen3.5-397B-A17B) architecture in pure JAX (no Flax/NNX). Built for studying and profiling inference performance on TPU.

## Architecture

Qwen3.5 is a hybrid-attention sparse MoE model:

```
For each group of 4 layers:
  Layer 0: RMSNorm → Gated DeltaNet (linear attention) → Residual → RMSNorm → MoE → Residual
  Layer 1: RMSNorm → Gated DeltaNet (linear attention) → Residual → RMSNorm → MoE → Residual
  Layer 2: RMSNorm → Gated DeltaNet (linear attention) → Residual → RMSNorm → MoE → Residual
  Layer 3: RMSNorm → GQA (full attention)              → Residual → RMSNorm → MoE → Residual
```

Key components:
- **Gated DeltaNet**: Linear attention with fixed-size recurrent state (O(1) memory per step)
- **GQA**: Grouped Query Attention with RoPE, QK norm, and sigmoid output gate
- **MoE**: Top-k routing with `jax.lax.ragged_dot` for sparse expert computation + shared expert
- **lax.scan decode loop**: Entire decode compiles to a single HLO program (zero host-device roundtrips)

## Model Configs

| Config | Layers | d_model | Experts | Params | Memory (bf16) | Target |
|--------|--------|---------|---------|--------|---------------|--------|
| `mini` | 8 | 1024 | 4 | ~600M | ~1.2 GB | Mac development |
| `mid` | 32 | 4096 | 64 | ~32B | ~64 GB | 4x TPU v5p |
| `full` | 60 | 4096 | 512 | ~397B | ~794 GB | 8x+ TPU v5p / Ironwood |

## Quick Start (Mac)

```bash
# Run all unit tests (55 tests)
python -m pytest tests/qwen35/ --ignore=tests/qwen35/test_sharding_multidevice.py -v

# Run multi-device sharding tests (11 tests, simulates 8 CPU devices)
XLA_FLAGS=--xla_force_host_platform_device_count=8 \
    python -m pytest tests/qwen35/test_sharding_multidevice.py -v

# Run inference benchmarks
python -m pytest tests/qwen35/test_inference_benchmark.py -v -s

# Run benchmark script
python scripts/qwen35_benchmark.py --config mini
```

## Running on TPU v5p

### Setup

```bash
# SSH into your TPU VM
gcloud compute tpus tpu-vm ssh <your-tpu-name> --zone=<zone>

# Clone and install
git clone <your-repo-url>
cd jax-gpt
pip install -r requirements.txt
pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

# Verify TPU access
python -c "import jax; print(f'{jax.device_count()} TPU devices: {jax.devices()}')"
```

### Run Benchmarks

```bash
# Mid config on 4x TPU v5p — Config B (uniform TP=4)
python scripts/qwen35_benchmark.py \
    --config mid \
    --sharding B \
    --devices 4 \
    --batch-size 1 \
    --prompt-len 512 \
    --decode-steps 128 \
    --dtype bfloat16

# Same with Config A (TP=4 DeltaNet, GQA replicated)
python scripts/qwen35_benchmark.py \
    --config mid \
    --sharding A \
    --devices 4 \
    --batch-size 1 \
    --prompt-len 512 \
    --decode-steps 128 \
    --dtype bfloat16

# With TPU profiling (view in TensorBoard)
PROFILE_DIR=/tmp/qwen35_profiles python scripts/qwen35_benchmark.py \
    --config mid \
    --sharding B \
    --devices 4 \
    --profile

# View profile
tensorboard --logdir=/tmp/qwen35_profiles
```

### What to Look For in TPU Profiles

**Prefill:**
- Single dense compute block (matmuls, norms, MoE routing)
- No idle gaps between ops
- MXU utilization should be high

**Decode (lax.scan):**
- All decode steps fused into one HLO program
- No host-device sync gaps between steps
- Compare with Python-loop decode to see the roundtrip elimination

**Config A vs B:**
- Config A: GQA layers have no allreduce (replicated) — less ICI traffic
- Config B: GQA layers have allreduce across TP — more uniform but more communication
- Check ICI bandwidth utilization in the profile

## Experimenting with Model Configs

### Creating a Custom Config

Edit `jax_gpt/models/qwen35/config.py`:

```python
@classmethod
def my_config(cls) -> Qwen35Config:
    return cls(
        d_model=2048,
        n_layers=16,          # must be divisible by full_attention_interval (4)
        max_position_embeddings=4096,

        # GQA — q_heads must be divisible by kv_heads
        gqa_n_q_heads=16,
        gqa_n_kv_heads=2,
        gqa_head_dim=128,

        # DeltaNet — v_heads must be divisible by qk_heads
        delta_n_qk_heads=8,
        delta_n_v_heads=32,
        delta_qk_head_dim=128,
        delta_v_head_dim=128,

        # MoE — n_routed_experts should be divisible by TP
        n_routed_experts=32,
        n_experts_per_token=4,
        moe_intermediate_size=512,
        shared_expert_intermediate_size=512,
    )
```

Then add it to the benchmark script's config dict in `get_config()`:

```python
configs = {
    'mini': Qwen35Config.mini,
    'mid': Qwen35Config.mid,
    'full': Qwen35Config.full,
    'my_config': Qwen35Config.my_config,  # add here
}
```

### Divisibility Constraints

For clean sharding, ensure:
- `n_routed_experts % TP == 0` (experts split evenly across devices)
- `delta_n_v_heads % TP == 0` (DeltaNet heads split evenly)
- `gqa_n_q_heads % TP == 0` (for Config B; Config A replicates GQA)
- `delta_n_v_heads % delta_n_qk_heads == 0` (QK→V head grouping)
- `n_layers % 4 == 0` (groups of 4 layers)

Dimensions that don't divide evenly automatically fall back to replicated.

## Experimenting with Sharding Configs

### Modifying Axis Rules

Edit `jax_gpt/models/qwen35/sharding.py`:

```python
# Example: 2D mesh with separate EP and TP axes
MY_AXIS_RULES = {
    'vocab':           'tp',
    'embed':           None,
    'delta_v_heads':   'tp',
    'delta_qk_heads':  'tp',
    'gqa_q_heads':     'tp',
    'gqa_kv_heads':    None,
    'experts':         'ep',       # separate expert parallelism axis
}
```

For a 2D mesh, update `make_mesh()`:

```python
def make_mesh_2d(ep: int, tp: int) -> Mesh:
    devices = np.array(jax.devices()).reshape(ep, tp)
    return Mesh(devices, ('ep', 'tp'))
```

Add to the benchmark script:

```python
# In get_axis_rules():
rules = {
    'none': None,
    'A': AXIS_RULES_A,
    'B': AXIS_RULES_B,
    'my_rules': MY_AXIS_RULES,
}
```

### Adding Activation Sharding Constraints

For better XLA sharding propagation, add `jax.lax.with_sharding_constraint`
inside the forward pass. Key places:

```python
# In gqa.py, after Q/K/V projection:
q = jax.lax.with_sharding_constraint(q, P(None, 'gqa_q_heads', None, None))

# In moe.py, after expert routing:
x_sorted = jax.lax.with_sharding_constraint(x_sorted, P(None, None))

# In deltanet.py, after QKV projection:
mixed_qkv = jax.lax.with_sharding_constraint(mixed_qkv, P(None, None, 'delta_v_heads'))
```

## File Structure

```
jax_gpt/models/qwen35/
    config.py        — Qwen35Config (mini, mid, full)
    primitives.py    — RMSNorm, RoPE, SwiGLU
    deltanet.py      — Gated DeltaNet (recurrent + prefill)
    gqa.py           — GQA with output gate and QK norm
    moe.py           — Top-k routing + ragged_dot experts
    cache.py         — HybridCache (DeltaNet state + GQA KV cache)
    block.py         — 4-layer group with lax.scan
    model.py         — Full model with init_params, forward, generate
    fp8.py           — FP8 matmul utilities for native TPU fp8 ops
    sharding.py      — Logical axis rules + sharding configs
    weight_loader.py — HF state_dict → JAX pytree mapping

tests/qwen35/       — 58 unit tests + 11 multi-device sharding tests
scripts/
    qwen35_benchmark.py — CLI benchmark runner
```

## Design Decisions

**Pure JAX (no Flax NNX):** Every layer is a pure function `f(params, x, state) -> (y, new_state)`. This enables `lax.scan` for both layer iteration and the decode loop — impossible with NNX due to trace-level conflicts.

**Params as nested dicts:** Stacked with leading scan axes via `jnp.stack`. `lax.scan` slices along these axes automatically.

**Native FP8:** Weights stored as `float8_e4m3fn` with per-row `weight_scale_inv`. On TPU, `jax.lax.dot_general` emits native fp8 MXU ops at 2x bf16 FLOPS.

**Logical axis sharding:** Every param annotated with logical axis names. Physical mesh mapping is a separate dict — swap the dict to change parallelism without touching model code.
