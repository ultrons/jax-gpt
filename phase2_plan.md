# Phase 2 Implementation Plan: JAX/Flax NNX GPT-2 Training Infrastructure

> Best-practices reference implementation for large-scale distributed training on TPU/GPU with GCS data.

---

## Key Technical Decisions

### Data Loading: `grain` (not tf.data or gcsfs)
- Built for JAX + GCS: deterministic step resumption, multi-host sharding via `ShardByJaxProcess`, streaming without materializing full datasets
- `grain.ShardOptions(shard_index=jax.process_index(), shard_count=jax.process_count())` handles multi-host automatically
- Falls back to local `.bin`/`.npy` files for M1 development — identical code path

### Device Mesh: 4-axis `('dp', 'fsdp', 'tp', 'sp')`
```python
# e.g. (2, 1, 4, 1) = DP+TP on 8 devices
devices = np.array(jax.devices()).reshape(dp, fsdp, tp, sp)
mesh = Mesh(devices, ('dp', 'fsdp', 'tp', 'sp'))
```
- `dp`: data parallelism (replicated weights, sharded batch)
- `fsdp`: ZeRO-3 style sharding of params + optimizer state + grads
- `tp`: tensor/model parallelism (column/row parallel weight matrices)
- `sp`: sequence parallelism (shard sequence dimension in attention)
- **Pipeline parallelism hook**: `make_mesh()` accepts `pp=1` now; Phase 3 adds `'pp'` as outermost axis

### Sharding Annotation Strategy: Hybrid
- `NamedSharding` on params pytree at `TrainState` init (explicit, XLA never guesses)
- `jax.lax.with_sharding_constraint` on activations inside `__call__` for SP/TP intermediates
- Both are gated: `mesh=None` → no-op → Phase 1 tests pass unchanged

### TrainState: NNX Functional Split Pattern
```python
# Extract params pytree for jax.jit differentiation
params = nnx.state(model, nnx.Param)
loss, grads = jax.value_and_grad(loss_fn)(params)
# After update:
nnx.update(model, new_params)
```

### Mixed Precision: BF16 compute + FP32 master weights
- Cast activations to BF16 at `train_step` entry, not inside model (keeps model dtype-agnostic)
- Loss computed in FP32 (upcast before cross-entropy)
- LayerNorm in FP32 (Flax NNX default)
- Controlled by `TrainConfig.dtype = 'bfloat16'`

### Gradient Accumulation: Python-level loop (not inside jit)
- `grad_accum_steps` micro-batches accumulated in Python loop to avoid recompilation
- Effective batch = `micro_batch_size * seq_len * grad_accum_steps * num_devices * num_hosts`

---

## File Structure

```
jax_gpt/
  trainer/
    config.py           MODIFY  — expand TrainConfig significantly
    train_state.py      CREATE  — TrainState dataclass, create_train_state
    optimizer.py        CREATE  — AdamW with weight decay mask, cosine schedule
    train_step.py       CREATE  — train_step (jit+donate), eval_step, accum wrapper
    sharding.py         CREATE  — make_mesh, param sharding specs, shard_train_state
    metrics.py          CREATE  — MFU, TFLOPs/chip, tok/s, WandB + stdout logger
    checkpointing.py    CREATE  — orbax async checkpoint manager (GCS-compatible)

jax_gpt/
  data/
    __init__.py         CREATE
    pipeline.py         CREATE  — grain-based GCS/local dataset, multi-host sharding
    tokenize_to_gcs.py  CREATE  — offline tokenization → ArrayRecord shards on GCS

scripts/
  train.py              MODIFY  — main training loop (currently empty stub)
  utils.py              MODIFY  — device setup helpers, CLI config parsing

tests/
  test_sharding.py      CREATE  — golden output: sharded == unsharded
  test_train_step.py    CREATE  — loss decreases, gradients non-zero, initial loss ≈ ln(vocab)
  test_data_pipeline.py CREATE  — batch shapes, y == x shifted by 1, multi-host sim
  test_checkpointing.py CREATE  — save/restore round-trip
  test_metrics.py       CREATE  — MFU formula, TFLOPs formula
```

---

## Task Breakdown

### Group A: Configuration & State (no dependencies — start immediately)

#### A1: Expand `TrainConfig`
**File:** `jax_gpt/trainer/config.py`

```python
@dataclass
class TrainConfig:
    # Batch & sequence
    global_batch_tokens: int = 524288   # ~0.5M tokens
    micro_batch_size: int = 16
    seq_len: int = 1024
    grad_accum_steps: int = 1

    # Optimizer
    learning_rate: float = 6e-4
    min_lr: float = 6e-5
    warmup_steps: int = 2000
    lr_decay_steps: int = 600000
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0

    # Mixed precision
    dtype: str = 'bfloat16'        # compute dtype
    param_dtype: str = 'float32'   # master weights dtype

    # Sharding (default = single device safe)
    dp: int = 1
    fsdp: int = 1
    tp: int = 1
    sp: int = 1
    pp: int = 1                    # Phase 3 hook — not wired yet

    # Data
    data_source: str = 'local'     # 'local' | 'gcs'
    gcs_bucket: str = ''
    gcs_dataset_path: str = ''
    local_data_path: str = 'data/'
    num_workers: int = 4

    # Training duration
    max_steps: int = 600000
    eval_interval: int = 250
    eval_steps: int = 20
    log_interval: int = 10
    save_interval: int = 1000

    # Checkpointing
    checkpoint_dir: str = 'checkpoints/'
    max_checkpoints_to_keep: int = 3
    resume_from: str = ''

    # Logging
    wandb_project: str = ''
    wandb_run_name: str = ''
    log_to_wandb: bool = False
    hardware: str = 'tpu_v4'       # for MFU calculation peak FLOPS lookup
```

**Test:** `python -c "from jax_gpt.trainer.config import TrainConfig; TrainConfig()"` — no errors.

---

#### A2: Create `TrainState`
**File:** `jax_gpt/trainer/train_state.py`

```python
@dataclass
class TrainState:
    step: int
    model: GPT                         # mutable NNX module
    tx: optax.GradientTransformation
    opt_state: optax.OptState          # pure pytree, JIT-able

def create_train_state(model, tx) -> TrainState:
    params = nnx.state(model, nnx.Param)
    opt_state = tx.init(params)
    return TrainState(step=0, model=model, tx=tx, opt_state=opt_state)
```

**Test:** Instantiation works, `opt_state` not None.

---

### Group B: Optimizer (depends on A1, A2)

#### B1: Create `optimizer.py`
**File:** `jax_gpt/trainer/optimizer.py`

AdamW with weight decay separation:
- **no decay**: biases, LayerNorm params, embeddings
- **with decay**: all Linear kernels

```python
def make_weight_decay_mask(model) -> pytree_of_bool:
    """Walk NNX graph, return bool pytree matching nnx.state(model, nnx.Param)."""
    ...

def create_optimizer(model, config: TrainConfig) -> optax.GradientTransformation:
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=config.learning_rate,
        warmup_steps=config.warmup_steps,
        decay_steps=config.lr_decay_steps,
        end_value=config.min_lr,
    )
    return optax.chain(
        optax.clip_by_global_norm(config.grad_clip),
        optax.scale_by_adam(b1=config.beta1, b2=config.beta2, eps=1e-8),
        optax.add_decayed_weights(config.weight_decay, mask=make_weight_decay_mask(model)),
        optax.scale_by_learning_rate(schedule),
    )
```

**Test:** Check that embedding weights get `mask=False`, Linear kernels get `mask=True`.

---

### Group C: Data Pipeline (depends on A1 only — parallelize with B, D, G)

#### C1: Create `data/pipeline.py`
**File:** `jax_gpt/data/pipeline.py`

```python
class DataPipeline:
    def __init__(self, config: TrainConfig, split: str = 'train'):
        source = _make_source(config, split)  # local memmap or GCS ArrayRecord
        shard_options = grain.ShardOptions(
            shard_index=jax.process_index(),
            shard_count=jax.process_count(),
        )
        self._loader = grain.DataLoader(
            data_source=source,
            sampler=grain.SequentialSampler(len(source), shard_options),
            operations=[_SlidingWindowOp(seq_len=config.seq_len)],
            worker_count=config.num_workers,
            read_options=grain.ReadOptions(prefetch_buffer_size=512),
        )

    def get_batch(self) -> tuple[np.ndarray, np.ndarray]:
        """Returns (x, y) each shape (micro_batch_size, seq_len)."""
        ...

    def skip_to_step(self, step: int, grad_accum_steps: int):
        """Deterministic fast-forward for checkpoint resumption."""
        ...
```

**Test:**
```python
x, y = pipe.get_batch()
assert x.shape == (micro_batch_size, seq_len)
assert np.array_equal(x[:, 1:], y[:, :-1])  # y is x shifted by 1
```

Multi-host simulation on M1: `XLA_FLAGS=--xla_force_host_platform_device_count=8` + two `DataPipeline` instances must return non-overlapping ranges.

---

#### C2: Create `data/tokenize_to_gcs.py`
**File:** `jax_gpt/data/tokenize_to_gcs.py`

Offline script: JSONL → tiktoken → ArrayRecord shards on GCS (or flat `.bin` locally).
- Concatenates documents with EOT token `50256`
- Splits into train/val at configurable ratio
- Writes to `gs://bucket/dataset/train/` and `.../val/`

**Test:** Run on 1MB sample, verify `.bin` is loadable by `DataPipeline`.

---

### Group D: Sharding Infrastructure (depends on A1, A2 — parallelize with B, C, G)

#### D1: Create `trainer/sharding.py`
**File:** `jax_gpt/trainer/sharding.py`

```python
def make_mesh(config: TrainConfig) -> Mesh:
    """
    4D mesh: ('dp', 'fsdp', 'tp', 'sp').
    Phase 3 hook: if config.pp > 1, creates 5D mesh ('pp', 'dp', 'fsdp', 'tp', 'sp').
    """
    total = config.dp * config.fsdp * config.tp * config.sp
    assert total == jax.device_count()
    devices = np.array(jax.devices()).reshape(config.dp, config.fsdp, config.tp, config.sp)
    return Mesh(devices, ('dp', 'fsdp', 'tp', 'sp'))


def get_param_sharding_specs(model_config: GPTConfig) -> dict[str, PartitionSpec]:
    """
    Per-parameter PartitionSpec for TP sharding.
    All specs include a leading None for the layers axis (nnx.vmap).

    Attention:
      c_attn.kernel: [None, None, 'tp']   (column-parallel, output sharded)
      c_attn.bias:   [None, 'tp']
      c_proj.kernel: [None, 'tp', None]   (row-parallel, input sharded)
      c_proj.bias:   [None, None]         (replicated — added after allreduce)

    MLP:
      c_fc.kernel:   [None, None, 'tp']   (column-parallel)
      c_proj.kernel: [None, 'tp', None]   (row-parallel)

    Embeddings:
      wte: ['tp', None]                   (vocab-parallel)
      wpe: [None, None]                   (replicated — small)
    """
    ...


def shard_train_state(state: TrainState, mesh: Mesh, model_config: GPTConfig) -> TrainState:
    """Apply NamedSharding to all params and opt_state. Returns new TrainState."""
    ...


def activation_sharding_constraint(x, spec: PartitionSpec, mesh: Mesh | None):
    """No-op if mesh is None. Used inside model forward pass."""
    if mesh is None:
        return x
    return jax.lax.with_sharding_constraint(x, NamedSharding(mesh, spec))
```

**Test (Tier 1 — M1 single device):**
- `make_mesh(TrainConfig(dp=1, fsdp=1, tp=1, sp=1))` doesn't crash

**Test (Tier 2 — M1 with fake devices):**
- `XLA_FLAGS=--xla_force_host_platform_device_count=8` + `make_mesh(TrainConfig(tp=8))` creates valid mesh

---

#### D2: Add activation sharding to `model.py`
**File:** `jax_gpt/models/gpt2/model.py`

Add optional `mesh=None` parameter to `GPT.__call__` and `CausalSelfAttention.__call__`.

In `CausalSelfAttention`, after QKV reshape:
```python
# SP: shard sequence dimension
q = activation_sharding_constraint(q, P('dp', 'tp', 'sp', None), mesh)
k = activation_sharding_constraint(k, P('dp', 'tp', 'sp', None), mesh)
v = activation_sharding_constraint(v, P('dp', 'tp', 'sp', None), mesh)
```

In `MLP`, after intermediate activation:
```python
# TP: shard d_ff dimension (allreduce happens at c_proj)
x = activation_sharding_constraint(x, P('dp', None, 'tp'), mesh)
```

**Critical invariant:** All existing Phase 1 tests must pass without modification — `mesh=None` is the default.

---

### Group E: Correctness / Golden Output Tests (depends on D1, D2)

#### E1: Create `tests/test_sharding.py`

**M1 trick:** `XLA_FLAGS=--xla_force_host_platform_device_count=8` gives 8 fake CPU devices for full sharding test coverage before TPU access.

**Tier 1: trivial mesh (always runs on M1)**
```python
def test_unsharded_equals_sharded_trivial_mesh():
    """(dp=1, fsdp=1, tp=1, sp=1) sharded output must be bitwise identical to unsharded."""
    ...
    np.testing.assert_allclose(logits_ref, logits_sharded, rtol=1e-5)
```

**Tier 2: multi-device CPU (M1 with fake devices)**
```python
@pytest.mark.parametrize("tp", [2, 4, 8])
def test_tp_matches_reference(tp):
    """TP-sharded forward pass must match FP32 reference within tolerance."""
    # rtol=1e-4 for BF16, rtol=1e-6 for FP32
    ...
```

**Tier 3: full training correctness (TPU only, marked `@pytest.mark.tpu`)**
- Train 10 steps unsharded and sharded with same seed + same data
- Loss curves within tolerance

---

### Group F: Training Step (depends on A2, B1)

#### F1: Create `trainer/train_step.py`

```python
@functools.partial(jax.jit, donate_argnums=(0,))
def train_step(state: TrainState, batch, mesh=None) -> tuple[TrainState, dict]:
    """
    donate_argnums=(0,) donates state buffer — critical for TPU memory efficiency.
    """
    x, y = batch
    model = state.model

    def loss_fn(params):
        nnx.update(model, params)
        logits, _ = model(x, deterministic=False, mesh=mesh)
        logits_fp32 = logits.astype(jnp.float32)
        return optax.softmax_cross_entropy_with_integer_labels(
            logits_fp32.reshape(-1, logits_fp32.shape[-1]), y.reshape(-1)
        ).mean()

    params = nnx.state(model, nnx.Param)
    loss, grads = jax.value_and_grad(loss_fn)(params)
    grad_norm = optax.global_norm(grads)

    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)
    nnx.update(model, new_params)

    return (
        TrainState(step=state.step + 1, model=model, tx=state.tx, opt_state=new_opt_state),
        {'loss': loss, 'grad_norm': grad_norm},
    )


def train_step_with_accumulation(state, micro_batches, mesh=None):
    """Python-level gradient accumulation. Avoids recompilation."""
    ...


@jax.jit
def eval_step(model, batch) -> jax.Array:
    x, y = batch
    logits, _ = model(x, deterministic=True)
    return optax.softmax_cross_entropy_with_integer_labels(...).mean()
```

**Tests:**
```python
def test_loss_decreases():
    """50 train_step calls on tiny model, same batch → loss decreases."""

def test_initial_loss_near_log_vocab():
    """Initial loss ≈ ln(vocab_size) = ln(50257) ≈ 10.82"""

def test_gradients_nonzero_and_finite():
    """No NaN, no zero gradients after first step."""
```

---

### Group G: Metrics & Logging (depends on A1 only — parallelize with everything)

#### G1: Create `trainer/metrics.py`

```python
# TFLOPs estimate (Chinchilla / nanoGPT formula)
# Per token, forward only: 2 * N_non_embedding_params
# Per token, training (fwd+bwd): 6 * N_non_embedding_params
def compute_model_flops(config: GPTConfig) -> int:
    N = 12 * config.n_layers * config.d_model**2  # dominant attn+MLP terms
    return 2 * N  # forward only, per token

# MFU = (achieved FLOPS) / (peak hardware FLOPS)
# achieved = flops_per_token * tokens_per_sec * 3  (training)
HARDWARE_PEAK_TFLOPS = {
    'tpu_v3': 123e12,
    'tpu_v4': 275e12,
    'a100': 312e12,
    'h100': 989e12,
    'cpu': 1e12,   # placeholder for M1 testing
}

class MetricsLogger:
    def log(self, step, metrics, num_tokens_this_step):
        # Computes: tokens/s, tokens/s/chip, TFLOPs/chip, MFU%
        # Prints formatted stdout line
        # Calls wandb.log() if enabled
        ...
```

**Stdout format:**
```
step   1000 | loss 4.2341 | lr 6.00e-04 | gnorm 1.234 | tok/s 145230 | tok/s/chip 18154 | TFLOPs/chip 42.3 | MFU 15.4%
```

**Tests:**
```python
def test_mfu_formula():
    """GPT-2-small at 10k tok/s on 1 TPU v4 → MFU in range [1%, 10%]."""

def test_tflops_per_chip():
    """Verify formula: 6 * N * tokens_per_sec / num_chips / 1e12"""
```

---

### Group H: Checkpointing (depends on A2)

#### H1: Create `trainer/checkpointing.py`

```python
class CheckpointManager:
    """Orbax async checkpointing with GCS support."""

    def save(self, step, state: TrainState):
        """Non-blocking. Call wait_until_finished() before exit."""
        params = nnx.state(state.model, nnx.Param)
        # Save: {'step', 'params', 'opt_state', 'model_config', 'train_config'}
        ...

    def restore(self, step, state: TrainState) -> TrainState:
        """Restore params via nnx.update(state.model, restored_params)."""
        ...

    def wait_until_finished(self): ...
    def latest_step(self) -> int | None: ...
```

Key: model structure is **never saved** — reconstructed from `GPTConfig` at startup, then populated. This keeps checkpoints portable and Phase 3 PP-compatible (each PP stage can checkpoint its own layer slice).

GCS path: `gs://bucket/run_name/checkpoints/` — works transparently via orbax's GCS backend.

**Test:**
```python
def test_checkpoint_roundtrip(tmp_path):
    """Save at step 42, restore into fresh model, params bitwise identical, step==42."""
```

---

### Group I: Main Training Loop (depends on all above)

#### I1: Rewrite `scripts/train.py`

```python
def main(train_config, model_config):
    # 1. mesh
    mesh = make_mesh(train_config)
    # 2. model
    with mesh:
        model = GPT(model_config, rngs=nnx.Rngs(0))
    # 3. optimizer + state
    tx = create_optimizer(model, train_config)
    state = create_train_state(model, tx)
    # 4. shard
    state = shard_train_state(state, mesh, model_config)
    # 5. resume
    ckpt_mgr = CheckpointManager(train_config, model_config)
    if ckpt_mgr.latest_step():
        state = ckpt_mgr.restore(None, state)
    # 6. data
    train_pipe = DataPipeline(train_config, 'train')
    val_pipe = DataPipeline(train_config, 'val')
    train_pipe.skip_to_step(state.step, train_config.grad_accum_steps)
    # 7. logger
    logger = MetricsLogger(train_config, model_config)
    # 8. loop
    for step in range(state.step, train_config.max_steps):
        micro_batches = [train_pipe.get_batch() for _ in range(train_config.grad_accum_steps)]
        t0 = time.time()
        state, metrics = train_step_with_accumulation(state, micro_batches, mesh)
        jax.effects_barrier()
        ...
```

**Smoke test on M1:**
```bash
python scripts/train.py --max_steps=5 --micro_batch_size=2 --seq_len=64 \
  --data_source=local --local_data_path=data/ --log_interval=1
```
Expected: 5 steps, loss decreases, metrics printed.

---

## Dependency Graph

```
A1 (TrainConfig)
├── A2 (TrainState)
│   ├── B1 (Optimizer)
│   │   └── F1 (train_step)
│   ├── H1 (Checkpointing)
│   └── D1 (Sharding)
│       └── D2 (model.py mods)
│           └── E1 (Golden tests)
├── C1 (Data Pipeline)
├── C2 (Tokenize to GCS)
└── G1 (Metrics)

F1 + D1 + D2 + E1 + C1 + H1 + G1
└── I1 (train.py main loop)
    └── I2 (utils.py helpers)
```

## Agent Parallelization

All tasks after A1+A2 are complete can be parallelized across 5 agents:

| Agent | Tasks | Start condition |
|---|---|---|
| Agent 1 | A1, A2 | Immediately |
| Agent 2 | B1 (Optimizer) | After A1, A2 |
| Agent 3 | C1, C2 (Data) | After A1 |
| Agent 4 | D1, D2 (Sharding) | After A1, A2 |
| Agent 5 | G1 (Metrics) | After A1 |

Synchronization points:
- **First sync**: Wait for A2, B1 → then start F1 (train_step)
- **Second sync**: Wait for D1 → then start D2
- **Third sync**: Wait for D2 → then start E1 (golden tests)
- **Final sync**: All groups done → start I1 (main loop)

H1 (Checkpointing) can start after A2 and run in parallel with B1, C1, D1, G1.

---

## Pipeline Parallelism Hooks (Phase 3)

Decisions made in Phase 2 that keep Phase 3 unblocked:

1. **`make_mesh(config.pp)`** — accepts `pp=1` now; Phase 3 adds `'pp'` axis as outermost dimension. No other changes needed.

2. **`nnx.vmap`+`nnx.scan` layer stack** — all layer weights have a leading `n_layers` axis. Phase 3 partitions this axis over `pp` by adding `PartitionSpec('pp', ...)` to the layers dimension in `get_param_sharding_specs`.

3. **`train_step` is layer-count agnostic** — `nnx.scan` makes the entire stack one operation. Phase 3 breaks it into `pp` segments (one per stage) by modifying the scan, not the optimizer or state.

4. **Checkpointing is mesh-independent** — uses `nnx.state(model, nnx.Param)` pytree. Each PP stage can save its layer slice independently.

5. **NOTE IN PLAN**: Pipeline parallelism (`pp > 1`) requires:
   - Splitting `nnx.scan` into `pp` pipeline stages
   - Implementing micro-batch pipeline schedule (1F1B or GPipe)
   - Handling pipeline flush at sequence boundaries
   - Estimated complexity: 10-15 hours

---

## New Dependencies

Add to `requirements.txt`:
```
grain-nightly          # data pipeline (use grain>=0.2 when stable)
gcsfs                  # GCS filesystem for grain
orbax-checkpoint       # async checkpointing
optax                  # optimizer
wandb                  # logging
pytest                 # testing
```

Note: Use `jax[cpu]` for M1 dev, `jax[tpu]` for production. Keep separate.

---

## Testing on M1 vs TPU

| What | M1 CPU | TPU |
|---|---|---|
| Config, TrainState, Optimizer | ✅ Full | ✅ Full |
| Data pipeline (local .bin) | ✅ Full | ✅ Full (GCS) |
| train_step correctness (tiny model) | ✅ Full | ✅ Full |
| Loss decreases | ✅ Full | ✅ Full |
| Checkpointing round-trip | ✅ Full (local) | ✅ Full (GCS) |
| Trivial mesh sharding (dp=fsdp=tp=sp=1) | ✅ Full | ✅ Full |
| Multi-device sharding (fake devices) | ✅ `XLA_FLAGS=--xla_force_host_platform_device_count=8` | ✅ Full |
| BF16 training | ⚠️ CPU BF16 is slow but correct | ✅ Native |
| Performance / MFU numbers | ❌ Not representative | ✅ Full |
| Multi-host data pipeline | ❌ Single process only | ✅ Full |
| Actual convergence at scale | ❌ Too slow | ✅ Full |

**Bottom line**: Everything in Phase 2 can be verified for correctness on M1 CPU. Move to TPU only when validating scale, performance, or multi-host behavior.
