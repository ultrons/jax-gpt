import jax
from flax import nnx

from jax_gpt.models.gpt2.config import GPTConfig
from jax_gpt.models.gpt2.model import GPT
from jax_gpt.trainer.config import TrainConfig
from jax_gpt.trainer.optimizer import build_decay_mask, create_optimizer


def _make_model():
    config = GPTConfig(
        d_model=64, d_head=16, d_ff=256, n_head=4, n_kv_head=4, n_layers=2, d_context=32
    )
    rngs = nnx.Rngs(jax.random.PRNGKey(0))
    return GPT(config, rngs=rngs)


def _make_train_config():
    return TrainConfig(
        learning_rate=6e-4,
        min_lr=6e-5,
        warmup_steps=10,
        lr_decay_steps=1000,
        weight_decay=0.1,
        beta1=0.9,
        beta2=0.95,
        grad_clip=1.0,
    )


def test_optimizer_instantiation():
    model = _make_model()
    config = _make_train_config()
    tx = create_optimizer(model, config)
    params = nnx.state(model, nnx.Param)
    opt_state = tx.init(params)
    assert opt_state is not None


def test_weight_decay_mask_embeddings():
    model = _make_model()
    mask = build_decay_mask(model)
    # wte and wpe are nnx.Embed — their parameter is stored under 'embedding'
    # mask is a nested State; access leaves via jax.tree_util.tree_leaves_with_path
    mask_flat = {}
    def collect(path, val):
        mask_flat[str(path)] = val
        return val
    jax.tree_util.tree_map_with_path(collect, mask)
    embedding_entries = {k: v for k, v in mask_flat.items() if 'embedding' in k.lower()}
    assert len(embedding_entries) > 0, "No embedding params found in mask"
    for key, val in embedding_entries.items():
        assert val is False or val == False, f"Expected False for {key}, got {val}"


def test_weight_decay_mask_linear_kernels():
    model = _make_model()
    mask = build_decay_mask(model)
    mask_flat = {}
    def collect(path, val):
        mask_flat[str(path)] = val
        return val
    jax.tree_util.tree_map_with_path(collect, mask)
    kernel_entries = {
        k: v for k, v in mask_flat.items()
        if 'kernel' in k.lower() and 'embedding' not in k.lower()
    }
    assert len(kernel_entries) > 0, "No kernel params found in mask"
    for key, val in kernel_entries.items():
        assert val is True or val == True, f"Expected True for {key}, got {val}"


def test_weight_decay_mask_biases():
    model = _make_model()
    mask = build_decay_mask(model)
    mask_flat = {}
    def collect(path, val):
        mask_flat[str(path)] = val
        return val
    jax.tree_util.tree_map_with_path(collect, mask)
    bias_entries = {k: v for k, v in mask_flat.items() if 'bias' in k.lower()}
    assert len(bias_entries) > 0, "No bias params found in mask"
    for key, val in bias_entries.items():
        assert val is False or val == False, f"Expected False for {key}, got {val}"


def test_lr_schedule_warmup():
    warmup_steps = 100
    config = TrainConfig(
        learning_rate=6e-4,
        min_lr=6e-5,
        warmup_steps=warmup_steps,
        lr_decay_steps=10000,
        weight_decay=0.1,
        beta1=0.9,
        beta2=0.95,
        grad_clip=1.0,
    )
    import optax
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=config.learning_rate,
        warmup_steps=config.warmup_steps,
        decay_steps=config.lr_decay_steps,
        end_value=config.min_lr,
    )
    lr_at_0 = schedule(0)
    lr_at_warmup = schedule(warmup_steps)
    assert float(lr_at_0) < 1e-5, f"LR at step 0 should be near 0, got {lr_at_0}"
    assert abs(float(lr_at_warmup) - config.learning_rate) < 1e-5, (
        f"LR at warmup_steps should be near peak {config.learning_rate}, got {lr_at_warmup}"
    )
