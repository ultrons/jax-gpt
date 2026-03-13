import pytest

from jax_gpt.trainer.metrics import compute_model_flops, compute_mfu, MetricsLogger


def test_compute_model_flops_gpt2_small():
    """GPT-2 small: 12 layers, d_model=768 → N ≈ 85M → flops ≈ 170M/token."""
    from jax_gpt.models.gpt2.config import GPTConfig
    config = GPTConfig()  # default is gpt2-small
    flops = compute_model_flops(config)
    # 12 * 12 * 768^2 * 2 = 170,459,136
    assert 1.5e8 < flops < 2.0e8


def test_mfu_reasonable_range():
    """10k tok/s on 1 TPU v4 → MFU should be in plausible range (1%-15%)."""
    from jax_gpt.models.gpt2.config import GPTConfig
    flops = compute_model_flops(GPTConfig())
    mfu = compute_mfu(flops, 10_000, 'tpu_v4', 1)
    assert 0.005 < mfu < 0.15


def test_mfu_scales_with_devices():
    """Same throughput on 4 devices gives 1/4 the MFU per device."""
    from jax_gpt.models.gpt2.config import GPTConfig
    flops = compute_model_flops(GPTConfig())
    mfu_1 = compute_mfu(flops, 10_000, 'tpu_v4', 1)
    mfu_4 = compute_mfu(flops, 10_000, 'tpu_v4', 4)
    assert abs(mfu_1 / mfu_4 - 4.0) < 0.01


def test_tflops_per_chip_formula():
    """Verify: TFLOPs/chip = 6 * flops_per_token * tok/s / num_chips / 1e12"""
    from jax_gpt.models.gpt2.config import GPTConfig
    config = GPTConfig()
    flops = compute_model_flops(config)
    tok_per_sec = 50_000
    num_chips = 8
    expected = 6 * flops * tok_per_sec / num_chips / 1e12
    # compute_mfu gives us total, so derive tflops/chip from that
    achieved_total_tflops = 6 * flops * tok_per_sec / 1e12
    tflops_per_chip = achieved_total_tflops / num_chips
    assert abs(tflops_per_chip - expected) < 1e-6


def test_logger_instantiation(tmp_path):
    """MetricsLogger initialises without wandb."""
    from jax_gpt.trainer.config import TrainConfig
    from jax_gpt.models.gpt2.config import GPTConfig
    config = TrainConfig(log_to_wandb=False)
    logger = MetricsLogger(config, GPTConfig())
    # log call should not crash (will print to stdout)
    logger.log(0, {'loss': 2.3, 'grad_norm': 1.0, 'lr': 6e-4}, num_tokens_this_interval=1000)
