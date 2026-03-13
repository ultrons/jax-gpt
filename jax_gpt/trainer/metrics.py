import time

import jax

from jax_gpt.models.gpt2.config import GPTConfig
from jax_gpt.trainer.config import TrainConfig


def compute_model_flops(config: GPTConfig) -> int:
    # Non-embedding parameter count (dominant terms)
    # Attention: Q,K,V projections + output = 4 * d_model^2 per layer
    # MLP: 2 layers = 8 * d_model^2 per layer (with d_ff = 4*d_model)
    # Total: 12 * n_layers * d_model^2
    N = 12 * config.n_layers * config.d_model ** 2
    return 2 * N  # FLOPs per token, forward pass only


HARDWARE_PEAK_TFLOPS = {
    'tpu_v3':  123e12,   # BF16
    'tpu_v4':  275e12,   # BF16
    'tpu_v5e': 393e12,   # BF16
    'a100':    312e12,   # BF16 (SXM)
    'h100':    989e12,   # BF16 (SXM)
    'cpu':       1e12,   # placeholder for M1 testing
}


def compute_mfu(
    flops_per_token: int,
    tokens_per_second: float,
    hardware: str,
    num_devices: int,
) -> float:
    """
    Model FLOP Utilization = achieved_flops / peak_flops

    Training FLOPs = 6 * flops_per_token (fwd + bwd + bwd for inputs)
    """
    peak = HARDWARE_PEAK_TFLOPS.get(hardware, HARDWARE_PEAK_TFLOPS['cpu'])
    achieved = 6 * flops_per_token * tokens_per_second
    return achieved / (peak * num_devices)


class MetricsLogger:
    def __init__(self, config: TrainConfig, model_config: GPTConfig):
        self._flops_per_token = compute_model_flops(model_config)
        self._config = config
        self._last_time = time.monotonic()
        self._last_step = 0
        self._wandb = None

        if config.log_to_wandb:
            import wandb
            self._wandb = wandb.init(
                project=config.wandb_project,
                name=config.wandb_run_name or None,
                resume='allow',
            )

    def log(self, step: int, metrics: dict, num_tokens_this_interval: int) -> None:
        """
        Compute derived metrics and emit to stdout + wandb.

        metrics dict from train_step should contain: loss, grad_norm.
        Also accepts optional: lr.
        """
        now = time.monotonic()
        dt = now - self._last_time

        tokens_per_sec = num_tokens_this_interval / dt if dt > 0 else 0.0
        num_devices = jax.device_count()
        tokens_per_sec_per_device = tokens_per_sec / num_devices

        mfu = compute_mfu(self._flops_per_token, tokens_per_sec,
                          self._config.hardware, num_devices)
        tflops_per_chip = (6 * self._flops_per_token * tokens_per_sec
                           / num_devices / 1e12)

        log_dict = {
            'step': step,
            'loss': float(metrics.get('loss', float('nan'))),
            'grad_norm': float(metrics.get('grad_norm', float('nan'))),
            'lr': float(metrics.get('lr', float('nan'))),
            'tokens_per_sec': tokens_per_sec,
            'tokens_per_sec_per_device': tokens_per_sec_per_device,
            'mfu_pct': mfu * 100,
            'tflops_per_chip': tflops_per_chip,
        }

        # Stdout — always emit on process 0
        if jax.process_index() == 0:
            print(
                f"step {step:6d} | "
                f"loss {log_dict['loss']:.4f} | "
                f"lr {log_dict['lr']:.2e} | "
                f"gnorm {log_dict['grad_norm']:.3f} | "
                f"tok/s {tokens_per_sec:,.0f} | "
                f"tok/s/chip {tokens_per_sec_per_device:,.0f} | "
                f"TFLOPs/chip {tflops_per_chip:.1f} | "
                f"MFU {log_dict['mfu_pct']:.1f}%"
            )

        if self._wandb is not None:
            self._wandb.log({k: v for k, v in log_dict.items() if k != 'step'}, step=step)

        self._last_time = now
        self._last_step = step

    def log_val(self, step: int, val_loss: float) -> None:
        if jax.process_index() == 0:
            print(f"step {step:6d} | val_loss {val_loss:.4f}")
        if self._wandb is not None:
            self._wandb.log({'val_loss': val_loss}, step=step)
