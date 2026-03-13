from dataclasses import dataclass


@dataclass
class TrainConfig:
    # --- Batch & sequence ---
    global_batch_tokens: int = 524288   # target tokens per gradient update (~0.5M)
    micro_batch_size: int = 16          # per-device batch size
    seq_len: int = 1024
    grad_accum_steps: int = 1           # set such that micro_batch*seq_len*accum*devices ≈ global_batch_tokens

    # --- Optimizer ---
    learning_rate: float = 6e-4
    min_lr: float = 6e-5                # cosine decay floor (≈ 0.1 * learning_rate)
    warmup_steps: int = 2000
    lr_decay_steps: int = 600000
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0

    # --- Mixed precision ---
    dtype: str = 'bfloat16'             # compute dtype: 'float32' | 'bfloat16'
    param_dtype: str = 'float32'        # master weight dtype (always float32 for stability)

    # --- Parallelism (all default to 1 = single-device safe) ---
    dp: int = 1                         # data parallelism
    fsdp: int = 1                       # fully-sharded data parallelism (ZeRO-3)
    tp: int = 1                         # tensor / model parallelism
    sp: int = 1                         # sequence parallelism
    pp: int = 1                         # pipeline parallelism — Phase 3 hook, not wired yet

    # --- Data ---
    data_source: str = 'local'          # 'local' | 'gcs'
    gcs_bucket: str = ''
    gcs_dataset_path: str = ''
    local_data_path: str = 'data/'
    num_workers: int = 4

    # --- Training duration ---
    max_steps: int = 600000
    eval_interval: int = 250
    eval_steps: int = 20
    log_interval: int = 10
    save_interval: int = 1000

    # --- Checkpointing ---
    checkpoint_dir: str = 'checkpoints/'
    max_checkpoints_to_keep: int = 3
    resume_from: str = ''               # empty = start fresh; 'latest' = auto-resume

    # --- Logging ---
    wandb_project: str = ''
    wandb_run_name: str = ''
    log_to_wandb: bool = False
    # Used to look up peak device FLOPS for MFU calculation.
    # Options: 'tpu_v3', 'tpu_v4', 'tpu_v5e', 'a100', 'h100', 'cpu'
    hardware: str = 'tpu_v4'
