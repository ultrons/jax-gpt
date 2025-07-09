# Step 1.1: Model Configuration
# TODO: Implement GPTConfig dataclass for model hyperparameters.
from dataclasses import dataclass

@dataclass
class GPTConfig:
    d_model: int = 1024
    d_head: int = 128
    d_ff: int = 4 * 1024
    d_context: int = 256
    n_head: int = 8
    n_kv_head: int = 8
    n_layers: int = 4
    use_bias: bool = False
    dropout: int = 0.2
