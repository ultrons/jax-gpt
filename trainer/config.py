# Step 2.1: Training Configuration
# TODO: Implement TrainConfig dataclass for training hyperparameters.
from dataclasses import dataclass

@dataclass
class TrainConfig:
    global_batch: int = 4096 # tokens
    learning_rate: float = 1e-4

