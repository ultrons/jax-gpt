# Step 1.1: Model Configuration
# TODO: Implement GPTConfig dataclass for model hyperparameters.
from dataclasses import dataclass

@dataclass
class GPTConfig:
    d_model: int = 768
    d_head: int = 64
    d_ff: int = 3072 # 4 * 768
    d_context: int = 1024
    n_head: int = 12
    n_kv_head: int = 12
    n_layers: int = 12
    vocab_size: int = 50257 # Correct GPT-2 vocab size
    use_bias: bool = True
    dropout: float = 0.0 # Should be a float

    @classmethod
    def from_model_type(cls, model_type: str, **override_args):
        configs = {
            'gpt2':         dict(n_layers=12, n_head=12, d_model=768),
            'gpt2-medium':  dict(n_layers=24, n_head=16, d_model=1024),
            'gpt2-large':   dict(n_layers=36, n_head=20, d_model=1280),
            'gpt2-xl':      dict(n_layers=48, n_head=25, d_model=1600),
        }
        if model_type not in configs:
            raise ValueError(f"Unknown model type: {model_type}")
            
        config_dict = configs[model_type]
        # update d_ff and d_head based on d_model
        config_dict['d_ff'] = 4 * config_dict['d_model']
        config_dict['d_head'] = config_dict['d_model'] // config_dict['n_head']
        
        config_dict.update(override_args)
        return cls(**config_dict)
