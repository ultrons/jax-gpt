"""Qwen3.5 model configuration.

Mirrors the text config from Qwen/Qwen3.5-397B-A17B on HuggingFace.
Provides mini() for Mac development and full() for TPU inference.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Qwen35Config:
    # Core dimensions
    d_model: int = 4096
    vocab_size: int = 248320
    n_layers: int = 60
    rms_norm_eps: float = 1e-6
    max_position_embeddings: int = 262144

    # Layer pattern: every full_attention_interval-th layer is GQA,
    # the rest are DeltaNet linear attention.  With interval=4 this gives
    # layers 0,1,2 = linear, 3 = full, 4,5,6 = linear, 7 = full, ...
    full_attention_interval: int = 4

    # GQA (full attention) params
    gqa_n_q_heads: int = 32
    gqa_n_kv_heads: int = 2
    gqa_head_dim: int = 256
    gqa_partial_rotary_factor: float = 0.25   # RoPE on 25% of head_dim
    gqa_rope_theta: float = 10_000_000.0
    attn_output_gate: bool = True              # learnable output gate

    # DeltaNet (linear attention) params
    delta_n_qk_heads: int = 16
    delta_n_v_heads: int = 64
    delta_qk_head_dim: int = 128
    delta_v_head_dim: int = 128
    delta_conv_kernel: int = 4                 # causal conv1d width
    delta_chunk_size: int = 256                # chunk size for prefill

    # MoE params
    n_routed_experts: int = 512
    n_experts_per_token: int = 10
    moe_intermediate_size: int = 1024
    shared_expert_intermediate_size: int = 1024
    router_aux_loss_coef: float = 0.001

    # Dtype
    dtype: str = 'bfloat16'

    @classmethod
    def mini(cls) -> Qwen35Config:
        """Scaled-down config for development/smoke-testing.

        n_routed_experts=32 so it shards evenly across TP=32 (mlperf-v5p).
        For single-device Mac runs use tp=1 or tp=4.
        """
        return cls(
            d_model=1024,
            n_layers=8,
            max_position_embeddings=2048,
            # GQA — same head counts as full config so TP=32 sharding works
            gqa_n_q_heads=32,
            gqa_n_kv_heads=2,
            gqa_head_dim=128,
            gqa_rope_theta=10_000_000.0,
            # DeltaNet — same head counts as full config
            delta_n_qk_heads=16,
            delta_n_v_heads=64,
            delta_qk_head_dim=128,
            delta_v_head_dim=128,
            delta_chunk_size=4,  # small for fast compilation in tests
            # MoE — 32 experts divides evenly across TP=4/8/16/32
            n_routed_experts=32,
            n_experts_per_token=2,
            moe_intermediate_size=512,
            shared_expert_intermediate_size=512,
        )

    @classmethod
    def mid(cls) -> Qwen35Config:
        """Mid-size config for 4x TPU v5p (~8B params, ~16GB bf16).

        Same d_model and head geometry as full model, but 16 layers (4 groups)
        and 16 experts. Fits in host RAM during init and exercises the full
        sharding strategy on 4 TPU chips.
        """
        return cls(
            d_model=4096,
            n_layers=16,
            max_position_embeddings=8192,
            # GQA — same as full
            gqa_n_q_heads=32,
            gqa_n_kv_heads=2,
            gqa_head_dim=256,
            # DeltaNet — same as full
            delta_n_qk_heads=16,
            delta_n_v_heads=64,
            delta_qk_head_dim=128,
            delta_v_head_dim=128,
            # MoE — 16 experts (4 per device with TP=4)
            n_routed_experts=16,
            n_experts_per_token=4,
            moe_intermediate_size=1024,
            shared_expert_intermediate_size=1024,
        )

    @classmethod
    def mid_large(cls) -> Qwen35Config:
        """Larger mid config (~32B params). Needs ~64GB host RAM for init.

        Use if your TPU VM has sufficient host memory (200GB+).
        """
        return cls(
            d_model=4096,
            n_layers=32,
            max_position_embeddings=8192,
            gqa_n_q_heads=32,
            gqa_n_kv_heads=2,
            gqa_head_dim=256,
            delta_n_qk_heads=16,
            delta_n_v_heads=64,
            delta_qk_head_dim=128,
            delta_v_head_dim=128,
            n_routed_experts=64,
            n_experts_per_token=4,
            moe_intermediate_size=1024,
            shared_expert_intermediate_size=1024,
        )

    @classmethod
    def full(cls) -> Qwen35Config:
        """Full Qwen3.5-397B config (defaults)."""
        return cls()

    @property
    def n_groups(self) -> int:
        """Number of repeating 4-layer groups."""
        assert self.n_layers % self.full_attention_interval == 0
        return self.n_layers // self.full_attention_interval

    @property
    def gqa_rope_dim(self) -> int:
        """Number of dims that get RoPE (25% of head_dim)."""
        return int(self.gqa_head_dim * self.gqa_partial_rotary_factor)

    def layer_type(self, layer_idx: int) -> str:
        """Return 'linear' or 'full' for a given layer index."""
        if (layer_idx + 1) % self.full_attention_interval == 0:
            return 'full'
        return 'linear'
