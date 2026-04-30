"""DeepSeek-V4-Pro configuration.

Mirrors the released config.json from
huggingface.co/deepseek-ai/DeepSeek-V4-Pro (preview, 2026-04-24).

The interesting bits relative to qwen35:
- MLA-style attention (q-LoRA + kv-LoRA), not GQA. head_dim=512 is the
  latent KV head dim; qk_rope_head_dim=64 is the partial-RoPE split.
- Hybrid sparse attention: every layer runs SWA-128 in parallel with one
  of {C4, C128} compressed-KV branches. Per-layer schedule via
  COMPRESS_RATIOS.
- Indexer (separate small attention head) selects top-1024 positions per
  query for the C4 branch.
- 384 routed experts + 1 shared, top-6 routing with sqrtsoftplus + noaux_tc.
- mHC residuals with hc_mult=4 (Sinkhorn over a (4,4) matrix per layer
  per token, 20 iters).
- FP8 e4m3 weights with [128,128] block scales; MoE experts in FP4.
"""

from __future__ import annotations

from dataclasses import dataclass, field


# Per-layer KV compression schedule, length = num_hidden_layers + num_nextn_predict_layers.
# 128 -> HCA layer (dense over 128:1-compressed KV)
#   4 -> CSA layer (top-1024 indexer over 4:1-compressed KV)
#   0 -> uncompressed (the trailing MTP head)
# Lifted verbatim from config.json.
COMPRESS_RATIOS: tuple[int, ...] = (
    128, 128,
    4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128,
    4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128,
    4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128,
    4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128,
    4, 128, 4,
    0,
)
assert len(COMPRESS_RATIOS) == 62  # 61 layers + 1 MTP


@dataclass(frozen=True)
class DSv4Config:
    # Core
    d_model: int = 7168
    vocab_size: int = 129280
    n_layers: int = 61
    n_mtp_layers: int = 1
    rms_norm_eps: float = 1e-6
    max_position_embeddings: int = 1_048_576
    original_max_position_embeddings: int = 65_536  # YaRN base

    # MLA attention
    n_attention_heads: int = 128
    n_kv_heads: int = 1
    head_dim: int = 512  # latent KV head dim
    qk_rope_head_dim: int = 64
    q_lora_rank: int = 1536
    o_lora_rank: int = 1024
    o_groups: int = 16
    rope_theta: float = 10_000.0
    compress_rope_theta: float = 160_000.0
    yarn_factor: float = 16.0
    yarn_beta_fast: float = 32.0
    yarn_beta_slow: float = 1.0

    # Indexer (CSA top-k selector)
    index_n_heads: int = 64
    index_head_dim: int = 128
    index_topk: int = 1024

    # Sliding window (every layer)
    sliding_window: int = 128

    # MoE
    n_routed_experts: int = 384
    n_shared_experts: int = 1
    n_experts_per_token: int = 6
    moe_intermediate_size: int = 3072
    routed_scaling_factor: float = 2.5
    norm_topk_prob: bool = True
    scoring_func: str = 'sqrtsoftplus'
    topk_method: str = 'noaux_tc'
    swiglu_limit: float = 10.0

    # mHC (Manifold-Constrained Hyper-Connections)
    hc_mult: int = 4              # n in the paper (expansion rate)
    hc_eps: float = 1e-6
    hc_sinkhorn_iters: int = 20

    # Hash layers (TBD — present in config, mechanism not yet documented)
    n_hash_layers: int = 3

    # Quantization
    weight_dtype: str = 'fp8_e4m3'           # most weights
    expert_dtype: str = 'fp4'                # MoE expert tensors
    quant_block_size: tuple[int, int] = (128, 128)
    scale_dtype: str = 'ue8m0'

    # Per-layer compression schedule
    compress_ratios: tuple[int, ...] = field(default_factory=lambda: COMPRESS_RATIOS)

    # ---- helpers ----
    @property
    def qk_nope_head_dim(self) -> int:
        return self.head_dim - self.qk_rope_head_dim

    def layer_kind(self, layer_idx: int) -> str:
        """Return 'csa' (top-k over 4:1) | 'hca' (dense over 128:1) | 'mtp' (no compr)."""
        ratio = self.compress_ratios[layer_idx]
        if ratio == 4:
            return 'csa'
        if ratio == 128:
            return 'hca'
        if ratio == 0:
            return 'mtp'
        raise ValueError(f'Unexpected compress_ratio={ratio} at layer {layer_idx}')

    @classmethod
    def full(cls) -> 'DSv4Config':
        return cls()

    @classmethod
    def mini(cls) -> 'DSv4Config':
        """Tiny config for unit tests / Mac dev. Architecture-faithful, scaled-down."""
        return cls(
            d_model=512,
            vocab_size=4096,
            n_layers=6,
            n_mtp_layers=0,
            n_attention_heads=8,
            n_kv_heads=1,
            head_dim=64,
            qk_rope_head_dim=16,
            q_lora_rank=128,
            o_lora_rank=128,
            o_groups=2,
            index_n_heads=4,
            index_head_dim=32,
            index_topk=64,
            sliding_window=32,
            n_routed_experts=8,
            n_shared_experts=1,
            n_experts_per_token=2,
            moe_intermediate_size=256,
            max_position_embeddings=4096,
            original_max_position_embeddings=1024,
            yarn_factor=4.0,
            # 6 layers: HCA, HCA, CSA, HCA, CSA, HCA  (matches the V4-Pro pattern shape)
            compress_ratios=(128, 128, 4, 128, 4, 128),
        )
