#!/usr/bin/env python3
"""DeepSeek-v3 671B parametric roofline analysis.

Estimates theoretical performance limits for training on v7x TPU slices.
All FLOPs, memory, and communication costs are computed from first principles.

Usage:
    python roofline/dsv3_roofline.py --pdbs 2 --topology 4x8x8
    python roofline/dsv3_roofline.py --pdbs 1 --topology 4x4x4 --mini
"""

import argparse
import dataclasses
import math


# ============================================================================
# Hardware specs
# ============================================================================

@dataclasses.dataclass
class TPUSpec:
    name: str
    mxu_bf16_tflops: float    # Peak MXU BF16 TFLOP/s per chip
    hbm_bw_gbs: float         # Peak HBM bandwidth GB/s per chip (read+write)
    hbm_capacity_gb: float    # HBM capacity GB per chip
    vmem_mb: float            # VMEM per core MB
    cores_per_chip: int
    ici_bw_per_link_gbs: float  # Unidirectional ICI BW per link
    ici_links_per_chip: int     # Total ICI links (2 per axis × 3 axes)

    @property
    def ridge_point(self) -> float:
        """FLOP/byte where compute = memory bound."""
        return self.mxu_bf16_tflops * 1e3 / self.hbm_bw_gbs  # FLOP/byte


V7X = TPUSpec(
    name="v7x",
    mxu_bf16_tflops=2307.0,    # Per chip (per core = 1153.5)
    hbm_bw_gbs=7373.0,         # 7373 GB/s per direction per chip (conservative for roofline)
    hbm_capacity_gb=190.0,     # 95 GB per core × 2 = 190 GB per chip
    vmem_mb=64.0,              # Per core
    cores_per_chip=2,
    ici_bw_per_link_gbs=90.0,  # 90 GB/s per link per direction
    ici_links_per_chip=6,      # 6 links (2 per axis × 3 axes)
)

V5P = TPUSpec(
    name="v5p",
    mxu_bf16_tflops=459.0,
    hbm_bw_gbs=2800.0,
    hbm_capacity_gb=95.0,
    vmem_mb=128.0,
    cores_per_chip=2,
    ici_bw_per_link_gbs=50.0,  # TODO: confirm
    ici_links_per_chip=6,
)

V4 = TPUSpec(
    name="v4",
    mxu_bf16_tflops=275.0,     # Per chip
    hbm_bw_gbs=1200.0,         # 1.2 TB/s per chip
    hbm_capacity_gb=32.0,      # 32 GiB per chip
    vmem_mb=16.0,              # Per core
    cores_per_chip=1,          # Single core per chip
    ici_bw_per_link_gbs=50.0,  # 50 GB/s per link per direction
    ici_links_per_chip=6,      # 3D torus, 2 per axis
)


# ============================================================================
# Model config
# ============================================================================

@dataclasses.dataclass
class DSv3Config:
    """DeepSeek-v3 model configuration."""
    name: str = "deepseek3-671b"

    # Dimensions
    D: int = 7168           # hidden dim
    V: int = 129280         # vocab size
    L: int = 61             # total layers
    L_dense: int = 3        # dense MLP layers
    S: int = 4096           # sequence length

    # MLA
    H: int = 128            # num heads
    R_q: int = 1536         # query LoRA rank
    R_kv: int = 512         # KV LoRA rank
    d_nope: int = 128       # non-positional head dim
    d_rope: int = 64        # RoPE head dim
    d_v: int = 128          # value head dim

    # MoE
    E: int = 256            # num experts
    K: int = 8              # top-k experts per token
    D_moe: int = 2048       # MoE intermediate dim
    n_shared: int = 1       # shared experts

    # Dense MLP (for dense layers only)
    D_mlp: int = 18432

    @property
    def L_moe(self) -> int:
        return self.L - self.L_dense

    @property
    def qk_dim(self) -> int:
        return self.d_nope + self.d_rope


def mini_dsv3() -> DSv3Config:
    """Mini DeepSeek-v3 for iteration on smaller clusters."""
    return DSv3Config(
        name="deepseek3-mini",
        D=2048,
        V=32000,
        L=12,        # 2 dense + 10 MoE
        L_dense=2,
        S=2048,
        H=16,
        R_q=512,
        R_kv=256,
        d_nope=64,
        d_rope=32,
        d_v=64,
        E=32,
        K=4,
        D_moe=1024,
        n_shared=1,
        D_mlp=5504,
    )


def tiny_dsv3() -> DSv3Config:
    """Tiny DeepSeek-v3 for single-host smoke tests."""
    return DSv3Config(
        name="deepseek3-tiny",
        D=512, V=8192, L=4, L_dense=1, S=512,
        H=8, R_q=128, R_kv=64, d_nope=32, d_rope=16, d_v=32,
        E=8, K=2, D_moe=256, n_shared=1, D_mlp=1376,
    )


# ============================================================================
# Parallelism config
# ============================================================================

@dataclasses.dataclass
class ParallelConfig:
    """Parallelism strategy."""
    fsdp: int = 128       # FSDP weight sharding
    ep: int = 4           # Expert parallelism
    dp: int = 1           # Data parallelism
    tp: int = 1           # Tensor parallelism (unused for batch_split)

    # Topology
    topo_x: int = 4
    topo_y: int = 8
    topo_z: int = 8

    @property
    def num_chips(self) -> int:
        return self.topo_x * self.topo_y * self.topo_z

    @property
    def num_devices(self) -> int:
        """Num devices = num_chips × cores_per_chip (for megacore)."""
        return self.num_chips * 2  # v7x: 2 cores/chip


# ============================================================================
# ICI communication model
# ============================================================================

def chip_submesh(n_chips: int, topo: tuple) -> tuple:
    """Infer rectangular chip sub-mesh dimensions from chip count and physical topology.

    Greedily assigns full topo axes from largest to smallest until n_chips is allocated.
    Example: chip_submesh(32, (4,8,8)) → (8, 4, 1)  [8×4 plane within 8×8 plane]
             chip_submesh(4,  (4,8,8)) → (4, 1, 1)  [x-axis only]
    """
    dims = sorted(topo, reverse=True)   # descending: [8, 8, 4] for 4×8×8
    result = []
    remaining = n_chips
    for d in dims:
        if remaining >= d:
            result.append(d)
            remaining //= d
        else:
            result.append(remaining)
            remaining = 1
    return tuple(sorted(result, reverse=True))


@dataclasses.dataclass
class ICIModel:
    """ICI communication cost model for 3D torus.

    Bandwidth formula (user-validated for v7x):
        BW_per_chip = 2 * B_link / (max(a, b, c) / 4)
    where (a, b, c) are the chip dimensions of the collective sub-mesh
    and B_link = bw_per_link_gbs (90 GB/s for v7x).

    Intuition: a longer axis means more hops, reducing effective BW.
    The /4 factor is empirically calibrated for v7x ICI torus geometry.
    """

    bw_per_link_gbs: float = 90.0  # GB/s per link, unidirectional (v7x)
    latency_us: float = 1.0        # Per-hop latency in microseconds

    def collective_bw_gbs(self, chip_dims: tuple) -> float:
        """Effective bandwidth per chip for a collective on a chip sub-mesh.

        Formula: 2 * B_link / (max(chip_dims) / 4)
        """
        if not chip_dims:
            return float('inf')
        max_dim = max(chip_dims)
        if max_dim <= 1:
            return float('inf')
        return 2 * self.bw_per_link_gbs / (max_dim / 4)

    def collective_time_s(self, total_bytes: float, chip_dims: tuple) -> float:
        """Time for a collective (all-gather, all-reduce, A2A) on a chip sub-mesh.

        Uses total_bytes / BW_per_chip. For all-reduce, total_bytes = 2×msg_bytes
        (reduce-scatter + all-gather). Caller should pass the appropriate byte count.
        """
        if not chip_dims or max(chip_dims) <= 1 or total_bytes <= 0:
            return 0.0
        bw_bytes_s = self.collective_bw_gbs(chip_dims) * 1e9
        return total_bytes / bw_bytes_s

    # Legacy ring methods kept for compatibility (use collective_time_s for new code)
    def ring_allreduce_time_s(self, msg_bytes: float, axis_size: int,
                               num_links: int = 1) -> float:
        """All-reduce on a ring of `axis_size` nodes."""
        if axis_size <= 1:
            return 0.0
        factor = 2.0 * (axis_size - 1) / axis_size
        bw = self.bw_per_link_gbs * num_links * 1e9
        return factor * msg_bytes / bw + (2 * (axis_size - 1)) * self.latency_us * 1e-6

    def ring_allgather_time_s(self, msg_bytes: float, axis_size: int,
                               num_links: int = 1) -> float:
        """All-gather on a ring."""
        if axis_size <= 1:
            return 0.0
        factor = (axis_size - 1) / axis_size
        bw = self.bw_per_link_gbs * num_links * 1e9
        return factor * msg_bytes / bw + (axis_size - 1) * self.latency_us * 1e-6

    def ring_reducescatter_time_s(self, msg_bytes: float, axis_size: int,
                                   num_links: int = 1) -> float:
        """Reduce-scatter on a ring (same cost as all-gather)."""
        return self.ring_allgather_time_s(msg_bytes, axis_size, num_links)

    def hbm_transfer_time_s(self, msg_bytes: float, hbm_bw_gbs: float) -> float:
        """HBM read/write time with BW scaling for small transfers."""
        if msg_bytes <= 0:
            return 0.0
        # BW efficiency as function of transfer size
        if msg_bytes < 1e3:        # < 1 KB
            efficiency = 0.10
        elif msg_bytes < 1e4:      # < 10 KB
            efficiency = 0.30
        elif msg_bytes < 1e5:      # < 100 KB
            efficiency = 0.60
        elif msg_bytes < 1e6:      # < 1 MB
            efficiency = 0.80
        elif msg_bytes < 1e7:      # < 10 MB
            efficiency = 0.90
        else:                      # >= 10 MB
            efficiency = 1.00
        effective_bw = hbm_bw_gbs * efficiency * 1e9
        return msg_bytes / effective_bw


# ============================================================================
# Per-layer FLOP and memory computation
# ============================================================================

@dataclasses.dataclass
class LayerCosts:
    """FLOP and memory costs for one forward pass of a layer."""
    name: str
    flops: float             # Total FLOPs
    hbm_read_bytes: float    # Bytes read from HBM
    hbm_write_bytes: float   # Bytes written to HBM
    weight_bytes: float      # Weight bytes (for FSDP gather)
    description: str = ""


def compute_mla_costs(cfg: DSv3Config, B: int) -> list[LayerCosts]:
    """MLA attention costs for B tokens per device."""
    D, R_q, R_kv, H = cfg.D, cfg.R_q, cfg.R_kv, cfg.H
    d_nope, d_rope, d_v, qk_dim = cfg.d_nope, cfg.d_rope, cfg.d_v, cfg.qk_dim
    S = cfg.S
    costs = []

    # wq_a: [D, R_q]
    flops = 2 * B * D * R_q
    w_bytes = D * R_q * 2
    act_bytes = B * D * 2 + B * R_q * 2
    costs.append(LayerCosts("mla/wq_a", flops, act_bytes + w_bytes, B * R_q * 2, w_bytes,
                            f"[{D},{R_q}] matmul"))

    # wq_b: [R_q, H × qk_dim]
    flops = 2 * B * R_q * H * qk_dim
    w_bytes = R_q * H * qk_dim * 2
    costs.append(LayerCosts("mla/wq_b", flops, B * R_q * 2 + w_bytes, B * H * qk_dim * 2, w_bytes,
                            f"[{R_q},{H*qk_dim}] matmul"))

    # wkv_a: [D, R_kv + d_rope]
    kv_out = R_kv + d_rope
    flops = 2 * B * D * kv_out
    w_bytes = D * kv_out * 2
    costs.append(LayerCosts("mla/wkv_a", flops, B * D * 2 + w_bytes, B * kv_out * 2, w_bytes,
                            f"[{D},{kv_out}] matmul"))

    # wkv_b: [R_kv, H × (d_nope + d_v)]
    kv_head = d_nope + d_v
    flops = 2 * B * R_kv * H * kv_head
    w_bytes = R_kv * H * kv_head * 2
    costs.append(LayerCosts("mla/wkv_b", flops, B * R_kv * 2 + w_bytes, B * H * kv_head * 2, w_bytes,
                            f"[{R_kv},{H*kv_head}] matmul"))

    # Attention (QK^T + softmax + AV)
    attn_flops = 2 * B * H * S * qk_dim + 2 * B * H * S * d_v  # QK + AV
    # Memory: Q,K,V read from VMEM (already there from projections)
    # Output: B × H × d_v written
    costs.append(LayerCosts("mla/attention", attn_flops,
                            B * H * qk_dim * 2 * 2 + B * H * d_v * 2,  # Q,K,V
                            B * H * d_v * 2, 0,
                            f"flash attention S={S}"))

    # Output projection: [H × d_v, D]
    out_dim = H * d_v
    flops = 2 * B * out_dim * D
    w_bytes = out_dim * D * 2
    costs.append(LayerCosts("mla/out_proj", flops, B * out_dim * 2 + w_bytes, B * D * 2, w_bytes,
                            f"[{out_dim},{D}] matmul"))

    return costs


def compute_moe_costs(cfg: DSv3Config, B: int, par: ParallelConfig) -> list[LayerCosts]:
    """MoE layer costs for B tokens per device."""
    D, E, K, D_moe = cfg.D, cfg.E, cfg.K, cfg.D_moe
    EP = par.ep
    costs = []

    # Gate logits: [D, E]
    flops = 2 * B * D * E
    w_bytes = D * E * 2
    costs.append(LayerCosts("moe/gate", flops, B * D * 2 + w_bytes, B * E * 2, w_bytes,
                            f"[{D},{E}] gate logits"))

    # Router top-k + dispatch (negligible FLOPs, but communication)
    costs.append(LayerCosts("moe/route_dispatch", 0,
                            B * E * 2,  # read logits
                            B * K * 4,  # write indices
                            0, "top-k + sort"))

    # EP all-gather for activations
    gather_bytes = B * D * 2 * EP  # All tokens from all EP peers
    costs.append(LayerCosts("moe/ep_allgather", 0, 0, 0, 0,
                            f"all-gather {gather_bytes/1e6:.1f} MB across EP={EP}"))

    # GMM: wi_0 (gate projection)
    # Tokens: B × EP × K (all expert-token pairs)
    B_moe = B * EP * K
    # Weight per device: [E, D, D_moe/EP] = [256, 7168, 512]
    flops = 2 * B_moe * D * (D_moe // EP)
    w_bytes = E * D * (D_moe // EP) * 2
    costs.append(LayerCosts("moe/gmm_wi0", flops,
                            B_moe * D * 2 + w_bytes,
                            B_moe * (D_moe // EP) * 2,
                            w_bytes,
                            f"GMM [{E},{D},{D_moe//EP}] × {B_moe} tokens"))

    # GMM: wi_1 (up projection) — same shape
    costs.append(LayerCosts("moe/gmm_wi1", flops,
                            B_moe * D * 2 + w_bytes,
                            B_moe * (D_moe // EP) * 2,
                            w_bytes,
                            f"GMM [{E},{D},{D_moe//EP}] × {B_moe} tokens"))

    # SiLU + element-wise multiply (negligible)
    costs.append(LayerCosts("moe/activation", 2 * B_moe * (D_moe // EP),
                            B_moe * (D_moe // EP) * 2 * 2,
                            B_moe * (D_moe // EP) * 2,
                            0, "silu(gate) * up"))

    # GMM: wo (down projection)
    flops = 2 * B_moe * (D_moe // EP) * D
    w_bytes = E * (D_moe // EP) * D * 2
    costs.append(LayerCosts("moe/gmm_wo", flops,
                            B_moe * (D_moe // EP) * 2 + w_bytes,
                            B_moe * D * 2,
                            w_bytes,
                            f"GMM [{E},{D_moe//EP},{D}] × {B_moe} tokens"))

    # EP reduce-scatter
    scatter_bytes = B * D * 2 * EP
    costs.append(LayerCosts("moe/ep_reducescatter", 0, 0, 0, 0,
                            f"reduce-scatter {scatter_bytes/1e6:.1f} MB across EP={EP}"))

    # Shared expert: wi_0 + wi_1 + wo
    for name, shape_desc, fl in [
        ("moe/shared_wi0", f"[{D},{D_moe}]", 2 * B * D * D_moe),
        ("moe/shared_wi1", f"[{D},{D_moe}]", 2 * B * D * D_moe),
        ("moe/shared_wo", f"[{D_moe},{D}]", 2 * B * D_moe * D),
    ]:
        w = D * D_moe * 2 if "wi" in name else D_moe * D * 2
        costs.append(LayerCosts(name, fl, B * D * 2 + w, B * D * 2, w, shape_desc))

    return costs


def compute_dense_mlp_costs(cfg: DSv3Config, B: int) -> list[LayerCosts]:
    """Dense MLP costs for B tokens."""
    D, D_mlp = cfg.D, cfg.D_mlp
    costs = []
    for name, fl in [
        ("dense_mlp/wi_gate", 2 * B * D * D_mlp),
        ("dense_mlp/wi_up", 2 * B * D * D_mlp),
        ("dense_mlp/wo", 2 * B * D_mlp * D),
    ]:
        w = D * D_mlp * 2
        costs.append(LayerCosts(name, fl, B * D * 2 + w, B * D * 2, w))
    return costs


# ============================================================================
# Roofline estimation
# ============================================================================

def estimate_step_time(cfg: DSv3Config, par: ParallelConfig, spec: TPUSpec,
                       pdbs: int, verbose: bool = True,
                       gradient_checkpoint: bool = False,
                       moe_backend: str = "jax_ep",
                       weight_dtype: str = "bf16") -> dict:
    """Estimate per-step training time.

    Args:
        pdbs: Per-device batch size (sequences per device).
        gradient_checkpoint: Whether gradient checkpointing is used.
            If True, adds one extra forward recompute in backward. FLOPs = 4×fwd.
            If False, FLOPs = 3×fwd.
        moe_backend: Which MoE implementation to model:
            "jax_ep" — shard_map+psum, computes all (E/EP) local experts for all tokens.
                       FLOPs: (E/EP) × T × 3 matmuls. Works for any EP.
            "ragged_dot" — token-sorted grouped matmul, only computes K expert-token pairs.
                           FLOPs: K × T × 3 matmuls. Currently only works for EP=1 in mini_dsv3.
        weight_dtype: "bf16" (2 bytes/param) or "fp8" (1 byte/param).
            FP8 halves FSDP gather bytes (weights stored/transferred in FP8).
            FLOPs are unchanged (compute still in bf16/fp32 via dequant).
    """
    weight_bytes_per_param = 1 if weight_dtype == "fp8" else 2  # bytes per parameter
    B = pdbs * cfg.S  # tokens per device
    EP = par.ep
    ici = ICIModel(bw_per_link_gbs=spec.ici_bw_per_link_gbs)
    cores_per_chip = spec.cores_per_chip
    topo = (par.topo_x, par.topo_y, par.topo_z)

    # Derive chip-level sub-mesh dimensions for each collective axis.
    # JAX devices = chips × cores_per_chip (v7x: 2 cores/chip).
    fsdp_chips = par.fsdp // cores_per_chip
    ep_chips   = par.ep   // cores_per_chip
    fsdp_submesh = chip_submesh(fsdp_chips, topo)
    ep_submesh   = chip_submesh(ep_chips,   topo)
    fsdp_bw_gbs  = ici.collective_bw_gbs(fsdp_submesh)
    ep_bw_gbs    = ici.collective_bw_gbs(ep_submesh)

    # === Compute costs (forward) ===
    mla_costs = compute_mla_costs(cfg, B)
    dense_costs = compute_dense_mlp_costs(cfg, B)

    # MoE FLOPs depend on backend:
    # - "jax_ep": each device processes all B tokens × E/EP local experts (wastes (E/EP)/K × compute)
    # - "ragged_dot": each device processes only the K-assigned tokens per expert
    if moe_backend == "ragged_dot":
        # Ideal: K expert-token pairs per token (only EP=1 currently working in mini_dsv3)
        moe_costs = compute_moe_costs(cfg, B, par)  # existing model is correct for ragged_dot
    else:
        # "jax_ep": all-expert einsum, (E/EP) experts × all T tokens
        # FLOPs per device per layer = 3 matmuls × 2 × T × (E/EP) × D × D_moe
        # vs ragged_dot: 3 × 2 × T × K × D × D_moe
        # Overhead factor: (E/EP) / K
        moe_costs_baseline = compute_moe_costs(cfg, B, par)
        E_over_EP = cfg.E // EP if EP > 0 else cfg.E
        overhead = E_over_EP / cfg.K  # how many more ops than ideal
        # Scale only the GMM FLOPs (not gate or collectives)
        moe_costs = []
        for c in moe_costs_baseline:
            if c.name.startswith("moe/gmm"):
                moe_costs.append(LayerCosts(
                    c.name, c.flops * overhead, c.hbm_read_bytes * overhead,
                    c.hbm_write_bytes * overhead, c.weight_bytes, c.description))
            else:
                moe_costs.append(c)

    # Total forward FLOPs per layer type
    mla_fwd_flops = sum(c.flops for c in mla_costs)
    moe_fwd_flops = sum(c.flops for c in moe_costs)
    dense_fwd_flops = sum(c.flops for c in dense_costs)

    # Output head
    output_flops = 2 * B * cfg.D * cfg.V

    total_fwd = (mla_fwd_flops * cfg.L + moe_fwd_flops * cfg.L_moe +
                 dense_fwd_flops * cfg.L_dense + output_flops)

    # Backward: dX + dW per matmul = 2× forward FLOPs
    total_bwd = total_fwd * 2.0
    # Gradient checkpointing: rerun forward in backward for activation recompute
    total_recompute = total_fwd if gradient_checkpoint else 0.0
    total_flops = total_fwd + total_bwd + total_recompute

    # === Compute time at peak ===
    compute_time_s = total_flops / (spec.mxu_bf16_tflops * 1e12)

    # === FSDP weight gather cost ===
    # Each FSDP gather must happen:
    #   - Once in forward pass (all-gather weights before each layer)
    #   - Once in backward pass (re-gather weights for grad computation)
    #   - Once for gradient checkpointing recompute (if enabled)
    # PLUS reduce-scatter of gradients (= 1× gather volume)
    # Total passes:
    #   Without ckpt: 2 gather + 1 reduce-scatter = 3× forward gather volume
    #   With ckpt:    3 gather + 1 reduce-scatter = 4× forward gather volume
    #
    # Note: ring_allgather_time_s expects msg_bytes = TOTAL bytes after gather
    #       (not per-device shard). See ICIModel.ring_allgather_time_s.

    # Weight dtype scaling: FP8 = 1 byte/param vs BF16 = 2 bytes/param.
    # Applies to all FSDP gather sizes (weights are stored and gathered at this dtype).
    w_scale = weight_bytes_per_param / 2  # 0.5 for fp8, 1.0 for bf16

    # MLA weights: sum of all MLA weight matrices (full, unsharded)
    mla_weight_bytes = sum(c.weight_bytes for c in mla_costs) * w_scale
    # All-gather on FSDP sub-mesh: total_bytes = full weight size
    mla_gather_time_1pass = ici.collective_time_s(mla_weight_bytes, fsdp_submesh)

    # MoE weights: inside shard_map, 3 separate all-gathers per layer (wi_0, wi_1, wo)
    # Each gathers the FSDP (D) dimension for E/EP local experts.
    # Full weight per EP partition: [E/EP, D, D_moe] per matrix.
    E_local = cfg.E // EP if EP > 0 else cfg.E
    moe_w_per_ep_partition = E_local * cfg.D * cfg.D_moe * 2 * w_scale  # bytes per matrix
    moe_gather_time_1pass = 3 * ici.collective_time_s(moe_w_per_ep_partition, fsdp_submesh)

    # Dense MLP weights (only 3 layers)
    dense_weight_bytes = sum(c.weight_bytes for c in dense_costs) * w_scale
    dense_gather_time_1pass = ici.collective_time_s(dense_weight_bytes, fsdp_submesh)

    # Total gather passes per step
    num_passes = 4 if gradient_checkpoint else 3  # (fwd + recompute + bwd) gathers + 1 RS
    total_mla_gather   = mla_gather_time_1pass * cfg.L * num_passes
    total_moe_gather   = moe_gather_time_1pass * cfg.L_moe * num_passes
    total_dense_gather = dense_gather_time_1pass * cfg.L_dense * num_passes
    total_gather_time  = total_mla_gather + total_moe_gather + total_dense_gather

    # === EP collective per MoE layer ===
    # Our shard_map+psum uses psum (all-reduce) per MoE layer.
    # psum size = B×D (activations, bf16); all-reduce = 2 × one-way transfer.
    psum_bytes = B * cfg.D * 2
    ep_time_per_layer = ici.collective_time_s(2 * psum_bytes, ep_submesh)  # ×2 for all-reduce
    # Happens in each forward pass (+ recompute if ckpt)
    ep_passes = 3 if gradient_checkpoint else 2  # fwd + (recompute) + bwd
    total_ep_time = ep_time_per_layer * cfg.L_moe * ep_passes

    # === HBM bandwidth limited time ===
    total_hbm_bytes = 0
    for costs, n_layers in [(mla_costs, cfg.L), (moe_costs, cfg.L_moe), (dense_costs, cfg.L_dense)]:
        layer_bytes = sum(c.hbm_read_bytes + c.hbm_write_bytes for c in costs)
        total_hbm_bytes += layer_bytes * n_layers
    # Scale by pass count (fwd + bwd + optional recompute)
    hbm_passes = 4.0 if gradient_checkpoint else 3.0  # fwd + 2×bwd matmuls + optional recompute
    total_hbm_bytes *= hbm_passes / 2.0  # base is fwd; bwd = 2× fwd; divide by 2 to normalize
    hbm_time_s = total_hbm_bytes / (spec.hbm_bw_gbs * 1e9)

    # === Total estimate ===
    # Lower bound: max(compute, HBM) — neither can be escaped
    # Roofline estimate: assumes 50% overlap of compute with communication
    # (XLA async collectives + pipeline can achieve 50-70% overlap in practice)
    comm_time = total_gather_time + total_ep_time
    theoretical_min = max(compute_time_s, hbm_time_s)
    roofline_estimate = max(compute_time_s, hbm_time_s) + comm_time * 0.5

    # Achieved efficiency vs roofline (accounts for kernel gaps, XLA overhead, etc.)
    ACHIEVED_EFFICIENCY = 0.70
    achieved_estimate = roofline_estimate / ACHIEVED_EFFICIENCY

    tps_chip_roofline = B / roofline_estimate
    tps_chip_achieved = B / achieved_estimate

    results = {
        "pdbs": pdbs,
        "tokens_per_device": B,
        "total_flops": total_flops,
        "total_fwd_flops": total_fwd,
        "compute_time_s": compute_time_s,
        "hbm_time_s": hbm_time_s,
        "fsdp_gather_time_s": total_gather_time,
        "fsdp_mla_gather_s": total_mla_gather,
        "fsdp_moe_gather_s": total_moe_gather,
        "ep_collective_time_s": total_ep_time,
        "total_comm_time_s": comm_time,
        "theoretical_min_s": theoretical_min,
        "roofline_estimate_s": roofline_estimate,
        "achieved_estimate_s": achieved_estimate,
        "mfu_roofline": total_flops / (roofline_estimate * spec.mxu_bf16_tflops * 1e12),
        "mfu_achieved": total_flops / (achieved_estimate * spec.mxu_bf16_tflops * 1e12),
        "tflops_per_device": total_flops / 1e12,
        "tps_per_chip_roofline": tps_chip_roofline,
        "tps_per_chip_achieved": tps_chip_achieved,
        "tps_cluster_roofline": tps_chip_roofline * par.num_chips,
        "tps_cluster_achieved": tps_chip_achieved * par.num_chips,
        "achieved_efficiency": ACHIEVED_EFFICIENCY,
        "gradient_checkpoint": gradient_checkpoint,
        "moe_backend": moe_backend,
    }

    if verbose:
        ckpt_str = "+ckpt" if gradient_checkpoint else ""
        print(f"{'='*72}")
        print(f"DeepSeek-v3 Roofline: {cfg.name}")
        print(f"Hardware: {spec.name} {par.topo_x}x{par.topo_y}x{par.topo_z} "
              f"({par.num_chips} chips / {par.num_devices} devices)")
        print(f"PDBS={pdbs}, S={cfg.S}, FSDP={par.fsdp}, EP={par.ep}, "
              f"backend={moe_backend}{ckpt_str}")
        print(f"{'='*72}")
        print()

        fwd_mult = 4 if gradient_checkpoint else 3
        print(f"FLOPs ({fwd_mult}×fwd, bwd=2×fwd):    {total_flops/1e12:.1f} TFLOP")
        print(f"  MLA ({cfg.L} layers):            {mla_fwd_flops*cfg.L/1e12:.1f} TFLOP fwd")
        print(f"  MoE ({cfg.L_moe} layers, {moe_backend}): "
              f"{moe_fwd_flops*cfg.L_moe/1e12:.1f} TFLOP fwd")
        e_over_ep = cfg.E // EP if EP > 0 else cfg.E
        if moe_backend == "jax_ep":
            print(f"    ({e_over_ep} local experts × T, vs ideal K={cfg.K} × T = "
                  f"{e_over_ep/cfg.K:.0f}× overhead)")
        print(f"  Dense ({cfg.L_dense} layers):        "
              f"{dense_fwd_flops*cfg.L_dense/1e12:.1f} TFLOP fwd")
        print()

        print(f"Compute at peak ({spec.mxu_bf16_tflops:.0f} TFLOP/s/chip):")
        print(f"  Time:                      {compute_time_s:.3f} s")
        print()

        print(f"HBM bandwidth ({spec.hbm_bw_gbs:.0f} GB/s/chip):")
        print(f"  Total bytes:               {total_hbm_bytes/1e9:.1f} GB")
        print(f"  Time:                      {hbm_time_s:.3f} s")
        print()

        print(f"Communication (ICI {spec.ici_bw_per_link_gbs:.0f} GB/s/link):")
        print(f"  FSDP sub-mesh: {fsdp_submesh} chips → {fsdp_bw_gbs:.0f} GB/s/chip")
        print(f"  EP   sub-mesh: {ep_submesh} chips   → {ep_bw_gbs:.0f} GB/s/chip")
        print(f"  FSDP MLA gather ({num_passes}×):    {total_mla_gather:.3f} s")
        print(f"  FSDP MoE gather ({num_passes}× 3w): {total_moe_gather:.3f} s")
        print(f"  FSDP dense gather ({num_passes}×):  {total_dense_gather:.3f} s")
        print(f"  EP psum ({ep_passes}×, EP={EP}):      {total_ep_time:.3f} s")
        print(f"  TOTAL comm:                {comm_time:.3f} s")
        print(f"  Comm / Compute ratio:      {comm_time/max(compute_time_s,1e-9):.1f}×")
        print()

        print(f"Bottleneck: {'COMPUTE' if compute_time_s > comm_time else 'COMMUNICATION'} "
              f"(comm = {comm_time/max(compute_time_s,1e-9):.1f}× compute)")
        print()

        tps_chip_theor = B / theoretical_min
        print(f"Throughput estimates:")
        print(f"  {'':32s} {'Step(s)':>8} {'MFU':>6} {'TPS/chip':>10} {'TPS/clust':>10}")
        print(f"  {'-'*32} {'-'*8} {'-'*6} {'-'*10} {'-'*10}")
        print(f"  {'Theoretical (compute only)':<32s} "
              f"{theoretical_min:>8.3f} {'100%':>6} "
              f"{tps_chip_theor:>10,.0f} {tps_chip_theor*par.num_chips:>10,.0f}")
        print(f"  {'Roofline (50% comm overlap)':<32s} "
              f"{roofline_estimate:>8.3f} {results['mfu_roofline']*100:>5.1f}% "
              f"{tps_chip_roofline:>10,.0f} {tps_chip_roofline*par.num_chips:>10,.0f}")
        print(f"  {'Achieved (~70% of roofline)':<32s} "
              f"{achieved_estimate:>8.3f} {results['mfu_achieved']*100:>5.1f}% "
              f"{tps_chip_achieved:>10,.0f} {tps_chip_achieved*par.num_chips:>10,.0f}")
        print()

    return results


def sharding_search(cfg: DSv3Config, spec: TPUSpec, pdbs_list: list[int],
                    gradient_checkpoint: bool = True,
                    moe_backend: str = "jax_ep",
                    topo: tuple[int, int, int] = (4, 8, 8),
                    weight_dtype: str = "bf16") -> None:
    """Print a table comparing all EP configurations for given PDBS values."""
    num_chips = topo[0] * topo[1] * topo[2]
    num_devices = num_chips * 2  # v7x: 2 cores/chip

    ep_options = [ep for ep in [1, 2, 4, 8, 16, 32, 64]
                  if num_devices % ep == 0 and num_devices // ep >= 1]

    ckpt_str = "+ckpt" if gradient_checkpoint else ""
    dtype_str = f" [{weight_dtype}]"
    print(f"\n{'='*90}")
    print(f"Sharding search: {cfg.name} on {spec.name} {topo[0]}x{topo[1]}x{topo[2]} "
          f"({num_chips} chips, backend={moe_backend}{ckpt_str}{dtype_str})")
    print(f"{'='*90}")

    # Memory feasibility check: min FSDP to fit params+optimizer on device HBM
    hbm_per_device_gb = spec.hbm_capacity_gb / spec.cores_per_chip  # per JAX device
    # Rough: 10 bytes/param for Adam (2 bf16 + 4 fp32 m + 4 fp32 v), 2 for SGD
    param_bytes_per_param = 10  # conservative (Adam)
    total_param_bytes = 651e9 * param_bytes_per_param  # ~651B params
    min_fsdp_adam = math.ceil(total_param_bytes / (hbm_per_device_gb * 1e9))
    print(f"Memory note: min FSDP={min_fsdp_adam} for Adam (10 bytes/param, "
          f"{hbm_per_device_gb:.0f} GB/device); "
          f"min FSDP={math.ceil(min_fsdp_adam/5)} for SGD (2 bytes/param)")

    for pdbs in pdbs_list:
        B = pdbs * cfg.S
        print(f"\nPDBS={pdbs}  (B={B} tokens/device)")
        print(f"  {'EP':>4} {'FSDP':>6} {'Compute':>10} {'FSDP-comm':>10} "
              f"{'EP-comm':>8} {'Roofline':>10} {'TPS/chip':>10} {'Mem?':>6} {'Bottleneck'}")
        print(f"  {'-'*4} {'-'*6} {'-'*10} {'-'*10} {'-'*8} {'-'*10} {'-'*10} {'-'*6} {'-'*12}")

        best_tps = 0.0
        best_ep = None
        for ep in ep_options:
            fsdp = num_devices // ep
            par = ParallelConfig(fsdp=fsdp, ep=ep,
                                 topo_x=topo[0], topo_y=topo[1], topo_z=topo[2])
            r = estimate_step_time(cfg, par, spec, pdbs, verbose=False,
                                   gradient_checkpoint=gradient_checkpoint,
                                   moe_backend=moe_backend,
                                   weight_dtype=weight_dtype)
            tps = r["tps_per_chip_roofline"]
            comm_s = r["fsdp_gather_time_s"] + r["ep_collective_time_s"]
            bottleneck = "COMPUTE" if r["compute_time_s"] > comm_s else "COMM"
            mem_ok = "OK" if fsdp >= min_fsdp_adam else ("SGD" if fsdp >= min_fsdp_adam // 5 else "OOM")
            flag = " ◄" if tps > best_tps else ""
            if tps > best_tps and mem_ok != "OOM":
                best_tps = tps
                best_ep = ep
            print(f"  {ep:>4} {fsdp:>6} {r['compute_time_s']:>9.3f}s "
                  f"{r['fsdp_gather_time_s']:>9.3f}s "
                  f"{r['ep_collective_time_s']:>7.3f}s "
                  f"{r['roofline_estimate_s']:>9.3f}s "
                  f"{tps:>10,.0f}{flag}  {mem_ok:>6}  [{bottleneck}]")
        print(f"  → Best feasible EP={best_ep} (FSDP={num_devices//best_ep if best_ep else '?'}) "
              f"at PDBS={pdbs}: {best_tps:,.0f} TPS/chip")


def main():
    parser = argparse.ArgumentParser(description="DeepSeek-v3 roofline analysis")
    parser.add_argument("--pdbs", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--topology", default="4x8x8", help="Chip topology (e.g., 4x8x8)")
    parser.add_argument("--ep", type=int, default=8)
    parser.add_argument("--fsdp", type=int, default=64)
    parser.add_argument("--model", default="full", choices=["full", "mini", "tiny"])
    parser.add_argument("--mini", action="store_true", help="Use mini model (legacy)")
    parser.add_argument("--hw", default="v7x", choices=["v7x", "v5p", "v4"])
    parser.add_argument("--gradient_checkpoint", action="store_true",
                        help="Model gradient checkpointing (adds recompute forward pass)")
    parser.add_argument("--moe_backend", default="jax_ep",
                        choices=["jax_ep", "ragged_dot"],
                        help="jax_ep: (E/EP)×T compute (current impl); "
                             "ragged_dot: K×T compute (ideal, EP=1 only today)")
    parser.add_argument("--search", action="store_true",
                        help="Search all EP configurations and print comparison table")
    parser.add_argument("--weight_dtype", default="bf16", choices=["bf16", "fp8"],
                        help="Weight dtype for FSDP gather size: bf16 (2 bytes) or fp8 (1 byte)")
    args = parser.parse_args()

    topo = [int(x) for x in args.topology.split("x")]
    assert len(topo) == 3

    MODEL_MAP = {"full": DSv3Config, "mini": mini_dsv3, "tiny": tiny_dsv3}
    if args.mini:
        cfg = mini_dsv3()
    else:
        cfg = MODEL_MAP[args.model]()
    HW_MAP = {"v7x": V7X, "v5p": V5P, "v4": V4}
    spec = HW_MAP[args.hw]

    if args.search:
        sharding_search(cfg, spec, args.pdbs,
                        gradient_checkpoint=args.gradient_checkpoint,
                        moe_backend=args.moe_backend,
                        topo=tuple(topo),
                        weight_dtype=args.weight_dtype)
    else:
        par = ParallelConfig(
            fsdp=args.fsdp, ep=args.ep,
            topo_x=topo[0], topo_y=topo[1], topo_z=topo[2],
        )
        for pdbs in args.pdbs:
            estimate_step_time(cfg, par, spec, pdbs,
                               gradient_checkpoint=args.gradient_checkpoint,
                               moe_backend=args.moe_backend,
                               weight_dtype=args.weight_dtype)
            print()


if __name__ == "__main__":
    main()
