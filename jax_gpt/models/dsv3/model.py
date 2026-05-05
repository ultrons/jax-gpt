"""DeepSeek-V3 671B — pure JAX training model.

Architecture: MLA attention + MoE FFN (EP=8, FSDP=32, TP=2).
Every component has jax.named_scope for xla_shell profile analysis.

MoE backends:
  "jax"           — JAX EP shard_map + jax.vjp backward. No token dropping.
  "fused_ep_moe_v4" — Pallas forward (streaming v1) + Pallas backward (v4 kernel).

Sharding target (4×8×8 bodaborg, 256 chips × 2 TCs = 512 JAX devices):
  Mesh axes: dp=1, ep=8, fsdp=32, tp=2
  Tokens:    P("fsdp", None)  — T/32 per device; replicated across EP
  MoE wts:   wi_0/wi_1 = _moe_wi_spec(); wo = _moe_wo_spec()
  Attn wts:  wq_b/wkv_b = P("fsdp", "tp") column-parallel; w_out = P("fsdp", None)
  TP scope:  attention only — heads column-parallel; AG before w_out + dense wo.
             MoE uses EP+FSDP only (TP axis ignored inside shard_map).

SC gather constraints (v7x SparseCore BF16):
  (1) Source rows ≤ 65536 — enforced by T_fsdp chunking and embed chunking.
  (2) Gather indices must be 1-D — enforced by flat_tok in embedding lookup.
"""

from __future__ import annotations

import dataclasses
import functools
import math

import jax
import jax.numpy as jnp
from jax import random
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

# Set by train.py: --moe_shard_e_with_fsdp / --moe_shard_d_with_fsdp.
# Default: F (=2048) on fsdp.  D mode: D (=7168) on fsdp.  E mode: E on (ep,fsdp).
_SHARD_E_WITH_FSDP = False
_MOE_NO_WEIGHT_AG = False   # set by train.py --moe_no_weight_ag
_SHARD_D_WITH_FSDP = False

def _moe_wi_spec():
    if _SHARD_E_WITH_FSDP:
        return P(("ep", "fsdp"), None, None)
    if _SHARD_D_WITH_FSDP:
        return P("ep", "fsdp", None)        # D on fsdp (wi's D is dim 1)
    return P("ep", None, "fsdp")  # noqa: do-not-substitute

def _moe_wo_spec():
    if _SHARD_E_WITH_FSDP:
        return P(("ep", "fsdp"), None, None)
    if _SHARD_D_WITH_FSDP:
        return P("ep", None, "fsdp")        # D on fsdp (wo's D is dim 2)
    return P("ep", "fsdp", None)  # noqa: do-not-substitute

def _moe_wi_ag_axis():
    if _SHARD_E_WITH_FSDP: return 0
    if _SHARD_D_WITH_FSDP: return 1
    return 2

def _moe_wo_ag_axis():
    if _SHARD_E_WITH_FSDP: return 0
    if _SHARD_D_WITH_FSDP: return 2
    return 1


# ============================================================================
# Config
# ============================================================================

@dataclasses.dataclass
class ModelConfig:
    # Transformer dimensions
    D: int = 7168        # model dimension
    H: int = 128         # attention heads
    d_v: int = 128       # value head dim
    d_nope: int = 128    # non-RoPE key/query dim per head
    d_rope: int = 64     # RoPE dim per head
    R_q: int = 1536      # MLA query low-rank dim
    R_kv: int = 512      # MLA KV low-rank dim
    # MoE
    E: int = 256         # number of routed experts
    K: int = 8           # top-k experts per token
    D_moe: int = 2048    # expert hidden dim (D_ffn = 2048 per expert, SwiGLU)
    # Dense FFN (for the first L_dense layers)
    D_mlp: int = 18432   # dense MLP hidden dim
    # Topology
    L: int = 61          # total transformer layers
    L_dense: int = 3     # first L_dense layers use dense FFN
    V: int = 102400      # vocabulary size (padded to multiple of 128)
    S: int = 4096        # sequence length
    # Training
    norm_eps: float = 1e-6
    moe_aux_loss_coeff: float = 1e-4
    # Routing scale applied to top-K weights after normalization. MaxText DSv3
    # 671B uses 2.5 (configs/models/deepseek3-671b.yml: routed_scaling_factor).
    # We default to 1.0 because production training (v304-) was tuned for
    # implicit-1.0 — the model converges fine without it. Set to 2.5 to match
    # MaxText behavior when loading their checkpoints.
    routed_scaling_factor: float = 1.0
    # Group-limited routing (DSv3): split E experts into n_routing_groups,
    # pick top topk_routing_group groups by per-group top-2 score sum, then
    # take top-K experts only within selected groups. -1 = disabled (flat
    # top-K). MaxText DSv3 671B uses (8, 4). Default -1 preserves v304.
    n_routing_groups: int = -1
    topk_routing_group: int = -1
    # MLA softmax scale: when use_yarn_scale=True, apply MaxText's YaRN-
    # adjusted scale = (1/sqrt(qk_dim)) * (0.1 * attn_mscale * log(rope_factor) + 1.0)^2.
    # MaxText DSv3 671B: rope_factor=40, attn_mscale=1.0 → effective scale
    # ≈ 0.1352 (vs vanilla 0.0722, ~1.87× sharper softmax). Default off for
    # v304 backward-compat.
    use_yarn_scale: bool = False
    attn_mscale: float = 1.0
    rope_factor: float = 1.0
    # Backend
    moe_backend: str = "jax"     # "jax" | "fused_ep_moe_v4"
    attn_backend: str = "splash" # "splash" | "jax"
    # Runtime (set by train.py)
    name: str = "full_671b"
    mesh: Mesh | None = None
    gradient_checkpoint: bool = False
    dtype: str = "bfloat16"
    use_cp: bool = True  # context parallelism on EP axis (sequence sharding)
    moe_xlayer_prefetch: bool = False  # cross-layer FSDP weight AG prefetch (gmm_ag only)
    moe_use_sc_scatter: bool = False   # use SC gather-reduce instead of HBM scatter (v305 — slower in practice)
    moe_use_gmm_v2: bool = False       # use Pallas gmm_v2 with fused silu (v307+); jax.vjp backward
    moe_n_chunks: int = 2              # token chunking inside MoE body (1=no chunking)
    moe_shard_e_with_fsdp: bool = False  # shard E along (ep,fsdp); AG along axis 0
    moe_fp8_weights: bool = False        # cast wi_0/wi_1/wo to fp8_e4m3fn before FSDP AG
                                         # (halves the per-layer 7 GB AG allocation)
    moe_no_weight_ag: bool = False       # use colrow+A2A body for tp>1+ep>1, removes
                                         # per-layer weight AllGather (1.4 GB transient)
    moe_debug_nans: bool = False         # insert breakpoint_if_nonfinite checks at strategic
                                         # points in _expert_mlp_gmm_ag_body (post-AG, post-sort,
                                         # post-ragged_dot×3, post-scatter, pre-psum_scatter).
                                         # Halts in pdb on first NaN. v304 default is OFF.

    @property
    def L_moe(self) -> int:
        return self.L - self.L_dense

    @property
    def qk_dim(self) -> int:
        return self.d_nope + self.d_rope

    @property
    def jax_dtype(self):
        return jnp.bfloat16 if self.dtype == "bfloat16" else jnp.float32


def full_671b_config() -> ModelConfig:
    return ModelConfig(name="full_671b")


def mini_config() -> ModelConfig:
    """Full 671B dimensions, 2 layers — catches all SC/shape bugs at low compile cost."""
    return ModelConfig(name="mini", L=2, L_dense=1)


def debug_671b_config() -> ModelConfig:
    """Alias for mini_config (full-shape, 2-layer)."""
    return mini_config()


def full_671b_maxtext_config() -> ModelConfig:
    """DSv3 671B with all DSv3-specific architectural knobs that MaxText
    trains with (configs/models/deepseek3-671b.yml). Preserves v304 default
    of full_671b_config() in every other respect.

    Use this when loading the MaxText DSv3-FSDP checkpoint
    (gs://mlperf-6-submission/ckpt0424-fsdp/0/items) — its weights were
    trained against this exact set of architectural choices, and any single
    mismatch silently corrupts the forward pass.
    """
    cfg = full_671b_config()
    cfg.name = "full_671b_maxtext"
    cfg.V = 129280                       # MaxText DSv3 vocab (vs our 102400 default)
    cfg.routed_scaling_factor = 2.5      # post-norm routing weight scale
    cfg.n_routing_groups = 8             # split 256 experts into 8 groups
    cfg.topk_routing_group = 4           # select top-4 groups per token
    cfg.use_yarn_scale = True            # YaRN-modified MLA softmax scale
    cfg.attn_mscale = 1.0                # mscale=1.0 in MaxText config
    cfg.rope_factor = 40.0               # YaRN extrapolation factor
    return cfg


# Registry for train.py --config flag
CONFIGS = {
    "full":           full_671b_config,
    "full_maxtext":   full_671b_maxtext_config,
    "mini":           mini_config,
    "debug":          debug_671b_config,
}


# ============================================================================
# Mesh / sharding helpers
# ============================================================================

@dataclasses.dataclass
class ShardConfig:
    fsdp: int = 1
    ep: int = 1
    tp: int = 1
    dp: int = 1
    explicit_axes: bool = False  # Mesh axis_types=Explicit (needed for pcast/reduced)

    def create_mesh(self, devices=None) -> Mesh:
        import numpy as np
        if devices is None:
            devices = jax.devices()
        n = len(devices)
        self.dp = n // (self.fsdp * self.ep * self.tp)
        assert n == self.fsdp * self.ep * self.tp * self.dp, (
            f"Device count {n} != fsdp={self.fsdp}*ep={self.ep}"
            f"*tp={self.tp}*dp={self.dp}")
        # Sort devices into (x, y, z, core) physical grid.
        dev_map = {}
        for d in devices:
            x, y, z = d.coords
            c = d.core_on_chip
            dev_map[(x, y, z, c)] = d
        xs = sorted(set(k[0] for k in dev_map))
        ys = sorted(set(k[1] for k in dev_map))
        zs = sorted(set(k[2] for k in dev_map))
        cs = sorted(set(k[3] for k in dev_map))
        grid = np.empty((len(xs), len(ys), len(zs), len(cs)), dtype=object)
        for (x, y, z, c), d in dev_map.items():
            grid[xs.index(x), ys.index(y), zs.index(z), cs.index(c)] = d
        nx, ny, nz, nc = grid.shape

        # Topology-aware mapping policy:
        #   * TP (intra-chip) → cores axis (C) when tp == nc
        #   * EP (critical path, A2A or AG dispatch) → smallest physical axes
        #     whose product equals self.ep — keeps EP on a tight torus
        #   * FSDP → all remaining axes
        # Physical axis order [X, Y, Z, C] is consumed greedily by EP.
        #
        # Examples:
        #   4×4×16, ep=16, fsdp=32, tp=1 → EP=X·Y (4·4), FSDP=Z·C (16·2)
        #   4×4×16, ep=16, fsdp=16, tp=2 → TP=C (2), EP=X·Y (4·4), FSDP=Z (16)
        #   4×8×8,  ep=4,  fsdp=128, tp=1 → EP=X (4), FSDP=Y·Z·C (8·8·2)
        #   4×8×8,  ep=8,  fsdp=64,  tp=1 → EP=X·Y2(?) — error if not factorable
        physical = [("X", nx), ("Y", ny), ("Z", nz), ("C", nc)]

        # Reserve cores axis for TP if requested and matches.
        tp_takes_cores = self.tp > 1 and nc == self.tp
        if self.tp > 1 and not tp_takes_cores:
            raise ValueError(
                f"tp={self.tp} requested but cores axis nc={nc} doesn't match. "
                f"Implement multi-axis TP placement if needed.")
        ep_pool = [p for p in physical if not (tp_takes_cores and p[0] == "C")]

        # Pick EP axes: prefer a single physical axis whose size matches self.ep
        # exactly (e.g. EP=8 → Y(8) on 4×8×8). Fall back to greedy contiguous
        # prefix if no single axis matches (e.g. EP=16 → X·Y on 4×4×16).
        ep_axes = []  # list of (name, size, idx_in_grid)
        prod = 1
        idx_map = {"X": 0, "Y": 1, "Z": 2, "C": 3}
        for name, sz in ep_pool:
            if sz == self.ep:
                ep_axes = [(name, sz, idx_map[name])]
                prod = sz
                break
        if prod != self.ep:
            for name, sz in ep_pool:
                if prod == self.ep:
                    break
                ep_axes.append((name, sz, idx_map[name]))
                prod *= sz
        if prod != self.ep:
            raise ValueError(
                f"ep={self.ep} cannot be formed from physical axes {ep_pool} "
                f"by single-axis match or contiguous prefix. Got product {prod}.")
        fsdp_axes = [(name, sz, idx_map[name]) for name, sz in ep_pool
                     if (name, sz, idx_map[name]) not in ep_axes]
        fsdp_prod = 1
        for _, sz, _ in fsdp_axes:
            fsdp_prod *= sz
        assert fsdp_prod == self.fsdp, (
            f"fsdp={self.fsdp} but remaining physical axes give {fsdp_prod}: "
            f"{fsdp_axes}")

        # Build (dp, ep, fsdp, tp) mesh by transposing grid into the chosen order.
        # Order on the input grid is [X=0, Y=1, Z=2, C=3].
        ep_idxs   = [a[2] for a in ep_axes]
        fsdp_idxs = [a[2] for a in fsdp_axes]
        tp_idxs   = [3] if tp_takes_cores else []
        # New axis order: ep first, then fsdp, then tp.
        perm = ep_idxs + fsdp_idxs + tp_idxs
        mesh_grid = grid.transpose(perm)
        mesh_devices = mesh_grid.reshape(1, self.ep, self.fsdp, self.tp)

        ep_desc   = "·".join(f"{a[0]}({a[1]})" for a in ep_axes)
        fsdp_desc = "·".join(f"{a[0]}({a[1]})" for a in fsdp_axes)
        tp_desc   = f"C({nc})" if tp_takes_cores else "—"
        type_desc = "Explicit" if self.explicit_axes else "Auto"
        print(f"[mesh] X={nx} Y={ny} Z={nz} C={nc} → "
              f"EP={ep_desc} | FSDP={fsdp_desc} | TP={tp_desc} ({type_desc})")
        if self.explicit_axes:
            from jax.sharding import AxisType
            return Mesh(mesh_devices,
                        axis_names=("dp", "ep", "fsdp", "tp"),
                        axis_types=(AxisType.Explicit,) * 4)
        return Mesh(mesh_devices, axis_names=("dp", "ep", "fsdp", "tp"))


def _has_tp(cfg: ModelConfig) -> bool:
    return cfg.mesh is not None and cfg.mesh.shape.get("tp", 1) > 1


def _batch_ax(cfg: ModelConfig):
    """Primary batch sharding axis name."""
    if cfg.mesh is None:
        return None
    if cfg.mesh.shape.get("dp", 1) > 1:
        return "dp"
    return "fsdp"


def _seq_ax(cfg: ModelConfig):
    """Sequence sharding axis (for context parallelism)."""
    if cfg.mesh is None or not cfg.use_cp:
        return None
    if cfg.mesh.shape.get("ep", 1) > 1:
        return "ep"  # EP axis does sequence parallelism (CP=EP)
    return None


def _carry_spec(cfg: ModelConfig):
    """Per-layer carry spec for `(B, S, D)` activation.

    CP on  → P(batch, ep, None)         — B by fsdp, S by ep
    CP off + EP>1 → P((batch, ep), None, None) — B jointly by fsdp×ep
    EP=1   → P(batch, None, None)        — B by fsdp only
    """
    if cfg.mesh is None:
        return None
    batch_ax = _batch_ax(cfg)
    seq_ax   = _seq_ax(cfg)
    ep_size  = cfg.mesh.shape.get("ep", 1)
    if seq_ax is not None:
        return P(batch_ax, seq_ax, None)
    if ep_size > 1:
        return P((batch_ax, "ep"), None, None)
    return P(batch_ax, None, None)


# ============================================================================
# Parameter sharding
# ============================================================================

def shard_params(params: dict, cfg: ModelConfig, mesh: Mesh) -> dict:
    """Re-shard an existing params dict onto mesh (used when loading checkpoints).
    init_params already handles sharding during training; this is for eval flows.
    """
    # TODO: implement per-tensor sharding for checkpoint loading.
    # For now, use init_params sharding as the reference.
    raise NotImplementedError(
        "shard_params not yet implemented for the rewritten model. "
        "Use init_params(cfg, key, mesh) for training."
    )


# ============================================================================
# Normalization
# ============================================================================

def rms_norm(x, weight, eps: float = 1e-6):
    with jax.named_scope("rms_norm"):
        variance = jnp.mean(x * x, axis=-1, keepdims=True)
        x = x * jax.lax.rsqrt(variance + eps)
        return x * weight


# ============================================================================
# MLA Attention
# ============================================================================

def _attention_softmax_scale(cfg) -> float:
    """Softmax scale used in MLA. Defaults to 1/sqrt(qk_dim). With YaRN
    (cfg.use_yarn_scale), applies MaxText DSv3 671B formula:
        scale = (1/sqrt(qk_dim)) * (0.1 * attn_mscale * log(rope_factor) + 1.0)^2
    """
    base = 1.0 / math.sqrt(cfg.qk_dim)
    if not cfg.use_yarn_scale:
        return base
    m = 0.1 * cfg.attn_mscale * math.log(cfg.rope_factor) + 1.0
    return base * m * m


def _splash_attention(query, key, value, scale: float):
    """Causal Splash attention via Pallas (TPU-only).

    Block sizes follow MaxText DSv3 v5p config: 2048 on Q/KV blocks, fused
    backward kernel that produces dQ and dKV in one Q/K/V pass (vs separate
    splash_mha_dq + splash_mha_dkv kernels each re-reading Q/K/V).
    """
    from jax.experimental.pallas.ops.tpu.splash_attention import (
        splash_attention_kernel as splash,
        splash_attention_mask as mask_lib,
    )
    B, H, S, _ = query.shape
    mask = mask_lib.CausalMask(shape=(S, S))
    multi_mask = mask_lib.MultiHeadMask(masks=[mask] * H)
    BLK = 2048
    block_sizes = splash.BlockSizes(
        block_q=min(BLK, S), block_kv=min(BLK, S),
        block_kv_compute=min(BLK, S), block_q_dkv=min(BLK, S),
        block_kv_dkv=min(BLK, S), block_kv_dkv_compute=min(BLK, S),
        block_q_dq=None, block_kv_dq=None,  # fused bwd kernel produces dQ + dKV together
        use_fused_bwd_kernel=True,
    )
    fn = splash.make_splash_mha(mask=multi_mask, head_shards=1,
                                 q_seq_shards=1, block_sizes=block_sizes)
    return jax.vmap(fn)(query * scale, key, value)


def _splash_cp_attention(query, key, value, scale: float,
                         S_local: int, S_full: int, rank: int, H: int):
    """CP Splash: Q(S_local) × K(S_full) with offset causal mask for EP rank."""
    from jax.experimental.pallas.ops.tpu.splash_attention import (
        splash_attention_kernel as splash,
        splash_attention_mask as mask_lib,
    )
    mask = mask_lib.CausalMask(shape=(S_local, S_full), offset=rank * S_local)
    multi_mask = mask_lib.MultiHeadMask(masks=[mask] * H)
    BLK = 2048
    block_q = min(BLK, S_local)
    block_kv = min(BLK, S_full)
    block_sizes = splash.BlockSizes(
        block_q=block_q, block_kv=block_kv,
        block_kv_compute=block_kv, block_q_dkv=block_q,
        block_kv_dkv=block_kv, block_kv_dkv_compute=block_kv,
        block_q_dq=None, block_kv_dq=None,
        use_fused_bwd_kernel=True,
    )
    fn = splash.make_splash_mha(mask=multi_mask, head_shards=1,
                                 q_seq_shards=1, block_sizes=block_sizes)
    return jax.vmap(fn)(query * scale, key, value)



def mla_attention(x, params, positions, cfg: ModelConfig, use_cp: bool | None = None):
    """Multi-head Latent Attention (DeepSeek-V3).

    Low-rank KV compression: kv_a (D→R_kv+d_rope), kv_b (R_kv→H*(d_nope+d_v)).
    Low-rank Q  compression: q_a  (D→R_q),          q_b  (R_q→H*qk_dim).

    use_cp=None (default) → take from cfg.use_cp; explicit False overrides.
    """
    if use_cp is None:
        use_cp = cfg.use_cp
    from jax.experimental.shard_map import shard_map as _smap

    B, S, D = x.shape
    H, d_v = cfg.H, cfg.d_v
    d_nope, d_rope = cfg.d_nope, cfg.d_rope
    qk_dim = cfg.qk_dim
    cp_axis = _seq_ax(cfg) if use_cp else None
    softmax_scale = _attention_softmax_scale(cfg)
    ck = jax._src.ad_checkpoint.checkpoint_name

    with jax.named_scope("mla_attention"):
        h = rms_norm(x, params["pre_attn_norm"], cfg.norm_eps)

        if cp_axis is not None and cfg.mesh is not None:
            # Context parallelism: do projections with GSPMD (handles FSDP weight sharding),
            # then wrap only the KV AllGather + attention core in shard_map.
            batch_ax = _batch_ax(cfg)

            # Q projection (GSPMD handles weight AllGather)
            with jax.named_scope("q_proj"):
                q_a = h @ params["wq_a"]
                q_a = rms_norm(q_a, params["q_norm_scale"], cfg.norm_eps)
                q_a = ck(q_a, "q_a")  # save+offload: 96 MB/layer, skip wq_b recompute in bwd
                q   = (q_a @ params["wq_b"]).reshape(B, S, H, qk_dim)

            # KV projection — local S_local
            with jax.named_scope("kv_proj"):
                kv_a = h @ params["wkv_a"]   # (B, S_local, R_kv+d_rope)
                kv_a = ck(kv_a, "kv_a")  # save+offload: 36 MB/layer, skip wkv_b expansion in bwd

            # Shard_map for: AllGather kv_a on EP + attention + output proj
            _bsrd_spec = P(batch_ax, "ep", None, None)  # (B, S, H, d)
            _bsd_spec  = P(batch_ax, "ep", None)        # (B, S, D)
            _bs_spec   = P(batch_ax, "ep")               # (B, S)
            _bsr_spec  = P(batch_ax, "ep", None)         # (B, S, R_kv+d_rope)

            @functools.partial(_smap, mesh=cfg.mesh,
                in_specs=(_bsrd_spec,   # q: (B, S_local, H, qk_dim)
                          _bsr_spec,    # kv_a: (B, S_local, R_kv+d_rope)
                          _bs_spec,     # positions: (B, S_local)
                          P(None),        # kv_norm_scale
                          P(None, None),  # wkv_b — needs full R_kv after KV AllGather
                          P(None, None)), # w_out
                out_specs=_bsd_spec, check_rep=False)
            def _cp_attn_core(q_, kv_a_, pos_, kvn_, wkv_b_, wout_):
                B_l, S_l, _ = kv_a_.shape

                # ★ AllGather compressed KV on "ep" — 288 MB ★
                kv_a_full = jax.lax.all_gather(kv_a_, "ep", axis=1, tiled=True)
                S_full = kv_a_full.shape[1]

                # Expand KV
                kv_a_norm = rms_norm(kv_a_full[..., :cfg.R_kv], kvn_, cfg.norm_eps)
                kv = (kv_a_norm @ wkv_b_).reshape(B_l, S_full, H, d_nope + d_v)
                k_nope, v = kv[..., :d_nope], kv[..., d_nope:]
                k_rope = jnp.broadcast_to(
                    kv_a_full[..., cfg.R_kv:].reshape(B_l, S_full, 1, d_rope),
                    (B_l, S_full, H, d_rope))

                # RoPE
                q_nope, q_rope = q_[..., :d_nope], q_[..., d_nope:]
                cos_q, sin_q = _rope_freqs(S_l, d_rope, pos_, q_.dtype)
                q_rope = _apply_rope(q_rope, cos_q, sin_q)
                positions_full = jnp.broadcast_to(jnp.arange(S_full), (B_l, S_full))
                cos_k, sin_k = _rope_freqs(S_full, d_rope, positions_full, q_.dtype)
                k_rope = _apply_rope(k_rope, cos_k, sin_k)

                q_full = jnp.concatenate([q_nope, q_rope], axis=-1)
                k_full = jnp.concatenate([k_nope, k_rope], axis=-1)

                scale = softmax_scale  # captured from outer scope (cfg-driven)

                # Splash Attention: Q(S_local) × K(S_full) with offset causal mask.
                # CausalMask(shape=(S_local, S_full), offset=ep_rank*S_local)
                # mask[i,j] = (i + offset) >= j — correct causal for this CP shard.
                q_t = q_full.transpose(0, 2, 1, 3)   # (B, H, S_local, qk_dim)
                k_t = k_full.transpose(0, 2, 1, 3)   # (B, H, S_full, qk_dim)
                v_t = v.transpose(0, 2, 1, 3)         # (B, H, S_full, d_v)

                # Splash with per-rank static causal offset.
                # ep_rank is a tracer (one shard_map body for all devices), but
                # its value space is the static enumeration 0..ep_size-1, so we
                # build N Splash kernels (one per rank, each with its own
                # CausalMask offset) and dispatch via lax.switch.
                ep_size_static = cfg.mesh.shape["ep"]
                S_l_static  = q_t.shape[2]
                S_full_static = k_t.shape[2]

                def _make_branch(rank: int):
                    def branch(q_, k_, v_):
                        return _splash_cp_attention(
                            q_, k_, v_, scale,
                            S_l_static, S_full_static, rank, H)
                    return branch

                ep_rank = jax.lax.axis_index("ep")
                attn_out = jax.lax.switch(
                    ep_rank,
                    [_make_branch(r) for r in range(ep_size_static)],
                    q_t, k_t, v_t,
                )

                # Output projection
                attn_flat = attn_out.transpose(0, 2, 1, 3).reshape(B_l, S_l, H * d_v)
                return attn_flat @ wout_

            out = _cp_attn_core(q, kv_a, positions,
                                params["kv_norm_scale"], params["wkv_b"], params["w_out"])
            out = ck(out, "attn_proj_out")  # offload: 448 MB/layer, skip Splash bwd recompute
            return out

        # Non-CP path: standard MLA attention
        # Q projection (low-rank)
        with jax.named_scope("q_proj"):
            q_a = h @ params["wq_a"]
            q_a = rms_norm(q_a, params["q_norm_scale"], cfg.norm_eps)
            q_a = ck(q_a, "q_a")  # save+offload: 96 MB/layer, skip wq_b recompute in bwd
            q   = q_a @ params["wq_b"]
            q   = q.reshape(B, S, H, qk_dim)

        # KV projection (low-rank)
        with jax.named_scope("kv_proj"):
            kv_a = h @ params["wkv_a"]
            kv_a = ck(kv_a, "kv_a")  # save+offload: 36 MB/layer, skip wkv_b expansion in bwd
            S_kv = S
            kv_a_norm = rms_norm(kv_a[..., :cfg.R_kv],
                                 params["kv_norm_scale"], cfg.norm_eps)
            kv    = kv_a_norm @ params["wkv_b"]
            kv    = kv.reshape(B, S_kv, H, d_nope + d_v)
            k_nope, v = kv[..., :d_nope], kv[..., d_nope:]
            k_rope = kv_a[..., cfg.R_kv:].reshape(B, S_kv, 1, d_rope)
            k_rope = jnp.broadcast_to(k_rope, (B, S_kv, H, d_rope))

        # Apply RoPE
        with jax.named_scope("rope"):
            q_nope, q_rope = q[..., :d_nope], q[..., d_nope:]
            cos, sin = _rope_freqs(S, d_rope, positions, x.dtype)
            q_rope = _apply_rope(q_rope, cos, sin)
            k_rope = _apply_rope(k_rope, cos, sin)
            q = jnp.concatenate([q_nope, q_rope], axis=-1)
            k = jnp.concatenate([k_nope, k_rope], axis=-1)

        scale = softmax_scale

        # Attention compute
        with jax.named_scope("attn_compute"):
            q_t = q.transpose(0, 2, 1, 3)
            k_t = k.transpose(0, 2, 1, 3)
            v_t = v.transpose(0, 2, 1, 3)

            if cfg.attn_backend == "jax":
                attn_w = jnp.einsum("bhsq,bhtq->bhst", q_t, k_t) * scale
                causal = jnp.tril(jnp.ones((S, S), dtype=jnp.bool_))
                attn_w = jnp.where(causal[None, None], attn_w, jnp.finfo(q_t.dtype).min)
                attn_w = jax.nn.softmax(attn_w.astype(jnp.float32), axis=-1).astype(q_t.dtype)
                attn_out = jnp.einsum("bhst,bhtd->bhsd", attn_w, v_t)
            elif cfg.mesh is not None:
                batch_ax = _batch_ax(cfg)
                ep_size = cfg.mesh.shape.get("ep", 1)
                if _has_tp(cfg) and ep_size > 1 and not use_cp:
                    # TP shards heads, EP+FSDP joint-shard batch — keeps PDBS small.
                    _bspec = P((batch_ax, "ep"), "tp", None, None)
                elif _has_tp(cfg):
                    _bspec = P(batch_ax, "tp", None, None)
                elif ep_size > 1 and not use_cp:
                    # Dense layers with EP: batch sharded by (fsdp, ep), full S.
                    _bspec = P((batch_ax, "ep"), None, None, None)
                else:
                    _bspec = P(batch_ax, None, None, None)

                @functools.partial(_smap, mesh=cfg.mesh,
                                   in_specs=(_bspec, _bspec, _bspec),
                                   out_specs=_bspec, check_rep=False)
                def _attn_local(q_, k_, v_):
                    return _splash_attention(q_, k_, v_, scale)

                attn_out = _attn_local(q_t, k_t, v_t)
            else:
                attn_out = _splash_attention(q_t, k_t, v_t, scale)

        # Output projection
        with jax.named_scope("out_proj"):
            attn_flat = attn_out.transpose(0, 2, 1, 3).reshape(B, S, H * d_v)
            out = attn_flat @ params["w_out"]
            out = ck(out, "attn_proj_out")  # offload: 448 MB/layer, skip Splash bwd recompute

    return out


def _rope_freqs(S: int, d: int, positions, dtype):
    """Compute cos/sin for RoPE."""
    inv_freq = 1.0 / (10000.0 ** (jnp.arange(0, d, 2, dtype=jnp.float32) / d))
    t = positions[..., None].astype(jnp.float32)           # (B, S, 1)
    freqs = t * inv_freq[None, None, :]                    # (B, S, d//2)
    emb = jnp.concatenate([freqs, freqs], axis=-1)         # (B, S, d)
    return jnp.cos(emb).astype(dtype), jnp.sin(emb).astype(dtype)


def _apply_rope(x, cos, sin):
    """Apply rotary position embedding. x: (B, S, H, d)."""
    d = x.shape[-1]
    x1, x2 = x[..., :d // 2], x[..., d // 2:]
    rotated = jnp.concatenate([-x2, x1], axis=-1)
    return x * cos[:, :, None, :] + rotated * sin[:, :, None, :]


# ============================================================================
# MoE routing
# ============================================================================

def moe_routing(x, gate_weight, gate_bias, cfg: ModelConfig):
    """Top-k sigmoid routing (DeepSeek-V3 style).

    Uses sort_key_val instead of lax.top_k AND instead of take_along_axis to
    avoid SC packed-element-gather errors on v7x.

    Both lax.top_k and jnp.take_along_axis with K-column index tensors
    (shape (T, K)) hit the SC gather constraint at EP=32 TP=1:
      "Expected packed element gather indices to either be 1 dimensional
       or to be bitpacked" (gather_emitter.cc, num_indices_per_output != 1).
    With TP>1 GSPMD's extra sharding constraints happened to avoid this; with
    TP=1 they don't — the 2D index tensor goes straight to SC.

    Fix: sort the unbiased scores array by the same neg_biased key in a second
    sort_key_val call.  sort_key_val is a sort op (not a gather), so it never
    touches the SC gather emitter.
    """
    with jax.named_scope("moe_routing"):
        with jax.named_scope("gate_logits"):
            logits = jnp.einsum("bsd,de->bse", x, gate_weight)

        with jax.named_scope("top_k"):
            scores = jax.nn.sigmoid(logits.astype(jnp.float32))
            biased = scores + gate_bias.astype(jnp.float32)

            # DSv3 group-limited routing (MaxText: layers/moe.py:expert_group_mask).
            # Split E into n_routing_groups, pick top topk_routing_group groups
            # by per-group top-2-sum, mask out experts in unselected groups.
            if cfg.n_routing_groups > 0:
                B_, S_, E_ = biased.shape
                groups = cfg.n_routing_groups
                epg = E_ // groups
                topk_g = cfg.topk_routing_group
                biased_g = biased.reshape(B_, S_, groups, epg)
                top2_in_group, _ = jax.lax.top_k(biased_g, k=2)
                group_scores = jnp.sum(top2_in_group, axis=-1)  # (B,S,groups)
                _, group_idx = jax.lax.top_k(group_scores, k=topk_g)
                gmask = jax.nn.one_hot(group_idx, num_classes=groups,
                                       dtype=jnp.float32).sum(axis=-2)  # (B,S,groups)
                expert_mask = jnp.broadcast_to(
                    gmask[..., None], (B_, S_, groups, epg)).reshape(B_, S_, E_)
                biased = jnp.where(expert_mask > 0, biased, jnp.float32(-1e30))

            # Sort by descending biased score (sort neg_biased ascending).
            neg_biased = -biased
            B, S, E = neg_biased.shape
            flat_neg_biased = neg_biased.reshape(B * S, E)

            # Sort 1: expert indices sorted by biased score descending.
            idx_all = jnp.broadcast_to(jnp.arange(E), (B * S, E))
            _, sorted_idx = jax.lax.sort_key_val(flat_neg_biased, idx_all, dimension=1)
            top_k_indices = sorted_idx[:, :cfg.K].reshape(B, S, cfg.K)

            # Sort 2: unbiased scores sorted by the same key → no take_along_axis,
            # no 2D SC gather.  sorted_scores[:, :K] gives the unbiased sigmoid
            # scores for the top-K biased experts (same permutation as sort 1).
            _, sorted_scores = jax.lax.sort_key_val(
                flat_neg_biased, scores.reshape(B * S, E), dimension=1)
            top_k_weights = sorted_scores[:, :cfg.K].reshape(B, S, cfg.K)

        # Normalize weights (sum to 1 per token), then DeepSeek-V3 routing scale.
        top_k_weights = top_k_weights / (top_k_weights.sum(axis=-1, keepdims=True) + 1e-9)
        if cfg.routed_scaling_factor != 1.0:
            top_k_weights = top_k_weights * jnp.float32(cfg.routed_scaling_factor)
        return top_k_weights, top_k_indices, scores


# ============================================================================
# MoE JAX backend — EP shard_map, no token dropping
# ============================================================================

def _expert_mlp_ep_body(flat_x, wi_0, wi_1, wo, flat_indices, flat_weights,
                         K: int, ep_axis: str, max_tpe: int):
    """Per-EP-device expert MoE body. Called inside shard_map over ("ep","fsdp").

    Tokens are P("fsdp") — replicated across EP, no EP token dispatch needed.
    Each EP device handles E_local experts. EP psum aggregates contributions.

    SC gather constraint: source rows ≤ 65536.
    When T_fsdp > 65536, split into ceil(T/65536) equal chunks.

    flat_x:       (T_fsdp, D)                   FSDP-sharded tokens
    wi_0, wi_1:   (E_local, D, D_moe/fsdp)      gate/up, D_moe on FSDP axis
    wo:           (E_local, D_moe/fsdp, D)       down proj
    flat_indices: (T_fsdp, K)                    global expert IDs
    flat_weights: (T_fsdp, K)                    routing weights
    """
    from .kernels.fused_moe_bwd.backward_kernel import sc_gather_rows  # noqa: E402

    T, D = flat_x.shape
    E_local = wi_0.shape[0]
    my_idx    = jax.lax.axis_index(ep_axis)
    local_start = my_idx * E_local

    def _compute_block(x_b, idx_b, w_b):
        """Compute MoE output for one T_b-token block (T_b ≤ SC_MAX_ROWS)."""
        T_b = x_b.shape[0]
        flat_idx  = idx_b.reshape(-1)                   # (T_b*K,)
        flat_w    = w_b.reshape(-1)                     # (T_b*K,)
        token_ids = jnp.repeat(jnp.arange(T_b), K)     # (T_b*K,)

        valid     = (flat_idx >= local_start) & (flat_idx < local_start + E_local)
        local_exp = jnp.where(valid, flat_idx - local_start, E_local).astype(jnp.int32)

        with jax.named_scope("sort_by_expert"):
            argsorted   = jnp.argsort(local_exp, stable=True)
            sorted_tids = token_ids[argsorted]
            sorted_ws   = flat_w[argsorted]
            sorted_exp  = local_exp[argsorted]

        sorted_tids = jnp.concatenate([sorted_tids, jnp.zeros(max_tpe, dtype=jnp.int32)])
        sorted_ws   = jnp.concatenate([sorted_ws,   jnp.zeros(max_tpe, dtype=jnp.float32)])

        with jax.named_scope("expert_boundaries"):
            expert_starts = jnp.searchsorted(sorted_exp, jnp.arange(E_local))
            expert_ends   = jnp.searchsorted(sorted_exp, jnp.arange(1, E_local + 1))

        tids_list, valid_list, ws_list = [], [], []
        for e in range(E_local):
            start_e = expert_starts[e]
            n_e     = expert_ends[e] - start_e
            e_tids  = jax.lax.dynamic_slice(sorted_tids, (start_e,), (max_tpe,))
            e_ws    = jax.lax.dynamic_slice(sorted_ws,   (start_e,), (max_tpe,))
            valid_e = jnp.arange(max_tpe) < n_e
            tids_list.append(jnp.where(valid_e, e_tids, 0))
            valid_list.append(valid_e)
            ws_list.append(e_ws)

        all_tids = jnp.concatenate(tids_list)   # (E_local * max_tpe,)

        out_parts = []
        for e in range(E_local):
            with jax.named_scope(f"expert_{e}"):
                sel_x  = sc_gather_rows(x_b, tids_list[e])   # (max_tpe, D)
                gate_e = jax.nn.silu(sel_x @ wi_0[e])
                up_e   = sel_x @ wi_1[e]
                out_e  = (gate_e * up_e) @ wo[e]
                out_parts.append(
                    (out_e * ws_list[e][:, None] * valid_list[e][:, None]).astype(x_b.dtype))

        stacked = jnp.concatenate(out_parts, axis=0)
        return jnp.zeros((T_b, D), dtype=x_b.dtype).at[all_tids].add(stacked)

    # Chunk T_fsdp so each block has ≤ SC_MAX_ROWS rows.
    partial_out = _compute_block(flat_x, flat_indices, flat_weights)

    # EP psum: sum contributions from all EP devices (each handles its experts).
    return jax.lax.psum(partial_out, ep_axis)


@functools.partial(jax.custom_vjp, nondiff_argnums=(6, 7, 8, 9, 10))
def _moe_jax_ep(fx, fi, fw, w0, w1, wout,
                mesh, K: int, act_spec, ep_axis: str, max_tpe: int):
    """Module-level custom_vjp for JAX EP MoE.

    Defined at module level (not inside expert_mlp_jax) so lax.scan over
    L_moe layers reuses a single compiled trace — O(1) compilation.

    The custom_vjp barrier prevents scan from accumulating the full
    (L_moe, B/fsdp, S, D) activation carry in the backward pass, which
    would overflow INT32 indices at EP=8, L=58.
    """
    from jax.experimental.shard_map import shard_map

    @functools.partial(shard_map, mesh=mesh,
                       in_specs=(act_spec, act_spec, act_spec,
                                 _moe_wi_spec(),
                                 _moe_wi_spec(),
                                 _moe_wo_spec()),
                       out_specs=act_spec, check_rep=False)
    def _fn(fx_, fi_, fw_, w0_, w1_, wout_):
        return _expert_mlp_ep_body(fx_, w0_, w1_, wout_, fi_, fw_,
                                    K, ep_axis, max_tpe)

    return _fn(fx, fi, fw, w0, w1, wout)


def _moe_jax_ep_fwd(fx, fi, fw, w0, w1, wout,
                    mesh, K, act_spec, ep_axis, max_tpe):
    out = _moe_jax_ep(fx, fi, fw, w0, w1, wout, mesh, K, act_spec, ep_axis, max_tpe)
    return out, (fx, fi, fw, w0, w1, wout)


def _moe_jax_ep_bwd(mesh, K, act_spec, ep_axis, max_tpe, res, g):
    """Backward via jax.vjp. Safe here — custom_vjp barrier means we are NOT
    inside lax.scan's unrolled backward, so no carry size explosion."""
    from jax.experimental.shard_map import shard_map

    fx, fi, fw, w0, w1, wout = res

    def _fwd(fx_, fw_, w0_, w1_, wout_):
        @functools.partial(shard_map, mesh=mesh,
                           in_specs=(act_spec, act_spec, act_spec,
                                     _moe_wi_spec(),
                                     _moe_wi_spec(),
                                     _moe_wo_spec()),
                           out_specs=act_spec, check_rep=False)
        def _fn(fx__, fi__, fw__, w0__, w1__, wout__):
            return _expert_mlp_ep_body(fx__, w0__, w1__, wout__, fi__, fw__,
                                        K, ep_axis, max_tpe)
        return _fn(fx_, fi, fw_, w0_, w1_, wout_)

    _, vjp_fn = jax.vjp(_fwd, fx, fw, w0, w1, wout)
    d_fx, d_fw, d_w0, d_w1, d_wout = vjp_fn(g)
    return (d_fx, jnp.zeros_like(fi), d_fw, d_w0, d_w1, d_wout)


_moe_jax_ep.defvjp(_moe_jax_ep_fwd, _moe_jax_ep_bwd)


def expert_mlp_jax(x, wi_0, wi_1, wo, top_k_weights, top_k_indices, cfg: ModelConfig):
    """JAX EP MoE: shard_map over ep+fsdp, jax.vjp backward. No token dropping."""
    B, S, D = x.shape
    K = cfg.K
    flat_x       = x.reshape(B * S, D)
    flat_indices = top_k_indices.reshape(B * S, K)
    flat_weights = top_k_weights.reshape(B * S, K)

    ep_size   = 1 if cfg.mesh is None else cfg.mesh.shape.get("ep", 1)
    fsdp_size = 1 if cfg.mesh is None else cfg.mesh.shape.get("fsdp", 1)

    if ep_size == 1:
        # EP=1: simple einsum over all E experts.
        E = cfg.E
        dispatch = jnp.zeros((B * S, E), dtype=x.dtype)
        token_idx = jnp.arange(B * S)[:, None]
        dispatch = dispatch.at[token_idx, flat_indices].add(flat_weights)
        gate_all = jax.nn.silu(jnp.einsum("td,edm->tem", flat_x, wi_0))
        up_all   = jnp.einsum("td,edm->tem", flat_x, wi_1)
        out_all  = jnp.einsum("tem,emd->ted", gate_all * up_all, wo)
        return jnp.einsum("ted,te->td", out_all, dispatch).reshape(B, S, D)

    T_fsdp  = B * S // fsdp_size
    # max_tpe: 2× expected avg tokens per expert per device.
    # Kept for static buffer sizing inside _expert_mlp_ep_body.
    max_tpe = max(1, 2 * T_fsdp * K // cfg.E)
    act_spec = P("fsdp", None)

    with jax.named_scope("moe_ep_shardmap"):
        out = _moe_jax_ep(flat_x, flat_indices, flat_weights,
                          wi_0, wi_1, wo,
                          cfg.mesh, K, act_spec, "ep", max_tpe)
    return out.reshape(B, S, D)


# ============================================================================
# MoE GMM backend — megablox Mosaic-native grouped matmul
# ============================================================================

def _expert_mlp_gmm_body(flat_x, wi_0, wi_1, wo, flat_indices, flat_weights,
                          K: int, ep_axis: str, fsdp_axis: str):
    """MoE body using megablox GMM kernel. Called inside shard_map over (ep, fsdp).

    Weight sharding: all-gather FSDP slices → full F. EP routing: each device
    processes only its local experts' tokens (≈ T*K/EP rows, not T*K) so the
    SC BF16 gather limit (16 GiB) is never exceeded.

    flat_x:       (T_fsdp, D)
    wi_0, wi_1:   (E_local, D, F_local)   F_local = D_moe / fsdp
    wo:           (E_local, F_local, D)
    flat_indices: (T_fsdp, K)             global expert IDs  0..E_global-1
    flat_weights: (T_fsdp, K)             routing weights (float32)
    """
    from jax.experimental.pallas.ops.tpu.megablox import gmm as megablox_gmm

    # All-gather FSDP weight slices → full weight tensors.
    # wi_0/wi_1: (E_local, D, F_local) → (E_local, D, F_full)
    # wo:        (E_local, F_local, D) → (E_local, F_full, D)
    with jax.named_scope("weight_allgather"):
        wi_0_f = jax.lax.all_gather(wi_0, fsdp_axis, axis=_moe_wi_ag_axis(), tiled=True)
        wi_1_f = jax.lax.all_gather(wi_1, fsdp_axis, axis=_moe_wi_ag_axis(), tiled=True)
        wo_f   = jax.lax.all_gather(wo, fsdp_axis, axis=_moe_wo_ag_axis(), tiled=True)

    T, D = flat_x.shape
    E_local     = wi_0_f.shape[0]
    ep_size     = jax.lax.axis_size(ep_axis)   # static Python int inside shard_map
    my_ep_rank  = jax.lax.axis_index(ep_axis)
    local_start = my_ep_rank * E_local

    # Sort INTEGER arrays only — never build a (T*K, D) sorted_x (SC 16 GiB limit).
    TK        = T * K
    exp_ids   = flat_indices.reshape(-1).astype(jnp.int32)   # (T*K,) global 0..E_global-1
    exp_ws    = flat_weights.reshape(-1)                       # (T*K,)
    token_ids = jnp.repeat(jnp.arange(T, dtype=jnp.int32), K)

    with jax.named_scope("sort_by_expert"):
        argsorted        = jnp.argsort(exp_ids, stable=True)
        sorted_token_ids = token_ids[argsorted]   # (T*K,) int32
        sorted_exp_ids   = exp_ids[argsorted]     # (T*K,) int32 — sorted
        sorted_weights   = exp_ws[argsorted]      # (T*K,)

    # Locate the contiguous local-expert block in sorted_exp_ids.
    blk_start = jnp.searchsorted(sorted_exp_ids, local_start)
    blk_end   = jnp.searchsorted(sorted_exp_ids, local_start + E_local)
    n_local   = blk_end - blk_start   # dynamic; ≈ T*K // ep_size with balanced routing

    # Static upper bound on local token count (assumes ~balanced expert load).
    # max_local × D × 2 bytes = (T*K/EP) × 7168 × 2 ≈ 7.5 GB < SC 16 GiB limit ✓
    max_local = TK // ep_size

    # Extract local block token IDs, weights, and expert IDs (static shape max_local).
    with jax.named_scope("local_block_extract"):
        local_tids_raw  = jax.lax.dynamic_slice(sorted_token_ids, [blk_start], [max_local])
        local_ws_raw    = jax.lax.dynamic_slice(sorted_weights,   [blk_start], [max_local])
        local_eids_raw  = jax.lax.dynamic_slice(sorted_exp_ids,   [blk_start], [max_local])

    # Map global expert IDs → relative [0, E_local); excess padding → E_local (dummy).
    # valid[:n_local]=True; valid[n_local:]=False → padded entries routed to dummy expert.
    # The resulting local_exp_rel is sorted: [0..E_local-1] then E_local (padding).
    valid         = jnp.arange(max_local) < n_local
    local_exp_rel = jnp.where(valid, local_eids_raw - local_start, E_local).astype(jnp.int32)
    local_ws      = jnp.where(valid, local_ws_raw, jnp.zeros_like(local_ws_raw))

    # Gather ONLY local tokens from flat_x: (max_local, D) — SC-safe.
    # flat_x has T_fsdp rows (≤ 14 GiB); result is max_local rows (≤ 7 GiB). ✓
    local_x = flat_x[local_tids_raw]                                           # (max_local, D)

    # group_sizes: E_local real + 1 dummy absorbing padding, sums to max_local.
    with jax.named_scope("group_sizes"):
        ends        = jnp.searchsorted(local_exp_rel, jnp.arange(1, E_local + 2))
        starts      = jnp.searchsorted(local_exp_rel, jnp.arange(E_local + 1))
        group_sizes = (ends - starts).astype(jnp.int32)

    # Extend weights with zero dummy expert row.
    zero_wi  = jnp.zeros((1, D, wi_0_f.shape[2]), dtype=wi_0_f.dtype)
    zero_wo  = jnp.zeros((1, wo_f.shape[1], D),   dtype=wo_f.dtype)
    wi_0_ext = jnp.concatenate([wi_0_f, zero_wi], axis=0)
    wi_1_ext = jnp.concatenate([wi_1_f, zero_wi], axis=0)
    wo_ext   = jnp.concatenate([wo_f,   zero_wo], axis=0)

    # GMM: (max_local, D) × (E_local+1, D, F) → (max_local, F), tiling (512, 1024, 1024)
    _gmm = functools.partial(megablox_gmm,
                             preferred_element_type=flat_x.dtype,
                             tiling=(512, 1024, 1024))
    with jax.named_scope("gmm_gate_up"):
        gate     = jax.nn.silu(_gmm(local_x, wi_0_ext, group_sizes))
        up       = _gmm(local_x, wi_1_ext, group_sizes)
    hidden = gate * up

    with jax.named_scope("gmm_down"):
        out_local = _gmm(hidden, wo_ext, group_sizes)

    out_local = out_local * local_ws[:, None].astype(out_local.dtype)

    # Scatter-add back. Padding entries have zeroed weights → contribute 0.
    partial_out = jnp.zeros((T, D), dtype=flat_x.dtype).at[local_tids_raw].add(
        out_local.astype(flat_x.dtype))

    # EP psum: aggregate contributions from all EP devices (each handles its experts).
    return jax.lax.psum(partial_out, ep_axis)


def _expert_mlp_rd_body_for_bwd(flat_x, wi_0, wi_1, wo, flat_indices, flat_weights,
                                  K: int, ep_axis: str, fsdp_axis: str):
    """Differentiable MoE body for use inside jax.vjp in the backward pass.

    Same lazy local-expert token gather as _expert_mlp_gmm_body, but uses
    jax.lax.ragged_dot (which has a registered VJP) instead of megablox_gmm
    (which is @jax.jit-wrapped with no VJP → NotImplementedError).

    Forward pass uses megablox_gmm for performance; this function is only
    traced by jax.vjp in _moe_jax_ep_rd_bwd.
    """
    # All-gather FSDP weight slices → full weight tensors.
    with jax.named_scope("weight_allgather"):
        wi_0_f = jax.lax.all_gather(wi_0, fsdp_axis, axis=_moe_wi_ag_axis(), tiled=True)
        wi_1_f = jax.lax.all_gather(wi_1, fsdp_axis, axis=_moe_wi_ag_axis(), tiled=True)
        wo_f   = jax.lax.all_gather(wo, fsdp_axis, axis=_moe_wo_ag_axis(), tiled=True)

    T, D = flat_x.shape
    E_local     = wi_0_f.shape[0]
    ep_size     = jax.lax.axis_size(ep_axis)
    my_ep_rank  = jax.lax.axis_index(ep_axis)
    local_start = my_ep_rank * E_local

    TK        = T * K
    exp_ids   = flat_indices.reshape(-1).astype(jnp.int32)
    exp_ws    = flat_weights.reshape(-1)
    token_ids = jnp.repeat(jnp.arange(T, dtype=jnp.int32), K)

    with jax.named_scope("sort_by_expert"):
        argsorted        = jnp.argsort(exp_ids, stable=True)
        sorted_token_ids = token_ids[argsorted]
        sorted_exp_ids   = exp_ids[argsorted]
        sorted_weights   = exp_ws[argsorted]

    blk_start = jnp.searchsorted(sorted_exp_ids, local_start)
    blk_end   = jnp.searchsorted(sorted_exp_ids, local_start + E_local)
    n_local   = blk_end - blk_start
    max_local = TK // ep_size  # static Python int

    with jax.named_scope("local_block_extract"):
        local_tids_raw  = jax.lax.dynamic_slice(sorted_token_ids, [blk_start], [max_local])
        local_ws_raw    = jax.lax.dynamic_slice(sorted_weights,   [blk_start], [max_local])
        local_eids_raw  = jax.lax.dynamic_slice(sorted_exp_ids,   [blk_start], [max_local])

    valid         = jnp.arange(max_local) < n_local
    local_exp_rel = jnp.where(valid, local_eids_raw - local_start, E_local).astype(jnp.int32)
    local_ws      = jnp.where(valid, local_ws_raw, jnp.zeros_like(local_ws_raw))

    local_x = flat_x[local_tids_raw]  # (max_local, D)

    with jax.named_scope("group_sizes"):
        ends        = jnp.searchsorted(local_exp_rel, jnp.arange(1, E_local + 2))
        starts      = jnp.searchsorted(local_exp_rel, jnp.arange(E_local + 1))
        group_sizes = (ends - starts).astype(jnp.int32)

    zero_wi  = jnp.zeros((1, D, wi_0_f.shape[2]), dtype=wi_0_f.dtype)
    zero_wo  = jnp.zeros((1, wo_f.shape[1], D),   dtype=wo_f.dtype)
    wi_0_ext = jnp.concatenate([wi_0_f, zero_wi], axis=0)
    wi_1_ext = jnp.concatenate([wi_1_f, zero_wi], axis=0)
    wo_ext   = jnp.concatenate([wo_f,   zero_wo], axis=0)

    # ragged_dot: lhs=(max_local, k), rhs=(E_local+1, k, n), group_sizes=(E_local+1,)
    # Fully differentiable — JAX has a registered VJP for ragged_dot.
    with jax.named_scope("rd_gate_up"):
        gate     = jax.nn.silu(jax.lax.ragged_dot(
            local_x.astype(wi_0_ext.dtype), wi_0_ext, group_sizes))
        up       = jax.lax.ragged_dot(
            local_x.astype(wi_1_ext.dtype), wi_1_ext, group_sizes)
    hidden = gate * up

    with jax.named_scope("rd_down"):
        out_local = jax.lax.ragged_dot(
            hidden.astype(wo_ext.dtype), wo_ext, group_sizes)

    out_local = out_local * local_ws[:, None].astype(out_local.dtype)

    partial_out = jnp.zeros((T, D), dtype=flat_x.dtype).at[local_tids_raw].add(
        out_local.astype(flat_x.dtype))

    return jax.lax.psum(partial_out, ep_axis)


@functools.partial(jax.custom_vjp, nondiff_argnums=(6, 7, 8, 9, 10))
def _moe_jax_ep_rd(fx, fi, fw, w0, w1, wout,
                   mesh, K: int, act_spec, ep_axis: str, max_tpe: int):
    """GMM EP MoE — module-level custom_vjp prevents scan carry explosion."""
    from jax.experimental.shard_map import shard_map

    @functools.partial(shard_map, mesh=mesh,
                       in_specs=(act_spec, act_spec, act_spec,
                                 _moe_wi_spec(),
                                 _moe_wi_spec(),
                                 _moe_wo_spec()),
                       out_specs=act_spec, check_rep=False)
    def _fn(fx_, fi_, fw_, w0_, w1_, wout_):
        return _expert_mlp_gmm_body(fx_, w0_, w1_, wout_, fi_, fw_,
                                    K, ep_axis, "fsdp")

    return _fn(fx, fi, fw, w0, w1, wout)


def _moe_jax_ep_rd_fwd(fx, fi, fw, w0, w1, wout,
                        mesh, K, act_spec, ep_axis, max_tpe):
    out = _moe_jax_ep_rd(fx, fi, fw, w0, w1, wout, mesh, K, act_spec, ep_axis, max_tpe)
    return out, (fx, fi, fw, w0, w1, wout)


def _moe_jax_ep_rd_bwd(mesh, K, act_spec, ep_axis, max_tpe, res, g):
    """Backward via jax.vjp — safe here (custom_vjp barrier, not inside scan carry)."""
    from jax.experimental.shard_map import shard_map

    fx, fi, fw, w0, w1, wout = res

    def _fwd(fx_, fw_, w0_, w1_, wout_):
        @functools.partial(shard_map, mesh=mesh,
                           in_specs=(act_spec, act_spec, act_spec,
                                     _moe_wi_spec(),
                                     _moe_wi_spec(),
                                     _moe_wo_spec()),
                           out_specs=act_spec, check_rep=False)
        def _fn(fx__, fi__, fw__, w0__, w1__, wout__):
            # Use ragged_dot (differentiable) not megablox_gmm (@jax.jit, no VJP).
            return _expert_mlp_rd_body_for_bwd(fx__, w0__, w1__, wout__, fi__, fw__,
                                               K, ep_axis, "fsdp")
        return _fn(fx_, fi, fw_, w0_, w1_, wout_)

    _, vjp_fn = jax.vjp(_fwd, fx, fw, w0, w1, wout)
    d_fx, d_fw, d_w0, d_w1, d_wout = vjp_fn(g)
    return (d_fx, jnp.zeros_like(fi), d_fw, d_w0, d_w1, d_wout)


_moe_jax_ep_rd.defvjp(_moe_jax_ep_rd_fwd, _moe_jax_ep_rd_bwd)


# ============================================================================
# AllReduce-GMM backend — no weight AllGather; partial F compute + psum
# ============================================================================

def _expert_mlp_gmm_colrow_body(flat_x, wi_0, wi_1, wo, flat_indices, flat_weights,
                                 K: int, ep_axis: str, fsdp_axis: str):
    """Col/row parallel MoE body — v251/v264 pattern (1740 TPS/chip at TP=2).

    Used when tp>1 + ep==1. NO weight AllGather. ONE psum on fsdp at end.
    See feedback_dsv3_tp2_colrow_pattern memory for context.

    Per-device input shapes (after shard_map view, with current init storage):
      flat_x: (T, D)             D is tp-sharded inside shard_map
      wi_0/wi_1/wo: (E_local, F_local, D_local)  F=fsdp-sharded, D=tp-sharded

    Pattern:
      Column-parallel gate/up: ragged_dot produces (max_local, F_local) directly
      Row-parallel down:       ragged_dot produces partial-sum on D, then psum on fsdp
      D stays tp-sharded throughout (no psum on tp needed).
    """
    T, D = flat_x.shape
    E_local = wi_0.shape[0]
    F_local = wi_0.shape[1]

    ep_size    = jax.lax.axis_size(ep_axis)
    my_ep_rank = jax.lax.axis_index(ep_axis)
    local_start = my_ep_rank * E_local

    TK        = T * K
    exp_ids   = flat_indices.reshape(-1).astype(jnp.int32)
    exp_ws    = flat_weights.reshape(-1)
    token_ids = jnp.repeat(jnp.arange(T, dtype=jnp.int32), K)

    with jax.named_scope("sort_by_expert"):
        argsorted        = jnp.argsort(exp_ids, stable=True)
        sorted_token_ids = token_ids[argsorted]
        sorted_exp_ids   = exp_ids[argsorted]
        sorted_weights   = exp_ws[argsorted]

    blk_start = jnp.searchsorted(sorted_exp_ids, local_start)
    blk_end   = jnp.searchsorted(sorted_exp_ids, local_start + E_local)
    n_local   = blk_end - blk_start
    max_local = TK // ep_size

    with jax.named_scope("local_block_extract"):
        local_tids_raw  = jax.lax.dynamic_slice(sorted_token_ids, [blk_start], [max_local])
        local_ws_raw    = jax.lax.dynamic_slice(sorted_weights,   [blk_start], [max_local])
        local_eids_raw  = jax.lax.dynamic_slice(sorted_exp_ids,   [blk_start], [max_local])

    valid         = jnp.arange(max_local) < n_local
    local_exp_rel = jnp.where(valid, local_eids_raw - local_start, E_local).astype(jnp.int32)
    local_ws      = jnp.where(valid, local_ws_raw, jnp.zeros_like(local_ws_raw))

    local_x = flat_x[local_tids_raw]  # (max_local, D)

    with jax.named_scope("group_sizes"):
        ends        = jnp.searchsorted(local_exp_rel, jnp.arange(1, E_local + 2))
        starts      = jnp.searchsorted(local_exp_rel, jnp.arange(E_local + 1))
        group_sizes = (ends - starts).astype(jnp.int32)

    # Transpose wi from (E, F, D) → (E, D, F) so D is contraction dim for ragged_dot.
    wi_0_t = wi_0.transpose(0, 2, 1)  # (E_local, D, F_local)
    wi_1_t = wi_1.transpose(0, 2, 1)
    # wo stays (E, F, D) — F is contraction dim for the down matmul.

    # Add dummy expert row for over-padded tokens (their weight is masked to 0).
    zero_wi = jnp.zeros((1, D, F_local), dtype=wi_0.dtype)
    zero_wo = jnp.zeros((1, F_local, D), dtype=wo.dtype)
    wi_0_ext = jnp.concatenate([wi_0_t, zero_wi], axis=0)  # (E_local+1, D, F_local)
    wi_1_ext = jnp.concatenate([wi_1_t, zero_wi], axis=0)
    wo_ext   = jnp.concatenate([wo,     zero_wo], axis=0)  # (E_local+1, F_local, D)

    # Gate/up matmul: x is D-tp-sharded × wi is D-tp-sharded → partial sum on tp.
    # MUST psum("tp") to complete D contraction BEFORE silu (silu is non-linear).
    # F output is fsdp-sharded (each fsdp rank has different F_local columns).
    with jax.named_scope("colrow_gate_up"):
        gate_partial = jax.lax.ragged_dot(
            local_x.astype(wi_0_ext.dtype), wi_0_ext, group_sizes)  # (max_local, F_local)
        up_partial = jax.lax.ragged_dot(
            local_x.astype(wi_1_ext.dtype), wi_1_ext, group_sizes)
        # Complete D contraction across TP shards.
        gate_local = jax.nn.silu(jax.lax.psum(gate_partial, "tp"))
        up_local   = jax.lax.psum(up_partial, "tp")
    hidden_local = gate_local * up_local  # (max_local, F_local) — full D contracted

    # Down matmul: hidden is fsdp-sharded F × wo is (F=fsdp,D=tp)-sharded.
    # Output (M, D_local) is partial sum on F. psum("fsdp") completes F → full F.
    # D stays tp-sharded throughout (no extra psum on tp needed).
    with jax.named_scope("colrow_down"):
        out_partial = jax.lax.ragged_dot(
            hidden_local.astype(wo_ext.dtype), wo_ext, group_sizes)  # (max_local, D_local)
        out_local = jax.lax.psum(out_partial, fsdp_axis)  # AR on fsdp → full F sum

    out_local = out_local * local_ws[:, None].astype(out_local.dtype)

    # Scatter back to per-token rows; EP psum is no-op at EP=1.
    partial_out = jnp.zeros((T, D), dtype=flat_x.dtype).at[local_tids_raw].add(
        out_local.astype(flat_x.dtype))
    out = jax.lax.psum(partial_out, ep_axis)
    # AllGather D across TP so output is D-replicated, matching carry_spec expectation
    # (carry has D=None at ep=1+tp>1; MoE output goes into residual `x + mlp_out`).
    out = jax.lax.all_gather(out, "tp", axis=1, tiled=True)
    return out


def _expert_mlp_gmm_ar_body(flat_x, wi_0, wi_1, wo, flat_indices, flat_weights,
                              K: int, ep_axis: str, fsdp_axis: str):
    """A2A MoE body: sort tokens by dest EP, all_to_all dispatch, compute, return.

    Communication:
      A2A dispatch:    T_local × K/EP × D per device (targeted, not broadcast)
      FSDP weight AG:  3 × E_local × D_moe × D (fixed, overlappable)
      A2A return:      same as dispatch

    vs AllGather: A2A sends ~K/EP fraction per token, AllGather sends all tokens.
    With token sharding on EP: A2A volume is 1/EP of non-sharded AllGather.

    wi_0, wi_1: (E_local, F_local, D_local)  stored transposed for T(8,128) layout
    wo:         (E_local, F_local, D_local)
    """
    T, D = flat_x.shape
    E_local    = wi_0.shape[0]
    ep_size    = jax.lax.axis_size(ep_axis)
    my_ep_rank = jax.lax.axis_index(ep_axis)
    local_start = my_ep_rank * E_local

    # Capacity per EP device: T*K/EP (uniform routing assumption).
    TK = T * K
    capacity = TK // ep_size  # tokens-expert pairs per EP device

    # --- Step 1: Sort token-expert pairs by destination EP device ---
    with jax.named_scope("a2a_prepare"):
        pair_tids = jnp.repeat(jnp.arange(T, dtype=jnp.int32), K)   # (TK,)
        pair_eids = flat_indices.reshape(-1).astype(jnp.int32)        # (TK,)
        pair_ws   = flat_weights.reshape(-1)                           # (TK,)
        pair_dests = pair_eids // E_local                              # (TK,) dest EP device

        # Sort by destination EP device
        sort_idx = jnp.argsort(pair_dests, stable=True)
        sorted_tids  = pair_tids[sort_idx]
        sorted_eids  = pair_eids[sort_idx]
        sorted_ws    = pair_ws[sort_idx]

        # Extract per-EP-device buckets (each of size 'capacity')
        bucket_starts = jnp.searchsorted(pair_dests[sort_idx],
                                          jnp.arange(ep_size))

        # Build dispatch buffers: (ep_size, capacity) for IDs/weights
        # and (ep_size, capacity, D) for token embeddings
        def _extract_bucket(start):
            tids = jax.lax.dynamic_slice(sorted_tids, [start], [capacity])
            eids = jax.lax.dynamic_slice(sorted_eids, [start], [capacity])
            ws   = jax.lax.dynamic_slice(sorted_ws,   [start], [capacity])
            return tids, eids, ws

        all_tids = jax.vmap(_extract_bucket)(bucket_starts)   # (ep, capacity)
        all_eids = all_tids[1]
        all_ws   = all_tids[2]
        all_tids = all_tids[0]
        dispatch_x = flat_x[all_tids.reshape(-1)].reshape(ep_size, capacity, D)

    # --- Step 2: all_to_all token dispatch ---
    with jax.named_scope("a2a_dispatch"):
        recv_x    = jax.lax.all_to_all(dispatch_x, ep_axis,
                                        split_axis=0, concat_axis=0, tiled=True)
        recv_eids = jax.lax.all_to_all(all_eids, ep_axis,
                                        split_axis=0, concat_axis=0, tiled=True)
        recv_ws   = jax.lax.all_to_all(all_ws, ep_axis,
                                        split_axis=0, concat_axis=0, tiled=True)
    # recv_x: (ep_size, capacity, D) — tokens from all EP devices for my experts

    # --- Step 3: Sort received tokens by local expert ---
    with jax.named_scope("local_sort"):
        recv_x_flat    = recv_x.reshape(-1, D)          # (ep*capacity, D)
        recv_eids_flat = recv_eids.reshape(-1)           # (ep*capacity,)
        recv_ws_flat   = recv_ws.reshape(-1)             # (ep*capacity,)
        max_local      = ep_size * capacity              # total received pairs

        local_eids = recv_eids_flat - local_start
        # Mark out-of-range as dummy expert
        valid = (local_eids >= 0) & (local_eids < E_local)
        local_eids = jnp.where(valid, local_eids, E_local).astype(jnp.int32)
        local_ws_masked = jnp.where(valid, recv_ws_flat, 0.0)

        sort_local = jnp.argsort(local_eids, stable=True)
        sorted_recv_x  = recv_x_flat[sort_local]
        sorted_local_e = local_eids[sort_local]
        sorted_local_w = local_ws_masked[sort_local]

    with jax.named_scope("group_sizes"):
        ends   = jnp.searchsorted(sorted_local_e, jnp.arange(1, E_local + 2))
        starts = jnp.searchsorted(sorted_local_e, jnp.arange(E_local + 1))
        group_sizes = (ends - starts).astype(jnp.int32)

    # --- Step 4: AllGather weights on FSDP → full D_moe ---
    with jax.named_scope("weight_allgather_f"):
        wi_0_f = jax.lax.all_gather(wi_0, fsdp_axis, axis=_moe_wi_ag_axis(), tiled=True)
        wi_1_f = jax.lax.all_gather(wi_1, fsdp_axis, axis=_moe_wi_ag_axis(), tiled=True)
        wo_f   = jax.lax.all_gather(wo, fsdp_axis, axis=_moe_wo_ag_axis(), tiled=True)

    F_full = wi_0_f.shape[1]
    wi_0_t = wi_0_f.transpose(0, 2, 1)
    wi_1_t = wi_1_f.transpose(0, 2, 1)

    zero_wi = jnp.zeros((1, D, F_full), dtype=wi_0_t.dtype)
    zero_wo = jnp.zeros((1, F_full, D), dtype=wo_f.dtype)
    wi_0_ext = jnp.concatenate([wi_0_t, zero_wi], axis=0)
    wi_1_ext = jnp.concatenate([wi_1_t, zero_wi], axis=0)
    wo_ext   = jnp.concatenate([wo_f, zero_wo], axis=0)

    # --- Step 5: Compute with gathered weights ---
    with jax.named_scope("a2a_gate_up"):
        gate = jax.nn.silu(jax.lax.ragged_dot(
            sorted_recv_x.astype(wi_0_ext.dtype), wi_0_ext, group_sizes))
        up = jax.lax.ragged_dot(
            sorted_recv_x.astype(wi_1_ext.dtype), wi_1_ext, group_sizes)
    hidden = gate * up

    with jax.named_scope("a2a_down"):
        out_sorted = jax.lax.ragged_dot(
            hidden.astype(wo_ext.dtype), wo_ext, group_sizes)

    out_sorted = out_sorted * sorted_local_w[:, None].astype(out_sorted.dtype)

    # Un-sort back to recv order
    out_recv = jnp.zeros_like(recv_x_flat).at[sort_local].set(
        out_sorted.astype(recv_x_flat.dtype))
    out_recv = out_recv.reshape(ep_size, capacity, D)

    # --- Step 6: Reverse all_to_all — send results back ---
    with jax.named_scope("a2a_return"):
        result_buckets = jax.lax.all_to_all(out_recv, ep_axis,
                                             split_axis=0, concat_axis=0, tiled=True)
    # result_buckets: (ep_size, capacity, D) — my dispatched tokens returned

    # Scatter-add back to original token positions
    result = jnp.zeros((T, D), dtype=flat_x.dtype)
    result = result.at[all_tids.reshape(-1)].add(
        result_buckets.reshape(-1, D).astype(flat_x.dtype))

    return result  # (T, D) — complete MoE output for this device's tokens


def _expert_mlp_gmm_colrow_a2a_body(flat_x, wi_0, wi_1, wo, flat_indices, flat_weights,
                                      K: int, ep_axis: str, fsdp_axis: str):
    """A2A-dispatch + colrow TP body — no weight AllGather.

    Mesh assumption: EP>1 + TP>1 (e.g. EP=8, TP=2). Combines:
      • A2A token dispatch on EP (from _expert_mlp_gmm_ar_body)
      • Column/row TP pattern on D_model (from _expert_mlp_gmm_colrow_body):
        - x is D-tp-sharded
        - wi/wo stored as (E_local, F_local, D_local) where F=fsdp-sharded, D=tp-sharded
        - gate/up: ragged_dot gives (M, F_local) partial-sum on D → psum("tp")
        - down:    ragged_dot gives (M, D_local) partial-sum on F → psum(fsdp)

    Removes the per-layer 7 GB weight AG that gmm_ar's body pays. Activation
    transients per layer drop from ~10 GB (with full F-AG'd weights) to
    ~5 GB (F stays sharded). Total resident weight in HBM is ~2.5 GB across
    all 58 layers — comfortably fits without host streaming.
    """
    T, D = flat_x.shape   # D here is D_model/TP (TP-sharded)
    E_local    = wi_0.shape[0]
    F_local    = wi_0.shape[1]   # F=fsdp-sharded
    ep_size    = jax.lax.axis_size(ep_axis)
    my_ep_rank = jax.lax.axis_index(ep_axis)
    local_start = my_ep_rank * E_local

    # Capacity per EP device: T*K/EP.
    TK = T * K
    capacity = TK // ep_size

    # --- Step 1: Sort token-expert pairs by destination EP device ---
    with jax.named_scope("a2a_prepare"):
        pair_tids = jnp.repeat(jnp.arange(T, dtype=jnp.int32), K)
        pair_eids = flat_indices.reshape(-1).astype(jnp.int32)
        pair_ws   = flat_weights.reshape(-1)
        pair_dests = pair_eids // E_local

        sort_idx = jnp.argsort(pair_dests, stable=True)
        sorted_tids  = pair_tids[sort_idx]
        sorted_eids  = pair_eids[sort_idx]
        sorted_ws    = pair_ws[sort_idx]

        bucket_starts = jnp.searchsorted(pair_dests[sort_idx],
                                          jnp.arange(ep_size))

        def _extract_bucket(start):
            tids = jax.lax.dynamic_slice(sorted_tids, [start], [capacity])
            eids = jax.lax.dynamic_slice(sorted_eids, [start], [capacity])
            ws   = jax.lax.dynamic_slice(sorted_ws,   [start], [capacity])
            return tids, eids, ws

        all_tids = jax.vmap(_extract_bucket)(bucket_starts)
        all_eids = all_tids[1]
        all_ws   = all_tids[2]
        all_tids = all_tids[0]
        # x has shape (T, D/TP) — dispatch the tp-sharded D too.
        dispatch_x = flat_x[all_tids.reshape(-1)].reshape(ep_size, capacity, D)

    # --- Step 2: all_to_all token dispatch on EP ---
    with jax.named_scope("a2a_dispatch"):
        recv_x    = jax.lax.all_to_all(dispatch_x, ep_axis,
                                        split_axis=0, concat_axis=0, tiled=True)
        recv_eids = jax.lax.all_to_all(all_eids, ep_axis,
                                        split_axis=0, concat_axis=0, tiled=True)
        recv_ws   = jax.lax.all_to_all(all_ws, ep_axis,
                                        split_axis=0, concat_axis=0, tiled=True)

    # --- Step 3: Sort received tokens by local expert ---
    with jax.named_scope("local_sort"):
        recv_x_flat    = recv_x.reshape(-1, D)
        recv_eids_flat = recv_eids.reshape(-1)
        recv_ws_flat   = recv_ws.reshape(-1)
        max_local      = ep_size * capacity

        local_eids = recv_eids_flat - local_start
        valid = (local_eids >= 0) & (local_eids < E_local)
        local_eids = jnp.where(valid, local_eids, E_local).astype(jnp.int32)
        local_ws_masked = jnp.where(valid, recv_ws_flat, 0.0)

        sort_local = jnp.argsort(local_eids, stable=True)
        sorted_recv_x  = recv_x_flat[sort_local]
        sorted_local_e = local_eids[sort_local]
        sorted_local_w = local_ws_masked[sort_local]

    with jax.named_scope("group_sizes"):
        ends   = jnp.searchsorted(sorted_local_e, jnp.arange(1, E_local + 2))
        starts = jnp.searchsorted(sorted_local_e, jnp.arange(E_local + 1))
        group_sizes = (ends - starts).astype(jnp.int32)

    # NO weight AG. Use weights at their sharded shape (E_local, F_local, D/TP).
    # ragged_dot needs (E, contract_dim, output_dim). Transpose wi (F,D)→(D,F)
    # so the matmul contracts D (which matches sorted_recv_x's last dim).
    wi_0_t = wi_0.transpose(0, 2, 1)   # (E_local, D/TP, F_local)
    wi_1_t = wi_1.transpose(0, 2, 1)
    # wo stays (E_local, F_local, D/TP) — F is the contract dim.

    zero_wi = jnp.zeros((1, D, F_local), dtype=wi_0_t.dtype)
    zero_wo = jnp.zeros((1, F_local, D), dtype=wo.dtype)
    wi_0_ext = jnp.concatenate([wi_0_t, zero_wi], axis=0)
    wi_1_ext = jnp.concatenate([wi_1_t, zero_wi], axis=0)
    wo_ext   = jnp.concatenate([wo,     zero_wo], axis=0)

    # --- Step 4: Gate/up matmul (column-parallel) ---
    # x is D-tp-sharded × wi is D-tp-sharded → partial sum on D → psum("tp").
    # Output (M, F_local): F is fsdp-sharded.
    with jax.named_scope("colrow_a2a_gate_up"):
        gate_partial = jax.lax.ragged_dot(
            sorted_recv_x.astype(wi_0_ext.dtype), wi_0_ext, group_sizes)
        up_partial = jax.lax.ragged_dot(
            sorted_recv_x.astype(wi_1_ext.dtype), wi_1_ext, group_sizes)
        gate = jax.nn.silu(jax.lax.psum(gate_partial, "tp"))
        up   = jax.lax.psum(up_partial, "tp")
    hidden = gate * up   # (M, F_local) full D contracted

    # --- Step 5: Down matmul (row-parallel) ---
    # hidden is F-fsdp-sharded × wo is F-fsdp-sharded → partial sum on F → psum(fsdp).
    # Output (M, D/TP): D stays tp-sharded.
    with jax.named_scope("colrow_a2a_down"):
        out_partial = jax.lax.ragged_dot(
            hidden.astype(wo_ext.dtype), wo_ext, group_sizes)
        out_sorted = jax.lax.psum(out_partial, fsdp_axis)

    out_sorted = out_sorted * sorted_local_w[:, None].astype(out_sorted.dtype)

    # Un-sort back to recv order
    out_recv = jnp.zeros_like(recv_x_flat).at[sort_local].set(
        out_sorted.astype(recv_x_flat.dtype))
    out_recv = out_recv.reshape(ep_size, capacity, D)

    # --- Step 6: Reverse A2A — send results back ---
    with jax.named_scope("a2a_return"):
        result_buckets = jax.lax.all_to_all(out_recv, ep_axis,
                                             split_axis=0, concat_axis=0, tiled=True)

    # Scatter-add back to original token positions; output stays (T, D/TP).
    result = jnp.zeros((T, D), dtype=flat_x.dtype)
    result = result.at[all_tids.reshape(-1)].add(
        result_buckets.reshape(-1, D).astype(flat_x.dtype))

    return result   # (T, D/TP) — D stays tp-sharded


def _gmm_ar_specs(mesh, act_spec):
    """Pick activation/weight/body specs for _moe_gmm_ar.

    See feedback_dsv3_tp2_colrow_pattern memory for full table. Summary:
    - tp>1 + ep==1: col/row parallel (v251/v264). NO weight AG. psum("tp")
                    after gate/up + psum("fsdp") after wo + AG("tp") at end
                    so output D is replicated for the residual stream.
    - tp>1 + ep>1:  AG-D fallback (existing).
    - ep>1 + tp=1:  A2A dispatch (v304 baseline path).
    - ep=1 + tp=1:  Simple FSDP-only.

    Returns: in_act_x, out_act_x, act_iw, wt, body, reduce_axis.
    """
    tp = mesh.shape.get("tp", 1) if mesh else 1
    ep = mesh.shape.get("ep", 1) if mesh else 1
    if tp > 1 and ep == 1:
        return dict(in_act_x=P("fsdp", "tp"), out_act_x=P("fsdp", None),
                    act_iw=P("fsdp", None), wt=P("ep", "fsdp", "tp"),
                    body=_expert_mlp_gmm_colrow_body, reduce_axis="fsdp")
    if tp > 1:
        # tp>1 + ep>1. With --moe_no_weight_ag: use colrow+A2A body (no weight AG;
        # psum on TP after wi, psum on FSDP after wo). Default: gmm_ar body which
        # AG's the F dim per layer.
        body = (_expert_mlp_gmm_colrow_a2a_body
                if _MOE_NO_WEIGHT_AG else _expert_mlp_gmm_ar_body)
        return dict(in_act_x=P(("fsdp", "ep"), "tp"), out_act_x=P(("fsdp", "ep"), "tp"),
                    act_iw=P(("fsdp", "ep"), None), wt=P("ep", "fsdp", "tp"),
                    body=body, reduce_axis="fsdp")
    if ep > 1:
        return dict(in_act_x=P(("fsdp", "ep"), None), out_act_x=P(("fsdp", "ep"), None),
                    act_iw=P(("fsdp", "ep"), None), wt=_moe_wo_spec(),
                    body=_expert_mlp_gmm_ar_body, reduce_axis="fsdp")
    return dict(in_act_x=act_spec, out_act_x=act_spec,
                act_iw=act_spec, wt=_moe_wo_spec(),
                body=_expert_mlp_gmm_ar_body, reduce_axis="fsdp")


@functools.partial(jax.custom_vjp, nondiff_argnums=(6, 7, 8, 9, 10))
def _moe_gmm_ar(fx, fi, fw, w0, w1, wout,
                mesh, K: int, act_spec, ep_axis: str, max_tpe: int):
    """AllGather-F MoE — gather full D_moe, no activation psum needed."""
    from jax.experimental.shard_map import shard_map
    s = _gmm_ar_specs(mesh, act_spec)

    @functools.partial(shard_map, mesh=mesh,
                       in_specs=(s["in_act_x"], s["act_iw"], s["act_iw"],
                                 s["wt"], s["wt"], s["wt"]),
                       out_specs=s["out_act_x"], check_rep=False)
    def _fn(fx_, fi_, fw_, w0_, w1_, wout_):
        return s["body"](fx_, w0_, w1_, wout_, fi_, fw_,
                         K, ep_axis, s["reduce_axis"])

    return _fn(fx, fi, fw, w0, w1, wout)


def _moe_gmm_ar_fwd(fx, fi, fw, w0, w1, wout,
                    mesh, K, act_spec, ep_axis, max_tpe):
    out = _moe_gmm_ar(fx, fi, fw, w0, w1, wout, mesh, K, act_spec, ep_axis, max_tpe)
    return out, (fx, fi, fw, w0, w1, wout)


def _moe_gmm_ar_bwd(mesh, K, act_spec, ep_axis, max_tpe, res, g):
    """Backward via jax.vjp through the same body picked by _gmm_ar_specs."""
    from jax.experimental.shard_map import shard_map
    s = _gmm_ar_specs(mesh, act_spec)
    fx, fi, fw, w0, w1, wout = res

    def _fwd(fx_, fw_, w0_, w1_, wout_):
        @functools.partial(shard_map, mesh=mesh,
                           in_specs=(s["in_act_x"], s["act_iw"], s["act_iw"],
                                     s["wt"], s["wt"], s["wt"]),
                           out_specs=s["out_act_x"], check_rep=False)
        def _fn(fx__, fi__, fw__, w0__, w1__, wout__):
            return s["body"](fx__, w0__, w1__, wout__, fi__, fw__,
                             K, ep_axis, s["reduce_axis"])
        return _fn(fx_, fi, fw_, w0_, w1_, wout_)

    _, vjp_fn = jax.vjp(_fwd, fx, fw, w0, w1, wout)
    d_fx, d_fw, d_w0, d_w1, d_wout = vjp_fn(g)
    return (d_fx, jnp.zeros_like(fi), d_fw, d_w0, d_w1, d_wout)


_moe_gmm_ar.defvjp(_moe_gmm_ar_fwd, _moe_gmm_ar_bwd)


# ============================================================================
# AG-dispatch GMM backend — AllGather tokens on EP + AllGather F on FSDP
# Best when EP < K+1 (e.g. EP=4): per-device dispatch comm = (EP-1)·T·D,
# smaller than A2A's K·T·D.
# ============================================================================

@functools.partial(jax.custom_vjp, nondiff_argnums=(3, 4, 5, 6))
def _sc_combine_with_vjp(out_local, idx, local_tids,
                          zero_rows: int, K: int, T_all: int, D: int):
    """SC gather-reduce wrapped in custom_vjp.

    fwd: out_with_zero = concat(out_local, zeros((zero_rows, D)))
         result[t] = sum over k of out_with_zero[idx[t*K + k]]      (T_all, D)

    bwd: each local_pos maps to EXACTLY ONE token (the route's owner),
         so d_out_local[local_pos] = d_result[local_tids[local_pos]] — pure gather.
         d_idx, d_local_tids: not differentiable (zero gradient).
    """
    return _sc_combine_fwd(out_local, idx, local_tids,
                           zero_rows, K, T_all, D)[0]


def _sc_combine_fwd(out_local, idx, local_tids, zero_rows, K, T_all, D):
    # v322: switched from kernels.gather_reduce_sc (MLIR-direct, used in v305)
    # to kernels.gather_reduce_pallas (production Pallas, lower per-call overhead).
    # Same signature, drop-in replacement.
    from kernels.gather_reduce_pallas import sc_gather_reduce
    zero_pad = jnp.zeros((zero_rows, D), dtype=out_local.dtype)
    out_with_zero = jnp.concatenate([out_local, zero_pad], axis=0)
    result = sc_gather_reduce(
        out_with_zero, idx, reduce_group_size=K, single_sc=True)
    return result, (idx, local_tids)


def _sc_combine_bwd(zero_rows, K, T_all, D, res, d_result):
    idx, local_tids = res
    # bwd of fwd-gather-then-sum: each local row got contribution from exactly
    # one (t, k); so d_out_local[i] = d_result[local_tids[i]] — pure gather.
    d_out_local = d_result[local_tids].astype(d_result.dtype)
    # idx and local_tids are integer arrays — return zero grads (placeholders).
    return d_out_local, jnp.zeros_like(idx), jnp.zeros_like(local_tids)


_sc_combine_with_vjp.defvjp(_sc_combine_fwd, _sc_combine_bwd)




def _maybe_check_finite(label: str, x, c: int, debug_nans: bool) -> None:
    """Print {any_nan, any_inf, max_abs} for `x` via jax.debug.print when
    debug_nans is True.

    ordered=True is NOT used: it raises 'OrderedDebugEffect not supported
    for more than 1 device' under pjit/shard_map on our 64-host setup
    (v341h discovery). Prints from different layers/chunks will interleave
    in the kubectl log, but the per-line label encodes (location, chunk)
    so we can still pinpoint the first non-finite tensor.
    """
    if not debug_nans:
        return
    jax.debug.print(
        "[finite-check] " + label + " c={c}: nan={n} inf={i} max_abs={m}",
        c=c,
        n=jnp.isnan(x).any(),
        i=jnp.isinf(x).any(),
        m=jnp.max(jnp.abs(x.astype(jnp.float32))),
    )


def _expert_mlp_gmm_ag_body(flat_x, wi_0, wi_1, wo, flat_indices, flat_weights,
                              K: int, ep_axis: str, fsdp_axis: str,
                              n_chunks: int = 2, use_sc_scatter: bool = False,
                              use_gmm_v2: bool = False,
                              use_fp8_weights: bool = False,
                              debug_nans: bool = False):
    """AG-dispatch MoE body with token chunking for compute/comm overlap.

    Chunks the post-AG processing into n_chunks token chunks. Per-chunk
    scatter+psum_scatter run in parallel with next chunk's sort+ragged_dot
    (XLA latency hiding scheduler). Token AG and weight AG happen ONCE at
    body entry; only the per-token compute is chunked.

    At v296 (T_all=131072, EP=4, K=8, E_local=64), n_chunks=2 gives
    rows-per-expert = 2048 — MXU sweet spot.

    wi_0, wi_1, wo: (E_local, F_local, D) sharded P("ep","fsdp",None)
    """
    T, D = flat_x.shape
    E_local    = wi_0.shape[0]
    ep_size    = jax.lax.axis_size(ep_axis)
    my_ep_rank = jax.lax.axis_index(ep_axis)
    local_start = my_ep_rank * E_local

    if use_fp8_weights:
        # Cast to fp8 BEFORE the AG so the AG runs in fp8 (halves the 7 GB
        # gathered weight allocation to 3.5 GB). barrier prevents XLA from
        # pushing the cast across the AG boundary.
        wi_0 = wi_0.astype(jnp.float8_e4m3fn)
        wi_1 = wi_1.astype(jnp.float8_e4m3fn)
        wo   = wo.astype(jnp.float8_e4m3fn)
        wi_0, wi_1, wo = jax.lax.optimization_barrier((wi_0, wi_1, wo))

    with jax.named_scope("weight_allgather_f"):
        wi_0_f = jax.lax.all_gather(wi_0, fsdp_axis, axis=_moe_wi_ag_axis(), tiled=True)
        wi_1_f = jax.lax.all_gather(wi_1, fsdp_axis, axis=_moe_wi_ag_axis(), tiled=True)
        wo_f   = jax.lax.all_gather(wo, fsdp_axis, axis=_moe_wo_ag_axis(), tiled=True)

    wi_0_t = wi_0_f.transpose(0, 2, 1)   # (E_local, D, F_full)
    wi_1_t = wi_1_f.transpose(0, 2, 1)

    # Per-chunk decomposition: slice LOCAL tokens BEFORE AG so each chunk has
    # its own AG → no cross-chunk dependency at the AG layer.
    assert T % n_chunks == 0, f"T={T} not divisible by n_chunks={n_chunks}"
    sz_local  = T // n_chunks                 # local tokens per chunk
    chunk_size = sz_local * ep_size           # tokens per chunk after AG (= T_all / n_chunks)

    # Pre-slice per-chunk inputs so the staggered_call loop sees them as a list.
    local_inputs = []
    for c in range(n_chunks):
        cs_local = c * sz_local
        local_inputs.append((
            jax.lax.dynamic_slice(flat_x,       [cs_local, 0], [sz_local, D]),
            jax.lax.dynamic_slice(flat_indices, [cs_local, 0], [sz_local, K]),
            jax.lax.dynamic_slice(flat_weights, [cs_local, 0], [sz_local, K]),
        ))

    def _process_chunk(c: int, inp):
        chunk_x_local, chunk_indices_local, chunk_weights_local = inp
        with jax.named_scope(f"chunk{c}"):
            # v323: REMOVED optimization_barriers around AG. Profile of v304
            # showed 4× ep_token_gather AGs took 20.2s exposed stall (1.26s/chip)
            # — the barriers prevented XLA from interleaving chunk0_AG with
            # chunk1_compute. Letting XLA schedule freely should pipeline.
            with jax.named_scope("ep_token_gather"):
                chunk_x       = jax.lax.all_gather(chunk_x_local,       ep_axis, axis=0, tiled=True)
                chunk_indices = jax.lax.all_gather(chunk_indices_local, ep_axis, axis=0, tiled=True)
                chunk_weights = jax.lax.all_gather(chunk_weights_local, ep_axis, axis=0, tiled=True)
            # v326: re-add post-AG barrier. v325 (no barriers) hit 2034 TPS/chip
            # but NaN — XLA reordered AG outputs in a way that broke correctness.
            chunk_x, chunk_indices, chunk_weights = jax.lax.optimization_barrier(
                (chunk_x, chunk_indices, chunk_weights))
            _maybe_check_finite("post_token_AG", chunk_x, c, debug_nans)

            # Phase 2: sort by expert, slice local block.
            TK_c = chunk_size * K
            exp_ids_c   = chunk_indices.reshape(-1).astype(jnp.int32)
            exp_ws_c    = chunk_weights.reshape(-1)
            token_ids_c = jnp.repeat(jnp.arange(chunk_size, dtype=jnp.int32), K)

            argsorted_c   = jnp.argsort(exp_ids_c, stable=True)
            sorted_tids_c = token_ids_c[argsorted_c]
            sorted_eids_c = exp_ids_c[argsorted_c]
            sorted_ws_c   = exp_ws_c[argsorted_c]

            blk_start_c = jnp.searchsorted(sorted_eids_c, local_start)
            blk_end_c   = jnp.searchsorted(sorted_eids_c, local_start + E_local)
            n_local_c   = blk_end_c - blk_start_c
            max_local_c = TK_c // ep_size

            local_tids_c     = jax.lax.dynamic_slice(sorted_tids_c, [blk_start_c], [max_local_c])
            local_ws_raw_c   = jax.lax.dynamic_slice(sorted_ws_c,   [blk_start_c], [max_local_c])
            local_eids_raw_c = jax.lax.dynamic_slice(sorted_eids_c, [blk_start_c], [max_local_c])

            valid_c = jnp.arange(max_local_c) < n_local_c
            local_eids_c = jnp.where(valid_c, local_eids_raw_c - local_start, 0).astype(jnp.int32)
            local_ws_c   = jnp.where(valid_c, local_ws_raw_c, 0.0)
            local_x_c    = chunk_x[local_tids_c]
            _maybe_check_finite("post_sort_local_x", local_x_c, c, debug_nans)
            _maybe_check_finite("post_sort_local_ws", local_ws_c, c, debug_nans)

            ends_c   = jnp.searchsorted(local_eids_c, jnp.arange(1, E_local + 1))
            starts_c = jnp.searchsorted(local_eids_c, jnp.arange(E_local))
            group_sizes_c = (ends_c - starts_c).astype(jnp.int32)

            # Phase 3: ragged_dots.
            if use_gmm_v2:
                # Pallas gmm_v2 with fused gate+up+silu; jax.vjp backward through ragged_dot reference.
                from kernels.gmm_v2_train import gmm_v2_train, gmm_v2_fused_silu_train
                # Fused gate+up+silu: 3 ragged_dots → 2 gmm_v2 calls.
                # vmem_limit=48M required for fused at our M=131072, N=2*2048 (default OOMs).
                hidden = gmm_v2_fused_silu_train(
                    local_x_c.astype(wi_0_t.dtype), wi_0_t, wi_1_t,
                    group_sizes_c, 48 * 1024 * 1024)
                out_local_c = gmm_v2_train(
                    hidden.astype(wo_f.dtype), wo_f, group_sizes_c, 0)
            elif use_fp8_weights:
                # Mixed-precision ragged_dot: bf16 activation × fp8 weight → bf16.
                # Keep activation in bf16 (DON'T downcast to wi_0_t.dtype, which is fp8).
                x_bf16 = local_x_c.astype(jnp.bfloat16)
                gate = jax.nn.silu(jax.lax.ragged_dot(
                    x_bf16, wi_0_t, group_sizes_c,
                    preferred_element_type=jnp.bfloat16))
                up = jax.lax.ragged_dot(
                    x_bf16, wi_1_t, group_sizes_c,
                    preferred_element_type=jnp.bfloat16)
                hidden = (gate * up).astype(jnp.bfloat16)
                out_local_c = jax.lax.ragged_dot(
                    hidden, wo_f, group_sizes_c,
                    preferred_element_type=jnp.bfloat16)
            else:
                gate = jax.nn.silu(jax.lax.ragged_dot(
                    local_x_c.astype(wi_0_t.dtype), wi_0_t, group_sizes_c))
                _maybe_check_finite("post_gate_silu", gate, c, debug_nans)
                up = jax.lax.ragged_dot(
                    local_x_c.astype(wi_1_t.dtype), wi_1_t, group_sizes_c)
                _maybe_check_finite("post_up", up, c, debug_nans)
                hidden = gate * up
                out_local_c = jax.lax.ragged_dot(
                    hidden.astype(wo_f.dtype), wo_f, group_sizes_c)
                _maybe_check_finite("post_wo", out_local_c, c, debug_nans)

            out_local_c = out_local_c * local_ws_c[:, None].astype(out_local_c.dtype)
            _maybe_check_finite("post_weight_mask", out_local_c, c, debug_nans)

            # Phase 4: scatter back to per-token rows, then psum_scatter across EP.
            # v323: removed optimization_barrier around scatter — same reason as Phase 1.
            if use_sc_scatter:
                # v305 attempt — SC gather-reduce. Slower in practice
                # (~5% TPS regression at ~230 calls/step due to per-call overhead).
                # Kept behind flag for experimentation.
                inv_argsorted_c = jnp.argsort(argsorted_c)
                in_slice = ((inv_argsorted_c >= blk_start_c) &
                            (inv_argsorted_c < blk_start_c + max_local_c))
                idx_c = jnp.where(in_slice,
                                  inv_argsorted_c - blk_start_c,
                                  max_local_c).astype(jnp.int32)
                # zero_rows multiple of K (kernel constraint).
                full_out_c = _sc_combine_with_vjp(
                    out_local_c.astype(flat_x.dtype), idx_c, local_tids_c,
                    zero_rows=K, K=K, T_all=chunk_size, D=D)
            else:
                # Default — HBM scatter-add. Best perf at our chunk count.
                full_out_c = jnp.zeros((chunk_size, D), dtype=flat_x.dtype).at[
                    local_tids_c].add(out_local_c.astype(flat_x.dtype))
            _maybe_check_finite("post_scatter", full_out_c, c, debug_nans)
            result_c = jax.lax.psum_scatter(full_out_c, ep_axis,
                                             scatter_dimension=0, tiled=True)
            _maybe_check_finite("post_psum_scatter", result_c, c, debug_nans)
            # v323 removed this barrier for n_chunks=2 perf; v341 found NaN at
            # n_chunks=4 (same class as v325). Re-add ONLY when n_chunks > 2 —
            # forces each chunk's psum_scatter output to be sealed before the
            # downstream concat, preventing XLA from cross-chunk reorders that
            # corrupt scatter-add ordering at higher chunk counts.
            if n_chunks > 2:
                result_c = jax.lax.optimization_barrier(result_c)
        return result_c

    # Sibling chunks: each _process_chunk has its own AG → no cross-chunk
    # operand. Intra-chunk barriers (around AG and scatter blocks above)
    # keep each chunk atomic; XLA can schedule the two chunks in parallel
    # on different engines (SC for AG/scatter, TC for ragged_dots).
    # NO cross-chunk barrier — that would serialize them (MaxText's
    # staggered_call pattern is for whole-microbatch serialization, not
    # within-layer chunking).
    chunks = [_process_chunk(c, local_inputs[c]) for c in range(n_chunks)]
    return jnp.concatenate(chunks, axis=0)   # (T, D)


@functools.partial(jax.custom_vjp, nondiff_argnums=(6, 7, 8, 9, 10, 11, 12, 13, 14, 15))
def _moe_gmm_ag(fx, fi, fw, w0, w1, wout,
                mesh, K: int, act_spec, ep_axis: str, max_tpe: int,
                use_sc_scatter: bool = False, use_gmm_v2: bool = False,
                n_chunks: int = 2, use_fp8_weights: bool = False,
                debug_nans: bool = False):
    """AG-dispatch GMM — AllGather tokens on EP + AllGather F on FSDP."""
    from jax.experimental.shard_map import shard_map
    ep = mesh.shape.get("ep", 1) if mesh else 1
    if ep > 1:
        _act_x  = P(("fsdp", "ep"), None)
        _act_iw = P(("fsdp", "ep"), None)
    else:
        _act_x  = act_spec
        _act_iw = act_spec
    _wt_i = _moe_wi_spec()
    _wt_o = _moe_wo_spec()

    @functools.partial(shard_map, mesh=mesh,
                       in_specs=(_act_x, _act_iw, _act_iw, _wt_i, _wt_i, _wt_o),
                       out_specs=_act_x, check_rep=False)
    def _fn(fx_, fi_, fw_, w0_, w1_, wout_):
        return _expert_mlp_gmm_ag_body(fx_, w0_, w1_, wout_, fi_, fw_,
                                       K, ep_axis, "fsdp",
                                       n_chunks=n_chunks,
                                       use_sc_scatter=use_sc_scatter,
                                       use_gmm_v2=use_gmm_v2,
                                       use_fp8_weights=use_fp8_weights,
                                       debug_nans=debug_nans)
    return _fn(fx, fi, fw, w0, w1, wout)


def _moe_gmm_ag_fwd(fx, fi, fw, w0, w1, wout,
                    mesh, K, act_spec, ep_axis, max_tpe, use_sc_scatter, use_gmm_v2,
                    n_chunks, use_fp8_weights, debug_nans):
    out = _moe_gmm_ag(fx, fi, fw, w0, w1, wout, mesh, K, act_spec, ep_axis,
                      max_tpe, use_sc_scatter, use_gmm_v2, n_chunks, use_fp8_weights,
                      debug_nans)
    return out, (fx, fi, fw, w0, w1, wout)


def _moe_gmm_ag_bwd(mesh, K, act_spec, ep_axis, max_tpe, use_sc_scatter, use_gmm_v2,
                    n_chunks, use_fp8_weights, debug_nans, res, g):
    from jax.experimental.shard_map import shard_map
    ep = mesh.shape.get("ep", 1) if mesh else 1
    if ep > 1:
        _act_x  = P(("fsdp", "ep"), None)
        _act_iw = P(("fsdp", "ep"), None)
    else:
        _act_x  = act_spec
        _act_iw = act_spec
    _wt_i = _moe_wi_spec()
    _wt_o = _moe_wo_spec()

    fx, fi, fw, w0, w1, wout = res

    _maybe_check_finite("bwd_in_g", g, 0, debug_nans)

    def _fwd(fx_, fw_, w0_, w1_, wout_):
        @functools.partial(shard_map, mesh=mesh,
                           in_specs=(_act_x, _act_iw, _act_iw, _wt_i, _wt_i, _wt_o),
                           out_specs=_act_x, check_rep=False)
        def _fn(fx__, fi__, fw__, w0__, w1__, wout__):
            return _expert_mlp_gmm_ag_body(fx__, w0__, w1__, wout__, fi__, fw__,
                                           K, ep_axis, "fsdp",
                                           n_chunks=n_chunks,
                                           use_sc_scatter=use_sc_scatter,
                                           use_gmm_v2=use_gmm_v2,
                                           use_fp8_weights=use_fp8_weights,
                                           debug_nans=debug_nans)
        return _fn(fx_, fi, fw_, w0_, w1_, wout_)

    _, vjp_fn = jax.vjp(_fwd, fx, fw, w0, w1, wout)
    d_fx, d_fw, d_w0, d_w1, d_wout = vjp_fn(g)

    _maybe_check_finite("bwd_out_d_fx",   d_fx,   0, debug_nans)
    _maybe_check_finite("bwd_out_d_fw",   d_fw,   0, debug_nans)
    _maybe_check_finite("bwd_out_d_w0",   d_w0,   0, debug_nans)
    _maybe_check_finite("bwd_out_d_w1",   d_w1,   0, debug_nans)
    _maybe_check_finite("bwd_out_d_wout", d_wout, 0, debug_nans)

    return (d_fx, jnp.zeros_like(fi), d_fw, d_w0, d_w1, d_wout)


_moe_gmm_ag.defvjp(_moe_gmm_ag_fwd, _moe_gmm_ag_bwd)


def expert_mlp_gmm_ag(x, wi_0, wi_1, wo, top_k_weights, top_k_indices,
                      cfg: ModelConfig):
    """AG-dispatch GMM (best when EP < K+1, e.g. EP=4)."""
    B, S, D = x.shape
    K = cfg.K
    flat_x       = x.reshape(B * S, D)
    flat_indices = top_k_indices.reshape(B * S, K)
    flat_weights = top_k_weights.reshape(B * S, K)

    if cfg.mesh is None:
        with jax.named_scope("moe_gmm_ag"):
            out = _ragged_dot_no_mesh(flat_x, wi_0, wi_1, wo,
                                       flat_indices, flat_weights, K, cfg.E)
        return out.reshape(B, S, D)

    act_spec = P("fsdp", None)
    fsdp_size = cfg.mesh.shape.get("fsdp", 1)
    ep_size   = cfg.mesh.shape.get("ep",   1)
    T_local   = B * S // (fsdp_size * max(ep_size, 1))
    max_tpe   = max(1, 2 * T_local * K // cfg.E)

    with jax.named_scope("moe_gmm_ag"):
        out = _moe_gmm_ag(flat_x, flat_indices, flat_weights,
                          wi_0, wi_1, wo,
                          cfg.mesh, K, act_spec, "ep", max_tpe,
                          cfg.moe_use_sc_scatter, cfg.moe_use_gmm_v2,
                          cfg.moe_n_chunks, cfg.moe_fp8_weights,
                          cfg.moe_debug_nans)
    return out.reshape(B, S, D)


# ============================================================================
# Cross-layer FSDP weight prefetch (v295) — uses jax.lax.pcast / "reduced"
# AG semantics so the bwd transposes to a reduce_scatter automatically and
# d_ws_ag stays as a per-shard tensor (no 109 GB stack like v294).
# Pattern from MaxText src/maxtext/models/deepseek_batchsplit.py:124.
# ============================================================================

def _ag_one_moe_layer(lp_dict, mesh):
    """AllGather one MoE layer's wi_0/wi_1/wo along F using pcast/reduced.

    Uses an inline Explicit-axes mesh so reduced semantics work even when the
    global mesh is Auto. Inputs are jax.reshard'd from Auto→Explicit before
    the shard_map; output NamedSharding ties to the Explicit mesh and is
    bridged back to Auto by downstream consumers.
    """
    from jax.experimental.shard_map import shard_map
    from jax.sharding import AxisType, Mesh as _Mesh, NamedSharding
    explicit_mesh = _Mesh(mesh.devices, mesh.axis_names,
                          axis_types=(AxisType.Explicit,) * len(mesh.axis_names))
    in_sharding  = NamedSharding(explicit_mesh, _moe_wo_spec())

    @functools.partial(shard_map, mesh=explicit_mesh,
                       in_specs=_moe_wo_spec(),
                       out_specs=P("ep", None, None,
                                    reduced={"dp", "fsdp"}),
                       check_rep=False)
    def _ag(w):
        w = jax.lax.pcast(w, axis_name="dp", to="reduced")
        w = jax.lax.all_gather(w, axis_name="fsdp",
                               tiled=True, axis=1, to="reduced")
        return w

    def _ag_one(w):
        w_explicit = jax.reshard(w, in_sharding)
        return _ag(w_explicit)

    return {**lp_dict,
            "wi_0": _ag_one(lp_dict["wi_0"]),
            "wi_1": _ag_one(lp_dict["wi_1"]),
            "wo":   _ag_one(lp_dict["wo"])}


def _expert_mlp_gmm_ag_body_pre(flat_x, wi_0_full, wi_1_full, wo_full,
                                 flat_indices, flat_weights,
                                 K: int, ep_axis: str):
    """Like _expert_mlp_gmm_ag_body but weights are already FSDP-AG'd."""
    T, D = flat_x.shape
    E_local    = wi_0_full.shape[0]
    ep_size    = jax.lax.axis_size(ep_axis)
    my_ep_rank = jax.lax.axis_index(ep_axis)
    local_start = my_ep_rank * E_local

    with jax.named_scope("ep_token_gather"):
        all_x       = jax.lax.all_gather(flat_x,       ep_axis, axis=0, tiled=True)
        all_indices = jax.lax.all_gather(flat_indices, ep_axis, axis=0, tiled=True)
        all_weights = jax.lax.all_gather(flat_weights, ep_axis, axis=0, tiled=True)

    wi_0_t = wi_0_full.transpose(0, 2, 1)
    wi_1_t = wi_1_full.transpose(0, 2, 1)

    T_all = all_x.shape[0]
    TK    = T_all * K
    exp_ids   = all_indices.reshape(-1).astype(jnp.int32)
    exp_ws    = all_weights.reshape(-1)
    token_ids = jnp.repeat(jnp.arange(T_all, dtype=jnp.int32), K)

    with jax.named_scope("sort_by_expert"):
        argsorted    = jnp.argsort(exp_ids, stable=True)
        sorted_tids  = token_ids[argsorted]
        sorted_eids  = exp_ids[argsorted]
        sorted_ws    = exp_ws[argsorted]

    blk_start = jnp.searchsorted(sorted_eids, local_start)
    blk_end   = jnp.searchsorted(sorted_eids, local_start + E_local)
    n_local   = blk_end - blk_start
    max_local = TK // ep_size

    with jax.named_scope("local_block_extract"):
        local_tids     = jax.lax.dynamic_slice(sorted_tids, [blk_start], [max_local])
        local_ws_raw   = jax.lax.dynamic_slice(sorted_ws,   [blk_start], [max_local])
        local_eids_raw = jax.lax.dynamic_slice(sorted_eids, [blk_start], [max_local])

    valid = jnp.arange(max_local) < n_local
    local_eids = jnp.where(valid, local_eids_raw - local_start, 0).astype(jnp.int32)
    local_ws   = jnp.where(valid, local_ws_raw, 0.0)
    local_x    = all_x[local_tids]

    with jax.named_scope("group_sizes"):
        ends   = jnp.searchsorted(local_eids, jnp.arange(1, E_local + 1))
        starts = jnp.searchsorted(local_eids, jnp.arange(E_local))
        group_sizes = (ends - starts).astype(jnp.int32)

    with jax.named_scope("ag_gate_up"):
        gate = jax.nn.silu(jax.lax.ragged_dot(
            local_x.astype(wi_0_t.dtype), wi_0_t, group_sizes))
        up = jax.lax.ragged_dot(
            local_x.astype(wi_1_t.dtype), wi_1_t, group_sizes)
    hidden = gate * up

    with jax.named_scope("ag_down"):
        out_local = jax.lax.ragged_dot(
            hidden.astype(wo_full.dtype), wo_full, group_sizes)

    out_local = out_local * local_ws[:, None].astype(out_local.dtype)

    full_out = jnp.zeros((T_all, D), dtype=flat_x.dtype).at[local_tids].add(
        out_local.astype(flat_x.dtype))

    with jax.named_scope("ep_scatter"):
        result = jax.lax.psum_scatter(full_out, ep_axis,
                                      scatter_dimension=0, tiled=True)
    return result


@functools.partial(jax.custom_vjp, nondiff_argnums=(6, 7, 8, 9, 10))
def _moe_gmm_ag_pre(fx, fi, fw, w0_full, w1_full, wout_full,
                    mesh, K: int, act_spec, ep_axis: str, max_tpe: int):
    """AG-dispatch GMM with pre-AG'd weights (cross-layer prefetch path).

    Weights arrive with sharding P("ep", None, None, reduced={"dp","fsdp"}).
    """
    from jax.experimental.shard_map import shard_map
    ep = mesh.shape.get("ep", 1) if mesh else 1
    if ep > 1:
        _act_x  = P(("fsdp", "ep"), None)
        _act_iw = P(("fsdp", "ep"), None)
        _wt     = P("ep", None, None, reduced={"dp", "fsdp"})
    else:
        _act_x  = act_spec
        _act_iw = act_spec
        _wt     = P("ep", None, None, reduced={"dp", "fsdp"})

    @functools.partial(shard_map, mesh=mesh,
                       in_specs=(_act_x, _act_iw, _act_iw, _wt, _wt, _wt),
                       out_specs=_act_x, check_rep=False)
    def _fn(fx_, fi_, fw_, w0_, w1_, wout_):
        return _expert_mlp_gmm_ag_body_pre(fx_, w0_, w1_, wout_, fi_, fw_,
                                           K, ep_axis)
    return _fn(fx, fi, fw, w0_full, w1_full, wout_full)


def _moe_gmm_ag_pre_fwd(fx, fi, fw, w0, w1, wout, mesh, K, act_spec, ep_axis, max_tpe):
    out = _moe_gmm_ag_pre(fx, fi, fw, w0, w1, wout, mesh, K, act_spec, ep_axis, max_tpe)
    return out, (fx, fi, fw, w0, w1, wout)


def _moe_gmm_ag_pre_bwd(mesh, K, act_spec, ep_axis, max_tpe, res, g):
    from jax.experimental.shard_map import shard_map
    ep = mesh.shape.get("ep", 1) if mesh else 1
    if ep > 1:
        _act_x  = P(("fsdp", "ep"), None)
        _act_iw = P(("fsdp", "ep"), None)
        _wt     = P("ep", None, None, reduced={"dp", "fsdp"})
    else:
        _act_x  = act_spec
        _act_iw = act_spec
        _wt     = P("ep", None, None, reduced={"dp", "fsdp"})

    fx, fi, fw, w0, w1, wout = res

    def _fwd(fx_, fw_, w0_, w1_, wout_):
        @functools.partial(shard_map, mesh=mesh,
                           in_specs=(_act_x, _act_iw, _act_iw, _wt, _wt, _wt),
                           out_specs=_act_x, check_rep=False)
        def _fn(fx__, fi__, fw__, w0__, w1__, wout__):
            return _expert_mlp_gmm_ag_body_pre(fx__, w0__, w1__, wout__, fi__, fw__,
                                               K, ep_axis)
        return _fn(fx_, fi, fw_, w0_, w1_, wout_)

    _, vjp_fn = jax.vjp(_fwd, fx, fw, w0, w1, wout)
    d_fx, d_fw, d_w0, d_w1, d_wout = vjp_fn(g)
    return (d_fx, jnp.zeros_like(fi), d_fw, d_w0, d_w1, d_wout)


_moe_gmm_ag_pre.defvjp(_moe_gmm_ag_pre_fwd, _moe_gmm_ag_pre_bwd)


def expert_mlp_gmm_ag_pre(x, wi_0_full, wi_1_full, wo_full,
                          top_k_weights, top_k_indices, cfg: ModelConfig):
    """AG-dispatch GMM with pre-AG'd weights (cross-layer prefetch path)."""
    B, S, D = x.shape
    K = cfg.K
    flat_x       = x.reshape(B * S, D)
    flat_indices = top_k_indices.reshape(B * S, K)
    flat_weights = top_k_weights.reshape(B * S, K)
    fsdp_size = cfg.mesh.shape.get("fsdp", 1)
    ep_size   = cfg.mesh.shape.get("ep",   1)
    T_local   = B * S // (fsdp_size * max(ep_size, 1))
    max_tpe   = max(1, 2 * T_local * K // cfg.E)

    act_spec = P("fsdp", None)
    with jax.named_scope("moe_gmm_ag_pre"):
        out = _moe_gmm_ag_pre(flat_x, flat_indices, flat_weights,
                              wi_0_full, wi_1_full, wo_full,
                              cfg.mesh, K, act_spec, "ep", max_tpe)
    return out.reshape(B, S, D)


def expert_mlp_gmm_ar(x, wi_0, wi_1, wo, top_k_weights, top_k_indices,
                       cfg: ModelConfig):
    """AllReduce-GMM MoE: partial F compute + AllReduce, no weight AllGather.

    Enables large FSDP (256+) without materialising full expert weight matrices.
    Compatible with EP≥1; for EP=1 the ep psum is a no-op.
    """
    B, S, D = x.shape
    K = cfg.K
    flat_x       = x.reshape(B * S, D)
    flat_indices = top_k_indices.reshape(B * S, K)
    flat_weights = top_k_weights.reshape(B * S, K)

    if cfg.mesh is None:
        # No mesh: fall back to single-device ragged_dot (no FSDP/EP)
        with jax.named_scope("moe_gmm_ar"):
            out = _ragged_dot_no_mesh(flat_x, wi_0, wi_1, wo,
                                       flat_indices, flat_weights, K, cfg.E)
        return out.reshape(B, S, D)

    act_spec  = P("fsdp", None)
    fsdp_size = cfg.mesh.shape.get("fsdp", 1)
    tp_size   = cfg.mesh.shape.get("tp",   1)
    ep_size   = cfg.mesh.shape.get("ep",   1)
    # Tokens per device: sharded by fsdp (and ep if token-sharded).
    # EP>1 with token sharding: tokens split by fsdp×ep.
    T_local   = B * S // (fsdp_size * max(ep_size, 1))
    max_tpe   = max(1, 2 * T_local * K // cfg.E)

    with jax.named_scope("moe_gmm_ar"):
        out = _moe_gmm_ar(flat_x, flat_indices, flat_weights,
                          wi_0, wi_1, wo,
                          cfg.mesh, K, act_spec, "ep", max_tpe)
    return out.reshape(B, S, D)


def _ragged_dot_no_mesh(flat_x, wi_0, wi_1, wo, flat_indices, flat_weights, K, E):
    """Single-device ragged_dot path (no mesh, no shard_map, no psum)."""
    T, D = flat_x.shape
    SC_MAX_ROWS = 65536
    n_chunks = 1
    while T // n_chunks > SC_MAX_ROWS:
        n_chunks *= 2
    T_c = T // n_chunks

    def _rd_block(x_b, idx_b, w_b):
        T_b = x_b.shape[0]
        exp_ids   = idx_b.reshape(-1).astype(jnp.int32)
        exp_ws    = w_b.reshape(-1)
        token_ids = jnp.repeat(jnp.arange(T_b, dtype=jnp.int32), K)
        argsorted        = jnp.argsort(exp_ids, stable=True)
        sorted_token_ids = token_ids[argsorted]
        sorted_exp_ids   = exp_ids[argsorted]
        sorted_weights   = exp_ws[argsorted]
        sorted_x = x_b[sorted_token_ids]
        ends        = jnp.searchsorted(sorted_exp_ids, jnp.arange(1, E + 1))
        starts      = jnp.searchsorted(sorted_exp_ids, jnp.arange(E))
        group_sizes = (ends - starts).astype(jnp.int32)
        gate = jax.nn.silu(jax.lax.ragged_dot(
            sorted_x.astype(wi_0.dtype), wi_0, group_sizes))
        up   = jax.lax.ragged_dot(sorted_x.astype(wi_1.dtype), wi_1, group_sizes)
        out_sorted = jax.lax.ragged_dot(
            (gate * up).astype(wo.dtype), wo, group_sizes)
        out_sorted = out_sorted * sorted_weights[:, None].astype(out_sorted.dtype)
        return jnp.zeros((T_b, D), dtype=x_b.dtype).at[sorted_token_ids].add(
            out_sorted.astype(x_b.dtype))

    chunks = [
        _rd_block(flat_x[c * T_c:(c + 1) * T_c],
                  flat_indices[c * T_c:(c + 1) * T_c],
                  flat_weights[c * T_c:(c + 1) * T_c])
        for c in range(n_chunks)
    ]
    return jnp.concatenate(chunks, axis=0) if n_chunks > 1 else chunks[0]


def expert_mlp_ragged_dot(x, wi_0, wi_1, wo, top_k_weights, top_k_indices,
                           cfg: ModelConfig):
    """Ragged-dot EP=1 MoE: O(T×K) computation, no SC gather, pure JAX."""
    B, S, D = x.shape
    K, E = cfg.K, cfg.E
    flat_x       = x.reshape(B * S, D)
    flat_indices = top_k_indices.reshape(B * S, K)
    flat_weights = top_k_weights.reshape(B * S, K)

    if cfg.mesh is None:
        with jax.named_scope("moe_ragged_dot"):
            out = _ragged_dot_no_mesh(flat_x, wi_0, wi_1, wo, flat_indices, flat_weights,
                                      K, E)
        return out.reshape(B, S, D)

    act_spec = P("fsdp", None)

    with jax.named_scope("moe_ragged_dot"):
        out = _moe_jax_ep_rd(flat_x, flat_indices, flat_weights,
                              wi_0, wi_1, wo,
                              cfg.mesh, K, act_spec, "ep", 1)
    return out.reshape(B, S, D)


# ============================================================================
# MoE Pallas backend — fused_ep_moe_v4 (Pallas fwd + Pallas bwd v4)
# ============================================================================

def _moe_pallas_call(fx, fi, fw, w0, w1, wo, mesh, K, act_spec, ep_axis,
                     collective_id: int = 0):
    """Standalone pallas forward kernel call — no custom_vjp wrapping.

    v204: extracted from _moe_pallas_v4 primal so that both the primal eval
    (value path of value_and_grad) and the VJP fwd rule can call it with
    DISTINCT collective_id values.

    Background: jax.value_and_grad evaluates the primal ONCE (for the loss
    value) and the VJP forward ONCE (for the gradient).  For lax.scan with
    L_moe=1, the primal eval is unrolled directly into the main SPMD
    computation (main.139_spmd), while the VJP forward scan lives in a
    while-loop body computation.  Both paths call fused_ep_moe_fwd_streaming_v1
    and produce separate HLO computations (.0 and .1).  If they share
    collective_id=0, XLA's async collective fusion can pipeline the two ICI
    ring-reduce transfers onto the same hardware channel → collision → corrupted
    ring-reduce metadata → start_off OOB → BoundsCheck 21.

    Fix: primal path uses collective_id=0; VJP fwd path uses collective_id=1.
    Different IDs → different ICI channels → no collision.
    """
    from forward_kernel import fused_ep_moe_fwd_streaming_v1
    from jax.experimental.shard_map import shard_map

    E_global = w0.shape[0]
    tp_size = mesh.shape["tp"] if "tp" in mesh.shape else 1
    tp_iota = jnp.arange(tp_size, dtype=jnp.int32)

    @functools.partial(shard_map, mesh=mesh,
                       in_specs=(act_spec, act_spec, act_spec,
                                 _moe_wi_spec(),
                                 _moe_wi_spec(),
                                 _moe_wo_spec(),
                                 P("tp")),  # brings "tp" into shard_map axis scope
                       out_specs=act_spec, check_rep=False)
    def _fn(fx_, fi_, fw_, w0_, w1_, wo_, _tp_dummy):
        w1_stk = jnp.stack([w0_, w1_], axis=1)
        # lax.axis_index("tp") at JAX/shard_map level lowers to XLA partition_id —
        # a hardware primitive that SPMD never folds to a constant.
        tp_rank = jax.lax.axis_index("tp").astype(jnp.int32)  # 0 or 1, per device
        tp_rank_arr_local = jnp.full((1,), tp_rank, dtype=jnp.int32)  # shape (1,)
        out = fused_ep_moe_fwd_streaming_v1(
            fx_, w1_stk, wo_,
            gating_output=None, top_k=K, act_fn="silu",
            ep_axis_name=ep_axis,
            tp_axis_name=None,
            tp_rank_arr=tp_rank_arr_local,
            top_k_indices_precomputed=fi_,
            top_k_weights_precomputed=fw_,
            E_global_override=E_global,
            collective_id=collective_id,
        )
        return jax.lax.psum(out, "fsdp")

    return _fn(fx, fi, fw, w0, w1, wo, tp_iota)


@functools.partial(jax.custom_vjp, nondiff_argnums=(7, 8, 9, 10, 11))
def _moe_pallas_v4(fx, fi, fw, fg, w0, w1, wo,
                   mesh, K: int, act_spec, ep_axis: str, max_tpe: int):
    """Pallas EP MoE: streaming-v1 forward. Backward via fused_ep_moe_bwd_v4.

    fi, fw: precomputed top-k routing (with gate_bias) — used by both forward
            and backward, ensuring identical expert selection in both passes.
    fg: biased sigmoid scores — saved in residuals only (not used for routing).
    """
    # collective_id=0: primal eval (value path of value_and_grad).
    return _moe_pallas_call(fx, fi, fw, w0, w1, wo, mesh, K, act_spec, ep_axis,
                            collective_id=0)


def _moe_pallas_v4_fwd(fx, fi, fw, fg, w0, w1, wo,
                        mesh, K, act_spec, ep_axis, max_tpe):
    out = _moe_pallas_v4(fx, fi, fw, fg, w0, w1, wo,
                          mesh, K, act_spec, ep_axis, max_tpe)
    # checkpoint_name: saves these tensors into the residual stash so that
    # jax.remat / gradient_checkpoint does NOT re-execute the Pallas A2A kernel
    # in the backward pass (would hang on ICI DMA in non-forward context).
    ck = jax._src.ad_checkpoint.checkpoint_name
    return ck(out, 'p_out'), (
        ck(fx, 'p_fx'), ck(fi, 'p_fi'), ck(fw, 'p_fw'), ck(fg, 'p_fg'),
        ck(w0, 'p_w0'), ck(w1, 'p_w1'), ck(wo, 'p_wo'),
    )


def _moe_pallas_v4_bwd(mesh, K, act_spec, ep_axis, max_tpe, res, g):
    """Backward via fused_ep_moe_bwd_v4 Pallas kernel."""
    from .kernels.fused_moe_bwd.backward_kernel_v4 import fused_ep_moe_bwd_v4
    from jax.experimental.shard_map import shard_map

    fx, fi, fw, fg, w0, w1, wo = res
    E_global = w0.shape[0]

    def _bwd_fn(g_l, fx_l, fw_l, fi_l, w0_l, w1_l, wo_l):
        # v4 kernel requires F_shard % 128 == 0.
        # With FSDP=32: F_shard = D_moe/32 = 64 — too small, gather to full F first.
        # With FSDP=16: F_shard = 128 — OK directly.
        F_shard = w0_l.shape[2]
        needs_gather = (F_shard % 128 != 0)

        if needs_gather:
            w0_k = jax.lax.all_gather(w0_l, "fsdp", axis=2, tiled=True)
            w1_k = jax.lax.all_gather(w1_l, "fsdp", axis=2, tiled=True)
            wo_k = jax.lax.all_gather(wo_l, "fsdp", axis=1, tiled=True)
        else:
            w0_k, w1_k, wo_k = w0_l, w1_l, wo_l

        w1_stk = jnp.stack([w0_k, w1_k], axis=1)

        d_tok_p, d_w1_p, d_wo_p, d_topk_p = fused_ep_moe_bwd_v4(
            g_l, fx_l, w1_stk, wo_k,
            gating_output=None,
            top_k=K,
            scoring_fn="sigmoid",
            renormalize_topk_logits=True,
            act_fn="silu",
            ep_axis_name=ep_axis,
            top_k_indices_precomputed=fi_l,
            top_k_weights_precomputed=fw_l.astype(jnp.float32),
            return_dtopk=True,
            E_global_override=E_global,
        )

        if needs_gather:
            fsdp_id = jax.lax.axis_index("fsdp")
            d_w0_l = jax.lax.dynamic_slice_in_dim(d_w1_p[:, 0], fsdp_id * F_shard, F_shard, 2)
            d_w1_l = jax.lax.dynamic_slice_in_dim(d_w1_p[:, 1], fsdp_id * F_shard, F_shard, 2)
            d_wo_l = jax.lax.dynamic_slice_in_dim(d_wo_p,       fsdp_id * F_shard, F_shard, 1)
        else:
            d_w0_l = d_w1_p[:, 0]
            d_w1_l = d_w1_p[:, 1]
            d_wo_l = d_wo_p

        d_tok  = jax.lax.psum(d_tok_p.astype(jnp.bfloat16), ep_axis)
        d_topk = jax.lax.psum(d_topk_p, ep_axis)
        return (d_tok.astype(g_l.dtype),
                d_topk.astype(fw_l.dtype),
                d_w0_l.astype(w0_l.dtype),
                d_w1_l.astype(w1_l.dtype),
                d_wo_l.astype(wo_l.dtype))

    d_fx, d_fw, d_w0, d_w1, d_wo = shard_map(
        _bwd_fn, mesh=mesh,
        in_specs=(act_spec, act_spec, act_spec, act_spec,
                  _moe_wi_spec(), _moe_wi_spec(), _moe_wo_spec()),
        out_specs=(act_spec, act_spec,
                   _moe_wi_spec(), _moe_wi_spec(), _moe_wo_spec()),
        check_rep=False,
    )(g, fx, fw, fi, w0, w1, wo)

    return (d_fx, jnp.zeros_like(fi), d_fw, jnp.zeros_like(fg), d_w0, d_w1, d_wo)


_moe_pallas_v4.defvjp(_moe_pallas_v4_fwd, _moe_pallas_v4_bwd)


# --------------------------------------------------------------------------
# Hybrid: JAX forward + Pallas v4 backward (fast compile, tests backward)
# --------------------------------------------------------------------------

@functools.partial(jax.custom_vjp, nondiff_argnums=(7, 8, 9, 10, 11))
def _moe_jax_fwd_pv4_bwd(fx, fi, fw, fg, w0, w1, wo,
                           mesh, K: int, act_spec, ep_axis: str, max_tpe: int):
    """JAX EP MoE forward + Pallas v4 backward (gradient correctness test)."""
    from jax.experimental.shard_map import shard_map

    @functools.partial(shard_map, mesh=mesh,
                       in_specs=(act_spec, act_spec, act_spec,
                                 _moe_wi_spec(),
                                 _moe_wi_spec(),
                                 _moe_wo_spec()),
                       out_specs=act_spec, check_rep=False)
    def _fn(fx_, fi_, fw_, w0_, w1_, wo_):
        return _expert_mlp_ep_body(fx_, w0_, w1_, wo_, fi_, fw_, K, ep_axis, max_tpe)

    return _fn(fx, fi, fw, w0, w1, wo)


def _moe_jax_fwd_pv4_bwd_fwd(fx, fi, fw, fg, w0, w1, wo,
                               mesh, K, act_spec, ep_axis, max_tpe):
    out = _moe_jax_fwd_pv4_bwd(fx, fi, fw, fg, w0, w1, wo,
                                 mesh, K, act_spec, ep_axis, max_tpe)
    ck = jax._src.ad_checkpoint.checkpoint_name
    return ck(out, 'p_out'), (
        ck(fx, 'p_fx'), ck(fi, 'p_fi'), ck(fw, 'p_fw'), ck(fg, 'p_fg'),
        ck(w0, 'p_w0'), ck(w1, 'p_w1'), ck(wo, 'p_wo'),
    )


_moe_jax_fwd_pv4_bwd.defvjp(_moe_jax_fwd_pv4_bwd_fwd, _moe_pallas_v4_bwd)


# --------------------------------------------------------------------------
# Hybrid: Pallas forward + JAX backward (tests forward kernel correctness)
# --------------------------------------------------------------------------

@functools.partial(jax.custom_vjp, nondiff_argnums=(7, 8, 9, 10, 11))
def _moe_pallas_fwd_jax_bwd(fx, fi, fw, fg, w0, w1, wo,
                              mesh, K: int, act_spec, ep_axis: str, max_tpe: int):
    """Pallas forward + JAX EP backward — isolates forward kernel correctness.

    v204: calls _moe_pallas_call directly with collective_id=0 (primal eval path).
    The _fwd rule below uses collective_id=1 (VJP fwd path) to ensure the two
    HLO instances use separate ICI channels and cannot collide.
    """
    return _moe_pallas_call(fx, fi, fw, w0, w1, wo, mesh, K, act_spec, ep_axis,
                            collective_id=0)


def _moe_pallas_fwd_jax_bwd_fwd(fx, fi, fw, fg, w0, w1, wo,
                                  mesh, K, act_spec, ep_axis, max_tpe):
    # v204: call _moe_pallas_call directly (not _moe_pallas_v4) with collective_id=1.
    # Using _moe_pallas_v4 here would: (a) go through its custom_vjp dispatch in the
    # grad context (creating nested _moe_pallas_v4_fwd call), and (b) share
    # collective_id=0 with the primal eval instance — risking ICI channel collision
    # when XLA async-fuses the two ring-reduce transfers.
    # collective_id=1 → distinct ICI channel → no collision with primal's id=0.
    out = _moe_pallas_call(fx, fi, fw, w0, w1, wo, mesh, K, act_spec, ep_axis,
                           collective_id=1)
    # Save inputs as residuals — backward re-runs JAX EP forward for VJP.
    # No checkpoint_name needed: there are no Pallas A2A calls in the backward.
    return out, (fx, fi, fw, fg, w0, w1, wo)


def _moe_pallas_fwd_jax_bwd_bwd(mesh, K, act_spec, ep_axis, max_tpe, res, g):
    """Backward: VJP through JAX EP shard_map (no Pallas kernel, no remat issues)."""
    from jax.experimental.shard_map import shard_map
    fx, fi, fw, fg, w0, w1, wo = res

    def _jax_ep(fx_, fi_, fw_, w0_, w1_, wo_):
        @functools.partial(shard_map, mesh=mesh,
                           in_specs=(act_spec, act_spec, act_spec,
                                     _moe_wi_spec(),
                                     _moe_wi_spec(),
                                     _moe_wo_spec()),
                           out_specs=act_spec, check_rep=False)
        def _fn(x, i, w, w0__, w1__, wo__):
            return _expert_mlp_ep_body(x, w0__, w1__, wo__, i, w, K, ep_axis, max_tpe)
        return _fn(fx_, fi_, fw_, w0_, w1_, wo_)

    _, vjp_fn = jax.vjp(_jax_ep, fx, fi, fw, w0, w1, wo)
    d_fx, d_fi, d_fw, d_w0, d_w1, d_wo = vjp_fn(g)
    return (d_fx, jnp.zeros_like(fi), d_fw, jnp.zeros_like(fg), d_w0, d_w1, d_wo)


_moe_pallas_fwd_jax_bwd.defvjp(_moe_pallas_fwd_jax_bwd_fwd, _moe_pallas_fwd_jax_bwd_bwd)


def expert_mlp_pallas_fwd_jax_bwd(x, wi_0, wi_1, wo, top_k_weights, top_k_indices,
                                    biased_scores, cfg: ModelConfig):
    """Pallas forward + JAX backward — tests forward kernel correctness in isolation."""
    B, S, D = x.shape
    K = cfg.K
    flat_x       = x.reshape(B * S, D)
    flat_indices = top_k_indices.reshape(B * S, K)
    flat_weights = top_k_weights.reshape(B * S, K)
    flat_scores  = biased_scores.reshape(B * S, cfg.E)

    fsdp_size = 1 if cfg.mesh is None else cfg.mesh.shape.get("fsdp", 1)
    T_fsdp    = B * S // fsdp_size
    max_tpe   = max(1, 2 * T_fsdp * K // cfg.E)
    act_spec  = P("fsdp", None)

    with jax.named_scope("moe_pallas_fwd_jax_bwd"):
        out = _moe_pallas_fwd_jax_bwd(flat_x, flat_indices, flat_weights, flat_scores,
                                       wi_0, wi_1, wo,
                                       cfg.mesh, K, act_spec, "ep", max_tpe)
    return out.reshape(B, S, D)


def expert_mlp_jax_fwd_pv4_bwd(x, wi_0, wi_1, wo, top_k_weights, top_k_indices,
                                 biased_scores, cfg: ModelConfig):
    """JAX EP MoE forward + Pallas v4 backward — fast-compile hybrid for cluster tests."""
    B, S, D = x.shape
    K = cfg.K
    flat_x       = x.reshape(B * S, D)
    flat_indices = top_k_indices.reshape(B * S, K)
    flat_weights = top_k_weights.reshape(B * S, K)
    flat_scores  = biased_scores.reshape(B * S, cfg.E)

    fsdp_size = 1 if cfg.mesh is None else cfg.mesh.shape.get("fsdp", 1)
    T_fsdp    = B * S // fsdp_size
    max_tpe   = max(1, 2 * T_fsdp * K // cfg.E)
    act_spec  = P("fsdp", None)

    with jax.named_scope("moe_jax_fwd_pv4_bwd"):
        out = _moe_jax_fwd_pv4_bwd(flat_x, flat_indices, flat_weights, flat_scores,
                                     wi_0, wi_1, wo,
                                     cfg.mesh, K, act_spec, "ep", max_tpe)
    return out.reshape(B, S, D)


def expert_mlp_pallas_v4(x, wi_0, wi_1, wo, top_k_weights, top_k_indices,
                          biased_scores, cfg: ModelConfig):
    """Pallas EP MoE: streaming-v1 forward + fused_ep_moe_bwd_v4 backward."""
    B, S, D = x.shape
    K = cfg.K
    flat_x       = x.reshape(B * S, D)
    flat_indices = top_k_indices.reshape(B * S, K)
    flat_weights = top_k_weights.reshape(B * S, K)
    flat_scores  = biased_scores.reshape(B * S, cfg.E)

    fsdp_size = 1 if cfg.mesh is None else cfg.mesh.shape.get("fsdp", 1)
    T_fsdp    = B * S // fsdp_size
    max_tpe   = max(1, 2 * T_fsdp * K // cfg.E)
    act_spec  = P("fsdp", None)

    with jax.named_scope("moe_pallas_v4"):
        out = _moe_pallas_v4(flat_x, flat_indices, flat_weights, flat_scores,
                              wi_0, wi_1, wo,
                              cfg.mesh, K, act_spec, "ep", max_tpe)
    return out.reshape(B, S, D)


# ============================================================================
# MoE layer (routing + routed experts + shared expert + aux loss)
# ============================================================================

def moe_layer(x, params, cfg: ModelConfig):
    B, S, D = x.shape

    top_k_weights, top_k_indices, scores = moe_routing(
        x, params["gate"], params["gate_bias"], cfg)

    with jax.named_scope("moe_experts"):
        if cfg.moe_backend == "jax":
            routed_out = expert_mlp_jax(
                x, params["wi_0"], params["wi_1"], params["wo"],
                top_k_weights, top_k_indices, cfg)

        elif cfg.moe_backend == "fused_ep_moe_v4":
            biased_scores = scores + params["gate_bias"].astype(jnp.float32)
            routed_out = expert_mlp_pallas_v4(
                x, params["wi_0"], params["wi_1"], params["wo"],
                top_k_weights, top_k_indices, biased_scores, cfg)

        elif cfg.moe_backend == "fused_ep_moe_v4_jax_fwd":
            # JAX forward + Pallas v4 backward — fast-compile hybrid.
            # Use this to verify backward kernel correctness on cluster without
            # the >45 min Pallas forward kernel Mosaic compilation overhead.
            biased_scores = scores + params["gate_bias"].astype(jnp.float32)
            routed_out = expert_mlp_jax_fwd_pv4_bwd(
                x, params["wi_0"], params["wi_1"], params["wo"],
                top_k_weights, top_k_indices, biased_scores, cfg)

        elif cfg.moe_backend == "pfwd_jbwd":
            # Pallas forward + JAX backward — isolates forward kernel correctness.
            # No gradient_checkpoint needed; JAX backward has no ICI remat issues.
            biased_scores = scores + params["gate_bias"].astype(jnp.float32)
            routed_out = expert_mlp_pallas_fwd_jax_bwd(
                x, params["wi_0"], params["wi_1"], params["wo"],
                top_k_weights, top_k_indices, biased_scores, cfg)

        elif cfg.moe_backend == "ragged_dot":
            # Pure JAX ragged_dot: O(T×K) per MoE layer, EP=1, any FSDP.
            # No SC gather — sorted token groups go directly into ragged_dot.
            routed_out = expert_mlp_ragged_dot(
                x, params["wi_0"], params["wi_1"], params["wo"],
                top_k_weights, top_k_indices, cfg)

        elif cfg.moe_backend == "gmm":
            # A2A dispatch GMM: targeted token routing; best when EP > K+1.
            routed_out = expert_mlp_gmm_ar(
                x, params["wi_0"], params["wi_1"], params["wo"],
                top_k_weights, top_k_indices, cfg)

        elif cfg.moe_backend == "gmm_ag":
            # AG dispatch GMM: AllGather tokens on EP + AllGather F on FSDP.
            # Best when EP < K+1 (e.g. EP=4 with K=8).
            routed_out = expert_mlp_gmm_ag(
                x, params["wi_0"], params["wi_1"], params["wo"],
                top_k_weights, top_k_indices, cfg)

        else:
            raise ValueError(f"Unknown moe_backend={cfg.moe_backend!r}. "
                             f"Valid: 'jax', 'ragged_dot', 'gmm', 'gmm_ag', "
                             f"'fused_ep_moe_v4', 'fused_ep_moe_v4_jax_fwd', 'pfwd_jbwd'")

    with jax.named_scope("shared_expert"):
        gate   = jax.nn.silu(x @ params["shared_wi_0"])  # column-parallel (TP on D_moe)
        up     = x @ params["shared_wi_1"]
        hidden = gate * up
        hidden = jax._src.ad_checkpoint.checkpoint_name(hidden, "shared_hidden")  # save+offload: 128 MB/layer
        shared_out = hidden @ params["shared_wo"]  # row-parallel (TP on D_moe, GSPMD AllReduces)

    # Load-balancing auxiliary loss (DeepSeek-V3 §2.3).
    # Compute LOCAL per-shard aux_loss (no implicit AR per layer).
    # The global AR is done ONCE at end-of-forward, not per layer.
    # Mathematically: global_aux ≈ sum_devices(local_aux_per_layer) / D, where D is
    # the number of devices. We rely on the scan accumulator + single end-AR.
    with jax.named_scope("aux_loss"):
        E = cfg.E
        if cfg.mesh is not None:
            from jax.experimental.shard_map import shard_map as _smap
            batch_ax = (_batch_ax(cfg), "ep") if cfg.mesh.shape.get("ep", 1) > 1 \
                       else _batch_ax(cfg)
            idx_spec = P(batch_ax, None, None)
            scr_spec = P(batch_ax, None, None)

            def _local_aux(idx, scr):
                oh = jax.nn.one_hot(idx, E, dtype=jnp.float32)
                Bl, Sl = idx.shape[:2]
                f_i = oh.sum(axis=(0, 1, 2)) / (Bl * Sl * cfg.K)
                P_i = scr.mean(axis=(0, 1))
                return cfg.moe_aux_loss_coeff * E * jnp.sum(f_i * P_i)

            aux_loss = _smap(_local_aux, mesh=cfg.mesh,
                             in_specs=(idx_spec, scr_spec),
                             out_specs=P(), check_rep=False)(top_k_indices, scores)
        else:
            one_hot = jax.nn.one_hot(top_k_indices, E, dtype=jnp.float32)
            f_i = one_hot.sum(axis=(0, 1, 2)) / (B * S * cfg.K)
            P_i = scores.mean(axis=(0, 1))
            aux_loss = cfg.moe_aux_loss_coeff * E * jnp.sum(f_i * P_i)

    return routed_out + shared_out, aux_loss


def moe_layer_pre_ag(x, params, ag_moe_weights, cfg: ModelConfig):
    """MoE layer for the cross-layer prefetch path (gmm_ag only).

    Same as moe_layer but routed-expert weights come from `ag_moe_weights`
    (already FSDP-AG'd via _ag_one_moe_layer with pcast/reduced semantics).
    """
    B, S, D = x.shape
    top_k_weights, top_k_indices, scores = moe_routing(
        x, params["gate"], params["gate_bias"], cfg)

    with jax.named_scope("moe_experts"):
        if cfg.moe_backend != "gmm_ag":
            raise ValueError(
                f"moe_xlayer_prefetch only implemented for moe_backend=gmm_ag, "
                f"got {cfg.moe_backend!r}")
        routed_out = expert_mlp_gmm_ag_pre(
            x, ag_moe_weights["wi_0"], ag_moe_weights["wi_1"], ag_moe_weights["wo"],
            top_k_weights, top_k_indices, cfg)

    with jax.named_scope("shared_expert"):
        gate   = jax.nn.silu(x @ params["shared_wi_0"])
        up     = x @ params["shared_wi_1"]
        hidden = gate * up
        shared_out = hidden @ params["shared_wo"]

    with jax.named_scope("aux_loss"):
        E = cfg.E
        one_hot = jax.nn.one_hot(top_k_indices, E, dtype=jnp.float32)
        f_i = one_hot.sum(axis=(0, 1, 2)) / (B * S * cfg.K)
        P_i = scores.mean(axis=(0, 1))
        aux_loss = cfg.moe_aux_loss_coeff * E * jnp.sum(f_i * P_i)

    return routed_out + shared_out, aux_loss


# ============================================================================
# Dense FFN
# ============================================================================

def dense_mlp(x, params, cfg: ModelConfig):
    with jax.named_scope("dense_mlp"):
        gate   = jax.nn.silu(x @ params["wi_gate"])  # column-parallel (TP on output dim)
        up     = x @ params["wi_up"]
        hidden = gate * up
        return hidden @ params["wo_mlp"]  # row-parallel (TP on input dim, GSPMD AllReduces)


# ============================================================================
# Parameter initialization
# ============================================================================

def _make_sharded(key, shape, mesh, spec, dt, scale=None):
    """Initialize a sharded parameter using make_array_from_callback.
    Each host generates only its own shard — avoids host OOM on large tensors.
    """
    import math as _math
    if scale is None:
        scale = 1.0 / _math.sqrt(shape[0])
    if mesh is None:
        return (random.normal(key, shape, dtype=jnp.float32) * scale).astype(dt)

    sharding = NamedSharding(mesh, spec)

    def _cb(index):
        def _start(s):
            return s.start if isinstance(s, slice) and s.start is not None else 0
        def _size(i, s):
            if isinstance(s, slice) and s.stop is not None and s.start is not None:
                return s.stop - s.start
            return shape[i]
        shard_id  = sum(_start(s) for s in index)
        shard_key = random.fold_in(key, shard_id)
        shard_shape = tuple(_size(i, s) for i, s in enumerate(index))
        return (random.normal(shard_key, shard_shape, dtype=jnp.float32) * scale).astype(dt)

    return jax.make_array_from_callback(shape, sharding, _cb)


def _make_ones(shape, mesh, spec, dt):
    if isinstance(shape, int):
        shape = (shape,)
    if mesh is None:
        return jnp.ones(shape, dtype=dt)
    sharding = NamedSharding(mesh, spec)
    def _cb(index):
        sz = tuple(
            s.stop - s.start if isinstance(s, slice) and s.stop is not None else shape[i]
            for i, s in enumerate(index))
        return jnp.ones(sz, dtype=dt)
    return jax.make_array_from_callback(shape, sharding, _cb)


def _init_attn_params(key, cfg, mesh, dt):
    res_scale = 1.0 / math.sqrt(2 * cfg.L)
    tp = 1 if mesh is None else mesh.shape.get("tp", 1)
    _col = P("fsdp", "tp") if tp > 1 else P(None, "fsdp")
    # Row-parallel output: TP shards input dim (H*d_v); GSPMD AllReduces after matmul.
    _row = P("tp", None) if tp > 1 else P("fsdp", None)
    keys = random.split(key, 6)
    return {
        "pre_attn_norm":  _make_ones(cfg.D, mesh, P(None), dt),
        "wq_a":           _make_sharded(keys[0], (cfg.D, cfg.R_q),         mesh, P("fsdp", None), dt),
        "wq_b":           _make_sharded(keys[1], (cfg.R_q, cfg.H * cfg.qk_dim), mesh, _col, dt),
        "q_norm_scale":   _make_ones(cfg.R_q, mesh, P(None), dt),
        "wkv_a":          _make_sharded(keys[2], (cfg.D, cfg.R_kv + cfg.d_rope), mesh, P("fsdp", None), dt),
        "wkv_b":          _make_sharded(keys[3], (cfg.R_kv, cfg.H * (cfg.d_nope + cfg.d_v)), mesh, _col, dt),
        "kv_norm_scale":  _make_ones(cfg.R_kv, mesh, P(None), dt),
        "w_out":          _make_sharded(keys[4], (cfg.H * cfg.d_v, cfg.D), mesh, _row, dt,
                                        scale=res_scale / math.sqrt(cfg.H * cfg.d_v)),
        "post_attn_norm": _make_ones(cfg.D, mesh, P(None), dt),
    }


def init_params(cfg: ModelConfig, key: jax.Array, mesh: Mesh | None = None) -> dict:
    """Initialize all parameters, sharded if mesh is provided.

    Returns a dict with stacked layer params (for lax.scan):
        embed:        (V, D)
        dense_layers: {param: (L_dense, ...)}
        moe_layers:   {param: (L_moe, ...)}
        final_norm:   (D,)
        output_head:  (D, V)
    """
    dt   = cfg.jax_dtype
    tp   = 1 if mesh is None else mesh.shape.get("tp", 1)
    _col = P("fsdp", "tp") if tp > 1 else P("fsdp", None)
    # Row-parallel: TP shards input dim; GSPMD AllReduces after matmul.
    _row = P("tp", None) if tp > 1 else P("fsdp", None)
    # MoE expert weight: column/row parallel — no weight AllGather needed.
    # All MoE weights stored as (E, D_moe, D) so D is the last (minor) dim.
    # With T(8,128) tiling: D (3584 or 7168) as minor → zero padding.
    # Storing as (E, D, D_moe) puts F_local=8 as minor → pads 8→128 = 93.8% waste.
    # wi transposed to (E, D, F) inside the body before ragged_dot (free in XLA).
    _moe_w = P("ep", "fsdp", "tp") if tp > 1 else _moe_wo_spec()
    res  = 1.0 / math.sqrt(2 * cfg.L)

    keys = random.split(key, cfg.L * 10 + 10)
    ki   = 0
    params = {}

    # Embedding: V × D, sharded on D across FSDP.
    # V=102400 > SC_MAX_ROWS=65536 — chunked lookup in forward() handles this.
    params["embed"] = _make_sharded(keys[ki], (cfg.V, cfg.D), mesh, P(None, "fsdp"), dt)
    ki += 1
    if mesh is not None:
        print("  embed initialized", flush=True)

    # Dense layers
    dense_list = []
    for i in range(cfg.L_dense):
        layer = _init_attn_params(keys[ki], cfg, mesh, dt); ki += 1
        layer["wi_gate"] = _make_sharded(keys[ki], (cfg.D, cfg.D_mlp), mesh, _col, dt); ki += 1
        layer["wi_up"]   = _make_sharded(keys[ki], (cfg.D, cfg.D_mlp), mesh, _col, dt); ki += 1
        layer["wo_mlp"]  = _make_sharded(keys[ki], (cfg.D_mlp, cfg.D), mesh, _row, dt,
                                          scale=res / math.sqrt(cfg.D_mlp)); ki += 1
        dense_list.append(layer)
        if mesh is not None:
            print(f"  dense layer {i}/{cfg.L_dense} initialized", flush=True)

    params["dense_layers"] = (
        jax.tree.map(lambda *a: jnp.stack(a, 0), *dense_list)
        if dense_list else {})

    # MoE layers
    moe_list = []
    for i in range(cfg.L_moe):
        layer = _init_attn_params(keys[ki], cfg, mesh, dt); ki += 1
        layer["gate"]        = _make_sharded(keys[ki], (cfg.D, cfg.E), mesh, P(None, None), dt); ki += 1
        layer["gate_bias"]   = jnp.zeros((cfg.E,), dtype=jnp.float32)
        layer["wi_0"]        = _make_sharded(keys[ki], (cfg.E, cfg.D_moe, cfg.D),
                                              mesh, _moe_w, dt,
                                              scale=1.0 / math.sqrt(cfg.D)); ki += 1
        layer["wi_1"]        = _make_sharded(keys[ki], (cfg.E, cfg.D_moe, cfg.D),
                                              mesh, _moe_w, dt,
                                              scale=1.0 / math.sqrt(cfg.D)); ki += 1
        layer["wo"]          = _make_sharded(keys[ki], (cfg.E, cfg.D_moe, cfg.D),
                                              mesh, _moe_w, dt,
                                              scale=res / math.sqrt(cfg.D_moe)); ki += 1
        layer["shared_wi_0"] = _make_sharded(keys[ki], (cfg.D, cfg.D_moe), mesh, _col, dt); ki += 1
        layer["shared_wi_1"] = _make_sharded(keys[ki], (cfg.D, cfg.D_moe), mesh, _col, dt); ki += 1
        layer["shared_wo"]   = _make_sharded(keys[ki], (cfg.D_moe, cfg.D), mesh, _row, dt,
                                              scale=res / math.sqrt(cfg.D_moe)); ki += 1
        moe_list.append(layer)
        if mesh is not None:
            print(f"  moe layer {i}/{cfg.L_moe} initialized", flush=True)

    params["moe_layers"] = (
        jax.tree.map(lambda *a: jnp.stack(a, 0), *moe_list)
        if moe_list else {})

    params["final_norm"]  = _make_ones(cfg.D, mesh, P(None), dt)
    params["output_head"] = _make_sharded(keys[ki], (cfg.D, cfg.V), mesh, P("fsdp", None), dt)

    if mesh is not None:
        print("  all layers initialized", flush=True)
    return params


# ============================================================================
# Layer bodies (used inside lax.scan)
# ============================================================================

def _dense_layer_body(x, layer_params, positions, cfg):
    h = rms_norm(x, layer_params["pre_attn_norm"], cfg.norm_eps)
    x = x + mla_attention(h, layer_params, positions, cfg, use_cp=False)
    h = rms_norm(x, layer_params["post_attn_norm"], cfg.norm_eps)
    return x + dense_mlp(h, layer_params, cfg)


def _moe_layer_body(x, layer_params, positions, cfg):
    # Name the layer input for selective host offloading via checkpoint policy.
    x = jax._src.ad_checkpoint.checkpoint_name(x, "moe_layer_input")
    # Re-annotate sharding inside scan to prevent GSPMD from losing placement.
    # CP on: P(batch, ep, None); CP off + EP>1: P((batch,ep), None, None).
    carry_spec = _carry_spec(cfg)
    if carry_spec is not None:
        x = jax.lax.with_sharding_constraint(
            x, NamedSharding(cfg.mesh, carry_spec))
    h = rms_norm(x, layer_params["pre_attn_norm"], cfg.norm_eps)
    x = x + mla_attention(h, layer_params, positions, cfg)
    h = rms_norm(x, layer_params["post_attn_norm"], cfg.norm_eps)
    mlp_out, aux = moe_layer(h, layer_params, cfg)
    return x + mlp_out, aux


def _moe_layer_body_pre_ag(x, layer_params, ag_moe_weights, positions, cfg):
    """Cross-layer prefetch variant: MoE weights arrive already FSDP-AG'd."""
    x = jax._src.ad_checkpoint.checkpoint_name(x, "moe_layer_input")
    carry_spec = _carry_spec(cfg)
    if carry_spec is not None:
        x = jax.lax.with_sharding_constraint(
            x, NamedSharding(cfg.mesh, carry_spec))
    h = rms_norm(x, layer_params["pre_attn_norm"], cfg.norm_eps)
    x = x + mla_attention(h, layer_params, positions, cfg)
    h = rms_norm(x, layer_params["post_attn_norm"], cfg.norm_eps)
    mlp_out, aux = moe_layer_pre_ag(h, layer_params, ag_moe_weights, cfg)
    return x + mlp_out, aux


# ============================================================================
# Forward pass
# ============================================================================

def forward(params, tokens, cfg: ModelConfig, return_final_x: bool = False):
    """Full forward pass. Returns (logits, total_aux_loss).

    Uses lax.scan for O(1) compilation (one compiled function reused L times).
    Dense layers scanned separately from MoE layers.
    MoE scan chunked to cap backward carry HBM: 30 GB / chunk.

    return_final_x: if True, return (x_final_normed, aux_loss) without the
    output_head matmul.  Used by compute_loss for chunked vocab CE.
    """
    B, S = tokens.shape
    positions = jnp.broadcast_to(jnp.arange(S), (B, S))

    # Embedding lookup — chunked to satisfy SC gather constraints on v7x:
    #   (1) flatten 2-D tokens to 1-D (SC requires 1-D indices for BF16)
    #   (2) split vocab into ≤65536-row chunks (SC uses 16-bit row indices)
    # V=102400 → 2 chunks: [0, 65536) and [65536, 102400).
    with jax.named_scope("embedding"):
        SC_MAX_V = 65536
        flat_tok = tokens.reshape(-1)
        V_total  = params["embed"].shape[0]
        D_emb    = params["embed"].shape[1]
        x_flat   = jnp.zeros((flat_tok.shape[0], D_emb), dtype=params["embed"].dtype)
        v_start  = 0
        while v_start < V_total:
            v_end   = min(v_start + SC_MAX_V, V_total)
            chunk   = params["embed"][v_start:v_end]
            in_c    = (flat_tok >= v_start) & (flat_tok < v_end)
            loc_idx = jnp.where(in_c, flat_tok - v_start, 0)
            x_flat  = x_flat + jnp.where(in_c[:, None], chunk[loc_idx], 0)
            v_start = v_end
        x = x_flat.reshape(B, S, D_emb)

    # Sharding constraint: batch by FSDP (and EP if CP off + EP>1), seq by EP (if CP).
    if cfg.mesh is not None:
        carry_spec = _carry_spec(cfg)
        x = jax.lax.with_sharding_constraint(
            x, NamedSharding(cfg.mesh, carry_spec))
        seq_ax = _seq_ax(cfg)
        if seq_ax is not None:
            positions = jax.lax.with_sharding_constraint(
                positions, NamedSharding(cfg.mesh, P(_batch_ax(cfg), seq_ax)))

    # Dense layers — always shard batch jointly by (fsdp, ep), full sequence.
    # With CP: reshard from P(fsdp, ep, None) to P((fsdp,ep), None, None).
    # No CP + EP>1: already in P((fsdp,ep), None, None) layout.
    if cfg.L_dense > 0:
        if cfg.mesh is not None:
            ep_size = cfg.mesh.shape.get("ep", 1)
            if ep_size > 1:
                dense_spec   = P((_batch_ax(cfg), "ep"), None, None)
                pos_spec     = P((_batch_ax(cfg), "ep"), None)
            else:
                dense_spec   = P(_batch_ax(cfg), None, None)
                pos_spec     = P(_batch_ax(cfg), None)
            x = jax.lax.with_sharding_constraint(
                x, NamedSharding(cfg.mesh, dense_spec))
            positions_dense = jax.lax.with_sharding_constraint(
                positions, NamedSharding(cfg.mesh, pos_spec))
        else:
            positions_dense = positions

        def _dense_scan_fn(x, lp):
            with jax.named_scope("dense_layer"):
                fn = functools.partial(_dense_layer_body, positions=positions_dense, cfg=cfg)
                x = jax.checkpoint(fn)(x, lp) if cfg.gradient_checkpoint else fn(x, lp)
            return x, None
        with jax.named_scope("dense_scan"):
            x, _ = jax.lax.scan(_dense_scan_fn, x, params["dense_layers"])

        # Reshard back to the carry layout for MoE layers
        if cfg.mesh is not None:
            x = jax.lax.with_sharding_constraint(
                x, NamedSharding(cfg.mesh, _carry_spec(cfg)))

    # MoE layers
    total_aux = jnp.float32(0.0)
    if cfg.L_moe > 0:
        # Host offload: only when scan carry would exceed HBM.
        # B_l = GBS / (FSDP * TP).  Carry = L * B_l * S * D * 2 bytes.
        # At FSDP=512 TP=1: B_l=8 → 25 GB carry → fits in HBM.
        # At FSDP=256 TP=2: B_l=16 → 50 GB carry → needs host offload.
        fsdp_size = 1 if cfg.mesh is None else cfg.mesh.shape.get("fsdp", 1)
        ep_size = 1 if cfg.mesh is None else cfg.mesh.shape.get("ep", 1)
        # Per-device tokens in the carry: (B*S)/(fsdp*ep) regardless of CP layout —
        # CP shards S on ep, no-CP shards B jointly on fsdp×ep. TP doesn't divide tokens.
        B_l = B * S // (fsdp_size * ep_size) if cfg.mesh else B * S
        # carry is (B_l, D) per layer in bf16; stack L layers across scan VJP
        carry_bytes = cfg.L_moe * B_l * cfg.D * 2  # bf16
        need_offload = carry_bytes > 20 * 1024**3  # >20 GB → offload (was 30; v311 OOM at 27 GB just below trigger)

        if need_offload:
            # v315: revert to just moe_layer_input. The small offloads (q_a,
            # kv_a, shared_hidden) had per-DUS overhead ~25-35 ms × 58 layers
            # = 1.5-2 s of pure-overhead per offload — way more than the cheap
            # matmul recompute they would save. Only large activations
            # (moe_layer_input, attn_proj_out) have favorable DUS:save ratio.
            _ckpt_policy = jax.checkpoint_policies.save_and_offload_only_these_names(
                names_which_can_be_saved=(),
                names_which_can_be_offloaded=("moe_layer_input",),
                offload_src="device",
                offload_dst="pinned_host",
            )
        else:
            _ckpt_policy = None

        if cfg.moe_xlayer_prefetch and cfg.moe_backend == "gmm_ag":
            # Cross-layer FSDP weight prefetch via pcast/reduced AG.
            # 3-tuple carry: (x, aux, ws_ag). ws_ag is "reduced" along {dp,fsdp},
            # so the carry's gradient stays as a per-shard (small) tensor — no
            # 109 GB stash like v294 attempted with plain AG.
            first_lp = jax.tree.map(lambda w: w[0], params["moe_layers"])
            first_ag = _ag_one_moe_layer(first_lp, cfg.mesh)

            def _shift_left(w):
                return jnp.concatenate([w[1:], jnp.zeros_like(w[:1])], axis=0)
            shifted_lps = jax.tree.map(_shift_left, params["moe_layers"])

            def _moe_scan_fn_pf(carry, scan_in):
                x, aux, ws_ag = carry
                cur_lp, next_lp = scan_in
                # Issue prefetch AG for next layer's MoE weights — depends only
                # on next_lp, so XLA can schedule concurrently with current
                # layer's compute.
                next_ws_ag = _ag_one_moe_layer(next_lp, cfg.mesh)
                fn = functools.partial(
                    _moe_layer_body_pre_ag, positions=positions, cfg=cfg)
                if cfg.gradient_checkpoint:
                    x_new, a = jax.checkpoint(
                        fn, policy=_ckpt_policy, prevent_cse=False)(
                            x, cur_lp, ws_ag)
                else:
                    x_new, a = fn(x, cur_lp, ws_ag)
                return (x_new, aux + a, next_ws_ag), None

            with jax.named_scope("moe_scan_pf"):
                (x, total_aux, _), _ = jax.lax.scan(
                    _moe_scan_fn_pf,
                    (x, total_aux, first_ag),
                    (params["moe_layers"], shifted_lps))
        else:
            def _moe_scan_fn(carry, lp):
                x, aux = carry
                fn = functools.partial(_moe_layer_body, positions=positions, cfg=cfg)
                if cfg.gradient_checkpoint:
                    x_new, a = jax.checkpoint(
                        fn, policy=_ckpt_policy, prevent_cse=False)(x, lp)
                else:
                    x_new, a = fn(x, lp)
                return (x_new, aux + a), None
            with jax.named_scope("moe_scan"):
                (x, total_aux), _ = jax.lax.scan(
                    _moe_scan_fn, (x, total_aux), params["moe_layers"])

        # Single global psum on the accumulated per-shard local aux_loss.
        # moe_layer now returns LOCAL aux_loss (no per-layer AR); we do ONE AR
        # at end of forward instead of 58 (saves ~16.5 ms × 57 = ~940 ms/step).
        if cfg.mesh is not None:
            from jax.experimental.shard_map import shard_map as _smap
            ep_size = cfg.mesh.shape.get("ep", 1)
            batch_ax = (_batch_ax(cfg), "ep") if ep_size > 1 else _batch_ax(cfg)
            with jax.named_scope("aux_loss_global_psum"):
                total_aux = _smap(
                    lambda a: jax.lax.psum(a, batch_ax),
                    mesh=cfg.mesh, in_specs=P(), out_specs=P(),
                    check_rep=False)(total_aux)

    # Output head
    with jax.named_scope("output_head"):
        x = rms_norm(x, params["final_norm"], cfg.norm_eps)
        if return_final_x:
            return x, total_aux  # caller handles chunked logit/loss computation
        logits = x @ params["output_head"]

    return logits, total_aux


def forward_with_layer_stats(params, tokens, cfg: ModelConfig):
    """Debug-only forward: runs layers in a Python loop (no scan) and
    collects per-layer residual-norm + per-MoE-layer aux contribution.
    SLOWER and consumes more compile time than `forward()`, but lets you
    diff our forward against a reference (e.g. MaxText) one layer at a
    time — exactly what was missing during the Tier 3 calibration arc.

    Returns (x_final_normed, layer_stats) where layer_stats is a list of
    dicts (one per dense + MoE layer), each with:
      - layer_idx (int, 0..L-1)
      - kind ('dense' | 'moe')
      - resid_norm (scalar f32) — L2 norm of x[0, last_real_pos, :]
      - aux_local (scalar f32, MoE only) — per-layer routing aux loss
    """
    B, S = tokens.shape
    positions = jnp.broadcast_to(jnp.arange(S), (B, S))

    # Re-use the embedding logic from forward() — keep this in sync.
    with jax.named_scope("embedding"):
        SC_MAX_V = 65536
        flat_tok = tokens.reshape(-1)
        V_total  = params["embed"].shape[0]
        D_emb    = params["embed"].shape[1]
        x_flat   = jnp.zeros((flat_tok.shape[0], D_emb), dtype=params["embed"].dtype)
        v_start  = 0
        while v_start < V_total:
            v_end   = min(v_start + SC_MAX_V, V_total)
            chunk   = params["embed"][v_start:v_end]
            in_c    = (flat_tok >= v_start) & (flat_tok < v_end)
            loc_idx = jnp.where(in_c, flat_tok - v_start, 0)
            x_flat  = x_flat + jnp.where(in_c[:, None], chunk[loc_idx], 0)
            v_start = v_end
        x = x_flat.reshape(B, S, D_emb)

    def _resid_norm(x_):
        # L2 norm of last position of batch row 0, in fp32.
        return jnp.linalg.norm(x_[0, -1, :].astype(jnp.float32))

    stats: list[dict] = []

    # Dense layers
    for i in range(cfg.L_dense):
        layer_p = jax.tree.map(lambda w: w[i], params["dense_layers"])
        x = _dense_layer_body(x, layer_p, positions, cfg)
        stats.append({
            "layer_idx": i, "kind": "dense",
            "resid_norm": _resid_norm(x),
        })

    # MoE layers
    for i in range(cfg.L_moe):
        layer_p = jax.tree.map(lambda w: w[i], params["moe_layers"])
        x, aux = _moe_layer_body(x, layer_p, positions, cfg)
        stats.append({
            "layer_idx": cfg.L_dense + i, "kind": "moe",
            "resid_norm": _resid_norm(x),
            "aux_local": aux.astype(jnp.float32),
        })

    # Final norm
    x = rms_norm(x, params["final_norm"], cfg.norm_eps)
    return x, stats


# ============================================================================
# Loss — chunked vocab CE with explicit custom_vjp (v245).
#
# History of failures at full scale (FSDP=16, TP=2, GBS=4096):
#   v241: monolithic x @ output_head → bf16[256,2048,102400] = 107 GB OOM.
#   v242: Python for-loop (25 iters) → unrolled in HLO → 1.05 TB OOM.
#   v243: lax.scan without checkpoint → VJP stacks lg as f32[25,B,S,4096]
#         = 200 GB, hits Jellyfish int32 DMA limit (3.35 B words > INT32_MAX).
#   v244: lax.scan + jax.checkpoint → XLA unrolls bwd → 1.04 TB OOM.
#
# Fix: custom_vjp with two explicit lax.scan while-loops.
#   Forward scan: one pass, accumulates (max, sum_exp, tgt_logit) → loss.
#     Saves log_sum_exp for bwd. 8 GB/step, O(1) compile-time. ✓
#   Backward scan: one pass, recomputes logits, accumulates d_x_pred +
#     d_output_head. ~28 GB/step (temporary), carry ~4 GB. O(1). ✓
#   No scan VJP derivation → no residual stacking, no checkpoint unrolling.
# ============================================================================

@functools.partial(jax.custom_vjp, nondiff_argnums=(3, 4))
def _vocab_ce(x_pred, output_head, targets, n_chunks: int, V_CHUNK: int):
    """Chunked vocab CE — primal (non-differentiating path)."""
    B_l, S_l = x_pred.shape[:2]
    D = x_pred.shape[-1]
    NEG_INF = jnp.finfo(jnp.float32).min

    def _fwd_body(carry, v_idx):
        mx, se, tl = carry
        vs = v_idx * V_CHUNK
        w  = jax.lax.dynamic_slice(output_head, (0, vs), (D, V_CHUNK))
        lg = (x_pred @ w).astype(jnp.float32)
        cm = lg.max(-1);  nm = jnp.maximum(mx, cm)
        se = se * jnp.exp(mx - nm) + jnp.exp(lg - nm[:, :, None]).sum(-1)
        in_v = (targets >= vs) & (targets < vs + V_CHUNK)
        oh   = jax.nn.one_hot(jnp.where(in_v, targets - vs, 0), V_CHUNK, dtype=jnp.float32)
        tl  += (lg * oh).sum(-1) * in_v.astype(jnp.float32)
        return (nm, se, tl), None

    (mx, se, tl), _ = jax.lax.scan(
        _fwd_body,
        (jnp.full((B_l, S_l), NEG_INF, jnp.float32),
         jnp.zeros((B_l, S_l), jnp.float32),
         jnp.zeros((B_l, S_l), jnp.float32)),
        jnp.arange(n_chunks, dtype=jnp.int32))
    return jnp.mean(jnp.log(se) + mx - tl)


def _vocab_ce_fwd(x_pred, output_head, targets, n_chunks, V_CHUNK):
    """Forward rule: compute loss, save log_sum_exp for backward."""
    B_l, S_l = x_pred.shape[:2]
    D = x_pred.shape[-1]
    NEG_INF = jnp.finfo(jnp.float32).min

    def _body(carry, v_idx):
        mx, se, tl = carry
        vs = v_idx * V_CHUNK
        w  = jax.lax.dynamic_slice(output_head, (0, vs), (D, V_CHUNK))
        lg = (x_pred @ w).astype(jnp.float32)
        cm = lg.max(-1);  nm = jnp.maximum(mx, cm)
        se = se * jnp.exp(mx - nm) + jnp.exp(lg - nm[:, :, None]).sum(-1)
        in_v = (targets >= vs) & (targets < vs + V_CHUNK)
        oh   = jax.nn.one_hot(jnp.where(in_v, targets - vs, 0), V_CHUNK, dtype=jnp.float32)
        tl  += (lg * oh).sum(-1) * in_v.astype(jnp.float32)
        return (nm, se, tl), None

    (mx, se, tl), _ = jax.lax.scan(
        _body,
        (jnp.full((B_l, S_l), NEG_INF, jnp.float32),
         jnp.zeros((B_l, S_l), jnp.float32),
         jnp.zeros((B_l, S_l), jnp.float32)),
        jnp.arange(n_chunks, dtype=jnp.int32))
    log_sum_exp = jnp.log(se) + mx          # (B_l, S_l) f32 — needed in bwd
    loss = jnp.mean(log_sum_exp - tl)
    return loss, (x_pred, output_head, log_sum_exp, targets)


def _vocab_ce_bwd(n_chunks, V_CHUNK, residuals, g):
    """Backward rule: explicit forward scan computing d_x_pred and d_output_head.
    Carry: d_x_pred (~3.75 GB/dev f32) + d_output_head (~175 MB/dev f32).
    NOT stacked — custom_vjp bwd is not differentiated further in training.
    ~28 GB/step temporary (lg, p_v, d_lg). O(1) compile-time memory. ✓
    """
    x_pred, output_head, log_sum_exp, targets = residuals
    B_l, S_l = x_pred.shape[:2]
    D = x_pred.shape[-1]
    scale = g / jnp.asarray(B_l * S_l, jnp.float32)   # grad of mean: 1/N

    # In-graph diagnostics for bwd-NaN bisection. Gated by env var because
    # jax.debug.print is NOT pure side-effect — it constrains XLA op ordering
    # via side-effect tokens, which can flip the scheduler to expose latent
    # NaN-producing patterns (v341m mini-grad finding). Set
    # VCE_BWD_FINITE_CHECK=1 to enable.
    import os as _os
    if _os.environ.get("VCE_BWD_FINITE_CHECK", "").lower() in ("1", "true", "yes"):
        jax.debug.print(
            "[vce-bwd] inputs: x_pred nan={a} max={b} | LSE nan={c} max={d} min={e} | g={g}",
            a=jnp.isnan(x_pred).any(),
            b=jnp.max(jnp.abs(x_pred.astype(jnp.float32))),
            c=jnp.isnan(log_sum_exp).any(),
            d=jnp.max(log_sum_exp),
            e=jnp.min(log_sum_exp),
            g=g)

    def _bwd_body(carry, v_idx):
        d_x, d_w = carry
        vs  = v_idx * V_CHUNK
        w   = jax.lax.dynamic_slice(output_head, (0, vs), (D, V_CHUNK))
        lg  = (x_pred @ w).astype(jnp.float32)         # (B_l, S_l, V_CHUNK)
        pv  = jnp.exp(lg - log_sum_exp[:, :, None])    # softmax this chunk
        in_v = (targets >= vs) & (targets < vs + V_CHUNK)
        oh   = jax.nn.one_hot(jnp.where(in_v, targets - vs, 0), V_CHUNK, dtype=jnp.float32)
        d_lg = (pv - oh * in_v[:, :, None].astype(jnp.float32)) * scale
        # Accumulate d_x_pred: (B_l, S_l, V_CHUNK) × (V_CHUNK, D) → (B_l, S_l, D)
        d_x = d_x + (d_lg.astype(w.dtype) @ w.T).astype(d_x.dtype)
        # Accumulate d_output_head[:, vs:vs+V_CHUNK]: einsum over batch/seq dims.
        # GSPMD emits all-reduce over FSDP+TP for the batch/seq contraction.
        dw  = jnp.einsum('bsd,bsv->dv', x_pred.astype(jnp.float32), d_lg)
        d_w = jax.lax.dynamic_update_slice(d_w, dw.astype(d_w.dtype), (0, vs))
        return (d_x, d_w), None

    # Initialise d_x with same sharding as x_pred (GSPMD infers from zeros_like).
    # Initialise d_w in f32 matching output_head shape; cast output to param dtype.
    d_x0 = jnp.zeros_like(x_pred).astype(jnp.float32)
    d_w0 = jnp.zeros_like(output_head).astype(jnp.float32)
    (d_x_pred_f32, d_w_f32), _ = jax.lax.scan(
        _bwd_body, (d_x0, d_w0), jnp.arange(n_chunks, dtype=jnp.int32))

    if _os.environ.get("VCE_BWD_FINITE_CHECK", "").lower() in ("1", "true", "yes"):
        jax.debug.print(
            "[vce-bwd] outputs: d_x nan={a} max={b} | d_w nan={c} max={d}",
            a=jnp.isnan(d_x_pred_f32).any(),
            b=jnp.max(jnp.abs(d_x_pred_f32)),
            c=jnp.isnan(d_w_f32).any(),
            d=jnp.max(jnp.abs(d_w_f32)))

    return d_x_pred_f32.astype(x_pred.dtype), d_w_f32.astype(output_head.dtype), None


_vocab_ce.defvjp(_vocab_ce_fwd, _vocab_ce_bwd)


def compute_loss(params, tokens, cfg: ModelConfig):
    """Cross-entropy loss + MoE auxiliary load-balancing loss.

    Returns (total_loss, aux) where aux = {'lm_loss': ..., 'aux_loss': ...} so
    callers using `jax.value_and_grad(compute_loss, has_aux=True)` can report
    the LM and MoE-balance components separately. Critical for diagnosing
    whether a high reported loss reflects a forward-pass issue (LM term high)
    or just the expected pre-training MoE imbalance penalty (aux term high).
    """
    x_final, aux_loss = forward(params, tokens, cfg, return_final_x=True)
    x_pred  = x_final[:, :-1]                          # (B_l, S_l, D)
    B_l, S_l = x_pred.shape[:2]
    targets = tokens[:, 1:1+S_l].astype(jnp.int32)    # (B_l, S_l)
    V_CHUNK  = 4096
    n_chunks = cfg.V // V_CHUNK                         # 102400 // 4096 = 25
    lm_loss = _vocab_ce(x_pred, params["output_head"], targets, n_chunks, V_CHUNK)
    total = lm_loss + aux_loss
    return total, {"lm_loss": lm_loss, "aux_loss": aux_loss}
