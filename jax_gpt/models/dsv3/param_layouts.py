"""Single source of truth for DSv3 parameter shapes + shardings.

The Tier 3 t-series spent most of its iterations chasing parameter-layout
bugs (wi_0 axis order, RoPE interleave permute, etc.). Each one only
surfaced as a `dot_general contracting dims mismatch` deep inside a 5-min
JIT compile. Catching them at load time turns "wait 5 min for the cluster
to crash" into "fail in 100 ms with a precise error".

This module declares the canonical layout of every DSv3 parameter — shape
(per layer), PartitionSpec on the serving mesh, dtype. External-checkpoint
loaders (load_maxtext_ckpt, load_hf_weights, ...) call `validate()` after
mapping their format into our params dict. Mismatches raise immediately.

NOT (yet) consumed by `init_params` — that path is production-tested and
out of scope for this refactor. The intent is for init_params and this
registry to drift together when shapes/specs change; a future change can
have init_params iterate this registry to enforce alignment.
"""
from __future__ import annotations

import dataclasses
from typing import Iterable

import jax
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P


@dataclasses.dataclass(frozen=True)
class ParamLayout:
    """Canonical layout of one DSv3 parameter.

    `shape` is the per-tensor shape AS STORED IN params[...]. For per-layer
    parameters this includes the leading `L` axis (e.g. dense_layers/wi_gate
    has shape `(L_dense, D, D_mlp)`).

    `spec` is the JAX PartitionSpec applied via NamedSharding on the
    serving mesh. For per-layer params the leading L axis is replicated
    (None) because lax.scan iterates over it.
    """
    name: str
    shape: tuple[int, ...]
    spec: P
    dtype: str = "bfloat16"
    notes: str = ""


def dsv3_layouts(cfg) -> dict[str, ParamLayout]:
    """Return the full DSv3 parameter layout registry for a given config.

    Keys are dotted paths matching how params dict is indexed:
      - "embed", "final_norm", "output_head"
      - "dense_layers.<param>"  — leading axis is L_dense
      - "moe_layers.<param>"    — leading axis is L_moe
    """
    L_dense = cfg.L_dense
    L_moe = cfg.L_moe
    D = cfg.D
    V = cfg.V
    H = cfg.H
    d_v = cfg.d_v
    qk_dim = cfg.qk_dim
    R_q = cfg.R_q
    R_kv = cfg.R_kv
    d_rope = cfg.d_rope
    D_mlp = cfg.D_mlp
    D_moe = cfg.D_moe
    E = cfg.E

    L = {}

    # ── Top-level ─────────────────────────────────────────────────────────
    L["embed"]       = ParamLayout("embed",       (V, D),     P(None, "fsdp"))
    L["final_norm"]  = ParamLayout("final_norm",  (D,),       P(None))
    L["output_head"] = ParamLayout("output_head", (D, V),     P("fsdp", None))

    # ── Per-layer attention (MLA) ─────────────────────────────────────────
    # Same layout for dense_layers and moe_layers; differ only in L axis.
    def _mla(prefix: str, L_axis: int):
        L[f"{prefix}.pre_attn_norm"]  = ParamLayout(
            f"{prefix}.pre_attn_norm",  (L_axis, D),                   P(None, None))
        L[f"{prefix}.post_attn_norm"] = ParamLayout(
            f"{prefix}.post_attn_norm", (L_axis, D),                   P(None, None))
        L[f"{prefix}.q_norm_scale"]   = ParamLayout(
            f"{prefix}.q_norm_scale",   (L_axis, R_q),                 P(None, None))
        L[f"{prefix}.kv_norm_scale"]  = ParamLayout(
            f"{prefix}.kv_norm_scale",  (L_axis, R_kv),                P(None, None))
        L[f"{prefix}.wq_a"]           = ParamLayout(
            f"{prefix}.wq_a",           (L_axis, D, R_q),              P(None, "fsdp", None))
        L[f"{prefix}.wq_b"]           = ParamLayout(
            f"{prefix}.wq_b",           (L_axis, R_q, H * qk_dim),     P(None, None, "fsdp"),
            notes="last dim is H heads * qk_dim, H-outer flatten; "
                  "DSv3 trains with interleaved RoPE → loaders must permute "
                  "the trailing d_rope dims of each head from interleaved "
                  "[r0,i0,...] to concatenated [r0,r1,...,i0,i1,...].")
        L[f"{prefix}.wkv_a"]          = ParamLayout(
            f"{prefix}.wkv_a",          (L_axis, D, R_kv + d_rope),    P(None, "fsdp", None),
            notes="last d_rope dims are k_rope (no head dim); same "
                  "interleaved→concat permute required as wq_b.")
        L[f"{prefix}.wkv_b"]          = ParamLayout(
            f"{prefix}.wkv_b",          (L_axis, R_kv, H * (cfg.d_nope + d_v)), P(None, None, "fsdp"))
        L[f"{prefix}.w_out"]          = ParamLayout(
            f"{prefix}.w_out",          (L_axis, H * d_v, D),          P(None, "fsdp", None))

    if L_dense > 0:
        _mla("dense_layers", L_dense)
        # Dense FFN
        L["dense_layers.wi_gate"] = ParamLayout(
            "dense_layers.wi_gate", (L_dense, D, D_mlp),     P(None, "fsdp", None))
        L["dense_layers.wi_up"]   = ParamLayout(
            "dense_layers.wi_up",   (L_dense, D, D_mlp),     P(None, "fsdp", None))
        L["dense_layers.wo_mlp"]  = ParamLayout(
            "dense_layers.wo_mlp",  (L_dense, D_mlp, D),     P(None, "fsdp", None))

    if L_moe > 0:
        _mla("moe_layers", L_moe)
        # MoE router + experts. CANONICAL LAYOUT (init_params model.py:2826):
        # ALL THREE expert weights are (E, D_moe, D) — D is the minor dim.
        L["moe_layers.gate"]        = ParamLayout(
            "moe_layers.gate",        (L_moe, D, E),         P(None, None, None))
        L["moe_layers.gate_bias"]   = ParamLayout(
            "moe_layers.gate_bias",   (L_moe, E),            P(None, None),
            dtype="float32",
            notes="init_params uses jnp.zeros; loaders may load bf16 — both OK.")
        L["moe_layers.wi_0"]        = ParamLayout(
            "moe_layers.wi_0",        (L_moe, E, D_moe, D),  P(None, "ep", "fsdp", None))
        L["moe_layers.wi_1"]        = ParamLayout(
            "moe_layers.wi_1",        (L_moe, E, D_moe, D),  P(None, "ep", "fsdp", None))
        L["moe_layers.wo"]          = ParamLayout(
            "moe_layers.wo",          (L_moe, E, D_moe, D),  P(None, "ep", "fsdp", None))
        # Shared expert (3-D, no E dim)
        L["moe_layers.shared_wi_0"] = ParamLayout(
            "moe_layers.shared_wi_0", (L_moe, D, D_moe),     P(None, "fsdp", None))
        L["moe_layers.shared_wi_1"] = ParamLayout(
            "moe_layers.shared_wi_1", (L_moe, D, D_moe),     P(None, "fsdp", None))
        L["moe_layers.shared_wo"]   = ParamLayout(
            "moe_layers.shared_wo",   (L_moe, D_moe, D),     P(None, "fsdp", None))

    return L


def _walk(params: dict, prefix: str = "") -> Iterable[tuple[str, jax.Array]]:
    """Yield ('top.sub.leaf', array) for every leaf in a nested dict."""
    for k, v in params.items():
        path = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            yield from _walk(v, path)
        else:
            yield path, v


def validate(params: dict, cfg, *, strict: bool = True) -> list[str]:
    """Validate `params` against dsv3_layouts(cfg).

    Reports (and optionally raises on) shape, dtype, or registry mismatches.
    Returns list of human-readable error strings (empty = all good).

    Use right after `load_maxtext_dsv3()` / `load_hf_weights()`. A loader
    bug that produces e.g. (E, D, D_moe) instead of (E, D_moe, D) for wi_0
    will be caught here in 100ms instead of 5 minutes into JIT compile.
    """
    expected = dsv3_layouts(cfg)
    errors: list[str] = []

    # 1. Every loaded leaf must be in the registry.
    for path, arr in _walk(params):
        if path not in expected:
            errors.append(
                f"unexpected param '{path}' (shape {tuple(arr.shape)}) — "
                f"not in dsv3_layouts(cfg). Loader produced extra key.")
            continue
        layout = expected[path]
        # 2. Shape must match exactly.
        if tuple(arr.shape) != layout.shape:
            errors.append(
                f"'{path}' shape mismatch: got {tuple(arr.shape)}, "
                f"expected {layout.shape}. Notes: {layout.notes or '(none)'}")
        # 3. dtype is a soft check — bf16/f32 are interchangeable for many
        #    norms; we only warn if drastically off (e.g. f64 / int).
        got_dtype = str(arr.dtype)
        if got_dtype not in ("bfloat16", "float32", "float16"):
            errors.append(
                f"'{path}' suspicious dtype: {got_dtype} "
                f"(expected bfloat16 / float32 family).")

    # 4. Every registry entry must be loaded.
    loaded = {p for p, _ in _walk(params)}
    for path in expected:
        if path not in loaded:
            errors.append(f"missing param '{path}' — loader did not produce it.")

    if errors and strict:
        raise ValueError(
            f"ParamLayout validation failed ({len(errors)} error(s)):\n  - "
            + "\n  - ".join(errors))
    return errors
