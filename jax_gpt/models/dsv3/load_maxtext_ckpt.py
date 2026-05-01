"""Load a MaxText DSv3 orbax checkpoint into our DSv3 param structure.

The MaxText checkpoint at gs://mlperf-6-submission/ckpt0424-fsdp/0/items
is the reference for the convergence baseline (first-step llm_loss=5.337).
This loader maps MaxText's parameter naming convention into the dict
shape that load_hf_weights produces, so the rest of our forward path
works unchanged.

Mapping summary (MaxText → ours):
    token_embedder/embedding             →  embed
    decoder/decoder_norm/scale           →  final_norm
    decoder/logits_dense/kernel          →  output_head
    decoder/dense_layers/* (L=3 axis)    →  dense_layers/* (L axis-0 stack)
    decoder/moe_layers/* (L=58 axis)     →  moe_layers/* (L axis-0 stack)

Per-layer MLA (move L axis to front, flatten n_heads × head_dim):
    pre_self_attention_layer_norm/scale  →  pre_attn_norm
    post_self_attention_layer_norm/scale →  post_attn_norm
    self_attention/wq_a/kernel           →  wq_a
    self_attention/wq_b/kernel           →  wq_b   (flatten n_heads, head_dim)
    self_attention/q_norm/scale          →  q_norm_scale
    self_attention/wkv_a/kernel          →  wkv_a
    self_attention/wkv_b/kernel          →  wkv_b  (flatten n_heads, head_dim)
    self_attention/kv_norm/scale         →  kv_norm_scale
    self_attention/out/kernel            →  w_out  (flatten n_heads, head_dim)

Per dense layer FFN:
    mlp/wi_0/kernel  →  wi_gate
    mlp/wi_1/kernel  →  wi_up
    mlp/wo/kernel    →  wo_mlp

Per MoE layer:
    moe_layers/.../MoeBlock_0/gate/kernel    →  gate
    moe_layers/.../MoeBlock_0/gate/bias      →  gate_bias
    moe_layers/.../MoeBlock_0/wi_0           →  wi_0
    moe_layers/.../MoeBlock_0/wi_1           →  wi_1
    moe_layers/.../MoeBlock_0/wo             →  wo
    moe_layers/.../shared_experts/wi_0/kernel →  shared_wi_0
    moe_layers/.../shared_experts/wi_1/kernel →  shared_wi_1
    moe_layers/.../shared_experts/wo/kernel   →  shared_wo

NOT loaded (we don't use these in our model):
    mtp_block/*                          (multi-token-prediction head)
    /step                                (training step counter)

KNOWN VOCAB MISMATCH: MaxText / HF DSv3 use vocab=129280 but our
cfg.V=102400. This loader passes through MaxText's vocab dimension
unchanged; the caller MUST set cfg.V=129280 BEFORE forward, or the
chunked _vocab_ce will silently miss the trailing 26880 vocab dims.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from .model import ModelConfig


def _shard(arr, mesh: Mesh, spec: P) -> jax.Array:
    """Move a numpy array onto the JAX mesh with the given PartitionSpec."""
    return jax.device_put(arr, NamedSharding(mesh, spec))


def _move_layer_axis_to_front(arr, layer_axis: int):
    """Pull the given axis to position 0. axis < 0 → from-end indexing."""
    return np.moveaxis(arr, layer_axis, 0)


def _flatten_heads(arr):
    """Flatten the (n_heads, head_dim) dims into one. Assumes those are the
    last two dims of the input."""
    *leading, n_heads, head_dim = arr.shape
    return arr.reshape(*leading, n_heads * head_dim)


def _per_layer_mla(d, layer_axis: int, mesh: Mesh) -> dict:
    """Pull MLA tensors out of a dense_layers / moe_layers subtree.

    `d` is the MaxText subtree (a dict). `layer_axis` is the axis where the
    L dimension lives in each tensor (typically 1 for dense_layers/moe_layers
    after the leading non-L dim).
    """
    sa = d["self_attention"]

    # The L axis lives at different positions per tensor. We move it to 0.
    pre  = _move_layer_axis_to_front(d["pre_self_attention_layer_norm"]["scale"],  layer_axis=1)
    post = _move_layer_axis_to_front(d["post_self_attention_layer_norm"]["scale"], layer_axis=1)
    wq_a = _move_layer_axis_to_front(sa["wq_a"]["kernel"], layer_axis=1)            # (D, L, q_lora) -> (L, D, q_lora)
    wq_b = _move_layer_axis_to_front(sa["wq_b"]["kernel"], layer_axis=1)            # (q_lora, L, H, 192) -> (L, q_lora, H, 192)
    wq_b = _flatten_heads(wq_b)                                                      # -> (L, q_lora, H*192)
    qn   = _move_layer_axis_to_front(sa["q_norm"]["scale"], layer_axis=1)           # (q_lora, L) -> (L, q_lora)
    wkv_a = _move_layer_axis_to_front(sa["wkv_a"]["kernel"], layer_axis=1)
    wkv_b = _move_layer_axis_to_front(sa["wkv_b"]["kernel"], layer_axis=1)           # (kv_lora, L, H, 256)
    wkv_b = _flatten_heads(wkv_b)                                                    # -> (L, kv_lora, H*256)
    kvn   = _move_layer_axis_to_front(sa["kv_norm"]["scale"], layer_axis=1)
    out  = _move_layer_axis_to_front(sa["out"]["kernel"], layer_axis=1)              # (H, L, head_dim, D)
    out  = out.reshape(out.shape[0], out.shape[1] * out.shape[2], out.shape[3])      # (L, H*head_dim, D)

    return {
        "pre_attn_norm":  _shard(pre,   mesh, P(None, None)),
        "post_attn_norm": _shard(post,  mesh, P(None, None)),
        "wq_a":           _shard(wq_a,  mesh, P(None, "fsdp", None)),
        "wq_b":           _shard(wq_b,  mesh, P(None, None, "fsdp")),
        "q_norm_scale":   _shard(qn,    mesh, P(None, None)),
        "wkv_a":          _shard(wkv_a, mesh, P(None, "fsdp", None)),
        "wkv_b":          _shard(wkv_b, mesh, P(None, None, "fsdp")),
        "kv_norm_scale":  _shard(kvn,   mesh, P(None, None)),
        "w_out":          _shard(out,   mesh, P(None, "fsdp", None)),
    }


def _per_layer_dense_ffn(d, mesh: Mesh) -> dict:
    """Dense FFN tensors (only present in the dense_layers subtree)."""
    wi0 = _move_layer_axis_to_front(d["mlp"]["wi_0"]["kernel"], layer_axis=1)  # (D, L, ffn) -> (L, D, ffn)
    wi1 = _move_layer_axis_to_front(d["mlp"]["wi_1"]["kernel"], layer_axis=1)
    wo  = _move_layer_axis_to_front(d["mlp"]["wo"]["kernel"],  layer_axis=1)   # (ffn, L, D) -> (L, ffn, D)
    return {
        "wi_gate": _shard(wi0, mesh, P(None, "fsdp", None)),
        "wi_up":   _shard(wi1, mesh, P(None, "fsdp", None)),
        "wo_mlp":  _shard(wo,  mesh, P(None, None, "fsdp")),
    }


def _per_layer_moe(d, mesh: Mesh) -> dict:
    """MoE-block tensors. d = decoder/moe_layers."""
    moe = d["DeepSeekMoeBlock_0"]
    block = moe["MoeBlock_0"]

    # Router gate + bias (move L axis to front)
    gate_kernel = _move_layer_axis_to_front(block["gate"]["kernel"], layer_axis=1)  # (D, L, E) -> (L, D, E)
    gate_bias   = _move_layer_axis_to_front(block["gate"]["bias"],   layer_axis=1)  # (E, L) -> (L, E)

    # Expert weights (already (E, L, D, D_moe) — swap axes to (L, E, D, D_moe))
    wi_0 = np.moveaxis(block["wi_0"], 1, 0)  # (E, L, D, D_moe) -> (L, E, D, D_moe)
    wi_1 = np.moveaxis(block["wi_1"], 1, 0)
    wo   = np.moveaxis(block["wo"],   1, 0)  # (E, L, D_moe, D) -> (L, E, D_moe, D)

    # Shared expert (D, L, D_moe) -> (L, D, D_moe)
    shared = moe["shared_experts"]
    shared_wi0 = _move_layer_axis_to_front(shared["wi_0"]["kernel"], layer_axis=1)
    shared_wi1 = _move_layer_axis_to_front(shared["wi_1"]["kernel"], layer_axis=1)
    shared_wo  = _move_layer_axis_to_front(shared["wo"]["kernel"],   layer_axis=1)

    return {
        "gate":       _shard(gate_kernel, mesh, P(None, None, None)),
        "gate_bias":  _shard(gate_bias,   mesh, P(None, None)),
        "wi_0":       _shard(wi_0, mesh, P(None, "ep", "fsdp", None)),
        "wi_1":       _shard(wi_1, mesh, P(None, "ep", "fsdp", None)),
        "wo":         _shard(wo,   mesh, P(None, "ep", None, "fsdp")),
        "shared_wi_0": _shard(shared_wi0, mesh, P(None, "fsdp", None)),
        "shared_wi_1": _shard(shared_wi1, mesh, P(None, "fsdp", None)),
        "shared_wo":   _shard(shared_wo,  mesh, P(None, None, "fsdp")),
    }


def _spec_for_path(path_parts: tuple, shape: tuple, mesh_axes: set) -> P:
    """Sharding rule: shard the most-shardable dim of each tensor along an
    appropriate mesh axis. Conservative — replicate small / hard-to-divide
    tensors. Goal is restore-fits-in-memory, NOT optimal layout.

    path_parts is a tuple of nested dict keys (lowercased fragments).
    """
    last = path_parts[-1] if path_parts else ""

    def divisible_axes(dim: int) -> list[str]:
        out = []
        for ax in ("fsdp", "ep", "tp"):
            if ax in mesh_axes and dim % {"fsdp": 128, "ep": 4, "tp": 1}.get(ax, 1) == 0:
                out.append(ax)
        return out

    nd = len(shape)
    # Norms / biases / scales: small, replicate.
    if last == "scale" or last == "bias" or sum(shape) < 16384:
        return P(*([None] * nd))

    # Embedding (V, D) — shard V on fsdp.
    if path_parts[-2:] == ("token_embedder", "embedding"):
        return P("fsdp", None)

    # Output head (D, V) — shard V on fsdp (V=129280 / 128 = 1010 ✓).
    if "logits_dense" in path_parts:
        return P(None, "fsdp")

    # MoE expert weights (E, L, D, D_moe) or (E, L, D_moe, D) — shard E on ep.
    # Match by leading dim 256 = num_experts.
    if shape and shape[0] == 256 and "moe_layers" in path_parts:
        # E on ep, last D-axis on fsdp.
        spec = ["ep", None, None, None][:nd]
        if nd >= 3 and shape[-2] % 128 == 0:
            spec[-2] = "fsdp"
        elif nd >= 1 and shape[-1] % 128 == 0:
            spec[-1] = "fsdp"
        return P(*spec)

    # Default: shard the FIRST dim divisible by fsdp on fsdp; else replicate.
    for i, d in enumerate(shape):
        if d >= 1024 and d % 128 == 0:
            spec = [None] * nd
            spec[i] = "fsdp"
            return P(*spec)
    return P(*([None] * nd))


def _build_abstract_target(metadata, mesh: Mesh):
    """Walk the metadata tree and build an abstract_target with per-tensor
    sharding suitable for multi-host restore."""
    mesh_axes = set(mesh.axis_names)

    def _recurse(t, path_parts: tuple):
        if hasattr(t, "shape") and hasattr(t, "dtype"):
            shape = tuple(t.shape)
            spec = _spec_for_path(path_parts, shape, mesh_axes)
            return jax.ShapeDtypeStruct(
                shape, t.dtype, sharding=NamedSharding(mesh, spec))
        if isinstance(t, dict):
            return {k: _recurse(v, path_parts + (k,)) for k, v in t.items()}
        return t  # int / scalar metadata

    return _recurse(metadata, ())


def load_maxtext_dsv3(ckpt_path: str, cfg: ModelConfig, mesh: Mesh) -> dict:
    """Restore a MaxText DSv3 orbax checkpoint into our params dict.

    Args:
        ckpt_path: gs:// or local path to the orbax 'items' directory.
        cfg:       Our DSv3 ModelConfig (must have cfg.V == 129280).
        mesh:      Target JAX mesh.

    Returns:
        params dict matching what load_hf_weights produces — directly usable
        by `forward(params, tokens, cfg)`.
    """
    import orbax.checkpoint as ocp

    print(f"Restoring MaxText orbax checkpoint from {ckpt_path} ...")
    print("  reading metadata ...")
    md_raw = ocp.PyTreeCheckpointer().metadata(ckpt_path)
    print(f"  metadata type: {type(md_raw).__name__}")
    # Newer orbax (>=0.6.x) returns StepMetadata wrapping the actual tree at
    # .metadata or .tree; older versions return a dict directly.
    if isinstance(md_raw, dict):
        metadata = md_raw
    elif hasattr(md_raw, "metadata") and isinstance(md_raw.metadata, dict):
        metadata = md_raw.metadata
    elif hasattr(md_raw, "tree") and isinstance(md_raw.tree, dict):
        metadata = md_raw.tree
    else:
        # Print diagnostics so the next iteration knows what to do.
        attrs = [a for a in dir(md_raw) if not a.startswith("_")]
        raise RuntimeError(
            f"Don't know how to extract tree from metadata of type "
            f"{type(md_raw).__name__}. Public attrs: {attrs}")
    print(f"  metadata tree top-level keys: {list(metadata.keys())[:5]}")
    print("  building abstract target with per-tensor sharding ...")
    abstract = _build_abstract_target(metadata, mesh)
    print("  restoring (multi-host) ...")
    raw = ocp.PyTreeCheckpointer().restore(
        ckpt_path,
        args=ocp.args.PyTreeRestore(item=abstract),
    )

    # MaxText nests under /params/params; peel the wrapper.
    p = raw["params"]["params"]
    decoder = p["decoder"]

    # Top-level: embed, final_norm, output_head.
    embed = p["token_embedder"]["embedding"]           # (V=129280, D=7168)
    final_norm = decoder["decoder_norm"]["scale"]      # (D,)
    output_head = decoder["logits_dense"]["kernel"]    # (D, V)

    params = {
        "embed":       _shard(embed,        mesh, P(None, "fsdp")),
        "final_norm":  _shard(final_norm,   mesh, P(None)),
        "output_head": _shard(output_head,  mesh, P("fsdp", None)),
    }

    # Dense layers (3) — MLA + dense FFN.
    print("  Mapping dense_layers ...")
    dense_subtree = decoder["dense_layers"]
    dense_mla = _per_layer_mla(dense_subtree, layer_axis=1, mesh=mesh)
    dense_ffn = _per_layer_dense_ffn(dense_subtree, mesh=mesh)
    params["dense_layers"] = {**dense_mla, **dense_ffn}

    # MoE layers (58) — MLA + MoE block.
    print("  Mapping moe_layers ...")
    moe_subtree = decoder["moe_layers"]
    moe_mla = _per_layer_mla(moe_subtree, layer_axis=1, mesh=mesh)
    moe_blk = _per_layer_moe(moe_subtree, mesh=mesh)
    params["moe_layers"] = {**moe_mla, **moe_blk}

    # MTP block — not yet wired into our forward; skip.
    print("  (skipping mtp_block — not modeled in our DSv3)")

    return params
