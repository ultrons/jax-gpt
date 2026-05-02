"""Load a MaxText DSv3 orbax checkpoint into our DSv3 param structure.

Manifest-driven restore. The MaxText DSv3-FSDP checkpoint at
gs://mlperf-6-submission/ckpt0424-fsdp/0/items was saved on a 12-axis JAX
mesh where ALL axes have size 1 except `fsdp` (size 16). Every per-tensor
PartitionSpec entry is either None or a list of MaxText axis names; only
entries whose bundle contains `fsdp` are actually sharded — everything else
is effectively replicated.

So at restore time:
  1. Read orbax metadata to get per-tensor save-time NamedShardingMetadata
     (shape + dtype + partition_spec).
  2. Build a `(fsdp=N, replicate=K)` restore mesh covering all available
     devices (N | save_dim_gcd, K = n_devs / N).
  3. For each tensor, convert MaxText partition_spec entries → 'fsdp' if the
     bundle contains 'fsdp', else None.
  4. Restore using abstract_target with these simplified shardings.
  5. Map MaxText's tree → our params dict, with axis-moves & reshapes done
     in jnp (kept on TPU, not pulled to host), then jax.device_put each
     tensor onto our serving mesh.

MaxText param-name mapping (→ our dict keys):
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

Per dense FFN:
    mlp/wi_0/kernel  →  wi_gate
    mlp/wi_1/kernel  →  wi_up
    mlp/wo/kernel    →  wo_mlp

Per MoE:
    moe_layers/.../MoeBlock_0/gate/kernel    →  gate
    moe_layers/.../MoeBlock_0/gate/bias      →  gate_bias
    moe_layers/.../MoeBlock_0/wi_0           →  wi_0   (E, L, D, D_moe) → (L, E, D, D_moe)
    moe_layers/.../MoeBlock_0/wi_1           →  wi_1
    moe_layers/.../MoeBlock_0/wo             →  wo     (E, L, D_moe, D) → (L, E, D_moe, D)
    moe_layers/.../shared_experts/wi_0/kernel →  shared_wi_0
    moe_layers/.../shared_experts/wi_1/kernel →  shared_wi_1
    moe_layers/.../shared_experts/wo/kernel   →  shared_wo

NOT loaded:
    mtp_block/*  — multi-token-prediction head, not modeled in our DSv3.
    step         — training step counter.

VOCAB: MaxText uses V=129280. Caller MUST set cfg.V=129280 before forward,
otherwise the chunked _vocab_ce will silently miss the trailing dims.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from .model import ModelConfig


# Set of MaxText mesh axis names that had non-trivial size at save time.
# The DSv3-FSDP ckpt only used `fsdp` (size 16) — verified via manifest probe.
_MAXTEXT_SHARDED_AXES = {"fsdp"}


def _maxtext_spec_to_restore_spec(maxtext_spec):
    """Convert a MaxText `partition_spec` (tuple of None | str | list[str])
    into a JAX PartitionSpec on our 2-axis restore mesh: each entry is None
    (replicated) or 'fsdp' (sharded along the restore mesh's fsdp axis)."""
    if maxtext_spec is None:
        return None
    out = []
    for entry in maxtext_spec:
        if entry is None:
            out.append(None)
        elif isinstance(entry, str):
            out.append("fsdp" if entry in _MAXTEXT_SHARDED_AXES else None)
        elif isinstance(entry, (list, tuple)):
            out.append("fsdp" if any(a in _MAXTEXT_SHARDED_AXES for a in entry) else None)
        else:
            out.append(None)
    return tuple(out)


def _build_abstract_target_from_manifest(metadata, restore_mesh: Mesh):
    """Walk metadata tree; for each leaf with shape/dtype/sharding, build a
    `jax.ShapeDtypeStruct` with the simplified NamedSharding on restore_mesh."""

    def recurse(t):
        if hasattr(t, "shape") and hasattr(t, "dtype"):
            shape = tuple(t.shape) if t.shape is not None else ()
            sh = getattr(t, "sharding", None)
            ps = getattr(sh, "partition_spec", None) if sh is not None else None
            simple = _maxtext_spec_to_restore_spec(ps)
            if simple is None:
                ns = NamedSharding(restore_mesh, P())
            else:
                ns = NamedSharding(restore_mesh, P(*simple))
            return jax.ShapeDtypeStruct(shape, t.dtype, sharding=ns)
        if isinstance(t, dict):
            return {k: recurse(v) for k, v in t.items()}
        return t

    return recurse(metadata)


def _build_restore_mesh():
    """Build a `(fsdp, replicate)` restore mesh covering ALL devices.

    fsdp_n must divide every fsdp-sharded dim. Manifest probe of the DSv3
    MaxText ckpt confirmed the smallest fsdp-sharded dim is 512 (wkv_b dim
    0); gcd of all observed sharded dims (512, 1536, 2048-not-sharded,
    7168, 14336, 18432, 129280) is 256. We use `min(128, n_devs)`:
      - 128 divides every observed sharded dim (smallest 512 → 4 elem/dev);
      - matches our serving fsdp axis size, so cross-mesh re-shard becomes
        a near-no-op for fsdp-sharded tensors (just an axis-name rename);
      - per-device load 1.34 TB / 128 ≈ 10.5 GB on v7x (101 GB HBM).
    """
    n = jax.device_count()
    fsdp_n = min(128, n)
    while n % fsdp_n != 0 and fsdp_n > 1:
        fsdp_n -= 1
    replicate_n = n // fsdp_n
    devs = np.array(jax.devices()).reshape(fsdp_n, replicate_n)
    mesh = Mesh(devs, ("fsdp", "replicate"))
    return mesh, fsdp_n, replicate_n


def _shard(arr, mesh: Mesh, spec: P):
    """Move/re-shard `arr` onto `mesh` with the given PartitionSpec."""
    return jax.device_put(arr, NamedSharding(mesh, spec))


def _move_layer_axis_to_front(arr, layer_axis: int):
    """Pull the given axis to position 0 on TPU (no host transfer)."""
    return jnp.moveaxis(arr, layer_axis, 0)


def _flatten_heads(arr):
    """Flatten the trailing (n_heads, head_dim) into one axis."""
    s = arr.shape
    return arr.reshape(*s[:-2], s[-2] * s[-1])


# DSv3 671B head dims (hardcoded — this loader is DSv3-671B-specific anyway).
_D_NOPE = 128
_D_ROPE = 64
_R_KV = 512


def _interleaved_to_concat_per_head(arr_per_head):
    """Convert per-head qk dim from MaxText interleaved RoPE layout
    [r0,i0,r1,i1,...] to our split-half layout [r0,r1,...,i0,i1,...].

    arr_per_head shape: (..., H, qk_dim) where qk_dim = d_nope + d_rope.
    Only the trailing d_rope dims are permuted; nope dims pass through.
    """
    nope = arr_per_head[..., :_D_NOPE]                               # (..., H, d_nope)
    rope = arr_per_head[..., _D_NOPE:]                               # (..., H, d_rope) interleaved
    rope_concat = jnp.concatenate([rope[..., 0::2], rope[..., 1::2]], axis=-1)
    return jnp.concatenate([nope, rope_concat], axis=-1)             # (..., H, qk_dim)


def _interleaved_to_concat_kv_a(wkv_a):
    """Permute wkv_a's trailing d_rope dims (k_rope) from interleaved → concat.
    wkv_a shape: (..., R_kv + d_rope) — no head dim (k_rope shared across heads).
    """
    kv = wkv_a[..., :_R_KV]
    rope = wkv_a[..., _R_KV:]
    rope_concat = jnp.concatenate([rope[..., 0::2], rope[..., 1::2]], axis=-1)
    return jnp.concatenate([kv, rope_concat], axis=-1)


def _per_layer_mla(d, mesh: Mesh) -> dict:
    """Pull MLA tensors (move L axis to 0, flatten heads).

    Critical RoPE-layout fix: MaxText DSv3 MLA trains with interleaved RoPE
    (rope_interleave=True; layers/attention_mla.py:669, embeddings.py:964).
    Our model's _apply_rope uses split-half/concatenated layout. So we permute
    the trailing d_rope=64 dims of wq_b (per head) and wkv_a (no head) from
    [r0,i0,r1,i1,...] → [r0,r1,...,i0,i1,...] at load time.
    """
    sa = d["self_attention"]
    pre   = _move_layer_axis_to_front(d["pre_self_attention_layer_norm"]["scale"],  layer_axis=1)
    post  = _move_layer_axis_to_front(d["post_self_attention_layer_norm"]["scale"], layer_axis=1)
    wq_a  = _move_layer_axis_to_front(sa["wq_a"]["kernel"], layer_axis=1)            # (D, L, q_lora) → (L, D, q_lora)
    wq_b  = _move_layer_axis_to_front(sa["wq_b"]["kernel"], layer_axis=1)            # (q_lora, L, H, 192) → (L, q_lora, H, 192)
    wq_b  = _interleaved_to_concat_per_head(wq_b)                                     # permute rope dims per head
    wq_b  = _flatten_heads(wq_b)                                                      # → (L, q_lora, H*192)
    qn    = _move_layer_axis_to_front(sa["q_norm"]["scale"], layer_axis=1)
    wkv_a = _move_layer_axis_to_front(sa["wkv_a"]["kernel"], layer_axis=1)            # (D, L, R_kv+d_rope) → (L, D, 576)
    wkv_a = _interleaved_to_concat_kv_a(wkv_a)                                        # permute trailing 64 dims
    wkv_b = _move_layer_axis_to_front(sa["wkv_b"]["kernel"], layer_axis=1)
    wkv_b = _flatten_heads(wkv_b)                                                     # wkv_b has no rope dims, no permute
    kvn   = _move_layer_axis_to_front(sa["kv_norm"]["scale"], layer_axis=1)
    out   = _move_layer_axis_to_front(sa["out"]["kernel"], layer_axis=1)              # (H, L, head_dim, D) → (L, H, head_dim, D)
    out   = out.reshape(out.shape[0], out.shape[1] * out.shape[2], out.shape[3])      # (L, H*head_dim, D)
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
    # init_params layout (model.py:2798-2800):
    #   wi_gate, wi_up:  (L, D=7168, D_mlp=18432)
    #   wo_mlp:          (L, D_mlp=18432, D=7168)
    # MaxText after moveaxis(L→0): MATCHES exactly. No swap needed.
    # FSDP shards dim 1 of each (per init_params._col / _row = P("fsdp", None)).
    wi0 = _move_layer_axis_to_front(d["mlp"]["wi_0"]["kernel"], layer_axis=1)  # (L, D, D_mlp)
    wi1 = _move_layer_axis_to_front(d["mlp"]["wi_1"]["kernel"], layer_axis=1)
    wo  = _move_layer_axis_to_front(d["mlp"]["wo"]["kernel"],  layer_axis=1)   # (L, D_mlp, D)
    return {
        "wi_gate": _shard(wi0, mesh, P(None, "fsdp", None)),
        "wi_up":   _shard(wi1, mesh, P(None, "fsdp", None)),
        "wo_mlp":  _shard(wo,  mesh, P(None, "fsdp", None)),
    }


def _per_layer_moe(d, mesh: Mesh) -> dict:
    moe = d["DeepSeekMoeBlock_0"]
    block = moe["MoeBlock_0"]
    gate_kernel = _move_layer_axis_to_front(block["gate"]["kernel"], layer_axis=1)
    gate_bias   = _move_layer_axis_to_front(block["gate"]["bias"],   layer_axis=1)
    # init_params layout (model.py:2816-2822): ALL THREE MoE weights are
    # (E, D_moe=2048, D=7168) — D_moe first, D last. Sharded P("ep", "fsdp", None).
    # MaxText:
    #   wi_0/1: (E, L, D=7168, D_moe=2048)  — D before D_moe.  → SWAP needed.
    #   wo:     (E, L, D_moe=2048, D=7168)  — D_moe before D. → no swap.
    wi_0 = jnp.moveaxis(block["wi_0"], 1, 0).swapaxes(-1, -2)  # (E,L,D,D_moe)→(L,E,D_moe,D)
    wi_1 = jnp.moveaxis(block["wi_1"], 1, 0).swapaxes(-1, -2)
    wo   = jnp.moveaxis(block["wo"],   1, 0)                   # (E,L,D_moe,D)→(L,E,D_moe,D)
    # Shared experts (3-D, per-layer):
    #   shared_wi_0/1: (L, D=7168, D_moe=2048)  ← matches init_params (D, D_moe).
    #   shared_wo:     (L, D_moe=2048, D=7168)  ← matches init_params (D_moe, D).
    # MaxText after moveaxis(L→0) MATCHES — no swap.
    shared = moe["shared_experts"]
    shared_wi0 = _move_layer_axis_to_front(shared["wi_0"]["kernel"], layer_axis=1)
    shared_wi1 = _move_layer_axis_to_front(shared["wi_1"]["kernel"], layer_axis=1)
    shared_wo  = _move_layer_axis_to_front(shared["wo"]["kernel"],   layer_axis=1)
    return {
        "gate":        _shard(gate_kernel, mesh, P(None, None, None)),
        "gate_bias":   _shard(gate_bias,   mesh, P(None, None)),
        "wi_0":        _shard(wi_0, mesh, P(None, "ep", "fsdp", None)),
        "wi_1":        _shard(wi_1, mesh, P(None, "ep", "fsdp", None)),
        "wo":          _shard(wo,   mesh, P(None, "ep", "fsdp", None)),
        "shared_wi_0": _shard(shared_wi0, mesh, P(None, "fsdp", None)),
        "shared_wi_1": _shard(shared_wi1, mesh, P(None, "fsdp", None)),
        "shared_wo":   _shard(shared_wo,  mesh, P(None, "fsdp", None)),
    }


def load_maxtext_dsv3(ckpt_path: str, cfg: ModelConfig, mesh: Mesh) -> dict:
    """Restore a MaxText DSv3 orbax checkpoint into our DSv3 params dict.

    Args:
        ckpt_path: gs:// or local path to the orbax 'items' directory.
        cfg:       Our DSv3 ModelConfig (caller MUST set cfg.V=129280).
        mesh:      Target SERVING mesh (axes: dp / fsdp / ep / tp).

    Returns:
        params dict matching what load_hf_weights produces — directly usable
        by `forward(params, tokens, cfg)`.
    """
    import orbax.checkpoint as ocp

    print(f"Restoring MaxText orbax checkpoint from {ckpt_path} ...")

    # 1. Read manifest
    md_raw = ocp.PyTreeCheckpointer().metadata(ckpt_path)
    print(f"  metadata type: {type(md_raw).__name__}")
    if isinstance(md_raw, dict):
        metadata = md_raw
    elif hasattr(md_raw, "item_metadata"):
        im = md_raw.item_metadata
        metadata = im if isinstance(im, dict) else getattr(im, "tree", im)
    elif hasattr(md_raw, "tree"):
        metadata = md_raw.tree
    else:
        attrs = [a for a in dir(md_raw) if not a.startswith("_")]
        raise RuntimeError(
            f"Cannot extract tree from metadata of type "
            f"{type(md_raw).__name__}. Public attrs: {attrs}")
    print(f"  metadata top-level keys: {list(metadata.keys())[:5]}")

    # 2. Build restore mesh
    restore_mesh, fsdp_n, replicate_n = _build_restore_mesh()
    print(f"  restore mesh: fsdp={fsdp_n}, replicate={replicate_n} "
          f"(over {jax.device_count()} devices)")

    # 3. Build abstract_target from manifest
    print("  building abstract target from manifest shardings ...")
    abstract = _build_abstract_target_from_manifest(metadata, restore_mesh)

    # 4. Restore — use the v0 API path that MaxText uses
    # (src/maxtext/common/checkpointing.py:_load_full_state_from_path,
    # the `else` branch). The plain PyTreeCheckpointer().restore(args=
    # PyTreeRestore(item=abstract)) path does NOT respect the abstract's
    # sharding for tensorstore chunk reads — it falls back to save-time
    # sharding, which leaves most hosts with no addressable shards when
    # restoring from a 16-fsdp save onto our 512-core cluster.
    #
    # The fix: explicit `restore_args` with ArrayRestoreArgs(sharding=...)
    # per leaf, on a PyTreeCheckpointHandler with use_ocdbt=True.
    print("  building per-leaf ArrayRestoreArgs ...")

    def _make_restore_args(x):
        if isinstance(x, jax.ShapeDtypeStruct) and x.sharding is not None:
            return ocp.type_handlers.ArrayRestoreArgs(sharding=x.sharding)
        return ocp.RestoreArgs()

    restore_args = jax.tree_util.tree_map(
        _make_restore_args, abstract,
        is_leaf=lambda x: isinstance(x, jax.ShapeDtypeStruct))

    handler = ocp.PyTreeCheckpointHandler(use_ocdbt=True)
    print("  restoring (Checkpointer + per-leaf ArrayRestoreArgs) ...")
    raw = ocp.Checkpointer(handler).restore(
        ckpt_path, abstract, restore_args=restore_args,
    )
    print("  restore complete; mapping MaxText tree → our params ...")

    # Free opt_state (Adam m,v ≈ 2-3x params = ~3 TB) and the unused MTP head
    # before we start re-sharding params (which briefly holds source +
    # destination copies). gc.collect() is needed for JAX HBM to actually be
    # released — `del` alone doesn't trigger XLA dealloc.
    import gc
    if "opt_state" in raw:
        print("  freeing opt_state ...")
        del raw["opt_state"]
    if "mtp_block" in raw.get("params", {}).get("params", {}):
        print("  freeing mtp_block ...")
        del raw["params"]["params"]["mtp_block"]
    gc.collect()

    # 5. Map MaxText tree → our params, re-sharded onto serving mesh
    p = raw["params"]["params"]
    decoder = p["decoder"]
    embed       = p["token_embedder"]["embedding"]    # (V, D)
    final_norm  = decoder["decoder_norm"]["scale"]    # (D,)
    output_head = decoder["logits_dense"]["kernel"]   # (D, V)

    params = {
        "embed":       _shard(embed,       mesh, P(None, "fsdp")),
        "final_norm":  _shard(final_norm,  mesh, P(None)),
        "output_head": _shard(output_head, mesh, P("fsdp", None)),
    }

    print("  mapping dense_layers (3 layers) ...")
    dense = decoder["dense_layers"]
    params["dense_layers"] = {
        **_per_layer_mla(dense, mesh),
        **_per_layer_dense_ffn(dense, mesh),
    }

    print("  mapping moe_layers (58 layers) ...")
    moe = decoder["moe_layers"]
    params["moe_layers"] = {
        **_per_layer_mla(moe, mesh),
        **_per_layer_moe(moe, mesh),
    }

    print("  (skipping mtp_block, step — not modeled in our DSv3)")
    return params
