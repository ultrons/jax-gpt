"""Layer-by-layer weight offloading for the full Qwen3.5-397B model.

Enables full-model correctness testing on hardware where total HBM < model size.
Only one group's weights (4 layers: 3 DeltaNet + 1 GQA) reside in HBM at a time.

Memory per device with TP=4:
  - One group params (BF16): ~52 GB total / 4 devices = ~13 GB/device
  - Activations + cache: < 1 GB/device
  - Total: ~14 GB/device, well within 96 GB TPU v5p HBM

Usage:
    from jax_gpt.models.qwen35.offload import forward_offload
    from jax_gpt.models.qwen35.sharding import make_mesh, AXIS_RULES_B

    mesh = make_mesh(n_devices=4)
    logits, _ = forward_offload(tokens, cfg, '/mnt/disks/tpu_data/qwen3.5-397b',
                                 mesh, AXIS_RULES_B)
"""

from __future__ import annotations

import json
from pathlib import Path

import jax
import jax.numpy as jnp

from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from jax_gpt.models.qwen35.block import group_forward, HAS_RPA
from jax_gpt.models.qwen35.cache import HybridCache, init_cache
from jax_gpt.models.qwen35.config import Qwen35Config
from jax_gpt.models.qwen35.primitives import precompute_rope_freqs, rms_norm
from jax_gpt.models.qwen35.sharding import (
    AXIS_RULES_B,
    _flatten_path,
    _pad_spec_to_ndim,
    _param_logical_axes,
    _resolve_spec,
)

if HAS_RPA:
    from jax_gpt.models.qwen35.block import group_forward_rpa
    from jax_gpt.models.qwen35.paged_cache import (
        PagedGQACache,
        contiguous_to_paged,
        make_decode_metadata,
    )
    from tpu_inference.kernels.ragged_paged_attention.v3.util import cdiv

# HF uses this prefix for the language model weights in the multimodal checkpoint.
# (weight_loader.py incorrectly uses 'model.layers.X' — only works for mini test models.)
_LM_PREFIX = 'model.language_model'


# ---------------------------------------------------------------------------
# Safetensors lazy loader
# ---------------------------------------------------------------------------

class _SafetensorsIndex:
    """Lazy per-tensor reader backed by a sharded safetensors checkpoint.

    Opens shard files on demand and keeps handles open for reuse within a
    group load.  Call close() between groups to release file handles.
    """

    def __init__(self, model_dir: str | Path):
        self.model_dir = Path(model_dir)
        with open(self.model_dir / 'model.safetensors.index.json') as f:
            idx = json.load(f)
        self._weight_map: dict[str, str] = idx['weight_map']
        self._handles: dict[str, object] = {}

    def get(self, key: str) -> jax.Array:
        """Return a JAX array for the given tensor key.

        Uses DLPack to transfer from PyTorch without going through numpy,
        which is needed because numpy doesn't natively support bfloat16.
        """
        if key not in self._weight_map:
            raise KeyError(f"Tensor not found in index: {key!r}")
        shard = self._weight_map[key]
        path = str(self.model_dir / shard)
        if path not in self._handles:
            from safetensors.torch import load_file
            self._handles[path] = load_file(path)
        tensor = self._handles[path][key]
        # DLPack: zero-copy bfloat16 transfer torch -> JAX (CPU)
        return jax.dlpack.from_dlpack(tensor.detach())

    def close(self):
        """Release all open shard handles."""
        self._handles.clear()

    def __contains__(self, key: str) -> bool:
        return key in self._weight_map


# ---------------------------------------------------------------------------
# Per-layer param loaders
# ---------------------------------------------------------------------------

def _j(arr: jax.Array) -> jax.Array:
    """Identity — already a JAX array from DLPack."""
    return arr


def _jT(arr: jax.Array) -> jax.Array:
    """Transpose: HF stores (out, in); we use (in, out)."""
    return jnp.asarray(arr).T


def _load_moe(st: _SafetensorsIndex, prefix: str, cfg: Qwen35Config) -> dict:
    I = cfg.moe_intermediate_size

    gate_up = st.get(f'{prefix}.mlp.experts.gate_up_proj')   # (E, 2*I, D) JAX array
    gate_proj = jnp.transpose(gate_up[:, :I, :], (0, 2, 1))  # (E, D, I)
    up_proj   = jnp.transpose(gate_up[:, I:, :], (0, 2, 1))  # (E, D, I)

    down_hf = st.get(f'{prefix}.mlp.experts.down_proj')       # (E, D, I)
    down_proj = jnp.transpose(down_hf, (0, 2, 1))             # (E, I, D)

    return {
        'gate_weight':               _jT(st.get(f'{prefix}.mlp.gate.weight')),
        'gate_proj':                 gate_proj,
        'up_proj':                   up_proj,
        'down_proj':                 down_proj,
        'shared_gate_proj':          _jT(st.get(f'{prefix}.mlp.shared_expert.gate_proj.weight')),
        'shared_up_proj':            _jT(st.get(f'{prefix}.mlp.shared_expert.up_proj.weight')),
        'shared_down_proj':          _jT(st.get(f'{prefix}.mlp.shared_expert.down_proj.weight')),
        'shared_expert_gate_weight': _jT(st.get(f'{prefix}.mlp.shared_expert_gate.weight')),
    }


def _load_delta_layer(st: _SafetensorsIndex, layer_idx: int, cfg: Qwen35Config) -> dict:
    prefix = f'{_LM_PREFIX}.layers.{layer_idx}'
    ap = f'{prefix}.linear_attn'

    attn = {
        'in_proj_qkv':  _jT(st.get(f'{ap}.in_proj_qkv.weight')),
        'in_proj_z':    _jT(st.get(f'{ap}.in_proj_z.weight')),
        'in_proj_b':    _jT(st.get(f'{ap}.in_proj_b.weight')),
        'in_proj_a':    _jT(st.get(f'{ap}.in_proj_a.weight')),
        # conv1d stores (conv_dim, 1, kernel_size) — squeeze the singleton dim
        'conv_weight':  _j(st.get(f'{ap}.conv1d.weight')).squeeze(1),
        # A_log and norm.weight are float32 in the checkpoint
        'A_log':        _j(st.get(f'{ap}.A_log')),
        'dt_bias':      _j(st.get(f'{ap}.dt_bias')),
        'norm_weight':  _j(st.get(f'{ap}.norm.weight')),
        'out_proj':     _jT(st.get(f'{ap}.out_proj.weight')),
    }
    return {
        'attn_norm': _j(st.get(f'{prefix}.input_layernorm.weight')),
        'attn':      attn,
        'moe_norm':  _j(st.get(f'{prefix}.post_attention_layernorm.weight')),
        'moe':       _load_moe(st, prefix, cfg),
    }


def _load_gqa_layer(st: _SafetensorsIndex, layer_idx: int, cfg: Qwen35Config) -> dict:
    prefix = f'{_LM_PREFIX}.layers.{layer_idx}'
    ap = f'{prefix}.self_attn'

    # q_proj shape is (q_dim * 2, D) in HF — the *2 includes the output gate.
    attn = {
        'q_proj':  _jT(st.get(f'{ap}.q_proj.weight')),
        'k_proj':  _jT(st.get(f'{ap}.k_proj.weight')),
        'v_proj':  _jT(st.get(f'{ap}.v_proj.weight')),
        'o_proj':  _jT(st.get(f'{ap}.o_proj.weight')),
        'q_norm':  _j(st.get(f'{ap}.q_norm.weight')),
        'k_norm':  _j(st.get(f'{ap}.k_norm.weight')),
    }
    return {
        'attn_norm': _j(st.get(f'{prefix}.input_layernorm.weight')),
        'attn':      attn,
        'moe_norm':  _j(st.get(f'{prefix}.post_attention_layernorm.weight')),
        'moe':       _load_moe(st, prefix, cfg),
    }


def _load_group(st: _SafetensorsIndex, group_idx: int, cfg: Qwen35Config) -> dict:
    """Load one 4-layer group's params into CPU memory as a JAX pytree."""
    base = group_idx * cfg.full_attention_interval
    n_delta = cfg.full_attention_interval - 1

    delta_list = [_load_delta_layer(st, base + d, cfg) for d in range(n_delta)]
    delta_stacked = jax.tree.map(lambda *arrs: jnp.stack(arrs, axis=0), *delta_list)

    gqa_layer = _load_gqa_layer(st, base + n_delta, cfg)

    return {'delta_layers': delta_stacked, 'gqa_layer': gqa_layer}


# ---------------------------------------------------------------------------
# Sharding helper for a single group (not the full stacked params pytree)
# ---------------------------------------------------------------------------

def _shard_group(g_params: dict, mesh: Mesh, cfg: Qwen35Config,
                 axis_rules: dict) -> dict:
    """Apply NamedSharding to one group's params.

    Reuses the logical-axis + path-matching logic from sharding.py.
    The group params have leading (n_delta,) axes on delta_layers entries —
    these get None-padded specs, identical to how shard_params handles them.
    """
    logical_axes = _param_logical_axes(cfg)

    def _shard_leaf(path, leaf):
        path_str = _flatten_path(path)
        spec = _resolve_spec(path_str, logical_axes, axis_rules)
        spec = _pad_spec_to_ndim(spec, leaf.ndim)
        # Fall back to replicated for dims not divisible by the mesh size
        safe = []
        for i, ax in enumerate(spec):
            if ax is not None and leaf.shape[i] % mesh.shape[ax] != 0:
                safe.append(None)
            else:
                safe.append(ax)
        return jax.device_put(leaf, NamedSharding(mesh, P(*safe)))

    return jax.tree_util.tree_map_with_path(_shard_leaf, g_params)


# ---------------------------------------------------------------------------
# JIT-compiled group forward (compiled once, reused for all 15 groups)
# ---------------------------------------------------------------------------

_jit_group_forward = jax.jit(
    group_forward,
    static_argnames=('config', 'is_decode', 'n_devices', 'axis_name', 'mesh'),
)

if HAS_RPA:
    _jit_group_forward_rpa = jax.jit(
        group_forward_rpa,
        static_argnames=('config', 'n_devices', 'axis_name', 'mesh'),
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def forward_offload(
    tokens: jax.Array,
    config: Qwen35Config,
    model_dir: str | Path,
    mesh: Mesh,
    axis_rules: dict | None = None,
    cache: HybridCache | None = None,
    is_decode: bool = False,
    verbose: bool = True,
    use_rpa: bool = False,
    paged_cache: PagedGQACache | None = None,
) -> tuple[jax.Array, HybridCache]:
    """Full Qwen3.5 forward pass with one-group-at-a-time weight offloading.

    Parameters
    ----------
    tokens:     (B, T) int32 token ids, already on device.
    config:     Qwen35Config (use Qwen35Config.full() for the 397B model).
    model_dir:  Path to directory containing safetensors shards + index.
    mesh:       Device mesh (e.g. make_mesh(n_devices=4)).
    axis_rules: Sharding rules (default: AXIS_RULES_B).
    cache:      HybridCache for incremental decode, or None for prefill.
    is_decode:  True for single-token decode mode.
    verbose:    Print group progress.
    use_rpa:    Use RPA v3 kernel for GQA decode (requires tpu-inference).
    paged_cache: PagedGQACache for RPA decode, or None (will be created from
                 contiguous cache on first use).

    Returns
    -------
    logits:    (B, T, vocab_size)
    new_cache: Updated HybridCache (or tuple with PagedGQACache when use_rpa=True).
    """
    if axis_rules is None:
        axis_rules = AXIS_RULES_B

    st = _SafetensorsIndex(model_dir)
    B, T = tokens.shape
    n_groups = config.n_groups
    n_delta = config.full_attention_interval - 1
    n_devices = mesh.devices.size

    # ------------------------------------------------------------------
    # Load always-resident tensors (small — embedding, norms, lm_head)
    # ------------------------------------------------------------------
    embed_w = _j(st.get(f'{_LM_PREFIX}.embed_tokens.weight'))
    final_norm_w = _j(st.get(f'{_LM_PREFIX}.norm.weight'))
    lm_head_w = _jT(st.get('lm_head.weight'))

    # Shard embedding and lm_head; replicate final_norm (tiny)
    vocab_ax = axis_rules.get('vocab')
    embed_w      = jax.device_put(embed_w,      NamedSharding(mesh, P(vocab_ax, None)))
    lm_head_w    = jax.device_put(lm_head_w,    NamedSharding(mesh, P(None, vocab_ax)))
    final_norm_w = jax.device_put(final_norm_w, NamedSharding(mesh, P(None)))

    # ------------------------------------------------------------------
    # Embedding lookup
    # ------------------------------------------------------------------
    tokens_dev = jax.device_put(tokens, NamedSharding(mesh, P(None, None)))
    x = embed_w[tokens_dev]   # (B, T, D)

    rope_freqs = precompute_rope_freqs(
        config.gqa_rope_dim, config.max_position_embeddings, config.gqa_rope_theta,
    )

    # ------------------------------------------------------------------
    # Prepare cache arrays (allocate zeros if no cache provided)
    # ------------------------------------------------------------------
    key_dim   = config.delta_n_qk_heads * config.delta_qk_head_dim
    value_dim = config.delta_n_v_heads  * config.delta_v_head_dim
    conv_dim  = key_dim * 2 + value_dim

    if cache is not None:
        cache_pos    = cache.pos
        all_delta_Ms   = cache.delta_M    # (n_groups, n_delta, B, n_v_heads, qk_dim, v_dim)
        all_delta_convs = cache.delta_conv  # (n_groups, n_delta, B, conv_dim, kernel)
        all_gqa_ks   = cache.gqa_k        # (n_groups, B, n_kv_heads, max_len, head_dim)
        all_gqa_vs   = cache.gqa_v
    else:
        cache_pos = None
        # Dummy zero arrays — only used for scan signature consistency when cache=None
        all_delta_Ms    = jnp.zeros((n_groups, n_delta, B,
                                     config.delta_n_v_heads,
                                     config.delta_qk_head_dim,
                                     config.delta_v_head_dim))
        all_delta_convs = jnp.zeros((n_groups, n_delta, B,
                                     conv_dim, config.delta_conv_kernel))
        all_gqa_ks      = jnp.zeros((n_groups, B, config.gqa_n_kv_heads,
                                     T, config.gqa_head_dim))
        all_gqa_vs      = jnp.zeros((n_groups, B, config.gqa_n_kv_heads,
                                     T, config.gqa_head_dim))

    # ------------------------------------------------------------------
    # RPA setup: convert contiguous cache to paged on first decode
    # ------------------------------------------------------------------
    _use_rpa = use_rpa and is_decode and HAS_RPA
    if _use_rpa and paged_cache is None and cache is not None:
        # First decode step with RPA: convert contiguous GQA KV to paged
        page_size = 64
        paged_kv_list = contiguous_to_paged(
            all_gqa_ks, all_gqa_vs,
            prefill_len=int(cache_pos),
            page_size=page_size,
        )
        pages_per_seq = cdiv(all_gqa_ks.shape[3], page_size)
        paged_cache = PagedGQACache(
            kv_pages=paged_kv_list,
            kv_lens=jnp.full((B,), int(cache_pos), dtype=jnp.int32),
            page_indices=jnp.arange(B * pages_per_seq, dtype=jnp.int32),
            page_size=page_size,
            pages_per_seq=pages_per_seq,
        )

    # ------------------------------------------------------------------
    # Group loop — load, shard, forward, discard
    # ------------------------------------------------------------------
    new_delta_Ms:   list[jax.Array] = []
    new_delta_convs: list[jax.Array] = []
    new_gqa_ks:     list[jax.Array] = []
    new_gqa_vs:     list[jax.Array] = []
    new_paged_caches: list[jax.Array] = []

    if _use_rpa:
        cu_q_lens, distribution = make_decode_metadata(
            B, paged_cache.kv_lens, paged_cache.pages_per_seq,
        )

    for g in range(n_groups):
        if verbose:
            print(f'\r  group {g + 1:2d}/{n_groups}', end='', flush=True)

        # Load one group from disk → CPU JAX arrays
        g_params_cpu = _load_group(st, g, config)
        # Shard to devices
        g_params = _shard_group(g_params_cpu, mesh, config, axis_rules)
        del g_params_cpu  # free CPU copy

        # Slice cache for this group
        g_dM   = all_delta_Ms[g]
        g_dC   = all_delta_convs[g]

        with mesh:
            if _use_rpa:
                x, new_dM, new_dC, updated_kv = _jit_group_forward_rpa(
                    x, g_params,
                    g_dM, g_dC,
                    paged_cache.kv_pages[g],
                    paged_cache.kv_lens,
                    paged_cache.page_indices,
                    cu_q_lens, distribution,
                    cache_pos, config, rope_freqs,
                    n_devices=n_devices, mesh=mesh,
                )
                new_paged_caches.append(updated_kv)
            else:
                g_gk = all_gqa_ks[g]
                g_gv = all_gqa_vs[g]
                x, new_dM, new_dC, new_gk, new_gv = _jit_group_forward(
                    x, g_params,
                    g_dM, g_dC, g_gk, g_gv,
                    cache_pos, config, rope_freqs, is_decode,
                    n_devices=n_devices, mesh=mesh,
                )
                new_gqa_ks.append(new_gk)
                new_gqa_vs.append(new_gv)

        # Ensure XLA has committed the computation before we free params
        jax.effects_barrier()
        del g_params

        # Close shard file handles no longer needed
        st.close()

        new_delta_Ms.append(new_dM)
        new_delta_convs.append(new_dC)

    if verbose:
        print()  # newline after progress

    # ------------------------------------------------------------------
    # Final norm + lm_head
    # ------------------------------------------------------------------
    x = rms_norm(x, final_norm_w, config.rms_norm_eps)
    logits = x @ lm_head_w   # (B, T, vocab_size)

    # ------------------------------------------------------------------
    # Rebuild cache
    # ------------------------------------------------------------------
    new_pos = jnp.array(
        (0 if cache_pos is None else int(cache_pos)) + T,
        dtype=jnp.int32,
    )

    if _use_rpa:
        new_cache = HybridCache(
            delta_M=jnp.stack(new_delta_Ms, axis=0),
            delta_conv=jnp.stack(new_delta_convs, axis=0),
            gqa_k=cache.gqa_k,  # keep contiguous cache unchanged (paged is authoritative)
            gqa_v=cache.gqa_v,
            pos=new_pos,
        )
        new_paged = PagedGQACache(
            kv_pages=new_paged_caches,
            kv_lens=paged_cache.kv_lens + 1,
            page_indices=paged_cache.page_indices,
            page_size=paged_cache.page_size,
            pages_per_seq=paged_cache.pages_per_seq,
        )
        return logits, new_cache, new_paged
    else:
        new_cache = HybridCache(
            delta_M=jnp.stack(new_delta_Ms, axis=0),
            delta_conv=jnp.stack(new_delta_convs, axis=0),
            gqa_k=jnp.stack(new_gqa_ks, axis=0),
            gqa_v=jnp.stack(new_gqa_vs, axis=0),
            pos=new_pos,
        )
        return logits, new_cache
