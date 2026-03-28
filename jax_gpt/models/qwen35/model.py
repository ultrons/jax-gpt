"""Qwen3.5-style model: pure JAX implementation.

Full model: embedding -> lax.scan over groups -> final norm -> lm_head.
All functions are pure — params are nested dicts, cache is a pytree.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from jax.sharding import PartitionSpec as P

from jax_gpt.models.qwen35.fp8 import matmul_maybe_fp8
from jax_gpt.models.qwen35.block import group_forward, HAS_RPA
from jax_gpt.models.qwen35.cache import HybridCache, init_cache
from jax_gpt.models.qwen35.config import Qwen35Config
from jax_gpt.models.qwen35.primitives import precompute_rope_freqs, rms_norm

if HAS_RPA:
    from jax_gpt.models.qwen35.block import group_forward_rpa
    from jax_gpt.models.qwen35.paged_cache import make_decode_metadata

    _jit_group_forward_rpa = jax.jit(
        group_forward_rpa,
        static_argnames=('config', 'n_devices', 'axis_name', 'mesh', 'moe_backend'),
    )


# ---------------------------------------------------------------------------
# Parameter initialization
# ---------------------------------------------------------------------------

def _init_linear(key: jax.Array, in_dim: int, out_dim: int, scale: float = 0.02,
                  dtype: jnp.dtype = jnp.float32, fp8: bool = False):
    """Init a linear weight. If fp8=True, returns {'w': fp8, 'scale_inv': f32} dict."""
    if fp8:
        from jax_gpt.models.qwen35.fp8 import FP8_DTYPE
        # Generate in bf16, transpose to (out, in) for fp8_matmul, cast to fp8
        w = jax.random.normal(key, (out_dim, in_dim), dtype=jnp.bfloat16) * scale
        w_fp8 = w.astype(FP8_DTYPE)
        # scale_inv = 1.0 for random weights (values already in fp8 range)
        scale_inv = jnp.ones((out_dim, 1), dtype=jnp.float32)
        return {'w': w_fp8, 'scale_inv': scale_inv}
    return jax.random.normal(key, (in_dim, out_dim), dtype=dtype) * scale


def _init_deltanet_attn_params(key: jax.Array, config: Qwen35Config,
                               dtype: jnp.dtype = jnp.float32, fp8: bool = False) -> dict:
    """Initialize one DeltaNet attention sub-layer's params."""
    keys = jax.random.split(key, 10)
    D = config.d_model
    key_dim = config.delta_n_qk_heads * config.delta_qk_head_dim
    value_dim = config.delta_n_v_heads * config.delta_v_head_dim
    conv_dim = key_dim * 2 + value_dim

    return {
        'in_proj_qkv': _init_linear(keys[0], D, conv_dim, dtype=dtype, fp8=fp8),
        'in_proj_z': _init_linear(keys[1], D, value_dim, dtype=dtype, fp8=fp8),
        'in_proj_b': _init_linear(keys[2], D, config.delta_n_v_heads, dtype=dtype, fp8=fp8),
        'in_proj_a': _init_linear(keys[3], D, config.delta_n_v_heads, dtype=dtype, fp8=fp8),
        'conv_weight': (jax.random.normal(keys[4], (conv_dim, config.delta_conv_kernel), dtype=dtype) * 0.02),
        'A_log': jnp.log(jax.random.uniform(keys[5], (config.delta_n_v_heads,), minval=0.1, maxval=16.0)),
        'dt_bias': jnp.ones(config.delta_n_v_heads, dtype=dtype),
        'norm_weight': jnp.ones(config.delta_v_head_dim, dtype=dtype),
        'out_proj': _init_linear(keys[6], value_dim, D, dtype=dtype, fp8=fp8),
    }


def _init_gqa_attn_params(key: jax.Array, config: Qwen35Config,
                          dtype: jnp.dtype = jnp.float32, fp8: bool = False) -> dict:
    """Initialize one GQA attention sub-layer's params."""
    keys = jax.random.split(key, 5)
    D = config.d_model
    q_dim = config.gqa_n_q_heads * config.gqa_head_dim
    kv_dim = config.gqa_n_kv_heads * config.gqa_head_dim

    return {
        'q_proj': _init_linear(keys[0], D, q_dim * 2, dtype=dtype, fp8=fp8),
        'k_proj': _init_linear(keys[1], D, kv_dim, dtype=dtype, fp8=fp8),
        'v_proj': _init_linear(keys[2], D, kv_dim, dtype=dtype, fp8=fp8),
        'o_proj': _init_linear(keys[3], q_dim, D, dtype=dtype, fp8=fp8),
        'q_norm': jnp.zeros(config.gqa_head_dim, dtype=dtype),
        'k_norm': jnp.zeros(config.gqa_head_dim, dtype=dtype),
    }


def _init_moe_params(key: jax.Array, config: Qwen35Config,
                     dtype: jnp.dtype = jnp.float32, fp8: bool = False) -> dict:
    """Initialize one MoE layer's params."""
    keys = jax.random.split(key, 8)
    D = config.d_model
    E = config.n_routed_experts
    I = config.moe_intermediate_size
    SI = config.shared_expert_intermediate_size

    # Expert weights: 3D (E, K, N) in ragged_dot layout — no transpose needed
    if fp8:
        from jax_gpt.models.qwen35.fp8 import FP8_DTYPE
        def _init_expert_fp8(k, shape):
            # Store in ragged_dot layout (E, K, N) with per-output scale (E, 1, N)
            E_, K_, N_ = shape
            w = jax.random.normal(k, (E_, K_, N_), dtype=jnp.bfloat16) * 0.02
            return {'w': w.astype(FP8_DTYPE),
                    'scale_inv': jnp.ones((E_, 1, N_), dtype=jnp.float32)}
        gate_proj = _init_expert_fp8(keys[1], (E, D, I))
        up_proj = _init_expert_fp8(keys[2], (E, D, I))
        down_proj = _init_expert_fp8(keys[3], (E, I, D))
    else:
        gate_proj = jax.random.normal(keys[1], (E, D, I), dtype=dtype) * 0.02
        up_proj = jax.random.normal(keys[2], (E, D, I), dtype=dtype) * 0.02
        down_proj = jax.random.normal(keys[3], (E, I, D), dtype=dtype) * 0.02

    return {
        'gate_weight': _init_linear(keys[0], D, E, dtype=dtype, fp8=fp8),
        'gate_proj': gate_proj,
        'up_proj': up_proj,
        'down_proj': down_proj,
        'shared_gate_proj': _init_linear(keys[4], D, SI, dtype=dtype, fp8=fp8),
        'shared_up_proj': _init_linear(keys[5], D, SI, dtype=dtype, fp8=fp8),
        'shared_down_proj': _init_linear(keys[6], SI, D, dtype=dtype, fp8=fp8),
        'shared_expert_gate_weight': _init_linear(keys[7], D, 1, dtype=dtype, fp8=fp8),
    }


def _init_delta_layer_params(key: jax.Array, config: Qwen35Config,
                             dtype: jnp.dtype = jnp.float32, fp8: bool = False) -> dict:
    """Initialize one DeltaNet layer (attn_norm + attn + moe_norm + moe)."""
    k1, k2 = jax.random.split(key)
    return {
        'attn_norm': jnp.zeros(config.d_model, dtype=dtype),
        'attn': _init_deltanet_attn_params(k1, config, dtype, fp8),
        'moe_norm': jnp.zeros(config.d_model, dtype=dtype),
        'moe': _init_moe_params(k2, config, dtype, fp8),
    }


def _init_gqa_layer_params(key: jax.Array, config: Qwen35Config,
                           dtype: jnp.dtype = jnp.float32, fp8: bool = False) -> dict:
    """Initialize one GQA layer (attn_norm + attn + moe_norm + moe)."""
    k1, k2 = jax.random.split(key)
    return {
        'attn_norm': jnp.zeros(config.d_model, dtype=dtype),
        'attn': _init_gqa_attn_params(k1, config, dtype, fp8),
        'moe_norm': jnp.zeros(config.d_model, dtype=dtype),
        'moe': _init_moe_params(k2, config, dtype, fp8),
    }


def _stack_tree(trees: list[dict]) -> dict:
    """Stack a list of identical-structure param dicts into one dict with
    leading axis. E.g. [{a: (D,), b: (D, D)}, ...] -> {a: (N, D), b: (N, D, D)}."""
    return jax.tree.map(lambda *arrs: jnp.stack(arrs, axis=0), *trees)


def init_params(config: Qwen35Config, key: jax.Array, dtype: jnp.dtype = jnp.float32, fp8: bool = False) -> dict:
    """Initialize all model parameters as a nested dict pytree.

    Args:
        config: model config.
        key: PRNG key.
        dtype: parameter dtype (use jnp.bfloat16 for large models to save memory).

    Structure:
        embed: (vocab_size, d_model)
        groups: stacked group params with leading n_groups axis
            delta_layers: stacked (3 per group) DeltaNet layer params
            gqa_layer: GQA layer params (1 per group)
        final_norm: (d_model,)
        lm_head: (d_model, vocab_size)
    """
    keys = jax.random.split(key, 3 + config.n_groups * 2)
    key_idx = 0

    # Embedding
    embed = jax.random.normal(keys[key_idx], (config.vocab_size, config.d_model), dtype=dtype) * 0.02
    key_idx += 1

    # Groups
    group_params_list = []
    for g in range(config.n_groups):
        # 3 DeltaNet layers
        delta_keys = jax.random.split(keys[key_idx], 3)
        key_idx += 1
        delta_layers = _stack_tree([
            _init_delta_layer_params(delta_keys[i], config, dtype, fp8) for i in range(3)
        ])

        # 1 GQA layer
        gqa_layer = _init_gqa_layer_params(keys[key_idx], config, dtype, fp8)
        key_idx += 1

        group_params_list.append({
            'delta_layers': delta_layers,
            'gqa_layer': gqa_layer,
        })

    groups = _stack_tree(group_params_list)

    # Final norm + lm_head (embed stays as regular array for lookup; lm_head can be fp8)
    final_norm = jnp.zeros(config.d_model, dtype=dtype)
    lm_head = _init_linear(keys[key_idx], config.d_model, config.vocab_size, dtype=dtype, fp8=fp8)

    return {
        'embed': embed,
        'groups': groups,
        'final_norm': final_norm,
        'lm_head': lm_head,
    }


# ---------------------------------------------------------------------------
# Output head helpers
# ---------------------------------------------------------------------------

def _topk_output_head(
    logits_shard: jax.Array,
    vocab_size: int,
    k: int,
    mesh,
    axis_name: str,
) -> jax.Array:
    """Replace full vocab all-gather with local top-k + tiny all-gather.

    Instead of gathering all vocab/tp logits to every device (1+ GB), each
    device takes its local top-k, all-gathers only k*tp candidates, then
    selects the global top-k.  Communication: B * k * tp * 6 bytes (bf16 val
    + int32 idx) vs B * vocab * 2 bytes previously.

    Args:
        logits_shard: (B, T, vocab/tp) — vocab-sharded logits on this device.
        vocab_size: total vocabulary size.
        k: number of top tokens to return.
        mesh: device mesh.
        axis_name: TP axis name in the mesh.

    Returns:
        top_ids: (B, T, k) int32 global token ids, replicated across TP.
    """
    from jax.experimental.shard_map import shard_map
    from jax.sharding import PartitionSpec as P

    tp_size = mesh.shape[axis_name]
    vocab_shard_size = vocab_size // tp_size

    # Build in/out specs: batch on dp axis (if present), vocab on tp axis.
    dp_axis = next((n for n in mesh.axis_names if n != axis_name), None)
    if dp_axis is not None:
        in_spec = P(dp_axis, None, axis_name)
        out_spec = P(dp_axis, None, None)
    else:
        in_spec = P(None, None, axis_name)
        out_spec = P(None, None, None)

    def _local_topk(shard):
        # shard: (B/dp, T, vocab/tp)
        B_local, T_local, _ = shard.shape
        flat = shard.reshape(B_local * T_local, -1)          # (B*T, vocab/tp)
        vals, local_idx = jax.lax.top_k(flat, k)             # (B*T, k)
        tp_rank = jax.lax.axis_index(axis_name)
        global_idx = local_idx + tp_rank * vocab_shard_size   # (B*T, k)
        # All-gather k candidates from each TP device along k-axis (axis=1)
        all_vals = jax.lax.all_gather(vals, axis_name, axis=1, tiled=True)      # (B*T, k*tp)
        all_idx  = jax.lax.all_gather(global_idx, axis_name, axis=1, tiled=True)
        # Global top-k
        _, positions = jax.lax.top_k(all_vals, k)                       # (B*T, k)
        final_idx = jnp.take_along_axis(all_idx, positions, axis=-1)    # (B*T, k)
        return final_idx.reshape(B_local, T_local, k)                   # (B/dp, T, k)

    return shard_map(
        _local_topk,
        mesh=mesh,
        in_specs=in_spec,
        out_specs=out_spec,
        check_rep=False,
    )(logits_shard)


# ---------------------------------------------------------------------------
# Forward pass
# ---------------------------------------------------------------------------

def forward(
    params: dict,
    tokens: jax.Array,
    config: Qwen35Config,
    cache: HybridCache | None = None,
    is_decode: bool = False,
    cache_sharding: dict | None = None,
    n_devices: int = 1,
    axis_name: str = 'tp',
    mesh=None,
    last_logit_only: bool = False,
    use_rpa: bool = False,
    scan_mode: str = 'scan',
    moe_backend: str = 'ragged_dot',
    output_top_k: int = 0,
) -> tuple[jax.Array, HybridCache | None]:
    """Full model forward pass.

    Args:
        params: nested dict pytree from init_params.
        tokens: (B, T) int32 token ids.
        config: model config.
        cache: HybridCache or None.
        is_decode: True for single-token decode mode.
        cache_sharding: optional dict with PartitionSpecs for cache arrays.
            Keys: 'delta_M', 'delta_conv', 'gqa_kv'. When provided,
            jax.lax.with_sharding_constraint is applied to cache outputs
            inside the scan to prevent XLA from inferring incompatible sharding.
        scan_mode: 'scan' (default) uses jax.lax.scan over groups.
            'unrolled' uses a Python for-loop over groups inside the JIT,
            avoiding XLA WhileOp copy insertion that OOMs with large MoE
            expert weights. Trades longer compilation for no scan copies.
        moe_backend: 'ragged_dot' (XLA) or 'gmm' (Pallas kernel).

    Returns:
        (logits, updated_cache)
        logits: (B, T, vocab_size)
    """
    assert scan_mode in ('scan', 'unrolled'), (
        f"scan_mode must be 'scan' or 'unrolled', got {scan_mode!r}"
    )
    B, T = tokens.shape

    with jax.named_scope('embedding'):
        x = params['embed'][tokens]  # (B, T, D)

    # Precompute RoPE frequencies
    rope_freqs = precompute_rope_freqs(
        config.gqa_rope_dim,
        config.max_position_embeddings,
        config.gqa_rope_theta,
    )

    # Prepare cache slices for scan
    if cache is not None:
        cache_pos = cache.pos
        delta_Ms = cache.delta_M        # (n_groups, 3, B, ...)
        delta_convs = cache.delta_conv   # (n_groups, 3, B, ...)
        gqa_ks = cache.gqa_k             # (n_groups, B, ...)
        gqa_vs = cache.gqa_v             # (n_groups, B, ...)
    else:
        cache_pos = None
        delta_Ms = None
        delta_convs = None
        gqa_ks = None
        gqa_vs = None

    # Determine whether to use RPA path
    _use_rpa = use_rpa and is_decode and HAS_RPA and cache is not None

    # RPA metadata (used by both scan and unrolled RPA paths)
    if _use_rpa:
        paged_kv = cache.paged_kv       # tuple of n_groups arrays, each (total_pages, ps, kv_dim, pk, hd)
        kv_lens = cache.kv_lens          # (B,)
        page_indices = cache.page_indices  # (B * pages_per_seq,)
        pages_per_seq = paged_kv[0].shape[0] // B

        if mesh is not None and 'dp' in mesh.axis_names:
            dp = mesh.shape['dp']
            B_local = B // dp
        else:
            B_local = B
        cu_q_lens, distribution = make_decode_metadata(B_local, kv_lens, pages_per_seq)
        page_indices_local = jnp.arange(B_local * pages_per_seq, dtype=jnp.int32)

    # ── Unrolled path: Python for-loop inside JIT ──────────────────────
    # Traces 15 sequential group calls into one XLA program.
    # No WhileOp → no XLA copy insertion of expert weights.
    #
    # If params contains 'groups_list' (a Python list of pre-split per-group
    # dicts), we use those directly — avoids the expensive jax.tree.map(a[g])
    # squeeze/slice operations that dominate device time when slicing from
    # stacked (n_groups, ...) arrays.
    groups_list = params.get('groups_list', None)

    def _dyn_idx(tensor, g):
        """Dynamic index into axis-0 of tensor at position g.

        Using dynamic_index_in_dim instead of static tensor[g] prevents XLA
        from fusing all 15 group reads into one giant slice_bitcast_fusion.
        Static indexing causes XLA to pre-materialize all group slices from
        the same buffer simultaneously — each 4-6 GB tensor × 15 groups = 60+
        GB read per step for cache tensors (paged_kv, delta_M, delta_conv).
        """
        return jax.lax.dynamic_index_in_dim(tensor, jnp.int32(g), axis=0, keepdims=False)

    def _slice_group(groups, g):
        """Slice group g from stacked params using dynamic_index_in_dim.

        Uses dynamic_index_in_dim instead of static a[g] indexing to prevent
        XLA from fusing all 15 group slices into one giant retiling op.
        Static a[g] causes XLA to pre-materialize all 15 slices with a tiling
        change (T(8,128) -> T(32,128)), consuming 58% of decode device time.
        """
        g_idx = jnp.int32(g)
        return jax.tree.map(
            lambda a: jax.lax.dynamic_index_in_dim(a, g_idx, axis=0, keepdims=False),
            groups)

    def _slice_moe_flat(moe_flat, idx):
        """Slice one layer's MoE params from the flattened (n_groups*n_delta, ...)
        array using dynamic_index_in_dim.

        Uses dynamic_index_in_dim (same as _slice_group) instead of
        dynamic_slice_in_dim + squeeze, which caused XLA to fuse all 15
        group slices into one giant slice_bitcast_fusion consuming ~0.8ms/step.
        """
        idx_arr = jnp.int32(idx)
        return jax.tree.map(
            lambda a: jax.lax.dynamic_index_in_dim(a, idx_arr, axis=0, keepdims=False),
            moe_flat)

    if scan_mode == 'unrolled':
        # ── Separate MoE weights for reduced squeeze overhead ──────────
        # MoE expert weights (gate_proj, up_proj, down_proj) are the
        # largest tensors (~96GB each for gate/up) and cause 58% of decode
        # device time as XLA slice_bitcast_fusion retiling when sliced from
        # stacked (n_groups, n_delta, E, K, N) arrays.
        #
        # When moe_backend == 'gmm': pass stacked expert weights to
        # gmm_v2 with rhs_group_offset. The Pallas kernel's DMA index map
        # reads weights directly from the stacked array — no JAX-level
        # squeeze at all. Only small MoE params (gate_weight, shared_*)
        # are sliced per-layer.
        #
        # When moe_backend != 'gmm': use v69 approach — separate MoE,
        # reshape to (45, ...), dynamic_slice_in_dim + squeeze.
        n_delta = config.full_attention_interval - 1
        n_groups = config.n_groups

        _expert_keys = {'gate_proj', 'up_proj', 'down_proj'}
        _use_gmm_stacking = moe_backend == 'gmm' and groups_list is None

        stacked_delta_expert = None
        stacked_gqa_expert = None

        if groups_list is None:
            delta_moe_raw = params['groups']['delta_layers']['moe']
            gqa_moe_raw = params['groups']['gqa_layer']['moe']

            if _use_gmm_stacking:
                # ── gmm stacking: keep expert weights 4D, slice only small params
                delta_expert_raw = {k: v for k, v in delta_moe_raw.items()
                                    if k in _expert_keys}
                delta_small_raw = {k: v for k, v in delta_moe_raw.items()
                                   if k not in _expert_keys}
                gqa_expert_raw = {k: v for k, v in gqa_moe_raw.items()
                                  if k in _expert_keys}
                gqa_small_raw = {k: v for k, v in gqa_moe_raw.items()
                                 if k not in _expert_keys}

                # Stacked expert weights: (15, 3, E, K, N) → (45, E, K, N)
                stacked_delta_expert = jax.tree.map(
                    lambda a: a.reshape(n_groups * n_delta, *a.shape[2:]),
                    delta_expert_raw)
                stacked_gqa_expert = gqa_expert_raw  # (15, E, K, N) — already 4D

                # Small params: (15, 3, ...) → (45, ...) for per-layer slicing
                delta_moe_flat = jax.tree.map(
                    lambda a: a.reshape(n_groups * n_delta, *a.shape[2:]),
                    delta_small_raw)
                gqa_moe_flat = gqa_small_raw
            else:
                # ── v69 approach: slice ALL MoE params
                delta_moe_flat = jax.tree.map(
                    lambda a: a.reshape(n_groups * n_delta, *a.shape[2:]),
                    delta_moe_raw)
                gqa_moe_flat = gqa_moe_raw

        if _use_rpa:
            new_delta_Ms = []
            new_delta_convs = []
            new_paged_kvs = []

            for g in range(n_groups):
                if groups_list is not None:
                    g_params = groups_list[g]
                    g_delta_moe_list = None
                    g_gqa_moe = None
                else:
                    g_params = _slice_group(params['groups'], g)
                    g_delta_moe_list = [
                        _slice_moe_flat(delta_moe_flat, g * n_delta + i)
                        for i in range(n_delta)
                    ]
                    g_gqa_moe = _slice_moe_flat(gqa_moe_flat, g)

                x, new_dM, new_dC, updated_kv = group_forward_rpa(
                    x, g_params,
                    _dyn_idx(delta_Ms, g), _dyn_idx(delta_convs, g),
                    paged_kv[g], kv_lens, page_indices_local,
                    cu_q_lens, distribution,
                    cache_pos, config, rope_freqs,
                    n_devices=n_devices, mesh=mesh, axis_name=axis_name,
                    moe_backend=moe_backend,
                    delta_moe_list=g_delta_moe_list,
                    gqa_moe=g_gqa_moe,
                    stacked_delta_expert_weights=stacked_delta_expert,
                    stacked_gqa_expert_weights=stacked_gqa_expert,
                    group_idx=g if _use_gmm_stacking else None,
                )
                if cache_sharding is not None:
                    new_dM = jax.lax.with_sharding_constraint(new_dM, cache_sharding['delta_M'])
                    new_dC = jax.lax.with_sharding_constraint(new_dC, cache_sharding['delta_conv'])
                    if 'paged_kv' in cache_sharding:
                        updated_kv = jax.lax.with_sharding_constraint(updated_kv, cache_sharding['paged_kv'])
                new_delta_Ms.append(new_dM)
                new_delta_convs.append(new_dC)
                new_paged_kvs.append(updated_kv)

            # Stack delta states; keep paged_kv as tuple of per-group arrays.
            # Tuple paged_kv: paged_kv[g] inside JIT is Python tuple indexing at
            # trace time — a direct input parameter reference, zero JAX slice op,
            # no slice_bitcast_fusion. jnp.stack would recreate the bottleneck.
            result_delta_M = jnp.stack(new_delta_Ms)
            result_delta_conv = jnp.stack(new_delta_convs)
            result_paged_kv = tuple(new_paged_kvs)

            new_cache = HybridCache(
                delta_M=result_delta_M,
                delta_conv=result_delta_conv,
                gqa_k=cache.gqa_k,
                gqa_v=cache.gqa_v,
                pos=cache_pos + T,
                paged_kv=result_paged_kv,
                kv_lens=kv_lens + 1,
                page_indices=page_indices,
            )

        elif cache is not None:
            result_delta_M = delta_Ms
            result_delta_conv = delta_convs
            result_gqa_k = gqa_ks
            result_gqa_v = gqa_vs

            for g in range(n_groups):
                if groups_list is not None:
                    g_params = groups_list[g]
                    g_delta_moe_list = None
                    g_gqa_moe = None
                else:
                    g_params = _slice_group(params['groups'], g)
                    g_delta_moe_list = [
                        _slice_moe_flat(delta_moe_flat, g * n_delta + i)
                        for i in range(n_delta)
                    ]
                    g_gqa_moe = _slice_moe_flat(gqa_moe_flat, g)

                x, new_dM, new_dC, new_gk, new_gv = group_forward(
                    x, g_params,
                    _dyn_idx(delta_Ms, g), _dyn_idx(delta_convs, g),
                    _dyn_idx(gqa_ks, g), _dyn_idx(gqa_vs, g),
                    cache_pos, config, rope_freqs, is_decode,
                    n_devices=n_devices, mesh=mesh, axis_name=axis_name,
                    moe_backend=moe_backend,
                    delta_moe_list=g_delta_moe_list,
                    gqa_moe=g_gqa_moe,
                    stacked_delta_expert_weights=stacked_delta_expert,
                    stacked_gqa_expert_weights=stacked_gqa_expert,
                    group_idx=g if _use_gmm_stacking else None,
                )
                if cache_sharding is not None:
                    new_dM = jax.lax.with_sharding_constraint(new_dM, cache_sharding['delta_M'])
                    new_dC = jax.lax.with_sharding_constraint(new_dC, cache_sharding['delta_conv'])
                    new_gk = jax.lax.with_sharding_constraint(new_gk, cache_sharding['gqa_kv'])
                    new_gv = jax.lax.with_sharding_constraint(new_gv, cache_sharding['gqa_kv'])
                result_delta_M = result_delta_M.at[g].set(new_dM)
                result_delta_conv = result_delta_conv.at[g].set(new_dC)
                result_gqa_k = result_gqa_k.at[g].set(new_gk)
                result_gqa_v = result_gqa_v.at[g].set(new_gv)

            new_cache = HybridCache(
                delta_M=result_delta_M,
                delta_conv=result_delta_conv,
                gqa_k=result_gqa_k,
                gqa_v=result_gqa_v,
                pos=cache_pos + T,
            )

        else:
            # No cache — prefill with unrolled groups
            key_dim = config.delta_n_qk_heads * config.delta_qk_head_dim
            value_dim = config.delta_n_v_heads * config.delta_v_head_dim
            conv_dim = key_dim * 2 + value_dim
            dummy_dM = jnp.zeros((n_delta, B,
                                  config.delta_n_v_heads, config.delta_qk_head_dim, config.delta_v_head_dim))
            dummy_dC = jnp.zeros((n_delta, B, conv_dim, config.delta_conv_kernel))
            dummy_gK = jnp.zeros((B, config.gqa_n_kv_heads, T, config.gqa_head_dim))
            dummy_gV = jnp.zeros((B, config.gqa_n_kv_heads, T, config.gqa_head_dim))

            for g in range(n_groups):
                if groups_list is not None:
                    g_params = groups_list[g]
                    g_delta_moe_list = None
                    g_gqa_moe = None
                else:
                    g_params = _slice_group(params['groups'], g)
                    g_delta_moe_list = [
                        _slice_moe_flat(delta_moe_flat, g * n_delta + i)
                        for i in range(n_delta)
                    ]
                    g_gqa_moe = _slice_moe_flat(gqa_moe_flat, g)

                x, _, _, _, _ = group_forward(
                    x, g_params,
                    dummy_dM, dummy_dC, dummy_gK, dummy_gV,
                    None, config, rope_freqs, False,
                    n_devices=n_devices, mesh=mesh, axis_name=axis_name,
                    moe_backend=moe_backend,
                    delta_moe_list=g_delta_moe_list,
                    gqa_moe=g_gqa_moe,
                    stacked_delta_expert_weights=stacked_delta_expert,
                    stacked_gqa_expert_weights=stacked_gqa_expert,
                    group_idx=g if _use_gmm_stacking else None,
                )
            new_cache = None

    # ── Scan path (default): jax.lax.scan over groups ──────────────────
    elif _use_rpa:
        def _group_step_rpa(carry, group_inputs):
            x_carry = carry
            g_params, g_delta_M, g_delta_conv, g_paged_kv = group_inputs

            x_carry, new_dM, new_dC, updated_kv = group_forward_rpa(
                x_carry, g_params,
                g_delta_M, g_delta_conv,
                g_paged_kv, kv_lens, page_indices_local,
                cu_q_lens, distribution,
                cache_pos, config, rope_freqs,
                n_devices=n_devices, mesh=mesh, axis_name=axis_name,
                moe_backend=moe_backend,
            )

            if cache_sharding is not None:
                new_dM = jax.lax.with_sharding_constraint(new_dM, cache_sharding['delta_M'])
                new_dC = jax.lax.with_sharding_constraint(new_dC, cache_sharding['delta_conv'])
                if 'paged_kv' in cache_sharding:
                    updated_kv = jax.lax.with_sharding_constraint(updated_kv, cache_sharding['paged_kv'])

            return x_carry, (new_dM, new_dC, updated_kv)

        # Scan path requires paged_kv as a stacked array (leading axis = n_groups).
        # Tuple paged_kv is only supported in the unrolled path (scan_mode='unrolled').
        if isinstance(paged_kv, tuple):
            raise ValueError(
                "scan_mode='scan' requires paged_kv as a stacked jax.Array "
                "(n_groups, total_pages, ...). Got a tuple. "
                "Use scan_mode='unrolled' when paged_kv is a tuple."
            )
        scan_inputs = (
            params['groups'], delta_Ms, delta_convs, paged_kv,
        )
        x, (new_dMs, new_dConvs, new_paged_kv) = jax.lax.scan(
            _group_step_rpa, x, scan_inputs,
        )
        new_cache = HybridCache(
            delta_M=new_dMs,
            delta_conv=new_dConvs,
            gqa_k=cache.gqa_k,
            gqa_v=cache.gqa_v,
            pos=cache_pos + T,
            paged_kv=new_paged_kv,
            kv_lens=kv_lens + 1,
            page_indices=page_indices,
        )

    elif cache is not None:
        def _group_step(carry, group_inputs):
            x_carry = carry
            g_params, g_delta_M, g_delta_conv, g_gqa_k, g_gqa_v = group_inputs

            x_carry, new_dM, new_dC, new_gk, new_gv = group_forward(
                x_carry, g_params,
                g_delta_M, g_delta_conv,
                g_gqa_k, g_gqa_v,
                cache_pos, config, rope_freqs, is_decode,
                n_devices=n_devices, mesh=mesh, axis_name=axis_name,
                moe_backend=moe_backend,
            )

            if cache_sharding is not None:
                new_dM = jax.lax.with_sharding_constraint(new_dM, cache_sharding['delta_M'])
                new_dC = jax.lax.with_sharding_constraint(new_dC, cache_sharding['delta_conv'])
                new_gk = jax.lax.with_sharding_constraint(new_gk, cache_sharding['gqa_kv'])
                new_gv = jax.lax.with_sharding_constraint(new_gv, cache_sharding['gqa_kv'])

            return x_carry, (new_dM, new_dC, new_gk, new_gv)

        scan_inputs = (
            params['groups'], delta_Ms, delta_convs, gqa_ks, gqa_vs,
        )
        x, (new_dMs, new_dConvs, new_gKs, new_gVs) = jax.lax.scan(
            _group_step, x, scan_inputs,
        )
        new_cache = HybridCache(
            delta_M=new_dMs,
            delta_conv=new_dConvs,
            gqa_k=new_gKs,
            gqa_v=new_gVs,
            pos=cache_pos + T,
        )
    else:
        n_groups = config.n_groups
        n_delta = config.full_attention_interval - 1
        key_dim = config.delta_n_qk_heads * config.delta_qk_head_dim
        value_dim = config.delta_n_v_heads * config.delta_v_head_dim
        conv_dim = key_dim * 2 + value_dim

        dummy_dM = jnp.zeros((n_groups, n_delta, B,
                              config.delta_n_v_heads, config.delta_qk_head_dim, config.delta_v_head_dim))
        dummy_dC = jnp.zeros((n_groups, n_delta, B, conv_dim, config.delta_conv_kernel))
        dummy_gK = jnp.zeros((n_groups, B, config.gqa_n_kv_heads, T, config.gqa_head_dim))
        dummy_gV = jnp.zeros((n_groups, B, config.gqa_n_kv_heads, T, config.gqa_head_dim))

        def _group_step(carry, group_inputs):
            x_carry = carry
            g_params, g_delta_M, g_delta_conv, g_gqa_k, g_gqa_v = group_inputs
            x_carry, new_dM, new_dC, new_gk, new_gv = group_forward(
                x_carry, g_params,
                g_delta_M, g_delta_conv,
                g_gqa_k, g_gqa_v,
                None, config, rope_freqs, False,
                n_devices=n_devices, mesh=mesh, axis_name=axis_name,
                moe_backend=moe_backend,
            )
            return x_carry, (new_dM, new_dC, new_gk, new_gv)

        scan_inputs = (
            params['groups'], dummy_dM, dummy_dC, dummy_gK, dummy_gV,
        )
        x, _ = jax.lax.scan(_group_step, x, scan_inputs)
        new_cache = None

    with jax.named_scope('output_head'):
        x = rms_norm(x, params['final_norm'], config.rms_norm_eps)
        if last_logit_only:
            x = x[:, -1:, :]  # (B, 1, D) — only last position
        logits = matmul_maybe_fp8(x, params['lm_head'])  # (B, T_or_1, vocab_size/tp)
        if mesh is not None:
            from jax.sharding import NamedSharding, PartitionSpec as P
            if output_top_k > 0:
                logits = _topk_output_head(
                    logits, config.vocab_size, output_top_k, mesh, axis_name)
                return logits, new_cache
            logits = jax.lax.with_sharding_constraint(
                logits, NamedSharding(mesh, P(None, None, None)))

    return logits, new_cache


# ---------------------------------------------------------------------------
# RPA decode (per-group JIT — avoids scan+Pallas OOM)
# ---------------------------------------------------------------------------

_jit_group_forward = jax.jit(
    group_forward,
    static_argnames=('config', 'is_decode', 'n_devices', 'axis_name', 'mesh', 'moe_backend'),
)


def forward_rpa_decode(
    params: dict,
    tokens: jax.Array,
    config: Qwen35Config,
    cache: HybridCache,
    cache_sharding: dict | None = None,
    n_devices: int = 1,
    axis_name: str = 'tp',
    mesh=None,
    last_logit_only: bool = True,
    moe_backend: str = 'ragged_dot',
    output_top_k: int = 0,
) -> tuple[jax.Array, HybridCache]:
    """Decode forward pass using RPA, with per-group JIT to avoid OOM.

    Unlike forward(..., use_rpa=True) which puts the RPA Pallas kernel inside
    a jax.lax.scan (causing a multi-GB compiled program), this function uses
    a Python for-loop over groups with separately JITted group_forward_rpa.
    Each group reuses the same compiled program (~200 MB instead of ~4 GB).

    Args:
        params: nested dict pytree from init_params (all groups stacked).
        tokens: (B, 1) int32 token ids (single decode step).
        config: model config.
        cache: HybridCache with paged_kv, kv_lens, page_indices populated.
        cache_sharding: optional dict with PartitionSpecs for cache arrays.
        n_devices: number of devices.
        axis_name: TP axis name.
        mesh: device mesh (required for shard_map in RPA kernel).
        last_logit_only: if True, return logits for last position only.

    Returns:
        (logits, updated_cache)
    """
    B, T = tokens.shape
    n_groups = config.n_groups

    with jax.named_scope('embedding'):
        x = params['embed'][tokens]

    rope_freqs = precompute_rope_freqs(
        config.gqa_rope_dim,
        config.max_position_embeddings,
        config.gqa_rope_theta,
    )

    cache_pos = cache.pos
    paged_kv = cache.paged_kv        # tuple of n_groups arrays, each (total_pages, ps, kv_dim, pk, hd)
    kv_lens = cache.kv_lens          # (B,)
    page_indices = cache.page_indices  # (B * pages_per_seq,)

    # Infer pages_per_seq
    pages_per_seq = paged_kv[0].shape[0] // B

    # With dp sharding, shard_map splits batch across dp devices.
    # Compute dp-local metadata: each shard processes B_local sequences.
    if mesh is not None and 'dp' in mesh.axis_names:
        dp = mesh.shape['dp']
        B_local = B // dp
    else:
        B_local = B
    cu_q_lens, distribution = make_decode_metadata(B_local, kv_lens, pages_per_seq)
    # Local page indices: contiguous 0-based mapping for B_local sequences.
    page_indices_local = jnp.arange(B_local * pages_per_seq, dtype=jnp.int32)

    # Per-group loop — each call reuses the same compiled program.
    # Collect outputs then stack once (single XLA concatenate, no serial DUS).
    new_delta_Ms = []
    new_delta_convs = []
    new_paged_kvs = []

    ctx = mesh if mesh is not None else __import__('contextlib').nullcontext()

    for g in range(n_groups):
        # Extract this group's params (index leading axis of stacked tree)
        g_params = jax.tree.map(lambda leaf: leaf[g], params['groups'])
        g_dM = cache.delta_M[g]
        g_dC = cache.delta_conv[g]
        g_paged = paged_kv[g]

        with ctx:
            x, new_dM, new_dC, updated_kv = _jit_group_forward_rpa(
                x, g_params,
                g_dM, g_dC,
                g_paged, kv_lens, page_indices_local,
                cu_q_lens, distribution,
                cache_pos, config, rope_freqs,
                n_devices=n_devices, mesh=mesh, axis_name=axis_name,
                moe_backend=moe_backend,
            )

        new_delta_Ms.append(new_dM)
        new_delta_convs.append(new_dC)
        new_paged_kvs.append(updated_kv)

    result_delta_M = jnp.stack(new_delta_Ms)
    result_delta_conv = jnp.stack(new_delta_convs)
    result_paged_kv = tuple(new_paged_kvs)

    with jax.named_scope('output_head'):
        x = rms_norm(x, params['final_norm'], config.rms_norm_eps)
        if last_logit_only:
            x = x[:, -1:, :]
        logits = matmul_maybe_fp8(x, params['lm_head'])
        if mesh is not None:
            from jax.sharding import NamedSharding, PartitionSpec as P
            if output_top_k > 0:
                logits = _topk_output_head(
                    logits, config.vocab_size, output_top_k, mesh, axis_name)
            else:
                logits = jax.lax.with_sharding_constraint(
                    logits, NamedSharding(mesh, P(None, None, None)))

    new_cache = HybridCache(
        delta_M=result_delta_M,
        delta_conv=result_delta_conv,
        gqa_k=cache.gqa_k,
        gqa_v=cache.gqa_v,
        pos=cache_pos + T,
        paged_kv=result_paged_kv,
        kv_lens=kv_lens + 1,
        page_indices=page_indices,
    )

    return logits, new_cache


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate(
    params: dict,
    prompt_tokens: jax.Array,
    config: Qwen35Config,
    max_new_tokens: int,
    max_seq_len: int | None = None,
    temperature: float = 1.0,
    top_k: int | None = None,
    key: jax.Array | None = None,
) -> jax.Array:
    """Autoregressive generation with prefill + lax.scan decode loop.

    Args:
        params: model parameters.
        prompt_tokens: (B, T) prompt token ids.
        config: model config.
        max_new_tokens: number of new tokens to generate.
        max_seq_len: max sequence length for KV cache.
        temperature: sampling temperature.
        top_k: if set, sample from top-k logits.
        key: PRNG key for sampling.

    Returns:
        (B, max_new_tokens) generated token ids.
    """
    if key is None:
        key = jax.random.key(0)
    if max_seq_len is None:
        max_seq_len = prompt_tokens.shape[1] + max_new_tokens

    B = prompt_tokens.shape[0]

    # Initialize cache
    cache = init_cache(config, B, max_seq_len)

    # Prefill
    logits, cache = forward(params, prompt_tokens, config, cache=cache, is_decode=False)

    # Sample first token from last position
    first_logits = logits[:, -1, :]  # (B, vocab)
    key, subkey = jax.random.split(key)
    first_token = _sample(first_logits, temperature, top_k, subkey)  # (B,)

    # Decode loop via lax.scan
    def _decode_step(carry, _):
        token, cache_carry, rng = carry
        token_input = token[:, None]  # (B, 1)
        logits, new_cache = forward(params, token_input, config, cache=cache_carry, is_decode=True)
        next_logits = logits[:, 0, :]  # (B, vocab)
        rng, subkey = jax.random.split(rng)
        next_token = _sample(next_logits, temperature, top_k, subkey)
        return (next_token, new_cache, rng), next_token

    init_carry = (first_token, cache, key)
    _, generated = jax.lax.scan(_decode_step, init_carry, None, length=max_new_tokens - 1)

    # generated: (max_new_tokens-1, B) -> (B, max_new_tokens)
    all_tokens = jnp.concatenate([first_token[:, None], generated.T], axis=1)
    return all_tokens


def _sample(
    logits: jax.Array,
    temperature: float,
    top_k: int | None,
    key: jax.Array,
) -> jax.Array:
    """Sample from logits with temperature and optional top-k."""
    if temperature <= 0:
        return jnp.argmax(logits, axis=-1)

    logits = logits / temperature

    if top_k is not None:
        top_k_vals, _ = jax.lax.top_k(logits, top_k)
        threshold = top_k_vals[:, -1:]
        logits = jnp.where(logits >= threshold, logits, jnp.finfo(logits.dtype).min)

    return jax.random.categorical(key, logits, axis=-1)
