"""Pallas backward kernel v4 for fused_ep_moe.

Fixes all 4 v3 bugs:
  1. Weight grad accumulation: VMEM += with prev_scale=0 on first block;
     write d_w to HBM once per (expert, D-tile) after all blocks.
     (v3 bug: d_w1g_acc[...] = dot(...)  — overwrite, not accumulate)
  2. max_tpe = 2 * avg_per_expert, NOT TK.
     (v3 bug: max_tpe = cdiv(TK, bte)*bte = TK → 917 GB OOM at full scale)
  3. No (TK, D, F) intermediate for d_gating.
     (v3 bug: w1g_pairs = w1_f32[ids, 0] → (TK, D, F) → 58 TB)
  4. top_k_indices_precomputed API added.
     (v3 bug: always calls compute_routing internally → wrong routing for DSv3)

Loop structure — D-tile outermost, block innermost:
  for each expert e:
    for each D-tile dt:
      for each token block b:
        pass1: fori over ALL D-tiles → h_gate(bt,F), h_up(bt,F), d_h_act(bt,F)
        act_backward → d_h_gate, d_h_up, h_act
        pass2 for D-tile dt only:
          scale = (b != 0).astype(float32)   ← 0 on first block
          d_w1g_acc = scale*acc + tok.T @ d_h_gate   VMEM accumulate
          d_w2_acc  = scale*acc + h_act.T @ dexp     VMEM accumulate
          write d_tok_tile → sorted_d_tokens_hbm     (non-overlapping slice)
      write d_w1g_acc/d_w2_acc → HBM once per (e, dt)

EP>1: call inside shard_map; each device processes E_local experts.
      Caller must lax.psum(d_tokens, ep_axis) and lax.psum(d_gating, ep_axis).
"""

import functools
import sys

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

cdiv = pl.cdiv


# ---------------------------------------------------------------------------
# Routing helper
# ---------------------------------------------------------------------------

def _compute_routing(gating_output, top_k, scoring_fn, renormalize_topk_logits):
    if scoring_fn == "softmax":
        scores = jax.nn.softmax(gating_output.astype(jnp.float32), axis=-1)
    elif scoring_fn == "sigmoid":
        scores = jax.nn.sigmoid(gating_output.astype(jnp.float32))
    else:
        raise NotImplementedError(scoring_fn)
    top_k_weights, top_k_indices = lax.top_k(scores, top_k)
    if renormalize_topk_logits:
        top_k_weights = top_k_weights / top_k_weights.sum(axis=-1, keepdims=True)
    return top_k_indices, top_k_weights


# ---------------------------------------------------------------------------
# Pallas kernel body
# ---------------------------------------------------------------------------

def _fused_ep_moe_bwd_v4_kernel(
    # HBM inputs — bins layout: expert e occupies rows [e*max_tpe, (e+1)*max_tpe)
    bins_tokens_hbm,      # (E_local*max_tpe, D) float32
    bins_d_exp_hbm,       # (E_local*max_tpe, D) float32
    w1_hbm,               # (E_local, 2, D, F_shard)
    w2_hbm,               # (E_local, F_shard, D)
    expert_sizes_hbm,     # (E_local,) int32
    # HBM outputs (same bins layout)
    bins_d_tokens_hbm,    # (E_local*max_tpe, D) float32
    d_w1_hbm,             # (E_local, 2, D, F_shard)
    d_w2_hbm,             # (E_local, F_shard, D)
    # SMEM scratch
    expert_sizes_smem,    # (E_local,) int32
    # VMEM scratch
    tok_vmem,             # (bt, tile_D)
    dexp_vmem,            # (bt, tile_D)
    w1g_vmem,             # (tile_D, F_shard)
    w1u_vmem,             # (tile_D, F_shard)
    w2_vmem,              # (F_shard, tile_D)
    d_w1g_acc,            # (tile_D, F_shard) — weight grad VMEM accumulator
    d_w1u_acc,            # (tile_D, F_shard)
    d_w2_acc,             # (F_shard, tile_D)
    d_tok_vmem,           # (bt, tile_D) — staging buf for d_tok HBM writes
    # DMA semaphores
    sem_w, sem_tok, sem_out,
    *,
    E_local: int,
    bt: int,
    max_blocks_per_expert: int,
    max_tpe: int,
    tile_D: int,
    n_D_tiles: int,
    act_fn: str,
):
    # Grid dimension 0 = expert id.  The outer E_local loop is lifted into the
    # Pallas grid so Mosaic sees a simpler per-expert kernel body (3 nested loops
    # instead of 4).  This reduces Mosaic compile time ~16× vs. the previous
    # single grid-cell design with lax.fori_loop(0, E_local, ...).
    #
    # tok_start = e_id * max_tpe + blk_id * bt
    # max_tpe is a multiple of bt, bt is a multiple of 8 → tok_start always 8-aligned ✓
    e_id   = pl.program_id(0)
    F_shard = w1_hbm.shape[-1]

    # Load expert_sizes HBM → SMEM (fence via self-DMA wait)
    pltpu.make_async_copy(expert_sizes_hbm, expert_sizes_smem, sem_w).start()
    pltpu.make_async_copy(expert_sizes_smem, expert_sizes_smem, sem_w).wait()
    size = expert_sizes_smem[e_id]

    def _act_bwd(h_gate, h_up, d_h_act):
        """Activation backward for silu."""
        sig      = jax.nn.sigmoid(h_gate)
        silu_g   = sig * (1.0 + h_gate * (1.0 - sig))
        h_act    = jax.nn.silu(h_gate) * h_up
        d_h_up   = d_h_act * jax.nn.silu(h_gate)
        d_h_gate = d_h_act * h_up * silu_g
        return h_act, d_h_gate, d_h_up

    def process_dt(d_tile, _):
        """D-tile loop: accumulate d_w for one tile, write once after all blocks."""
        d_start = d_tile * tile_D

        # Zero-initialize VMEM accumulators for this D-tile.
        # Without this, uninitialized VMEM is NaN → 0 * NaN = NaN in first block.
        d_w1g_acc[...] = jnp.zeros((tile_D, F_shard), jnp.float32)
        d_w1u_acc[...] = jnp.zeros((tile_D, F_shard), jnp.float32)
        d_w2_acc[...]  = jnp.zeros((F_shard, tile_D), jnp.float32)

        def process_block(blk_id, _):
            # Bins layout: expert e starts at e_id * max_tpe. Always bt-aligned.
            tok_start = e_id * max_tpe + blk_id * bt
            remaining = size - blk_id * bt
            valid_f   = (jnp.arange(bt) < remaining).astype(jnp.float32)  # (bt,)

            # ---- Pass 1: accumulate h_gate, h_up, d_h_act over ALL D-tiles ----
            def accum_all_dt(dt_, carry):
                h_g_acc, h_u_acc, dha_acc = carry
                ds = dt_ * tile_D

                pltpu.make_async_copy(
                    bins_tokens_hbm.at[pl.ds(tok_start, bt), pl.ds(ds, tile_D)],
                    tok_vmem, sem_tok).start()
                pltpu.make_async_copy(
                    bins_d_exp_hbm.at[pl.ds(tok_start, bt), pl.ds(ds, tile_D)],
                    dexp_vmem, sem_tok).start()
                pltpu.make_async_copy(
                    w1_hbm.at[e_id, 0, pl.ds(ds, tile_D)], w1g_vmem, sem_w).start()
                pltpu.make_async_copy(
                    w1_hbm.at[e_id, 1, pl.ds(ds, tile_D)], w1u_vmem, sem_w).start()
                pltpu.make_async_copy(
                    w2_hbm.at[e_id, :, pl.ds(ds, tile_D)], w2_vmem, sem_w).start()
                pltpu.make_async_copy(tok_vmem,  tok_vmem,  sem_tok).wait()
                pltpu.make_async_copy(dexp_vmem, dexp_vmem, sem_tok).wait()
                pltpu.make_async_copy(w1g_vmem,  w1g_vmem,  sem_w).wait()
                pltpu.make_async_copy(w1u_vmem,  w1u_vmem,  sem_w).wait()
                pltpu.make_async_copy(w2_vmem,   w2_vmem,   sem_w).wait()

                vf   = lax.broadcast_in_dim(valid_f, (bt, tile_D), (0,))
                tok  = vf * tok_vmem[...]    # (bt, tile_D)
                dexp = vf * dexp_vmem[...]   # (bt, tile_D)
                wg   = w1g_vmem[...]         # (tile_D, F_shard)
                wu   = w1u_vmem[...]         # (tile_D, F_shard)
                wd   = w2_vmem[...]          # (F_shard, tile_D)

                return (
                    h_g_acc + jnp.dot(tok, wg),    # (bt, F_shard)
                    h_u_acc + jnp.dot(tok, wu),    # (bt, F_shard)
                    dha_acc + jnp.dot(dexp, wd.T), # (bt, F_shard)
                )

            zero_bF = jnp.zeros((bt, F_shard), jnp.float32)
            h_gate, h_up, d_h_act = lax.fori_loop(
                0, n_D_tiles, accum_all_dt, (zero_bF, zero_bF, zero_bF))

            # ---- Activation backward ----
            if act_fn == "silu":
                h_act, d_h_gate, d_h_up = _act_bwd(h_gate, h_up, d_h_act)
            else:
                raise NotImplementedError(act_fn)

            # ---- Pass 2: accumulate d_w for D-tile d_start; write d_tok ----
            pltpu.make_async_copy(
                bins_tokens_hbm.at[pl.ds(tok_start, bt), pl.ds(d_start, tile_D)],
                tok_vmem, sem_tok).start()
            pltpu.make_async_copy(
                bins_d_exp_hbm.at[pl.ds(tok_start, bt), pl.ds(d_start, tile_D)],
                dexp_vmem, sem_tok).start()
            pltpu.make_async_copy(
                w1_hbm.at[e_id, 0, pl.ds(d_start, tile_D)], w1g_vmem, sem_w).start()
            pltpu.make_async_copy(
                w1_hbm.at[e_id, 1, pl.ds(d_start, tile_D)], w1u_vmem, sem_w).start()
            pltpu.make_async_copy(tok_vmem,  tok_vmem,  sem_tok).wait()
            pltpu.make_async_copy(dexp_vmem, dexp_vmem, sem_tok).wait()
            pltpu.make_async_copy(w1g_vmem,  w1g_vmem,  sem_w).wait()
            pltpu.make_async_copy(w1u_vmem,  w1u_vmem,  sem_w).wait()

            vf  = lax.broadcast_in_dim(valid_f, (bt, tile_D), (0,))
            tok  = vf * tok_vmem[...]    # (bt, tile_D)
            dexp = vf * dexp_vmem[...]   # (bt, tile_D)
            wg   = w1g_vmem[...]         # (tile_D, F_shard)
            wu   = w1u_vmem[...]         # (tile_D, F_shard)

            # Simple accumulate: accumulators were zero-initialized by process_dt.
            d_w1g_acc[...] = d_w1g_acc[...] + jnp.dot(tok.T, d_h_gate)
            d_w1u_acc[...] = d_w1u_acc[...] + jnp.dot(tok.T, d_h_up)
            d_w2_acc[...]  = d_w2_acc[...]  + jnp.dot(h_act.T, dexp)

            # Write d_tok for this (block, D-tile) — non-overlapping HBM slice
            d_tok_vmem[...] = vf * (jnp.dot(d_h_gate, wg.T) + jnp.dot(d_h_up, wu.T))
            pltpu.make_async_copy(
                d_tok_vmem,
                bins_d_tokens_hbm.at[pl.ds(tok_start, bt), pl.ds(d_start, tile_D)],
                sem_out).start()
            # Fence: ensure write is done before next block reuses d_tok_vmem
            pltpu.make_async_copy(
                bins_d_tokens_hbm.at[pl.ds(tok_start, bt), pl.ds(d_start, tile_D)],
                bins_d_tokens_hbm.at[pl.ds(tok_start, bt), pl.ds(d_start, tile_D)],
                sem_out).wait()

        lax.fori_loop(0, max_blocks_per_expert, process_block, None)

        # Write d_w for this (expert, D-tile) once — after all blocks
        pltpu.make_async_copy(
            d_w1g_acc,
            d_w1_hbm.at[e_id, 0, pl.ds(d_start, tile_D)], sem_out).start()
        pltpu.make_async_copy(
            d_w1u_acc,
            d_w1_hbm.at[e_id, 1, pl.ds(d_start, tile_D)], sem_out).start()
        pltpu.make_async_copy(
            d_w2_acc,
            d_w2_hbm.at[e_id, :, pl.ds(d_start, tile_D)], sem_out).start()
        pltpu.make_async_copy(
            d_w1_hbm.at[e_id, 0, pl.ds(d_start, tile_D)],
            d_w1_hbm.at[e_id, 0, pl.ds(d_start, tile_D)], sem_out).wait()
        pltpu.make_async_copy(
            d_w1_hbm.at[e_id, 1, pl.ds(d_start, tile_D)],
            d_w1_hbm.at[e_id, 1, pl.ds(d_start, tile_D)], sem_out).wait()
        pltpu.make_async_copy(
            d_w2_hbm.at[e_id, :, pl.ds(d_start, tile_D)],
            d_w2_hbm.at[e_id, :, pl.ds(d_start, tile_D)], sem_out).wait()

    lax.fori_loop(0, n_D_tiles, process_dt, None)


# ---------------------------------------------------------------------------
# Public wrapper
# ---------------------------------------------------------------------------

def fused_ep_moe_bwd_v4(
    d_out,            # (T, D) float32
    tokens,           # (T, D)
    w1,               # (E_local, 2, D, F_shard) — gate[0]+up[1], FSDP-sharded
    w2,               # (E_local, F_shard, D)
    gating_output,    # (T, E_global) raw logits — or None if precomputed routing given
    top_k: int,
    *,
    scoring_fn: str = "sigmoid",
    renormalize_topk_logits: bool = True,
    act_fn: str = "silu",
    ep_axis_name: str = "model",
    bt: int | None = None,
    tile_D: int | None = None,
    vmem_limit_bytes: int = 30 * 1024 * 1024,
    top_k_indices_precomputed=None,  # (T, K) int32 global expert IDs
    top_k_weights_precomputed=None,  # (T, K) float32
    return_dtopk: bool = False,
    E_global_override: int | None = None,
):
    """Pallas v4 backward for fused_ep_moe.

    EP=1: call directly (processes all E experts on single device).
    EP>1: call inside shard_map; each device handles E_local = E_global/EP experts.
          Returns *partial* gradients — caller does lax.psum(d_tokens, ep_axis).

    Returns (d_tokens, d_w1, d_w2, d_gating) or with return_dtopk=True:
            (d_tokens, d_w1, d_w2, d_top_k_weights).

    w1 layout:  (E_local, 2, D, F_shard)  — gate[0] + up[1], FSDP-sharded on axis 3.
    w2 layout:  (E_local, F_shard, D)     — FSDP-sharded on axis 1.
    """
    T, D = tokens.shape
    E_local, _, _, F_shard = w1.shape
    if gating_output is not None:
        E_global = gating_output.shape[1]
    else:
        assert E_global_override is not None
        E_global = E_global_override

    assert F_shard % 128 == 0, (
        f"F_shard={F_shard} must be 128-aligned (HBM tile constraint). "
        f"Use FSDP ≤ F//128 to reduce F_shard to a valid size.")

    tokens_f32 = tokens.astype(jnp.float32)
    w1_f32     = w1.astype(jnp.float32)
    w2_f32     = w2.astype(jnp.float32)
    d_out_f32  = d_out.astype(jnp.float32)

    # ---- 1. Routing ----
    if top_k_indices_precomputed is not None:
        top_k_indices = top_k_indices_precomputed.astype(jnp.int32)
        top_k_weights = top_k_weights_precomputed.astype(jnp.float32)
    else:
        gating_f32    = gating_output.astype(jnp.float32)
        top_k_indices, top_k_weights = _compute_routing(
            gating_f32, top_k, scoring_fn, renormalize_topk_logits)

    # ---- 1b. EP>1: remap global IDs → local; zero non-local slots ----
    ep_sharded = (E_local < E_global)
    if ep_sharded:
        device_id     = lax.axis_index(ep_axis_name)
        expert_offset = jnp.int32(device_id) * jnp.int32(E_local)
        is_local      = (top_k_indices >= expert_offset) & (
                         top_k_indices < expert_offset + E_local)
        is_local_flat = is_local.reshape(-1)
        top_k_indices_kernel = jnp.where(is_local, top_k_indices - expert_offset, E_local)
        top_k_weights_kernel = jnp.where(is_local, top_k_weights, 0.0)
    else:
        expert_offset        = jnp.int32(0)
        is_local_flat        = jnp.ones(T * top_k, dtype=bool)
        top_k_indices_kernel = top_k_indices
        top_k_weights_kernel = top_k_weights

    TK = T * top_k

    # ---- 2. Sort tokens by local expert ----
    expert_ids_flat = top_k_indices_kernel.reshape(TK)
    token_ids_flat  = jnp.repeat(jnp.arange(T, dtype=jnp.int32), top_k)
    k_ids_flat      = jnp.tile(jnp.arange(top_k, dtype=jnp.int32), T)
    flat_weights    = top_k_weights_kernel[token_ids_flat, k_ids_flat]  # (TK,)

    sort_order        = jnp.argsort(expert_ids_flat, stable=True)  # (TK,)
    sorted_expert_ids = expert_ids_flat[sort_order]                # (TK,)
    sorted_token_ids  = token_ids_flat[sort_order]                 # (TK,)
    d_exp_weights     = flat_weights[sort_order]                   # (TK,)
    is_local_sorted   = is_local_flat[sort_order]                  # (TK,) bool

    expert_sizes  = jnp.bincount(sorted_expert_ids, length=E_local).astype(jnp.int32)
    expert_starts = jnp.concatenate([
        jnp.zeros(1, jnp.int32),
        jnp.cumsum(expert_sizes[:-1]),
    ])  # (E_local,)

    # ---- 3. Block size bt (must be multiple of 8 for HBM tile alignment) ----
    if bt is None:
        avg_per_expert = max(cdiv(TK, E_local), 1)
        bt = min(avg_per_expert, 128)
        bt = max((bt // 8) * 8, 8)
    else:
        assert bt % 8 == 0, f"bt={bt} must be a multiple of 8 (HBM tile alignment)"

    # ---- 4. D-tile size ----
    if tile_D is None:
        # Conservative estimate: 9 VMEM buffers of shape (tile_D, F_shard) or (F_shard, tile_D)
        # plus (bt, tile_D) buffers. Budget ~50% of vmem_limit to weight/grad tiles.
        bytes_per_tileD = 9 * F_shard * 4 + 4 * bt * 4  # per unit of tile_D
        max_tile_D = vmem_limit_bytes // (2 * bytes_per_tileD)
        cand = (max_tile_D // 128) * 128
        # Find largest multiple of 128 that divides D
        while cand > 0 and D % cand != 0:
            cand -= 128
        tile_D = cand if cand >= 128 else D
        tile_D = min(tile_D, D)

    assert D % tile_D == 0, f"tile_D={tile_D} must divide D={D}"
    n_D_tiles = D // tile_D

    # ---- 5. Padding: TK_padded = E_local * max_tpe ----
    # Budget per expert slot: 2× the global average (TK / E_global).
    #
    # At EP>1 each local expert only receives E_local/E_global ≈ 1/16 of all
    # token-expert pairs.  max_tpe must be sized against the *global* average
    # (TK / E_global), NOT the naïve TK / E_local which is 16× too large.
    #
    # Correct formula:
    #   avg_per_expert = TK / E_global           (expected local assignments)
    #   max_tpe        = 2 × avg_per_expert      (2× safety for load imbalance)
    #
    # At EP=16 (E_local=16, E_global=256, TK=524288):
    #   max_tpe = 2 × (524288/256) = 4096   → TK_padded = 16×4096 = 65536
    #   bins = (65536, 7168) f32 = 1.88 GB each  (vs 22.6 GB with the wrong formula)
    #
    # At EP=1 (E_local=E_global): formula is identical to the old E_local version.
    # DSv3 load-balancing loss keeps per-expert load within 1.3× avg in practice.
    avg_tpe    = max(cdiv(TK, E_global) * 2, 128)  # 2× global avg per expert
    max_tpe    = cdiv(avg_tpe, bt) * bt
    max_blocks = max_tpe // bt
    TK_padded  = E_local * max_tpe

    # ---- 6. Build bins layout for tokens and d_exp ----
    # Scatter sorted tokens into bins: expert e → rows [e*max_tpe, (e+1)*max_tpe).
    # This ensures tok_start = e*max_tpe + blk_id*bt is always bt-aligned (HBM tile req).
    sorted_tokens = tokens_f32[sorted_token_ids]                          # (TK, D)
    d_exp_flat    = d_out_f32[token_ids_flat] * flat_weights[:, None]     # (TK, D)
    d_sorted_exp  = d_exp_flat[sort_order]                                # (TK, D)

    expert_starts_per_slot = expert_starts[sorted_expert_ids]             # (TK,)
    local_indices          = jnp.arange(TK, dtype=jnp.int32) - expert_starts_per_slot
    bin_positions          = sorted_expert_ids * max_tpe + local_indices  # (TK,) ∈ [0, TK_padded)

    bins_tokens = jnp.zeros((TK_padded, D), jnp.float32).at[bin_positions].set(sorted_tokens)
    bins_d_exp  = jnp.zeros((TK_padded, D), jnp.float32).at[bin_positions].set(d_sorted_exp)

    # ---- 7. Pallas kernel ----
    hbm = pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM)

    bwd_kernel = pl.pallas_call(
        functools.partial(
            _fused_ep_moe_bwd_v4_kernel,
            E_local=E_local,
            bt=bt,
            max_blocks_per_expert=max_blocks,
            max_tpe=max_tpe,
            tile_D=tile_D,
            n_D_tiles=n_D_tiles,
            act_fn=act_fn,
        ),
        out_shape=[
            jax.ShapeDtypeStruct((TK_padded, D),           jnp.float32),  # bins_d_tokens
            jax.ShapeDtypeStruct((E_local, 2, D, F_shard), jnp.float32),  # d_w1
            jax.ShapeDtypeStruct((E_local, F_shard, D),    jnp.float32),  # d_w2
        ],
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=(E_local,),
            in_specs=[hbm] * 5,
            out_specs=[hbm] * 3,
            scratch_shapes=[
                pltpu.SMEM((E_local,), jnp.int32),             # expert_sizes_smem
                pltpu.VMEM((bt, tile_D), jnp.float32),         # tok_vmem
                pltpu.VMEM((bt, tile_D), jnp.float32),         # dexp_vmem
                pltpu.VMEM((tile_D, F_shard), jnp.float32),    # w1g_vmem
                pltpu.VMEM((tile_D, F_shard), jnp.float32),    # w1u_vmem
                pltpu.VMEM((F_shard, tile_D), jnp.float32),    # w2_vmem
                pltpu.VMEM((tile_D, F_shard), jnp.float32),    # d_w1g_acc
                pltpu.VMEM((tile_D, F_shard), jnp.float32),    # d_w1u_acc
                pltpu.VMEM((F_shard, tile_D), jnp.float32),    # d_w2_acc
                pltpu.VMEM((bt, tile_D), jnp.float32),         # d_tok_vmem
                pltpu.SemaphoreType.DMA,                        # sem_w
                pltpu.SemaphoreType.DMA,                        # sem_tok
                pltpu.SemaphoreType.DMA,                        # sem_out
            ],
        ),
        compiler_params=pltpu.CompilerParams(vmem_limit_bytes=vmem_limit_bytes),
        name=f"fused-moe-bwd-v4-E{E_local}-D{D}-F{F_shard}-tD{tile_D}-bt{bt}",
    )

    bins_d_tokens, d_w1, d_w2 = bwd_kernel(
        bins_tokens,
        bins_d_exp,
        w1_f32,
        w2_f32,
        expert_sizes,
    )

    # ---- 8. Gather and unsort d_tokens ----
    # bins_d_tokens[bin_positions] → sorted order; segment_sum → original token order
    d_sorted_tokens = bins_d_tokens[bin_positions]                         # (TK, D)
    d_tokens = jax.ops.segment_sum(d_sorted_tokens, sorted_token_ids, T)  # (T, D)

    # ---- 9. d_gating via per-expert loop ----
    # Uses 1D bin index arrays (cheap — no (TK,D) data materialization).
    # bin_positions already computed in step 6.
    bins_tok_ids  = jnp.zeros(TK_padded, jnp.int32  ).at[bin_positions].set(sorted_token_ids)
    bins_weights  = jnp.zeros(TK_padded, jnp.float32).at[bin_positions].set(d_exp_weights)
    bins_is_local = jnp.zeros(TK_padded, bool        ).at[bin_positions].set(is_local_sorted)
    bins_valid    = jnp.zeros(TK_padded, bool         ).at[bin_positions].set(
                        jnp.ones(TK, bool))

    def _silu_grad(x):
        sig = jax.nn.sigmoid(x)
        return sig * (1.0 + x * (1.0 - sig))

    d_tw_bin_list = []
    for e in range(E_local):
        start_e    = e * max_tpe
        tok_ids_e  = lax.dynamic_slice(bins_tok_ids,  (start_e,), (max_tpe,))
        weights_e  = lax.dynamic_slice(bins_weights,  (start_e,), (max_tpe,))
        is_local_e = lax.dynamic_slice(bins_is_local, (start_e,), (max_tpe,))
        valid_e    = lax.dynamic_slice(bins_valid,    (start_e,), (max_tpe,)).astype(jnp.float32)

        toks_e   = tokens_f32[tok_ids_e] * valid_e[:, None]    # (max_tpe, D)
        d_out_e  = d_out_f32[tok_ids_e]                        # (max_tpe, D) raw upstream grad

        h_g = toks_e @ w1_f32[e, 0]
        h_u = toks_e @ w1_f32[e, 1]
        if act_fn == "silu":
            h_act = jax.nn.silu(h_g) * h_u
        else:
            raise NotImplementedError(act_fn)
        out_e = h_act @ w2_f32[e]                             # (max_tpe, D)

        # d_routing_weights[bin_pos] = dot(d_out, expert_out) — only for local, valid slots
        d_tw_e = ((d_out_e * out_e * valid_e[:, None]).sum(-1)
                  * is_local_e.astype(jnp.float32))            # (max_tpe,)
        d_tw_bin_list.append(d_tw_e)

    d_tw_padded = jnp.concatenate(d_tw_bin_list, axis=0)      # (TK_padded,) in bin order

    # Invert bin scatter: gather from bin layout → sorted order → original flat order
    d_tw_sorted     = d_tw_padded[bin_positions]               # (TK,) sorted order
    sort_order_inv  = jnp.argsort(sort_order)
    d_tw_flat       = d_tw_sorted[sort_order_inv]              # (TK,) original (token, k) order
    d_top_k_weights = d_tw_flat.reshape(T, top_k)             # (T, K)
    if ep_sharded:
        # Non-local slots may have gathered garbage from OOB bin positions; zero them.
        d_top_k_weights = jnp.where(is_local, d_top_k_weights, 0.0)

    if return_dtopk:
        return d_tokens, d_w1, d_w2, d_top_k_weights

    # ---- 10. Routing backward → d_gating ----
    assert gating_output is not None
    from backward import moe_bwd_routing
    gating_f32    = gating_output.astype(jnp.float32)
    gating_scores = (jax.nn.softmax(gating_f32, axis=-1) if scoring_fn == "softmax"
                     else jax.nn.sigmoid(gating_f32))
    d_gating = moe_bwd_routing(
        d_top_k_weights, top_k_indices, top_k_weights,
        gating_scores, gating_f32,
        scoring_fn=scoring_fn,
        renormalize_topk_logits=renormalize_topk_logits,
    )

    return d_tokens, d_w1, d_w2, d_gating
