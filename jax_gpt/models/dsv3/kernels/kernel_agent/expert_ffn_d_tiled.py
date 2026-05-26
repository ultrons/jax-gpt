"""DSv3 fused EP-MoE — v_outside D-tiled expert FFN for full D=7168.

The default `expert_ffn_v_outside` kernel holds the FULL per-device W1
`(E_local, D, 2F)` in VMEM as a BlockSpec window. At E_local=8 D=3840
F=128 that's ~16 MB — right at the VMEM cap. D=7168 needs ~29 MB.

D.6 splits both the contraction dim (E_local) AND the OUTPUT D dim
via the grid:

  Grid = (num_bt, E_local, num_d_out)  — inner axis = d_out

Per-grid-step VMEM at D=7168, F=128, bt=128, D_tile=1024:
  W1 block        (1, D_full, 2F)  bf16  ≈ 3.7 MB  (changes only on e step)
  W_d block       (1, F, D_tile)    bf16  ≈ 256 KB (changes per d step)
  tok block       (bt, D_full)      bf16  ≈ 1.8 MB (changes only on i step)
  out block       (bt, D_tile)      f32  ≈ 512 KB (×2 buf ≈ 1 MB)
  act_scratch     (bt, F)           f32  = 64 KB  (cached across d axis)
                                                  ─────
                                       total ≈ 7-9 MB (well under 16 MB)

The out block is read-modify-write across the E_local axis: at e=0 we
initialize from zero, at e>0 we accumulate. Mosaic's double-buffered
output handles the RMW automatically.

`act_scratch` is computed once per (bt, e_local) pair at d_out=0 and
re-used across all d_out tiles to avoid `num_d_out`× redundant
up-matmul work.

Frontmatter:
  slug: dsv3-v-outside-expert-ffn-d-tiled
  intent: kernel
  status: v0 (D.6) — true D-tiling for full D=7168
  sources:
    - build/v_outside/expert_ffn.py (the structural base)
    - distilled/patterns/pallas-call-skeleton.md
"""
from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def _expert_ffn_d_tiled_body(
    # HBM input refs (BlockSpec-windowed):
    tok_ref,             # (bt, D)        bf16 — changes only on i axis
    eids_ref,            # (bt,)          int32
    W1_ref,              # (1, D, 2F)     bf16 — changes only on e axis
    W_d_ref,             # (1, F, D_tile) bf16 — changes per d_out step
    # HBM output ref:
    out_ref,             # (bt, D_tile)   f32  — RMW across E axis
    # Scratches:
    act_scratch,         # (bt, F)        f32  — cached across d_out steps
    *,
    E_local: int,
    num_d_out: int,
    bt: int,
    D: int,
    D_tile: int,
    F: int,
):
    """One (bt-tile, expert, d_out-tile) per grid step.

    Grid axes (i, e, d):
      i = bt-tile, e = local expert id, d = output D-tile.
    Inner axis = d, so W1[e] stays in VMEM across d steps and act_scratch
    is computed once per (i, e) pair (at d=0) and re-used.
    """
    e_local = pl.program_id(1)
    d_idx   = pl.program_id(2)
    is_first_d = d_idx == 0
    is_first_e = e_local == 0

    # ---- d == 0: refresh act for this expert ----
    @pl.when(is_first_d)
    def _refresh_act():
        tok = tok_ref[...].astype(jnp.float32)            # (bt, D)
        w1_e = W1_ref[0].astype(jnp.float32)              # (D, 2F)
        gate_up = tok @ w1_e                              # (bt, 2F)
        gate, up = jnp.split(gate_up, 2, axis=-1)         # (bt, F) each
        act_scratch[...] = jax.nn.silu(gate) * up         # (bt, F)

    # ---- d-tile down matmul ----
    act = act_scratch[...]                                # (bt, F) f32
    w_d_e_tile = W_d_ref[0].astype(jnp.float32)           # (F, D_tile)
    out_e_tile = act @ w_d_e_tile                         # (bt, D_tile)

    # Per-row mask vs `e_local` (LOCAL expert id; caller `local_permute`
    # converts global → local before invoking the FFN).
    eids = eids_ref[...]                                  # (bt,) int32
    mask_bt = (1 - jnp.minimum(jnp.abs(eids - e_local), 1)).astype(jnp.float32)
    mask_bd = lax.broadcast_in_dim(mask_bt, (bt, D_tile),
                                    broadcast_dimensions=(0,))
    masked_contrib = mask_bd * out_e_tile                 # (bt, D_tile) f32

    # ---- Accumulate into out tile (RMW across E_local axis) ----
    # At e=0 we initialize from zero; at e>0 we accumulate prior value.
    @pl.when(is_first_e)
    def _init():
        out_ref[...] = masked_contrib

    @pl.when(jnp.logical_not(is_first_e))
    def _accum():
        out_ref[...] = out_ref[...] + masked_contrib


def _expert_ffn_d_tiled_grid(
    sorted_tokens, sorted_eids, W1, W_d, *, bt: int, D_tile: int,
):
    """D.6 D-tiled kernel. Grid = (num_bt, E_local, num_d_out).

    D_tile chooses the output D-tile size. Must evenly divide D.
    """
    M, D = sorted_tokens.shape
    E_local, _, twoF = W1.shape
    F = twoF // 2
    num_bt = M // bt
    assert D % D_tile == 0, f"D={D} must be divisible by D_tile={D_tile}"
    num_d_out = D // D_tile

    return pl.pallas_call(
        functools.partial(_expert_ffn_d_tiled_body,
                          E_local=E_local, num_d_out=num_d_out,
                          bt=bt, D=D, D_tile=D_tile, F=F),
        grid=(num_bt, E_local, num_d_out),
        in_specs=[
            # tokens: change only on i axis. Index e/d as 0 → same block.
            pl.BlockSpec((bt, D),       lambda i, e, d: (i, 0)),
            pl.BlockSpec((bt,),         lambda i, e, d: (i,)),
            # W1[e]: change only on e axis.
            pl.BlockSpec((1, D, 2 * F), lambda i, e, d: (e, 0, 0)),
            # W_d[e, :, d_tile]: change on both e and d axes.
            pl.BlockSpec((1, F, D_tile), lambda i, e, d: (e, 0, d)),
        ],
        # Output: D-tiled, changes per (i, d). RMW across e_local axis.
        out_specs=pl.BlockSpec((bt, D_tile),
                                lambda i, e, d: (i, d)),
        out_shape=jax.ShapeDtypeStruct((M, D), jnp.float32),
        scratch_shapes=[
            pltpu.VMEM((bt, F), jnp.float32),    # act_scratch
        ],
    )(sorted_tokens, sorted_eids, W1, W_d)


def expert_ffn_v_outside_d_tiled(
    sorted_tokens: jax.Array,    # (M, D) bf16
    sorted_eids: jax.Array,      # (M,) int32 — local expert id (0..E_local-1)
    W1: jax.Array,               # (E_local, D, 2F) bf16
    W_d: jax.Array,              # (E_local, F, D)  bf16
    *,
    bt: int = 128,
    D_tile: int | None = None,
) -> jax.Array:
    """D.6 truly D-tiled variant — fits full DSv3 D=7168.

    Drop-in replacement for `expert_ffn_v_outside(..., impl='grid')`.

    D_tile defaults to the largest power-of-2 ≤ 1024 that divides D.
    Smaller D_tile → more grid steps but lower peak VMEM.
    """
    M, D = sorted_tokens.shape
    E_local, _, twoF = W1.shape
    F = twoF // 2
    assert M % bt == 0, f"M={M} must be divisible by bt={bt}"
    assert sorted_eids.shape == (M,)
    assert W_d.shape == (E_local, F, D)
    assert bt >= 128, "D-tiled grid requires bt >= 128 (Mosaic rank-1 BlockSpec)"

    if D_tile is None:
        # Pick the largest power-of-2 D_tile ≤ 1024 that divides D.
        for cand in (1024, 512, 256, 128):
            if D % cand == 0:
                D_tile = cand
                break
        assert D_tile is not None, f"no D_tile candidate divides D={D}"

    return _expert_ffn_d_tiled_grid(
        sorted_tokens, sorted_eids, W1, W_d, bt=bt, D_tile=D_tile)
