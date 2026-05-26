"""DSv3 fused EP-MoE — v_outside D+F-tiled expert FFN for production F=2048.

D.6 closes D=7168 by tiling the OUTPUT D dim, but it still holds the full
per-expert W1 window `(1, D, 2F)` in VMEM. At production F=2048 that window
is `(1, 7168, 4096) bf16 = 56 MiB`, which double-buffered exceeds the v7x
64 MiB VMEM cap (the autoperf agent reproduced this OOM at
E=256 D=7168 F=2048 K=8 on tpu7x:4x8x8).

D.7 adds an **F-output tile axis** to the D.6 grid. The PUBLIC W1 layout
is `(E_local, D, 2, F)` (matches the Megatron wrapper). Internally the
kernel transposes W1 to `(E_local, 2, D, F)` before the `pallas_call`,
moving the size-2 (gate/up) axis from position 2 to position 1 so it is
*never* between the sublane axis (D) and the multi-lane-block axis
(F_tile). This is the D.7-correctness-fix (2026-05-22): the original
`(1, D, 2, F_tile)` BlockSpec compiled cleanly but produced WRONG
results at cluster scale when F_tile spanned >1 lane block (F_tile=256
at v7x), because Mosaic's slicing of the size-2 axis between the D-row
sublane and F_tile-column-lane-block boundary was mis-lowered. With
the `(E, 2, D, F)` internal layout the BlockSpec is `(1, 2, D, F_tile)`
and the body slices the size-2 axis at position 1 (W1_int_ref[0, 0]
vs W1_int_ref[0, 1]) — the trailing two dims are the clean
sublane×lane `(D, F_tile)` pair Mosaic expects.

Grid = `(num_bt, E_local, num_f_tile, num_d_out)` with **d innermost**:
- W1_int block changes on `f` (and `e`).
- W_d block changes on `f` and `d`.
- `act_scratch (bt, F_tile) f32` is computed once per `(i, e, f)` tuple
  at `d_idx == 0` and re-used across all `d` tiles.
- Output `(bt, D_tile) f32` is **RMW** across BOTH the `E_local` axis
  AND the F-tile axis (each (i, d) block is touched `E_local *
  num_f_tile` times). Initialise on `(e == 0) & (f == 0)`; accumulate
  otherwise.

VMEM budget at production (E_local=64, D=7168, F=2048, bt=128, F_tile=256,
D_tile=1024) — bf16 weights, f32 act/out (double-buffered W blocks):

```
W1_int block   (1, 2, 7168, 256) bf16   ≈  7.0 MB  ×2 buf ≈ 14 MB
W_d block      (1, 256, 1024)    bf16   ≈  0.5 MB  ×2 buf ≈  1 MB
tok block      (128, 7168)       bf16   ≈  1.8 MB
out block      (128, 1024)       f32    ≈  0.5 MB  ×2 buf ≈  1 MB
act_scratch    (128, 256)        f32    ≈  0.125 MB
internal matmul scratch                    ≈ 30-32 MB (Mosaic accumulators)
                                    total ≈ 48-52 MB  (<64 MB cap)
```

The "internal matmul scratch" is the Mosaic-generated f32 accumulator
+ sublane-replication padding for the gate+up matmul. It scales with
the matmul output size (bt × 2 × F_tile) so cutting F_tile in half
roughly halves this term too.

At F_tile=F (no actual F-tiling) the kernel degenerates to D.6 modulo
the layout change; this is the small-shape cross-check entry point.

Frontmatter:
  slug: dsv3-v-outside-expert-ffn-f-tiled
  intent: kernel
  status: v1 (D.7-correctness-fix 2026-05-22) — (E, 2, D, F) internal layout
  sources:
    - build/v_outside/expert_ffn_d_tiled.py (D.6 base)
    - build/v_inside/moe_block_ep_megatron.py (origin of (E,D,2,F) public layout)
    - distilled/patterns/pallas-call-skeleton.md
    - distilled/debugging-runbooks/size2-axis-multi-lane-block.md  (to be authored)
"""
from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def _expert_ffn_f_tiled_body(
    # HBM input refs (BlockSpec-windowed):
    tok_ref,             # (bt, D)              bf16 — changes only on i axis
    eids_ref,            # (bt,)                int32
    W1_ref,              # (1, 2, D, F_tile)    bf16 — changes on (e, f)
    W_d_ref,             # (1, F_tile, D_tile)  bf16 — changes per (e, f, d) step
    # HBM output ref:
    out_ref,             # (bt, D_tile)         f32  — RMW across E AND F axes
    # Scratches:
    act_scratch,         # (bt, F_tile)         f32  — cached across d_out steps
    *,
    E_local: int,
    num_f_tile: int,
    num_d_out: int,
    bt: int,
    D: int,
    D_tile: int,
    F: int,
    F_tile: int,
):
    """One (bt-tile, d-tile, expert, f-tile) per grid step.

    Grid axes (i, d, e, f), d OUTER and f innermost:
      i = bt-tile, d = output D-tile, e = local expert id, f = F-tile.

    Output blocks (i, d) RMW across both e and f. With d OUTER, all
    (E_local × num_f_tile) accumulations into a given (i, d) block happen
    consecutively; the block stays in Pallas's double-buffered VMEM
    throughout. Then the grid advances to the next (i, d).

    D.7-correctness-fix (2026-05-23): the original grid order (d
    INNERMOST) caused the same (i, d) output block to be revisited
    non-monotonically across the (e, f) loop, with `num_d_out - 1`
    different output blocks written between consecutive revisits.
    Mosaic mishandled the HBM coherence for these non-monotonic revisits
    when num_d_out >= 4 (D >= 4096), producing wrong accumulations
    proportional to num_d_out. With d OUTER, output traversal is
    monotonic and correctness holds at all num_d_out.

    Cost of the order swap: the previous (d-innermost) order cached
    `act = silu(gate)*up` once per (i, e, f) and reused it across
    d-tiles (num_d_out down-matmuls share one up-matmul). With d outer,
    every grid step recomputes act → num_d_out× redundant up-matmul
    work. For D=7168 num_d_out=7 → 7× up-side overhead. Acceptable
    trade since up-matmul is small relative to down (F_tile dim is
    smaller than D_tile).

    W1 layout INSIDE the kernel: `(1, 2, D, F_tile)` — size-2 (gate/up)
    axis is at position 1, i.e. OUTSIDE the trailing `(D, F_tile)` pair
    that Mosaic maps to sublane×lane. The caller (`expert_ffn_v_outside_f_tiled`)
    transposes the public `(E_local, D, 2, F)` weight to `(E_local, 2, D, F)`
    once before the grid.
    """
    e_local = pl.program_id(2)
    f_idx   = pl.program_id(3)
    is_first_ef = jnp.logical_and(e_local == 0, f_idx == 0)

    # Always refresh act for this (e, f) — with d OUTER we never revisit
    # the same (e, f) for a different d, so caching across d is moot.
    tok = tok_ref[...].astype(jnp.float32)                 # (bt, D)
    gate_w = W1_ref[0, 0, :, :].astype(jnp.float32)        # (D, F_tile)
    up_w   = W1_ref[0, 1, :, :].astype(jnp.float32)        # (D, F_tile)
    act_scratch[...] = jax.nn.silu(tok @ gate_w) * (tok @ up_w)
    act = act_scratch[...]                                 # (bt, F_tile) f32
    w_d_e_tile = W_d_ref[0].astype(jnp.float32)            # (F_tile, D_tile)
    out_e_f_tile = act @ w_d_e_tile                        # (bt, D_tile)

    # Per-row mask vs `e_local`. eids carries the local expert id (caller
    # `local_permute` converts global→local before invoking the FFN).
    eids = eids_ref[...]                                   # (bt,) int32
    mask_bt = (1 - jnp.minimum(jnp.abs(eids - e_local), 1)).astype(jnp.float32)
    mask_bd = lax.broadcast_in_dim(mask_bt, (bt, D_tile),
                                    broadcast_dimensions=(0,))
    masked_contrib = mask_bd * out_e_f_tile                # (bt, D_tile) f32

    # ---- Accumulate into out tile (RMW across E_local AND F-tile axes) ----
    # First touch of (i, d) is at (e=0, f=0) — init then; accumulate otherwise.
    @pl.when(is_first_ef)
    def _init():
        out_ref[...] = masked_contrib

    @pl.when(jnp.logical_not(is_first_ef))
    def _accum():
        out_ref[...] = out_ref[...] + masked_contrib


def _expert_ffn_f_tiled_grid(
    sorted_tokens, sorted_eids, W1_int, W_d, *, bt: int, D_tile: int, F_tile: int,
):
    """D.7 D+F-tiled kernel. Grid = (num_bt, E_local, num_f_tile, num_d_out).

    Args:
      W1_int: shape `(E_local, 2, D, F)` bf16 — gate/up SPLIT layout with
          the size-2 axis at position 1 (NOT position 2). The public-facing
          `expert_ffn_v_outside_f_tiled` accepts the Megatron `(E, D, 2, F)`
          layout and transposes to `(E, 2, D, F)` here. The transpose is
          required to avoid a Mosaic mis-lowering when the size-2 axis sits
          between the sublane (D) and multi-lane-block (F_tile) dimensions
          — see module docstring and `results/phase_d7.md` Gate C debug.
      D_tile: output D-tile size. Must evenly divide D.
      F_tile: F-tile size. Must evenly divide F.
    """
    M, D = sorted_tokens.shape
    E_local, two, _, F = W1_int.shape
    assert two == 2, f"W1_int axis 1 must be 2 (gate/up), got {two}"
    assert W1_int.shape[2] == D, (
        f"W1_int axis 2 must equal D={D}, got {W1_int.shape[2]}")
    num_bt = M // bt
    assert D % D_tile == 0, f"D={D} must be divisible by D_tile={D_tile}"
    assert F % F_tile == 0, f"F={F} must be divisible by F_tile={F_tile}"
    num_d_out = D // D_tile
    num_f_tile = F // F_tile

    return pl.pallas_call(
        functools.partial(_expert_ffn_f_tiled_body,
                          E_local=E_local, num_f_tile=num_f_tile,
                          num_d_out=num_d_out,
                          bt=bt, D=D, D_tile=D_tile, F=F, F_tile=F_tile),
        # Grid order: (i, d, e, f) — d OUTER, f innermost. Forces output
        # block (i, d) to be visited E_local*num_f_tile consecutive times
        # before advancing to the next d. Required for correctness at
        # num_d_out >= 4; see body docstring D.7-correctness-fix note.
        grid=(num_bt, num_d_out, E_local, num_f_tile),
        in_specs=[
            # tokens: change only on i axis.
            pl.BlockSpec((bt, D),                lambda i, d, e, f: (i, 0)),
            pl.BlockSpec((bt,),                  lambda i, d, e, f: (i,)),
            # W1_int[e, :, :, f_tile]: change on (e, f).
            pl.BlockSpec((1, 2, D, F_tile),      lambda i, d, e, f: (e, 0, 0, f)),
            # W_d[e, f_tile, d_tile]: change on (e, f, d).
            pl.BlockSpec((1, F_tile, D_tile),    lambda i, d, e, f: (e, f, d)),
        ],
        # Output: D-tiled, changes per (i, d). RMW across (e, f) — same
        # (i, d) is hit E_local*num_f_tile times consecutively.
        out_specs=pl.BlockSpec((bt, D_tile),     lambda i, d, e, f: (i, d)),
        out_shape=jax.ShapeDtypeStruct((M, D), jnp.float32),
        scratch_shapes=[
            pltpu.VMEM((bt, F_tile), jnp.float32),    # act_scratch
        ],
    )(sorted_tokens, sorted_eids, W1_int, W_d)


def _pick_F_tile(F: int, D: int, *, target_w1_block_bytes: int = 2 * 1024 * 1024) -> int:
    """Largest power-of-2 F_tile dividing F such that the W1 block
    `(1, D, 2, F_tile) bf16` is ≤ target_w1_block_bytes.

    Hard floor: F_tile ≥ 128. Mosaic's "last two dimensions divisible by
    (8, 128) OR equal to overall array dim" rule means a partial F-tile
    smaller than 128 is rejected; in that case the caller should pass
    F_tile=F (degenerate single-tile) instead.

    History: the original `(1, D, 2, F_tile)` BlockSpec gave NUMERICALLY
    WRONG results at cluster scale when F_tile spanned >1 lane block
    (F_tile=256 on v7x: max_rel=1.8 vs JAX f32 reference at
    E_local=16 D=7168 F=2048). Root cause was Mosaic's slicing of the
    size-2 axis sitting between the sublane (D) and multi-lane-block
    (F_tile) dimensions. The D.7-correctness-fix (2026-05-22) moves
    the size-2 axis to position 1 via an internal transpose
    `(E, D, 2, F) → (E, 2, D, F)`; the auto-picker is now safe at all
    valid F_tile values. See module docstring and
    `results/phase_d7.md` for the substrate-bug debug trail.
    """
    # W1 block bytes = D * 2 * F_tile * 2
    max_F_tile_by_vmem = max(128, target_w1_block_bytes // (4 * D))
    for cand in (2048, 1024, 512, 256, 128):
        if cand <= F and F % cand == 0 and cand <= max_F_tile_by_vmem:
            return cand
    # No power-of-2 ≥ 128 divides F → fall back to F_tile = F (degenerate).
    return F


def _pick_D_tile(D: int) -> int:
    """Largest power-of-2 D_tile ≤ 1024 that divides D (matches D.6)."""
    for cand in (1024, 512, 256, 128):
        if D % cand == 0:
            return cand
    raise AssertionError(f"no D_tile candidate divides D={D}")


def expert_ffn_v_outside_f_tiled(
    sorted_tokens: jax.Array,    # (M, D) bf16
    sorted_eids: jax.Array,      # (M,) int32 — local expert id (0..E_local-1)
    W1: jax.Array,               # (E_local, D, 2, F) bf16 — split gate/up
    W_d: jax.Array,              # (E_local, F, D)  bf16
    *,
    bt: int = 128,
    D_tile: int | None = None,
    F_tile: int | None = None,
) -> jax.Array:
    """D.7 D+F-tiled variant — fits full DSv3 (D=7168, F=2048).

    PUBLIC W1 layout: `(E_local, D, 2, F)`. The "2" axis separates gate
    vs up columns; F_tile slices along the F dim only, preserving the
    gate/up pairing so `silu(gate) * up` is local to each tile.

    INTERNAL: this wrapper transposes W1 to `(E_local, 2, D, F)` before
    calling the grid, moving the size-2 axis from position 2 to
    position 1. This is REQUIRED for numerical correctness at cluster
    scale when F_tile spans more than one lane block — see module
    docstring (D.7-correctness-fix, 2026-05-22) for the Mosaic
    substrate-bug background. The transpose is a one-time data
    movement (per kernel call) and is irrelevant in the v_outside
    flow where W1 is replicated and only the kernel sees it.
    """
    M, D = sorted_tokens.shape
    E_local, _, two, F = W1.shape
    assert two == 2, f"W1 axis 2 must be 2 (gate/up), got {two}"
    assert M % bt == 0, f"M={M} must be divisible by bt={bt}"
    assert sorted_eids.shape == (M,)
    assert W_d.shape == (E_local, F, D)
    assert bt >= 128, "F-tiled grid requires bt >= 128 (Mosaic rank-1 BlockSpec)"

    if D_tile is None:
        D_tile = _pick_D_tile(D)
    if F_tile is None:
        F_tile = _pick_F_tile(F, D)

    # (E_local, D, 2, F) → (E_local, 2, D, F). Move the size-2 (gate/up)
    # axis OUTSIDE the trailing (D, F) sublane×lane pair so Mosaic can
    # lower the multi-lane-block F_tile cleanly. See module docstring.
    W1_int = jnp.transpose(W1, (0, 2, 1, 3))

    return _expert_ffn_f_tiled_grid(
        sorted_tokens, sorted_eids, W1_int, W_d,
        bt=bt, D_tile=D_tile, F_tile=F_tile)


# -----------------------------------------------------------------------------
# AOT spec for tools/aot_check.py
# -----------------------------------------------------------------------------

def make_aot_spec(variant: str = "v_outside", topo_key: str = "4x8x8"):
    """AOT spec at the autoperf failure shape (E=256 D=7168 F=2048 K=8)
    on tpu7x:4x8x8 with the production mesh (dp=1, ep=4, fsdp=128, tp=1).

    Per SPEC §5.4 production: E_local = E/ep = 64. We don't shard W
    inside this AOT spec because the kernel is v_outside (weights are
    full-F per device; the JAX-side AG happens before this kernel).
    Just the per-shard shapes are used so Mosaic sees the same VMEM
    budget it will see at runtime.
    """
    from jax.sharding import PartitionSpec as P

    if topo_key == "4x8x8":
        # Production mesh: (dp=1, ep=4, fsdp=128, tp=1).
        mesh_axes_shape = (1, 4, 128, 1)
    elif topo_key == "2x2x1":
        # Iteration mesh: (dp=1, ep=4, fsdp=2, tp=1) — autoperf parity
        # at small T_global for the harness self-test.
        mesh_axes_shape = (1, 4, 2, 1)
    else:
        raise NotImplementedError(f"topo_key={topo_key} not wired for D.7 AOT spec")

    # Autoperf shape: DSv3-671B production.
    E, D, F, K = 256, 7168, 2048, 8
    _dp, ep, fsdp, _tp = mesh_axes_shape
    E_local = E // ep                                  # 64 at production
    # Token batch (M = T_local * K) — small enough that one bt-tile = M
    # so the AOT check focuses on the per-tile VMEM, not overall T size.
    bt = 128
    M = bt   # one tile

    abstract_inputs = (
        jax.ShapeDtypeStruct((M, D),                  jnp.bfloat16),  # sorted_tokens
        jax.ShapeDtypeStruct((M,),                    jnp.int32),     # sorted_eids
        jax.ShapeDtypeStruct((E_local, D, 2, F),      jnp.bfloat16),  # W1 (split)
        jax.ShapeDtypeStruct((E_local, F, D),         jnp.bfloat16),  # W_d
    )
    # Inside this shard_map: all four args are replicated (the v_outside
    # caller has done EP-permute + FSDP-AG before this kernel call). For
    # AOT we just want Mosaic to compile against the per-device shapes.
    in_specs  = (P(None, None), P(None,),
                 P(None, None, None, None), P(None, None, None))
    out_specs = P(None, None)
    return abstract_inputs, in_specs, out_specs, mesh_axes_shape


def kernel(sorted_tokens, sorted_eids, W1, W_d):
    """AOT entry point — uses default auto F_tile/D_tile."""
    return expert_ffn_v_outside_f_tiled(sorted_tokens, sorted_eids, W1, W_d)
