"""DSv3 fused EP-MoE — v_outside Phase 3 expert FFN (TC pallas_call).

Implements SPEC v0.4 §3 steps 3.1-3.6 for the v_outside variant. EP=1 path
(weights are full-F, already AG'd by JAX outside the kernel) and no A2A
inside the kernel — that's separate (added in B.4 for EP>1 G3).

Inputs (HBM, sorted by expert per Phase 1.6's argsort):
  sorted_tokens : (M, D) bf16        — M = T_local * K
  sorted_eids   : (M,) int32         — expert id of each row
  sorted_w      : (M,) bf16          — top-K weight (after renormalize)
  W1            : (E_local, D, 2F) bf16 — gate+up fused
  W_d           : (E_local, F, D) bf16

Output (HBM):
  out : (M, D) bf16  — per-row FFN output, scaled by sorted_w; ready for
                       Phase 4.3 SC gather_reduce to unsort+combine.

Approach: a single-tile `pallas_call` that processes one (bt, D) chunk of
sorted tokens through all E_local experts. The wrapper slices the (M, D)
array into num_bt static tiles and calls the kernel num_bt times,
concatenating outputs. Per-tile structure (rather than a grid with dynamic
indexing) avoids Mosaic's tiled-memref alignment constraints on rank-1
sub-tile loads at small G2 test shapes. Phase D may revert to a grid for
production-scale perf.

Frontmatter:
  slug: dsv3-v-outside-expert-ffn
  intent: kernel
  status: v0 (B.1) — EP=1 path; full-weight VMEM resident; per-tile pallas_call
  sources:
    - distilled/patterns/pallas-call-skeleton.md
    - distilled/antipatterns/jax-mosaic-rules.md (A3, A4, A5, B3, D1)
  related: targets/dsv3-fused-ep-moe/SPEC.md §3
"""
from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def _expert_ffn_tile_body(
    # HBM input refs (no grid; one call processes one tile):
    tok_ref,             # (bt, D) bf16
    eids_ref,            # (bt,) int32  — local expert id (0..E_local-1)
    W1_ref,              # (E_local, D, 2F) bf16
    W_d_ref,             # (E_local, F, D) bf16
    # HBM output ref:
    out_ref,             # (bt, D) bf16
    # Scratches:
    out_acc,             # (bt, D) f32 — output accumulator
    *,
    E_local: int,
    bt: int,
    D: int,
    F: int,
):
    """Process one (bt, D) tile through all E_local experts. Output is UNSCALED;
    callers apply the per-row sorted_w (top-K renormalized weight) externally
    in the unpermute step (matches production sparse_moe_distributed_fwd)."""
    # ---- D1: zero-init accumulator. ----
    out_acc[...] = jnp.zeros_like(out_acc)

    tok = tok_ref[...]                            # (bt, D) bf16
    eids = eids_ref[...]                          # (bt,) int32

    tok_f32 = tok.astype(jnp.float32)

    # Per-expert loop — Python `for` (B3).
    for e in range(E_local):
        # Gate+up matmul: (bt, D) @ (D, 2F) = (bt, 2F).
        # No transpose — see _inbox/blocker-spec-matmul-transpose-nit.md.
        w1_e = W1_ref[e].astype(jnp.float32)              # (D, 2F)
        gate_up = tok_f32 @ w1_e                          # (bt, 2F)
        gate, up = jnp.split(gate_up, 2, axis=-1)         # (bt, F) each
        act = jax.nn.silu(gate) * up                      # (bt, F)

        # Down matmul: (bt, F) @ (F, D) = (bt, D).
        w_d_e = W_d_ref[e].astype(jnp.float32)            # (F, D)
        out_e = act @ w_d_e                               # (bt, D)

        # Mask via int-arithmetic (A3): 1 where eids == e.
        mask_bt = (1 - jnp.minimum(jnp.abs(eids - e), 1)).astype(jnp.float32)
        mask_bd = lax.broadcast_in_dim(mask_bt, (bt, D), broadcast_dimensions=(0,))
        out_acc[...] = out_acc[...] + mask_bd * out_e

    # Emit f32 — the per-row top-K weight is applied externally in f32 and
    # the final segment_sum should also accumulate in f32 (to match
    # jax_ref.moe_forward's precision). Casting to bf16 here would round to
    # bf16 boundaries before the weight scale, accumulating drift across K.
    out_ref[...] = out_acc[...]


def _expert_ffn_one_tile(
    tok: jax.Array,        # (bt, D) bf16
    eids: jax.Array,       # (bt,) int32 — local expert id (0..E_local-1)
    W1: jax.Array,         # (E_local, D, 2F) bf16
    W_d: jax.Array,        # (E_local, F, D) bf16
) -> jax.Array:
    """Single pallas_call processing one (bt, D) tile. Output is UNSCALED.

    Kept for backward compat / debugging. Production path goes through
    `_expert_ffn_grid` (D.2) which uses one pallas_call with grid=(num_bt,).
    """
    bt, D = tok.shape
    E_local, _, twoF = W1.shape
    F = twoF // 2

    return pl.pallas_call(
        functools.partial(_expert_ffn_tile_body,
                          E_local=E_local, bt=bt, D=D, F=F),
        out_shape=jax.ShapeDtypeStruct((bt, D), jnp.float32),
        scratch_shapes=[
            pltpu.VMEM((bt, D), jnp.float32),    # out_acc
        ],
    )(tok, eids, W1, W_d)


def _expert_ffn_grid(
    sorted_tokens: jax.Array,    # (M, D) bf16
    sorted_eids: jax.Array,      # (M,) int32
    W1: jax.Array,               # (E_local, D, 2F) bf16
    W_d: jax.Array,              # (E_local, F, D) bf16
    *,
    bt: int,
) -> jax.Array:
    """D.2: ONE pallas_call with grid=(num_bt,). BlockSpec windows the M-dim;
    W1/W_d stay resident (full-buffer per grid step). Body is identical to the
    per-tile path — Mosaic handles the grid sequencing + DMA double-buffer.

    This removes the per-tile dispatch overhead that showed up as a 7-16%
    regression vs jax_ref at T≥1024 in the Phase C G5 bench
    (`results/phase_c_g5_bench.md`).
    """
    M, D = sorted_tokens.shape
    E_local, _, twoF = W1.shape
    F = twoF // 2
    num_bt = M // bt

    return pl.pallas_call(
        functools.partial(_expert_ffn_tile_body,
                          E_local=E_local, bt=bt, D=D, F=F),
        grid=(num_bt,),
        in_specs=[
            pl.BlockSpec((bt, D),                    lambda i: (i, 0)),  # tokens
            pl.BlockSpec((bt,),                      lambda i: (i,)),    # eids
            pl.BlockSpec((E_local, D, 2 * F),        lambda i: (0, 0, 0)),  # W1 full
            pl.BlockSpec((E_local, F, D),            lambda i: (0, 0, 0)),  # W_d full
        ],
        out_specs=pl.BlockSpec((bt, D),              lambda i: (i, 0)),
        out_shape=jax.ShapeDtypeStruct((M, D), jnp.float32),
        scratch_shapes=[
            pltpu.VMEM((bt, D), jnp.float32),    # out_acc — per-tile
        ],
    )(sorted_tokens, sorted_eids, W1, W_d)


def expert_ffn_v_outside(
    sorted_tokens: jax.Array,    # (M, D) bf16
    sorted_eids: jax.Array,      # (M,) int32 — LOCAL expert id (0..E_local-1)
    W1: jax.Array,               # (E_local, D, 2F) bf16
    W_d: jax.Array,              # (E_local, F, D) bf16
    *,
    bt: int = 8,
    impl: str = "auto",
) -> jax.Array:
    """Phase 3 expert FFN over sorted tokens. Returns (M, D) bf16 unscaled.

    Per-row top-K weight scaling is applied EXTERNALLY (in the unpermute step),
    matching production `sparse_moe_distributed_fwd` (`unpermute_fn` applies
    router_weights). Keeps the FFN kernel weight-agnostic so the EP path can
    leave sorted_w on the source device through the A2A.

    impl:
      - "auto" (default):
          * "f_tiled" when per-tile W1 (1, D, 2F) bf16 doesn't fit
            double-buffered in 32 MB (production F=2048 case). Internally
            reshapes (E, D, 2F) → (E, D, 2, F) and routes to D.7.
          * "d_tiled" when full W1 > 12 MB but per-tile fits (D=7168
            F=128 case).
          * "grid" when bt ≥ 128 and full W1 fits.
          * "tile" otherwise.
      - "f_tiled" (D.7): grid=(num_bt, num_d_out, E_local, num_f_tile).
        E + D + F tiled. The current production kernel — fits autoperf's
        E=256 D=7168 F=2048 K=8 shape on v7x 64 MB VMEM.
      - "d_tiled" (D.6): grid=(num_bt, E_local, num_d_out). E + D-output
        tiled but W1 single-tile per expert. Fits D=7168 only at F≤128.
      - "grid" (D.2): one pallas_call with grid=(num_bt,). Requires bt ≥ 128.
        Fits up to D=3840 at E_local=8.
      - "tile" (B.1): one pallas_call per M-tile. Works at any bt; kept for
        small-shape G2 cross-check.

    The auto-route makes this entry point a DROP-IN for callers (incl. the
    autoperf agent's pin to legacy v_outside): same (E, D, 2F) W1
    signature, same (M, D) f32 output. Big-F shapes that would OOM D.6
    now succeed transparently via the F-tiled internal path.
    """
    M, D = sorted_tokens.shape
    E_local, _, twoF = W1.shape
    F = twoF // 2
    assert M % bt == 0, f"M={M} must be divisible by bt={bt}"
    assert sorted_eids.shape == (M,)
    assert W_d.shape == (E_local, F, D)

    if impl == "auto":
        # Per-tile W1 block at the D.6 kernel: shape (1, D, 2F) bf16 =
        # D * 2F * 2 bytes single-buffered. Empirically D.6 starts hitting
        # Mosaic matmul-scratch overhead OOMs around 4 MB per-tile
        # (e.g. D=1024 F=2048 → 8 MB single-buf already trips it).
        # When over that threshold we MUST F-tile (D.7) — this captures
        # the autoperf production case at D=7168 F=2048 (56 MB per-tile).
        per_tile_w1_bytes = D * 2 * F * 2  # bf16, single-buffered
        full_w1_bytes = E_local * D * 2 * F * 2
        if per_tile_w1_bytes > 4 * 1024 * 1024:
            # Per-tile too large for D.6's Mosaic budget.
            # Reshape (E, D, 2F) → (E, D, 2, F) and route to D.7.
            impl = "f_tiled"
        elif full_w1_bytes > 12 * 1024 * 1024:
            # Full W1 doesn't fit; tile E + D via D.6.
            impl = "d_tiled"
        elif bt >= 128:
            impl = "grid"
        else:
            impl = "tile"

    if impl == "f_tiled":
        # D.7: F-tiled kernel. Takes (E, D, 2, F) layout natively; reshape
        # legacy (E, D, 2F) → (E, D, 2, F). The reshape is a metadata-only
        # view (gate/up are contiguous in the trailing 2F dim, so the split
        # at position 2 is a no-op transpose followed by a stride-only view).
        from .expert_ffn_f_tiled import expert_ffn_v_outside_f_tiled
        W1_split = W1.reshape(E_local, D, 2, F)
        return expert_ffn_v_outside_f_tiled(
            sorted_tokens, sorted_eids, W1_split, W_d, bt=bt)

    if impl == "d_tiled":
        from .expert_ffn_d_tiled import expert_ffn_v_outside_d_tiled
        return expert_ffn_v_outside_d_tiled(
            sorted_tokens, sorted_eids, W1, W_d, bt=bt)

    if impl == "grid":
        return _expert_ffn_grid(sorted_tokens, sorted_eids, W1, W_d, bt=bt)

    if impl != "tile":
        raise ValueError(
            f"impl must be 'auto', 'grid', 'd_tiled', 'f_tiled' or 'tile', "
            f"got {impl!r}")

    num_bt = M // bt
    out_pieces = []
    for i in range(num_bt):
        tok_i  = lax.dynamic_slice_in_dim(sorted_tokens, i * bt, bt, axis=0)
        eids_i = lax.dynamic_slice_in_dim(sorted_eids,   i * bt, bt, axis=0)
        out_pieces.append(_expert_ffn_one_tile(tok_i, eids_i, W1, W_d))
    return jnp.concatenate(out_pieces, axis=0)


# -----------------------------------------------------------------------------
# AOT spec for tools/aot_check.py
# -----------------------------------------------------------------------------

def make_aot_spec(variant: str = "v_outside", topo_key: str = "2x2x1"):
    from jax.sharding import PartitionSpec as P

    if topo_key == "2x2x1":
        mesh_axes_shape = (1, 4, 2, 1)
    else:
        raise NotImplementedError(f"topo_key={topo_key} not wired for AOT spec yet")

    E_local, D, F, K = 8, 64, 32, 2
    T = 16
    M = T * K

    abstract_inputs = (
        jax.ShapeDtypeStruct((M, D),                 jnp.bfloat16),  # sorted_tokens
        jax.ShapeDtypeStruct((M,),                   jnp.int32),     # sorted_eids
        jax.ShapeDtypeStruct((E_local, D, 2 * F),    jnp.bfloat16),  # W1
        jax.ShapeDtypeStruct((E_local, F, D),        jnp.bfloat16),  # W_d
    )
    in_specs = (P(None, None), P(None,),
                P(None, None, None), P(None, None, None))
    out_specs = P(None, None)
    return abstract_inputs, in_specs, out_specs, mesh_axes_shape


def kernel(sorted_tokens, sorted_eids, W1, W_d):
    return expert_ffn_v_outside(sorted_tokens, sorted_eids, W1, W_d, bt=8)
