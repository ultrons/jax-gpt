# Training forward kernel wrapper — imports from fwd_kernel_train.py (grid-based).
#
# See fwd_kernel_train.py for the full design explanation.
#
# Import strategy: try flat (Docker /app/ layout) then package (local dev layout).

try:
    from fwd_kernel_train import fused_ep_moe_fwd_train_v1          # Docker: /app/fwd_kernel_train.py
except ImportError:
    from fused_moe_bwd.fwd_kernel_train import fused_ep_moe_fwd_train_v1  # local: fused_moe_bwd/


def fused_ep_moe_fwd_streaming_v1(
    tokens,
    w1,
    w2,
    gating_output,
    top_k: int,
    *,
    ep_axis_name: str,
    act_fn: str = "silu",
    scoring_fn: str = "identity",
    renormalize_topk_logits: bool = True,
    bt=None,
    bd1=None,
    bd2=None,
    btc=None,
    bd1c=None,
    bd2c=None,
    top_k_indices_precomputed=None,
    top_k_weights_precomputed=None,
    E_global_override=None,
    tp_axis_name: str | None = None,   # v200: when set, kernel uses lax.axis_index(tp_axis_name)
    tp_rank=None,                       # unused; kept for backward compat
    tp_rank_int: int = 0,              # static Python int tp rank (v199 legacy; use tp_axis_name instead)
    tp_rank_arr=None,                   # (1,) int32 JAX-traced tp_rank per device (v193/v198; avoid)
    collective_id: int = 0,            # Mosaic ICI collective ID
):
    """Thin shim — delegates to fused_ep_moe_fwd_train_v1 (grid-based kernel).

    Called from within shard_map(ep, fsdp, tp).  Training mesh is (dp, ep, fsdp, tp).
    dp=0 is a static prefix.

    v200 preferred path (tp_axis_name):
      tp_axis_name="tp": kernel uses lax.axis_index("tp") directly in
      get_mesh_device_id — same mechanism as lax.axis_index("ep") and "fsdp".
      Single pallas_call, no lax.cond.  Avoids both v198 (dynamic scalar_prefetch
      crash) and v199 (lax.cond BoundsCheck 21 in .2 branch VMEM OOB).

    v199 legacy path (lax.cond dispatch):
      tp_rank_int: static Python int (0 or 1) — extra_device_id_suffix=(tp_rank_int,).
      lax.cond causes BoundsCheck 21 — do NOT use.

    v193/v198 legacy path (single pallas_call, dynamic scalar_prefetch):
      tp_rank_arr: (1,) int32 from lax.axis_index("tp").  Causes Mosaic runtime
      crash on real hardware.  Do NOT use.
    """
    return fused_ep_moe_fwd_train_v1(
        tokens, w1, w2, gating_output, top_k,
        ep_axis_name=ep_axis_name,
        act_fn=act_fn,
        scoring_fn=scoring_fn,
        renormalize_topk_logits=renormalize_topk_logits,
        non_ep_axis_name="fsdp",
        non_ep_first=False,
        extra_device_id_prefix=(0,),            # dp_rank = 0
        # v202: when tp_rank_arr is provided (concrete shard_map slice), set
        # extra_device_id_suffix=() so get_mesh_device_id uses scalar_prefetch
        # (tp_rank_scalar[0]) instead of static suffix. When tp_rank_arr is None
        # (TP=1 inference or static fallback), use tp_rank_int as static suffix.
        extra_device_id_suffix=() if tp_rank_arr is not None else (tp_rank_int,),
        tp_axis_name=None,                      # v202: lax.axis_index("tp") broken in pallas
        tp_rank_arr=tp_rank_arr,
        collective_id=collective_id,
        bt=bt, bd1=bd1, bd2=bd2,
        btc=btc, bd1c=bd1c, bd2c=bd2c,
        top_k_indices_precomputed=top_k_indices_precomputed,
        top_k_weights_precomputed=top_k_weights_precomputed,
        E_global_override=E_global_override,
    )
