"""Vendored fused-MoE expert FFN from ~/kernel-agent @ b4b63d1.

This package contains the Phase 3 expert-FFN Pallas kernel from
kernel-agent's `targets/dsv3-fused-ep-moe/build/v_outside/`. It is
gated in jax-gpt by `cfg.moe_use_kernel_agent_ffn` and only replaces
the three ragged_dot / gmm_v2 calls inside `_expert_mlp_gmm_ag_body`;
the surrounding EP token AG + sort + scatter + psum_scatter
machinery is preserved.

Vendored, not imported from the upstream repo, so that:
  1. jax-gpt training is reproducible from a single commit, even as
     kernel-agent continues to evolve.
  2. The exact upstream snapshot the kernel was copied from is
     pinned at research/dsv3/kernel-agent-snapshot-b4b63d1/.

To refresh the snapshot, re-run the snapshot step in
`research/dsv3/kernel_agent_integration_notes.md`.

Upstream source: kernel-agent 2cda804, files
  build/v_outside/expert_ffn.py             (auto-route to D.7 F-tiled when F=2048)
  build/v_outside/expert_ffn_d_tiled.py     (D.6 — D-axis tiling, F=128 case)
  build/v_outside/expert_ffn_f_tiled.py     (D.7 — F-axis tiling, F=2048 case; cluster-verified)

Prior pins kept in research/dsv3/kernel-agent-snapshot-{945964d,b4b63d1,2cda804}/
for diffability.
"""
from .expert_ffn import expert_ffn_v_outside

__all__ = ["expert_ffn_v_outside"]
