"""Megablox GMM v2 Pallas kernel for MoE expert computation.

Copied from tpu-inference (tpu_inference/kernels/megablox/gmm_v2.py).
Bypasses XLA tiling system — reads HBM directly via Pallas DMA index maps,
eliminating the squeeze/retiling overhead that dominates decode on TPU.
"""

from jax_gpt.models.qwen35.megablox.gmm_v2 import gmm_v2, TileSizes

__all__ = ['gmm_v2', 'TileSizes']
