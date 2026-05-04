"""Common interface for DSv3 external-checkpoint loaders.

Two formats today: HuggingFace safetensors (load_hf_weights) and MaxText
orbax (load_maxtext_dsv3). Each maps a foreign on-disk format to our
canonical params dict layout (defined in param_layouts.py).

The contract:

  loader(source: str, cfg: ModelConfig, mesh: Mesh) -> params: dict

  * `source` — gs:// path, local dir, etc. (loader-specific).
  * `cfg` — ModelConfig (must match the checkpoint's architecture; e.g.
    cfg.V must equal the saved vocab size).
  * `mesh` — JAX serving mesh; output tensors are NamedSharding'd on it.
  * returns a params dict matching the structure described in
    param_layouts.dsv3_layouts(cfg).

  After successful return, every loader MUST have called
  param_layouts.validate(params, cfg, strict=True) — this is the
  fail-fast guarantee that the rest of the model assumes.
"""
from __future__ import annotations

from typing import Callable, Protocol, runtime_checkable

import jax
from jax.sharding import Mesh

from .model import ModelConfig


@runtime_checkable
class CheckpointLoader(Protocol):
    """Protocol any DSv3 external-checkpoint loader must satisfy."""

    def __call__(self, source: str, cfg: ModelConfig, mesh: Mesh) -> dict:
        ...


def get_loader(name: str) -> CheckpointLoader:
    """Return a loader by short name. Lazy imports — avoids loading
    safetensors/orbax/transformers when not needed.
    """
    if name in ("hf", "huggingface", "safetensors"):
        from .load_weights import load_hf_weights
        return load_hf_weights
    if name in ("maxtext", "orbax", "maxtext_orbax"):
        from .load_maxtext_ckpt import load_maxtext_dsv3
        return load_maxtext_dsv3
    raise ValueError(
        f"unknown loader '{name}'. Known: 'hf', 'maxtext'.")


__all__ = ["CheckpointLoader", "get_loader"]
