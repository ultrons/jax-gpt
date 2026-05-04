"""Test that the ParamLayout registry stays in sync with init_params.

If init_params produces a tensor with a shape that the registry doesn't
expect (or vice-versa), this test fails — guaranteeing the two source
files don't silently drift.

Runs CPU-only (no mesh) — `init_params` accepts mesh=None and returns
plain jax.Arrays whose shape can be inspected.
"""
from __future__ import annotations
import os

# Force CPU before any JAX import — no TPU needed for shape checks.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import pytest

from jax_gpt.models.dsv3.model import (
    full_671b_config, mini_config, init_params,
)
from jax_gpt.models.dsv3.param_layouts import dsv3_layouts, validate


def test_registry_keys_match_mini_init():
    """mini_config has L_dense=1, L_moe=1; init_params output shapes
    should round-trip through validate() with zero errors."""
    cfg = mini_config()
    key = jax.random.PRNGKey(0)
    params = init_params(cfg, key, mesh=None)
    errors = validate(params, cfg, strict=False)
    if errors:
        pytest.fail(
            "ParamLayout registry drifted from init_params output:\n  - "
            + "\n  - ".join(errors[:20]))


def test_registry_keys_match_full_671b_shapes():
    """No actual init (would OOM on host) — synthesize abstract leaves
    matching dsv3_layouts(cfg) and verify validate() passes them."""
    cfg = full_671b_config()
    layouts = dsv3_layouts(cfg)

    # Build a fake params dict from layouts: shape + dtype only, with
    # nested-dict structure matching what init_params produces.
    import jax.numpy as jnp
    import numpy as np

    def _set(d, path, value):
        keys = path.split(".")
        for k in keys[:-1]:
            d = d.setdefault(k, {})
        d[keys[-1]] = value

    fake = {}
    for path, lay in layouts.items():
        # Tiny placeholder (don't actually allocate; numpy zeros of size 1
        # in the right shape would OOM the host for V*D=900M).
        # Use a custom Array-like with shape + dtype attrs.
        class _Stub:
            def __init__(self, shape, dtype):
                self.shape = tuple(shape)
                self.dtype = np.dtype(dtype)
        _set(fake, path, _Stub(lay.shape, lay.dtype))

    # Run validate; should produce 0 errors.
    errors = validate(fake, cfg, strict=False)
    assert not errors, (
        "Registry self-consistency failed:\n  - " + "\n  - ".join(errors[:20]))
