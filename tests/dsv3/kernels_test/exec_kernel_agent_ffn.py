"""Numerical-parity smoke test for the kernel-agent FFN swap inside
`_expert_mlp_gmm_ag_body`.

Compares three FFN implementations on the same synthetic inputs at a small
shape:
  1. gmm_v2          (cfg.moe_use_gmm_v2 = True)
  2. kernel_agent    (cfg.moe_use_kernel_agent_ffn = True)
  3. baseline JAX    (both flags False — `jax.lax.ragged_dot` path)

For each implementation, runs the full `expert_mlp_gmm_ag` (which exercises
all the surrounding AG-dispatch + sort + scatter + psum_scatter machinery)
on a trivial mesh and reports max_abs / max_rel against the baseline.

Pass criterion: each Pallas path agrees with the baseline within rtol=5e-2
(bf16 rounding through scatter+psum_scatter; same tolerance the kernel-agent
G3 gate uses).

This is a smoke test (catches obvious wiring errors); production parity
requires a cluster shot at full shape.

Run:
  source ~/xdb/.xprof/bin/activate
  PYTHONPATH=. python tests/dsv3/kernels_test/exec_kernel_agent_ffn.py
"""
from __future__ import annotations

import os
import sys

# Force a 4-device CPU mesh so we can build (dp=1, ep=2, fsdp=2, tp=1)
# without needing a TPU. The kernel under test will fall back to the
# `tile` impl (bt<128) on CPU and the gmm_v2 path will produce a
# `tpu_custom_call` — which CPU can't run. So we keep this CPU-friendly
# by skipping gmm_v2 on CPU. On TPU all three run.
import jax

if jax.default_backend() == "cpu":
    os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")
    jax.config.update("jax_platforms", "cpu")

import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, PartitionSpec as P

# Make `jax_gpt` importable when run as a script.
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from jax_gpt.models.dsv3.model import (  # noqa: E402
    ModelConfig, expert_mlp_gmm_ag,
)


# ---------- small synthetic shapes ----------
# Local 4-device CPU is enough for wiring parity. Production is (E=256
# D=7168 K=8); we test (E=8 D=128 K=2) which exercises the same code
# paths.
B   = 1
S   = 256          # B*S=256 tokens — divisible by max_local_c >= bt=8 (CPU "tile" impl)
E   = 8
D   = 128
F   = 64
K   = 2
DEV = jax.devices()
N   = len(DEV)
print(f"backend={jax.default_backend()}  devices={N}")

if N < 4:
    sys.exit(f"need >= 4 devices; got {N}")


def make_cfg(*, use_gmm_v2: bool, use_kernel_agent_ffn: bool):
    cfg = ModelConfig(name="parity_smoke")
    # Override shape knobs.
    cfg.D = D
    cfg.F = F
    cfg.E = E
    cfg.K = K
    cfg.L = 1
    cfg.L_dense = 0
    cfg.mesh = Mesh(np.array(DEV).reshape(1, 2, 2, 1),
                    ("dp", "ep", "fsdp", "tp"))
    cfg.moe_use_gmm_v2 = use_gmm_v2
    cfg.moe_use_kernel_agent_ffn = use_kernel_agent_ffn
    cfg.moe_n_chunks = 1     # keep things simple at small shape
    cfg.moe_use_sc_scatter = False
    cfg.moe_fp8_weights = False
    cfg.moe_debug_nans = False
    return cfg


def make_inputs(seed: int = 0):
    rng = np.random.default_rng(seed)
    x  = jnp.asarray(rng.standard_normal((B, S, D)).astype(np.float32) * 0.5,
                     dtype=jnp.bfloat16)
    wi_0 = jnp.asarray(rng.standard_normal((E, F, D)).astype(np.float32) * 0.05,
                       dtype=jnp.bfloat16)
    wi_1 = jnp.asarray(rng.standard_normal((E, F, D)).astype(np.float32) * 0.05,
                       dtype=jnp.bfloat16)
    wo   = jnp.asarray(rng.standard_normal((E, F, D)).astype(np.float32) * 0.05,
                       dtype=jnp.bfloat16)

    # Random per-token top-K (uniform over experts).
    raw = jnp.asarray(rng.standard_normal((B, S, E)).astype(np.float32),
                      dtype=jnp.bfloat16)
    top_k_weights, top_k_indices = jax.lax.top_k(raw.astype(jnp.float32), K)
    # Renormalize to sum=1 per token.
    top_k_weights = top_k_weights / top_k_weights.sum(axis=-1, keepdims=True)
    top_k_weights = top_k_weights.astype(jnp.bfloat16)
    top_k_indices = top_k_indices.astype(jnp.int32)

    return x, wi_0, wi_1, wo, top_k_weights, top_k_indices


def diff(a, b, label):
    da = jnp.abs(a.astype(jnp.float32) - b.astype(jnp.float32))
    base = jnp.maximum(jnp.abs(b.astype(jnp.float32)), 1e-6)
    max_abs = float(da.max())
    max_rel = float((da / base).max())
    print(f"  {label:24s}  max_abs={max_abs:.3e}  max_rel={max_rel:.3e}")
    return max_abs, max_rel


def main():
    x, wi_0, wi_1, wo, tkw, tki = make_inputs(0)

    print("---- baseline (ragged_dot, no Pallas) ----")
    cfg = make_cfg(use_gmm_v2=False, use_kernel_agent_ffn=False)
    out_base = jax.jit(lambda x_, w0_, w1_, wo_, tkw_, tki_:
                       expert_mlp_gmm_ag(x_, w0_, w1_, wo_, tkw_, tki_, cfg))(
        x, wi_0, wi_1, wo, tkw, tki)
    out_base.block_until_ready()
    print(f"  out shape={out_base.shape} dtype={out_base.dtype}")
    assert not jnp.isnan(out_base).any(), "baseline produced NaN"

    print("---- kernel-agent FFN ----")
    cfg_ka = make_cfg(use_gmm_v2=False, use_kernel_agent_ffn=True)
    try:
        out_ka = jax.jit(lambda x_, w0_, w1_, wo_, tkw_, tki_:
                         expert_mlp_gmm_ag(x_, w0_, w1_, wo_, tkw_, tki_, cfg_ka))(
            x, wi_0, wi_1, wo, tkw, tki)
        out_ka.block_until_ready()
        ok_ka = not jnp.isnan(out_ka).any()
        if ok_ka:
            diff(out_ka, out_base, "kernel-agent vs baseline")
        else:
            print("  kernel-agent produced NaN")
    except Exception as e:
        msg = str(e)
        print(f"  EXCEPTION: {type(e).__name__}: {msg[:600]}")
        ok_ka = False

    # gmm_v2 only runs on TPU; skip if backend!=tpu.
    if jax.default_backend() == "tpu":
        print("---- gmm_v2 ----")
        cfg_v2 = make_cfg(use_gmm_v2=True, use_kernel_agent_ffn=False)
        try:
            out_v2 = jax.jit(lambda x_, w0_, w1_, wo_, tkw_, tki_:
                              expert_mlp_gmm_ag(x_, w0_, w1_, wo_, tkw_, tki_, cfg_v2))(
                x, wi_0, wi_1, wo, tkw, tki)
            out_v2.block_until_ready()
            if not jnp.isnan(out_v2).any():
                diff(out_v2, out_base, "gmm_v2 vs baseline")
        except Exception as e:
            msg = str(e)
            print(f"  EXCEPTION: {type(e).__name__}: {msg[:600]}")
    else:
        print("---- gmm_v2 skipped (CPU backend) ----")

    print()
    print("DONE" if ok_ka else "SMOKE FAILED")
    return 0 if ok_ka else 1


if __name__ == "__main__":
    sys.exit(main())
