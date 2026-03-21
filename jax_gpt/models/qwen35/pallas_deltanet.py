"""Pallas kernel for fused DeltaNet recurrent state update on TPU.

Fuses the three einsum passes over the state matrix into a single kernel:
  1. kv_mem  = einsum('bhkv,bhk->bhv', state, k)   — readout
  2. state   = state * g + outer(k, delta)          — decay + rank-1 update
  3. output  = einsum('bhkv,bhk->bhv', state, q)    — query

Without fusion, each einsum reads the full state matrix from HBM separately.
The state is (B, H, dk, dv) = (B, 8, 128, 128) per device — 16 MB/batch @ f32.
Three passes = 48 MB/batch/layer. With 45 DeltaNet layers, that's 2.1 GB/batch.

This kernel reads the state ONCE, computes all three operations in VMEM,
and writes the updated state back — reducing HBM traffic by ~3x.

Usage:
    output, new_state = fused_deltanet_step(
        state, q, k, v, g_factor, beta,
    )
    # Replaces lines 180-193 of deltanet.py
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl


def _fused_deltanet_kernel(
    # Inputs (read-only refs)
    state_ref,      # [dk, dv] — one (batch, head) slice of state
    q_ref,          # [dk]
    k_ref,          # [dk]
    v_ref,          # [dv]
    g_factor_ref,   # [] scalar — exp(g), the decay factor
    beta_ref,       # [] scalar — sigmoid gate
    # Outputs (write refs)
    new_state_ref,  # [dk, dv]
    output_ref,     # [dv]
):
    """Pallas kernel body: fused decay + readout + rank-1 update + query.

    Processes one (batch, head) slice. Grid maps over (B, H).

    For dk=128, dv=128: state tile is 128×128 = 16K f32 values = 64 KB.
    This fits comfortably in TPU VMEM (16+ MB per core).
    """
    # Load inputs into VMEM and squeeze grid dims (1, 1, ...) → (...)
    state = state_ref[0, 0]          # [dk, dv] f32
    q = q_ref[0, 0]                  # [dk] f32
    k = k_ref[0, 0]                  # [dk] f32
    v = v_ref[0, 0]                  # [dv] f32
    g = g_factor_ref[0, 0]           # [] f32
    beta = beta_ref[0, 0]            # [] f32

    # Step 1: Decay state
    state = state * g               # [dk, dv]

    # Step 2: Readout — kv_mem = state.T @ k (contract over dk)
    kv_mem = jnp.dot(k, state)      # [dv] = [dk] . [dk, dv]

    # Step 3: Gated delta
    delta = (v - kv_mem) * beta      # [dv]

    # Step 4: Rank-1 update — state += outer(k, delta)
    state = state + k[:, None] * delta[None, :]  # [dk, dv]

    # Step 5: Query output — output = new_state.T @ q (contract over dk)
    output = jnp.dot(q, state)       # [dv] = [dk] . [dk, dv]

    # Write outputs
    new_state_ref[0, 0] = state
    output_ref[0, 0] = output


def fused_deltanet_step(
    state: jax.Array,       # (B, H, dk, dv) f32
    q: jax.Array,           # (B, H, dk) f32
    k: jax.Array,           # (B, H, dk) f32
    v: jax.Array,           # (B, H, dv) f32
    g_factor: jax.Array,    # (B, H) f32 — exp(g), NOT raw g
    beta: jax.Array,        # (B, H) f32 — sigmoid gate
) -> tuple[jax.Array, jax.Array]:
    """Fused DeltaNet recurrent step via Pallas.

    Replaces:
        state = state * g_factor[..., None, None]
        kv_mem = einsum('bhkv,bhk->bhv', state, k)
        delta = (v - kv_mem) * beta[..., None]
        state = state + einsum('bhk,bhv->bhkv', k, delta)
        output = einsum('bhkv,bhk->bhv', state, q)

    Args:
        state: (B, H, dk, dv) recurrent state in f32.
        q: (B, H, dk) normalized query.
        k: (B, H, dk) normalized key.
        v: (B, H, dv) value.
        g_factor: (B, H) decay factor = exp(g).
        beta: (B, H) gate = sigmoid(b).

    Returns:
        (output, new_state)
        output: (B, H, dv) f32
        new_state: (B, H, dk, dv) f32
    """
    B, H, dk, dv = state.shape

    return pl.pallas_call(
        _fused_deltanet_kernel,
        out_shape=[
            jax.ShapeDtypeStruct((B, H, dk, dv), jnp.float32),  # new_state
            jax.ShapeDtypeStruct((B, H, dv), jnp.float32),      # output
        ],
        grid=(B, H),
        in_specs=[
            pl.BlockSpec((1, 1, dk, dv), lambda b, h: (b, h, 0, 0)),   # state
            pl.BlockSpec((1, 1, dk), lambda b, h: (b, h, 0)),           # q
            pl.BlockSpec((1, 1, dk), lambda b, h: (b, h, 0)),           # k
            pl.BlockSpec((1, 1, dv), lambda b, h: (b, h, 0)),           # v
            pl.BlockSpec((1, 1), lambda b, h: (b, h)),                  # g_factor
            pl.BlockSpec((1, 1), lambda b, h: (b, h)),                  # beta
        ],
        out_specs=[
            pl.BlockSpec((1, 1, dk, dv), lambda b, h: (b, h, 0, 0)),   # new_state
            pl.BlockSpec((1, 1, dv), lambda b, h: (b, h, 0)),           # output
        ],
        compiler_params=dict(mosaic=dict(
            dimension_semantics=("parallel", "parallel"),
        )),
        name="fused_deltanet_step",
    )(state, q, k, v, g_factor, beta)


def fused_deltanet_step_ref(
    state: jax.Array,       # (B, H, dk, dv) f32
    q: jax.Array,           # (B, H, dk) f32
    k: jax.Array,           # (B, H, dk) f32
    v: jax.Array,           # (B, H, dv) f32
    g_factor: jax.Array,    # (B, H) f32
    beta: jax.Array,        # (B, H) f32
) -> tuple[jax.Array, jax.Array]:
    """Reference JAX implementation (no Pallas) for testing correctness."""
    state = state * g_factor[..., None, None]
    kv_mem = jnp.einsum('bhkv,bhk->bhv', state, k)
    delta = (v - kv_mem) * beta[..., None]
    state = state + jnp.einsum('bhk,bhv->bhkv', k, delta)
    output = jnp.einsum('bhkv,bhk->bhv', state, q)
    return state, output
