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

Grid: (B,) — one kernel invocation per batch element.
Each invocation processes all H heads. State tile = (H, dk, dv) = 512 KB @ f32
for H=8, dk=dv=128, fitting comfortably in TPU VMEM (16+ MB/core).

Usage:
    new_state, output = fused_deltanet_step(
        state, q, k, v, g_factor, beta,
    )
    # Replaces lines 180-193 of deltanet.py
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def _fused_deltanet_kernel(
    # Inputs (read-only refs)
    state_ref,      # [H, dk, dv]
    q_ref,          # [H, dk]
    k_ref,          # [H, dk]
    v_ref,          # [H, dv]
    g_factor_ref,   # [H, 1]
    beta_ref,       # [H, 1]
    # Outputs (write refs)
    new_state_ref,  # [H, dk, dv]
    output_ref,     # [H, dv]
):
    """Pallas kernel body: fused decay + readout + rank-1 update + query.

    Processes one batch element, all H heads.
    Uses matmul (1,dk) @ (dk,dv) → (1,dv) per head for TPU compatibility.
    """
    H = state_ref.shape[1]

    for h in range(H):
        state = state_ref[0, h]           # [dk, dv] f32
        q = q_ref[0, h]                   # [dk] f32
        k = k_ref[0, h]                   # [dk] f32
        v = v_ref[0, h]                   # [dv] f32
        g = g_factor_ref[0, h, 0]         # scalar f32
        beta = beta_ref[0, h, 0]          # scalar f32

        # Step 1: Decay state
        state = state * g                 # [dk, dv]

        # Step 2: Readout — kv_mem[v] = sum_k(k[k] * state[k,v])
        # TPU Pallas requires 2D+ operands for dot_general.
        # (1, dk) @ (dk, dv) → (1, dv), then squeeze.
        kv_mem = jax.lax.dot_general(
            k[None, :], state,
            dimension_numbers=(((1,), (0,)), ((), ())),
            preferred_element_type=jnp.float32,
        )[0]  # [dv]

        # Step 3: Gated delta
        delta = (v - kv_mem) * beta       # [dv]

        # Step 4: Rank-1 update — state += outer(k, delta)
        state = state + k[:, None] * delta[None, :]  # [dk, dv]

        # Step 5: Query output — output[v] = sum_k(q[k] * state[k,v])
        output = jax.lax.dot_general(
            q[None, :], state,
            dimension_numbers=(((1,), (0,)), ((), ())),
            preferred_element_type=jnp.float32,
        )[0]  # [dv]

        # Write outputs for this head
        new_state_ref[0, h] = state
        output_ref[0, h] = output


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
        (new_state, output)
        new_state: (B, H, dk, dv) f32
        output: (B, H, dv) f32
    """
    B, H, dk, dv = state.shape

    # Reshape g_factor/beta to (B, H, 1) for TPU tiling alignment:
    # block (H, 1) → last two dims (H, 1), H%8=0 ✓, 1==array_dim ✓
    g_3d = g_factor[..., None]    # (B, H, 1)
    beta_3d = beta[..., None]     # (B, H, 1)

    new_state, output = pl.pallas_call(
        _fused_deltanet_kernel,
        out_shape=[
            jax.ShapeDtypeStruct((B, H, dk, dv), jnp.float32),  # new_state
            jax.ShapeDtypeStruct((B, H, dv), jnp.float32),      # output
        ],
        grid=(B,),
        in_specs=[
            pl.BlockSpec((1, H, dk, dv), lambda b: (b, 0, 0, 0)),  # state
            pl.BlockSpec((1, H, dk), lambda b: (b, 0, 0)),          # q
            pl.BlockSpec((1, H, dk), lambda b: (b, 0, 0)),          # k
            pl.BlockSpec((1, H, dv), lambda b: (b, 0, 0)),          # v
            pl.BlockSpec((1, H, 1), lambda b: (b, 0, 0)),           # g_factor
            pl.BlockSpec((1, H, 1), lambda b: (b, 0, 0)),           # beta
        ],
        out_specs=[
            pl.BlockSpec((1, H, dk, dv), lambda b: (b, 0, 0, 0)),  # new_state
            pl.BlockSpec((1, H, dv), lambda b: (b, 0, 0)),          # output
        ],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel",),
        ),
        name="fused_deltanet_step",
    )(state, q, k, v, g_3d, beta_3d)

    return new_state, output


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
