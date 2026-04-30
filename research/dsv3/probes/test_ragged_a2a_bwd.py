"""Minimal repro for ragged_all_to_all backward crash.

Tests the interaction of:
  ragged_all_to_all + jax.checkpoint + jax.lax.scan + async collective fusion

Run on 4 devices (e.g. a v4-8 TPU VM, or TPU_CHIPS_PER_HOST_BOUNDS=2x2x1):
  python test_ragged_a2a_bwd.py

Expected: all tests pass with a working backward.
Buggy:    E0200 RuntimeUnexpectedCoreHalt at all_to_all.* in wide.region*
"""

import functools
import os
import sys

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P
from jax.experimental.shard_map import shard_map

print(f"JAX version: {jax.__version__}")
print(f"Devices: {jax.devices()}")

N_DEVICES = len(jax.devices())
if N_DEVICES < 2:
    print("Need at least 2 devices. Exiting.")
    sys.exit(0)

# ── Minimal ragged_all_to_all body ────────────────────────────────────────────

def _a2a_body(x, axis_name):
    """Send x[i] to device i%EP, receive from all."""
    T, D = x.shape
    EP = jax.lax.axis_size(axis_name)

    # Uniform: each device sends T//EP tokens to each peer
    chunk = T // EP
    send_sizes = jnp.full((EP,), chunk, dtype=jnp.int32)
    input_offsets = jnp.arange(EP, dtype=jnp.int32) * chunk

    recv_sizes   = jax.lax.all_to_all(send_sizes,    axis_name, 0, 0, tiled=True)
    recv_offsets = jnp.concatenate([
        jnp.zeros(1, jnp.int32), jnp.cumsum(recv_sizes)[:-1]])
    output_offsets = jax.lax.all_to_all(recv_offsets, axis_name, 0, 0, tiled=True)

    recv_buf = jnp.zeros_like(x)
    out = jax.lax.ragged_all_to_all(
        x, recv_buf,
        input_offsets, send_sizes,
        output_offsets, recv_sizes,
        axis_name=axis_name)
    return out


def _body_with_matmul(x, w, axis_name):
    """A2A dispatch, matmul, A2A gather — minimal MoE-like pattern."""
    dispatched = _a2a_body(x, axis_name)        # dispatch
    out = dispatched @ w                         # local compute
    gathered = _a2a_body(out, axis_name)         # gather
    return gathered


# ── Tests ─────────────────────────────────────────────────────────────────────

def make_mesh():
    devices = jax.devices()[:N_DEVICES]
    return Mesh(devices, axis_names=("ep",))

mesh = make_mesh()
EP = N_DEVICES
T_local = 16   # tokens per device
D = 8           # hidden dim

def run_test(name, use_checkpoint, use_scan, n_layers):
    print(f"\n{'='*60}")
    print(f"Test: {name}")
    print(f"  checkpoint={use_checkpoint}, scan={use_scan}, layers={n_layers}")

    rng = jax.random.PRNGKey(0)
    x = jax.random.normal(rng, (EP * T_local, D))   # (T_total, D)
    ws = jax.random.normal(rng, (n_layers, D, D))   # per-layer weights

    @functools.partial(shard_map, mesh=mesh,
                       in_specs=(P("ep", None), P(None, None, None)),
                       out_specs=P("ep", None),
                       check_rep=False)
    def _sharded_layer(x_local, w_local):
        def _fn(xl, wl):
            return _body_with_matmul(xl, wl, "ep")
        if use_checkpoint:
            _fn = jax.checkpoint(_fn)
        return _fn(x_local, w_local)

    def _one_layer(carry, w):
        x = carry
        out = _sharded_layer(x, w)
        return out, None

    def forward(x, ws):
        if use_scan:
            out, _ = jax.lax.scan(_one_layer, x, ws)
        else:
            out = x
            for i in range(n_layers):
                out = _sharded_layer(out, ws[i])
        return out.sum()

    grad_fn = jax.jit(jax.grad(forward))

    try:
        g = grad_fn(x, ws)
        g.block_until_ready()
        print(f"  PASSED — grad norm = {jnp.linalg.norm(g):.4f}")
        return True
    except Exception as e:
        print(f"  FAILED — {type(e).__name__}: {str(e)[:200]}")
        return False


results = {}

# ── Test raw ragged_all_to_all backward ───────────────────────────────────────
# Test 1: no checkpoint, no scan (baseline)
results["no_ckpt_no_scan"] = run_test(
    "raw A2A: no checkpoint, no scan", use_checkpoint=False, use_scan=False, n_layers=2)

# Test 2: checkpoint, no scan
results["ckpt_no_scan"] = run_test(
    "raw A2A: checkpoint, no scan", use_checkpoint=True, use_scan=False, n_layers=2)

# Test 3: no checkpoint, with scan
results["no_ckpt_scan"] = run_test(
    "raw A2A: no checkpoint, scan", use_checkpoint=False, use_scan=True, n_layers=2)

# Test 4: checkpoint + scan (the failing case in JAX 0.9.2)
results["ckpt_scan"] = run_test(
    "raw A2A: checkpoint + scan (JAX 0.9.2 crash)", use_checkpoint=True, use_scan=True, n_layers=4)

# ── Test custom_vjp A2A (ragged_dot backward) ─────────────────────────────────
import sys
sys.path.insert(0, "/home/sivaibhav_google_com/dsv3")

def _a2a_custom_vjp_body(x, axis_name):
    """Same as _a2a_body but uses custom_vjp: A2A fwd + all_gather bwd."""
    T, D = x.shape
    EP = jax.lax.axis_size(axis_name)
    chunk = T // EP
    send_sizes = jnp.full((EP,), chunk, dtype=jnp.int32)
    input_offsets = jnp.arange(EP, dtype=jnp.int32) * chunk
    recv_sizes   = jax.lax.all_to_all(send_sizes,    axis_name, 0, 0, tiled=True)
    recv_offsets = jnp.concatenate([jnp.zeros(1, jnp.int32), jnp.cumsum(recv_sizes)[:-1]])
    output_offsets = jax.lax.all_to_all(recv_offsets, axis_name, 0, 0, tiled=True)
    recv_buf = jnp.zeros_like(x)

    @jax.custom_vjp
    def _a2a(operand):
        return jax.lax.ragged_all_to_all(
            operand, recv_buf, input_offsets, send_sizes,
            output_offsets, recv_sizes, axis_name=axis_name)

    def _a2a_fwd(operand):
        return _a2a(operand), operand

    def _a2a_bwd(saved, g):
        # Backward via all_gather (the proven path) instead of transpose A2A
        d_operand = jax.lax.all_gather(g, axis_name, axis=0, tiled=True)
        # Only take back the slice this device originally owned
        my_idx = jax.lax.axis_index(axis_name)
        d_operand = jax.lax.dynamic_slice(d_operand, [my_idx * chunk, 0], [chunk, D])
        return (d_operand,)

    _a2a.defvjp(_a2a_fwd, _a2a_bwd)
    return _a2a(x)

def _body_custom_vjp(x, w, axis_name):
    dispatched = _a2a_custom_vjp_body(x, axis_name)
    out = dispatched @ w
    gathered = _a2a_custom_vjp_body(out, axis_name)
    return gathered

def run_custom_vjp_test(name, use_checkpoint, use_scan, n_layers):
    print(f"\n{'='*60}")
    print(f"Test (custom_vjp): {name}")
    print(f"  checkpoint={use_checkpoint}, scan={use_scan}, layers={n_layers}")
    rng = jax.random.PRNGKey(0)
    x = jax.random.normal(rng, (EP * T_local, D))
    ws = jax.random.normal(rng, (n_layers, D, D))

    @functools.partial(shard_map, mesh=mesh,
                       in_specs=(P("ep", None), P(None, None, None)),
                       out_specs=P("ep", None),
                       check_rep=False)
    def _sharded_layer(x_local, w_local):
        def _fn(xl, wl):
            return _body_custom_vjp(xl, wl, "ep")
        if use_checkpoint:
            _fn = jax.checkpoint(_fn)
        return _fn(x_local, w_local)

    def _one_layer(carry, w):
        return _sharded_layer(carry, w), None

    def forward(x, ws):
        if use_scan:
            out, _ = jax.lax.scan(_one_layer, x, ws)
        else:
            out = x
            for i in range(n_layers):
                out = _sharded_layer(out, ws[i])
        return out.sum()

    try:
        g = jax.jit(jax.grad(forward))(x, ws)
        g.block_until_ready()
        print(f"  PASSED — grad norm = {jnp.linalg.norm(g):.4f}")
        return True
    except Exception as e:
        print(f"  FAILED — {type(e).__name__}: {str(e)[:200]}")
        return False

results["cvjp_ckpt_scan"] = run_custom_vjp_test(
    "custom_vjp A2A: checkpoint + scan", use_checkpoint=True, use_scan=True, n_layers=4)

print(f"\n{'='*60}")
print("Summary:")
for k, v in results.items():
    print(f"  {k}: {'PASS' if v else 'FAIL'}")
