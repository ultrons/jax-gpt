"""Minimal repro of pl.kernel + lax.scan + jax.grad ref leak.

This file deliberately uses ONLY:
  - jax / jax.numpy
  - jax.experimental.pallas (+ tpu, tpu_sc submodules)

It defines an identity Pallas kernel and its (also-identity) backward kernel,
both via `pl.kernel(...)`, wraps them in `jax.custom_vjp`, asserts
correctness against a numpy reference, and then bisects the
ref-leak failure across three loop styles:

  (a) jax.lax.scan + jax.grad   → fails  (leaks an input Ref)
  (b) jax.lax.scan(unroll=N)+grad → fails  (full unroll is not enough)
  (c) Python for loop + jax.grad → succeeds (workaround, correct grad)

Run with `--entrypoint python` on a v7x SparseCore image; one host with
4 TPU chips is enough.
"""
import sys
import jax
import jax.numpy as jnp
import functools
import numpy as np
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas import tpu_sc as plsc

# Distributed init (no-op on single host)
try:
    jax.distributed.initialize()
except Exception:
    pass

print(f"[info] jax: {jax.__version__}")
print(f"[info] devices: {len(jax.devices())} "
      f"(process {jax.process_index()}/{jax.process_count()})")
if jax.process_index() != 0:
    sys.exit(0)

sc_info = pltpu.get_tpu_info().sparse_core
assert sc_info is not None, "needs SparseCore (v5p / v6e / v7x)"

SC_MESH = plsc.VectorSubcoreMesh(
    core_axis_name="core", subcore_axis_name="subcore", num_cores=1)

M, N = 2, 16             # SC `Get` only supports (32,) or (2, 16) bf16 tiles
DT = jnp.bfloat16


def _copy_block(in_blk, out_blk):
    out_blk[...] = in_blk[...]


def _identity_body(in_ref, out_ref):
    pltpu.emit_pipeline(
        _copy_block,
        grid=(1,),
        in_specs=pl.BlockSpec((M, N), lambda i: (0, 0)),
        out_specs=pl.BlockSpec((M, N), lambda i: (0, 0)),
    )(in_ref, out_ref)


# Forward Pallas kernel: identity
identity_fwd = pl.kernel(
    _identity_body,
    out_shape=jax.ShapeDtypeStruct((M, N), DT),
    mesh=SC_MESH,
)

# Backward Pallas kernel: also identity (d/dx of identity is identity)
identity_bwd = pl.kernel(
    _identity_body,
    out_shape=jax.ShapeDtypeStruct((M, N), DT),
    mesh=SC_MESH,
)


@jax.custom_vjp
def cvjp_identity(x):
    return cvjp_identity_fwd(x)[0]

def cvjp_identity_fwd(x):
    return identity_fwd(x), None

def cvjp_identity_bwd(_, g):
    # gradient of identity wrt x is the cotangent g, computed by the bwd kernel
    return (identity_bwd(g),)

cvjp_identity.defvjp(cvjp_identity_fwd, cvjp_identity_bwd)


# ---------------------------------------------------------------------------
# Step 1: confirm correctness of the kernel pair OUTSIDE any loop construct.
# ---------------------------------------------------------------------------
key = jax.random.key(0)
x = jax.random.normal(key, (M, N), dtype=DT)

y = jax.jit(cvjp_identity)(x)
assert y.shape == x.shape and y.dtype == x.dtype
np.testing.assert_array_equal(np.asarray(y), np.asarray(x))
print("[PASS] cvjp_identity(x) == x")

g = jax.jit(jax.grad(lambda x: jnp.sum(cvjp_identity(x).astype(jnp.float32))))(x)
np.testing.assert_array_equal(np.asarray(g), np.ones_like(np.asarray(x), dtype=np.float32))
print("[PASS] grad(sum(cvjp_identity(x))) == ones (reference: sum-of-identity)")


# ---------------------------------------------------------------------------
# Step 2: drop the kernel into a layered loop, take grad, observe leak.
# ---------------------------------------------------------------------------
N_LAYERS = 4

def layer(carry, _):
    """One 'layer': apply cvjp_identity, return new carry + per-layer output."""
    h = carry
    out = cvjp_identity(h)
    return out, out


def _trace(name, fn, *abs_args):
    print(f"\n=== {name} ===")
    try:
        jaxpr = jax.make_jaxpr(fn)(*abs_args)
        avals = [v.aval for v in jaxpr.jaxpr.outvars]
        is_ref = any("Ref" in type(a).__name__ for a in avals)
        flag = "*** REF LEAK ***" if is_ref else "ok"
        print(f"  out aval: {avals}")
        print(f"  out aval types: {[type(a).__name__ for a in avals]}  {flag}")
    except Exception as e:
        msg = str(e)
        if "mutable array reference" in msg:
            print(f"  REF LEAK: {type(e).__name__}: {msg[:300]}")
        else:
            print(f"  TRACE FAILED ({type(e).__name__}): {msg[:300]}")


x_abs = jax.ShapeDtypeStruct((M, N), DT)

# (a) lax.scan + grad — predict leak
@jax.jit
def loss_scan(x):
    _, ys = jax.lax.scan(layer, x, jnp.arange(N_LAYERS))
    return jnp.sum(ys.astype(jnp.float32))

@jax.jit
def grad_scan(x):
    return jax.grad(loss_scan)(x)

_trace("(a) jax.jit(grad(jax.lax.scan(cvjp_identity)))",
       grad_scan, x_abs)

# (b) lax.scan(unroll=N_LAYERS) + grad — predict leak (scan_p still emitted)
@jax.jit
def loss_scan_unrolled(x):
    _, ys = jax.lax.scan(layer, x, jnp.arange(N_LAYERS), unroll=N_LAYERS)
    return jnp.sum(ys.astype(jnp.float32))

@jax.jit
def grad_scan_unrolled(x):
    return jax.grad(loss_scan_unrolled)(x)

_trace("(b) jax.jit(grad(jax.lax.scan(unroll=N)(cvjp_identity)))",
       grad_scan_unrolled, x_abs)

# (c) Python for loop + grad — predict success
@jax.jit
def loss_pyloop(x):
    h, outs = x, []
    for i in range(N_LAYERS):
        h, out = layer(h, i)
        outs.append(out)
    return jnp.sum(jnp.stack(outs).astype(jnp.float32))

@jax.jit
def grad_pyloop(x):
    return jax.grad(loss_pyloop)(x)

_trace("(c) jax.jit(grad(python_for_loop(cvjp_identity)))",
       grad_pyloop, x_abs)

# Step 3: numerically verify the workaround
g_pyloop = grad_pyloop(x)
g_ref    = jax.grad(lambda x: jnp.sum(jnp.stack(
    [x] * N_LAYERS).astype(jnp.float32)))(x)  # reference: same operation in pure jnp
np.testing.assert_array_equal(np.asarray(g_pyloop), np.asarray(g_ref))
print("\n[PASS] (c) Python-for-loop grad matches jnp reference")


# ---------------------------------------------------------------------------
# K13: production sc_gather_reduce + custom_vjp + scan + grad
# (matches the actual DSv3 wiring at production shapes)
# ---------------------------------------------------------------------------
import sys as _sys, functools as _ft
_sys.path.insert(0, "/app")
from kernels.gather_reduce_pallas import sc_gather_reduce as real_sc

K_TOPK = 8
T_ALL  = 65536
D_DIM  = 7168

@_ft.partial(jax.custom_vjp, nondiff_argnums=(3, 4, 5, 6))
def _sc_combine_with_vjp(out_local, idx, local_tids,
                         zero_rows: int, K_top: int, T_all: int, D: int):
    return _sc_combine_fwd(out_local, idx, local_tids,
                           zero_rows, K_top, T_all, D)[0]
def _sc_combine_fwd(out_local, idx, local_tids,
                    zero_rows, K_top, T_all, D):
    zero_pad = jnp.zeros((zero_rows, D), dtype=out_local.dtype)
    out_with_zero = jnp.concatenate([out_local, zero_pad], axis=0)
    result = real_sc(out_with_zero, idx, reduce_group_size=K_top, single_sc=True)
    return result, (idx, local_tids)
def _sc_combine_bwd(zero_rows, K_top, T_all, D, res, d_result):
    idx, local_tids = res
    d_out_local = d_result[local_tids].astype(d_result.dtype)
    return d_out_local, jnp.zeros_like(idx), jnp.zeros_like(local_tids)
_sc_combine_with_vjp.defvjp(_sc_combine_fwd, _sc_combine_bwd)

def K13_layer(carry, _):
    out_local, idx, local_tids = carry
    out = _sc_combine_with_vjp(out_local, idx, local_tids,
                               64, K_TOPK, T_ALL, D_DIM)
    new_out_local = jnp.concatenate(
        [out, jnp.zeros((out_local.shape[0]-out.shape[0], D_DIM),
                        dtype=out_local.dtype)], axis=0)
    return (new_out_local, idx, local_tids), out

@jax.jit
def K13_loss(out_local, idx, local_tids):
    _, ys = jax.lax.scan(K13_layer,
                         (out_local, idx, local_tids),
                         jnp.arange(N_LAYERS))
    return jnp.sum(ys.astype(jnp.float32))

@jax.jit
def K13_grad(out_local, idx, local_tids):
    return jax.grad(K13_loss)(out_local, idx, local_tids)

out_local_a   = jax.ShapeDtypeStruct((T_ALL * K_TOPK - 64, D_DIM), jnp.bfloat16)
idx_a         = jax.ShapeDtypeStruct((T_ALL * K_TOPK,),             jnp.int32)
local_tids_a  = jax.ShapeDtypeStruct((T_ALL * K_TOPK - 64,),        jnp.int32)

_trace("K13: production sc_gather_reduce + cvjp + scan + grad",
       K13_grad, out_local_a, idx_a, local_tids_a)
