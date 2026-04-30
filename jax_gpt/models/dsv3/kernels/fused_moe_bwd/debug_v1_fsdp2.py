"""Debug v1 ratio=0.84 for FSDP=2, EP=2, K=2.

Compare intermediate values between jax.vjp and streaming_bwd v1.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "mini_dsv3"))

import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from jax.experimental.shard_map import shard_map
import functools

D     = 32   # small
D_moe = 8
E     = 4
K     = 2
T     = 16
SEED  = 0

def make_mesh(fsdp, ep):
    devs = jax.devices()
    arr  = np.array(devs[:fsdp*ep]).reshape(fsdp, ep)
    return Mesh(arr, ("fsdp", "ep"))

def make_sharded(data, mesh, spec):
    sharding  = NamedSharding(mesh, spec)
    idx_map   = sharding.addressable_devices_indices_map(data.shape)
    shards    = [jax.device_put(data[idx_map[d]], d) for d in idx_map]
    return jax.make_array_from_single_device_arrays(data.shape, sharding, shards)

def main():
    fsdp, ep = 2, 2
    mesh = make_mesh(fsdp, ep)
    act_spec = P(("ep","fsdp"), None)
    ep_axis  = "ep"

    T_per_dev = T // fsdp
    E_local   = E // ep
    avg_tpe   = T_per_dev * K // E_local
    max_tpe   = min(T_per_dev * K, max(4, 2 * avg_tpe))
    print(f"T_per_dev={T_per_dev} E_local={E_local} avg_tpe={avg_tpe} max_tpe={max_tpe}")

    key = jax.random.PRNGKey(SEED)
    k0,k1,k2,k3 = jax.random.split(key,4)
    fx_np = np.array(jax.random.normal(k0,(T,D),   dtype=jnp.bfloat16))
    w0_np = np.array(jax.random.normal(k1,(E,D,D_moe),dtype=jnp.bfloat16)*0.02)
    w1_np = np.array(jax.random.normal(k2,(E,D,D_moe),dtype=jnp.bfloat16)*0.02)
    wo_np = np.array(jax.random.normal(k3,(E,D_moe,D),dtype=jnp.bfloat16)*0.02)
    rng   = np.random.default_rng(SEED)
    fi_np = np.argsort(rng.random((T,E)),axis=-1)[:,:K].astype(np.int32)
    raw   = rng.random((T,K)).astype(np.float32)
    fw_np = (raw / raw.sum(axis=-1, keepdims=True)).astype(np.float32)

    print("\nfi_np (routing):")
    print(fi_np)

    # Shard inputs
    fx   = make_sharded(fx_np, mesh, act_spec)
    w0   = make_sharded(w0_np, mesh, P("ep",None,"fsdp"))
    w1   = make_sharded(w1_np, mesh, P("ep",None,"fsdp"))
    wout = make_sharded(wo_np, mesh, P("ep","fsdp",None))
    fw   = make_sharded(fw_np, mesh, act_spec)
    fi   = make_sharded(fi_np, mesh, act_spec)

    from jax_gpt.models.dsv3.model import _moe_jax_ep_fn

    # ---- Step 1: run both versions and get gradients ----
    def loss_fn(fx_, fw_, w0_, w1_, wout_):
        out = _moe_jax_ep_fn(fx_, fi, fw_, w0_, w1_, wout_, mesh, K,
                              act_spec, ep_axis, max_tpe, 0)  # version=0 (jax.vjp ref)
        return jnp.sum(out)

    def loss_fn_v1(fx_, fw_, w0_, w1_, wout_):
        out = _moe_jax_ep_fn(fx_, fi, fw_, w0_, w1_, wout_, mesh, K,
                              act_spec, ep_axis, max_tpe, 1)  # version=1
        return jnp.sum(out)

    val_ref, grads_ref = jax.value_and_grad(loss_fn,    argnums=(0,1,2,3,4))(fx,fw,w0,w1,wout)
    val_v1,  grads_v1  = jax.value_and_grad(loss_fn_v1, argnums=(0,1,2,3,4))(fx,fw,w0,w1,wout)

    print(f"\nloss ref={float(val_ref):.6f}, v1={float(val_v1):.6f}")

    d_fx_ref = np.array(jnp.asarray(grads_ref[0]).astype(jnp.float32))
    d_fx_v1  = np.array(jnp.asarray(grads_v1[0]).astype(jnp.float32))

    print(f"\nd_fx_ref norm={np.linalg.norm(d_fx_ref):.6f}")
    print(f"d_fx_v1  norm={np.linalg.norm(d_fx_v1):.6f}")
    print(f"ratio = {np.linalg.norm(d_fx_v1)/np.linalg.norm(d_fx_ref):.4f}")
    print(f"max_abs_err = {np.max(np.abs(d_fx_ref - d_fx_v1)):.4e}")

    # ---- Step 2: inspect g_full inside shard_map ----
    # We'll capture g_full by running a custom backward manually
    print("\n---- Manually running VJP pieces ----")

    # Global gradient = ones (from jnp.sum)
    g_global = jnp.ones((T, D), dtype=jnp.float32)
    g_sharded = make_sharded(np.ones((T, D), dtype=np.float32), mesh, act_spec)

    # Inside shard_map, each device gets its g_l according to act_spec
    # Let's capture g_l, g_l_fsdp, g_full from the shard_map body

    g_full_store = []
    d_tok_store  = []

    def _debug_bwd_fn(g_l, fx_l, fw_l, fi_l, w0_l, w1_l, wout_l):
        from backward_kernel import fused_ep_moe_bwd_streaming as _bwd
        D = g_l.shape[1]
        g_l_fsdp = jax.lax.psum(g_l, "fsdp")
        g_full   = jax.lax.all_gather(g_l_fsdp, "ep", axis=0, tiled=True)
        fx_full  = jax.lax.all_gather(fx_l, "ep", axis=0, tiled=True)
        fw_full  = jax.lax.all_gather(fw_l, "ep", axis=0, tiled=True)
        fi_full  = jax.lax.all_gather(fi_l, "ep", axis=0, tiled=True)
        w1_stk   = jnp.stack([w0_l, w1_l], axis=1)
        E_global = w0_l.shape[0] * ep

        d_tok_p, d_w1_p, d_wo_p, d_topk_p = _bwd(
            g_full, fx_full, w1_stk, wout_l,
            gating_output=None, top_k=K,
            scoring_fn="sigmoid", renormalize_topk_logits=True, act_fn="silu",
            ep_axis_name="ep", max_tpe=max_tpe,
            top_k_indices_precomputed=fi_full,
            top_k_weights_precomputed=fw_full.astype(jnp.float32),
            return_dtopk=True, E_global_override=E_global,
        )
        T_ep_local = g_l.shape[0]
        device_ep  = jax.lax.axis_index("ep")
        d_tok_full = jax.lax.psum(d_tok_p, "ep")
        d_tok_l    = jax.lax.dynamic_slice(d_tok_full, (device_ep*T_ep_local,0), (T_ep_local,D))

        # Debug: return g_full norm and d_tok_full norm as extra outputs
        g_full_norm  = jnp.linalg.norm(g_full)
        d_tok_l_norm = jnp.linalg.norm(d_tok_l)
        return (d_tok_l.astype(g_l.dtype),
                jnp.stack([g_full_norm, d_tok_l_norm]))

    # Run in shard_map to get norms
    debug_out = shard_map(
        _debug_bwd_fn, mesh=mesh,
        in_specs=(act_spec, act_spec, act_spec, act_spec,
                  P("ep",None,"fsdp"), P("ep",None,"fsdp"), P("ep","fsdp",None)),
        out_specs=(act_spec, P(("ep","fsdp"),None)),
        check_rep=False,
    )(g_sharded, fx, fw, fi, w0, w1, wout)

    d_tok_smap = np.array(jnp.asarray(debug_out[0]).astype(jnp.float32))
    norms      = np.array(jnp.asarray(debug_out[1]))  # (4, 2) = 4 devices x 2 values

    print(f"\nManual d_tok norm = {np.linalg.norm(d_tok_smap):.6f}")
    print(f"v1 d_fx   norm = {np.linalg.norm(d_fx_v1):.6f}")
    print(f"ref d_fx   norm = {np.linalg.norm(d_fx_ref):.6f}")
    print(f"\ng_full norms per device: {norms[:, 0]}")
    print(f"d_tok norms per device:  {norms[:, 1]}")

    # ---- Step 3: What does jax.vjp use for g_full? ----
    # Run jax.vjp version=0 with a custom version that captures g_full
    # We do this by forward-propagating g through the same ops
    g_l_samples = {}

    def _ref_capture_fn(g_l, fx_l, fw_l, fi_l, w0_l, w1_l, wout_l):
        # Reproduce what jax.vjp would compute as the gradient of partial_out
        g_l_fsdp = jax.lax.psum(g_l, "fsdp")
        g_full   = jax.lax.all_gather(g_l_fsdp, "ep", axis=0, tiled=True)
        g_norm   = jnp.linalg.norm(g_full)
        ep_idx   = jax.lax.axis_index("ep")
        fsdp_idx = jax.lax.axis_index("fsdp")
        return g_norm * jnp.ones((g_l.shape[0], 1))

    ref_capture = shard_map(
        _ref_capture_fn, mesh=mesh,
        in_specs=(act_spec, act_spec, act_spec, act_spec,
                  P("ep",None,"fsdp"), P("ep",None,"fsdp"), P("ep","fsdp",None)),
        out_specs=P(("ep","fsdp"),None),
        check_rep=False,
    )(g_sharded, fx, fw, fi, w0, w1, wout)

    print(f"\ng_full norms (should be same as streaming_bwd): {np.array(jnp.asarray(ref_capture)).flatten()}")

    print("\nDone.")

if __name__ == "__main__":
    main()
