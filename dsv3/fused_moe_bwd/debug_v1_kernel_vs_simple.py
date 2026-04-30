"""Compare streaming_bwd v1 kernel vs a simple pure-JAX backward inside the same shard_map.

If simple_bwd matches jax.vjp but v1_kernel does not → bug is in the kernel.
If simple_bwd also mismatches → bug is in the shard_map setup.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "mini_dsv3"))

import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from jax import lax
import functools

D     = 32
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
    sharding = NamedSharding(mesh, spec)
    idx_map  = sharding.addressable_devices_indices_map(data.shape)
    shards   = [jax.device_put(data[idx_map[d]], d) for d in idx_map]
    return jax.make_array_from_single_device_arrays(data.shape, sharding, shards)

# ---- Pure-JAX simple backward (no kernel) ----
def simple_moe_bwd(g_full, fx_full, fw_full, fi_full, w0_l, w1_l, wout_l,
                   ep_axis_name, max_tpe, K, E_global):
    """Minimal pure-JAX backward of the EP MoE forward (using partial D_moe weights).

    This mirrors _expert_mlp_ep_body_ep_sharded backward without using the
    fused_ep_moe_bwd_streaming kernel, so we can isolate kernel bugs.
    """
    T_local, D   = fx_full.shape  # T/FSDP
    E_local      = w0_l.shape[0]
    F_shard      = w0_l.shape[2]  # D_moe / FSDP

    ep_idx       = lax.axis_index(ep_axis_name)
    expert_offset = ep_idx * E_local

    # Flatten routing
    fi_flat = fi_full.reshape(-1)  # (T_local * K,)
    fw_flat = fw_full.reshape(-1).astype(jnp.float32)
    tok_ids = jnp.repeat(jnp.arange(T_local, dtype=jnp.int32), K)

    # Per-device expert: is this (token,k) pair assigned to a local expert?
    is_local  = (fi_flat >= expert_offset) & (fi_flat < expert_offset + E_local)
    local_exp = jnp.where(is_local, fi_flat - expert_offset, E_local)
    local_fw  = jnp.where(is_local, fw_flat, 0.0)

    # Sort by local expert id
    sort_ord      = jnp.argsort(local_exp, stable=True)
    sorted_exp    = local_exp[sort_ord]
    sorted_tok    = tok_ids[sort_ord]
    sorted_fw     = local_fw[sort_ord]

    expert_starts = jnp.searchsorted(sorted_exp, jnp.arange(E_local))
    expert_ends   = jnp.searchsorted(sorted_exp, jnp.arange(1, E_local+1))

    # Gather d_out and tokens per expert, compute d_tok contribution
    d_tok_accum = jnp.zeros((T_local, D), dtype=jnp.float32)
    d_w0_list, d_w1_list, d_wo_list = [], [], []

    for e in range(E_local):
        s_e = expert_starts[e]
        n_e = expert_ends[e] - s_e
        tids_e = lax.dynamic_slice(sorted_tok, (s_e,), (max_tpe,))
        fw_e   = lax.dynamic_slice(sorted_fw,  (s_e,), (max_tpe,))
        valid  = jnp.arange(max_tpe) < n_e
        valid_f = valid.astype(jnp.float32)
        safe_tids = jnp.where(valid, tids_e, 0)

        x_e   = fx_full[safe_tids] * valid_f[:, None]    # (max_tpe, D)
        g_e   = g_full[safe_tids]                          # (max_tpe, D)
        g_es  = g_e * (fw_e * valid_f)[:, None]           # scaled by routing weight

        # Forward recompute
        hg_e  = x_e @ w0_l[e]                             # (max_tpe, D_moe/FSDP)
        hu_e  = x_e @ w1_l[e]
        ha_e  = jax.nn.silu(hg_e) * hu_e

        # Backward
        d_ha  = g_es @ wout_l[e].T                        # (max_tpe, D_moe/FSDP)
        d_wo_list.append(ha_e.T @ g_es)                   # (D_moe/FSDP, D)

        sig_hg  = jax.nn.sigmoid(hg_e)
        silu_g  = sig_hg * (1 + hg_e * (1 - sig_hg))
        d_hu  = d_ha * jax.nn.silu(hg_e)
        d_hg  = d_ha * hu_e * silu_g

        d_w0_list.append((x_e.T @ d_hg) * valid_f[:, None].sum())   # not quite right
        d_w1_list.append((x_e.T @ d_hu) * valid_f[:, None].sum())

        # d_tok: shape (max_tpe, D) zeroed for padding
        d_x_e = (d_hg @ w0_l[e].T + d_hu @ w1_l[e].T) * valid_f[:, None]

        # Scatter-add back to d_tok
        d_tok_accum = d_tok_accum.at[safe_tids].add(d_x_e)

    return d_tok_accum

def main():
    fsdp, ep = 2, 2
    mesh = make_mesh(fsdp, ep)
    act_spec = P(("ep","fsdp"), None)
    ep_axis  = "ep"
    fsdp_axis = "fsdp"

    T_per_dev = T // fsdp
    E_local   = E // ep
    avg_tpe   = T_per_dev * K // E_local
    max_tpe   = min(T_per_dev * K, max(4, 2 * avg_tpe))

    key = jax.random.PRNGKey(SEED)
    k0,k1,k2,k3 = jax.random.split(key,4)
    fx_np = np.array(jax.random.normal(k0,(T,D),     dtype=jnp.float32))
    w0_np = np.array(jax.random.normal(k1,(E,D,D_moe),dtype=jnp.float32)*0.1)
    w1_np = np.array(jax.random.normal(k2,(E,D,D_moe),dtype=jnp.float32)*0.1)
    wo_np = np.array(jax.random.normal(k3,(E,D_moe,D),dtype=jnp.float32)*0.1)
    rng   = np.random.default_rng(SEED)
    fi_np = np.argsort(rng.random((T,E)),axis=-1)[:,:K].astype(np.int32)
    raw   = rng.random((T,K)).astype(np.float32)
    fw_np = (raw / raw.sum(axis=-1, keepdims=True)).astype(np.float32)

    print("fi_np routing:")
    print(fi_np)

    fx   = make_sharded(fx_np, mesh, act_spec)
    w0   = make_sharded(w0_np, mesh, P("ep",None,"fsdp"))
    w1   = make_sharded(w1_np, mesh, P("ep",None,"fsdp"))
    wout = make_sharded(wo_np, mesh, P("ep","fsdp",None))
    fw   = make_sharded(fw_np, mesh, act_spec)
    fi   = make_sharded(fi_np, mesh, act_spec)

    from model import _moe_jax_ep_fn

    def loss_fn(fx_, fw_, w0_, w1_, wout_, version):
        out = _moe_jax_ep_fn(fx_, fi, fw_, w0_, w1_, wout_, mesh, K,
                              act_spec, ep_axis, max_tpe, version)
        return jnp.sum(out)

    # Reference (jax.vjp)
    _, grads_ref = jax.value_and_grad(
        lambda *a: loss_fn(*a, 0), argnums=(0,1,2,3,4))(fx,fw,w0,w1,wout)
    # v1
    _, grads_v1  = jax.value_and_grad(
        lambda *a: loss_fn(*a, 1), argnums=(0,1,2,3,4))(fx,fw,w0,w1,wout)

    d_fx_ref = np.array(jnp.asarray(grads_ref[0]).astype(jnp.float32))
    d_fx_v1  = np.array(jnp.asarray(grads_v1[0]).astype(jnp.float32))

    print(f"\nRef d_fx norm: {np.linalg.norm(d_fx_ref):.6f}")
    print(f"V1  d_fx norm: {np.linalg.norm(d_fx_v1):.6f}")
    print(f"V1 ratio: {np.linalg.norm(d_fx_v1)/np.linalg.norm(d_fx_ref):.4f}")

    # ---- Now run simple_bwd inside shard_map, manually ----
    # We implement the full streaming_bwd wrapper manually but replace the kernel
    # with simple_moe_bwd

    def _simple_bwd_fn(g_l, fx_l, fw_l, fi_l, w0_l, w1_l, wout_l):
        D_ = g_l.shape[1]
        g_l_fsdp = lax.psum(g_l, fsdp_axis)
        g_full   = lax.all_gather(g_l_fsdp, ep_axis, axis=0, tiled=True)
        fx_full  = lax.all_gather(fx_l, ep_axis, axis=0, tiled=True)
        fw_full  = lax.all_gather(fw_l, ep_axis, axis=0, tiled=True)
        fi_full  = lax.all_gather(fi_l, ep_axis, axis=0, tiled=True)

        d_tok_p = simple_moe_bwd(g_full.astype(jnp.float32),
                                  fx_full.astype(jnp.float32),
                                  fw_full.astype(jnp.float32),
                                  fi_full, w0_l, w1_l, wout_l,
                                  ep_axis, max_tpe, K, E)

        d_tok_full = lax.psum(d_tok_p, ep_axis)
        T_ep_local = g_l.shape[0]
        device_ep  = lax.axis_index(ep_axis)
        d_tok_l    = lax.dynamic_slice(d_tok_full, (device_ep*T_ep_local,0), (T_ep_local,D_))
        return d_tok_l.astype(g_l.dtype)

    from jax.experimental.shard_map import shard_map

    g_ones = make_sharded(np.ones((T,D), dtype=np.float32), mesh, act_spec)

    d_fx_simple = shard_map(
        _simple_bwd_fn, mesh=mesh,
        in_specs=(act_spec, act_spec, act_spec, act_spec,
                  P("ep",None,"fsdp"), P("ep",None,"fsdp"), P("ep","fsdp",None)),
        out_specs=act_spec,
        check_rep=False,
    )(g_ones, fx, fw, fi, w0, w1, wout)

    d_fx_simple_np = np.array(jnp.asarray(d_fx_simple).astype(jnp.float32))
    print(f"\nSimple d_fx norm: {np.linalg.norm(d_fx_simple_np):.6f}")
    print(f"Simple ratio vs ref: {np.linalg.norm(d_fx_simple_np)/np.linalg.norm(d_fx_ref):.4f}")

    # ---- Compare simple vs ref element-wise ----
    print(f"\nd_fx_ref - d_fx_simple max_abs: {np.max(np.abs(d_fx_ref - d_fx_simple_np)):.4e}")
    print(f"d_fx_ref - d_fx_v1    max_abs: {np.max(np.abs(d_fx_ref - d_fx_v1)):.4e}")

    print("\nPer-token comparison (first 8 tokens, norm):")
    for t in range(min(T, 8)):
        r = np.linalg.norm(d_fx_ref[t])
        s = np.linalg.norm(d_fx_simple_np[t])
        v = np.linalg.norm(d_fx_v1[t])
        print(f"  t={t}: ref={r:.4f}  simple={s:.4f}  v1={v:.4f}  "
              f"simple/ref={s/(r+1e-12):.3f}  v1/ref={v/(r+1e-12):.3f}")

if __name__ == "__main__":
    main()
