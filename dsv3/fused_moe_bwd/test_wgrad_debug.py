"""Minimal debug test for weight gradient discrepancy in EP=2, FSDP=1 and EP=2, FSDP=2.

Compares streaming kernel d_w with reference (jax.vjp on forward shard_map).
Uses simple MoE formulation without the full model.py integration.
"""
import sys, types, logging
_vllm = types.ModuleType("vllm"); _vllm_logger = types.ModuleType("vllm.logger")
class _VL(logging.Logger):
    def warning_once(self,*a,**k): self.warning(*a,**k)
    def info_once(self,*a,**k): self.info(*a,**k)
    def debug_once(self,*a,**k): self.debug(*a,**k)
logging.setLoggerClass(_VL)
_vllm_logger.init_logger = lambda n: logging.getLogger(n)
_vllm_logger._VllmLogger = _VL
_vllm_logger.init_vllm_logger = lambda n: logging.getLogger(n)
sys.modules.setdefault("vllm", _vllm)
sys.modules.setdefault("vllm.logger", _vllm_logger)
sys.path.insert(0, "/home/sivaibhav_google_com/tpu-inference")
sys.path.insert(0, "/home/sivaibhav_google_com/dsv3/mini_dsv3")
sys.path.insert(0, "/home/sivaibhav_google_com/dsv3/fused_moe_bwd")

import numpy as np
import jax
import jax.numpy as jnp
from jax import lax
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from jax.experimental.shard_map import shard_map
from backward_kernel import fused_ep_moe_bwd_streaming


# Simple MoE forward (no gating recomputation — just use precomputed fi, fw)
def moe_fwd_ep(fx_l, fi_l, fw_l, w0_l, w1_l, wout_l, K, ep_axis, max_tpe):
    """EP-sharded MoE forward mimicking _expert_mlp_ep_body_ep_sharded."""
    from jax.experimental.shard_map import shard_map as sm
    flat_x   = lax.all_gather(fx_l, ep_axis, axis=0, tiled=True)
    flat_fi  = lax.all_gather(fi_l, ep_axis, axis=0, tiled=True)
    flat_fw  = lax.all_gather(fw_l, ep_axis, axis=0, tiled=True)

    T, D = flat_x.shape
    E_local = w0_l.shape[0]
    my_idx = lax.axis_index(ep_axis)
    local_start = jnp.int32(my_idx * E_local)

    flat_idx = flat_fi.reshape(-1)   # (T*K,)
    flat_w   = flat_fw.reshape(-1)   # (T*K,)
    tok_ids  = jnp.repeat(jnp.arange(T, dtype=jnp.int32), K)

    valid     = (flat_idx >= local_start) & (flat_idx < local_start + E_local)
    local_exp = jnp.where(valid, flat_idx - local_start, E_local).astype(jnp.int32)

    argsorted   = jnp.argsort(local_exp, stable=True)
    sorted_tids = tok_ids[argsorted]
    sorted_ws   = flat_w[argsorted]
    sorted_exp  = local_exp[argsorted]

    expert_starts = jnp.searchsorted(sorted_exp, jnp.arange(E_local))
    expert_ends   = jnp.searchsorted(sorted_exp, jnp.arange(1, E_local + 1))

    all_weighted_out = []
    all_safe_tids_list = []
    for e in range(E_local):
        start_e = expert_starts[e]
        n_e     = expert_ends[e] - start_e
        e_tids  = lax.dynamic_slice(sorted_tids, (start_e,), (max_tpe,))
        e_ws    = lax.dynamic_slice(sorted_ws,   (start_e,), (max_tpe,))
        valid_e = jnp.arange(max_tpe) < n_e
        safe_tids = jnp.where(valid_e, e_tids, 0)
        all_safe_tids_list.append(safe_tids)

        sel_x  = flat_x[safe_tids]
        gate_e = jax.nn.silu(sel_x @ w0_l[e]) * (sel_x @ w1_l[e])
        out_e  = gate_e @ wout_l[e]
        all_weighted_out.append(out_e * e_ws[:, None] * valid_e[:, None])

    all_out   = jnp.concatenate(all_weighted_out, axis=0)
    all_tids  = jnp.concatenate(all_safe_tids_list, axis=0)
    partial_out = jnp.zeros((T, D), dtype=flat_x.dtype).at[all_tids].add(all_out)

    partial_out_ep = lax.psum_scatter(partial_out, ep_axis, scatter_dimension=0, tiled=True)
    return lax.psum(partial_out_ep, "fsdp")


def run_test(EP, FSDP, T=32, D=64, F=32, E=8, K=4):
    print(f"\n=== EP={EP} FSDP={FSDP} T={T} D={D} F={F} E={E} K={K} ===")
    devs_count = EP * FSDP
    all_devs = jax.local_devices()[:devs_count]
    if len(all_devs) < devs_count:
        print(f"  SKIP: need {devs_count} devices, have {len(jax.local_devices())}")
        return

    devs = np.array(all_devs).reshape(EP, FSDP)
    mesh = Mesh(devs, ("ep", "fsdp"))

    E_local = E // EP
    F_local = F // FSDP
    max_tpe = max(1, 2 * (T // FSDP) * K // E)

    key = jax.random.PRNGKey(42)
    k1, k2, k3, k4, k5 = jax.random.split(key, 5)
    tokens = jax.random.normal(k1, (T, D), jnp.float32) * 0.1
    w0     = jax.random.normal(k2, (E, D, F), jnp.float32) * 0.02
    w1     = jax.random.normal(k3, (E, D, F), jnp.float32) * 0.02
    wout   = jax.random.normal(k4, (E, F, D), jnp.float32) * 0.02
    gate_logits = jax.random.normal(k5, (T, E), jnp.float32) * 0.1
    scores = jax.nn.sigmoid(gate_logits)
    fw_raw, fi = lax.top_k(scores, K)
    fw = fw_raw / fw_raw.sum(-1, keepdims=True)

    act_spec = P(("ep", "fsdp"), None)

    def shard(x, spec):
        return jax.device_put(x, NamedSharding(mesh, spec))

    fx_s    = shard(tokens.astype(jnp.bfloat16), act_spec)
    fi_s    = shard(fi,   act_spec)
    fw_s    = shard(fw,   act_spec)
    w0_s    = shard(w0,   P("ep", None, "fsdp"))
    w1_s    = shard(w1,   P("ep", None, "fsdp"))
    wout_s  = shard(wout, P("ep", "fsdp", None))

    # ---- Reference: jax.vjp on forward shard_map ----
    @functools.partial(shard_map, mesh=mesh,
                       in_specs=(act_spec, act_spec, act_spec,
                                 P("ep", None, "fsdp"), P("ep", None, "fsdp"), P("ep", "fsdp", None)),
                       out_specs=act_spec, check_rep=False)
    def fwd_ref(fx_l, fi_l, fw_l, w0_l, w1_l, wout_l):
        return moe_fwd_ep(fx_l, fi_l, fw_l, w0_l, w1_l, wout_l, K, "ep", max_tpe)

    def loss_ref(fx, fi, fw, w0, w1, wout):
        return jnp.sum(fwd_ref(fx, fi, fw, w0, w1, wout).astype(jnp.float32))

    _, (d_w0_ref, d_w1_ref, d_wout_ref) = jax.value_and_grad(
        loss_ref, argnums=(3, 4, 5))(fx_s, fi_s, fw_s, w0_s, w1_s, wout_s)

    # ---- Streaming: call streaming kernel inside shard_map ----
    def stream_bwd_fn(g_l, fx_l, fw_l, fi_l, w0_l, w1_l, wout_l):
        g_full  = lax.all_gather(g_l,  "ep", axis=0, tiled=True)
        fx_full = lax.all_gather(fx_l, "ep", axis=0, tiled=True)
        fw_full = lax.all_gather(fw_l, "ep", axis=0, tiled=True)
        fi_full = lax.all_gather(fi_l, "ep", axis=0, tiled=True)

        w1_stk = jnp.stack([w0_l, w1_l], axis=1)  # (E_local, 2, D, F_local)

        d_tok_partial, d_w1_p, d_wo_p, d_topk_partial = fused_ep_moe_bwd_streaming(
            g_full, fx_full.astype(jnp.float32), w1_stk, wout_l,
            gating_output=None, top_k=K,
            scoring_fn="sigmoid", renormalize_topk_logits=True, act_fn="silu",
            ep_axis_name="ep", max_tpe=max_tpe,
            top_k_indices_precomputed=fi_full,
            top_k_weights_precomputed=fw_full.astype(jnp.float32),
            return_dtopk=True,
            E_global_override=E,
        )
        return d_w1_p[:, 0], d_w1_p[:, 1], d_wo_p

    g_ones = jax.device_put(jnp.ones((T, D), jnp.float32), NamedSharding(mesh, act_spec))

    d_w0_stream, d_w1_stream, d_wout_stream = shard_map(
        stream_bwd_fn, mesh=mesh,
        in_specs=(act_spec, act_spec, act_spec, act_spec,
                  P("ep", None, "fsdp"), P("ep", None, "fsdp"), P("ep", "fsdp", None)),
        out_specs=(P("ep", None, "fsdp"), P("ep", None, "fsdp"), P("ep", "fsdp", None)),
        check_rep=False,
    )(g_ones, fx_s, fw_s, fi_s, w0_s, w1_s, wout_s)

    for name, (g_s, g_r) in zip(["d_w0", "d_w1", "d_wout"],
                                  [(d_w0_stream, d_w0_ref),
                                   (d_w1_stream, d_w1_ref),
                                   (d_wout_stream, d_wout_ref)]):
        ns = float(jnp.linalg.norm(g_s.astype(jnp.float32)))
        nr = float(jnp.linalg.norm(g_r.astype(jnp.float32)))
        ratio = ns / (nr + 1e-12)
        print(f"  {name}: stream_norm={ns:.4f} ref_norm={nr:.4f} ratio={ratio:.4f}")


import functools

if __name__ == "__main__":
    print(f"JAX devices: {jax.local_devices()}")
    run_test(EP=1, FSDP=1)
    run_test(EP=2, FSDP=1)
    run_test(EP=1, FSDP=2)
    run_test(EP=2, FSDP=2)
