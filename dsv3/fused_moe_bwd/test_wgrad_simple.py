"""Minimal: test weight gradient with EP=2, FSDP=1 to isolate FSDP vs EP issue.

Calls streaming kernel directly via shard_map with:
- EP=2: each device gets E_local experts
- No FSDP sharding (full F features per device)
- Compares with EP=1 reference (full computation)
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


def test_ep2_no_fsdp():
    """EP=2, FSDP=1: streaming kernel directly via shard_map.

    Compare with EP=1 reference (full computation on all experts).
    """
    print("\n=== EP=2, FSDP=1 (no FSDP sharding) ===")
    # Use only 2 devices (EP=2, FSDP=1)
    devs_ep2 = np.array(jax.local_devices()[:2]).reshape(2, 1)
    mesh_ep2 = Mesh(devs_ep2, ("ep", "fsdp"))

    T, D, F, E, K = 32, 64, 32, 8, 4
    EP = 2
    E_local = E // EP
    max_tpe = 2 * T * K // E  # 2 * 32 * 4 / 8 = 32

    key = jax.random.PRNGKey(42)
    k1, k2, k3, k4, k5 = jax.random.split(key, 5)
    tokens = jax.random.normal(k1, (T, D), jnp.float32) * 0.1
    # w1 shape: (E, 2, D, F) for streaming kernel
    w1 = jax.random.normal(k2, (E, 2, D, F), jnp.float32) * 0.02
    w2 = jax.random.normal(k3, (E, F, D), jnp.float32) * 0.02
    gate = jax.random.normal(k4, (T, E), jnp.float32) * 0.1
    d_out = jax.random.normal(k5, (T, D), jnp.float32)

    # EP=1 reference: call streaming kernel directly (all experts, all tokens)
    ref_grads = fused_ep_moe_bwd_streaming(
        d_out, tokens, w1, w2, gate, K,
        scoring_fn="sigmoid", renormalize_topk_logits=True, act_fn="silu")
    d_w1_ref, d_w2_ref = ref_grads[1], ref_grads[2]

    # EP=2: distribute experts across 2 devices, replicate tokens + d_out
    def shard(x, spec):
        return jax.device_put(x, NamedSharding(mesh_ep2, spec))

    d_out_s = shard(d_out, P())  # replicated
    tok_s   = shard(tokens, P())  # replicated
    gate_s  = shard(gate, P())   # replicated
    w1_s    = shard(w1, P("ep"))  # EP-sharded on axis 0 (E axis)
    w2_s    = shard(w2, P("ep"))  # EP-sharded

    def ep2_bwd(d_out_l, tok_l, w1_l, w2_l, gate_l):
        # All tokens replicated; only experts are EP-sharded.
        # d_out_l = (T, D), tok_l = (T, D) — full (replicated)
        # w1_l = (E_local, 2, D, F), w2_l = (E_local, F, D)
        d_tok, d_w1_l, d_w2_l, _ = fused_ep_moe_bwd_streaming(
            d_out_l, tok_l, w1_l, w2_l, gate_l, K,
            scoring_fn="sigmoid", renormalize_topk_logits=True, act_fn="silu",
            ep_axis_name="ep")
        d_tok_sum = lax.psum(d_tok, "ep")
        return d_tok_sum, d_w1_l, d_w2_l

    ep2_grads = shard_map(
        ep2_bwd, mesh=mesh_ep2,
        in_specs=(P(), P(), P("ep"), P("ep"), P()),
        out_specs=(P(), P("ep"), P("ep")),
        check_rep=False,
    )(d_out_s, tok_s, w1_s, w2_s, gate_s)

    d_tok_ep2, d_w1_ep2, d_w2_ep2 = ep2_grads
    d_tok_ref = ref_grads[0]

    for name, (g_s, g_r) in zip(["d_tok", "d_w1", "d_w2"],
                                  [(d_tok_ep2, d_tok_ref),
                                   (d_w1_ep2, d_w1_ref),
                                   (d_w2_ep2, d_w2_ref)]):
        ns = float(jnp.linalg.norm(g_s.astype(jnp.float32)))
        nr = float(jnp.linalg.norm(g_r.astype(jnp.float32)))
        ratio = ns / (nr + 1e-12)
        print(f"  {name}: ep2={ns:.4f} ref={nr:.4f} ratio={ratio:.4f}")


def test_ep2_fsdp2():
    """EP=2, FSDP=2: streaming kernel with FSDP-sharded weights.

    Tests specifically whether FSDP sharding breaks d_w.
    """
    print("\n=== EP=2, FSDP=2 ===")
    devs = np.array(jax.local_devices()).reshape(2, 2)
    mesh = Mesh(devs, ("ep", "fsdp"))

    T, D, F, E, K = 32, 64, 32, 8, 4
    EP, FSDP = 2, 2
    E_local = E // EP
    F_local = F // FSDP
    max_tpe = max(1, 2 * (T // FSDP) * K // E)

    key = jax.random.PRNGKey(42)
    k1, k2, k3, k4, k5 = jax.random.split(key, 5)
    # Use F_local split weights for FSDP
    tokens = jax.random.normal(k1, (T, D), jnp.float32) * 0.1
    w1 = jax.random.normal(k2, (E, 2, D, F), jnp.float32) * 0.02
    w2 = jax.random.normal(k3, (E, F, D), jnp.float32) * 0.02
    gate = jax.random.normal(k4, (T, E), jnp.float32) * 0.1
    d_out = jax.random.normal(k5, (T, D), jnp.float32)

    # EP=1, FSDP=1 reference
    ref_grads = fused_ep_moe_bwd_streaming(
        d_out, tokens, w1, w2, gate, K,
        scoring_fn="sigmoid", renormalize_topk_logits=True, act_fn="silu")
    d_w1_ref_full, d_w2_ref_full = ref_grads[1], ref_grads[2]

    def shard(x, spec):
        return jax.device_put(x, NamedSharding(mesh, spec))

    act_spec = P(("ep", "fsdp"), None)

    # For the w1 (E, 2, D, F) tensor: EP shards E, FSDP shards F
    # Shape: P("ep", None, None, "fsdp") for (E, 2, D, F)
    # But shard_map needs in_specs for the local tensor shape.
    # The inner in_spec for (E, 2, D, F) = P("ep", None, None, "fsdp")
    # Similarly for w2 (E, F, D) = P("ep", "fsdp", None)

    d_out_s = shard(d_out, act_spec)  # (T, D) sharded P(("ep","fsdp"),None)
    tok_s   = shard(tokens, act_spec)
    gate_s  = shard(gate, act_spec)
    w1_s    = shard(w1, P("ep", None, None, "fsdp"))  # (E, 2, D, F)
    w2_s    = shard(w2, P("ep", "fsdp", None))         # (E, F, D)

    def ep2fsdp2_bwd(d_out_l, tok_l, w1_l, w2_l, gate_l):
        # VJP of forward psum("fsdp"): psum the incoming gradient.
        d_out_l_sum = lax.psum(d_out_l, "fsdp")  # (T/(EP*FSDP), D)
        # EP all_gather for tokens
        d_out_full = lax.all_gather(d_out_l_sum, "ep", axis=0, tiled=True)  # (T/FSDP, D)
        tok_full   = lax.all_gather(tok_l,       "ep", axis=0, tiled=True)  # (T/FSDP, D)
        gate_full  = lax.all_gather(gate_l,      "ep", axis=0, tiled=True)  # (T/FSDP, E)

        # w1_l: (E_local, 2, D, F_local), w2_l: (E_local, F_local, D)
        d_tok_p, d_w1_l, d_w2_l, _ = fused_ep_moe_bwd_streaming(
            d_out_full, tok_full, w1_l, w2_l, gate_full, K,
            scoring_fn="sigmoid", renormalize_topk_logits=True, act_fn="silu",
            ep_axis_name="ep", max_tpe=max_tpe,
            E_global_override=E)

        # No psum("fsdp") on d_tok: FSDP factor already incorporated via d_out_l_sum.
        d_tok_full = lax.psum(d_tok_p, "ep")
        T_ep = d_out_l.shape[0]
        dev_ep = lax.axis_index("ep")
        d_tok_l = lax.dynamic_slice(d_tok_full, (dev_ep * T_ep, 0), (T_ep, D))
        return d_tok_l, d_w1_l, d_w2_l

    ep2fsdp2_grads = shard_map(
        ep2fsdp2_bwd, mesh=mesh,
        in_specs=(act_spec, act_spec,
                  P("ep", None, None, "fsdp"), P("ep", "fsdp", None), act_spec),
        out_specs=(act_spec, P("ep", None, None, "fsdp"), P("ep", "fsdp", None)),
        check_rep=False,
    )(d_out_s, tok_s, w1_s, w2_s, gate_s)

    d_tok_ep2fsdp2, d_w1_ep2fsdp2, d_w2_ep2fsdp2 = ep2fsdp2_grads

    for name, (g_s, g_r) in zip(["d_tok", "d_w1", "d_w2"],
                                  [(d_tok_ep2fsdp2, ref_grads[0]),
                                   (d_w1_ep2fsdp2, d_w1_ref_full),
                                   (d_w2_ep2fsdp2, d_w2_ref_full)]):
        ns = float(jnp.linalg.norm(g_s.astype(jnp.float32)))
        nr = float(jnp.linalg.norm(g_r.astype(jnp.float32)))
        ratio = ns / (nr + 1e-12)
        print(f"  {name}: ep2fsdp2={ns:.4f} ref={nr:.4f} ratio={ratio:.4f}")

    # Also compute sum (not norm) for element-wise comparison
    print("  d_w1 sum: streaming={:.4f} ref={:.4f}".format(
        float(jnp.sum(d_w1_ep2fsdp2)), float(jnp.sum(d_w1_ref_full))))


if __name__ == "__main__":
    print(f"JAX devices: {jax.local_devices()}")
    test_ep2_no_fsdp()
    test_ep2_fsdp2()
