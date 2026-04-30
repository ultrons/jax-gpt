"""NaN debug: exercise _moe_jax_ep_fn with use_streaming_bwd=True at EP=2, FSDP=2.

Runs on the 4 local TPU devices: mesh (ep=2, fsdp=2).
Compares streaming vs jax.vjp backward to identify the NaN source.
Adds debug prints inside _streaming_bwd_fn to binary-search the NaN.

Run: /home/sivaibhav_google_com/xdb/.xprof/bin/python3 test_nan_debug.py
"""

import sys, types, logging, functools
import numpy as np

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

import jax
import jax.numpy as jnp
from jax import lax
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from jax.experimental.shard_map import shard_map

# ---- Patch _moe_jax_ep_fn_bwd with debug prints ----
# We monkey-patch _streaming_bwd_fn to add NaN checks at key boundaries.

import model as _model_mod
import backward_kernel as _bk_mod

_orig_bwd = _model_mod._moe_jax_ep_fn_bwd.__wrapped__ if hasattr(
    _model_mod._moe_jax_ep_fn_bwd, '__wrapped__') else None

# Instead, patch by replacing the module-level function body with a debug version.
# We do this by importing and wrapping fused_ep_moe_bwd_streaming.

_orig_streaming = _bk_mod.fused_ep_moe_bwd_streaming

def _debug_streaming(d_out, tokens, w1, w2, gating_output, top_k, **kwargs):
    """Wrapper with NaN debug prints at per-expert granularity."""
    # Check inputs
    jax.debug.print("  [bwd-in] d_out nan:{} max:{:.3e}",
        jnp.any(jnp.isnan(d_out)), jnp.max(jnp.abs(d_out)))
    jax.debug.print("  [bwd-in] tokens nan:{} max:{:.3e}",
        jnp.any(jnp.isnan(tokens)), jnp.max(jnp.abs(tokens)))
    jax.debug.print("  [bwd-in] w1 nan:{} max:{:.3e}",
        jnp.any(jnp.isnan(w1)), jnp.max(jnp.abs(w1)))
    jax.debug.print("  [bwd-in] w2 nan:{} max:{:.3e}",
        jnp.any(jnp.isnan(w2)), jnp.max(jnp.abs(w2)))
    if kwargs.get("top_k_weights_precomputed") is not None:
        fw = kwargs["top_k_weights_precomputed"]
        jax.debug.print("  [bwd-in] fw_precomp nan:{} max:{:.3e} min:{:.3e}",
            jnp.any(jnp.isnan(fw)), jnp.max(jnp.abs(fw)),
            jnp.min(fw.astype(jnp.float32)))

    result = _orig_streaming(d_out, tokens, w1, w2, gating_output, top_k, **kwargs)

    jax.debug.print("  [bwd-out] d_tokens nan:{} max:{:.3e}",
        jnp.any(jnp.isnan(result[0])), jnp.max(jnp.abs(result[0])))
    jax.debug.print("  [bwd-out] d_w1 nan:{} max:{:.3e}",
        jnp.any(jnp.isnan(result[1])), jnp.max(jnp.abs(result[1])))
    jax.debug.print("  [bwd-out] d_w2 nan:{} max:{:.3e}",
        jnp.any(jnp.isnan(result[2])), jnp.max(jnp.abs(result[2])))
    jax.debug.print("  [bwd-out] d_topk/dgat nan:{} max:{:.3e}",
        jnp.any(jnp.isnan(result[3])), jnp.max(jnp.abs(result[3])))
    return result


# Patch the backward kernel import in model.py
import model
model.fused_ep_moe_bwd_streaming_debug = _debug_streaming


# ---- Build test: EP=2, FSDP=2, small dims ----
# E=8, EP=2 → E_local=4
# T=32 tokens, D=64, F=32, K=4 routing
# mesh: (ep=2, fsdp=2), 4 local devices

def make_mesh_ep2_fsdp2():
    devs = np.array(jax.local_devices()).reshape(2, 2)
    return Mesh(devs, ("ep", "fsdp"))


def run_nan_test():
    mesh = make_mesh_ep2_fsdp2()
    print(f"Mesh: {mesh}")

    T, D, F, E, K = 32, 64, 32, 8, 4
    EP, FSDP = 2, 2
    E_local = E // EP
    F_local = F // FSDP

    key = jax.random.PRNGKey(42)
    k1, k2, k3, k4 = jax.random.split(key, 4)

    # Token activations: bf16 to mimic training
    tokens_f32 = jax.random.normal(k1, (T, D), jnp.float32)

    # Global weights (will be sharded by shard_map)
    w0 = jax.random.normal(k2, (E, D, F), jnp.float32) * 0.02   # gate proj (E, D, F)
    w1 = jax.random.normal(k3, (E, D, F), jnp.float32) * 0.02   # up proj   (E, D, F)
    wout = jax.random.normal(k4, (E, F, D), jnp.float32) * 0.02  # down proj (E, F, D)

    # Routing: sigmoid + normalize, K=4 top experts per token
    gate_key, = jax.random.split(key, 1),
    gate_logits = jax.random.normal(gate_key[0], (T, E), jnp.float32) * 0.1
    scores = jax.nn.sigmoid(gate_logits)
    _, fi = jax.lax.top_k(scores, K)                          # (T, K) int32 global IDs
    fw_raw = jnp.take_along_axis(scores, fi, axis=-1)          # (T, K) raw scores
    fw = fw_raw / fw_raw.sum(-1, keepdims=True)                 # normalized (T, K)
    fw = (fw * 2.5).astype(jnp.bfloat16)                       # bf16, scaled like DSv3

    act_spec = P(("ep", "fsdp"), None)
    max_tpe = max(1, 2 * (T // FSDP) * K // E)
    print(f"T={T} D={D} F={F} E={E} K={K} EP={EP} FSDP={FSDP} max_tpe={max_tpe}")

    # Shard tensors onto the mesh
    def shard(x, spec):
        return jax.device_put(x, NamedSharding(mesh, spec))

    fx_s   = shard(tokens_f32.astype(jnp.bfloat16), act_spec)
    fi_s   = shard(fi, act_spec)
    fw_s   = shard(fw, act_spec)
    w0_s   = shard(w0, P("ep", None, "fsdp"))
    w1_s   = shard(w1, P("ep", None, "fsdp"))
    wout_s = shard(wout, P("ep", "fsdp", None))

    # ----- Reference: use_streaming_bwd=False -----
    print("\n--- Reference (use_streaming_bwd=False) ---")
    def loss_fn_ref(fx, fi, fw, w0, w1, wout):
        out = model._moe_jax_ep_fn(fx, fi, fw, w0, w1, wout,
                                    mesh, K, act_spec, "ep", max_tpe,
                                    use_streaming_bwd=False)
        return jnp.sum(out.astype(jnp.float32))

    ref_loss, ref_grads = jax.value_and_grad(loss_fn_ref,
                                              argnums=(0, 2, 3, 4, 5))(
        fx_s, fi_s, fw_s, w0_s, w1_s, wout_s)
    print(f"  ref loss={float(ref_loss):.4f}")
    for name, g in zip(["d_fx","d_fw","d_w0","d_w1","d_wout"], ref_grads):
        nan = bool(jnp.any(jnp.isnan(g)))
        print(f"  ref {name}: nan={nan}  norm={float(jnp.linalg.norm(g.astype(jnp.float32))):.4f}")

    # ----- Streaming: use_streaming_bwd=True -----
    print("\n--- Streaming (use_streaming_bwd=True) ---")

    # Monkey-patch _streaming_bwd_fn to use debug version
    # We patch directly in the module's _moe_jax_ep_fn_bwd closure
    _orig_import = None

    def loss_fn_stream(fx, fi, fw, w0, w1, wout):
        out = model._moe_jax_ep_fn(fx, fi, fw, w0, w1, wout,
                                    mesh, K, act_spec, "ep", max_tpe,
                                    use_streaming_bwd=True)
        return jnp.sum(out.astype(jnp.float32))

    # First do forward-only to check for NaN there
    fwd_out = model._moe_jax_ep_fn(fx_s, fi_s, fw_s, w0_s, w1_s, wout_s,
                                    mesh, K, act_spec, "ep", max_tpe,
                                    use_streaming_bwd=True)
    fwd_nan = bool(jnp.any(jnp.isnan(fwd_out)))
    print(f"  forward output: nan={fwd_nan}  norm={float(jnp.linalg.norm(fwd_out.astype(jnp.float32))):.4f}")

    stream_loss, stream_grads = jax.value_and_grad(loss_fn_stream,
                                                    argnums=(0, 2, 3, 4, 5))(
        fx_s, fi_s, fw_s, w0_s, w1_s, wout_s)
    print(f"  stream loss={float(stream_loss):.4f}")
    for name, (sg, rg) in zip(["d_fx","d_fw","d_w0","d_w1","d_wout"],
                               zip(stream_grads, ref_grads)):
        sg_f = sg.astype(jnp.float32)
        rg_f = rg.astype(jnp.float32)
        nan = bool(jnp.any(jnp.isnan(sg_f)))
        norm_s = float(jnp.linalg.norm(sg_f))
        norm_r = float(jnp.linalg.norm(rg_f))
        ratio = norm_s / (norm_r + 1e-12)
        print(f"  stream {name}: nan={nan}  norm={norm_s:.4f}  ref={norm_r:.4f}  ratio={ratio:.3f}")


if __name__ == "__main__":
    print(f"JAX devices: {jax.local_devices()}")
    run_nan_test()
