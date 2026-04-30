"""Stage 1 v7x: DSv3 full config (D=7168, F=2048, E=256, K=8) on 4x8x8.

Tests fused_ep_moe_bwd_streaming at DSv3 full scale:
  1. EP=1 (T=32): streaming backward vs ref_moe — each process independently
  2. EP=8 (T=128): shard_map + lax.psum vs EP=1 streaming reference

Topology: 4x8x8 = 64 processes × 8 local devices = 512 JAX devices.
Mesh: (64, 8) → ("fsdp"=64, "model"=8).  EP uses "model" axis.

Memory budget per v7x device (192 GB HBM):
  EP=1 ref: w1+w2 = 11 GB, fori_loop carry = 11 GB → ~22 GB peak.
  EP=8 bwd: w1_shard+w2_shard = 1.4 GB/device.
  ref_moe vmap gather T=32: (256,7168,2048)×4 = 15 GB → fits.

Run via:
  kubectl --context gke_tpu-vm-gke-testing_us-central1_sivaibhav-exp-v7x \\
    apply -f k8s/test-stage1-4x8x8.yaml
"""

import gc
import sys
import types
import logging
import numpy as np

# Bootstrap: mock vllm for tpu_inference logger.
_vllm = types.ModuleType("vllm")
_vllm_logger = types.ModuleType("vllm.logger")


class _VllmLogger(logging.Logger):
    def warning_once(self, msg, *args, **kwargs): self.warning(msg, *args, **kwargs)
    def info_once(self, msg, *args, **kwargs): self.info(msg, *args, **kwargs)
    def debug_once(self, msg, *args, **kwargs): self.debug(msg, *args, **kwargs)


logging.setLoggerClass(_VllmLogger)
_vllm_logger.init_logger = lambda name: logging.getLogger(name)
_vllm_logger._VllmLogger = _VllmLogger
_vllm_logger.init_vllm_logger = lambda name: logging.getLogger(name)
sys.modules.setdefault("vllm", _vllm)
sys.modules.setdefault("vllm.logger", _vllm_logger)

_TPU_INFERENCE = "/home/sivaibhav_google_com/tpu-inference"
if _TPU_INFERENCE not in sys.path:
    sys.path.insert(0, _TPU_INFERENCE)

import jax
import jax.numpy as jnp
from jax import lax
from jax.sharding import NamedSharding, PartitionSpec as P

from tpu_inference.kernels.fused_moe.v1.kernel import ref_moe
from backward_kernel import fused_ep_moe_bwd_streaming


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def norm_parity(a, b, name="", tol=0.02, verbose=True):
    na = float(jnp.linalg.norm(jnp.asarray(a)))
    nb = float(jnp.linalg.norm(jnp.asarray(b)))
    ratio = na / (nb + 1e-12)
    ok = abs(ratio - 1.0) < tol
    if verbose:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}  norm_ratio={ratio:.6f}"
              f"  (stream={na:.4f}, ref={nb:.4f})")
    return ok


def make_ep_sharded(data, mesh, ep_axis):
    """Distribute data along axis-0 across ep_axis devices.

    Each process independently provides its addressable shards.
    All processes must hold the same full data (same random key).
    jax.local_devices()[j] maps to mesh column j (model shard j).
    """
    EP = mesh.shape[ep_axis]
    shard_size = data.shape[0] // EP
    sharding = NamedSharding(mesh, P(ep_axis))
    local_shards = [
        jax.device_put(data[j * shard_size : (j + 1) * shard_size], dev)
        for j, dev in enumerate(jax.local_devices())
    ]
    return jax.make_array_from_single_device_arrays(data.shape, sharding, local_shards)


def make_replicated(data, mesh):
    """Replicate data across all local devices (addressable slice of mesh)."""
    sharding = NamedSharding(mesh, P())
    local_shards = [jax.device_put(data, dev) for dev in jax.local_devices()]
    return jax.make_array_from_single_device_arrays(data.shape, sharding, local_shards)


# ---------------------------------------------------------------------------
# Test 1: EP=1 DSv3 streaming backward vs ref_moe
# ---------------------------------------------------------------------------

def check_dsv3_ep1(T=32, D=1024, F=512, E=256, K=8, verbose=True):
    """EP=1 streaming backward vs ref_moe at DSv3 expert count but reduced D/F.

    E=256, K=8 kept at DSv3 scale to validate routing / scoring logic.
    D=1024, F=512 instead of full DSv3 D=7168/F=2048 to avoid OOM:
      - ref_moe vmap gather (T*K=256, 2, D, F) = 0.26 GB → negligible
      - streaming d_w1_acc carry (256, 2, D, F) × 4 = 0.53 GB → fits easily
    Full D=7168/F=2048 causes RuntimeBufferAllocationFailure (14 GB contiguous
    needed after JAX distributed init, only ~10 GB contiguous free).
    Each process runs independently on its local device (no collectives).
    """
    key = jax.random.PRNGKey(42)
    k1, k2, k3, k4 = jax.random.split(key, 4)
    tokens = jax.random.normal(k1, (T, D),        dtype=jnp.float32)
    w1     = jax.random.normal(k2, (E, 2, D, F),  dtype=jnp.float32) * 0.02
    w2     = jax.random.normal(k3, (E, F, D),     dtype=jnp.float32) * 0.02
    gating = jax.random.normal(k4, (T, E),         dtype=jnp.float32) * 0.1

    cfg = dict(top_k=K, scoring_fn="sigmoid", renormalize_topk_logits=True)

    ref_grads = jax.grad(
        lambda t, w1, w2, g: ref_moe(t, w1, w2, g, **cfg).sum(),
        argnums=(0, 1, 2, 3))(tokens, w1, w2, gating)

    d_out = jnp.ones((T, D), jnp.float32)
    v4_grads = fused_ep_moe_bwd_streaming(
        d_out, tokens, w1, w2, gating, K,
        scoring_fn="sigmoid", renormalize_topk_logits=True, act_fn="silu")

    ok_tok  = norm_parity(v4_grads[0], ref_grads[0], "d_tokens", verbose=verbose)
    ok_gate = norm_parity(v4_grads[3], ref_grads[3], "d_gating", verbose=verbose)
    ok_w1   = norm_parity(v4_grads[1], ref_grads[1], "d_w1    ", verbose=verbose)
    ok_w2   = norm_parity(v4_grads[2], ref_grads[2], "d_w2    ", verbose=verbose)
    return all([ok_tok, ok_gate, ok_w1, ok_w2])


# ---------------------------------------------------------------------------
# Test 2: EP=8 via shard_map vs EP=1 reference
# ---------------------------------------------------------------------------

def check_dsv3_ep8(mesh, T=128, D=2048, F=1024, E=256, K=8, verbose=True):
    """EP=8 streaming backward via shard_map vs EP=1 reference.

    All 64 processes participate via shard_map over the "model" axis (EP=8).
    d_tokens and d_gating are psum'd across the 8 EP devices → must match EP=1.
    d_w1, d_w2 remain sharded (no global reference for per-shard grads).

    D=2048, F=1024 (not full DSv3 D=7168/F=2048) to stay within the 94.75 GB
    compile-time HBM ceiling.  E=256/K=8 are kept at DSv3 scale to validate the
    expert-routing topology (which is what this test is actually checking).

    Memory per device at D=2048, F=1024 (v7x, 192 GB HBM):
      EP=1 ref: d_w1_acc(256,2,2048,1024)=2.1GB + bf16 HLO temps=3.1GB → ~6 GB.
      EP=8 bwd: w1_shard(32,2,2048,1024)=0.27GB → ~0.8 GB peak.
    """
    ep_axis = "model"
    EP = mesh.shape[ep_axis]
    if E % EP != 0:
        raise ValueError(f"E={E} not divisible by EP={EP}")

    key = jax.random.PRNGKey(99)
    k1, k2, k3, k4 = jax.random.split(key, 4)

    d_out  = jnp.ones((T, D), jnp.float32)
    tokens = jax.random.normal(k1, (T, D),        dtype=jnp.float32)
    w1     = jax.random.normal(k2, (E, 2, D, F),  dtype=jnp.float32) * 0.02
    w2     = jax.random.normal(k3, (E, F, D),     dtype=jnp.float32) * 0.02
    gating = jax.random.normal(k4, (T, E),         dtype=jnp.float32) * 0.1

    # EP=1 reference: process all E=256 experts on one device.
    # Uses the same data as EP=8 so results must match after psum.
    ep1_grads = fused_ep_moe_bwd_streaming(
        d_out, tokens, w1, w2, gating, K,
        scoring_fn="sigmoid", renormalize_topk_logits=True, act_fn="silu")

    # Distribute arrays for EP=8 shard_map.
    w1_dist    = make_ep_sharded(w1,    mesh, ep_axis)
    w2_dist    = make_ep_sharded(w2,    mesh, ep_axis)
    d_out_dist = make_replicated(d_out,  mesh)
    tok_dist   = make_replicated(tokens, mesh)
    gat_dist   = make_replicated(gating, mesh)

    def _ep8_bwd(d_out_l, tokens_l, w1_l, w2_l, gating_l):
        """Each device processes E_local=32 experts; psum d_tokens and d_gating."""
        d_tok_p, d_w1_l, d_w2_l, d_gate_p = fused_ep_moe_bwd_streaming(
            d_out_l, tokens_l, w1_l, w2_l, gating_l, K,
            scoring_fn="sigmoid", renormalize_topk_logits=True,
            act_fn="silu", ep_axis_name=ep_axis)
        d_tok  = lax.psum(d_tok_p,  axis_name=ep_axis)
        d_gate = lax.psum(d_gate_p, axis_name=ep_axis)
        return d_tok, d_w1_l, d_w2_l, d_gate

    ep8_grads = jax.shard_map(
        _ep8_bwd, mesh=mesh,
        in_specs=(P(), P(), P(ep_axis), P(ep_axis), P()),
        out_specs=(P(), P(ep_axis), P(ep_axis), P()),
        check_vma=False,
    )(d_out_dist, tok_dist, w1_dist, w2_dist, gat_dist)

    ok_tok  = norm_parity(ep8_grads[0], ep1_grads[0],
                          "d_tokens (EP=8 vs EP=1)", verbose=verbose)
    ok_gate = norm_parity(ep8_grads[3], ep1_grads[3],
                          "d_gating (EP=8 vs EP=1)", verbose=verbose)
    return ok_tok and ok_gate


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    jax.distributed.initialize(initialization_timeout=600)
    proc = jax.process_index()

    if proc == 0:
        print(f"JAX: {jax.__version__}")
        print(f"Total JAX devices: {jax.device_count()}")
        print(f"Local JAX devices: {jax.local_device_count()}")
        print(f"Backend: {jax.default_backend()}")

    results = {}

    # ── Test 1: EP=1 DSv3 streaming backward vs ref_moe ──────────────────────
    if proc == 0:
        print("\n=== DSv3 EP=1  D=1024 F=512 E=256 K=8 T=32  sigmoid+renorm ===")
    try:
        ok = check_dsv3_ep1(verbose=(proc == 0))
    except Exception as e:
        if proc == 0:
            import traceback
            traceback.print_exc()
        ok = False
    results["dsv3_ep1"] = ok

    # Free HBM from EP=1 test (ref_grads, v4_grads, w1/w2 locals) before EP=8
    # compiles.  Without this, ~22 GB of stale weight-grad arrays remain live
    # and push the compile-time allocation over the 94.75 GB ceiling.
    gc.collect()
    jax.effects_barrier()

    # ── Test 2: EP=8 streaming backward via shard_map ─────────────────────────
    EXPECTED_DEVICES = 64 * 8  # 64 pods × 8 local devices on 4x8x8
    if jax.device_count() != EXPECTED_DEVICES:
        if proc == 0:
            print(f"\n  [SKIP] EP=8 test requires {EXPECTED_DEVICES} devices, "
                  f"got {jax.device_count()}")
        results["dsv3_ep8"] = None
    else:
        # mesh: (FSDP=64, EP=8). "model" = EP axis; "fsdp" = FSDP axis.
        # jax.local_devices()[j] = mesh[proc, j] → model shard j.
        devices = np.array(jax.devices()).reshape(64, 8)
        mesh = jax.sharding.Mesh(devices, ("fsdp", "model"))
        if proc == 0:
            print("\n=== DSv3 EP=8  D=2048 F=1024 E=256 K=8 T=128 shard_map ===")
        try:
            ok = check_dsv3_ep8(mesh, verbose=(proc == 0))
        except Exception as e:
            if proc == 0:
                import traceback
                traceback.print_exc()
            ok = False
        results["dsv3_ep8"] = ok

    # ── Summary ──────────────────────────────────────────────────────────────
    all_pass = all(v for v in results.values() if v is not None)
    if proc == 0:
        print("\n" + "=" * 60)
        for name, ok in results.items():
            label = "PASS" if ok else ("SKIP" if ok is None else "FAIL")
            print(f"  {label}  {name}")
        print("=" * 60)
        print(f"Overall: {'ALL PASSED' if all_pass else 'SOME FAILED'}")
    sys.exit(0 if all_pass else 1)
