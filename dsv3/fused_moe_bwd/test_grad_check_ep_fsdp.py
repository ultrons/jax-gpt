"""Gradient check for make_fused_ep_moe_train_v4 with EP>1 and FSDP>1.

This is the UNTESTED path: backward.py _local_bwd_fsdp (lines 637-666).
It exercises the full custom_vjp including:
  psum(d_out, fsdp) → all_gather(d_out_sum, ep) → streaming_v1(F-sharded weights)
  → psum(d_tok, ep) → dynamic_slice

Run locally on a single host with ≥8 JAX devices (EP=2, FSDP=4):
  python test_grad_check_ep_fsdp.py

Run on cluster (requires jax.distributed):
  python test_grad_check_ep_fsdp.py --distributed
"""

import sys
import os
import types
import logging
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

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
from jax.sharding import NamedSharding, PartitionSpec as P

from tpu_inference.kernels.fused_moe.v1.kernel import ref_moe
from backward import make_fused_ep_moe_train_v4


def norm_parity(a, b, name="", tol=0.05):
    na = float(jnp.linalg.norm(jnp.asarray(a).reshape(-1)))
    nb = float(jnp.linalg.norm(jnp.asarray(b).reshape(-1)))
    ratio = na / (nb + 1e-12)
    ok = abs(ratio - 1.0) < tol
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}  norm_ratio={ratio:.6f}"
          f"  (v4={na:.4f}, ref={nb:.4f})")
    return ok


def make_sharded(data, mesh, spec):
    sharding = NamedSharding(mesh, spec)
    local_shards = []
    for dev in jax.local_devices():
        device_indices = sharding.addressable_devices_indices_map(data.shape)[dev]
        shard_data = data[device_indices]
        local_shards.append(jax.device_put(shard_data, dev))
    return jax.make_array_from_single_device_arrays(data.shape, sharding, local_shards)


def check_ep_fsdp(mesh, EP, FSDP, T=64, D=512, F=256, E=16, K=4):
    """Compare make_fused_ep_moe_train_v4 (EP>1, FSDP>1) grads vs ref_moe.

    Exercises backward.py _local_bwd_fsdp end-to-end via jax.grad.
    Token sharding: P(("model", "fsdp"), None) — T split across EP*FSDP.
    Weight sharding: P("model", None, None, "fsdp") / P("model", "fsdp", None).
    """
    assert E % EP == 0, f"E={E} not divisible by EP={EP}"
    assert F % FSDP == 0, f"F={F} not divisible by FSDP={FSDP}"
    assert T % (EP * FSDP) == 0, f"T={T} not divisible by EP*FSDP={EP*FSDP}"

    key = jax.random.PRNGKey(42)
    k1, k2, k3, k4 = jax.random.split(key, 4)
    tokens = jax.random.normal(k1, (T, D),        dtype=jnp.float32)
    w1     = jax.random.normal(k2, (E, 2, D, F),  dtype=jnp.float32) * 0.02
    w2     = jax.random.normal(k3, (E, F, D),     dtype=jnp.float32) * 0.02
    gating = jax.random.normal(k4, (T, E),         dtype=jnp.float32) * 0.1

    cfg = dict(top_k=K, scoring_fn="sigmoid", renormalize_topk_logits=True)

    # Reference: full JAX autodiff on single device.
    ref_grads = jax.grad(
        lambda t, w1, w2, g: ref_moe(t, w1, w2, g, **cfg).sum(),
        argnums=(0, 1, 2, 3))(tokens, w1, w2, gating)

    # v4 with EP+FSDP: shard inputs to match training mesh layout.
    ep_axis   = "model"
    fsdp_axis = "fsdp"
    act_spec  = P((ep_axis, fsdp_axis), None)
    w1_spec   = P(ep_axis, None, None, fsdp_axis)
    w2_spec   = P(ep_axis, fsdp_axis, None)
    gate_spec = P((ep_axis, fsdp_axis), None)

    tok_dist  = make_sharded(np.array(tokens), mesh, act_spec)
    w1_dist   = make_sharded(np.array(w1),     mesh, w1_spec)
    w2_dist   = make_sharded(np.array(w2),     mesh, w2_spec)
    gat_dist  = make_sharded(np.array(gating), mesh, gate_spec)

    fn = make_fused_ep_moe_train_v4(
        mesh, top_k=K,
        scoring_fn="sigmoid",
        renormalize_topk_logits=True,
        act_fn="silu",
        ep_axis_name=ep_axis,
        fsdp_axis_name=fsdp_axis,
    )

    v4_grads = jax.grad(
        lambda t, w1, w2, g: fn(t, w1, w2, g).sum(),
        argnums=(0, 1, 2, 3))(tok_dist, w1_dist, w2_dist, gat_dist)

    ok_tok  = norm_parity(v4_grads[0], ref_grads[0], "d_tokens")
    ok_gate = norm_parity(v4_grads[3], ref_grads[3], "d_gating")
    ok_w1   = norm_parity(v4_grads[1], ref_grads[1], "d_w1    ")
    ok_w2   = norm_parity(v4_grads[2], ref_grads[2], "d_w2    ")
    return all([ok_tok, ok_gate, ok_w1, ok_w2])


if __name__ == "__main__":
    is_distributed = "--distributed" in sys.argv
    if is_distributed:
        jax.distributed.initialize(initialization_timeout=600)

    proc = jax.process_index()
    n_local = jax.local_device_count()
    n_total = jax.device_count()

    if proc == 0:
        print(f"JAX: {jax.__version__}")
        print(f"Total devices: {n_total}  Local: {n_local}")
        print(f"Backend: {jax.default_backend()}")

    results = {}

    # ── Local: EP=2, FSDP=N using all available local devices ───────────────
    # Exercises _local_bwd_fsdp path. Requires n_local divisible by 2 (EP=2).
    if n_local >= 4 and n_local % 2 == 0:
        EP   = 2
        FSDP = n_local // EP          # 2 for 4 devices, 4 for 8 devices, etc.
        local_devs = jax.local_devices()
        # mesh: (FSDP, EP) → local devices ordered as [fsdp_rank, ep_rank]
        mesh = jax.sharding.Mesh(
            np.array(local_devs).reshape(FSDP, EP), ("fsdp", "model"))
        if proc == 0:
            print(f"\n=== EP={EP} FSDP={FSDP} (local, {n_local} devices)  "
                  f"D=512 F=256 E=16 K=4 T=64 ===")
        try:
            ok = check_ep_fsdp(mesh, EP, FSDP)
        except Exception:
            if proc == 0:
                import traceback; traceback.print_exc()
            ok = False
        results[f"ep2_fsdp{FSDP}_local"] = ok
    else:
        if proc == 0:
            print(f"\n  [SKIP] local EP=2 test needs ≥4 even devices, got {n_local}")
        results["ep2_fsdp_local"] = None

    # ── EP=8, FSDP=16 on 4x4x4 (128 devices) ────────────────────────────────
    if n_total == 16 * 8:
        EP, FSDP = 8, 16
        devices = np.array(jax.devices()).reshape(FSDP, EP)
        mesh = jax.sharding.Mesh(devices, ("fsdp", "model"))
        if proc == 0:
            print(f"\n=== EP={EP} FSDP={FSDP} (4x4x4)  D=512 F=256 E=64 K=8 T=128 ===")
        try:
            ok = check_ep_fsdp(mesh, EP, FSDP, T=128, E=64, K=8)
        except Exception:
            if proc == 0:
                import traceback; traceback.print_exc()
            ok = False
        results["ep8_fsdp16_4x4x4"] = ok

    # ── EP=8, FSDP=64 on 4x8x8 (512 devices) ────────────────────────────────
    elif n_total == 64 * 8:
        EP, FSDP = 8, 64
        devices = np.array(jax.devices()).reshape(FSDP, EP)
        mesh = jax.sharding.Mesh(devices, ("fsdp", "model"))
        if proc == 0:
            print(f"\n=== EP={EP} FSDP={FSDP} (4x8x8)  D=512 F=256 E=256 K=8 T=512 ===")
        try:
            ok = check_ep_fsdp(mesh, EP, FSDP, T=512, E=256, K=8)
        except Exception:
            if proc == 0:
                import traceback; traceback.print_exc()
            ok = False
        results["ep8_fsdp64_4x8x8"] = ok

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
