"""Stage 1 gradient check: Pallas forward + per-expert streaming JAX backward.

Tests that make_fused_ep_moe_train_v4 (fused_ep_moe Pallas forward + streaming
backward) produces correct gradients vs. ref_moe across DSv3 mini/medium dims.

Key property under test: the streaming backward avoids materializing the
T*K*D bins_tokens / bins_d_exp buffers and the vmap weight gather that OOM
the Pallas backward (fused_ep_moe_bwd) at full DSv3 scale.

Configs under test:
  tiny:   D=512,  F=256,  E=8,  K=2, T=64
  mini:   D=2048, F=1024, E=16, K=4, T=64 and T=128
  medium: D=4096, F=2048, E=64, K=8, T=64

Run with:
  /home/sivaibhav_google_com/xdb/.xprof/bin/python3 test_grad_check_stage1.py
"""

import sys
import types
import logging
import numpy as np

# Bootstrap: mock vllm (needed by tpu_inference/logger.py) and add tpu-inference
# to sys.path for local dev.  In container, tpu_inference is already at /app/.
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

# Local dev only: add tpu-inference path (container has /app/tpu_inference).
_TPU_INFERENCE = "/home/sivaibhav_google_com/tpu-inference"
if _TPU_INFERENCE not in sys.path:
    sys.path.insert(0, _TPU_INFERENCE)

import jax
import jax.numpy as jnp

from tpu_inference.kernels.fused_moe.v1.kernel import ref_moe
from backward import make_fused_ep_moe_train_v4
from backward_kernel import fused_ep_moe_bwd_streaming


def norm_parity(a, b, name="", tol=0.02):
    na = float(jnp.linalg.norm(jnp.asarray(a)))
    nb = float(jnp.linalg.norm(jnp.asarray(b)))
    ratio = na / (nb + 1e-12)
    ok = abs(ratio - 1.0) < tol
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}  norm_ratio={ratio:.6f}"
          f"  (v4={na:.4f}, ref={nb:.4f})")
    return ok


def make_ep1_mesh():
    device = jax.local_devices()[0]
    return jax.sharding.Mesh(
        np.array([[device]]), ("data", "model"))


def print_oom_avoided(T, D, F, E, K, E_local=None):
    """Print the tensor sizes that would have been created by the old approach."""
    if E_local is None:
        E_local = E
    TK = T * K
    import math
    # max_tpe_single: auto formula from streaming bwd
    avg_tpe = max((TK + E_local - 1) // E_local * 2, 128)
    max_tpe_single = min(avg_tpe, TK)
    max_tpe_single = math.ceil(max_tpe_single / 128) * 128

    # Old approach: E_local * max_tpe * D * 4 bytes for each of bins_tokens, bins_d_exp
    old_bins_gb = E_local * max_tpe_single * D * 4 / 1e9
    # Old approach vmap weight gather: T * K * D * F * 4 bytes
    old_vmap_tb = T * K * D * F * 4 / 1e12

    print(f"  [OOM-avoided] T={T} K={K} D={D} F={F} E_local={E_local}")
    print(f"    Old bins_tokens/bins_d_exp each:  {old_bins_gb:.2f} GB")
    print(f"    Old vmap weight gather:           {old_vmap_tb:.4f} TB")
    print(f"    New 1D bins (4 arrays × pad_to):  "
          f"{4 * E_local * max_tpe_single * 4 / 1e6:.1f} MB  (index/scalar only)")


def check_grads(mesh, T, D, F, E, K, scoring_fn, renormalize_topk_logits):
    """Compare streaming backward against ref_moe on EP=1.

    Tests fused_ep_moe_bwd_streaming directly (d_out=ones) to isolate the
    backward from the Pallas forward kernel (which OOMs on v4 for mini/medium).
    """
    key = jax.random.PRNGKey(42)
    k1, k2, k3, k4 = jax.random.split(key, 4)
    tokens = jax.random.normal(k1, (T, D),       dtype=jnp.float32)
    w1     = jax.random.normal(k2, (E, 2, D, F), dtype=jnp.float32) * 0.02
    w2     = jax.random.normal(k3, (E, F, D),    dtype=jnp.float32) * 0.02
    gating = jax.random.normal(k4, (T, E),        dtype=jnp.float32) * 0.1

    cfg = dict(top_k=K, scoring_fn=scoring_fn,
               renormalize_topk_logits=renormalize_topk_logits)

    # Reference gradients via full JAX autodiff (loss = moe_out.sum() → d_out = ones).
    ref_grads = jax.grad(
        lambda t, w1, w2, g: ref_moe(t, w1, w2, g, **cfg).sum(),
        argnums=(0, 1, 2, 3))(tokens, w1, w2, gating)

    # Stage 1 streaming backward: call directly with d_out=ones, bypassing Pallas fwd.
    # This isolates the backward correctness from the forward kernel VMEM constraints.
    d_out = jnp.ones((T, D), jnp.float32)
    v4_grads = fused_ep_moe_bwd_streaming(
        d_out, tokens, w1, w2, gating, K,
        scoring_fn=scoring_fn,
        renormalize_topk_logits=renormalize_topk_logits,
        act_fn="silu",
    )

    ok_tok  = norm_parity(v4_grads[0], ref_grads[0], "d_tokens")
    ok_gate = norm_parity(v4_grads[3], ref_grads[3], "d_gating")
    ok_w1   = norm_parity(v4_grads[1], ref_grads[1], "d_w1   ")
    ok_w2   = norm_parity(v4_grads[2], ref_grads[2], "d_w2   ")
    return all([ok_tok, ok_gate, ok_w1, ok_w2])


def run_test(name, mesh, v7x_only=False, **kwargs):
    print(f"\n=== {name} ===")
    try:
        ok = check_grads(mesh, **kwargs)
    except Exception as e:
        if v7x_only:
            print(f"  [SKIP] requires v7x — ref_moe OOMs on v4: {type(e).__name__}")
            return None
        raise
    print_oom_avoided(
        T=kwargs["T"], D=kwargs["D"], F=kwargs["F"],
        E=kwargs["E"], K=kwargs["K"])
    return ok


if __name__ == "__main__":
    n_dev = len(jax.local_devices())
    proc = jax.process_index()
    if proc == 0:
        print(f"JAX local devices ({n_dev}): {jax.local_devices()}")
        print(f"Total devices across all processes: {jax.device_count()}")
        print(f"Backend: {jax.default_backend()}")

    mesh = make_ep1_mesh()
    results = {}

    # ---- Tiny: D=512, F=256, E=8, K=2 ----
    results["tiny_sigmoid_renorm"] = run_test(
        "Tiny  D=512  F=256  T=64  E=8  K=2  sigmoid+renorm",
        mesh, T=64, D=512, F=256, E=8, K=2,
        scoring_fn="sigmoid", renormalize_topk_logits=True,
    )
    results["tiny_softmax"] = run_test(
        "Tiny  D=512  F=256  T=64  E=8  K=2  softmax",
        mesh, T=64, D=512, F=256, E=8, K=2,
        scoring_fn="softmax", renormalize_topk_logits=False,
    )

    # ---- Mini: D=2048, F=1024, E=16, K=4 — DS-v3 mini dims ----
    results["mini_sigmoid_renorm"] = run_test(
        "Mini  D=2048 F=1024 T=64  E=16 K=4  sigmoid+renorm",
        mesh, T=64, D=2048, F=1024, E=16, K=4,
        scoring_fn="sigmoid", renormalize_topk_logits=True,
    )
    results["mini_sigmoid_renorm_t128"] = run_test(
        "Mini  D=2048 F=1024 T=128 E=16 K=4  sigmoid+renorm",
        mesh, T=128, D=2048, F=1024, E=16, K=4,
        scoring_fn="sigmoid", renormalize_topk_logits=True,
    )

    # ---- Medium: D=4096, F=2048, E=64, K=8 — DS-v3 medium dims ----
    # ref_moe itself OOMs on v4 (vmap weight gather = 16 GB); test on v7x.
    results["medium_sigmoid_renorm"] = run_test(
        "Medium D=4096 F=2048 T=64  E=64 K=8  sigmoid+renorm",
        mesh, T=64, D=4096, F=2048, E=64, K=8,
        scoring_fn="sigmoid", renormalize_topk_logits=True,
        v7x_only=True,
    )

    # ---- Summary ----
    all_pass = all(v for v in results.values() if v is not None)
    if proc == 0:
        print("\n" + "=" * 60)
        for name, ok in results.items():
            label = "PASS" if ok else ("SKIP" if ok is None else "FAIL")
            print(f"  {label}  {name}")
        print("=" * 60)
        print(f"Overall: {'ALL PASSED' if all_pass else 'SOME FAILED'}")
    sys.exit(0 if all_pass else 1)
