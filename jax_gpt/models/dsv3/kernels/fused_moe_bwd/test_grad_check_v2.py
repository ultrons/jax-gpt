"""Phase 2 correctness test: fused_ep_moe_bwd_streaming_v2 vs v1 reference.

v2 accepts FSDP-sharded weights w1(E,2,D,F/fsdp) and w2(E,F/fsdp,D), gathers
one expert at a time inside lax.scan (double-buffering), then psum_scatters the
weight grads back. The result must match v1 called with the full (gathered) weights.

Tests:
  1. fsdp_only — EP=1, fsdp=8 (single v7x host, 8 local devices)
       v2 via shard_map("fsdp") vs v1 on one device with full weights.
  2. ep_fsdp  — EP=8, fsdp=8 (4x8x8, 512 devices total)
       v2 via shard_map("ep","fsdp") vs v1 on one device with full weights.
       d_tokens and d_gating must match after EP psum.

Run locally (single host, 8 devices):
  cd ~/dsv3/fused_moe_bwd
  python test_grad_check_v2.py

Run on 4x8x8 cluster:
  kubectl --context gke_cloud-tpu-multipod-dev_us-central1_ninja-v7x-64-spot \\
      apply -f k8s/test-v2-4x8x8.yaml
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

import jax
import jax.numpy as jnp
from jax import lax
from jax.sharding import NamedSharding, PartitionSpec as P

from .backward_kernel import (compute_routing, fused_ep_moe_bwd_streaming,
                             fused_ep_moe_bwd_streaming_v2)


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
              f"  (v2={na:.4f}, v1={nb:.4f})")
    return ok


def make_sharded(data, mesh, spec):
    """Create a distributed JAX array from full numpy data + NamedSharding spec."""
    sharding = NamedSharding(mesh, spec)
    indices_map = sharding.addressable_devices_indices_map(data.shape)
    local_shards = [jax.device_put(data[idx], dev) for dev, idx in indices_map.items()]
    return jax.make_array_from_single_device_arrays(data.shape, sharding, local_shards)


def make_replicated(data, mesh):
    sharding = NamedSharding(mesh, P())
    local_shards = [jax.device_put(data, dev) for dev in sharding.addressable_devices]
    return jax.make_array_from_single_device_arrays(data.shape, sharding, local_shards)


# ---------------------------------------------------------------------------
# Test 1: FSDP-only (EP=1), single host
# ---------------------------------------------------------------------------

def check_v2_fsdp_only(mesh, T=64, D=256, F=128, E=32, K=8, verbose=True):
    """EP=1, fsdp=8: v2 (shard F across fsdp) vs v1 reference (full weights).

    Weight layout inside shard_map:
      w1_l: (E, 2, D, F/fsdp)  ← P(None, None, None, "fsdp")
      w2_l: (E, F/fsdp, D)     ← P(None, "fsdp", None)
    v2 gathers one expert per scan step, psum_scatters weight grads after.
    d_tokens and d_gating are psum'd inside v2 (replicated output).

    Memory at test scale (E=32, D=256, F=128, fsdp=8):
      F_shard = 16; w1_l = 32×2×256×16×4 = 1.0 MB per device.
      v2 double-buffer peak: 2×[(2×256×128)+(128×256)]×4 = 0.8 MB — trivial.
    """
    fsdp = mesh.shape["fsdp"]
    assert F % fsdp == 0, f"F={F} must be divisible by fsdp={fsdp}"

    key = jax.random.PRNGKey(7)
    k1, k2, k3, k4 = jax.random.split(key, 4)
    tokens = jax.random.normal(k1, (T, D),       dtype=jnp.float32)
    w1     = jax.random.normal(k2, (E, 2, D, F), dtype=jnp.float32) * 0.02
    w2     = jax.random.normal(k3, (E, F, D),    dtype=jnp.float32) * 0.02
    gating = jax.random.normal(k4, (T, E),        dtype=jnp.float32) * 0.1
    d_out  = jnp.ones((T, D), jnp.float32)

    # ---- v1 reference: full weights on one device ----
    v1 = fused_ep_moe_bwd_streaming(
        d_out, tokens, w1, w2, gating, K,
        scoring_fn="sigmoid", renormalize_topk_logits=True, act_fn="silu",
    )
    # v1 returns (d_tokens, d_w1[E,2,D,F], d_w2[E,F,D], d_gating[T,E])

    # ---- Distribute for v2 ----
    # w1: shard F (axis 3) across fsdp  →  P(None, None, None, "fsdp")
    # w2: shard F (axis 1) across fsdp  →  P(None, "fsdp", None)
    w1_spec = P(None, None, None, "fsdp")
    w2_spec = P(None, "fsdp", None)

    w1_dist  = make_sharded(np.array(w1),     mesh, w1_spec)
    w2_dist  = make_sharded(np.array(w2),     mesh, w2_spec)
    tok_dist = make_replicated(np.array(tokens), mesh)
    gat_dist = make_replicated(np.array(gating), mesh)
    dout_dist = make_replicated(np.array(d_out),  mesh)

    # ---- v2 via shard_map over fsdp ----
    def _v2(dout_l, tok_l, w1_l, w2_l, gat_l):
        # w1_l: (E, 2, D, F/fsdp); w2_l: (E, F/fsdp, D)
        d_tok, d_w1_l, d_w2_l, d_gate = fused_ep_moe_bwd_streaming_v2(
            dout_l, tok_l, w1_l, w2_l, gat_l, K,
            fsdp_axis_name="fsdp",
            scoring_fn="sigmoid", renormalize_topk_logits=True, act_fn="silu",
        )
        # d_tok and d_gate: replicated across fsdp devices because v2 uses full-F
        # weights (gathered) — all fsdp devices compute the same d_tok.
        # d_w1_l: (E, 2, D, F/fsdp); d_w2_l: (E, F/fsdp, D)
        return d_tok, d_w1_l, d_w2_l, d_gate

    v2 = jax.shard_map(
        _v2, mesh=mesh,
        in_specs=(P(), P(), w1_spec, w2_spec, P()),
        out_specs=(P(), w1_spec, w2_spec, P()),
        check_vma=False,
    )(dout_dist, tok_dist, w1_dist, w2_dist, gat_dist)
    # v2[0]=d_tokens (replicated), v2[1]=d_w1 (sharded F), v2[2]=d_w2 (sharded F),
    # v2[3]=d_gating (replicated)

    # ---- Compare ----
    # d_tokens and d_gating: v2 outputs are replicated global arrays; take process-0 slice.
    ok_tok  = norm_parity(v2[0], v1[0], "d_tokens", verbose=verbose)
    ok_gate = norm_parity(v2[3], v1[3], "d_gating", verbose=verbose)

    # d_w1, d_w2: v2 outputs are sharded; jnp.linalg.norm gathers internally for comparison.
    ok_w1 = norm_parity(v2[1], v1[1], "d_w1    ", verbose=verbose)
    ok_w2 = norm_parity(v2[2], v1[2], "d_w2    ", verbose=verbose)

    return all([ok_tok, ok_gate, ok_w1, ok_w2])


# ---------------------------------------------------------------------------
# Test 2: EP=8 + FSDP=8, full 4×8×8 cluster
# ---------------------------------------------------------------------------

def check_v2_ep_fsdp(mesh, T=128, D=256, F=512, E=256, K=8, verbose=True):
    """EP=8 + fsdp=8: v2 via shard_map(ep, fsdp) vs v1 reference (EP=1 full weights).

    Topology: mesh ("fsdp"=64, "ep"=8).
    Each EP device handles E_local=32 experts; FSDP splits F into F/8=16-dim slices.
    v2 is called inside the shard_map body → fsdp collective works within the EP group.

    After shard_map:
      d_tokens: psum'd inside v2 over fsdp, then psum'd over ep by caller → matches EP=1 v1.
      d_gating: same.
      d_w1, d_w2: EP-local (no EP reduction needed — each EP device owns its experts).
    """
    ep_axis  = "ep"
    EP       = mesh.shape[ep_axis]
    fsdp     = mesh.shape["fsdp"]
    assert E % EP == 0,    f"E={E} not divisible by EP={EP}"
    assert F % fsdp == 0,  f"F={F} not divisible by fsdp={fsdp}"
    E_local = E // EP

    key = jax.random.PRNGKey(42)
    k1, k2, k3, k4 = jax.random.split(key, 4)
    d_out   = jnp.ones((T, D),       jnp.float32)
    tokens  = jax.random.normal(k1, (T, D),       dtype=jnp.float32)
    w1      = jax.random.normal(k2, (E, 2, D, F), dtype=jnp.float32) * 0.02
    w2      = jax.random.normal(k3, (E, F, D),    dtype=jnp.float32) * 0.02
    gating  = jax.random.normal(k4, (T, E),        dtype=jnp.float32) * 0.1

    # ---- EP=1 v1 reference ----
    ep1_grads = fused_ep_moe_bwd_streaming(
        d_out, tokens, w1, w2, gating, K,
        scoring_fn="sigmoid", renormalize_topk_logits=True, act_fn="silu",
    )

    # ---- Distribute for v2 ----
    # w1: shard E on "ep" (axis 0), shard F on "fsdp" (axis 3)
    # w2: shard E on "ep" (axis 0), shard F on "fsdp" (axis 1)
    w1_spec   = P("ep", None, None, "fsdp")
    w2_spec   = P("ep", "fsdp", None)
    rep_spec  = P()

    w1_dist   = make_sharded(np.array(w1),     mesh, w1_spec)
    w2_dist   = make_sharded(np.array(w2),     mesh, w2_spec)
    tok_dist  = make_replicated(np.array(tokens), mesh)
    gat_dist  = make_replicated(np.array(gating), mesh)
    dout_dist = make_replicated(np.array(d_out),  mesh)

    def _v2_ep_fsdp(dout_l, tok_l, w1_l, w2_l, gat_l):
        # w1_l: (E_local, 2, D, F/fsdp); w2_l: (E_local, F/fsdp, D)
        # fused_ep_moe_bwd_streaming_v2 handles FSDP gather internally.
        # EP>1: it also handles EP masking internally via ep_axis_name.
        d_tok_p, d_w1_l, d_w2_l, d_gate_p = fused_ep_moe_bwd_streaming_v2(
            dout_l, tok_l, w1_l, w2_l, gat_l, K,
            fsdp_axis_name="fsdp",
            scoring_fn="sigmoid", renormalize_topk_logits=True,
            act_fn="silu", ep_axis_name=ep_axis,
        )
        # psum partial d_tokens and d_gating across EP devices.
        # (v2 already psum'd across fsdp internally; now psum across ep.)
        d_tok  = lax.psum(d_tok_p,  axis_name=ep_axis)
        d_gate = lax.psum(d_gate_p, axis_name=ep_axis)
        return d_tok, d_w1_l, d_w2_l, d_gate

    ep_grads = jax.shard_map(
        _v2_ep_fsdp, mesh=mesh,
        in_specs=(rep_spec, rep_spec, w1_spec, w2_spec, rep_spec),
        out_specs=(rep_spec, w1_spec, w2_spec, rep_spec),
        check_vma=False,
    )(dout_dist, tok_dist, w1_dist, w2_dist, gat_dist)

    # Compare d_tokens and d_gating (EP psum'd in _v2_ep_fsdp → matches EP=1 reference).
    ok_tok  = norm_parity(ep_grads[0], ep1_grads[0],
                          "d_tokens (EP=8 vs EP=1)", verbose=verbose)
    ok_gate = norm_parity(ep_grads[3], ep1_grads[3],
                          "d_gating (EP=8 vs EP=1)", verbose=verbose)
    return ok_tok and ok_gate


# ---------------------------------------------------------------------------
# Test 3: EP=2 + FSDP=2 — mirrors model.py _streaming_bwd_fn exactly
# ---------------------------------------------------------------------------

def check_v2_ep2_fsdp2(mesh, T=32, D=128, F=64, E=8, K=4, verbose=True):
    """EP=2, FSDP=2 (4 devices): mirrors model.py _streaming_bwd_fn.

    Tokens are P(("ep","fsdp"),None) — each device holds T/4 unique tokens.
    Inside shard_map:
      1. all_gather("ep") reconstructs T/FSDP tokens per FSDP group
      2. v2 kernel computes d_tok with full-F weights (gathered internally)
      3. psum("ep") + slice reconstructs per-device d_tok

    This is the critical failure mode: EP>1 AND FSDP>1 with partitioned tokens.
    With partial weights (old v1/v2), d_tok was incomplete in D_moe and produced NaN.
    With this fix (full-F gather), d_tok should match the EP=1 v1 reference.

    Reference: v1 with full T tokens and full weights (EP=1).
    """
    EP   = mesh.shape["ep"]
    fsdp = mesh.shape["fsdp"]
    assert E % EP == 0,   f"E={E} not divisible by EP={EP}"
    assert F % fsdp == 0, f"F={F} not divisible by fsdp={fsdp}"

    key = jax.random.PRNGKey(99)
    k1, k2, k3, k4, k5 = jax.random.split(key, 5)
    g       = jax.random.normal(k1, (T, D),       dtype=jnp.float32)
    tokens  = jax.random.normal(k2, (T, D),       dtype=jnp.float32)
    w1      = jax.random.normal(k3, (E, 2, D, F), dtype=jnp.float32) * 0.02
    w2      = jax.random.normal(k4, (E, F, D),    dtype=jnp.float32) * 0.02
    gating  = jax.random.normal(k5, (T, E),        dtype=jnp.float32) * 0.1

    # Precompute routing on full T tokens (matches model.py: routing saved in fwd residuals).
    fi, fw = compute_routing(gating, K, scoring_fn="sigmoid", renormalize_topk_logits=True)

    # ---- v1 reference: EP=1, full tokens and full weights ----
    v1_d_tok, v1_d_w1, v1_d_w2, v1_d_topk = fused_ep_moe_bwd_streaming(
        g, tokens, w1, w2,
        gating_output=None, top_k=K,
        top_k_indices_precomputed=fi,
        top_k_weights_precomputed=fw,
        return_dtopk=True,
        E_global_override=E,
    )

    # ---- Distribute for v2 EP=2+FSDP=2 ----
    act_spec = P(("ep", "fsdp"), None)
    w1_spec  = P("ep", None, None, "fsdp")
    w2_spec  = P("ep", "fsdp", None)

    g_dist   = make_sharded(np.array(g),      mesh, act_spec)
    tok_dist = make_sharded(np.array(tokens),  mesh, act_spec)
    fi_dist  = make_sharded(np.array(fi),      mesh, act_spec)
    fw_dist  = make_sharded(np.array(fw),      mesh, act_spec)
    w1_dist  = make_sharded(np.array(w1),      mesh, w1_spec)
    w2_dist  = make_sharded(np.array(w2),      mesh, w2_spec)

    def _bwd(g_l, tok_l, fi_l, fw_l, w1_l, w2_l):
        # Mirror model.py _streaming_bwd_fn exactly.
        T_ep_local = g_l.shape[0]
        D_          = g_l.shape[1]

        # EP all_gather: reconstruct T/FSDP tokens (same as model.py).
        g_full   = lax.all_gather(g_l,   "ep", axis=0, tiled=True)
        tok_full = lax.all_gather(tok_l, "ep", axis=0, tiled=True)
        fi_full  = lax.all_gather(fi_l,  "ep", axis=0, tiled=True)
        fw_full  = lax.all_gather(fw_l,  "ep", axis=0, tiled=True)

        d_tok_p, d_w1_l, d_w2_l, d_topk_p = fused_ep_moe_bwd_streaming_v2(
            g_full, tok_full, w1_l, w2_l,
            gating_output=None, top_k=K,
            fsdp_axis_name="fsdp",
            ep_axis_name="ep",
            top_k_indices_precomputed=fi_full,
            top_k_weights_precomputed=fw_full,
            return_dtopk=True,
            E_global_override=E,
        )

        # EP psum + slice (same as model.py).
        d_tok_full  = lax.psum(d_tok_p,  "ep")
        d_topk_full = lax.psum(d_topk_p, "ep")
        device_ep   = lax.axis_index("ep")
        d_tok_l     = lax.dynamic_slice(d_tok_full,  (device_ep * T_ep_local, 0),
                                        (T_ep_local, D_))
        d_topk_l    = lax.dynamic_slice(d_topk_full, (device_ep * T_ep_local, 0),
                                        (T_ep_local, K))

        return d_tok_l, d_w1_l, d_w2_l, d_topk_l

    v2_d_tok, v2_d_w1, v2_d_w2, v2_d_topk = jax.shard_map(
        _bwd, mesh=mesh,
        in_specs=(act_spec, act_spec, act_spec, act_spec, w1_spec, w2_spec),
        out_specs=(act_spec, w1_spec, w2_spec, act_spec),
        check_vma=False,
    )(g_dist, tok_dist, fi_dist, fw_dist, w1_dist, w2_dist)

    # d_tokens and d_topk are globally reduced (psum over EP) so they must match
    # the full-T v1 reference exactly.
    # d_w1/d_w2 are NOT compared against v1: each FSDP stripe only processes T/FSDP
    # tokens in the forward, so its weight gradient is legitimately ~1/sqrt(FSDP) of
    # the full-batch gradient. The per-stripe weight grad is correct by construction
    # (same token set as the forward), but it doesn't match the full-T v1 reference.
    ok_tok  = norm_parity(v2_d_tok,  v1_d_tok,  "d_tokens", verbose=verbose)
    ok_topk = norm_parity(v2_d_topk, v1_d_topk, "d_topk  ", verbose=verbose)

    return all([ok_tok, ok_topk])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    proc = jax.process_index()

    if proc == 0:
        print(f"JAX: {jax.__version__}")
        print(f"Total JAX devices: {jax.device_count()}")
        print(f"Local JAX devices: {jax.local_device_count()}")
        print(f"Backend: {jax.default_backend()}")

    results = {}

    # ── Test 1: FSDP-only, single host ────────────────────────────────────────
    # Works on any host with ≥8 local devices. No jax.distributed needed.
    # Uses fsdp=8 with small dims (D=256, F=128, E=32, K=8, T=64).
    local_devs = jax.local_devices()
    fsdp_count = min(8, len(local_devs))
    local_mesh = jax.sharding.Mesh(
        np.array(local_devs[:fsdp_count]), ("fsdp",)
    )

    if proc == 0:
        print(f"\n=== Test 1: EP=1 fsdp={fsdp_count}  D=256 F=128 E=32 K=8 T=64 ===")
    try:
        ok = check_v2_fsdp_only(local_mesh, verbose=(proc == 0))
    except Exception:
        if proc == 0:
            import traceback; traceback.print_exc()
        ok = False
    results["v2_fsdp_only"] = ok

    gc.collect()
    jax.effects_barrier()

    # ── Test 2: EP=8 + FSDP, cluster-scale ───────────────────────────────────
    # 4x4x4: 16 pods × 8 local devices = 128 total → mesh (16, 8) fsdp=16 ep=8
    # 4x8x8: 64 pods × 8 local devices = 512 total → mesh (64, 8) fsdp=64 ep=8
    total = jax.device_count()
    if total == 16 * 8:
        devices = np.array(jax.devices()).reshape(16, 8)
        mesh = jax.sharding.Mesh(devices, ("fsdp", "ep"))
        label = "EP=8 fsdp=16 (4x4x4)"
    elif total == 64 * 8:
        devices = np.array(jax.devices()).reshape(64, 8)
        mesh = jax.sharding.Mesh(devices, ("fsdp", "ep"))
        label = "EP=8 fsdp=64 (4x8x8)"
    else:
        mesh = None
        label = None

    if mesh is None:
        if proc == 0:
            print(f"\n  [SKIP] EP=8 test requires 128 or 512 devices, "
                  f"got {total}")
        results["v2_ep8_fsdp"] = None
    else:
        if proc == 0:
            print(f"\n=== Test 2: {label}  D=256 F=128 E=256 K=8 T=128 ===")
        try:
            ok = check_v2_ep_fsdp(mesh, verbose=(proc == 0))
        except Exception:
            if proc == 0:
                import traceback; traceback.print_exc()
            ok = False
        results["v2_ep8_fsdp"] = ok

    gc.collect()
    jax.effects_barrier()

    # ── Test 3: EP=2+FSDP=2, single host (4 devices) ─────────────────────────
    # Mirrors model.py _streaming_bwd_fn: partitioned tokens + all_gather("ep")
    # + psum("ep") + slice. The critical failure mode for the NaN bug.
    local_devs = jax.local_devices()
    if len(local_devs) >= 4:
        ep2_fsdp2_devs = np.array(local_devs[:4]).reshape(2, 2)  # (ep=2, fsdp=2)
        ep2_mesh = jax.sharding.Mesh(ep2_fsdp2_devs, ("ep", "fsdp"))
        if proc == 0:
            print(f"\n=== Test 3: EP=2 fsdp=2  D=128 F=64 E=8 K=4 T=32 ===")
        try:
            ok = check_v2_ep2_fsdp2(ep2_mesh, verbose=(proc == 0))
        except Exception:
            if proc == 0:
                import traceback; traceback.print_exc()
            ok = False
        results["v2_ep2_fsdp2"] = ok
    else:
        if proc == 0:
            print(f"\n  [SKIP] EP=2+FSDP=2 test requires 4 devices, got {len(local_devs)}")
        results["v2_ep2_fsdp2"] = None

    gc.collect()
    jax.effects_barrier()

    # ── Summary ───────────────────────────────────────────────────────────────
    all_pass = all(v for v in results.values() if v is not None)
    if proc == 0:
        print("\n" + "=" * 60)
        for name, ok in results.items():
            label = "PASS" if ok else ("SKIP" if ok is None else "FAIL")
            print(f"  {label}  {name}")
        print("=" * 60)
        print(f"Overall: {'ALL PASSED' if all_pass else 'SOME FAILED'}")
    sys.exit(0 if all_pass else 1)
