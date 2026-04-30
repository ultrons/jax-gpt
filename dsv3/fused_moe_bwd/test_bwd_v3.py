"""Correctness test for fused_ep_moe_bwd_v3 Pallas backward kernel.

Strategy:
  - EP=1, FSDP=1 (single device): validates backward math and A2A (degenerates
    to local copies when num_devices=1). No ICI DMA exercised.
  - Reference: fused_ep_moe_bwd_streaming (Stage-1 JAX backward) for d_tok/d_w.
  - Routing: deterministic, constructed from gating scores to avoid buffer overflows.

Config: D=256, F=128, E=8, K=2, T=32 (F>=128 required for Mosaic DMA tiling alignment).

Run on a single v7x host:
  cd ~/ml-experiments/dsv3/fused_moe_bwd
  python test_bwd_v3.py
"""

import sys
import os
import logging
import types

# Bootstrap vllm mock for tpu_inference imports
_vllm = types.ModuleType("vllm")
_vllm_logger = types.ModuleType("vllm.logger")

class _VllmLogger(logging.Logger):
    def warning_once(self, msg, *a, **kw): self.warning(msg, *a, **kw)
    def info_once(self, msg, *a, **kw): self.info(msg, *a, **kw)
    def debug_once(self, msg, *a, **kw): self.debug(msg, *a, **kw)

logging.setLoggerClass(_VllmLogger)
_vllm_logger.init_logger = lambda name: logging.getLogger(name)
_vllm_logger._VllmLogger = _VllmLogger
_vllm_logger.init_vllm_logger = lambda name: logging.getLogger(name)
sys.modules.setdefault("vllm", _vllm)
sys.modules.setdefault("vllm.logger", _vllm_logger)

import numpy as np
import jax
import jax.numpy as jnp
from jax import lax
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from backward_kernel import fused_ep_moe_bwd_streaming, compute_routing
from backward_kernel_v3 import fused_ep_moe_bwd_v3


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

D = 256
F = 128  # must be >= 128 (Mosaic DMA tiling requires last dim multiple of 128)
E = 8
K = 2
T = 32
SEED = 42


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def norm_parity(a, b, name, tol=0.03, verbose=True):
    a = jnp.asarray(a).astype(jnp.float32)
    b = jnp.asarray(b).astype(jnp.float32)
    max_abs = float(jnp.max(jnp.abs(a - b)))
    na = float(jnp.linalg.norm(a))
    nb = float(jnp.linalg.norm(b))
    ratio = na / (nb + 1e-12)
    ok = abs(ratio - 1.0) < tol and max_abs < (nb * tol + 1e-6)
    status = "PASS" if ok else "FAIL"
    if verbose:
        print(f"  [{status}] {name:14s}  max_abs={max_abs:.3e}  norm_ratio={ratio:.4f}"
              f"  (v3={na:.4f} ref={nb:.4f})")
    return ok


def make_deterministic_gating(T, E, K, seed=0):
    """Construct gating logits that give deterministic, distinct top-K routing.

    Token t routes to experts (t*K % E), (t*K+1 % E), ..., (t*K+K-1 % E)
    (all distinct since K <= E). Gating scores are 10.0 for selected experts,
    decreasing by 1 per rank, -10.0 for all others.
    """
    rng = np.random.default_rng(seed)
    gating = -10.0 * np.ones((T, E), dtype=np.float32)
    fi = np.zeros((T, K), dtype=np.int32)
    for t in range(T):
        for k in range(K):
            e = (t * K + k) % E
            # Ensure distinct experts per token
            while e in fi[t, :k]:
                e = (e + 1) % E
            fi[t, k] = e
            gating[t, e] = 10.0 - k  # top-1 gets 10, top-2 gets 9, etc.

    # Add small noise to break ties cleanly (still won't change routing)
    gating += rng.uniform(-0.01, 0.01, gating.shape).astype(np.float32)
    return gating, fi


# ---------------------------------------------------------------------------
# EP=1 correctness test
# ---------------------------------------------------------------------------

def check_v3_ep1(D=D, F=F, E=E, K=K, T=T, verbose=True):
    """EP=1, FSDP=1: v3 Pallas kernel vs JAX streaming reference.

    With num_devices=1, all A2A is local (no ICI DMA). Tests the backward
    math: d_tok, d_w1, d_w2.
    """
    key = jax.random.PRNGKey(SEED)
    k1, k2, k3, k4 = jax.random.split(key, 4)

    tokens  = jax.random.normal(k1, (T, D), dtype=jnp.bfloat16)
    w1      = jax.random.normal(k2, (E, 2, D, F), dtype=jnp.bfloat16) * 0.02
    w2      = jax.random.normal(k3, (E, F, D), dtype=jnp.bfloat16) * 0.02
    d_out   = jax.random.normal(k4, (T, D), dtype=jnp.bfloat16)

    # Deterministic routing: no buffer overflow, exactly known fi/fw
    gating_logits_np, fi_np = make_deterministic_gating(T, E, K, seed=SEED)
    gating_logits = jnp.array(gating_logits_np)

    # Compute fw (renormalized sigmoid weights) matching what v3 kernel computes:
    #   v3 kernel receives sigmoid(gating).astype(bf16), then computes top_k and renorms.
    gating_sigmoid_bf16 = jnp.asarray(
        jax.nn.sigmoid(gating_logits).astype(jnp.bfloat16)
    ).astype(jnp.float32)
    # Gather per-token weights and renormalize
    fw_raw = gating_sigmoid_bf16[jnp.arange(T)[:, None], fi_np]  # (T, K)
    fw_np = np.array(fw_raw / fw_raw.sum(axis=-1, keepdims=True))  # renorm

    if verbose:
        print(f"  Routing sample: token 0 → experts {fi_np[0]}, weights {fw_np[0]}")

    # ---- JAX reference ----
    ref = fused_ep_moe_bwd_streaming(
        d_out.astype(jnp.float32),
        tokens.astype(jnp.float32),
        w1.astype(jnp.float32),
        w2.astype(jnp.float32),
        gating_output=None,
        top_k=K,
        scoring_fn="sigmoid",
        renormalize_topk_logits=True,
        act_fn="silu",
        top_k_indices_precomputed=jnp.array(fi_np),
        top_k_weights_precomputed=jnp.array(fw_np),
        return_dtopk=True,
        E_global_override=E,
    )
    ref_d_tok, ref_d_w1, ref_d_w2, ref_d_topk = ref

    # ---- v3 Pallas kernel ----
    # Inside shard_map(ep=1, fsdp=1): everything local, no collectives needed.
    # We call fused_ep_moe_bwd_v3 directly (as if inside shard_map with 1 device).
    #
    # The gating_l input to v3 is the post-sigmoid scores (already scored).
    gating_l = jax.nn.sigmoid(gating_logits).astype(jnp.bfloat16)  # (T, E) bf16

    # ep_axis_name and fsdp_axis_name are needed by lax.axis_index/axis_size.
    # For EP=1, FSDP=1 test, wrap in shard_map over a 1-device mesh.
    devs = np.array(jax.local_devices()[:1])
    mesh = jax.sharding.Mesh(devs.reshape(1, 1), ("ep", "fsdp"))

    w1_spec  = P("ep", None, None, "fsdp")  # EP shards E, FSDP shards F
    w2_spec  = P("ep", "fsdp", None)
    rep_spec = P()

    def _v3(d_out_l, tok_l, gat_l, fw_l, w1_l, w2_l):
        fi_placeholder = jnp.zeros((tok_l.shape[0], K), dtype=jnp.int32)
        d_tok_p, d_topk_p, d_w1_l, d_w2_l = fused_ep_moe_bwd_v3(
            d_out_l, tok_l,
            fi_placeholder,
            fw_l,
            gat_l,
            w1_l,
            w2_l,
            ep_axis_name="ep",
            fsdp_axis_name="fsdp",
            K=K,
            renormalize_topk_logits=True,
        )
        # EP=1: no psum/psum_scatter needed (single device)
        return d_tok_p, d_w1_l, d_w2_l

    def make_replicated(data, mesh):
        sharding = NamedSharding(mesh, P())
        return jax.make_array_from_single_device_arrays(
            data.shape, sharding,
            [jax.device_put(data, mesh.devices.flat[0])])

    def make_sharded(data, mesh, spec):
        sharding = NamedSharding(mesh, spec)
        idx = sharding.addressable_devices_indices_map(data.shape)
        return jax.make_array_from_single_device_arrays(
            data.shape, sharding,
            [jax.device_put(data[idx[d]], d) for d in idx])

    d_out_dist = make_replicated(np.array(d_out),   mesh)
    tok_dist   = make_replicated(np.array(tokens),  mesh)
    gat_dist   = make_replicated(np.array(gating_l), mesh)
    fw_dist    = make_replicated(np.array(fw_np).astype(np.float32), mesh)
    w1_dist    = make_sharded(np.array(w1), mesh, w1_spec)
    w2_dist    = make_sharded(np.array(w2), mesh, w2_spec)

    try:
        v3_d_tok, v3_d_w1, v3_d_w2 = jax.shard_map(
            _v3, mesh=mesh,
            in_specs=(rep_spec, rep_spec, rep_spec, rep_spec, w1_spec, w2_spec),
            out_specs=(rep_spec, w1_spec, w2_spec),
            check_vma=False,
        )(d_out_dist, tok_dist, gat_dist, fw_dist, w1_dist, w2_dist)
    except Exception as exc:
        print(f"  [CRASH] v3 kernel failed: {exc}")
        import traceback; traceback.print_exc()
        return False

    # ---- Compare ----
    ok_tok = norm_parity(v3_d_tok,  ref_d_tok, "d_tok",   tol=0.05, verbose=verbose)
    ok_w1  = norm_parity(v3_d_w1,  ref_d_w1,  "d_w1",    tol=0.05, verbose=verbose)
    ok_w2  = norm_parity(v3_d_w2,  ref_d_w2,  "d_w2",    tol=0.05, verbose=verbose)

    return all([ok_tok, ok_w1, ok_w2])


# ---------------------------------------------------------------------------
# Python simulation of v3 kernel's EP=4 computation (for debugging)
# ---------------------------------------------------------------------------

def _debug_ep4_simulation(tokens, d_out, w1, w2, fi, fw, ref_d_tok, ref_d_w1, EP, E, K, T):
    """Simulate the v3 kernel's computation in pure NumPy to find algorithm bugs.

    Reproduces what the kernel should do:
      - Each device d has tokens[d*TL:(d+1)*TL] and experts e in [d*EL:(d+1)*EL]
      - Scatter: device d sends (tokens[t], fw[t,k]*d_out[t]) to expert fi[t,k]
      - expert_bwd: compute d_tok, d_w1, d_w2 for each expert
      - Gather: d_tok returned to token owners

    If sim matches ref, the algorithm is correct and the bug is in Pallas DMA.
    If sim doesn't match ref, the algorithm itself has a bug.
    """
    import numpy as np
    from scipy.special import expit as sigmoid_np
    TL = T // EP
    EL = E // EP

    sim_d_tok = np.zeros((T, tokens.shape[1]), dtype=np.float32)
    sim_d_w1  = np.zeros_like(w1)
    sim_d_w2  = np.zeros_like(w2)

    # For each expert, collect all tokens routed to it (from ALL devices).
    for e_id in range(E):
        tok_e = []  # (tokens[t], fw_scaled_dout[t]) pairs
        for t_global in range(T):
            for k in range(K):
                if fi[t_global, k] == e_id:
                    tok_e.append((tokens[t_global], fw[t_global, k] * d_out[t_global], t_global))

        if not tok_e:
            continue

        tokens_e = np.stack([x[0] for x in tok_e], axis=0).astype(np.float32)  # (N, D)
        dout_e   = np.stack([x[1] for x in tok_e], axis=0).astype(np.float32)  # (N, D)
        t_ids    = [x[2] for x in tok_e]

        w1g = w1[e_id, 0].astype(np.float32)  # (D, F)
        w1u = w1[e_id, 1].astype(np.float32)
        w2e = w2[e_id].astype(np.float32)      # (F, D)

        # Forward recompute
        h_g = tokens_e @ w1g
        h_u = tokens_e @ w1u
        sig_hg  = 1.0 / (1.0 + np.exp(-h_g))
        silu_hg = h_g * sig_hg
        h_act   = silu_hg * h_u

        # Backward
        d_h_act = dout_e @ w2e.T
        silu_grad_hg = sig_hg * (1 + h_g * (1 - sig_hg))
        d_h_u = d_h_act * silu_hg
        d_h_g = d_h_act * h_u * silu_grad_hg

        d_tok_e = d_h_g @ w1g.T + d_h_u @ w1u.T
        d_w1g_e = tokens_e.T @ d_h_g
        d_w1u_e = tokens_e.T @ d_h_u
        d_w2e   = h_act.T @ dout_e

        for i, t_global in enumerate(t_ids):
            sim_d_tok[t_global] += d_tok_e[i]
        sim_d_w1[e_id, 0] += d_w1g_e
        sim_d_w1[e_id, 1] += d_w1u_e
        sim_d_w2[e_id]    += d_w2e

    print(f"\n  [DEBUG] Python simulation vs reference:")
    nr_tok = np.linalg.norm(sim_d_tok) / (np.linalg.norm(ref_d_tok) + 1e-12)
    nr_w1  = np.linalg.norm(sim_d_w1)  / (np.linalg.norm(ref_d_w1) + 1e-12)
    print(f"    sim d_tok norm_ratio={nr_tok:.4f} (sim={np.linalg.norm(sim_d_tok):.4f} ref={np.linalg.norm(ref_d_tok):.4f})")
    print(f"    sim d_w1  norm_ratio={nr_w1:.4f}  (sim={np.linalg.norm(sim_d_w1):.4f} ref={np.linalg.norm(ref_d_w1):.4f})")
    if abs(nr_tok - 1.0) < 0.01 and abs(nr_w1 - 1.0) < 0.01:
        print(f"    => Algorithm is CORRECT (bug is in Pallas DMA/semaphore layer)")
    else:
        print(f"    => Algorithm has a BUG (conceptual error in kernel design)")
        # Print per-token debug
        for t in range(min(4, T)):
            nr_t = np.linalg.norm(sim_d_tok[t]) / (np.linalg.norm(ref_d_tok[t]) + 1e-12)
            print(f"    token {t:2d}: sim_norm={np.linalg.norm(sim_d_tok[t]):.4f} ref_norm={np.linalg.norm(ref_d_tok[t]):.4f} ratio={nr_t:.4f}")


# ---------------------------------------------------------------------------
# EP=4 correctness test (real ICI DMA between chips within a pod)
# ---------------------------------------------------------------------------

def check_v3_ep4(D=D, F=F, E=E, K=K, T=T, EP=4, verbose=True, passthrough=False):
    """EP=4, FSDP=1: validates A2A scatter+gather with real ICI DMA.

    Uses 4 local devices. Each device owns T/EP tokens and E/EP experts.
    Compares full d_tok, d_w1, d_w2 against JAX streaming reference.

    passthrough=True: expert_bwd passes tokens straight through to acc buffer
    (skips GEMMs). Expected d_tok = 2*tokens. Tests ICI gather independently.
    """
    if jax.local_device_count() < EP:
        if verbose:
            print(f"  [SKIP] need {EP} local devices, have {jax.local_device_count()}")
        return None

    key = jax.random.PRNGKey(SEED)
    k1, k2, k3, k4 = jax.random.split(key, 4)

    tokens = jax.random.normal(k1, (T, D), dtype=jnp.bfloat16)
    w1     = jax.random.normal(k2, (E, 2, D, F), dtype=jnp.bfloat16) * 0.02
    w2     = jax.random.normal(k3, (E, F, D), dtype=jnp.bfloat16) * 0.02
    d_out  = jax.random.normal(k4, (T, D), dtype=jnp.bfloat16)

    gating_logits_np, fi_np = make_deterministic_gating(T, E, K, seed=SEED)
    gating_logits = jnp.array(gating_logits_np)
    gating_sigmoid_bf16 = jnp.asarray(
        jax.nn.sigmoid(gating_logits).astype(jnp.bfloat16)
    ).astype(jnp.float32)
    fw_raw = gating_sigmoid_bf16[jnp.arange(T)[:, None], fi_np]
    fw_np  = np.array(fw_raw / fw_raw.sum(axis=-1, keepdims=True))

    if verbose:
        print(f"  Config: T={T} E={E} K={K} D={D} F={F} EP={EP}")
        print(f"  T_local={T//EP} E_local={E//EP}")

    # ---- JAX reference (full tensors, no sharding) ----
    ref = fused_ep_moe_bwd_streaming(
        d_out.astype(jnp.float32),
        tokens.astype(jnp.float32),
        w1.astype(jnp.float32),
        w2.astype(jnp.float32),
        gating_output=None,
        top_k=K,
        scoring_fn="sigmoid",
        renormalize_topk_logits=True,
        act_fn="silu",
        top_k_indices_precomputed=jnp.array(fi_np),
        top_k_weights_precomputed=jnp.array(fw_np),
        return_dtopk=True,
        E_global_override=E,
    )
    ref_d_tok, ref_d_w1, ref_d_w2, _ = ref

    # ---- v3 EP=4 kernel ----
    gating_l = jax.nn.sigmoid(gating_logits).astype(jnp.bfloat16)  # (T, E)

    devs    = np.array(jax.local_devices()[:EP])
    mesh    = jax.sharding.Mesh(devs.reshape(EP, 1), ("ep", "fsdp"))
    tok_spec = P("ep", None)         # tokens/d_out/gating/fw sharded by EP along T
    w1_spec  = P("ep", None, None, "fsdp")
    w2_spec  = P("ep", "fsdp", None)

    def _v3(d_out_l, tok_l, gat_l, fw_l, w1_l, w2_l):
        fi_placeholder = jnp.zeros((tok_l.shape[0], K), dtype=jnp.int32)
        d_tok_p, _, d_w1_l, d_w2_l = fused_ep_moe_bwd_v3(
            d_out_l, tok_l,
            fi_placeholder,
            fw_l,
            gat_l,
            w1_l,
            w2_l,
            ep_axis_name="ep",
            fsdp_axis_name="fsdp",
            K=K,
            renormalize_topk_logits=True,
            passthrough=passthrough,
        )
        # FSDP=1: psum("fsdp") is identity; A2A gather already returned T_local d_tok
        d_tok = lax.psum(d_tok_p, "fsdp")
        return d_tok, d_w1_l, d_w2_l

    def make_sharded(data, mesh, spec):
        sharding = NamedSharding(mesh, spec)
        idx = sharding.addressable_devices_indices_map(data.shape)
        return jax.make_array_from_single_device_arrays(
            data.shape, sharding,
            [jax.device_put(data[idx[d]], d) for d in idx])

    d_out_dist = make_sharded(np.array(d_out),    mesh, tok_spec)
    tok_dist   = make_sharded(np.array(tokens),   mesh, tok_spec)
    gat_dist   = make_sharded(np.array(gating_l), mesh, tok_spec)
    fw_dist    = make_sharded(np.array(fw_np).astype(np.float32), mesh, tok_spec)
    w1_dist    = make_sharded(np.array(w1),        mesh, w1_spec)
    w2_dist    = make_sharded(np.array(w2),        mesh, w2_spec)

    try:
        v3_d_tok, v3_d_w1, v3_d_w2 = jax.shard_map(
            _v3, mesh=mesh,
            in_specs=(tok_spec, tok_spec, tok_spec, tok_spec, w1_spec, w2_spec),
            out_specs=(tok_spec, w1_spec, w2_spec),
            check_vma=False,
        )(d_out_dist, tok_dist, gat_dist, fw_dist, w1_dist, w2_dist)
    except Exception as exc:
        print(f"  [CRASH] v3 ep4 kernel failed: {exc}")
        import traceback; traceback.print_exc()
        return False

    if passthrough:
        # Passthrough mode: d_tok should equal 2*tokens (token echoed back from K=2 routes).
        # d_w1/d_w2 are uninitialized (not checked).
        expected_d_tok = (2.0 * jnp.array(tokens)).astype(jnp.float32)
        ok_tok = norm_parity(v3_d_tok, expected_d_tok, "d_tok(passthrough)", tol=0.01, verbose=verbose)
        if verbose:
            if ok_tok:
                print(f"  [Passthrough PASS] ICI gather path works — d_tok = 2*tokens as expected")
            else:
                print(f"  [Passthrough FAIL] ICI gather path BROKEN — d_tok != 2*tokens")
                vtok = np.array(v3_d_tok).astype(np.float32)
                etok = np.array(expected_d_tok).astype(np.float32)
                print(f"\n  Per-token d_tok vs 2*tokens:")
                for t in range(min(T, 8)):
                    vn = np.linalg.norm(vtok[t]); en = np.linalg.norm(etok[t])
                    print(f"    t={t:2d}: got={vn:.4f} expected={en:.4f} ratio={vn/(en+1e-12):.4f}")
        return ok_tok

    ok_tok = norm_parity(v3_d_tok, ref_d_tok, "d_tok", tol=0.05, verbose=verbose)
    ok_w1  = norm_parity(v3_d_w1,  ref_d_w1,  "d_w1",  tol=0.05, verbose=verbose)
    ok_w2  = norm_parity(v3_d_w2,  ref_d_w2,  "d_w2",  tol=0.05, verbose=verbose)

    if verbose and not all([ok_tok, ok_w1, ok_w2]):
        # Per-token d_tok analysis
        vtok = np.array(v3_d_tok).astype(np.float32)
        rtok = np.array(ref_d_tok).astype(np.float32)
        print(f"\n  Per-token d_tok norm ratios:")
        for t in range(min(T, 8)):
            vn = np.linalg.norm(vtok[t]); rn = np.linalg.norm(rtok[t])
            print(f"    t={t:2d}: v3={vn:.4f} ref={rn:.4f} ratio={vn/(rn+1e-12):.4f}")
        # Per-expert d_w1 analysis
        vw1 = np.array(v3_d_w1).astype(np.float32)
        rw1 = np.array(ref_d_w1).astype(np.float32)
        print(f"\n  Per-expert d_w1 norm ratios:")
        for e in range(E):
            vn = np.linalg.norm(vw1[e]); rn = np.linalg.norm(rw1[e])
            print(f"    e={e}: v3={vn:.4f} ref={rn:.4f} ratio={vn/(rn+1e-12):.4f}")

        # Python simulation: reproduce the kernel's computation to isolate
        # whether the bug is in the algorithm (Python level) or Pallas (DMA level).
        _debug_ep4_simulation(
            tokens=np.array(tokens).astype(np.float32),
            d_out=np.array(d_out).astype(np.float32),
            w1=np.array(w1).astype(np.float32),
            w2=np.array(w2).astype(np.float32),
            fi=fi_np, fw=fw_np,
            ref_d_tok=np.array(ref_d_tok).astype(np.float32),
            ref_d_w1=np.array(ref_d_w1).astype(np.float32),
            EP=EP, E=E, K=K, T=T,
        )

    return all([ok_tok, ok_w1, ok_w2])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"[__main__] jax.process_index={jax.process_index()} "
          f"local_device_count={jax.local_device_count()} "
          f"device_count={jax.device_count()} "
          f"backend={jax.default_backend()}", flush=True)

    proc = jax.process_index()

    if proc == 0:
        print(f"JAX: {jax.__version__}")
        print(f"devices: local={jax.local_device_count()} total={jax.device_count()}")
        print(f"backend: {jax.default_backend()}")
        print(f"Config: D={D} F={F} E={E} K={K} T={T}")

    results = {}

    # ── Test 1: EP=1, FSDP=1 — backward math on single device ────────────────
    if proc == 0:
        print(f"\n=== Test 1: EP=1 FSDP=1 (single device) ===")
    try:
        ok = check_v3_ep1(verbose=(proc == 0))
    except Exception:
        if proc == 0:
            import traceback; traceback.print_exc()
        ok = False
    results["v3_ep1_fsdp1"] = ok

    # ── Test 2a: EP=4 passthrough — ICI gather path test ──────────────────────
    # expert_bwd copies tokens→acc; expected d_tok = 2*tokens.
    # PASS → ICI gather works; bug is in expert_bwd computation.
    # FAIL → ICI gather itself is broken.
    if proc == 0:
        print(f"\n=== Test 2a: EP=4 FSDP=1 passthrough (ICI gather path) ===")
    try:
        ok = check_v3_ep4(verbose=(proc == 0), passthrough=True)
    except Exception:
        if proc == 0:
            import traceback; traceback.print_exc()
        ok = False
    results["v3_ep4_passthrough"] = ok

    # ── Test 2b: EP=4, FSDP=1 — A2A with ICI DMA + real backward math ────────
    if proc == 0:
        print(f"\n=== Test 2b: EP=4 FSDP=1 (ICI DMA + backward math) ===")
    try:
        ok = check_v3_ep4(verbose=(proc == 0))
    except Exception:
        if proc == 0:
            import traceback; traceback.print_exc()
        ok = False
    results["v3_ep4_fsdp1"] = ok

    # ── Test 3a: EP=8 passthrough — ICI gather path test ─────────────────────
    if proc == 0:
        print(f"\n=== Test 3a: EP=8 FSDP=1 passthrough (ICI gather path) ===")
    try:
        ok = check_v3_ep4(verbose=(proc == 0), passthrough=True, EP=8)
    except Exception:
        if proc == 0:
            import traceback; traceback.print_exc()
        ok = False
    results["v3_ep8_passthrough"] = ok

    # ── Test 3b: EP=8, FSDP=1 — A2A with ICI DMA + real backward math ────────
    if proc == 0:
        print(f"\n=== Test 3b: EP=8 FSDP=1 (ICI DMA + backward math) ===")
    try:
        ok = check_v3_ep4(verbose=(proc == 0), EP=8)
    except Exception:
        if proc == 0:
            import traceback; traceback.print_exc()
        ok = False
    results["v3_ep8_fsdp1"] = ok

    # ── Summary ───────────────────────────────────────────────────────────────
    if proc == 0:
        print("\n" + "=" * 60)
        for name, val in results.items():
            label = "PASS" if val else ("SKIP" if val is None else "FAIL")
            print(f"  {label}  {name}")
        print("=" * 60)
        all_pass = all(v for v in results.values() if v is not None)
        print(f"Overall: {'ALL PASSED' if all_pass else 'SOME FAILED'}")

    all_pass = all(v for v in results.values() if v is not None)
    sys.exit(0 if all_pass else 1)
