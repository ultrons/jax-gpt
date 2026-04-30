"""Test: does v1 with F-sharded weights produce correct d_tok after FSDP psum?

The production code (model.py _streaming_bwd_fn) calls fused_ep_moe_bwd_streaming
with FSDP-sharded weights (F/FSDP slice) but does NOT psum d_tok across FSDP.
The comment says "v1: d_tok is partial across D_moe."

This test checks whether d_tok IS partial, and whether summing across FSDP shards
recovers the correct gradient. Runs on a single device — no collectives needed.

If d_tok_shard0 + d_tok_shard1 != d_tok_ref: v1 needs fsdp_psum on d_tok (missing).
If they match: d_tok is somehow correct and the NaN is elsewhere.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import jax
import jax.numpy as jnp

from .backward_kernel import fused_ep_moe_bwd_streaming


def norm_parity(a, b, name="", tol=0.02):
    na = float(jnp.linalg.norm(jnp.asarray(a).reshape(-1)))
    nb = float(jnp.linalg.norm(jnp.asarray(b).reshape(-1)))
    ratio = na / (nb + 1e-12)
    ok = abs(ratio - 1.0) < tol
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}  norm_ratio={ratio:.6f}  "
          f"(test={na:.4f}, ref={nb:.4f})")
    return ok


def run(T=64, D=512, F=256, E=16, K=4, FSDP=2):
    assert F % FSDP == 0
    F_local = F // FSDP

    rng = np.random.default_rng(42)
    tokens = jnp.array(rng.standard_normal((T, D)).astype(np.float32))
    w1     = jnp.array(rng.standard_normal((E, 2, D, F)).astype(np.float32) * 0.02)
    w2     = jnp.array(rng.standard_normal((E, F, D)).astype(np.float32) * 0.02)
    gating = jnp.array(rng.standard_normal((T, E)).astype(np.float32) * 0.1)
    d_out  = jnp.ones((T, D), jnp.float32)

    bwd_kwargs = dict(top_k=K, scoring_fn="sigmoid",
                      renormalize_topk_logits=True, act_fn="silu")

    # ---- Reference: full F weights, FSDP=1 ----
    ref = fused_ep_moe_bwd_streaming(d_out, tokens, w1, w2, gating, **bwd_kwargs)
    d_tok_ref, d_w1_ref, d_w2_ref, d_gate_ref = ref
    print(f"Reference (full F={F}): d_tok norm={float(jnp.linalg.norm(d_tok_ref)):.4f}")

    # ---- FSDP-sharded: split F into FSDP slices ----
    # w1: (E, 2, D, F) → (E, 2, D, F_local) per shard
    # w2: (E, F, D) → (E, F_local, D) per shard
    d_tok_sum  = jnp.zeros_like(d_tok_ref)
    d_gate_sum = jnp.zeros_like(d_gate_ref)
    d_w1_shards = []
    d_w2_shards = []

    for shard in range(FSDP):
        f_start = shard * F_local
        w1_s = w1[:, :, :, f_start:f_start + F_local]      # (E, 2, D, F_local)
        w2_s = w2[:, f_start:f_start + F_local, :]          # (E, F_local, D)

        grads_s = fused_ep_moe_bwd_streaming(
            d_out, tokens, w1_s, w2_s, gating, **bwd_kwargs)
        d_tok_s, d_w1_s, d_w2_s, d_gate_s = grads_s

        d_tok_sum  += d_tok_s
        d_gate_sum += d_gate_s
        d_w1_shards.append(d_w1_s)
        d_w2_shards.append(d_w2_s)
        print(f"  shard {shard}: d_tok norm={float(jnp.linalg.norm(d_tok_s)):.4f}  "
              f"(fraction of ref: {float(jnp.linalg.norm(d_tok_s))/float(jnp.linalg.norm(d_tok_ref)):.3f})")

    # Reconstruct full d_w1/d_w2 from shards for comparison
    d_w1_combined = jnp.concatenate(d_w1_shards, axis=3)   # (E, 2, D, F)
    d_w2_combined = jnp.concatenate(d_w2_shards, axis=1)   # (E, F, D)

    print()
    print(f"=== After summing {FSDP} FSDP shards ===")
    ok_tok  = norm_parity(d_tok_sum,  d_tok_ref,  "d_tok  (sum-of-shards vs full-F ref)")
    ok_gate = norm_parity(d_gate_sum, d_gate_ref, "d_gate (sum-of-shards vs full-F ref)")
    ok_w1   = norm_parity(d_w1_combined, d_w1_ref, "d_w1   (concat-shards vs full-F ref)")
    ok_w2   = norm_parity(d_w2_combined, d_w2_ref, "d_w2   (concat-shards vs full-F ref)")

    print()
    print("=== d_tok without summing (what model.py v1 actually uses) ===")
    # This is what the production code uses — each FSDP device uses its own partial d_tok
    shard0_tok = fused_ep_moe_bwd_streaming(
        d_out, tokens, w1[:, :, :, :F_local], w2[:, :F_local, :], gating, **bwd_kwargs)[0]
    norm_parity(shard0_tok, d_tok_ref, "d_tok  (shard-0 only, NO psum vs full-F ref)")

    return all([ok_tok, ok_gate, ok_w1, ok_w2])


if __name__ == "__main__":
    print(f"JAX: {jax.__version__}  devices: {jax.local_device_count()}")
    print()

    results = {}

    print("=== FSDP=2, D=512, F=256, E=16, K=4, T=64 ===")
    results["fsdp2_small"] = run(T=64, D=512, F=256, E=16, K=4, FSDP=2)

    print()
    print("=== FSDP=4, D=512, F=256, E=16, K=4, T=64 ===")
    results["fsdp4_small"] = run(T=64, D=512, F=256, E=16, K=4, FSDP=4)

    print()
    all_pass = all(results.values())
    print("=" * 60)
    for name, ok in results.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    print("=" * 60)
    print(f"Overall: {'ALL PASSED' if all_pass else 'SOME FAILED'}")

    if not all_pass:
        print()
        print("DIAGNOSIS: d_tok IS partial across F — v1 needs psum(d_tok, 'fsdp')")
        print("in _streaming_bwd_fn (model.py line ~988) after the EP psum.")
