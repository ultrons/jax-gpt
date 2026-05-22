"""DSv3 Fused EP-MoE — pure-JAX reference (math contract + perf baseline).

Implements SPEC v0.4 §3 forward math; SPEC §4 backward is obtained for free via
jax.grad applied to the forward (the SPEC §4 formulas — including the §1'.2
renormalize VJP — are the autodiff-derived gradients of §3, by construction).

This file is the **single source of truth for math correctness** (G2/G3/G4) and
the **lower-bound perf baseline** (SPEC §8.1). It is not sharded and contains no
Pallas; the kernel under test must match this within rtol=1e-2 (forward) /
rtol=5e-2 (backward) per SPEC §11.

Frontmatter:
  slug: dsv3-fused-ep-moe-jax-ref
  intent: math-reference + perf-baseline
  status: v0 — bf16 forward + autodiff backward, single-device
  sources:
    - targets/dsv3-fused-ep-moe/SPEC.md (v0.4 §3 forward math, §4 backward math)
    - corpus/kernels/jax_ref__sparse_moe.py (production reference; vllm/qwix-laden)
    - corpus/kernels/jax_ref__moe_utils.py (sort/permute helpers)
    - corpus/kernels/fused_moe_bwd__backward.py:279-280 (renormalize VJP citation)
  related: targets/dsv3-fused-ep-moe/build/PHASE_A_PLAN.md §6, §9.5
"""
from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class MoEConfig:
    """Static parameters for the MoE block. Values per SPEC §5.3 production defaults."""
    E: int          # global experts
    D: int          # hidden
    F: int          # FFN intermediate
    K: int          # top-k


# Reference is for correctness, not perf. The Phase 3 expert loop unrolls at
# trace time (~30 min compile at production E=256). Hard-cap E for this file.
MAX_REFERENCE_E = 32

# Renormalize divides by `sum(top_w_unnorm)`. With K=1 and bf16 logits, the
# divisor can underflow to 0 -> NaN. K=1 is also outside SPEC §5.3 (K=8). Block.
MIN_REFERENCE_K = 2


# -----------------------------------------------------------------------------
# Forward — SPEC §3
# -----------------------------------------------------------------------------

def _sort_by_expert(expert_ids: jax.Array, K: int) -> tuple[jax.Array, jax.Array]:
    """SPEC §3 steps 1.5-1.6: flatten + stable sort by expert.

    expert_ids: (T, K) int — each row is the K experts a token routes to.
    Returns (sort_idx (T*K,), sorted_eids (T*K,)).
    """
    T = expert_ids.shape[0]
    flat_eids = expert_ids.reshape(-1)                          # (T*K,)
    sort_idx = jnp.argsort(flat_eids, stable=True)
    sorted_eids = flat_eids[sort_idx]
    return sort_idx, sorted_eids


def moe_forward(
    x_in: jax.Array,
    W_gate: jax.Array,
    W1: jax.Array,
    W_d: jax.Array,
    cfg: MoEConfig,
) -> jax.Array:
    """SPEC §3 forward (residual-included).

    Shapes:
      x_in   : (T, D) bf16
      W_gate : (E, D) bf16
      W1     : (E, D, 2F) bf16  — gate+up fused: W1 = concat(W_gate_proj, W_up_proj, axis=-1)
      W_d    : (E, F, D) bf16

    Returns:
      x_out  : (T, D) bf16  — `x_in + moe_residual_contribution` (SPEC §3 step 4.4)
    """
    T, D = x_in.shape
    E, _, twoF = W1.shape
    F = twoF // 2
    assert E == cfg.E and D == cfg.D and F == cfg.F
    assert cfg.E <= MAX_REFERENCE_E, (
        f"reference uses Python for-loop over experts; at E>{MAX_REFERENCE_E} "
        f"trace time is prohibitive (~30 min at production E=256). "
        f"Use the kernel under test for production-scale execution.")
    assert cfg.K >= MIN_REFERENCE_K, (
        f"K=1 makes the renormalize divisor `sum(top_w_unnorm)` equal to a single "
        f"bf16 value that can underflow to 0 -> NaN. SPEC §5.3 fixes K=8.")

    # ---- Phase 1: routing (SPEC §3 steps 1.1-1.4) ----
    # Routing is computed in f32 to match the kernel path: Mosaic doesn't
    # support bf16 argmax (per `iterative-argmax-topk.md`), so the kernel's
    # router casts to f32 internally. Production v1's `get_top_k`
    # (`fused_ep_moe_v1__kernel.py:353`) also casts to f32 before top-K.
    # Without this widening, bf16 score ties get resolved differently between
    # the two paths and a few tokens (~5-15% at E=32 K=4) pick different
    # experts, causing visible drift in the final MoE output.
    gate_logits = x_in.astype(jnp.float32) @ W_gate.astype(jnp.float32).T
    weights = jax.nn.sigmoid(gate_logits)                       # (T, E) f32
    top_w_unnorm, expert_ids = jax.lax.top_k(weights, cfg.K)    # (T, K), (T, K)
    s = top_w_unnorm.sum(axis=-1, keepdims=True)                # (T, 1)
    top_w = top_w_unnorm / s                                    # (T, K) — renormalize

    # ---- Phase 2: pack (SPEC §3 steps 1.5-1.7, 2.1) ----
    sort_idx, sorted_eids = _sort_by_expert(expert_ids, cfg.K)  # (T*K,), (T*K,)
    flat_token_ids = jnp.repeat(jnp.arange(T), cfg.K)           # (T*K,)
    sorted_token_ids = flat_token_ids[sort_idx]                 # (T*K,)
    sorted_w = top_w.reshape(-1)[sort_idx]                      # (T*K,)
    sorted_tokens = x_in[sorted_token_ids]                      # (T*K, D) — gather

    # ---- Phase 3: per-expert FFN (SPEC §3 steps 3.1-3.6) ----
    # Reference uses a Python for-loop over experts (E small at test shapes).
    # Each expert applies FFN to ALL T*K rows and writes the result only into
    # rows assigned to that expert; other rows pass through unchanged.
    #
    # Matmul convention (no transpose) — §5.1's W1: (E, D, 2F), W_d: (E, F, D).
    # See _inbox/blocker-spec-matmul-transpose-nit.md for why §3 text's `.T` is
    # dropped (verified against SPEC v0.4 — §3 still has the `.T`; SPEC v0.5
    # should drop it).
    out_sorted = jnp.zeros_like(sorted_tokens, dtype=jnp.float32)
    for e in range(cfg.E):
        gate_up = sorted_tokens.astype(jnp.float32) @ W1[e].astype(jnp.float32)   # (T*K, 2F)
        gate, up = jnp.split(gate_up, 2, axis=-1)                                  # (T*K, F) each
        act = jax.nn.silu(gate) * up                                               # (T*K, F)
        out_e = act @ W_d[e].astype(jnp.float32)                                   # (T*K, D)
        # jnp.where on the bool mask is more self-documenting than mask*out_e
        # (both are mathematically equivalent here; this is pure JAX, not Pallas
        # body, so antipattern A3 doesn't apply).
        mask_eq_e = (sorted_eids == e)[:, None]                                    # (T*K, 1) bool
        out_sorted = jnp.where(mask_eq_e, out_sorted + out_e, out_sorted)

    # ---- Phase 3 step 3.5: route weight scale ----
    out_sorted = out_sorted * sorted_w[:, None].astype(jnp.float32)                # (T*K, D)

    # ---- Phase 4 step 4.3: unsort + combine via segment_sum (K contributions/token) ----
    moe_out = jax.ops.segment_sum(out_sorted, sorted_token_ids, num_segments=T)    # (T, D)

    # ---- Phase 4 step 4.4: residual add ----
    x_out = x_in + moe_out.astype(x_in.dtype)
    return x_out


def _naive_moe_forward(
    x_in: jax.Array,
    W_gate: jax.Array,
    W1: jax.Array,
    W_d: jax.Array,
    cfg: MoEConfig,
) -> jax.Array:
    """Naive per-token-per-K reference using `jnp.take` + `vmap`. No sort,
    no segment_sum. Same math as `moe_forward`; used purely as a cross-check.

    Slower (`jnp.take` materializes (T, K, D, 2F)) but structurally distinct
    from the sort-based path — agreement between the two confirms the sort
    + segment_sum permutation logic is correct.
    """
    T, D = x_in.shape
    gate_logits = x_in.astype(jnp.float32) @ W_gate.astype(jnp.float32).T
    weights = jax.nn.sigmoid(gate_logits)
    top_w_unnorm, expert_ids = jax.lax.top_k(weights, cfg.K)
    s = top_w_unnorm.sum(axis=-1, keepdims=True)
    top_w = top_w_unnorm / s

    def per_token(token_x, eids, t_weights):
        W1_per_k = jnp.take(W1, eids, axis=0).astype(jnp.float32)         # (K, D, 2F)
        W_d_per_k = jnp.take(W_d, eids, axis=0).astype(jnp.float32)       # (K, F, D)
        gate_up = jnp.einsum("d,kdf->kf",
                             token_x.astype(jnp.float32), W1_per_k)        # (K, 2F)
        gate, up = jnp.split(gate_up, 2, axis=-1)                          # (K, F) each
        act = jax.nn.silu(gate) * up                                       # (K, F)
        out = jnp.einsum("kf,kfd->kd", act, W_d_per_k)                     # (K, D)
        return (out * t_weights[:, None].astype(jnp.float32)).sum(axis=0)   # (D,)

    moe_out = jax.vmap(per_token)(x_in, expert_ids, top_w)                 # (T, D) f32
    return (x_in + moe_out.astype(x_in.dtype))


# -----------------------------------------------------------------------------
# Backward — obtained via jax.grad of moe_forward
# -----------------------------------------------------------------------------
# Per SPEC §4 commentary (especially §1'.2's renormalize VJP citation pointing to
# fused_moe_bwd__backward.py:279-280): the SPEC's backward formulas ARE the
# autograd of the forward. So we expose backward as `jax.grad(loss(forward(...)))`
# rather than a hand-derived custom_vjp. This is the ground truth the kernel's
# custom_vjp must match.


def loss_fn(
    x_in: jax.Array,
    W_gate: jax.Array,
    W1: jax.Array,
    W_d: jax.Array,
    cfg: MoEConfig,
) -> jax.Array:
    """Sum-of-elements loss for grad-check. Used by smoke test + G4."""
    return moe_forward(x_in, W_gate, W1, W_d, cfg).sum()


# argnums=(0,1,2,3) for x_in, W_gate, W1, W_d
moe_grads = jax.jit(jax.grad(loss_fn, argnums=(0, 1, 2, 3)), static_argnums=(4,))


# -----------------------------------------------------------------------------
# Smoke test — run on default device (CPU or TPU); small synthetic inputs.
# -----------------------------------------------------------------------------

def _make_inputs(cfg: MoEConfig, T: int, seed: int = 0):
    """Synthetic small inputs in bf16."""
    key = jax.random.PRNGKey(seed)
    k_x, k_g, k_w1, k_wd = jax.random.split(key, 4)
    x_in   = (jax.random.normal(k_x,  (T, cfg.D))                  * 0.5).astype(jnp.bfloat16)
    W_gate = (jax.random.normal(k_g,  (cfg.E, cfg.D))              * 0.1).astype(jnp.bfloat16)
    W1     = (jax.random.normal(k_w1, (cfg.E, cfg.D, 2*cfg.F))     * 0.05).astype(jnp.bfloat16)
    W_d    = (jax.random.normal(k_wd, (cfg.E, cfg.F, cfg.D))       * 0.05).astype(jnp.bfloat16)
    return x_in, W_gate, W1, W_d


def _cross_check() -> None:
    """Numerical agreement between sort-based `moe_forward` and `_naive_moe_forward`.
    Both implement the same SPEC §3 math via structurally distinct code paths
    (sort+segment_sum vs vmap+take). Disagreement here is a real bug."""
    cfg = MoEConfig(E=8, D=64, F=32, K=2)
    T = 16
    x_in, W_gate, W1, W_d = _make_inputs(cfg, T, seed=0)

    out_sort  = jax.jit(moe_forward,        static_argnums=(4,))(x_in, W_gate, W1, W_d, cfg)
    out_naive = jax.jit(_naive_moe_forward, static_argnums=(4,))(x_in, W_gate, W1, W_d, cfg)

    diff = jnp.abs(out_sort.astype(jnp.float32) - out_naive.astype(jnp.float32))
    max_abs = float(diff.max())
    rel = max_abs / (float(jnp.abs(out_naive.astype(jnp.float32)).max()) + 1e-9)
    # Both are computed in f32 internally and cast back to bf16 at residual add;
    # bit-equivalence isn't expected, but small atol/rtol should hold.
    assert max_abs < 1e-2, f"sort-vs-naive max_abs={max_abs}, rel={rel}"
    print(f"[cross_check] sort-based vs naive agree: max_abs={max_abs:.2e}, rel={rel:.2e}")


def _smoke_test() -> None:
    """Synthetic inputs at small shapes. Verifies:
      - Forward runs, output shape matches input, no NaN.
      - jax.grad runs, all gradient shapes match parameter shapes, no NaN.
      - Renormalize: top_w sums to 1 per row.
      - Top-K: K largest sigmoid scores selected.
      - Routing gradient flows: ‖d_W_gate‖ > 0 (gradient flow check, not the
        analytic VJP identity).
      - Cross-check: sort-based moe_forward agrees with naive vmap path.
    """
    import time

    cfg = MoEConfig(E=8, D=64, F=32, K=2)
    T = 16
    x_in, W_gate, W1, W_d = _make_inputs(cfg, T, seed=0)

    # ---- Forward ----
    t0 = time.perf_counter()
    x_out = jax.jit(moe_forward, static_argnums=(4,))(x_in, W_gate, W1, W_d, cfg)
    x_out.block_until_ready()
    fwd_ms = (time.perf_counter() - t0) * 1000

    assert x_out.shape == x_in.shape, f"shape mismatch: {x_out.shape} vs {x_in.shape}"
    assert x_out.dtype == jnp.bfloat16, f"dtype: {x_out.dtype}"
    assert not jnp.isnan(x_out).any(), "NaN in forward output"
    print(f"[smoke] forward OK: shape={x_out.shape} dtype={x_out.dtype} time={fwd_ms:.1f}ms")

    # ---- Backward (jax.grad) ----
    t0 = time.perf_counter()
    grads = moe_grads(x_in, W_gate, W1, W_d, cfg)
    jax.block_until_ready(grads)
    bwd_ms = (time.perf_counter() - t0) * 1000
    g_x, g_Wg, g_W1, g_Wd = grads

    for name, g, ref in [("d_x_in", g_x, x_in),
                         ("d_W_gate", g_Wg, W_gate),
                         ("d_W1", g_W1, W1),
                         ("d_W_d", g_Wd, W_d)]:
        assert g.shape == ref.shape, f"{name} shape mismatch: {g.shape} vs {ref.shape}"
        assert not jnp.isnan(g).any(), f"NaN in {name}"
    print(f"[smoke] backward (jax.grad) OK: all 4 grads correct shape, no NaN, time={bwd_ms:.1f}ms")

    # ---- Renormalize: top_w sums to 1 per row ----
    gate_logits = x_in.astype(jnp.float32) @ W_gate.astype(jnp.float32).T
    weights = jax.nn.sigmoid(gate_logits)
    top_w_unnorm, _ = jax.lax.top_k(weights, cfg.K)
    top_w = top_w_unnorm / top_w_unnorm.sum(axis=-1, keepdims=True)
    row_sums = top_w.sum(axis=-1)
    assert jnp.allclose(row_sums, 1.0, atol=1e-5), f"renormalize broken: row sums {row_sums}"
    print(f"[smoke] renormalize OK: row sums {float(row_sums.min()):.6f} to {float(row_sums.max()):.6f}")

    # ---- Top-K: K largest sigmoid scores selected ----
    top_K_actual = jnp.sort(weights, axis=-1)[:, -cfg.K:]
    top_K_via_topk = jnp.sort(top_w_unnorm, axis=-1)
    assert jnp.allclose(top_K_actual, top_K_via_topk, rtol=1e-5), "top-k mismatch"
    print(f"[smoke] top-K OK: K={cfg.K} largest scores selected per token")

    # ---- Routing gradient flows (norm > 0) ----
    g_Wg_norm = jnp.linalg.norm(g_Wg.astype(jnp.float32))
    assert float(g_Wg_norm) > 1e-6, f"d_W_gate is zero — gradients aren't flowing through routing"
    print(f"[smoke] d_W_gate norm = {float(g_Wg_norm):.4f} (gradients flow through routing)")

    # ---- Cross-check sort-based vs naive ----
    _cross_check()

    print(f"[smoke] ALL CHECKS PASSED — JAX {jax.__version__} on {jax.devices()[0].platform}")


if __name__ == "__main__":
    _smoke_test()
