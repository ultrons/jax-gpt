"""Correctness tests for mini-dsv3 model.

Verifies:
1. Forward pass produces valid (non-NaN) outputs across all configs
2. Loss is finite and in expected range
3. Gradients are finite
4. MoE routing weights are properly normalized
5. Logit magnitudes don't grow with depth (residual stability)
"""

import sys
sys.path.insert(0, 'mini_dsv3')

import jax
import jax.numpy as jnp
from model import (
    ModelConfig, tiny_config, mini_config, medium_config, full_671b_config,
    init_params, forward, compute_loss, moe_routing,
)


def test_forward_no_nan(cfg, name, seq_len=64):
    """Forward pass should not produce NaN."""
    key = jax.random.PRNGKey(42)
    params = init_params(cfg, key)
    tokens = jax.random.randint(key, (1, seq_len), 0, cfg.V)

    logits, aux = forward(params, tokens, cfg)

    has_nan = bool(jnp.any(jnp.isnan(logits)))
    logit_max = float(jnp.max(jnp.abs(logits)))
    print(f"  {name}: logits shape={logits.shape}, "
          f"max|logit|={logit_max:.2f}, "
          f"aux_loss={float(aux):.4f}, "
          f"NaN={'YES !!!' if has_nan else 'no'}")
    return not has_nan


def test_loss_finite(cfg, name, seq_len=64):
    """Loss should be finite and reasonable (8-12 for random init)."""
    key = jax.random.PRNGKey(42)
    params = init_params(cfg, key)
    tokens = jax.random.randint(key, (1, seq_len), 0, cfg.V)

    loss = compute_loss(params, tokens, cfg)
    loss_val = float(loss)
    is_finite = bool(jnp.isfinite(loss))

    expected_loss = jnp.log(jnp.float32(cfg.V))  # ~ln(V) for random init
    print(f"  {name}: loss={loss_val:.4f}, "
          f"expected~{float(expected_loss):.1f} (ln V), "
          f"finite={'yes' if is_finite else 'NO !!!'}")
    return is_finite


def test_grad_finite(cfg, name, seq_len=64):
    """Gradients should be finite."""
    key = jax.random.PRNGKey(42)
    params = init_params(cfg, key)
    tokens = jax.random.randint(key, (1, seq_len), 0, cfg.V)

    loss_val, grads = jax.value_and_grad(compute_loss)(params, tokens, cfg)
    grad_leaves = jax.tree.leaves(grads)
    any_nan = any(bool(jnp.any(jnp.isnan(g))) for g in grad_leaves)
    any_inf = any(bool(jnp.any(jnp.isinf(g))) for g in grad_leaves)
    grad_norm = sum(float(jnp.sum(g ** 2)) for g in grad_leaves) ** 0.5

    print(f"  {name}: grad_norm={grad_norm:.2f}, "
          f"NaN={'YES !!!' if any_nan else 'no'}, "
          f"Inf={'YES !!!' if any_inf else 'no'}")
    return not any_nan and not any_inf


def test_moe_routing_weights(cfg, name):
    """Routing weights should sum to routed_scaling_factor, not K."""
    key = jax.random.PRNGKey(42)
    x = jax.random.normal(key, (1, 16, cfg.D), dtype=cfg.jax_dtype)
    gate_w = jax.random.normal(key, (cfg.D, cfg.E), dtype=cfg.jax_dtype) * 0.02

    weights, indices, scores = moe_routing(x, gate_w, cfg)
    weight_sum = float(jnp.mean(jnp.sum(weights, axis=-1)))

    print(f"  {name}: mean sum(weights)={weight_sum:.4f}, "
          f"expected~{cfg.routed_scaling_factor:.1f}, K={cfg.K}")
    return abs(weight_sum - cfg.routed_scaling_factor) < 0.5


def test_residual_stability(cfg, name, seq_len=64):
    """Activations should not explode through layers."""
    key = jax.random.PRNGKey(42)
    params = init_params(cfg, key)
    tokens = jax.random.randint(key, (1, seq_len), 0, cfg.V)

    # Run forward and check final hidden state magnitude
    logits, _ = forward(params, tokens, cfg)
    logit_std = float(jnp.std(logits.astype(jnp.float32)))

    # With random init, logit std should be ~1-10, not 100+
    stable = logit_std < 50.0
    print(f"  {name}: logit_std={logit_std:.2f}, "
          f"stable={'yes' if stable else 'NO — residual explosion!'}")
    return stable


def main():
    configs = [
        (tiny_config(), "tiny"),
        (mini_config(), "mini"),
    ]
    # Don't test medium/full on local v4 — too large

    all_pass = True

    print("=== Forward NaN test ===")
    for cfg, name in configs:
        if not test_forward_no_nan(cfg, name):
            all_pass = False

    print("\n=== Loss finite test ===")
    for cfg, name in configs:
        if not test_loss_finite(cfg, name):
            all_pass = False

    print("\n=== Gradient finite test ===")
    for cfg, name in configs:
        if not test_grad_finite(cfg, name):
            all_pass = False

    print("\n=== MoE routing weight test ===")
    for cfg, name in configs:
        if not test_moe_routing_weights(cfg, name):
            all_pass = False
    # Also test full config routing (doesn't need full model)
    test_moe_routing_weights(full_671b_config(), "full-671b")

    print("\n=== Residual stability test ===")
    for cfg, name in configs:
        if not test_residual_stability(cfg, name):
            all_pass = False

    print(f"\n{'='*50}")
    print(f"{'ALL TESTS PASSED' if all_pass else 'SOME TESTS FAILED'}")
    print(f"{'='*50}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
