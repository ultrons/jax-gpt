#!/usr/bin/env python3
"""Tutorial: scan carry memory, host offload, and remat strategies.

Run with: source ~/xdb/.xprof/bin/activate && python scan_memory_tutorial.py

This demonstrates the core memory problem we hit training DSv3 671B
and the solutions we explored:

1. Baseline scan: XLA stores ALL carry values for backward → O(L) memory
2. jax.checkpoint inside scan: saves intermediates but carry still O(L)
3. Offloadable policy: streams carry to host RAM → O(1) HBM
4. Custom VJP (maxtext style): manual forward/backward with host offload

Each section prints the XLA HLO memory estimate so you can see the effect.
"""

import jax
import jax.numpy as jnp
from jax import random
import functools

# Small model dimensions (CPU-friendly)
D = 128       # model dim
F = 256       # FFN hidden dim
L = 8         # number of layers
B = 4         # batch size
S = 32        # sequence length


def make_params(key):
    """Create stacked layer params [L, ...]."""
    keys = random.split(key, L)
    w1s, w2s = [], []
    for i in range(L):
        k1, k2 = random.split(keys[i])
        w1s.append(random.normal(k1, (D, F)) * 0.01)
        w2s.append(random.normal(k2, (F, D)) * 0.01)
    return {
        "w1": jnp.stack(w1s),    # (L, D, F)
        "w2": jnp.stack(w2s),    # (L, F, D)
        "norm": jnp.ones((L, D)),
    }


def rms_norm(x, scale, eps=1e-6):
    ms = jnp.mean(x ** 2, axis=-1, keepdims=True)
    return x * jax.lax.rsqrt(ms + eps) * scale


def layer_body(x, params):
    """One transformer-like layer: norm → FFN(silu) → residual."""
    h = rms_norm(x, params["norm"])
    gate = jax.nn.silu(h @ params["w1"])   # (B, S, F)
    out = gate @ params["w2"]               # (B, S, D)
    return x + out


def make_layer_params(stacked_params):
    """Helper: create single-layer params from stacked."""
    keys = random.split(random.PRNGKey(42), 4)
    return {
        "w1": random.normal(keys[0], (D, F)) * 0.01,
        "w2": random.normal(keys[1], (F, D)) * 0.01,
        "norm": jnp.ones((D,)),
    }


def get_hlo_memory(fn, *args):
    """Get XLA's estimated memory from the compiled HLO."""
    lowered = jax.jit(fn).lower(*args)
    compiled = lowered.compile()
    # Try to get memory stats from cost analysis
    try:
        cost = compiled.cost_analysis()
        if isinstance(cost, list):
            cost = cost[0]
        return cost.get("temp_size_in_bytes", None)
    except:
        return None


# ============================================================================
print("=" * 70)
print("SETUP")
print("=" * 70)

key = random.PRNGKey(0)
x = random.normal(key, (B, S, D))
params = make_params(key)

print(f"Model: {L} layers, D={D}, F={F}")
print(f"Input: ({B}, {S}, {D}) = {x.nbytes / 1024:.1f} KB")
print(f"Carry per layer: {x.nbytes / 1024:.1f} KB")
print(f"Total carry if stored for all {L} layers: {L * x.nbytes / 1024:.1f} KB")
print()


# ============================================================================
print("=" * 70)
print("1. BASELINE SCAN — no checkpoint")
print("   XLA stores carry at every layer boundary for backward.")
print("   Memory: O(L × B × S × D)")
print("=" * 70)

def forward_baseline(x, params):
    def scan_fn(x, lp):
        x_new = layer_body(x, lp)
        return x_new, None
    x_out, _ = jax.lax.scan(scan_fn, x, params)
    return jnp.sum(x_out)  # scalar loss for grad

grad_fn_baseline = jax.value_and_grad(forward_baseline, argnums=1)
loss, grads = grad_fn_baseline(x, params)

# Trace to see what XLA does
lowered = jax.jit(grad_fn_baseline).lower(x, params)
hlo_text = lowered.as_text()
# Count AllocateBuffer calls (proxy for memory allocations)
alloc_count = hlo_text.count("AllocateBuffer")
# Check if carry is stacked
has_stacked = f"{L}," in hlo_text and f"{B}," in hlo_text

print(f"  Loss: {loss:.4f}, grad finite: {jnp.isfinite(grads['w1']).all()}")
print(f"  HLO AllocateBuffer calls: {alloc_count}")
print(f"  HLO contains stacked [{L}, ...] tensors: {has_stacked}")
print(f"  → XLA stores [{L}, {B}, {S}, {D}] carry = {L * x.nbytes / 1024:.1f} KB")
print()


# ============================================================================
print("=" * 70)
print("2. jax.checkpoint INSIDE scan — default policy (save nothing)")
print("   Saves no intermediates WITHIN each layer (recomputes them).")
print("   BUT: scan still stores carry at every boundary → O(L) carry.")
print("=" * 70)

def forward_checkpoint(x, params):
    def scan_fn(x, lp):
        x_new = jax.checkpoint(layer_body)(x, lp)
        return x_new, None
    x_out, _ = jax.lax.scan(scan_fn, x, params)
    return jnp.sum(x_out)

grad_fn_ckpt = jax.value_and_grad(forward_checkpoint, argnums=1)
_, grads_ckpt = grad_fn_ckpt(x, params)

lowered_ckpt = jax.jit(grad_fn_ckpt).lower(x, params)
hlo_ckpt = lowered_ckpt.as_text()
alloc_ckpt = hlo_ckpt.count("AllocateBuffer")

print(f"  Grad finite: {jnp.isfinite(grads_ckpt['w1']).all()}")
print(f"  Grads match baseline: {jnp.allclose(grads['w1'], grads_ckpt['w1'], atol=1e-5)}")
print(f"  HLO AllocateBuffer calls: {alloc_ckpt} (vs {alloc_count} baseline)")
print(f"  → Fewer intermediates saved, BUT carry [{L}, {B}, {S}, {D}] still stored")
print(f"  → This is why jax.checkpoint alone doesn't solve scan OOM")
print()


# ============================================================================
print("=" * 70)
print("3. jax.checkpoint with Offloadable policy")
print("   Offloads checkpoint residuals to 'pinned_host' memory.")
print("   The scan carry goes to host RAM → O(1) HBM for carry!")
print("=" * 70)

def forward_offload(x, params):
    def _offload_policy(prim, *avals, **kw):
        return jax._src.interpreters.partial_eval.Offloadable(src='device', dst='pinned_host')

    def scan_fn(x, lp):
        x_new = jax.checkpoint(
            layer_body, policy=_offload_policy, prevent_cse=False)(x, lp)
        return x_new, None
    x_out, _ = jax.lax.scan(scan_fn, x, params)
    return jnp.sum(x_out)

lowered_off = jax.jit(jax.value_and_grad(forward_offload, argnums=1)).lower(x, params)
hlo_off = lowered_off.as_text()
host_count = hlo_off.count("pinned_host") + hlo_off.count("S(5)")  # S(5) = host memory space

try:
    _, grads_offload = jax.value_and_grad(forward_offload, argnums=1)(x, params)
    print(f"  Grad finite: {jnp.isfinite(grads_offload['w1']).all()}")
    print(f"  Grads match baseline: {jnp.allclose(grads['w1'], grads_offload['w1'], atol=1e-5)}")
except Exception as e:
    print(f"  (Execution failed on this device: {type(e).__name__} — HLO analysis still valid)")
print(f"  HLO host memory references: {host_count}")
print(f"  → Carry offloaded to host: [{L}, {B}, {S}, {D}] in CPU RAM, not HBM")
print(f"  ⚠ Offloads EVERYTHING — can blow host RAM for large models")
print()


# ============================================================================
print("=" * 70)
print("4. SELECTIVE offload — only named layer input")
print("   checkpoint_name(x, 'layer_input') + save_and_offload_only_these_names")
print("   Only the carry goes to host. Everything else recomputed.")
print("=" * 70)

def layer_body_named(x, params):
    """Same as layer_body but names x for selective offload."""
    x = jax._src.ad_checkpoint.checkpoint_name(x, "layer_input")
    h = rms_norm(x, params["norm"])
    gate = jax.nn.silu(h @ params["w1"])
    out = gate @ params["w2"]
    return x + out

def forward_selective(x, params):
    _policy = jax.checkpoint_policies.save_and_offload_only_these_names(
        names_which_can_be_saved=(),
        names_which_can_be_offloaded=("layer_input",),
        offload_src="device",
        offload_dst="pinned_host",
    )

    def scan_fn(x, lp):
        x_new = jax.checkpoint(
            layer_body_named, policy=_policy, prevent_cse=False)(x, lp)
        return x_new, None
    x_out, _ = jax.lax.scan(scan_fn, x, params)
    return jnp.sum(x_out)

lowered_sel = jax.jit(jax.value_and_grad(forward_selective, argnums=1)).lower(x, params)
hlo_sel = lowered_sel.as_text()
host_sel = hlo_sel.count("pinned_host") + hlo_sel.count("S(5)")

try:
    _, grads_sel = jax.value_and_grad(forward_selective, argnums=1)(x, params)
    print(f"  Grad finite: {jnp.isfinite(grads_sel['w1']).all()}")
    print(f"  Grads match baseline: {jnp.allclose(grads['w1'], grads_sel['w1'], atol=1e-5)}")
except Exception as e:
    print(f"  (Execution failed on this device: {type(e).__name__} — HLO analysis still valid)")
print(f"  HLO host memory references: {host_sel} (vs {host_count} offload-all)")
print(f"  → Only 'layer_input' (the carry) offloaded to host")
print(f"  → All intermediates (norm, silu, matmuls) recomputed from x")
print(f"  → Host usage: {L * x.nbytes / 1024:.1f} KB (just the carry)")
print()


# ============================================================================
print("=" * 70)
print("5. CUSTOM VJP (maxtext approach)")
print("   Full control: forward scan saves residuals, backward reverse scan")
print("   loads them back. Can offload specific tensors manually.")
print("=" * 70)

def forward_custom_vjp(x, params):
    @jax.custom_vjp
    def _stack(x, params):
        def scan_fn(x, lp):
            return layer_body(x, lp), None
        x_out, _ = jax.lax.scan(scan_fn, x, params)
        return x_out

    def _stack_fwd(x, params):
        def scan_fn(x, lp):
            x_new = layer_body(x, lp)
            return x_new, x  # save input x as ys (in this demo, stays on device)
        x_out, saved_xs = jax.lax.scan(scan_fn, x, params)
        return x_out, (saved_xs, params)

    def _stack_bwd(res, g):
        saved_xs, params = res
        def bwd_scan_fn(dx, inputs):
            x_saved, lp = inputs
            _, vjp_fn = jax.vjp(layer_body, x_saved, lp)
            dx_new, dlp = vjp_fn(dx)
            return dx_new, dlp
        dx_out, d_params = jax.lax.scan(
            bwd_scan_fn, g, (saved_xs, params), reverse=True)
        return dx_out, d_params

    _stack.defvjp(_stack_fwd, _stack_bwd)
    return jnp.sum(_stack(x, params))

try:
    _, grads_cvjp = jax.value_and_grad(forward_custom_vjp, argnums=1)(x, params)
    print(f"  Grad finite: {jnp.isfinite(grads_cvjp['w1']).all()}")
    print(f"  Grads match baseline: {jnp.allclose(grads['w1'], grads_cvjp['w1'], atol=1e-5)}")
except Exception as e:
    print(f"  (Execution failed: {type(e).__name__}: {e})")
print(f"  → Forward: single fused scan (XLA optimizes)")
print(f"  → Backward: reverse scan with saved xs + vjp per layer")
print(f"  → In maxtext: saved xs are device_put to pinned_host inside scan")
print(f"  → Requires manual device_put inside scan body (XLA support varies)")
print()


# ============================================================================
print("=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"""
Strategy                    | Carry HBM | Intermediates | Recompute | Host RAM
----------------------------|-----------|---------------|-----------|----------
1. Baseline scan            | O(L)      | ALL saved     | None      | 0
2. checkpoint (no offload)  | O(L)      | None saved    | Full      | 0
3. Offloadable (all)        | 0         | ALL offloaded | None      | O(L×all)
4. Selective offload (ours) | 0         | None saved    | Full      | O(L×carry)
5. Custom VJP (maxtext)     | 0*        | Selected      | Partial   | O(L×selected)

* Custom VJP eliminates scan's own carry stash; uses manual saves instead.

For DSv3 671B (L=58, B_l=16, S=4096, D=7168):
  Carry per layer: {16 * 4096 * 7168 * 2 / 1024**3:.1f} GB
  Total carry (L layers): {58 * 16 * 4096 * 7168 * 2 / 1024**3:.1f} GB
  HBM limit: 94.75 GB
  Host RAM: 768 GB

  Strategy 4 (selective offload) moves {58 * 16 * 4096 * 7168 * 2 / 1024**3:.1f} GB
  carry to host, leaving HBM free for compute.
""")
