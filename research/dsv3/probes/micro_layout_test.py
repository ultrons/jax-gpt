#!/usr/bin/env python3
"""Micro MoE layout experiment — test T(8,128) padding with Layout API.

Run on 4 local TPU v4 chips:
  source ~/xdb/.xprof/bin/activate && python micro_layout_test.py

Tests:
  1. Baseline: F_local=8 as minor dim → expect 16× padding
  2. Transpose: store wi as (E, F, D) → D as minor → expect 0 padding
  3. Layout API: explicit major_to_minor → force D as minor
  4. Check if XLA respects or overrides our layout choice

Shapes chosen to match production padding scenario:
  E=8 experts, D=256 (model), D_moe=16 (hidden), FSDP=2, TP=2
  → F_local = D_moe/FSDP = 8 (same as production 2048/256)
  → D_local = D/TP = 128 (clean multiple of 128)
"""

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from jax.experimental.shard_map import shard_map
import numpy as np
import functools

# Config — mimics production F_local=8 padding scenario
E = 256         # experts (production scale)
D = 256         # model dim (D/TP=128, multiple of 128 → clean tiling)
D_moe = 16      # FFN hidden (D_moe/FSDP=8 → F_local=8 → padding trigger)
K = 2           # top-k
L = 3           # layers
B = 2           # batch (small to avoid OOM)
S = 16          # seq len (small to avoid OOM)

print("=" * 70)
print("SETUP")
print("=" * 70)
devices = jax.devices()
n_dev = len(devices)
print(f"Devices: {n_dev} x {devices[0].device_kind}")

# Mesh: (fsdp=2, tp=2) if 4 devices, else adjust
assert n_dev >= 4, f"Need >= 4 devices, got {n_dev}"
fsdp = 2
tp = 2
mesh_devs = np.array(devices[:fsdp * tp]).reshape(fsdp, tp)
mesh = Mesh(mesh_devs, ("fsdp", "tp"))
print(f"Mesh: fsdp={fsdp}, tp={tp}")
print(f"F_local = D_moe/fsdp = {D_moe}/{fsdp} = {D_moe // fsdp}")
print(f"D_local = D/tp = {D}/{tp} = {D // tp}")
print()


def make_weights(key, shape, spec):
    """Create sharded random weights."""
    sharding = NamedSharding(mesh, spec)
    w = jax.random.normal(key, shape, dtype=jnp.bfloat16) * 0.01
    return jax.device_put(w, sharding)


def get_hlo_layouts(fn, *args):
    """Extract tensor layouts from compiled HLO."""
    lowered = jax.jit(fn).lower(*args)
    compiled = lowered.compile()
    hlo_text = compiled.as_text()
    # Find lines with our target shapes and T(8,128)
    lines = hlo_text.split('\n')
    padding_lines = []
    for line in lines:
        if 'T(8,128)' in line and any(f'{D_moe//fsdp}' in line for _ in [1]):
            if 'bf16[' in line:
                padding_lines.append(line.strip())
    return hlo_text, padding_lines


def simple_moe_body(flat_x, wi_0, wo, reduce_axis):
    """Minimal MoE body: just matmul, no routing (tests layout only)."""
    # Gate: (T, D_local) × (E, D_local, F_local) → (T, F_local)
    # Simplified: just use first expert for testing
    out = jnp.einsum('td,df->tf', flat_x, wi_0[0])
    out = jax.nn.silu(out)
    # Down: (T, F_local) × (E, F_local, D_local) → (T, D_local)
    out = jnp.einsum('tf,fd->td', out, wo[0])
    return jax.lax.psum(out, reduce_axis)


# ============================================================================
print("=" * 70)
print("TEST 1: Baseline — wi as (E, D, F), F_local=8 as minor dim")
print("=" * 70)

key = jax.random.PRNGKey(0)
k1, k2, k3 = jax.random.split(key, 3)

# wi: (E, D, D_moe) with P("ep_or_none", "tp", "fsdp") → per-dev (E, D/tp, F_local)
wi_baseline = make_weights(k1, (E, D, D_moe), P(None, "tp", "fsdp"))
wo_baseline = make_weights(k2, (E, D_moe, D), P(None, "fsdp", "tp"))
x = make_weights(k3, (B * S, D), P("fsdp", "tp"))

print(f"wi shape: {wi_baseline.shape}, spec: P(None, 'tp', 'fsdp')")
print(f"  per-device: ({E}, {D//tp}, {D_moe//fsdp}) = ({E}, {D//tp}, {D_moe//fsdp})")
print(f"  Expected: dim2={D_moe//fsdp} as minor → pads to 128")

@functools.partial(shard_map, mesh=mesh,
                   in_specs=(P("fsdp", "tp"), P(None, "tp", "fsdp"), P(None, "fsdp", "tp")),
                   out_specs=P("fsdp", "tp"), check_rep=False)
def baseline_fn(x_, wi_, wo_):
    return simple_moe_body(x_, wi_, wo_, "fsdp")

def baseline_grad(x, wi, wo):
    def loss_fn(wi, wo):
        return jnp.sum(baseline_fn(x, wi, wo))
    return jax.value_and_grad(loss_fn, argnums=(0, 1))(wi, wo)

hlo, pad_lines = get_hlo_layouts(baseline_grad, x, wi_baseline, wo_baseline)
# Count padding references
f_local = D_moe // fsdp
pad_count = hlo.count(f'T(8,128)')
print(f"  HLO T(8,128) references: {pad_count}")
# Check for the specific padded shape
padded_shape = f"{E},{D//tp},{f_local}"  # e.g. "8,128,8"
stacked_padded = f"{L},{E},{D//tp},{f_local}"
print(f"  Searching for shape [{padded_shape}] with T(8,128)...")
for line in pad_lines[:5]:
    print(f"    {line[:120]}")
print()


# ============================================================================
print("=" * 70)
print("TEST 2: Transpose — wi as (E, F, D), D as minor dim")
print("=" * 70)

# wi: (E, D_moe, D) with P(None, "fsdp", "tp") → per-dev (E, F_local, D/tp)
wi_transposed = make_weights(k1, (E, D_moe, D), P(None, "fsdp", "tp"))
wo_transposed = make_weights(k2, (E, D_moe, D), P(None, "fsdp", "tp"))

print(f"wi shape: {wi_transposed.shape}, spec: P(None, 'fsdp', 'tp')")
print(f"  per-device: ({E}, {D_moe//fsdp}, {D//tp}) = ({E}, {D_moe//fsdp}, {D//tp})")
print(f"  Expected: dim2={D//tp} as minor → {D//tp}%128=0 → zero pad")

@functools.partial(shard_map, mesh=mesh,
                   in_specs=(P("fsdp", "tp"), P(None, "fsdp", "tp"), P(None, "fsdp", "tp")),
                   out_specs=P("fsdp", "tp"), check_rep=False)
def transpose_fn(x_, wi_, wo_):
    # Transpose wi from (E, F, D) to (E, D, F) for matmul
    wi_t = wi_.transpose(0, 2, 1)
    return simple_moe_body(x_, wi_t, wo_, "fsdp")

def transpose_grad(x, wi, wo):
    def loss_fn(wi, wo):
        return jnp.sum(transpose_fn(x, wi, wo))
    return jax.value_and_grad(loss_fn, argnums=(0, 1))(wi, wo)

hlo2, pad_lines2 = get_hlo_layouts(transpose_grad, x, wi_transposed, wo_transposed)
pad_count2 = hlo2.count('T(8,128)')
print(f"  HLO T(8,128) references: {pad_count2}")
for line in pad_lines2[:5]:
    print(f"    {line[:120]}")
print()


# ============================================================================
print("=" * 70)
print("TEST 3: Layout API — explicit major_to_minor on baseline shape")
print("=" * 70)

try:
    from jax._src.layout import Layout, Format

    # Apply Layout constraint: force D as minor dim on wi (E, D, F) shape
    # Per-device: (E, D/tp, F_local) — want dim1 (D) minor, dim2 (F) second
    _wi_fmt = Format(Layout(major_to_minor=(0, 2, 1)),
                     NamedSharding(mesh, P(None, "tp", "fsdp")))

    @jax.jit
    def layout_grad(x, wi, wo):
        # Apply layout constraint
        wi = jax.lax.with_sharding_constraint(wi, _wi_fmt)
        def loss_fn(wi, wo):
            return jnp.sum(baseline_fn(x, wi, wo))
        return jax.value_and_grad(loss_fn, argnums=(0, 1))(wi, wo)

    hlo3, pad_lines3 = get_hlo_layouts(layout_grad, x, wi_baseline, wo_baseline)
    pad_count3 = hlo3.count('T(8,128)')
    print(f"  HLO T(8,128) references: {pad_count3}")
    for line in pad_lines3[:5]:
        print(f"    {line[:120]}")

    # Check if XLA respected our layout
    # Search for the minor_to_major we requested
    if '{1,2,0}' in hlo3 or '{0,2,1}' in hlo3:
        print("  ✓ Layout API: XLA appears to use our requested layout")
    else:
        print("  ✗ Layout API: XLA may have overridden our layout")

except ImportError as e:
    print(f"  Layout API not available: {e}")
except Exception as e:
    print(f"  Layout API failed: {e}")

print()


# ============================================================================
print("=" * 70)
print("TEST 4: Stacked params (L layers) — check padding on scan params")
print("=" * 70)

# Stacked wi: (L, E, D, D_moe) → per-dev (L, E, D/tp, F_local)
wi_stacked = make_weights(k1, (L, E, D, D_moe), P(None, None, "tp", "fsdp"))
print(f"Stacked wi shape: {wi_stacked.shape}, per-dev: ({L}, {E}, {D//tp}, {D_moe//fsdp})")

# Stacked transposed: (L, E, D_moe, D) → per-dev (L, E, F_local, D/tp)
wi_stacked_t = make_weights(k1, (L, E, D_moe, D), P(None, None, "fsdp", "tp"))
print(f"Stacked transposed: {wi_stacked_t.shape}, per-dev: ({L}, {E}, {D_moe//fsdp}, {D//tp})")

@jax.jit
def scan_baseline(x, wi_stk, wo_stk):
    def body(x, params):
        wi, wo = params
        @functools.partial(shard_map, mesh=mesh,
                           in_specs=(P("fsdp", "tp"), P(None, "tp", "fsdp"), P(None, "fsdp", "tp")),
                           out_specs=P("fsdp", "tp"), check_rep=False)
        def _fn(x_, wi_, wo_):
            return simple_moe_body(x_, wi_, wo_, "fsdp")
        return x + _fn(x, wi, wo), None
    x, _ = jax.lax.scan(body, x, (wi_stk, wo_stk))
    return jnp.sum(x)

wo_stacked = make_weights(k2, (L, E, D_moe, D), P(None, None, "fsdp", "tp"))

def scan_baseline_grad(x, wi, wo):
    return jax.value_and_grad(scan_baseline, argnums=(1,))(x, wi, wo)

hlo4, _ = get_hlo_layouts(scan_baseline_grad, x, wi_stacked, wo_stacked)

# Find the stacked shape in HLO to see its layout
for line in hlo4.split('\n'):
    if f'{L},{E},' in line and 'T(8,128)' in line and 'bf16' in line:
        # Extract just the shape and layout part
        for part in line.split():
            if 'bf16[' in part:
                print(f"  Stacked baseline: {part[:80]}")
                break

# Now with transposed storage
wo_stacked_t = make_weights(k2, (L, E, D_moe, D), P(None, None, "fsdp", "tp"))

@jax.jit
def scan_transposed(x, wi_stk, wo_stk):
    def body(x, params):
        wi, wo = params
        @functools.partial(shard_map, mesh=mesh,
                           in_specs=(P("fsdp", "tp"), P(None, "fsdp", "tp"), P(None, "fsdp", "tp")),
                           out_specs=P("fsdp", "tp"), check_rep=False)
        def _fn(x_, wi_, wo_):
            wi_t = wi_.transpose(0, 2, 1)
            return simple_moe_body(x_, wi_t, wo_, "fsdp")
        return x + _fn(x, wi, wo), None
    x, _ = jax.lax.scan(body, x, (wi_stk, wo_stk))
    return jnp.sum(x)

def scan_transposed_grad(x, wi, wo):
    return jax.value_and_grad(scan_transposed, argnums=(1,))(x, wi, wo)

hlo5, _ = get_hlo_layouts(scan_transposed_grad, x, wi_stacked_t, wo_stacked_t)

for line in hlo5.split('\n'):
    if f'{L},{E},' in line and 'T(8,128)' in line and 'bf16' in line:
        for part in line.split():
            if 'bf16[' in part:
                print(f"  Stacked transposed: {part[:80]}")
                break

print()

# ============================================================================
print("=" * 70)
print("TEST 5: Layout API on stacked params before scan")
print("=" * 70)

try:
    from jax._src.layout import Layout, Format

    _stk_fmt = Format(Layout(major_to_minor=(0, 1, 3, 2)),
                      NamedSharding(mesh, P(None, None, "tp", "fsdp")))

    @jax.jit
    def scan_layout(x, wi_stk, wo_stk):
        # Apply layout on stacked params before scan
        wi_stk = jax.lax.with_sharding_constraint(wi_stk, _stk_fmt)
        def body(x, params):
            wi, wo = params
            @functools.partial(shard_map, mesh=mesh,
                               in_specs=(P("fsdp", "tp"), P(None, "tp", "fsdp"), P(None, "fsdp", "tp")),
                               out_specs=P("fsdp", "tp"), check_rep=False)
            def _fn(x_, wi_, wo_):
                return simple_moe_body(x_, wi_, wo_, "fsdp")
            return x + _fn(x, wi, wo), None
        x, _ = jax.lax.scan(body, x, (wi_stk, wo_stk))
        return jnp.sum(x)

    def scan_layout_grad(x, wi, wo):
        return jax.value_and_grad(scan_layout, argnums=(1,))(x, wi, wo)

    hlo6, _ = get_hlo_layouts(scan_layout_grad, x, wi_stacked, wo_stacked)

    for line in hlo6.split('\n'):
        if f'{L},{E},' in line and 'T(8,128)' in line and 'bf16' in line:
            for part in line.split():
                if 'bf16[' in part:
                    print(f"  Stacked w/ Layout API: {part[:80]}")
                    break

except Exception as e:
    print(f"  Layout API on stacked failed: {e}")

print()
print("=" * 70)
print("DONE — compare T(8,128) padding across tests")
print("=" * 70)
