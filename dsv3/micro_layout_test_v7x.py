#!/usr/bin/env python3
"""AOT layout test targeting v7x — check if XLA respects transpose layout on v7x backend.

Run with: source ~/xdb/.xprof/bin/activate && python micro_layout_test_v7x.py

Uses jax.experimental.topologies to compile for v7x without hardware.
Compares baseline (E,D,F) vs transpose (E,F,D) layouts in the HLO.
"""

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from jax.experimental.shard_map import shard_map
from jax.experimental import topologies
import numpy as np
import functools

# Production-scale shapes (per-device after sharding)
E = 256         # experts
D = 7168        # model dim
D_moe = 2048    # FFN hidden
K = 8           # top-k
L = 3           # layers (small for compile speed)
B = 2048        # batch (global — needs B*S/FSDP >= E for ragged_dot)
S = 128         # seq

# Target: FSDP=256, TP=2 on v7x 4x8x8
FSDP = 256
TP = 2

print("=" * 70)
print("AOT v7x LAYOUT TEST")
print("=" * 70)

# Get v7x topology — 4x4x4 = 64 chips × 2 TCs = 128 devices minimum
# We need fsdp=256, tp=2 = 512 devices → 4x8x8
topo = topologies.get_topology_desc("tpu7x:4x8x8", platform="tpu")
devices = topo.devices
n_dev = len(devices)
print(f"Virtual devices: {n_dev} x v7x")

mesh_devs = np.array(devices[:FSDP * TP]).reshape(FSDP, TP)
mesh = Mesh(mesh_devs, ("fsdp", "tp"))
print(f"Mesh: fsdp={FSDP}, tp={TP}")
print(f"F_local = D_moe/FSDP = {D_moe}/{FSDP} = {D_moe // FSDP}")
print(f"D_local = D/TP = {D}/{TP} = {D // TP}")
print()

F_local = D_moe // FSDP  # 8
D_local = D // TP         # 3584


def simple_moe_body(flat_x, wi_0, wo, reduce_axis):
    """MoE body using ragged_dot — matches production layout behavior."""
    T_local, D_local = flat_x.shape
    E_local = wi_0.shape[0]
    F_local = wi_0.shape[2]

    # Uniform group sizes: tokens split evenly across experts
    tokens_per_expert = T_local // E_local
    group_sizes = jnp.full((E_local,), tokens_per_expert, dtype=jnp.int32)

    # Extend with zero dummy expert (production pattern)
    zero_wi = jnp.zeros((1, D_local, F_local), dtype=wi_0.dtype)
    zero_wo = jnp.zeros((1, F_local, D_local), dtype=wo.dtype)
    wi_ext = jnp.concatenate([wi_0, zero_wi], axis=0)
    wo_ext = jnp.concatenate([wo, zero_wo], axis=0)
    group_sizes_ext = jnp.concatenate([group_sizes, jnp.zeros(1, dtype=jnp.int32)])

    # Gate via ragged_dot: (T, D) × (E+1, D, F) → (T, F)
    gate = jax.nn.silu(jax.lax.ragged_dot(
        flat_x.astype(wi_ext.dtype), wi_ext, group_sizes_ext))
    # Down via ragged_dot: (T, F) × (E+1, F, D) → (T, D)
    out = jax.lax.ragged_dot(
        gate.astype(wo_ext.dtype), wo_ext, group_sizes_ext)
    return jax.lax.psum(out, reduce_axis)


def extract_layouts(hlo_text, shape_fragment):
    """Find all HLO lines matching a shape fragment and extract layout info."""
    results = []
    for line in hlo_text.split('\n'):
        if shape_fragment in line and 'T(8,128)' in line and 'bf16[' in line:
            # Extract the shape+layout part
            for part in line.split():
                if 'bf16[' in part and 'T(8,128)' in part:
                    results.append(part.rstrip(','))
                    break
    return results


# ============================================================================
print("=" * 70)
print("TEST A: Baseline — wi stored as (E, D, D_moe)")
print(f"  Spec P(None, 'tp', 'fsdp') → per-dev ({E}, {D_local}, {F_local})")
print("=" * 70)

wi_spec_baseline = P(None, "tp", "fsdp")
wo_spec_baseline = P(None, "fsdp", "tp")

@functools.partial(shard_map, mesh=mesh,
                   in_specs=(P("fsdp", "tp"), wi_spec_baseline, wo_spec_baseline),
                   out_specs=P("fsdp", "tp"), check_rep=False)
def baseline_body(x_, wi_, wo_):
    return simple_moe_body(x_, wi_, wo_, "fsdp")

def baseline_scan(x, wi_stk, wo_stk):
    def body(x, params):
        wi, wo = params
        return x + baseline_body(x, wi, wo), None
    x, _ = jax.lax.scan(body, x, (wi_stk, wo_stk))
    return jnp.sum(x)

def baseline_grad(x, wi, wo):
    return jax.value_and_grad(baseline_scan, argnums=(1,))(x, wi, wo)

# Abstract args — no memory allocated
x_abs = jax.ShapeDtypeStruct((B * S, D), jnp.bfloat16)
wi_abs = jax.ShapeDtypeStruct((L, E, D, D_moe), jnp.bfloat16)
wo_abs = jax.ShapeDtypeStruct((L, E, D_moe, D), jnp.bfloat16)

with jax.default_device(topo.devices[0]):
    lowered_a = jax.jit(
        baseline_grad,
        in_shardings=(
            NamedSharding(mesh, P("fsdp", "tp")),
            NamedSharding(mesh, P(None, None, "tp", "fsdp")),
            NamedSharding(mesh, P(None, None, "fsdp", "tp")),
        )
    ).lower(x_abs, wi_abs, wo_abs)
    compiled_a = lowered_a.compile()
    hlo_a = lowered_a.as_text()

# Check stacked wi layout
stk_shape_a = f"{L},{E},{D_local},{F_local}"  # 3,256,3584,8
layouts_a = extract_layouts(hlo_a, stk_shape_a)
print(f"  Stacked shape [{stk_shape_a}] layouts found: {len(layouts_a)}")
for lay in sorted(set(layouts_a)):
    count = layouts_a.count(lay)
    is_bad = f",{F_local}" in lay and lay.split('{')[1].startswith(str(lay.count(',') - 1))
    print(f"    {lay}  (×{count})")

print()


# ============================================================================
print("=" * 70)
print("TEST B: Transpose — wi stored as (E, D_moe, D)")
print(f"  Spec P(None, 'fsdp', 'tp') → per-dev ({E}, {F_local}, {D_local})")
print("=" * 70)

wt_spec = P(None, "fsdp", "tp")  # same for wi and wo

@functools.partial(shard_map, mesh=mesh,
                   in_specs=(P("fsdp", "tp"), wt_spec, wt_spec),
                   out_specs=P("fsdp", "tp"), check_rep=False)
def transpose_body(x_, wi_, wo_):
    wi_t = wi_.transpose(0, 2, 1)  # (E, F, D) → (E, D, F)
    return simple_moe_body(x_, wi_t, wo_, "fsdp")

def transpose_scan(x, wi_stk, wo_stk):
    def body(x, params):
        wi, wo = params
        return x + transpose_body(x, wi, wo), None
    x, _ = jax.lax.scan(body, x, (wi_stk, wo_stk))
    return jnp.sum(x)

def transpose_grad(x, wi, wo):
    return jax.value_and_grad(transpose_scan, argnums=(1,))(x, wi, wo)

wi_t_abs = jax.ShapeDtypeStruct((L, E, D_moe, D), jnp.bfloat16)
wo_t_abs = jax.ShapeDtypeStruct((L, E, D_moe, D), jnp.bfloat16)

with jax.default_device(topo.devices[0]):
    lowered_b = jax.jit(
        transpose_grad,
        in_shardings=(
            NamedSharding(mesh, P("fsdp", "tp")),
            NamedSharding(mesh, P(None, None, "fsdp", "tp")),
            NamedSharding(mesh, P(None, None, "fsdp", "tp")),
        )
    ).lower(x_abs, wi_t_abs, wo_t_abs)
    compiled_b = lowered_b.compile()
    hlo_b = lowered_b.as_text()

# Check stacked wi layout — transposed shape
stk_shape_b = f"{L},{E},{F_local},{D_local}"  # 3,256,8,3584
layouts_b = extract_layouts(hlo_b, stk_shape_b)
print(f"  Stacked shape [{stk_shape_b}] layouts found: {len(layouts_b)}")
for lay in sorted(set(layouts_b)):
    count = layouts_b.count(lay)
    print(f"    {lay}  (×{count})")

# Also check if XLA created any (E, D_local, F_local) tensors from the transpose
transpose_shape = f"{E},{D_local},{F_local}"  # 256,3584,8
layouts_bt = extract_layouts(hlo_b, transpose_shape)
print(f"  Body transposed shape [{transpose_shape}] layouts: {len(layouts_bt)}")
for lay in sorted(set(layouts_bt)):
    count = layouts_bt.count(lay)
    print(f"    {lay}  (×{count})")

print()


# ============================================================================
print("=" * 70)
print("SUMMARY")
print("=" * 70)

# Count padding occurrences
def count_padded(hlo, dim_size=F_local):
    """Count tensors where dim_size appears as a dimension with T(8,128)."""
    count = 0
    for line in hlo.split('\n'):
        if f'T(8,128)' in line and f'bf16[' in line:
            # Check if dim_size is the minor dim (last number before {)
            for part in line.split():
                if 'bf16[' in part and f',{dim_size}]' in part:
                    # Check layout — is it minor?
                    if '{' in part:
                        layout = part.split('{')[1].split(':')[0]
                        dims = [int(d) for d in layout.split(',')]
                        shape_str = part.split('[')[1].split(']')[0]
                        shape = [int(d) for d in shape_str.split(',')]
                        # Find which dim has value dim_size and check if it's minor (first in layout)
                        for i, s in enumerate(shape):
                            if s == dim_size and dims[0] == i:
                                count += 1
                                break
    return count

bad_a = count_padded(hlo_a)
bad_b = count_padded(hlo_b)

print(f"Baseline (E,D,F):     F={F_local}-as-minor tensors: {bad_a}")
print(f"Transpose (E,F,D):    F={F_local}-as-minor tensors: {bad_b}")
print()
if bad_b < bad_a:
    print(f"  ✓ Transpose reduces F-as-minor from {bad_a} → {bad_b} on v7x")
elif bad_b == bad_a:
    print(f"  ✗ No difference — v7x XLA overrides layout in both cases")
else:
    print(f"  ✗ Transpose is WORSE ({bad_b} > {bad_a})")
