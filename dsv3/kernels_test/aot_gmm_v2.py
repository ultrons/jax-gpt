"""Gate-1 AOT compile harness for gmm_v2.

Compiles gmm_v2 at DSv3 671B shapes WITHOUT a TPU:
- lhs (131072, 7168) bf16 — per-shard tokens after EP AG, post-sort
- rhs (64, 7168, 2048) bf16 — local experts × D × F_moe
- group_sizes (64,) int32

Catches every Mosaic shape-cast / relayout error documented in CLAUDE.md
(bool reshape, SMEM multi-D, packed-element-gather, trailing-1 reshape, etc.)
in ~2 min on any host with libtpu, no cluster cost.

Usage:
  source ~/xdb/.xprof/bin/activate
  python aot_gmm_v2.py
"""
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import topologies

# tpu_inference's package __init__ pulls in vllm; bypass by loading gmm_v2 directly.
import importlib.util
_gmm_path = "/home/sivaibhav_google_com/tpu-inference/tpu_inference/kernels/megablox/gmm_v2.py"
_spec = importlib.util.spec_from_file_location("gmm_v2_module", _gmm_path)
_gmm_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_gmm_mod)
gmm_v2 = _gmm_mod.gmm_v2

# DSv3 671B shapes (per-EP-shard, after AG, n_chunks=2).
LHS_M = 131072  # max_local_c = chunk_size * K / EP = 65536 * 8 / 4
LHS_K = 7168    # D (model hidden)
RHS_N = 2048    # F_moe (FFN intermediate)
N_GROUPS_LOCAL = 64  # E_local = E / EP = 256 / 4

# Output dim of the WO matmul.
WO_K = 2048
WO_N = 7168


def make_abstract(shape, dtype):
    return jax.ShapeDtypeStruct(shape, dtype)


def aot_compile(fn, *abstract_args, name=""):
    """AOT compile fn on tpu7x topology. Returns elapsed seconds or raises."""
    print(f"[{name}] AOT compiling ...", flush=True)
    t0 = time.perf_counter()
    topo = topologies.get_topology_desc("tpu7x:4x4x4", platform="tpu")
    with jax.default_device(topo.devices[0]):
        lowered = jax.jit(fn).lower(*abstract_args)
        compiled = lowered.compile()
    dt = time.perf_counter() - t0
    print(f"[{name}] OK in {dt:.1f}s", flush=True)

    # Cost / memory analysis — confirms tile sizes were picked, no spill, etc.
    try:
        cost = compiled.cost_analysis()
        mem = compiled.memory_analysis()
        if cost is not None:
            print(f"  flops={cost.get('flops', 'n/a'):,.0f} "
                  f"bytes_accessed={cost.get('bytes accessed', 'n/a'):,.0f}", flush=True)
        if mem is not None:
            for slot in ("output_size_in_bytes", "temp_size_in_bytes"):
                if slot in mem:
                    print(f"  {slot}={mem[slot]:,}", flush=True)
    except Exception as e:
        print(f"  (cost/memory analysis unavailable: {e})", flush=True)

    return dt


def case_plain_bf16():
    """Single gmm_v2 call: bf16 lhs × bf16 rhs, no activation fusion."""
    lhs = make_abstract((LHS_M, LHS_K), jnp.bfloat16)
    rhs = make_abstract((N_GROUPS_LOCAL, LHS_K, RHS_N), jnp.bfloat16)
    gs = make_abstract((N_GROUPS_LOCAL,), jnp.int32)

    def fn(lhs, rhs, gs):
        return gmm_v2(lhs, rhs, gs)

    return aot_compile(fn, lhs, rhs, gs, name="plain_bf16(M=131072,K=7168,N=2048)")


def case_wo_bf16():
    """The output matmul: gmm_v2 of hidden × wo."""
    lhs = make_abstract((LHS_M, WO_K), jnp.bfloat16)
    rhs = make_abstract((N_GROUPS_LOCAL, WO_K, WO_N), jnp.bfloat16)
    gs = make_abstract((N_GROUPS_LOCAL,), jnp.int32)

    def fn(lhs, rhs, gs):
        return gmm_v2(lhs, rhs, gs)

    return aot_compile(fn, lhs, rhs, gs, name="wo_bf16(M=131072,K=2048,N=7168)")


def case_fused_silu_bf16():
    """fuse_act='silu' — gate AND up in one kernel.

    Per gmm_v2 docstring + apply_act_fn: the kernel concatenates wi_0 and wi_1
    along the N dimension (size 2*N), accumulator splits via jnp.split(acc, 2, -1),
    then computes silu(gate) * up.
    """
    lhs = make_abstract((LHS_M, LHS_K), jnp.bfloat16)
    # Concatenated weights: (E_local, K, 2*N) where first half is wi_0, second wi_1.
    rhs_fused = make_abstract((N_GROUPS_LOCAL, LHS_K, 2 * RHS_N), jnp.bfloat16)
    gs = make_abstract((N_GROUPS_LOCAL,), jnp.int32)

    def fn(lhs, rhs, gs):
        # Default tile picker chose tile_n=768 → 69M VMEM (over 64M limit by 5M).
        # Force a smaller VMEM budget so picker chooses smaller tile_n.
        return gmm_v2(lhs, rhs, gs, fuse_act='silu',
                      vmem_limit_bytes=int(48 * 1024 * 1024))  # 48M of 64M

    return aot_compile(fn, lhs, rhs_fused, gs,
                       name="fused_silu_bf16(M=131072,K=7168,N=2*2048,vmem=48M)")


if __name__ == "__main__":
    print(f"jax={jax.__version__} devices={jax.devices()}", flush=True)
    cases = [
        ("plain_bf16 (gate or up alone)", case_plain_bf16),
        ("wo_bf16 (output proj)",         case_wo_bf16),
        ("fused_silu_bf16 (gate+up)",     case_fused_silu_bf16),
    ]
    failed = []
    for label, fn in cases:
        print(f"\n=== {label} ===")
        try:
            fn()
        except Exception as e:
            print(f"[{label}] FAILED: {type(e).__name__}: {e}", flush=True)
            failed.append(label)
    print("\n" + "="*60)
    if failed:
        print(f"FAILED ({len(failed)}/{len(cases)}):")
        for f in failed:
            print(f"  - {f}")
        sys.exit(1)
    print(f"PASSED all {len(cases)} cases")
