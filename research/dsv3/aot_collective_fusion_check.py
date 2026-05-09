"""AOT probe: does ragged_all_to_all lower async on v7x with production LIBTPU_INIT_ARGS?

Tests two hypotheses identified by the iter-14 Plan agent:
1. ragged_all_to_all NOT in SC-offload flag set in production manifest
   → may still go async via xla_tpu_enable_async_collective_fusion=true
2. If it lowers as plain `ragged-all-to-all` (sync TC) only, path C
   (collective fusion) is likely a NET REGRESSION on v7x training.

Decision rule:
- HLO contains `ragged-all-to-all-start` → async; proceed with iter-14b mini-repro
- HLO contains only `ragged-all-to-all` (sync) → abandon path C; pivot

Usage: python research/dsv3/aot_collective_fusion_check.py 2>&1 | tee /tmp/aot_collective_fusion_check.log
"""
import os

# Mirror manifests/jobset.yaml.j2:LIBTPU_INIT_ARGS exactly (production env).
os.environ["LIBTPU_INIT_ARGS"] = (
    "--xla_tpu_scoped_vmem_limit_kib=65536 "
    "--xla_tpu_bf16_emission_mode=NATIVE_EMISSION "
    "--xla_tpu_num_sparse_cores_for_gather_offloading=2 "
    "--xla_tpu_enable_sparse_core_collective_offload_all_gather=true "
    "--xla_tpu_enable_sparse_core_collective_offload_2d_all_gather=true "
    "--xla_tpu_enable_sparse_core_collective_offload_3d_all_gather=true "
    "--xla_tpu_enable_sparse_core_collective_offload_all_reduce=true "
    "--xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=true "
    "--xla_tpu_enable_sparse_core_collective_offload_nd_reduce_scatter=true "
    "--xla_tpu_enable_sparse_core_reduce_scatter_v2=true "
    "--xla_tpu_use_single_sparse_core_for_all_gather_offload=false "
    "--xla_tpu_enable_concurrent_sparse_core_offloading=true "
    "--xla_tpu_enable_offloading_gather_to_sparsecore=true "
    "--xla_tpu_use_tc_device_shape_on_sc=True "
    "--xla_sc_disable_megacore_partitioning=True "
    "--xla_tpu_enable_sparse_core_collective_aggregator=true "
    "--xla_tpu_enable_latency_hiding_layer_scheduler=true "
    "--xla_tpu_enable_layer_scheduler_for_dependent_collectives=true "
    "--xla_tpu_enable_multi_compute_overlap_in_layer_scheduler=true "
    "--xla_tpu_rerun_latency_hiding_scheduler_post_sc_assignment=true "
    "--xla_tpu_scheduler_percent_shared_memory_limit=150 "
    "--xla_tpu_enable_async_collective_fusion=true "
    "--xla_tpu_enable_async_collective_fusion_fuse_all_reduce=true "
    "--xla_tpu_enable_async_collective_fusion_fuse_all_gather=false "
    "--xla_tpu_enable_async_collective_fusion_fuse_reduce_scatter=true "
    "--xla_enable_async_reduce_scatter_fusion=true "
    "--xla_tpu_enable_async_collective_fusion_multiple_steps=true "
    "--xla_tpu_overlap_compute_collective_tc=true "
    "--xla_enable_async_all_gather=true "
    "--xla_enable_async_all_reduce=true "
    "--xla_tpu_prefer_async_allgather_to_allreduce=true "
    "--xla_tpu_enable_data_parallel_all_reduce_opt=true "
    "--xla_tpu_enable_ag_backward_pipelining=true "
    "--xla_tpu_enable_ici_ag_pipelining=true "
    "--xla_tpu_enable_ici_rs_pipelining=true "
    "--xla_max_concurrent_async_all_gathers=2 "
    "--xla_max_concurrent_async_reduce_scatters=2 "
    "--xla_tpu_pcie_bandwidth_multiplier=0.03 "
    "--xla_tpu_sparse_core_all_gather_latency_multiplier=1 "
    "--xla_tpu_sparse_core_reduce_scatter_latency_multiplier=3 "
    "--xla_tpu_enable_3d_reduce_scatter_decomposer=false "
    "--xla_tpu_enable_all_gather_offload_tracing=false "
    "--xla_tpu_enable_all_reduce_offload_tracing=false "
    "--xla_tpu_aggregate_data_dependent_sc_ops=false "
    "--xla_tpu_scheduling_annotation_deannotate_unsupported_groups=true "
)

# Force XLA HLO dump
HLO_DUMP_DIR = "/tmp/aot_collective_fusion_check_hlo"
os.makedirs(HLO_DUMP_DIR, exist_ok=True)
os.environ["XLA_FLAGS"] = f"--xla_dump_to={HLO_DUMP_DIR} --xla_dump_hlo_pass_re='.*'"

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import topologies
from jax.experimental.shard_map import shard_map
from jax.sharding import PartitionSpec as P, Mesh

print(f"jax version: {jax.__version__}")
print(f"jax.numpy backend: {jax.default_backend()}")

# Get a v7x topology — small one is fine for AOT (we just need the backend).
# tpu7x:2x2x1 = 4 chips × 2 cores = 8 logical devices.
topo = topologies.get_topology_desc("tpu7x:2x2x1", platform="tpu")
ep_size = 4  # use 4 of the 8 logical devices for ep
devs = np.array(topo.devices[:ep_size]).reshape(ep_size)
mesh = Mesh(devs, ("ep",))
print(f"mesh: {mesh}")


def _ragged_a2a_inner(x_, ss_, rs_):
    send_offsets = jnp.concatenate([jnp.zeros(1, dtype=jnp.int32), jnp.cumsum(ss_)[:-1]])
    recv_offsets = jnp.concatenate([jnp.zeros(1, dtype=jnp.int32), jnp.cumsum(rs_)[:-1]])
    recv_buf = jnp.zeros_like(x_)
    out = jax.lax.ragged_all_to_all(
        x_, recv_buf,
        input_offsets=send_offsets,
        send_sizes=ss_,
        output_offsets=recv_offsets,
        recv_sizes=rs_,
        axis_name="ep",
    )
    return out

ragged_a2a_fn = shard_map(
    _ragged_a2a_inner, mesh=mesh,
    in_specs=(P("ep"), P("ep"), P("ep")), out_specs=P("ep"),
    check_rep=False,
)


# Production-shape-aligned abstract args (per chunk in iter-2b):
# max_local_c = chunk_size * K / ep = (T_all/n_chunks) * K / ep = (131072/2) * 8 / 4 = 131072
# But for AOT probe at tpu7x:2x2x1 we need shape divisible by ep_size=4 in this mesh.
# Use 16384 (smaller, just to get a representative HLO; lowering decisions are
# shape-independent for the engine-assignment question).
M_per_shard = 16384  # rows per shard
D = 7168  # hidden dim
ep = ep_size

x_abs = jax.ShapeDtypeStruct((M_per_shard * ep, D), jnp.bfloat16)
ss_abs = jax.ShapeDtypeStruct((ep * ep,), jnp.int32)  # ep shards each have ep send sizes
rs_abs = jax.ShapeDtypeStruct((ep * ep,), jnp.int32)

# Symmetric variant for comparison: a plain reduce_scatter (the current pattern).
def _rs_inner(x_):
    return jax.lax.psum_scatter(x_, "ep", scatter_dimension=0, tiled=True)

reduce_scatter_fn = shard_map(
    _rs_inner, mesh=mesh, in_specs=P("ep"), out_specs=P("ep"), check_rep=False
)

# AOT both
print("\n=== AOT compile: ragged_all_to_all ===")
with jax.default_device(topo.devices[0]):
    lowered_a2a = jax.jit(ragged_a2a_fn).lower(x_abs, ss_abs, rs_abs)
    compiled_a2a = lowered_a2a.compile()
    hlo_a2a = compiled_a2a.as_text()
    print(f"Compile OK; HLO length: {len(hlo_a2a)} chars")

print("\n=== AOT compile: psum_scatter (reference) ===")
with jax.default_device(topo.devices[0]):
    x_for_rs = jax.ShapeDtypeStruct((M_per_shard * ep, D), jnp.bfloat16)
    lowered_rs = jax.jit(reduce_scatter_fn).lower(x_for_rs)
    compiled_rs = lowered_rs.compile()
    hlo_rs = compiled_rs.as_text()
    print(f"Compile OK; HLO length: {len(hlo_rs)} chars")

# Inspect the HLO output
def grep_collective_kinds(hlo: str, label: str):
    """Find collective op kinds in optimized HLO."""
    print(f"\n--- {label} HLO collective ops ---")
    keywords = [
        "ragged-all-to-all-start", "ragged-all-to-all-done", "ragged-all-to-all",
        "all-to-all-start", "all-to-all-done", "all-to-all",
        "reduce-scatter-start", "reduce-scatter-done", "reduce-scatter",
        "all-gather-start", "all-gather-done", "all-gather",
        "all-reduce-start", "all-reduce-done", "all-reduce",
    ]
    for kw in keywords:
        count = hlo.count(kw)
        if count > 0:
            # First occurrence line
            idx = hlo.index(kw)
            line_start = hlo.rfind("\n", 0, idx) + 1
            line_end = hlo.find("\n", idx)
            line = hlo[line_start:line_end] if line_end > 0 else hlo[line_start:line_start+200]
            print(f"  {kw:35s} count={count}  e.g.: {line.strip()[:120]}")
    print()

grep_collective_kinds(hlo_a2a, "ragged_all_to_all path")
grep_collective_kinds(hlo_rs,  "psum_scatter (reference)")

# Verdict
async_a2a = "ragged-all-to-all-start" in hlo_a2a
sync_a2a = ("ragged-all-to-all" in hlo_a2a) and not async_a2a
async_rs = "reduce-scatter-start" in hlo_rs
sync_rs = ("reduce-scatter" in hlo_rs) and not async_rs

print("=" * 60)
print(f"VERDICT:")
print(f"  ragged_all_to_all: {'ASYNC (start/done)' if async_a2a else ('SYNC' if sync_a2a else 'NOT FOUND')}")
print(f"  reduce_scatter:    {'ASYNC (start/done)' if async_rs else ('SYNC' if sync_rs else 'NOT FOUND')}")
print()
if async_a2a:
    print("  → Path C is viable. Proceed to iter-14b mini-repro.")
elif sync_a2a:
    print("  → Path C likely net regression. Abandon, pivot to FINDINGS.md option 4 or other multi-iter.")
else:
    print("  → ragged_all_to_all not in HLO; lowering may have rewritten it. Inspect HLO manually.")
print(f"  HLO dumped to: {HLO_DUMP_DIR}")
print("=" * 60)
