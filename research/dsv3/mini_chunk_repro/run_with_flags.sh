#!/bin/bash
# Run mini-repro with cluster XLA flags + profile capture.
set -e
cd "$(dirname "$0")"
source ~/xdb/.xprof/bin/activate

rm -rf /tmp/mini_chunk_hlo /tmp/mini_chunk_prof
mkdir -p /tmp/mini_chunk_prof

# Subset of v300 flags relevant to non-SC platforms (v4 doesn't have SC).
export LIBTPU_INIT_ARGS="
  --xla_tpu_enable_latency_hiding_layer_scheduler=true
  --xla_tpu_enable_layer_scheduler_for_dependent_collectives=true
  --xla_tpu_enable_multi_compute_overlap_in_layer_scheduler=true
  --xla_tpu_scheduler_percent_shared_memory_limit=150
  --xla_tpu_enable_async_collective_fusion=true
  --xla_tpu_enable_async_collective_fusion_fuse_all_reduce=true
  --xla_tpu_enable_async_collective_fusion_fuse_all_gather=false
  --xla_tpu_enable_async_collective_fusion_multiple_steps=true
  --xla_tpu_overlap_compute_collective_tc=true
  --xla_enable_async_all_gather=true
  --xla_enable_async_all_reduce=true
  --xla_tpu_prefer_async_allgather_to_allreduce=true
  --xla_tpu_enable_data_parallel_all_reduce_opt=true
  --xla_tpu_enable_ag_backward_pipelining=true
  --xla_tpu_enable_ici_ag_pipelining=true
  --xla_tpu_enable_ici_rs_pipelining=true
  --xla_max_concurrent_async_all_gathers=2
  --xla_max_concurrent_async_reduce_scatters=2
"

export XLA_FLAGS="--xla_dump_to=/tmp/mini_chunk_hlo --xla_dump_hlo_as_text"

python mini_chunked_body.py
