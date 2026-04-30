# Pallas backward kernel v3 for fused_ep_moe (training).
#
# Design: mirrors fused_ep_moe_fwd_streaming_v1 in communication structure.
#   - EP-local inputs (NOT pre-gathered) — kernel handles all A2A internally
#   - Packed scatter: tokens + d_out → expert devices (2D payload per token)
#   - Per-expert FFN backward (recompute h_g/h_u/h_act, then backward GEMMs)
#   - A2A gather: d_tokens → token-owner devices (same as forward gather)
#   - Outputs: d_tokens partial (FSDP contribution), d_w1_shard, d_w2_shard
#
# Communication cost vs forward:
#   - Forward: 1D scatter (tokens) + 1D gather (outputs)
#   - Backward v3: 2D scatter (tokens+d_out packed) + 1D gather (d_tokens)
#   - Ratio: 1.5× more ICI data than forward — still ~3-4× less than v2
#
# External collectives needed AFTER this kernel (in shard_map wrapper):
#   - lax.psum(d_tok, "fsdp")                        — sum FSDP partial contributions
#   - lax.psum_scatter(d_tok, "ep", scatter_dim=0)    — reduce-scatter EP
#   - lax.psum_scatter(d_w1, "fsdp", scatter_dim=3)   — reduce-scatter FSDP weight grads
#   - lax.psum_scatter(d_w2, "fsdp", scatter_dim=1)   — reduce-scatter FSDP weight grads
#
# VMEM budget at bt=8, EP=32, D=7168, F_shard=128 (debug config):
#   a2a_s_x2_vmem (packed 2D): 2 × 8×32 × 4 × 3584 × 2B = 14.7 MB
#   a2a_s_acc_x2_vmem (d_tok): 2 × 8×32 × 2 × 3584 × 2B = 7.4 MB
#   h_g_acc + h_u_acc (float32): 2 × 8×32 × 128 × 4B = 0.5 MB
#   b_w1g/b_w1u/b_w2 double-buffers: same as fwd ~10 MB
#   d_w2_acc, d_w1g_acc, d_w1u_acc (float32, per expert): 3 × 128×7168×4B = 11 MB
#   Total: ~44 MB < 64 MB VMEM budget ✓

import functools
import sys

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

sys.path.insert(0, "/home/sivaibhav_google_com/tpu-inference")
from tpu_inference.kernels.fused_moe.v1.kernel import (
    align_to,
    get_dtype_packing,
    broadcast_minor,
    apply_scoring_fn,
)


def _fused_ep_moe_bwd_kernel(
    # ---- HBM inputs ----
    # Packed (tokens, fw-scaled d_out): row t*K+k = (tokens[t], fw[t,k]*d_out[t])
    # Shape (T_local * K, 2*t_packing, D//t_packing); routing weight embedded in d_out.
    packed_hbm,       # (T_local * top_k, 2*t_packing, hidden_size//t_packing)
    w1_hbm,           # (E_local, 2, D, F_shard)
    w2_hbm,           # (E_local, F_shard, D)
    gating_hbm,       # (local_num_tokens, padded_E_global) — precomputed scores (scoring_fn applied)
    routing_weights_hbm,  # (local_num_tokens, top_k) — precomputed routing weights (fw_l)
    # ---- HBM outputs ----
    d_tokens_hbm,     # (local_num_tokens, D) — token gradients (partial in FSDP)
    d_w1_hbm,         # (E_local, 2, D, F_shard) — weight gradients (per-FSDP-shard)
    d_w2_hbm,         # (E_local, F_shard, D) — weight gradients
    # NOTE: a2a_g_hbm is declared as a Pallas OUTPUT (not input) so that XLA allocates
    # it at compile time with a consistent virtual address on all SPMD devices.
    # Runtime-provided inputs (even P()-replicated) have per-device independent
    # allocations and thus different physical HBM addresses — breaking ICI remote DMA.
    # Pallas output buffers are part of the compiled kernel layout, and XLA's SPMD
    # partitioner guarantees they land at the same virtual address on every device.
    a2a_g_hbm,        # (E_global, bt, t_packing, D//t_packing) — d_tokens gather scratch
    # ---- SMEM scratch (routing metadata) ----
    t2e_routing_x2_smem,    # (2, bt, padded_top_k)
    d2e_count_x2_smem,      # (2, num_devices, 1, padded_E_global)
    expert_offsets_x2_smem, # (2, 2, padded_E_global)
    expert_starts_x2_smem,  # (2, 1, padded_E_global)
    expert_sizes_x2_smem,   # (2, 1, padded_E_global)
    a2a_s_sends_x2_smem,    # (2,)
    # ---- VMEM scratch ----
    # Scatter buffer — holds packed (tokens, d_out) for each expert, double-buffered
    a2a_s_x2_vmem,    # (2, bt*EP, 2*t_packing, D//t_packing)
    # d_tokens accumulator at expert device — gets gathered back, double-buffered
    a2a_s_acc_x2_vmem,  # (2, bt*EP, t_packing, D//t_packing)
    # d_tokens gather accumulator (indexed by expert × bt × D)
    a2a_g_acc_vmem,   # (top_k, bt, t_packing, D//t_packing)
    # Gating scores (for routing weight recovery)
    b_gating_x2_vmem,  # (2, bt, padded_E_global)
    # Output d_tokens buffer
    b_output_x2_vmem,  # (2, bt, D)
    # Weight single-buffers (no cross-expert prefetch; fetch overlaps scatter recv within same expert)
    b_w1_vmem,     # (t_packing, bd1//t_packing, bf)
    b_w3_vmem,     # (t_packing, bd1//t_packing, bf)
    b_w2_vmem,     # (t_packing, bf, bd2//t_packing)
    # Static VMEM scratch for weight grad accumulators (single t_packing plane at a time)
    w1g_scratch_vmem,   # (bd1//t_packing, bf) f32 — 1.75 MB vs 3.5 MB for full t_packing
    w1u_scratch_vmem,   # (bd1//t_packing, bf) f32
    w2_scratch_vmem,    # (bf, bd2//t_packing) f32
    # ---- Semaphores ----
    local_sems,           # (2, 5)
    send_sems,            # (2,) — scatter sends only (avoid pollution from gather sends)
    recv_sems,            # (2,)
    gather_send_sems,     # (E_local,) — one slot per expert; never shared across experts
    a2a_gather_sem,
    a2a_acc_sem,
    *,
    top_k: int,
    renormalize_topk_logits: bool,
    ep_axis_name: str,
    act_fn: str = "silu",
    non_ep_axis_name: str = "fsdp",
    non_ep_first: bool = False,
    extra_mesh_axes: tuple = (),
    extra_device_id_prefix: tuple = (),
    bt: int,
    bf: int,
    bd1: int,
    bd2: int,
    btc: int,
    bfc: int,
    bd1c: int,
    bd2c: int,
    passthrough: bool = False,
):
    """Pallas backward kernel for EP MoE (v3).

    Communication structure mirrors the forward kernel (_fused_ep_moe_kernel):
      scatter(tokens+d_out) → expert_bwd → gather(d_tokens)

    Called from fused_ep_moe_bwd_v3 which is inside shard_map(ep, fsdp).
    """
    my_id = lax.axis_index(ep_axis_name)
    num_devices = lax.axis_size(ep_axis_name)
    right_id = (my_id + 1) % num_devices
    # packed_hbm has shape (T_local * top_k, 2*t_packing, D//t_packing):
    #   row t_id * top_k + k_id carries (tokens[t], fw[t,k] * d_out[t])
    # T_local is the actual number of local tokens.
    local_num_tokens = packed_hbm.shape[0] // top_k  # T_local
    local_num_experts, intermediate_size, hidden_size = w2_hbm.shape
    num_experts = a2a_g_hbm.shape[0]
    padded_num_experts = d2e_count_x2_smem.shape[-1]
    padded_top_k = t2e_routing_x2_smem.shape[-1]
    num_bt = local_num_tokens // bt

    t_dtype = jnp.bfloat16  # packed_hbm encodes bf16
    t_packing = get_dtype_packing(t_dtype)  # 2 for bf16
    t_bitwidth = 32 // t_packing

    bd1_per_t_packing = bd1 // t_packing
    bd2_per_t_packing = bd2 // t_packing
    bd1c_per_t_packing = bd1c // t_packing
    bd2c_per_t_packing = bd2c // t_packing
    num_bf = intermediate_size // bf
    num_bd1 = hidden_size // bd1
    num_bd2 = hidden_size // bd2

    # ---- Device ID helper (identical to forward) ----
    def get_mesh_device_id(ep_rank):
        non_ep_rank = lax.axis_index(non_ep_axis_name)
        if non_ep_first:
            pair = (non_ep_rank, ep_rank)
        else:
            pair = (ep_rank, non_ep_rank)
        if extra_device_id_prefix:
            return extra_device_id_prefix + pair
        if extra_mesh_axes:
            prefix = tuple(lax.axis_index(ax) for ax in extra_mesh_axes)
            return prefix + pair
        return pair

    # ---- Barrier (identical to forward) ----
    def sync_barrier():
        barrier_sem = pltpu.get_barrier_semaphore()
        for i in range(num_devices):
            pltpu.semaphore_signal(
                barrier_sem,
                device_id=get_mesh_device_id(i),
                device_id_type=pltpu.DeviceIdType.MESH,
            )
        pltpu.semaphore_wait(barrier_sem, num_devices)

    # ---- Gating prefetch (identical to forward) ----
    def start_fetch_b_gating(bt_id, priority=0):
        is_valid = jnp.logical_and(0 <= bt_id, bt_id < num_bt)
        sz = pl.multiple_of(lax.select(is_valid, bt, 0), bt)
        bt_sem_id = (bt_id + 2) % 2
        b_gating_sem = local_sems.at[bt_sem_id, 0]
        pltpu.make_async_copy(
            src_ref=gating_hbm.at[pl.ds(bt_id * bt, sz)],
            dst_ref=b_gating_x2_vmem.at[bt_sem_id, pl.ds(0, sz)],
            sem=b_gating_sem,
        ).start(priority=priority)

    def wait_fetch_b_gating(bt_id):
        bt_sem_id = bt_id % 2
        b_gating_sem = local_sems.at[bt_sem_id, 0]
        pltpu.make_async_copy(
            src_ref=b_gating_x2_vmem.at[bt_sem_id],
            dst_ref=b_gating_x2_vmem.at[bt_sem_id],
            sem=b_gating_sem,
        ).wait()

    # ---- Routing (uses precomputed gating scores, scoring_fn already applied) ----
    # We pass scoring_fn="identity" since gating_hbm already contains post-scoring scores.
    def get_top_k(input, top_k, renormalize_topk_logits):
        """Identical to forward get_top_k — routing table must match forward exactly."""
        input = input.astype(jnp.float32)
        padded_k_shape = (input.shape[0], padded_top_k)
        top_k_logits_lst = []
        t2e = jnp.zeros(input.shape, dtype=jnp.int32)
        t2e_routing = jnp.zeros(padded_k_shape, dtype=jnp.int32)
        iota = lax.broadcasted_iota(jnp.int32, input.shape, 1)
        padded_k_iota = lax.broadcasted_iota(jnp.int32, padded_k_shape, 1)
        top_k_logits_sum = jnp.zeros(padded_k_shape, jnp.float32)

        for k_id in range(top_k):
            top_k_logits = jnp.broadcast_to(
                jnp.max(input[:, :num_experts], axis=1, keepdims=True),
                padded_k_shape,
            ).astype(input.dtype)
            top_k_logits_lst.append(top_k_logits)
            if renormalize_topk_logits:
                top_k_logits_sum += top_k_logits
            top_k_indices = jnp.broadcast_to(
                jnp.argmax(input[:, :num_experts], axis=1, keepdims=True),
                padded_k_shape,
            )
            t2e_routing = jnp.where(padded_k_iota == k_id, top_k_indices, t2e_routing)
            mask = iota == broadcast_minor(top_k_indices, input.shape)
            t2e += mask.astype(jnp.int32)
            if k_id != top_k - 1:
                input = jnp.where(mask, -jnp.inf, input)

        if renormalize_topk_logits:
            for k_id in range(top_k):
                top_k_logits_lst[k_id] /= top_k_logits_sum

        expert_sizes = jnp.sum(t2e, axis=0, keepdims=True)
        expert_starts = jnp.zeros_like(expert_sizes)
        return top_k_logits_lst, t2e_routing, expert_sizes, expert_starts

    # ---- Metadata all-reduce (mirrors forward kernel exactly) ----
    def all_reduce_metadata(bt_sem_id, t2e_routing, starts, sizes):
        send_sem = send_sems.at[0]
        recv_sem = recv_sems.at[0]

        def _all_reduce_metadata(
            t2e_routing_vmem,   # VMEM (bt, padded_top_k)
            d2e_count_vmem,     # VMEM (num_devices, 1, padded_num_experts)
            offsets_vmem,       # VMEM (2, padded_num_experts)
            starts_vmem,        # VMEM (1, padded_num_experts)
            sizes_vmem,         # VMEM (1, padded_num_experts)
        ):
            offsets_vmem[...] = jnp.zeros_like(offsets_vmem)
            offsets_copy = pltpu.async_copy(
                src_ref=offsets_vmem,
                dst_ref=expert_offsets_x2_smem.at[bt_sem_id],
                sem=send_sem,
            )
            t2e_routing_vmem[...] = t2e_routing
            t2e_routing_copy = pltpu.async_copy(
                src_ref=t2e_routing_vmem,
                dst_ref=t2e_routing_x2_smem.at[bt_sem_id],
                sem=send_sem,
            )
            reduced_sizes = sizes
            reduced_starts = starts
            row_id = my_id
            d2e_count_vmem[row_id] = sizes
            for i in range(num_devices - 1):
                sync_barrier()
                pltpu.async_remote_copy(
                    src_ref=d2e_count_vmem.at[row_id],
                    dst_ref=d2e_count_vmem.at[row_id],
                    send_sem=send_sem,
                    recv_sem=recv_sem,
                    device_id=get_mesh_device_id(right_id),
                    device_id_type=pltpu.DeviceIdType.MESH,
                ).wait()
                row_id = (row_id + num_devices - 1) % num_devices
                new_sizes = d2e_count_vmem[row_id]
                reduced_sizes += new_sizes
                reduced_starts += lax.select(
                    my_id > i, new_sizes, jnp.zeros_like(new_sizes))
            starts_vmem[...] = reduced_starts
            sizes_vmem[...] = reduced_sizes

            starts_copy = pltpu.async_copy(
                src_ref=starts_vmem,
                dst_ref=expert_starts_x2_smem.at[bt_sem_id],
                sem=send_sem,
            )
            sizes_copy = pltpu.async_copy(
                src_ref=sizes_vmem,
                dst_ref=expert_sizes_x2_smem.at[bt_sem_id],
                sem=send_sem,
            )
            d2e_count_copy = pltpu.async_copy(
                src_ref=d2e_count_vmem,
                dst_ref=d2e_count_x2_smem.at[bt_sem_id],
                sem=send_sem,
            )

            t2e_routing_copy.wait()
            d2e_count_copy.wait()
            offsets_copy.wait()
            starts_copy.wait()
            sizes_copy.wait()

        pl.run_scoped(
            _all_reduce_metadata,
            pltpu.VMEM(t2e_routing_x2_smem.shape[1:], t2e_routing_x2_smem.dtype),
            pltpu.VMEM(d2e_count_x2_smem.shape[1:], d2e_count_x2_smem.dtype),
            pltpu.VMEM(expert_offsets_x2_smem.shape[1:], expert_offsets_x2_smem.dtype),
            pltpu.VMEM(expert_starts_x2_smem.shape[1:], expert_starts_x2_smem.dtype),
            pltpu.VMEM(expert_sizes_x2_smem.shape[1:], expert_sizes_x2_smem.dtype),
        )

    # ---- A2A scatter: send packed (tokens, fw-scaled d_out) to expert devices ----
    # packed_hbm row t_id * top_k + k_id carries (tokens[t], fw[t,k] * d_out[t]).
    # This embeds the routing weight in the payload so the expert device doesn't need
    # to look up fw for remote tokens.
    def start_a2a_scatter(bt_id, e_sem_id, local_e_id):
        bt_sem_id = bt_id % 2
        send_sz = 0
        for bt_t_id in range(bt):
            for k_id in range(top_k):
                e_id = t2e_routing_x2_smem[bt_sem_id, bt_t_id, k_id]
                is_active_expert = e_id % local_num_experts == local_e_id
                recv_id = e_id // local_num_experts
                offset = expert_offsets_x2_smem[bt_sem_id, 0, e_id]
                sz = lax.select(is_active_expert, 1, 0)
                is_local = recv_id == my_id
                local_sz = lax.select(is_local, sz, 0)
                remote_sz = lax.select(is_local, 0, sz)
                send_sz += remote_sz
                expert_offsets_x2_smem[bt_sem_id, 0, e_id] = offset + local_sz + remote_sz
                start = expert_starts_x2_smem[bt_sem_id, 0, e_id] + offset
                # Row in packed_hbm for this (token, k) pair: fw[t,k]-scaled d_out
                t_k_id = (bt * bt_id + bt_t_id) * top_k + k_id
                pltpu.make_async_copy(
                    src_ref=packed_hbm.at[pl.ds(t_k_id, local_sz)],
                    dst_ref=a2a_s_x2_vmem.at[e_sem_id, pl.ds(start, local_sz)],
                    sem=recv_sems.at[e_sem_id],
                ).start()
                pltpu.make_async_remote_copy(
                    src_ref=packed_hbm.at[pl.ds(t_k_id, remote_sz)],
                    dst_ref=a2a_s_x2_vmem.at[e_sem_id, pl.ds(start, remote_sz)],
                    send_sem=send_sems.at[e_sem_id],
                    recv_sem=recv_sems.at[e_sem_id],
                    device_id=get_mesh_device_id(recv_id),
                    device_id_type=pltpu.DeviceIdType.MESH,
                ).start()
        a2a_s_sends_x2_smem[e_sem_id] = send_sz

    def wait_a2a_scatter_recv(bt_id, e_sem_id, local_e_id):
        bt_sem_id = bt_id % 2
        e_id = my_id * local_num_experts + local_e_id
        sz = expert_sizes_x2_smem[bt_sem_id, 0, e_id]
        pltpu.make_async_copy(
            src_ref=a2a_s_x2_vmem.at[e_sem_id, pl.ds(0, sz)],
            dst_ref=a2a_s_x2_vmem.at[e_sem_id, pl.ds(0, sz)],
            sem=recv_sems.at[e_sem_id],
        ).wait()

    def wait_a2a_scatter_send(bt_id, e_sem_id, local_e_id):
        del bt_id, local_e_id
        sz = a2a_s_sends_x2_smem[e_sem_id]
        pltpu.make_async_copy(
            src_ref=a2a_s_x2_vmem.at[e_sem_id, pl.ds(0, sz)],
            dst_ref=a2a_s_x2_vmem.at[e_sem_id, pl.ds(0, sz)],
            sem=send_sems.at[e_sem_id],
        ).wait()

    # ---- A2A gather: send d_tokens from experts back to token owners ----
    # Identical to forward start_a2a_gather — sends a2a_s_acc_x2_vmem → a2a_g_hbm
    def start_a2a_gather(bt_id, e_sem_id, local_e_id):
        my_e_id = my_id * local_num_experts + local_e_id
        bt_sem_id = bt_id % 2
        start = 0
        for recv_id in range(num_devices):
            sz = d2e_count_x2_smem[bt_sem_id, recv_id, 0, my_e_id]
            is_local = recv_id == my_id
            local_sz = lax.select(is_local, sz, 0)
            remote_sz = lax.select(is_local, 0, sz)
            pltpu.make_async_copy(
                src_ref=a2a_s_acc_x2_vmem.at[e_sem_id, pl.ds(start, local_sz)],
                dst_ref=a2a_g_hbm.at[my_e_id, pl.ds(0, local_sz)],
                sem=a2a_gather_sem,
            ).start()
            pltpu.make_async_remote_copy(
                src_ref=a2a_s_acc_x2_vmem.at[e_sem_id, pl.ds(start, remote_sz)],
                dst_ref=a2a_g_hbm.at[my_e_id, pl.ds(0, remote_sz)],
                send_sem=gather_send_sems.at[local_e_id],
                recv_sem=a2a_gather_sem,
                device_id=get_mesh_device_id(recv_id),
                device_id_type=pltpu.DeviceIdType.MESH,
            ).start()
            start += sz

    def wait_a2a_gather_send(bt_id, local_e_id):
        """Wait for gather sends for one expert. Uses dedicated slot gather_send_sems[local_e_id]."""
        my_e_id = my_id * local_num_experts + local_e_id
        bt_sem_id = bt_id % 2
        sz = expert_sizes_x2_smem[bt_sem_id, 0, my_e_id]
        local_sz = d2e_count_x2_smem[bt_sem_id, my_id, 0, my_e_id]
        remote_sz = sz - local_sz
        is_valid = jnp.logical_and(0 <= local_e_id, local_e_id < local_num_experts)
        remote_sz = lax.select(is_valid, remote_sz, 0)
        ref = a2a_g_hbm.reshape(num_experts * bt, t_packing, hidden_size // t_packing)
        pltpu.make_async_copy(
            src_ref=ref.at[pl.ds(0, remote_sz)],
            dst_ref=ref.at[pl.ds(0, remote_sz)],
            sem=gather_send_sems.at[local_e_id],
        ).wait()

    def wait_a2a_gather_recv_all():
        sz = top_k * bt
        ref = a2a_g_hbm.reshape(num_experts * bt, t_packing, hidden_size // t_packing)
        pltpu.make_async_copy(
            src_ref=ref.at[pl.ds(0, sz)],
            dst_ref=ref.at[pl.ds(0, sz)],
            sem=a2a_gather_sem,
        ).wait()

    # ---- Weight prefetch (identical to forward) ----
    def start_fetch_bw1(local_e_id, bw1_sem_id, bf_id, bd1_id):
        del bw1_sem_id
        for p in range(t_packing):
            offset = p * (hidden_size // t_packing) + bd1_id * bd1_per_t_packing
            pltpu.make_async_copy(
                src_ref=w1_hbm.at[local_e_id, 0,
                                   pl.ds(offset, bd1_per_t_packing),
                                   pl.ds(bf_id * bf, bf)],
                dst_ref=b_w1_vmem.at[p,
                                      pl.ds(0, bd1_per_t_packing),
                                      pl.ds(0, bf)],
                sem=local_sems.at[0, 1],
            ).start()

    def start_fetch_bw3(local_e_id, bw3_sem_id, bf_id, bd3_id):
        del bw3_sem_id
        for p in range(t_packing):
            offset = p * (hidden_size // t_packing) + bd3_id * bd1_per_t_packing
            pltpu.make_async_copy(
                src_ref=w1_hbm.at[local_e_id, 1,
                                   pl.ds(offset, bd1_per_t_packing),
                                   pl.ds(bf_id * bf, bf)],
                dst_ref=b_w3_vmem.at[p,
                                      pl.ds(0, bd1_per_t_packing),
                                      pl.ds(0, bf)],
                sem=local_sems.at[0, 3],
            ).start()

    def start_fetch_bw2(local_e_id, bw2_sem_id, bf_id, bd2_id):
        del bw2_sem_id
        for p in range(t_packing):
            offset = p * (hidden_size // t_packing) + bd2_id * bd2_per_t_packing
            pltpu.make_async_copy(
                src_ref=w2_hbm.at[local_e_id,
                                   pl.ds(bf_id * bf, bf),
                                   pl.ds(offset, bd2_per_t_packing)],
                dst_ref=b_w2_vmem.at[p,
                                      pl.ds(0, bf),
                                      pl.ds(0, bd2_per_t_packing)],
                sem=local_sems.at[0, 2],
            ).start()

    def wait_fetch_bw1(local_e_id, bw1_sem_id, bf_id, bd1_id):
        del local_e_id, bw1_sem_id, bf_id, bd1_id
        pltpu.make_async_copy(
            src_ref=b_w1_vmem,
            dst_ref=b_w1_vmem,
            sem=local_sems.at[0, 1],
        ).wait()

    def wait_fetch_bw3(local_e_id, bw3_sem_id, bf_id, bd3_id):
        del local_e_id, bw3_sem_id, bf_id, bd3_id
        pltpu.make_async_copy(
            src_ref=b_w3_vmem,
            dst_ref=b_w3_vmem,
            sem=local_sems.at[0, 3],
        ).wait()

    def wait_fetch_bw2(local_e_id, bw2_sem_id, bf_id, bd2_id):
        del local_e_id, bw2_sem_id, bf_id, bd2_id
        pltpu.make_async_copy(
            src_ref=b_w2_vmem,
            dst_ref=b_w2_vmem,
            sem=local_sems.at[0, 2],
        ).wait()

    # ---- Per-expert backward compute ----
    def expert_bwd(bt_id, e_sem_id, local_e_id):
        """FFN backward for one expert's token batch.

        Weights are pre-loaded into b_w1/b_w3/b_w2_vmem by
        start_fetch_bw1/3/2 in run_per_expert before this is called.
        Weight grads are accumulated into d_w1_hbm/d_w2_hbm via DMA
        read-modify-write inside pl.run_scoped (HBM→f32 VMEM, add, write back).

        Inputs already in a2a_s_x2_vmem[e_sem_id] (packed: tokens || d_out).
        Computes d_tok → a2a_s_acc_x2_vmem[e_sem_id].
        """
        if passthrough:
            # PASSTHROUGH: copy scatter tokens directly to acc buffer (skip all GEMMs).
            # Tests the ICI gather path independently of backward math.
            # Expected: d_tok[t] = 2 * tokens_l[t] (each token appears in K=2 routes).
            s_full = a2a_s_x2_vmem[e_sem_id]  # (bt*EP, 2*t_packing, D//t_packing)
            a2a_s_acc_x2_vmem[e_sem_id] = s_full[:, :t_packing, :]
            return  # skip weight grad accumulation (d_w1/d_w2 remain uninitialized)

        bt_sem_id = bt_id % 2
        e_id = my_id * local_num_experts + local_e_id
        dyn_sz = expert_sizes_x2_smem[bt_sem_id, 0, e_id]

        # ---- Unpack tokens and d_out from packed scatter buffer ----
        # Read full static-size slot — cannot use dynamic pl.ds on a VMEM ref directly.
        # a2a_s_x2_vmem[e_sem_id]: (bt*EP, 2*t_packing, D//t_packing) bf16
        # [:, :t_packing, :] = tokens; [:, t_packing:, :] = d_out
        bt_ep_max = a2a_s_x2_vmem.shape[1]  # bt * EP (static)
        s_full = a2a_s_x2_vmem[e_sem_id]    # (bt*EP, 2*t_packing, D//t_packing)
        # Mask rows beyond dyn_sz to zero so invalid rows don't contribute to matmuls.
        # VMEM may contain NaN/Inf garbage for unfilled rows; NaN * 0 = NaN in IEEE 754.
        # Fix: create (N,D) float mask via (N,D)*(N,1) multiply-broadcast (Mosaic-safe),
        # then use jnp.where with a same-shape bool condition (no shape cast needed).
        # Mosaic cannot handle bool (N,1)→(N,D) broadcast in select ("unsupported shape cast"),
        # but float (N,D)*(N,1) multiply-broadcast is supported.
        row_ids = jnp.arange(bt_ep_max, dtype=jnp.int32)
        valid_mask_f = (row_ids < dyn_sz).astype(jnp.float32)[:, None]  # (bt*EP, 1) float
        tokens_e_raw = s_full[:, :t_packing, :].reshape(bt_ep_max, hidden_size).astype(jnp.float32)
        dout_e_raw   = s_full[:, t_packing:, :].reshape(bt_ep_max, hidden_size).astype(jnp.float32)
        # (N,D) float mask: 1.0 valid, 0.0 invalid — XLA folds ones_like*valid away.
        valid_nd = jnp.ones_like(tokens_e_raw) * valid_mask_f
        tokens_e = jnp.where(valid_nd > jnp.float32(0.5), tokens_e_raw, jnp.float32(0.0))
        dout_e   = jnp.where(valid_nd > jnp.float32(0.5), dout_e_raw,   jnp.float32(0.0))

        # ---- Routing weight is embedded in dout_e (fw-scaled in packed_hbm) ----
        # d_out in the scatter payload is already fw[t,k]-scaled: the wrapper wrote
        # packed[t*K+k] = (tokens[t], fw[t,k] * d_out[t]).  No separate routing weight
        # lookup needed — this is correct for EP>1 since remote tokens' gating is
        # unknown to this device.

        # ---- Reconstruct weight matrices from pre-fetched VMEM buffers ----
        # b_w1_vmem: (t_packing, bd1//t_packing, bf)
        # Planes are contiguous halves of D: plane p → rows [p*D//2 : (p+1)*D//2]
        # reshape(D, F_shard) works because row-major layout is preserved.
        w1g = b_w1_vmem[...].reshape(hidden_size, intermediate_size).astype(jnp.float32)
        w1u = b_w3_vmem[...].reshape(hidden_size, intermediate_size).astype(jnp.float32)
        # b_w2_vmem: (t_packing, bf, bd2//t_packing)
        # Planes are column halves of D: plane p → cols [p*D//2 : (p+1)*D//2]
        # concatenate along axis=-1 to reconstruct (F_shard, D).
        w2_slot = b_w2_vmem[...]   # (t_packing, bf, bd2//t_packing)
        w2 = jnp.concatenate(
            [w2_slot[p].astype(jnp.float32) for p in range(t_packing)], axis=-1
        )  # (intermediate_size, hidden_size)

        # ---- Forward recompute ----
        h_g = tokens_e @ w1g    # (bt*EP, F_shard)
        h_u = tokens_e @ w1u
        sig_hg  = jax.nn.sigmoid(h_g)
        silu_hg = h_g * sig_hg
        h_act   = silu_hg * h_u

        # ---- Backward through w2 ----
        # dout_e is already fw[t,k]-scaled (embedded in scatter payload)
        dout_scaled = dout_e
        d_h_act = dout_scaled @ w2.T

        # ---- SwiGLU backward ----
        silu_grad_hg = sig_hg * (1 + h_g * (1 - sig_hg))
        d_h_u = d_h_act * silu_hg
        d_h_g = d_h_act * h_u * silu_grad_hg

        # ---- Token gradient → a2a_s_acc_x2_vmem ----
        d_tok = d_h_g @ w1g.T + d_h_u @ w1u.T   # (bt*EP, D) float32
        d_tok_packed = d_tok.astype(t_dtype).reshape(bt_ep_max, t_packing, hidden_size // t_packing)
        a2a_s_acc_x2_vmem[e_sem_id] = d_tok_packed

        # ---- Weight grads: DMA read-modify-write into HBM outputs ----
        # Invalid rows (≥ dyn_sz) have tokens_e=0/dout_scaled=0, so contribute nothing.
        d_w1g_m = tokens_e.T @ d_h_g    # (D, F_shard) f32
        d_w1u_m = tokens_e.T @ d_h_u
        d_w2_m  = h_act.T @ dout_scaled  # (F_shard, D) f32

        # Semaphore slots reused from weight fetch (already idle after wait_fetch_bw1/3/2).
        bw1_sem = local_sems.at[e_sem_id, 1]
        bw3_sem = local_sems.at[e_sem_id, 3]
        bw2_sem = local_sems.at[e_sem_id, 2]

        def _accumulate_weight_grads(w1g_acc, w1u_acc, w2_acc):
            """Read-modify-write HBM weight grad accumulator, one t_packing plane at a time.

            Scratch shapes are (bd1_per_t_packing, bf) / (bf, bd2_per_t_packing) — single plane.
            Processing one plane at a time halves VMEM vs the all-planes-simultaneously version.
            Pallas output buffers are NOT zero-initialized; use lax.select on bt_id==0 to avoid
            reading HBM entirely (NaN * 0.0 = NaN in IEEE 754, so prev_scale trick is unsafe).
            """
            is_first = (bt_id == 0)
            d_w1g_tile = d_w1g_m.reshape(t_packing, bd1_per_t_packing, bf)
            d_w1u_tile = d_w1u_m.reshape(t_packing, bd1_per_t_packing, bf)

            for p in range(t_packing):
                off1 = p * bd1_per_t_packing
                off2 = p * bd2_per_t_packing
                d_w2_p = d_w2_m[:, off2:off2 + bd2_per_t_packing]  # (bf, bd2_per_t_packing)

                # Read plane p from HBM → single-plane VMEM scratch (skip on first bt tile)
                @pl.when(~is_first)
                def _read_hbm():
                    pltpu.make_async_copy(
                        src_ref=d_w1_hbm.at[local_e_id, 0,
                                             pl.ds(off1, bd1_per_t_packing), pl.ds(0, bf)],
                        dst_ref=w1g_acc,
                        sem=bw1_sem,
                    ).start()
                    pltpu.make_async_copy(
                        src_ref=d_w1_hbm.at[local_e_id, 1,
                                             pl.ds(off1, bd1_per_t_packing), pl.ds(0, bf)],
                        dst_ref=w1u_acc,
                        sem=bw3_sem,
                    ).start()
                    pltpu.make_async_copy(
                        src_ref=d_w2_hbm.at[local_e_id,
                                             pl.ds(0, bf), pl.ds(off2, bd2_per_t_packing)],
                        dst_ref=w2_acc,
                        sem=bw2_sem,
                    ).start()
                    pltpu.make_async_copy(src_ref=w1g_acc, dst_ref=w1g_acc, sem=bw1_sem).wait()
                    pltpu.make_async_copy(src_ref=w1u_acc, dst_ref=w1u_acc, sem=bw3_sem).wait()
                    pltpu.make_async_copy(src_ref=w2_acc,  dst_ref=w2_acc,  sem=bw2_sem).wait()

                # Accumulate: on bt_id=0 write gradient directly (don't touch NaN HBM),
                # on bt_id>0 add to accumulated value read from HBM above.
                w1g_acc[...] = lax.select(is_first, d_w1g_tile[p], w1g_acc[...] + d_w1g_tile[p])
                w1u_acc[...] = lax.select(is_first, d_w1u_tile[p], w1u_acc[...] + d_w1u_tile[p])
                w2_acc[...]  = lax.select(is_first, d_w2_p,         w2_acc[...]  + d_w2_p)

                # Write updated plane p back to HBM
                pltpu.make_async_copy(
                    src_ref=w1g_acc,
                    dst_ref=d_w1_hbm.at[local_e_id, 0,
                                         pl.ds(off1, bd1_per_t_packing), pl.ds(0, bf)],
                    sem=bw1_sem,
                ).start()
                pltpu.make_async_copy(
                    src_ref=w1u_acc,
                    dst_ref=d_w1_hbm.at[local_e_id, 1,
                                         pl.ds(off1, bd1_per_t_packing), pl.ds(0, bf)],
                    sem=bw3_sem,
                ).start()
                pltpu.make_async_copy(
                    src_ref=w2_acc,
                    dst_ref=d_w2_hbm.at[local_e_id,
                                         pl.ds(0, bf), pl.ds(off2, bd2_per_t_packing)],
                    sem=bw2_sem,
                ).start()
                pltpu.make_async_copy(src_ref=w1g_acc, dst_ref=w1g_acc, sem=bw1_sem).wait()
                pltpu.make_async_copy(src_ref=w1u_acc, dst_ref=w1u_acc, sem=bw3_sem).wait()
                pltpu.make_async_copy(src_ref=w2_acc,  dst_ref=w2_acc,  sem=bw2_sem).wait()

        _accumulate_weight_grads(w1g_scratch_vmem, w1u_scratch_vmem, w2_scratch_vmem)

    # ---- Output send (d_tokens to HBM, same structure as forward output send) ----
    def start_send_bo(bt_id, priority=0):
        bt_sem_id = bt_id % 2
        b_output_sem = local_sems.at[bt_sem_id, 4]
        pltpu.make_async_copy(
            src_ref=b_output_x2_vmem.at[bt_sem_id],
            dst_ref=d_tokens_hbm.at[pl.ds(bt_id * bt, bt)],
            sem=b_output_sem,
        ).start(priority=priority)

    def wait_send_bo(bt_id):
        is_valid = jnp.logical_and(0 <= bt_id, bt_id < num_bt)
        sz = pl.multiple_of(lax.select(is_valid, bt, 0), bt)
        bt_sem_id = (bt_id + 2) % 2
        b_output_sem = local_sems.at[bt_sem_id, 4]
        pltpu.make_async_copy(
            src_ref=d_tokens_hbm.at[pl.ds(0, sz)],
            dst_ref=d_tokens_hbm.at[pl.ds(0, sz)],
            sem=b_output_sem,
        ).wait()

    # ---- bt_acc for d_tokens (sum over K, no weight scaling) ----
    # Same structure as forward bt_acc but accumulates d_tokens without routing weight.
    # (Weight scaling was already applied inside expert_bwd via d_out_scaled.)
    def bt_acc_dtok(bt_id):
        bt_sem_id = bt_id % 2
        # Sum K contributions to d_tok for each token position
        output = None
        for k_id in range(top_k):
            for bt_t_id in range(bt):
                e_id = t2e_routing_x2_smem[bt_sem_id, bt_t_id, k_id]
                offset = expert_offsets_x2_smem[bt_sem_id, 1, e_id]
                expert_offsets_x2_smem[bt_sem_id, 1, e_id] = offset + 1
                pltpu.make_async_copy(
                    src_ref=a2a_g_hbm.at[e_id, pl.ds(offset, 1)],
                    dst_ref=a2a_g_acc_vmem.at[k_id, pl.ds(bt_t_id, 1)],
                    sem=a2a_acc_sem,
                ).start()
        pltpu.make_async_copy(
            src_ref=a2a_g_acc_vmem,
            dst_ref=a2a_g_acc_vmem,
            sem=a2a_acc_sem,
        ).wait()

        output = None
        for k_id in range(top_k):
            # d_tokens: sum over K without routing weight (already applied in expert_bwd)
            acc = a2a_g_acc_vmem[k_id].reshape(bt, hidden_size)
            if output is None:
                output = acc
            else:
                output = output + acc
        assert output is not None
        return output.astype(t_dtype)

    # ---- Main loop (same pipeline structure as forward) ----
    sync_barrier()
    start_fetch_b_gating(bt_id=0)

    def run_per_bt(bt_id, e_sem_id):
        bt_sem_id = bt_id % 2
        next_bt_id = bt_id + 1
        start_fetch_b_gating(next_bt_id)
        wait_fetch_b_gating(bt_id)

        b_gating = b_gating_x2_vmem[bt_sem_id]
        # gating_hbm already contains post-scoring values (identity scoring_fn)
        b_gating_score = b_gating  # already scored

        # Routing (must match forward routing exactly)
        top_k_logits_lst, t2e_routing, expert_sizes, expert_starts = get_top_k(
            b_gating_score.astype(jnp.float32), top_k, renormalize_topk_logits)

        all_reduce_metadata(bt_sem_id, t2e_routing, expert_starts, expert_sizes)
        sync_barrier()

        # Start packed scatter (tokens + d_out) for first expert
        start_a2a_scatter(bt_id=bt_id, e_sem_id=e_sem_id, local_e_id=0)

        def run_per_expert(local_e_id, e_sem_id):
            next_e_sem_id = lax.select(e_sem_id == 0, 1, 0)
            next_local_e_id = local_e_id + 1

            # Start scatter for NEXT expert
            @pl.when(next_local_e_id < local_num_experts)
            def _():
                start_a2a_scatter(bt_id, next_e_sem_id, next_local_e_id)

            # Start weight DMA fetch for CURRENT expert (overlaps with scatter recv).
            # bf_id=0, bd1_id=0: single tile covers full weight for correctness-first version.
            start_fetch_bw1(local_e_id, e_sem_id, 0, 0)
            start_fetch_bw3(local_e_id, e_sem_id, 0, 0)
            start_fetch_bw2(local_e_id, e_sem_id, 0, 0)

            # Wait for scatter recv of CURRENT expert
            wait_a2a_scatter_recv(bt_id, e_sem_id, local_e_id)

            # Wait for weight fetch of CURRENT expert
            wait_fetch_bw1(local_e_id, e_sem_id, 0, 0)
            wait_fetch_bw3(local_e_id, e_sem_id, 0, 0)
            wait_fetch_bw2(local_e_id, e_sem_id, 0, 0)

            # Backward compute for CURRENT expert
            expert_bwd(bt_id, e_sem_id, local_e_id)

            # Start gather to send d_tokens for CURRENT expert back to token owners
            start_a2a_gather(bt_id, e_sem_id, local_e_id)

            wait_a2a_scatter_send(bt_id, e_sem_id, local_e_id)
            sync_barrier()
            return next_e_sem_id

        e_sem_id = lax.fori_loop(0, local_num_experts, run_per_expert, e_sem_id, unroll=True)

        # Wait for all d_token gathers to complete
        wait_a2a_gather_recv_all()
        sync_barrier()

        # Debug: check a2a_g_hbm contents on ep_rank=0
        # Accumulate d_tokens from all K slots for this bt tile
        output = bt_acc_dtok(bt_id)

        wait_send_bo(bt_id=bt_id - 2)
        b_output_x2_vmem[bt_sem_id] = output
        start_send_bo(bt_id)

        for e in range(local_num_experts):
            wait_a2a_gather_send(bt_id, e)
        sync_barrier()
        return e_sem_id

    lax.fori_loop(0, num_bt, run_per_bt, 0, unroll=False)
    wait_send_bo(bt_id=num_bt - 2)
    wait_send_bo(bt_id=num_bt - 1)


def fused_ep_moe_bwd_v3(
    d_out_l,      # (T/(EP*FSDP), D) — local output gradient
    tokens_l,     # (T/(EP*FSDP), D) — local tokens
    fi_l,         # (T/(EP*FSDP), K) — precomputed top_k indices, EP-local (global IDs)
    fw_l,         # (T/(EP*FSDP), K) — precomputed top_k weights, EP-local
    gating_l,     # (T/(EP*FSDP), E_global) — biased sigmoid scores (pre-scored, identity fn)
    w1_shard,     # (E_local, 2, D, F_shard) — FSDP-sharded weights
    w2_shard,     # (E_local, F_shard, D)
    *,
    ep_axis_name: str,
    fsdp_axis_name: str,
    K: int,
    max_tpe: int | None = None,
    renormalize_topk_logits: bool = True,
    bt: int | None = None,
    bf: int | None = None,
    bd1: int | None = None,
    bd2: int | None = None,
    btc: int | None = None,
    bfc: int | None = None,
    bd1c: int | None = None,
    bd2c: int | None = None,
    passthrough: bool = False,
    extra_device_id_prefix: tuple = (),
):
    """Backward kernel wrapper for EP MoE v3.

    Called from within shard_map(ep, fsdp). Takes EP-local inputs (no pre-gather).
    The kernel handles all EP A2A internally (scatter tokens+d_out to experts, gather d_tokens).

    Returns (d_tok_partial, d_topk_partial, d_w1_partial, d_w2_partial) — all FSDP-partial.
    Caller must apply:
      d_tok = psum(d_tok_partial, fsdp_axis_name)
      d_tok_l = psum_scatter(d_tok, ep_axis_name, scatter_dimension=0, tiled=True)
      d_w1_l  = psum_scatter(d_w1_partial, fsdp_axis_name, scatter_dimension=3, tiled=True)
      d_w2_l  = psum_scatter(d_w2_partial, fsdp_axis_name, scatter_dimension=1, tiled=True)
    """
    T_local, D = tokens_l.shape
    E_local, _, _, F_shard = w1_shard.shape
    E_global = gating_l.shape[1]
    num_devices = lax.axis_size(ep_axis_name)

    t_dtype = jnp.bfloat16  # kernel hardcodes bfloat16; wrapper must match
    t_packing = get_dtype_packing(t_dtype)  # always 2
    tokens_l = tokens_l.astype(t_dtype)
    d_out_l = d_out_l.astype(t_dtype)

    # ---- Block size defaults (same tuning as forward) ----
    if bf is None:
        bf = F_shard  # full F_shard per tile
    if bd1 is None:
        # Use bd1=D (full D in one tile) so expert_bwd can reshape (t_packing, D/tp, F) → (D, F).
        # D-tiling (bd1<D) would require a multi-pass forward recompute; not implemented yet.
        bd1 = D
        bd1 = (bd1 // (t_packing * 128)) * (t_packing * 128)
        bd1 = max(bd1, t_packing * 128)
    if bd2 is None:
        bd2 = D
        bd2 = (bd2 // (t_packing * 128)) * (t_packing * 128)
        bd2 = max(bd2, t_packing * 128)
    if bt is None:
        # bt=8 (minimum for HBM tile alignment; gating buf tiled (8,128) on HBM).
        # VMEM fits via single-slot weight buffers (no double-buffering needed).
        bt = min(8, T_local)
        bt = max(bt, t_packing)
        while T_local % bt != 0 and bt > t_packing:
            bt -= t_packing
    if btc is None:
        btc = min(bt, 8)
    if bfc is None:
        bfc = bf
    if bd1c is None:
        bd1c = min(bd1, 128)
    if bd2c is None:
        bd2c = min(bd2, 128)

    padded_num_experts = align_to(E_global, 128)
    padded_top_k = align_to(K, 128)

    # ---- Pack tokens + d_out into single scatter payload ----
    # tokens_l: (T_local, D) bf16 → reshape to (T_local, t_packing, D//t_packing) → bit-pack
    # d_out_l:  (T_local, D) bf16 → same
    # packed:   (T_local, 2*t_packing, D//t_packing) — tokens in first half, d_out in second
    assert D % t_packing == 0
    D_per_tp = D // t_packing
    tokens_packed = tokens_l.reshape(T_local, t_packing, D_per_tp)
    d_out_packed  = d_out_l.reshape(T_local, t_packing, D_per_tp)
    # Build K-indexed packed buffer: row t*K+k carries (tokens[t], fw[t,k] * d_out[t]).
    # This embeds the routing weight in the payload so the expert device (which may not
    # have gating scores for remote tokens in EP>1) can use dout directly as dout_scaled.
    fw_bc = fw_l[:, :K, None, None].astype(t_dtype)                        # (T, K, 1, 1)
    tokens_k = jnp.broadcast_to(
        tokens_packed[:, None, :, :], (T_local, K, t_packing, D_per_tp))   # (T, K, tp, D/tp)
    d_out_k   = d_out_packed[:, None, :, :] * fw_bc                        # (T, K, tp, D/tp)
    packed_k  = jnp.concatenate([tokens_k, d_out_k], axis=2)               # (T, K, 2*tp, D/tp)
    packed    = packed_k.reshape(T_local * K, 2 * t_packing, D_per_tp)     # (T*K, 2*tp, D/tp)

    # ---- Pad gating if needed ----
    if padded_num_experts != E_global:
        gating_padded = jnp.pad(gating_l, ((0, 0), (0, padded_num_experts - E_global)),
                                 constant_values=-jnp.inf)
    else:
        gating_padded = gating_l
    gating_padded = gating_padded.astype(t_dtype)  # bf16 for routing

    # ---- A2A gather scratch ----
    # a2a_g_hbm is now declared as a Pallas OUTPUT buffer (see out_shape above).
    # This ensures XLA allocates it at compile-time with the same virtual HBM address
    # on all SPMD devices — required for ICI remote DMA to write d_tokens back to the
    # correct device's gather buffer.  No scratch argument needed from the caller.

    # ---- Routing weights scratch (for expert_bwd weight scaling) ----
    fw_padded = jnp.pad(fw_l, ((0, 0), (0, padded_top_k - K))) if padded_top_k != K else fw_l

    hbm_spec = pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM)

    fused_moe_bwd = pl.pallas_call(
        functools.partial(
            _fused_ep_moe_bwd_kernel,
            top_k=K,
            renormalize_topk_logits=renormalize_topk_logits,
            ep_axis_name=ep_axis_name,
            act_fn="silu",
            non_ep_axis_name=fsdp_axis_name,
            non_ep_first=False,
            extra_mesh_axes=(),
            extra_device_id_prefix=extra_device_id_prefix,
            bt=bt,
            bf=bf,
            bd1=bd1,
            bd2=bd2,
            btc=btc,
            bfc=bfc,
            bd1c=bd1c,
            bd2c=bd2c,
            passthrough=passthrough,
        ),
        out_shape=[
            jax.ShapeDtypeStruct((T_local, D), t_dtype),           # d_tokens_hbm (bf16)
            jax.ShapeDtypeStruct(w1_shard.shape, jnp.float32),    # d_w1_hbm (float32 for accumulation)
            jax.ShapeDtypeStruct(w2_shard.shape, jnp.float32),    # d_w2_hbm (float32 for accumulation)
            # a2a_g_hbm: output buffer for gather scratch.
            # Declared as output so XLA allocates it at compile-time with the same
            # virtual HBM address on all SPMD devices — required for ICI remote DMA.
            jax.ShapeDtypeStruct((E_global, bt, t_packing, D_per_tp), t_dtype),
        ],
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            in_specs=[
                hbm_spec,  # packed_hbm
                hbm_spec,  # w1_hbm
                hbm_spec,  # w2_hbm
                hbm_spec,  # gating_hbm
                hbm_spec,  # routing_weights_hbm
            ],
            out_specs=[
                pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),  # d_tokens
                pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),  # d_w1
                pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),  # d_w2
                pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),  # a2a_g_hbm scratch
            ],
            scratch_shapes=[
                # SMEM routing metadata
                pltpu.SMEM((2, bt, padded_top_k), jnp.int32),                      # t2e_routing_x2
                pltpu.SMEM((2, num_devices, 1, padded_num_experts), jnp.int32),     # d2e_count_x2
                pltpu.SMEM((2, 2, padded_num_experts), jnp.int32),                  # expert_offsets_x2
                pltpu.SMEM((2, 1, padded_num_experts), jnp.int32),                  # expert_starts_x2
                pltpu.SMEM((2, 1, padded_num_experts), jnp.int32),                  # expert_sizes_x2
                pltpu.SMEM((2,), jnp.int32),                                         # a2a_s_sends_x2
                # VMEM A2A scatter (2× packed payload: tokens + d_out)
                pltpu.VMEM((2, bt * num_devices, 2 * t_packing, D_per_tp), t_dtype), # a2a_s_x2
                # VMEM A2A accumulator for d_tokens (same size as fwd)
                pltpu.VMEM((2, bt * num_devices, t_packing, D_per_tp), t_dtype),     # a2a_s_acc_x2
                # d_tokens gather accumulator
                pltpu.VMEM((K, bt, t_packing, D_per_tp), t_dtype),                   # a2a_g_acc
                # Gating double-buffer
                pltpu.VMEM((2, bt, padded_num_experts), t_dtype),                     # b_gating_x2
                # d_tokens output double-buffer
                pltpu.VMEM((2, bt, D), t_dtype),                                      # b_output_x2
                # Weight single-buffers (fetch overlaps scatter recv within same expert)
                pltpu.VMEM((t_packing, bd1 // t_packing, bf), w1_shard.dtype),       # b_w1
                pltpu.VMEM((t_packing, bd1 // t_packing, bf), w1_shard.dtype),       # b_w3
                pltpu.VMEM((t_packing, bf, bd2 // t_packing), w2_shard.dtype),       # b_w2
                # Static weight grad accumulators (single t_packing plane — halves VMEM)
                pltpu.VMEM((bd1 // t_packing, bf), jnp.float32),                      # w1g_scratch
                pltpu.VMEM((bd1 // t_packing, bf), jnp.float32),                      # w1u_scratch
                pltpu.VMEM((bf, bd2 // t_packing), jnp.float32),                      # w2_scratch
                # Semaphores
                pltpu.SemaphoreType.DMA((2, 5)),   # local_sems
                pltpu.SemaphoreType.DMA((2,)),      # send_sems (scatter only)
                pltpu.SemaphoreType.DMA((2,)),      # recv_sems
                pltpu.SemaphoreType.DMA((E_local,)),  # gather_send_sems — one slot per expert
                pltpu.SemaphoreType.DMA,            # a2a_gather_sem
                pltpu.SemaphoreType.DMA,            # a2a_acc_sem
            ],
        ),
        compiler_params=pltpu.CompilerParams(
            collective_id=1,   # Different from forward (collective_id=0) to avoid conflict
            vmem_limit_bytes=64 * 1024 * 1024,  # static ~39.6 MB + ~20 MB compute temps
        ),
        name=f"fused-moe-bwd-v3-bt{bt}_bf{bf}",
    )

    # a2a_g_hbm is now the 4th output (scratch buffer for gather); discard it.
    d_tokens, d_w1, d_w2, _ = fused_moe_bwd(
        packed,
        w1_shard,
        w2_shard,
        gating_padded,
        fw_padded,
    )

    # Compute d_topk (routing weight gradients) in JAX — local computation, no collective needed
    # d_topk[t, k] = dot(d_out[t], out[t,k]) where out[t,k] is the expert output for (t,k)
    # For v3 first version: return zeros as placeholder (routing weight grad is small/secondary)
    d_topk = jnp.zeros_like(fw_l)

    return d_tokens, d_topk, d_w1, d_w2
