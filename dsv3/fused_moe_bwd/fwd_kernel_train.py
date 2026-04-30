# Copyright 2025 Google LLC
#
# Training-specific Pallas MoE forward kernel — single-body with lax.fori_loop.
#
# Baseline: ~/tpu-inference/tpu_inference/kernels/fused_moe/v1/kernel.py
#
# v216 change: REVERT grid lift — use lax.fori_loop inside single kernel body.
#
# Root cause of v209–v215 runtime hang: the grid=(num_bt,) pallas_call ran each
# bt_id as a SEPARATE program instance.  pltpu.get_barrier_semaphore() is a
# device-global semaphore shared across all grid cells on the same device.
# When device A is at grid cell K and device B is at grid cell K+2, device A's
# sync_barrier() signals land in device B's barrier for cell K, not K+2.  The
# barrier counts bleed between cells → one device gets 32 premature signals while
# another's wait never completes → deadlock → 60-second ICI timeout → exit(1).
#
# Fix: use lax.fori_loop(0, num_bt, run_per_bt, ..., unroll=False) inside a
# single-body kernel (no grid), exactly like the inference kernel (kernel.py).
# With no grid, there is only ONE program instance per device.  All bt_id
# iterations share the same barrier semaphore handle and run strictly in order.
# Barriers across EP devices are guaranteed to be matched.
#
# Compile time impact: Mosaic compiles the fori_loop body ONCE (unroll=False),
# same as the grid approach.  Expected compile time is unchanged.
#
# Training adaptations vs inference kernel
# -----------------------------------------
# 1. [REVERTED] Grid lift — removed; use lax.fori_loop inside single body.
#
# 2. Flat 1-D SMEM shapes:
#      Original multi-D SMEM (2, bt, padded_top_k), (2, num_devices, 1, E), …
#      → flat (bt*padded_top_k,), (num_devices*E,), …
#      Multi-D SMEM GatherOps fail SC bitpacking when E=256 at cluster scale.
#
# 3. b_acc_vmem: remove size-1 dim:
#      Original (2, bt*num_devices, 1, bf) F32 → (2, bt*num_devices, bf) F32.
#      The intermediate 1-dim maps to sublanes; downstream reshape to (2,bt*N,bf)
#      triggers Mosaic "unsupported shape cast".  Use correct shape from the start.
#      dtype stays float32 — dynamic_ffn1 accumulates with preferred_element_type=f32.
#
# 4. get_top_k: full rewrite using int32 arithmetic:
#      Original uses (bt,E) bool + argmax(keepdims=True) + 2-D float32 SC Get.
#      All three fail on v7x Mosaic. Fix: row-by-row 1-D float32 argmax,
#      int32 row-selector masks, no bool arrays.  See comments inside.
#
# 5. Multi-axis mesh support:
#      Training mesh is (dp, ep, fsdp, tp). Kernel runs inside shard_map(ep, fsdp, tp),
#      so ep and fsdp axes are dynamic via lax.axis_index.
#      dp=0 is a static prefix; tp_rank is a static int via extra_device_id_suffix
#      (see adaptation 6).
#
# 6. tp_rank via extra_device_id_suffix (v216):
#      lax.axis_index("tp") inside Pallas is broken (v202 finding).  scalar_prefetch
#      is also removed (no more grid).  With no grid, tp_rank is passed as a static
#      Python int via extra_device_id_suffix=(tp_rank_int,).  For TP=1 (all current
#      cluster runs), tp_rank_int=0.  get_mesh_device_id uses the elif branch.
#
# 6. a2a_g_hbm flat reshape:
#      a2a_g_hbm reshaped to (num_experts*bt, pack, D//pack) before any indexing.
#      a2a_g_hbm.at[e_id, pl.ds(offset, 1)] has TWO dynamic indices → SC gather
#      fails when num_experts > 128.  Single 1-D index e_id*bt+offset is safe.

import functools
import os

import jax
import jax.numpy as jnp
from jax import lax
from jax._src import dtypes
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

P = jax.sharding.PartitionSpec


# ---------------------------------------------------------------------------
# Module-level utilities (identical to inference kernel)
# ---------------------------------------------------------------------------

def cdiv(a, b):
    return (a + b - 1) // b


def align_to(x, a):
    return cdiv(x, a) * a


def get_dtype_packing(dtype):
    bits = dtypes.itemsize_bits(dtype)
    return 32 // bits


def broadcast_minor(src, shape):
    if src.shape == shape:
        return src
    assert src.shape[:-1] == shape[:-1]
    assert src.shape[-1] % 128 == 0
    target_minor = align_to(shape[-1], src.shape[-1])
    return jnp.concatenate([src for _ in range(target_minor // src.shape[-1])],
                            axis=-1)[..., :shape[-1]]


def apply_scoring_fn(scoring_fn: str, x):
    match scoring_fn:
        case "softmax":
            return jax.nn.softmax(x, axis=-1)
        case "sigmoid":
            return 1 / (1 + jnp.exp(-x))
        case "identity":
            return x
        case _:
            raise NotImplementedError(f"Unsupported scoring function: {scoring_fn}")


def apply_act_fn(acc1, acc3, act_fn):
    if act_fn == "silu":
        return jax.nn.silu(acc1) * acc3
    elif act_fn == "gelu":
        return jax.nn.gelu(acc1) * acc3
    else:
        raise NotImplementedError(f"Unsupported activation function: {act_fn}")


# ---------------------------------------------------------------------------
# Training forward kernel — single-body with lax.fori_loop (v216)
# ---------------------------------------------------------------------------

def _fused_ep_moe_kernel_train(
    # Inputs (HBM)
    tokens_hbm,       # (T_local, t_packing, D//t_packing)
    w1_hbm,           # (E_local, 2, D, F_shard) — gate+up stacked
    w2_hbm,           # (E_local, F_shard, D) — down
    w1_scale_hbm,     # None
    w2_scale_hbm,     # None
    b1_hbm,           # None
    b2_hbm,           # None
    gating_hbm,       # (T_local, padded_num_experts)
    a2a_g_hbm,        # (num_experts, bt, t_packing, D//t_packing)  scratch
    # Precomputed ring-reduce metadata (v222 — eliminates 31-step ICI ring)
    expert_sizes_precomp_hbm,   # (num_bt * padded_num_experts,) int32
    expert_starts_precomp_hbm,  # (num_bt * padded_num_experts,) int32
    d2e_count_compact_hbm,      # (num_bt * num_devices * E_local,) int32
    # Output (HBM)
    output_hbm,       # (T_local, D)
    # SMEM scratch (flat 1-D; see adaptation note 2 above)
    t2e_routing_x2_smem,    # (bt * padded_top_k,)
    d2e_count_compact_smem, # (num_devices * E_local,) — compact: only my local experts
    expert_offsets_x2_smem, # (2 * padded_num_experts,)  [0:E]=scatter [E:2E]=gather
    expert_starts_x2_smem,  # (padded_num_experts,)
    expert_sizes_x2_smem,   # (padded_num_experts,)
    a2a_s_sends_x2_smem,    # (2,)
    # VMEM scratch
    a2a_s_x2_vmem,          # (2, bt*num_devices, t_packing, D//t_packing)
    a2a_s_acc_x2_vmem,      # (2, bt*num_devices, t_packing, D//t_packing)
    a2a_g_acc_vmem,         # (top_k, bt, t_packing, D//t_packing)
    b_gating_x2_vmem,       # (2, bt, padded_num_experts)
    b_output_x2_vmem,       # (2, bt, D)
    b_w1_x2_vmem,           # (2, t_packing, bd1//t_packing, bf)
    b_w3_x2_vmem,           # (2, t_packing, bd1//t_packing, bf)
    b_w2_x2_vmem,           # (2, t_packing, bf, bd2//t_packing)
    b_w1_scale_x2_vmem,     # None
    b_w3_scale_x2_vmem,     # None
    b_w2_scale_x2_vmem,     # None
    b_b1_x2_vmem,           # None
    b_b3_x2_vmem,           # None
    b_b2_x2_vmem,           # None
    b_acc_vmem,             # (2, bt*num_devices, bf) float32 — no size-1 dim
    local_sems,             # DMA(2, 5)
    send_sems,              # DMA(2,)
    recv_sems,              # DMA(2,)
    a2a_gather_sem,         # DMA
    a2a_acc_sem,            # DMA
    *,
    # Static kernel parameters
    top_k: int,
    renormalize_topk_logits: bool,
    ep_axis_name: str,
    act_fn: str,
    scoring_fn: str,
    subc_quant_w1_sz=None,
    subc_quant_w2_sz=None,
    non_ep_axis_name: str = "fsdp",
    non_ep_first: bool = False,   # training mesh is (ep, fsdp) → ep first
    extra_device_id_prefix: tuple = (),  # (dp_rank=0,) prepended
    extra_device_id_suffix: tuple = (),  # fallback static suffix when tp_axis_name is None
    tp_axis_name: str | None = None,     # when set, use lax.axis_index(tp_axis_name) as suffix
    bt: int,
    bf: int,
    bd1: int,
    bd2: int,
    btc: int,
    bfc: int,
    bd1c: int,
    bd2c: int,
):
    # ------------------------------------------------------------------ #
    # Single-body kernel — bt_id comes from lax.fori_loop, not program_id
    # ------------------------------------------------------------------ #
    my_id       = lax.axis_index(ep_axis_name)
    num_devices = lax.axis_size(ep_axis_name)
    right_id    = (my_id + 1) % num_devices

    local_num_tokens     = tokens_hbm.shape[0]
    local_num_experts, intermediate_size, hidden_size = w2_hbm.shape
    num_experts    = a2a_g_hbm.shape[0]
    padded_num_experts = expert_offsets_x2_smem.shape[0] // 2
    padded_top_k       = t2e_routing_x2_smem.shape[0] // bt

    num_bt  = cdiv(local_num_tokens, bt)   # Python int (static shapes)
    num_bf  = cdiv(intermediate_size, bf)
    num_bd1 = cdiv(hidden_size, bd1)
    num_bd2 = cdiv(hidden_size, bd2)

    t_dtype   = tokens_hbm.dtype
    t_packing = get_dtype_packing(t_dtype)
    t_bitwidth = 32 // t_packing

    h_per_t_packing   = hidden_size // t_packing
    bd1_per_t_packing = bd1 // t_packing
    bd2_per_t_packing = bd2 // t_packing
    bd1c_per_t_packing = bd1c // t_packing
    bd2c_per_t_packing = bd2c // t_packing

    # ------------------------------------------------------------------ #
    # Mesh device-ID helper (training: (dp=0, ep, fsdp, tp))
    # ------------------------------------------------------------------ #
    def get_mesh_device_id(ep_rank):
        non_ep_rank = jax.lax.axis_index(non_ep_axis_name)
        if non_ep_first:
            pair = (non_ep_rank, ep_rank)
        else:
            pair = (ep_rank, non_ep_rank)
        # tp_rank: use lax.axis_index(tp_axis_name) when tp is in scope.
        # Fallback: static extra_device_id_suffix=(tp_rank_int,) for TP=1
        # (v216: scalar_prefetch removed; tp_rank always from static suffix).
        if tp_axis_name is not None:
            suffix_jax = (jax.lax.axis_index(tp_axis_name).astype(jnp.int32),)
        elif extra_device_id_suffix:
            suffix_jax = tuple(jnp.int32(x) for x in extra_device_id_suffix)
        else:
            suffix_jax = (jnp.int32(0),)   # TP=1 fallback: tp_rank=0
        if extra_device_id_prefix:
            prefix_jax = tuple(jnp.int32(x) for x in extra_device_id_prefix)
            return prefix_jax + pair + suffix_jax
        return pair + suffix_jax

    # ------------------------------------------------------------------ #
    # Barrier (all-reduce signal across EP devices)
    # ------------------------------------------------------------------ #
    def sync_barrier():
        # v232: pl.run_scoped(REGULAR) only, collective_id=None in pallas_call config.
        #
        # Root cause of ALL prior hangs (v222–v224, v230):
        #   get_barrier_semaphore() is a SINGLE device-global semaphore per pallas_call.
        #   8 bt_id iterations (Python for loop = 8 static HLO call sites) all share
        #   one slot.  Fast device K+1 signals arrive at slow device K-wait → K fires
        #   early → K+1 hangs → TPU watchdog kills all pods simultaneously.
        #
        # Why v225–v228 failed (pl.run_scoped, collective_id=0):
        #   pallas_call config had collective_id=0.  Mosaic requires get_barrier_semaphore()
        #   when collective_id != None.  Error: "collective_id has to be unspecified or
        #   None when not using a custom barrier."
        #
        # Why v231 failed (get_barrier_semaphore() dummy + pl.run_scoped, collective_id=0):
        #   Calling get_barrier_semaphore() but NOT using it (result discarded) violates
        #   a Mosaic C++ invariant: barrier semaphore allocated (via collective_id) but
        #   has no signal/wait ops → LOG(FATAL) / _exit(1) → empty pod logs, exit code 1.
        #
        # Fix (v232): set collective_id=None in CompilerParams → Mosaic drops the
        #   "must call get_barrier_semaphore()" constraint.  Use pl.run_scoped(REGULAR)
        #   for all 8 barriers.  Each Python-for call site is a STATIC HLO node → unique
        #   SMEM slot per bt_id → no aliasing.  REGULAR + DeviceIdType.MESH ICI works:
        #   proven by util.py local_barrier() double_barrier=True (second barrier uses
        #   pl.run_scoped(REGULAR) + DeviceIdType.MESH in production inference kernel).
        def do_barrier(b_sem):
            for i in range(num_devices):
                pltpu.semaphore_signal(
                    b_sem,
                    device_id=get_mesh_device_id(i),
                    device_id_type=pltpu.DeviceIdType.MESH,
                )
            pltpu.semaphore_wait(b_sem, num_devices)
        pl.run_scoped(do_barrier, b_sem=pltpu.SemaphoreType.REGULAR)

    # ------------------------------------------------------------------ #
    # Gating prefetch helpers (double-buffered by bt_id % 2)
    # ------------------------------------------------------------------ #
    def start_fetch_b_gating(bid, priority=0):
        is_valid = jnp.logical_and(0 <= bid, bid < num_bt)
        sz = pl.multiple_of(lax.select(is_valid, bt, 0), bt)
        sem_id = (bid + 2) % 2
        pltpu.make_async_copy(
            src_ref=gating_hbm.at[pl.ds(bid * bt, sz)],
            dst_ref=b_gating_x2_vmem.at[sem_id, pl.ds(0, sz)],
            sem=local_sems.at[sem_id, 0],
        ).start(priority=priority)

    def wait_fetch_b_gating(bid):
        sem_id = bid % 2
        pltpu.make_async_copy(
            src_ref=b_gating_x2_vmem.at[sem_id],
            dst_ref=b_gating_x2_vmem.at[sem_id],
            sem=local_sems.at[sem_id, 0],
        ).wait()

    # ------------------------------------------------------------------ #
    # get_top_k — rewritten to avoid (bt,E) bools and 2-D float32 SC Gets
    # (see adaptation note 4 above for full explanation)
    # ------------------------------------------------------------------ #
    def get_top_k(input, top_k_n, renorm):
        # Convert inner bt_local loops to lax.fori_loop to reduce static HLO count.
        # Before: top_k_n * bt_local = 8*8 = 64 Python iterations → 640+ HLO ops.
        # After:  top_k_n = 8 outer Python iters, each with 1 WhileOp (bt_local body).
        assert len(input.shape) == 2
        bt_local = input.shape[0]
        padded_k_shape = (bt_local, padded_top_k)
        top_k_logits_lst = []
        t2e = jnp.zeros(input.shape, dtype=jnp.int32)
        t2e_routing = jnp.zeros(padded_k_shape, dtype=jnp.int32)
        padded_k_iota = jax.lax.broadcasted_iota(jnp.int32, padded_k_shape, 1)
        row_iota_k   = jax.lax.broadcasted_iota(jnp.int32, padded_k_shape, 0)
        row_iota_1d  = jax.lax.broadcasted_iota(jnp.int32, (bt_local,), 0)
        row_iota_e   = jax.lax.broadcasted_iota(jnp.int32, input.shape, 0)
        expert_iota  = jax.lax.broadcasted_iota(jnp.int32, (input.shape[1],), 0)
        top_k_logits_sum = jnp.zeros((bt_local,), jnp.float32)

        for k_id in range(top_k_n):
            # Inner loop over rows — now a single WhileOp instead of bt_local static iters.
            # Row extraction: element-wise multiply with 2-D row mask then reduce along
            # axis=0. Avoids dynamic_slice (unsupported TC path), matmul (VMEM constant
            # capture issue), and [:, None] reshape (trailing-1 forbidden).
            # `input` is carried so it is NOT a closed-over VMEM constant.
            #
            # Carry structure depends on renorm to avoid dead elements:
            #   renorm=True:  (top_k_l, top_k_idx, hit_m, tks, inp)
            #   renorm=False: (top_k_l, top_k_idx, hit_m, inp) — tks omitted
            #
            # expert_hit: (padded_ne,) int32 — 1-D, NOT reshaped to (1, padded_ne).
            # Use implicit broadcast: row_eq_i * expert_hit = (bt,E)*(E,) → (bt,E).
            # (1, E) leading-1 reshape inside a WhileOp body causes Mosaic runtime crash
            # even when AOT compilation passes.
            if renorm:
                def _row_body(i, carry):
                    top_k_l, top_k_idx, hit_m, tks, inp = carry
                    row_eq_i   = 1 - jnp.minimum(jnp.abs(row_iota_e - i), 1)
                    row_f32    = jnp.sum(inp[:, :num_experts].astype(jnp.float32)
                                         * row_eq_i[:, :num_experts].astype(jnp.float32), axis=0)
                    row_max    = jnp.max(row_f32).astype(jnp.float32)
                    row_argmax = jnp.argmax(row_f32).astype(jnp.int32)
                    row_eq_i_k = 1 - jnp.minimum(jnp.abs(row_iota_k - i), 1)
                    top_k_l = (top_k_l * (1 - row_eq_i_k).astype(jnp.bfloat16)
                               + row_eq_i_k.astype(jnp.bfloat16) * row_max.astype(jnp.bfloat16))
                    top_k_idx = top_k_idx * (1 - row_eq_i_k) + row_argmax * row_eq_i_k
                    row_sel_1d = 1 - jnp.minimum(jnp.abs(row_iota_1d - i), 1)
                    tks = tks + row_sel_1d.astype(jnp.float32) * row_max
                    expert_hit = (expert_iota == row_argmax).astype(jnp.int32)  # (E,)
                    hit_m      = hit_m + row_eq_i * expert_hit                  # (bt,E)*(E,)→(bt,E)
                    return (top_k_l, top_k_idx, hit_m, tks, inp)
                init_carry = (
                    jnp.zeros(padded_k_shape, jnp.bfloat16),
                    jnp.zeros(padded_k_shape, jnp.int32),
                    jnp.zeros(input.shape, jnp.int32),
                    top_k_logits_sum,
                    input,
                )
                top_k_logits, top_k_indices, hit_mask, top_k_logits_sum, _ = lax.fori_loop(
                    0, bt_local, _row_body, init_carry)
            else:
                def _row_body(i, carry):
                    top_k_l, top_k_idx, hit_m, inp = carry
                    row_eq_i   = 1 - jnp.minimum(jnp.abs(row_iota_e - i), 1)
                    row_f32    = jnp.sum(inp[:, :num_experts].astype(jnp.float32)
                                         * row_eq_i[:, :num_experts].astype(jnp.float32), axis=0)
                    row_max    = jnp.max(row_f32).astype(jnp.float32)
                    row_argmax = jnp.argmax(row_f32).astype(jnp.int32)
                    row_eq_i_k = 1 - jnp.minimum(jnp.abs(row_iota_k - i), 1)
                    top_k_l = (top_k_l * (1 - row_eq_i_k).astype(jnp.bfloat16)
                               + row_eq_i_k.astype(jnp.bfloat16) * row_max.astype(jnp.bfloat16))
                    top_k_idx = top_k_idx * (1 - row_eq_i_k) + row_argmax * row_eq_i_k
                    expert_hit = (expert_iota == row_argmax).astype(jnp.int32)  # (E,)
                    hit_m      = hit_m + row_eq_i * expert_hit                  # (bt,E)*(E,)→(bt,E)
                    return (top_k_l, top_k_idx, hit_m, inp)
                init_carry = (
                    jnp.zeros(padded_k_shape, jnp.bfloat16),
                    jnp.zeros(padded_k_shape, jnp.int32),
                    jnp.zeros(input.shape, jnp.int32),
                    input,
                )
                top_k_logits, top_k_indices, hit_mask, _ = lax.fori_loop(
                    0, bt_local, _row_body, init_carry)

            top_k_logits_lst.append(top_k_logits)
            k_eq_iota   = 1 - jnp.minimum(jnp.abs(padded_k_iota - k_id), 1)
            t2e_routing = t2e_routing * (1 - k_eq_iota) + k_eq_iota * top_k_indices
            t2e        += hit_mask
            if k_id != top_k_n - 1:
                _large_neg = jnp.array(jnp.finfo(input.dtype).min, dtype=input.dtype)
                input = input + hit_mask.astype(input.dtype) * _large_neg

        if renorm:
            # Build scale_mat via fori_loop: one WhileOp instead of bt_local static iters.
            inv_sum = 1.0 / top_k_logits_sum  # (bt_local,) float32

            def _scale_body(i, scale_mat):
                row_eq_i_n = 1 - jnp.minimum(jnp.abs(row_iota_k - i), 1)
                # Use row selector to pick inv_sum[i] without gather_p
                inv_i = jnp.sum(inv_sum * row_eq_i_n[:, 0].astype(jnp.float32)).astype(jnp.bfloat16)
                return (scale_mat * (1 - row_eq_i_n).astype(jnp.bfloat16)
                        + row_eq_i_n.astype(jnp.bfloat16) * inv_i)

            scale_mat = lax.fori_loop(
                0, bt_local, _scale_body, jnp.zeros(padded_k_shape, jnp.bfloat16))
            for k_id in range(top_k_n):
                top_k_logits_lst[k_id] = top_k_logits_lst[k_id] * scale_mat

        expert_sizes  = jnp.sum(t2e, axis=0)
        expert_starts = jnp.zeros_like(expert_sizes)
        return top_k_logits_lst, t2e_routing, expert_sizes, expert_starts

    # ------------------------------------------------------------------ #
    # Load precomputed ring-reduce metadata from HBM (v222)
    #
    # v219–v221 used a 31-step ICI ring all-reduce (all_reduce_metadata) to
    # distribute per-device expert counts to all EP peers, requiring
    # 31 × num_bt × num_devices = 31 × 8192 × 32 ≈ 8.4M semaphore_signal ops
    # per kernel call.  Inside a lax.fori_loop (while_loop) this causes ICI
    # deadlocks at EP=32 scale: signals arrive out-of-order → semaphore_wait
    # never unblocks → 60-second XLA collective timeout → os._exit(1).
    #
    # v222 fix: precompute the ring output in JAX (lax.all_gather) before
    # calling pallas_call, store in HBM, load here per bt tile.  Zero ICI ops
    # inside the while_loop for metadata; all ICI moved to a single XLA
    # all_gather at the JAX level.
    # ------------------------------------------------------------------ #
    def load_precomp_metadata(bt_id, t2e_routing):
        send_sem     = send_sems.at[0]
        d2e_count_sz = num_devices * local_num_experts   # EP × E_local, static int
        precomp_off  = bt_id * padded_num_experts
        d2e_off      = bt_id * d2e_count_sz

        def _load(t2e_vmem, offsets_vmem, esz_vmem, est_vmem, d2e_vmem):
            # Zero-init per-expert offset counters (reset each bt tile)
            offsets_vmem[...] = jnp.zeros_like(offsets_vmem)
            offsets_copy = pltpu.async_copy(
                src_ref=offsets_vmem, dst_ref=expert_offsets_x2_smem, sem=send_sem)
            # Local routing table for scatter
            t2e_vmem[...] = t2e_routing.reshape(bt * padded_top_k)
            t2e_copy = pltpu.async_copy(
                src_ref=t2e_vmem, dst_ref=t2e_routing_x2_smem, sem=send_sem)
            # Load precomputed expert_sizes, expert_starts, d2e_count from HBM → VMEM
            pltpu.async_copy(
                src_ref=expert_sizes_precomp_hbm.at[pl.ds(precomp_off, padded_num_experts)],
                dst_ref=esz_vmem, sem=send_sem).wait()
            pltpu.async_copy(
                src_ref=expert_starts_precomp_hbm.at[pl.ds(precomp_off, padded_num_experts)],
                dst_ref=est_vmem, sem=send_sem).wait()
            pltpu.async_copy(
                src_ref=d2e_count_compact_hbm.at[pl.ds(d2e_off, d2e_count_sz)],
                dst_ref=d2e_vmem, sem=send_sem).wait()
            # VMEM → SMEM for all metadata
            esz_smem  = pltpu.async_copy(src_ref=esz_vmem, dst_ref=expert_sizes_x2_smem,  sem=send_sem)
            est_smem  = pltpu.async_copy(src_ref=est_vmem, dst_ref=expert_starts_x2_smem, sem=send_sem)
            d2e_smem  = pltpu.async_copy(src_ref=d2e_vmem, dst_ref=d2e_count_compact_smem, sem=send_sem)
            t2e_copy.wait()
            offsets_copy.wait()
            esz_smem.wait()
            est_smem.wait()
            d2e_smem.wait()

        pl.run_scoped(
            _load,
            pltpu.VMEM((bt * padded_top_k,),             t2e_routing_x2_smem.dtype),
            pltpu.VMEM(expert_offsets_x2_smem.shape,     expert_offsets_x2_smem.dtype),
            pltpu.VMEM(expert_sizes_x2_smem.shape,       expert_sizes_x2_smem.dtype),
            pltpu.VMEM(expert_starts_x2_smem.shape,      expert_starts_x2_smem.dtype),
            pltpu.VMEM((d2e_count_sz,),                  jnp.int32),
        )

    # ------------------------------------------------------------------ #
    # A2A scatter helpers
    # ------------------------------------------------------------------ #
    def start_a2a_scatter(bid, e_sem_id, local_e_id):
        # Static Python loops: bt_t_id/k_id are Python ints → static SMEM index → no SC GatherOp.
        # Matches inference kernel (kernel.py) exactly.
        send_sz = jnp.int32(0)
        for bt_t_id in range(bt):
            for k_id in range(top_k):
                e_id      = t2e_routing_x2_smem[bt_t_id * padded_top_k + k_id]
                is_active = e_id % local_num_experts == local_e_id
                recv_id   = e_id // local_num_experts
                offset    = expert_offsets_x2_smem[e_id]
                sz        = lax.select(is_active, jnp.int32(1), jnp.int32(0))
                is_local  = recv_id == my_id
                local_sz  = lax.select(is_local, sz, jnp.int32(0))
                remote_sz = lax.select(is_local, jnp.int32(0), sz)
                expert_offsets_x2_smem[e_id] = offset + local_sz + remote_sz
                start_off = expert_starts_x2_smem[e_id] + offset
                t_id      = bt * bid + bt_t_id
                pltpu.make_async_copy(
                    src_ref=tokens_hbm.at[pl.ds(t_id, local_sz)],
                    dst_ref=a2a_s_x2_vmem.at[e_sem_id, pl.ds(start_off, local_sz)],
                    sem=recv_sems.at[e_sem_id],
                ).start()
                pltpu.make_async_remote_copy(
                    src_ref=tokens_hbm.at[pl.ds(t_id, remote_sz)],
                    dst_ref=a2a_s_x2_vmem.at[e_sem_id, pl.ds(start_off, remote_sz)],
                    send_sem=send_sems.at[e_sem_id],
                    recv_sem=recv_sems.at[e_sem_id],
                    device_id=get_mesh_device_id(recv_id),
                    device_id_type=pltpu.DeviceIdType.MESH,
                ).start()
                send_sz = send_sz + remote_sz
        a2a_s_sends_x2_smem[e_sem_id] = send_sz

    def wait_a2a_scatter_recv(bid, e_sem_id, local_e_id):
        del bid
        # v222: compute sz from compact SMEM — fully static indices, no SC GatherOp.
        sz = jnp.int32(0)
        for _i in range(num_devices):
            sz = sz + d2e_count_compact_smem[_i * local_num_experts + local_e_id]
        pltpu.make_async_copy(
            src_ref=a2a_s_x2_vmem.at[e_sem_id, pl.ds(0, sz)],
            dst_ref=a2a_s_x2_vmem.at[e_sem_id, pl.ds(0, sz)],
            sem=recv_sems.at[e_sem_id],
        ).wait()

    def wait_a2a_scatter_send(bid, e_sem_id, local_e_id):
        del bid, local_e_id
        sz = a2a_s_sends_x2_smem[e_sem_id]
        pltpu.make_async_copy(
            src_ref=a2a_s_x2_vmem.at[e_sem_id, pl.ds(0, sz)],
            dst_ref=a2a_s_x2_vmem.at[e_sem_id, pl.ds(0, sz)],
            sem=send_sems.at[e_sem_id],
        ).wait()

    # ------------------------------------------------------------------ #
    # A2A gather helpers
    # ------------------------------------------------------------------ #
    def start_a2a_gather(bid, e_sem_id, local_e_id):
        del bid
        my_e_id = my_id * local_num_experts + local_e_id
        # v222: compact SMEM layout — recv_id * local_num_experts + local_e_id
        # Both recv_id (Python int, static for loop) and local_e_id (Python int)
        # are static → fully static index → zero SC GatherOps on d2e_count.
        start = jnp.int32(0)
        for recv_id in range(num_devices):
            sz        = d2e_count_compact_smem[recv_id * local_num_experts + local_e_id]
            is_local  = recv_id == my_id
            local_sz  = lax.select(is_local, sz, jnp.int32(0))
            remote_sz = lax.select(is_local, jnp.int32(0), sz)
            pltpu.make_async_copy(
                src_ref=a2a_s_acc_x2_vmem.at[e_sem_id, pl.ds(start, local_sz)],
                dst_ref=a2a_g_hbm.at[my_e_id, pl.ds(0, local_sz)],
                sem=a2a_gather_sem,
            ).start()
            pltpu.make_async_remote_copy(
                src_ref=a2a_s_acc_x2_vmem.at[e_sem_id, pl.ds(start, remote_sz)],
                dst_ref=a2a_g_hbm.at[my_e_id, pl.ds(0, remote_sz)],
                send_sem=send_sems.at[e_sem_id],
                recv_sem=a2a_gather_sem,
                device_id=get_mesh_device_id(recv_id),
                device_id_type=pltpu.DeviceIdType.MESH,
            ).start()
            start = start + sz

    def wait_a2a_gather_send(bid, e_sem_id, local_e_id):
        del bid
        my_e_id  = my_id * local_num_experts + local_e_id
        # v222: compute sz and local_sz from compact SMEM — fully static indices.
        # sz = sum of all sources' counts for my local expert local_e_id.
        # local_sz = this device's own count for local_e_id.
        sz       = jnp.int32(0)
        local_sz = jnp.int32(0)
        for _i in range(num_devices):
            val = d2e_count_compact_smem[_i * local_num_experts + local_e_id]
            sz       = sz + val
            local_sz = lax.select(my_id == _i, val, local_sz)
        remote_sz = sz - local_sz
        is_valid  = jnp.logical_and(0 <= local_e_id, local_e_id < local_num_experts)
        remote_sz = lax.select(is_valid, remote_sz, 0)
        ref = a2a_g_hbm.reshape(num_experts * bt, t_packing, hidden_size // t_packing)
        pltpu.make_async_copy(
            src_ref=ref.at[pl.ds(0, remote_sz)],
            dst_ref=ref.at[pl.ds(0, remote_sz)],
            sem=send_sems.at[e_sem_id],
        ).wait()

    def wait_a2a_gather_recv_all():
        sz  = top_k * bt
        ref = a2a_g_hbm.reshape(num_experts * bt, t_packing, hidden_size // t_packing)
        pltpu.make_async_copy(
            src_ref=ref.at[pl.ds(0, sz)],
            dst_ref=ref.at[pl.ds(0, sz)],
            sem=a2a_gather_sem,
        ).wait()

    # ------------------------------------------------------------------ #
    # Weight prefetch helpers (unchanged from inference kernel)
    # ------------------------------------------------------------------ #
    def start_fetch_bw1(local_e_id, bw1_sem_id, bf_id, bd1_id):
        for p in range(t_packing):
            offset = p * h_per_t_packing + bd1_id * bd1_per_t_packing
            pltpu.make_async_copy(
                src_ref=w1_hbm.at[local_e_id, 0,
                                   pl.ds(offset, bd1_per_t_packing),
                                   pl.ds(bf_id * bf, bf)],
                dst_ref=b_w1_x2_vmem.at[bw1_sem_id, p],
                sem=local_sems.at[bw1_sem_id, 1],
            ).start()

    def start_fetch_bw2(local_e_id, bw2_sem_id, bf_id, bd2_id):
        for p in range(t_packing):
            offset = p * h_per_t_packing + bd2_id * bd2_per_t_packing
            pltpu.make_async_copy(
                src_ref=w2_hbm.at[local_e_id,
                                   pl.ds(bf_id * bf, bf),
                                   pl.ds(offset, bd2_per_t_packing)],
                dst_ref=b_w2_x2_vmem.at[bw2_sem_id, p],
                sem=local_sems.at[bw2_sem_id, 2],
            ).start()

    def start_fetch_bw3(local_e_id, bw3_sem_id, bf_id, bd3_id):
        for p in range(t_packing):
            offset = p * h_per_t_packing + bd3_id * bd1_per_t_packing
            pltpu.make_async_copy(
                src_ref=w1_hbm.at[local_e_id, 1,
                                   pl.ds(offset, bd1_per_t_packing),
                                   pl.ds(bf_id * bf, bf)],
                dst_ref=b_w3_x2_vmem.at[bw3_sem_id, p],
                sem=local_sems.at[bw3_sem_id, 3],
            ).start()

    def wait_fetch_bw1(local_e_id, bw1_sem_id, bf_id, bd1_id):
        del local_e_id, bf_id, bd1_id
        pltpu.make_async_copy(
            src_ref=b_w1_x2_vmem.at[bw1_sem_id],
            dst_ref=b_w1_x2_vmem.at[bw1_sem_id],
            sem=local_sems.at[bw1_sem_id, 1],
        ).wait()

    def wait_fetch_bw2(local_e_id, bw2_sem_id, bf_id, bd2_id):
        del local_e_id, bf_id, bd2_id
        pltpu.make_async_copy(
            src_ref=b_w2_x2_vmem.at[bw2_sem_id],
            dst_ref=b_w2_x2_vmem.at[bw2_sem_id],
            sem=local_sems.at[bw2_sem_id, 2],
        ).wait()

    def wait_fetch_bw3(local_e_id, bw3_sem_id, bf_id, bd3_id):
        del local_e_id, bf_id, bd3_id
        pltpu.make_async_copy(
            src_ref=b_w3_x2_vmem.at[bw3_sem_id],
            dst_ref=b_w3_x2_vmem.at[bw3_sem_id],
            sem=local_sems.at[bw3_sem_id, 3],
        ).wait()

    def start_fetch_next_bw(local_e_id, bw_sem_id, bf_id, bd1_id, bd2_id):
        next_bd1_id = bd1_id + 1
        next_bd2_id = bd2_id + 1
        next_sem_id = (bw_sem_id + 1) % 2
        if bf_id >= num_bf:
            return
        if next_bd1_id < num_bd1:
            start_fetch_bw1(local_e_id, next_sem_id, bf_id, next_bd1_id)
            start_fetch_bw3(local_e_id, next_sem_id, bf_id, next_bd1_id)
        elif next_bd1_id == num_bd1:
            start_fetch_bw2(local_e_id, next_sem_id, bf_id, 0)
        elif next_bd2_id < num_bd2:
            start_fetch_bw2(local_e_id, next_sem_id, bf_id, next_bd2_id)
        elif next_bd2_id == num_bd2:
            start_fetch_next_bw(local_e_id, bw_sem_id, bf_id + 1, -1, -1)
        else:
            raise RuntimeError("Unreachable")

    # ------------------------------------------------------------------ #
    # FFN compute helpers (unchanged from inference kernel)
    # ------------------------------------------------------------------ #
    def dynamic_ffn1(t_b32_vmem, w1_vmem, w1_scale_vmem, b1_vmem,
                     w3_vmem, w3_scale_vmem, b3_vmem,
                     acc1_vmem, acc3_vmem, dyn_sz, should_init):
        num_loops = cdiv(dyn_sz, btc)
        repack_ty = jnp.dtype(f"int{t_bitwidth}")

        def body(btc_id, _):
            for bd1c_id in range(cdiv(bd1, bd1c)):
                t_b32 = t_b32_vmem[
                    pl.ds(btc_id * btc, btc),
                    pl.ds(bd1c_id * bd1c_per_t_packing, bd1c_per_t_packing)]
                for p_id in range(t_packing):
                    t = pltpu.bitcast(t_b32.astype(repack_ty), t_dtype)
                    t_b32 = t_b32 >> t_bitwidth
                    for bfc_id in range(cdiv(bf, bfc)):
                        w_slices = (p_id,
                                    pl.ds(bd1c_id * bd1c_per_t_packing, bd1c_per_t_packing),
                                    pl.ds(bfc_id * bfc, bfc))
                        acc1 = jnp.dot(t, w1_vmem[*w_slices],
                                       preferred_element_type=jnp.float32)
                        acc3 = jnp.dot(t, w3_vmem[*w_slices],
                                       preferred_element_type=jnp.float32)
                        acc_slices = (pl.ds(btc_id * btc, btc),
                                      pl.ds(bfc_id * bfc, bfc))
                        if should_init and p_id == bd1c_id == 0:
                            acc1_vmem[*acc_slices] = acc1
                            acc3_vmem[*acc_slices] = acc3
                        else:
                            acc1_vmem[*acc_slices] += acc1
                            acc3_vmem[*acc_slices] += acc3
        lax.fori_loop(0, num_loops, body, None)

    def dynamic_ffn2(acc1_vmem, acc3_vmem, w2_vmem, w2_scale_vmem, b2_vmem,
                     res_b32_vmem, dyn_sz, should_init):
        num_loops = cdiv(dyn_sz, btc)

        def body(btc_id, _):
            for bd2c_id in range(cdiv(bd2, bd2c)):
                res_lst = []
                for p_id in range(t_packing):
                    res = jnp.zeros((btc, bd2c_per_t_packing), dtype=jnp.float32)
                    for bfc_id in range(cdiv(bf, bfc)):
                        acc_slices = (pl.ds(btc_id * btc, btc),
                                      pl.ds(bfc_id * bfc, bfc))
                        act = apply_act_fn(acc1_vmem[*acc_slices],
                                           acc3_vmem[*acc_slices], act_fn)
                        w2  = w2_vmem[p_id,
                                      pl.ds(bfc_id * bfc, bfc),
                                      pl.ds(bd2c_id * bd2c_per_t_packing, bd2c_per_t_packing)]
                        res += jnp.dot(act, w2, preferred_element_type=jnp.float32)
                    res = pltpu.bitcast(res, jnp.uint32)
                    if t_packing == 2:
                        res = res >> 16 << (16 * p_id)
                    res_lst.append(res)
                res = res_lst[0]
                for i in range(1, t_packing):
                    res |= res_lst[i]
                sliced = res_b32_vmem.at[
                    pl.ds(btc_id * btc, btc),
                    pl.ds(bd2c_id * bd2c_per_t_packing, bd2c_per_t_packing)]
                if should_init:
                    sliced[...] = res
                else:
                    sliced[...] = pltpu.bitcast(
                        sliced.bitcast(t_dtype)[...] + pltpu.bitcast(res, t_dtype),
                        sliced.dtype)
        lax.fori_loop(0, num_loops, body, None)

    def expert_ffn(bid, e_sem_id, local_e_id):
        bw_sem_id = 0
        a2a_s_b32    = (a2a_s_x2_vmem.bitcast(jnp.uint32)
                        .reshape(2, bt * num_devices, hidden_size // t_packing)
                        .at[e_sem_id])
        a2a_s_acc_b32 = (a2a_s_acc_x2_vmem.bitcast(jnp.uint32)
                         .reshape(2, bt * num_devices, hidden_size // t_packing)
                         .at[e_sem_id])
        b_acc1 = b_acc_vmem.at[0]   # (bt*num_devices, bf) bfloat16
        b_acc3 = b_acc_vmem.at[1]
        # v222: compute dyn_sz from compact SMEM — fully static indices.
        dyn_sz = jnp.int32(0)
        for _i in range(num_devices):
            dyn_sz = dyn_sz + d2e_count_compact_smem[_i * local_num_experts + local_e_id]

        for bf_id in range(num_bf):
            for bd1_id in range(num_bd1):
                start_fetch_next_bw(local_e_id, bw_sem_id, bf_id, bd1_id, 0)
                wait_fetch_bw1(local_e_id, bw_sem_id, bf_id, bd1_id)
                wait_fetch_bw3(local_e_id, bw_sem_id, bf_id, bd1_id)
                dynamic_ffn1(
                    t_b32_vmem=a2a_s_b32.at[
                        ..., pl.ds(bd1_id * bd1_per_t_packing, bd1_per_t_packing)],
                    w1_vmem=b_w1_x2_vmem.at[bw_sem_id],
                    w1_scale_vmem=None,
                    b1_vmem=None,
                    w3_vmem=b_w3_x2_vmem.at[bw_sem_id],
                    w3_scale_vmem=None,
                    b3_vmem=None,
                    acc1_vmem=b_acc1,
                    acc3_vmem=b_acc3,
                    dyn_sz=dyn_sz,
                    should_init=(bd1_id == 0),
                )
                bw_sem_id = (bw_sem_id + 1) % 2
            for bd2_id in range(num_bd2):
                start_fetch_next_bw(local_e_id, bw_sem_id, bf_id, num_bd1, bd2_id)
                wait_fetch_bw2(local_e_id, bw_sem_id, bf_id, bd2_id)
                if bf_id == bd2_id == 0:
                    wait_a2a_gather_send(bid, e_sem_id, local_e_id - 2)
                dynamic_ffn2(
                    acc1_vmem=b_acc1,
                    acc3_vmem=b_acc3,
                    w2_vmem=b_w2_x2_vmem.at[bw_sem_id],
                    w2_scale_vmem=None,
                    b2_vmem=None,
                    res_b32_vmem=a2a_s_acc_b32.at[
                        ..., pl.ds(bd2_id * bd2_per_t_packing, bd2_per_t_packing)],
                    dyn_sz=dyn_sz,
                    should_init=(bf_id == 0),
                )
                bw_sem_id = (bw_sem_id + 1) % 2

    # ------------------------------------------------------------------ #
    # Output accumulation (bt_acc) — flat a2a_g_hbm reshape (adaptation 6)
    # ------------------------------------------------------------------ #
    def bt_acc(bid, top_k_logits_lst):
        # Flat reshape: (E, bt, pack, D//pack) → (E*bt, pack, D//pack)
        # so e_id*bt+offset is a single 1-D index (no SC gather on 2-D index).
        a2a_g_flat = a2a_g_hbm.reshape(
            num_experts * bt, t_packing, hidden_size // t_packing)
        # Static Python loops: bt_t_id/k_id are Python ints → static SMEM index → no SC GatherOp.
        # Matches inference kernel (kernel.py) exactly.
        for bt_t_id in range(bt):
            for k_id in range(top_k):
                e_id   = t2e_routing_x2_smem[bt_t_id * padded_top_k + k_id]
                offset = expert_offsets_x2_smem[padded_num_experts + e_id]
                expert_offsets_x2_smem[padded_num_experts + e_id] = offset + jnp.int32(1)
                pltpu.make_async_copy(
                    src_ref=a2a_g_flat.at[pl.ds(e_id * bt + offset, 1)],
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
            acc = a2a_g_acc_vmem[k_id].reshape(bt, hidden_size)
            acc *= broadcast_minor(top_k_logits_lst[k_id], acc.shape)
            output = acc if output is None else output + acc
        return output.astype(output_hbm.dtype)

    # ------------------------------------------------------------------ #
    # Output DMA helpers
    # ------------------------------------------------------------------ #
    def start_send_bo(bid, priority=0):
        sem_id = bid % 2
        pltpu.make_async_copy(
            src_ref=b_output_x2_vmem.at[sem_id],
            dst_ref=output_hbm.at[pl.ds(bid * bt, bt)],
            sem=local_sems.at[sem_id, 4],
        ).start(priority=priority)

    def wait_send_bo(bid):
        is_valid = jnp.logical_and(0 <= bid, bid < num_bt)
        sz       = pl.multiple_of(lax.select(is_valid, bt, 0), bt)
        sem_id   = (bid + 2) % 2
        pltpu.make_async_copy(
            src_ref=output_hbm.at[pl.ds(0, sz)],
            dst_ref=output_hbm.at[pl.ds(0, sz)],
            sem=local_sems.at[sem_id, 4],
        ).wait()

    # ------------------------------------------------------------------ #
    # Kernel body — single-body with lax.fori_loop (v216: grid lift reverted)
    #
    # v216 rationale: the grid=(num_bt,) caused ICI barrier deadlocks because
    # pltpu.get_barrier_semaphore() is shared across all grid cells on a device.
    # Signals from cell K bleed into cell K' waits on peer devices → 60-second
    # collective timeout → exit(1).  Single-body lax.fori_loop is the inference
    # kernel approach (kernel.py) and is known-correct for ICI barriers.
    # ------------------------------------------------------------------ #

    ### ------- Kernel start ------- ###
    sync_barrier()
    start_fetch_b_gating(0)

    def run_per_bt(bt_id, _):
        bt_sem_id  = bt_id % 2
        next_bt_id = bt_id + 1

        # Pipeline: prefetch gating for the NEXT bt tile, then wait for THIS one.
        # Guard: with Python for loop, bt_id is a static int so next_bt_id = bt_id+1
        # may equal num_bt.  pl.ds(num_bt*bt, 0) fails static bounds check even
        # though size=0.  The original lax.fori_loop had dynamic bt_id so Pallas
        # deferred the check to runtime (where is_valid=False made sz=0, no-op).
        if next_bt_id < num_bt:
            start_fetch_b_gating(next_bt_id)
        wait_fetch_b_gating(bt_id)

        b_gating       = b_gating_x2_vmem[bt_sem_id]
        b_gating_score = apply_scoring_fn(scoring_fn, b_gating)
        top_k_logits_lst, t2e_routing, _expert_sizes, _expert_starts = get_top_k(
            b_gating_score, top_k, renormalize_topk_logits)

        # v222: load precomputed metadata from HBM (no ICI ring).
        # Eliminates 31 × num_bt × 32 = ~8.4M semaphore_signal ops per kernel call.
        # No sync_barrier needed: load is local HBM→VMEM→SMEM DMA on each device
        # independently; scatter can start as soon as this device's metadata is ready.
        load_precomp_metadata(bt_id, t2e_routing)

        # Start A2A scatter for the first expert.
        # e_sem_id is always a static Python int: local_num_experts is even
        # (E_local = E_global/EP = 256/32 = 8) so alternating 0→1→...→0 after
        # local_num_experts flips.  Outer lax.fori_loop carry is constant 0.
        start_a2a_scatter(bt_id, 0, 0)

        # Static Python for loop (v221): eliminates inner lax.fori_loop.
        # Static loop makes all e_sem_id values Python ints (0,1,0,1,...)
        # → all semaphore accesses use static indices → no dynamic SC GatherOp.
        # v225: sync_barrier() uses pl.run_scoped(REGULAR) — each call site in
        # this unrolled loop is a DISTINCT HLO node with its own SMEM slot.
        e_sem_id = 0          # static Python int, alternates each iteration
        for local_e_id in range(local_num_experts):
            next_e_sem_id   = 1 - e_sem_id        # static Python int
            next_local_e_id = local_e_id + 1       # static Python int
            start_fetch_bw1(local_e_id, bw1_sem_id=0, bf_id=0, bd1_id=0)
            start_fetch_bw3(local_e_id, bw3_sem_id=0, bf_id=0, bd3_id=0)
            if next_local_e_id < local_num_experts:
                start_a2a_scatter(bt_id, next_e_sem_id, next_local_e_id)
            wait_a2a_scatter_recv(bt_id, e_sem_id, local_e_id)
            expert_ffn(bt_id, e_sem_id, local_e_id)
            start_a2a_gather(bt_id, e_sem_id, local_e_id)
            wait_a2a_scatter_send(bt_id, e_sem_id, local_e_id)
            sync_barrier()
            e_sem_id = next_e_sem_id

        # After local_num_experts=8 (even) iters starting from 0: e_sem_id=0.
        # last-2 expert (index 6) used e_sem_id=0 (even); last (index 7) used 1.
        wait_a2a_gather_recv_all()
        sync_barrier()

        output = bt_acc(bt_id, top_k_logits_lst)

        wait_send_bo(bt_id - 2)
        b_output_x2_vmem[bt_sem_id] = output
        start_send_bo(bt_id)

        wait_a2a_gather_send(bt_id, e_sem_id=(local_num_experts - 2) % 2,
                             local_e_id=local_num_experts - 2)
        wait_a2a_gather_send(bt_id, e_sem_id=(local_num_experts - 1) % 2,
                             local_e_id=local_num_experts - 1)
        sync_barrier()
        return jnp.int32(0)

    # v225: unroll outer bt_id loop to static Python for.
    # With lax.fori_loop the while_loop body is compiled ONCE; all 10
    # sync_barrier() calls per tile share a single pl.run_scoped SMEM slot
    # (the same slot is reused each while_loop iteration), defeating the
    # unique-semaphore fix.  Python for loop creates num_bt×10 = 80 DISTINCT
    # HLO call sites, each with its own unique SMEM slot from pl.run_scoped.
    for bt_id in range(num_bt):
        run_per_bt(bt_id, jnp.int32(0))

    wait_send_bo(num_bt - 2)
    wait_send_bo(num_bt - 1)

    ### ------- Kernel end ------- ###


# ---------------------------------------------------------------------------
# Public API: fused EP MoE forward for training (grid-based, fast compile)
# ---------------------------------------------------------------------------

def fused_ep_moe_fwd_train_v1(
    tokens,         # (T_local, D)  — after EP all_gather by caller
    w1,             # (E_local, 2, D, F_shard) — gate+up stacked, FSDP-sharded
    w2,             # (E_local, F_shard, D)     — down, FSDP-sharded
    gating_output,  # (T_local, E_global)       — biased scores
    top_k: int,
    *,
    ep_axis_name: str,
    act_fn: str = "silu",
    scoring_fn: str = "identity",
    renormalize_topk_logits: bool = True,
    # Training mesh axes not visible inside pallas_call (outside shard_map scope).
    # dp=0 is a static prefix.  tp is handled dynamically via tp_axis_name when
    # "tp" is an active axis in the enclosing shard_map (use a P("tp") dummy input).
    extra_device_id_prefix: tuple = (),  # e.g. (dp_rank=0,) prepended
    extra_device_id_suffix: tuple = (),  # fallback static tp suffix (TP=1 or lax.cond dispatch)
    tp_axis_name: str | None = None,     # unused; kept for backward compat
    non_ep_axis_name: str = "fsdp",
    non_ep_first: bool = False,           # training mesh: ep first, fsdp second
    tp_rank_arr: "jax.Array | None" = None,  # (1,) int32 JAX-traced tp_rank (v193/v198; avoid in v199+)
    collective_id: int = 0,              # Mosaic ICI collective ID; use 0 for tp=0 branch, 1 for tp=1
    bt: int | None = None,
    bd1: int | None = None,
    bd2: int | None = None,
    btc: int | None = None,
    bd1c: int | None = None,
    bd2c: int | None = None,
    top_k_indices_precomputed=None,       # (T_local, K) int32
    top_k_weights_precomputed=None,       # (T_local, K) float32
    E_global_override: int | None = None,
) -> jax.Array:
    """Pallas MoE forward for training: FSDP-sharded weights, single-body lax.fori_loop kernel.

    Called from within shard_map(ep, fsdp).  Tokens are already EP-all_gathered
    by the caller (shape T_local = T/FSDP).  Returns a partial output; caller must:

        out = lax.psum_scatter(out, ep_axis_name, scatter_dimension=0, tiled=True)
        out = lax.psum(out, fsdp_axis_name)
    """
    T_local, D = tokens.shape
    E_local, _, _, F_shard = w1.shape      # w1: (E_local, 2, D, F_shard)

    # ----- Optional precomputed routing -----
    if top_k_indices_precomputed is not None:
        assert top_k_weights_precomputed is not None and E_global_override is not None
        E_global = E_global_override
        fi = top_k_indices_precomputed.astype(jnp.int32)
        fw = top_k_weights_precomputed.astype(jnp.float32)
        neg_inf   = jnp.finfo(jnp.float32).min
        synthetic = jnp.full((T_local, E_global), neg_inf, dtype=jnp.float32)
        flat_t    = jnp.repeat(jnp.arange(T_local, dtype=jnp.int32), top_k)
        flat_e    = fi.reshape(-1)
        flat_w    = fw.reshape(-1)
        synthetic = synthetic.at[flat_t, flat_e].set(flat_w)
        gating_output = synthetic
        renormalize_topk_logits = False
    else:
        assert gating_output is not None
        E_global = (E_global_override if E_global_override is not None
                    else gating_output.shape[1])

    num_devices = lax.axis_size(ep_axis_name)

    t_dtype   = tokens.dtype
    t_packing = get_dtype_packing(t_dtype)

    # ----- Block sizes -----
    assert F_shard % 128 == 0, f"F_shard={F_shard} must be 128-aligned (FSDP ≤ F//128)"
    bf  = F_shard
    bfc = F_shard

    if bd1 is None:
        bd1 = D  # full hidden dim: num_bd1=1, minimizes Python loop unroll count
        bd1 = (bd1 // (t_packing * 128)) * (t_packing * 128)
        bd1 = max(bd1, t_packing * 128)
    if bd1c is None:
        bd1c = bd1
    if bd2 is None:
        bd2 = D  # full hidden dim: num_bd2=1, minimizes Python loop unroll count
        bd2 = (bd2 // (t_packing * 128)) * (t_packing * 128)
        bd2 = max(bd2, t_packing * 128)
    if bd2c is None:
        bd2c = bd2
    if bt is None:
        bt = min(128, T_local)
        bt = max(bt, t_packing)
        # Physical VMEM per kernel on v7x = 64 MB.
        # Reserve 75% for the two dominant a2a buffers → 48 MB budget.
        _vmem_limit     = 64 * 1024 * 1024
        _a2a_bytes_per_bt = 2 * num_devices * t_packing * (D // t_packing) * 2
        _bt_max = int(_vmem_limit * 0.75) // (2 * _a2a_bytes_per_bt)
        _bt_max = max((_bt_max // t_packing) * t_packing, t_packing)
        bt = min(bt, _bt_max)
        bt = min(bt, 8)   # hard cap: limits Python loop unroll count in kernel body
        while T_local % bt != 0 and bt > t_packing:
            bt -= t_packing
    if btc is None:
        btc = bt

    num_bt = T_local // bt   # loop iterations; T_local % bt == 0 by construction

    # v216: scalar_prefetch removed (grid lift reverted).
    # tp_rank is now a static suffix in extra_device_id_suffix=(tp_rank_int,).
    # No tp_rank_arr needed.

    padded_num_experts = align_to(E_global, 128)
    padded_top_k       = align_to(top_k, 128)

    if padded_num_experts != E_global:
        gating_output = jnp.pad(
            gating_output,
            ((0, 0), (0, padded_num_experts - E_global)),
            constant_values=-jnp.inf)

    gating_output = gating_output.astype(t_dtype)
    tokens_packed = tokens.reshape(T_local, t_packing, D // t_packing)

    def _full_hbm(shape):
        """Full-array HBM block spec — kernel sees entire array (no auto-slicing)."""
        ndim = len(shape)
        return pl.BlockSpec(
            block_shape=shape,
            index_map=lambda *_: (0,) * ndim,
            memory_space=pltpu.MemorySpace.HBM,
        )

    kernel_fn = functools.partial(
        _fused_ep_moe_kernel_train,
        top_k=top_k,
        renormalize_topk_logits=renormalize_topk_logits,
        ep_axis_name=ep_axis_name,
        act_fn=act_fn,
        scoring_fn=scoring_fn,
        subc_quant_w1_sz=None,
        subc_quant_w2_sz=None,
        non_ep_axis_name=non_ep_axis_name,
        non_ep_first=non_ep_first,
        extra_device_id_prefix=extra_device_id_prefix,
        extra_device_id_suffix=extra_device_id_suffix,
        tp_axis_name=tp_axis_name,
        bt=bt, bf=bf, bd1=bd1, bd2=bd2,
        btc=btc, bfc=bfc, bd1c=bd1c, bd2c=bd2c,
    )

    # v222: precompute ICI ring metadata in JAX (lax.all_gather) before pallas_call.
    # Previously all_reduce_metadata did a 31-step ring inside the kernel's
    # lax.fori_loop (while_loop), causing ICI deadlocks at EP=32.
    # Now: one JAX all_gather outside the kernel replaces 31×8192×32 ICI signals.
    assert top_k_indices_precomputed is not None, (
        "v222 kernel requires top_k_indices_precomputed (precomputed routing)")
    ep_size = num_devices   # Python int
    fi_bt_flat = fi.reshape(num_bt, bt * top_k).astype(jnp.int32)  # (num_bt, bt*K)

    # Per-bt-tile expert counts on THIS device: local_send_counts[b, e] = number of
    # bt-tile-b tokens routed to global expert e by this EP device.
    def _count_tile(idx):   # idx: (bt*K,) int32
        return jax.ops.segment_sum(
            jnp.ones(bt * top_k, jnp.int32), idx, num_segments=padded_num_experts)

    local_send_counts = jax.vmap(_count_tile)(fi_bt_flat)  # (num_bt, padded_num_experts)

    # All-gather across EP: all_send_counts[src, b, e] = tokens from src in bt tile b
    # going to global expert e.  Shape: (ep_size, num_bt, padded_num_experts).
    # Temporary 256 MB allocation (32 × 8192 × 256 × 4B); freed after sum below.
    all_send_counts = lax.all_gather(
        local_send_counts, ep_axis_name, axis=0, tiled=False)  # (EP, num_bt, E)

    # Global expert sizes = total tokens to each expert per bt tile.
    expert_sizes_precomp = jnp.sum(all_send_counts, axis=0)   # (num_bt, padded_num_experts)

    # Global expert starts = exclusive prefix sum over expert axis.
    expert_starts_precomp = (
        jnp.cumsum(expert_sizes_precomp, axis=1) - expert_sizes_precomp
    )  # (num_bt, padded_num_experts)

    # Compact d2e_count for MY local experts only: (num_bt, ep_size * E_local).
    # d2e_count_compact[b, src * E_local + e_local] = tokens from device src, bt tile b,
    # going to my local expert e_local.
    ep_rank = lax.axis_index(ep_axis_name)   # dynamic (0..ep_size-1)
    d2e_raw = lax.dynamic_slice_in_dim(
        all_send_counts, ep_rank * E_local, E_local, axis=2)  # (ep_size, num_bt, E_local)
    d2e_flat = d2e_raw.transpose(1, 0, 2).reshape(
        num_bt, ep_size * E_local)  # (num_bt, ep_size * E_local)

    # Flatten to 1-D for pallas_call HBM inputs (accessed via pl.ds per bt tile).
    expert_sizes_precomp_1d  = expert_sizes_precomp.reshape(-1)   # (num_bt * padded_num_experts,)
    expert_starts_precomp_1d = expert_starts_precomp.reshape(-1)  # (num_bt * padded_num_experts,)
    d2e_count_compact_1d     = d2e_flat.reshape(-1)               # (num_bt * ep_size * E_local,)

    # v232: pl.run_scoped(REGULAR) + collective_id=None + Python for loop.
    # v231: get_barrier_semaphore() dummy + pl.run_scoped — C++ fatal (unused barrier sem).
    # v230: get_barrier_semaphore() (used), EP=8 — SIGABRT 18:04:49 all pods (aliasing).
    # v229: get_barrier_semaphore() (used), EP=16 — HBM OOM (FSDP=8).
    # v225–v228: pl.run_scoped, collective_id=0, no get_barrier_semaphore() — ValueError.
    # v224: lax.fori_loop, get_barrier_semaphore(), EP=32 — hung (ICI aliasing).
    NUM_BT_CHUNK = 8   # bt_tiles per pallas_call (unchanged)
    assert num_bt % NUM_BT_CHUNK == 0, (
        f"num_bt={num_bt} must be divisible by NUM_BT_CHUNK={NUM_BT_CHUNK}")
    num_chunks = num_bt // NUM_BT_CHUNK   # = 1024 at T_local=65536, bt=8

    chunk_tok = NUM_BT_CHUNK * bt          # tokens per chunk (= 64)
    chunk_E   = NUM_BT_CHUNK * padded_num_experts
    chunk_d2e = NUM_BT_CHUNK * ep_size * E_local

    scratch_shapes_v223 = [
        pltpu.SMEM((bt * padded_top_k,), jnp.int32),
        pltpu.SMEM((num_devices * E_local,), jnp.int32),
        pltpu.SMEM((2 * padded_num_experts,), jnp.int32),
        pltpu.SMEM((padded_num_experts,), jnp.int32),
        pltpu.SMEM((padded_num_experts,), jnp.int32),
        pltpu.SMEM((2,), jnp.int32),
        pltpu.VMEM((2, bt * num_devices, t_packing, D // t_packing), t_dtype),
        pltpu.VMEM((2, bt * num_devices, t_packing, D // t_packing), t_dtype),
        pltpu.VMEM((top_k, bt, t_packing, D // t_packing), t_dtype),
        pltpu.VMEM((2, bt, padded_num_experts), t_dtype),
        pltpu.VMEM((2, bt, D), t_dtype),
        pltpu.VMEM((2, t_packing, bd1 // t_packing, bf), w1.dtype),
        pltpu.VMEM((2, t_packing, bd1 // t_packing, bf), w1.dtype),
        pltpu.VMEM((2, t_packing, bf, bd2 // t_packing), w2.dtype),
        None, None, None,
        None, None, None,
        pltpu.VMEM((2, bt * num_devices, bf), jnp.float32),
        pltpu.SemaphoreType.DMA((2, 5)),
        pltpu.SemaphoreType.DMA((2,)),
        pltpu.SemaphoreType.DMA((2,)),
        pltpu.SemaphoreType.DMA,
        pltpu.SemaphoreType.DMA,
    ]

    fused_moe_chunk = pl.pallas_call(
        kernel_fn,   # num_bt re-derived inside kernel as tokens_hbm.shape[0] // bt
        out_shape=jax.ShapeDtypeStruct((chunk_tok, D), t_dtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            in_specs=[
                _full_hbm((chunk_tok, t_packing, D // t_packing)),         # tokens_hbm
                _full_hbm((E_local, 2, D, F_shard)),                        # w1_hbm
                _full_hbm((E_local, F_shard, D)),                           # w2_hbm
                None, None, None, None,                                     # scale/bias
                _full_hbm((chunk_tok, padded_num_experts)),                  # gating_hbm
                _full_hbm((E_global, bt, t_packing, D // t_packing)),       # a2a_g_hbm scratch
                _full_hbm((chunk_E,)),                                      # expert_sizes_precomp_hbm
                _full_hbm((chunk_E,)),                                      # expert_starts_precomp_hbm
                _full_hbm((chunk_d2e,)),                                    # d2e_count_compact_hbm
            ],
            out_specs=_full_hbm((chunk_tok, D)),
            scratch_shapes=scratch_shapes_v223,
        ),
        compiler_params=pltpu.CompilerParams(
            collective_id=None,   # v232: None required for pl.run_scoped(REGULAR) ICI
            vmem_limit_bytes=64 * 1024 * 1024,
        ),
        name=f"fused-moe-fwd-train-v1-bt{bt}-bf{bf}-c8u-v232",
    )

    # lax.scan over num_chunks=1024: compact HLO (one while_loop, one pallas_call
    # body).  Each iteration processes chunk_tok=64 tokens with 8 bt_tiles,
    # producing 2,560 semaphore_signals — matches inference kernel scale.
    def chunk_body(_, ci):
        tok_off = ci * chunk_tok
        e_off   = ci * chunk_E
        d2e_off = ci * chunk_d2e
        a2a_sc  = jnp.zeros((E_global, bt, t_packing, D // t_packing), t_dtype)
        out_c = fused_moe_chunk(
            lax.dynamic_slice_in_dim(tokens_packed,             tok_off, chunk_tok,  0),
            w1, w2, None, None, None, None,
            lax.dynamic_slice_in_dim(gating_output,             tok_off, chunk_tok,  0),
            a2a_sc,
            lax.dynamic_slice_in_dim(expert_sizes_precomp_1d,  e_off,   chunk_E,    0),
            lax.dynamic_slice_in_dim(expert_starts_precomp_1d, e_off,   chunk_E,    0),
            lax.dynamic_slice_in_dim(d2e_count_compact_1d,     d2e_off, chunk_d2e,  0),
        )
        return _, out_c   # out_c: (chunk_tok, D)

    _, outputs = lax.scan(chunk_body, None, jnp.arange(num_chunks))
    # outputs: (num_chunks, chunk_tok, D) → reshape to (T_local, D)
    output = outputs.reshape(T_local, D)
    return output  # (T_local, D) partial — caller: psum_scatter(ep) + psum(fsdp)
