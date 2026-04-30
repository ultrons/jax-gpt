# Writing a Fused EP MoE Forward Kernel from Scratch in Pallas

This guide walks through writing a TPU Pallas kernel for the Mixture-of-Experts
forward pass with Expert Parallelism (EP) and ICI All-to-All communication.
Target config: DSv3 671B, EP=32, FSDP=16, TP=1 on TPU v7x.

---

## Part 1: Pallas/Mosaic Mental Model

### What Pallas is

Pallas is JAX's kernel authoring API. You write a Python function that Mosaic
(the TPU compiler) lowers to native TPU instructions. Unlike JAX, where you
describe a computation and XLA figures out tiling and scheduling, in Pallas
**you control every data movement explicitly**.

The key mental shift: in JAX you write `y = x @ w`. In Pallas you write:
"initiate DMA to bring a tile of x into VMEM, initiate DMA to bring a tile of w
into VMEM, wait for both, compute the matmul in VMEM, initiate DMA to write the
result back to HBM."

### TPU memory hierarchy

```
HBM (High Bandwidth Memory)
  ~100 GB total, ~1 TB/s bandwidth
  Off-chip relative to the TensorCore
  You never compute directly in HBM

  ↕ async DMA (initiate, do other work, then wait)

VMEM (Vector Memory)
  ~32 MB on-chip, ~10 TB/s bandwidth
  Where ALL compute happens
  Holds: weight tiles, activation tiles, accumulators, A2A buffers

  ↕ SC gather (hardware-accelerated indexed scatter/gather)

SMEM (SparseCore Memory)
  ~8 MB on-chip
  For index tables and routing metadata only — NOT for compute
  Loaded via scalar_prefetch before each grid step
```

The kernel body operates entirely on **VMEM Refs** — in-place mutable arrays.
You never touch HBM directly in the body. You initiate async DMAs to move data
between HBM and VMEM, then compute happens in VMEM.

### The grid abstraction

A Pallas kernel runs a body function over a grid of iterations:

```
grid = (num_bt,)  →  body runs num_bt times
bt_id = pl.program_id(0)  →  which iteration am I on?
```

Mosaic compiles the body **once** and instantiates it `num_bt` times. This is
fundamentally different from a Python loop (which unrolls at trace time) or a
`lax.fori_loop` (which creates a while-loop in HLO). The grid is the right
abstraction for TPU kernels.

---

## Part 2: pallas_call Structure

```python
import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

output = pl.pallas_call(
    kernel_fn,
    grid_spec=pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=1,        # how many scalar_prefetch args
        grid=(num_bt,),               # outer loop: num_bt iterations
        in_specs=[
            # BlockSpec tells Pallas how to slice each HBM input per grid step
            pl.BlockSpec((bt, D), lambda i: (i * bt, 0)),
            pl.BlockSpec((E_local, F, D), lambda i: (0, 0, 0)),  # load all
        ],
        out_specs=pl.BlockSpec((T, D), lambda i: (0, 0)),
        scratch_shapes=[
            pltpu.VMEM((2, bt, D), jnp.bfloat16),   # double-buffered VMEM
            pltpu.SMEM((E,), jnp.int32),              # routing table in SMEM
            pltpu.SemaphoreType.DMA,                  # async DMA semaphore
        ],
    ),
    out_shape=jax.ShapeDtypeStruct((T, D), jnp.bfloat16),
    compiler_params=pltpu.CompilerParams(collective_id=0),
)(scalar_arr, hbm_input_1, hbm_input_2)
```

### Kernel function signature

Arguments arrive in this strict order:
1. `scalar_prefetch` args (one per `num_scalar_prefetch`)
2. HBM inputs (one per `in_specs` entry)
3. HBM output (one per `out_specs` entry)
4. Scratch shapes (one per `scratch_shapes` entry, in order)

Keyword-only args after `*` are static compile-time constants.

```python
def kernel_fn(
    tp_rank_scalar,      # scalar_prefetch[0]: shape (1,) int32
    tokens_ref,          # HBM input 0
    weights_ref,         # HBM input 1
    output_ref,          # HBM output
    vmem_scratch_ref,    # scratch[0]: pltpu.VMEM
    smem_routing_ref,    # scratch[1]: pltpu.SMEM
    sem_ref,             # scratch[2]: pltpu.SemaphoreType.DMA
    *,
    top_k: int,          # static param
    ep_axis_name: str,   # static param
):
    bt_id = pl.program_id(0)
    ...
```

### BlockSpec and index maps

`pl.BlockSpec(block_shape, index_map)`:
- `block_shape`: the shape of the tile to load from HBM for this grid step
- `index_map(bt_id, ...) -> (i0, i1, ...)`: which tile to load

Use `None` in `block_shape` to load the entire dimension (no slicing):

```python
# Load bt rows of tokens, starting at bt_id * bt:
pl.BlockSpec((bt, D), lambda i: (i * bt, 0))

# Load all weight rows every step (weights don't change per bt_id):
pl.BlockSpec((E_local, D, F), lambda i: (0, 0, 0))
# equivalently with None:
pl.BlockSpec((None, D, F), lambda i: (0, 0, 0))
```

### pl.ds — dynamic slice

`pl.ds(start, size)` creates a dynamic slice:
- `start`: JAX scalar (dynamic, determined at runtime)
- `size`: Python int (static, must be known at compile time)

```python
vmem_ref.at[pl.ds(offset, bt)]   # rows [offset : offset+bt]
```

### pl.run_scoped — temporary VMEM allocation

For VMEM you only need within a function scope:

```python
def my_op(vmem_ref, smem_ref):
    # vmem_ref and smem_ref are allocated just for this call
    vmem_ref[...] = jnp.zeros_like(vmem_ref)
    result = vmem_ref[...] + 1.0

pl.run_scoped(
    my_op,
    pltpu.VMEM((bt, D), jnp.bfloat16),
    pltpu.SMEM((E,), jnp.int32),
)
```

---

## Part 3: Async DMA Primitives

### Local DMA (HBM → VMEM, same device)

```python
# Start an async copy from HBM to VMEM
copy = pltpu.make_async_copy(
    src_ref=hbm_ref.at[pl.ds(row_start, bt)],   # HBM source slice
    dst_ref=vmem_ref.at[buf_id],                  # VMEM destination
    sem=sem_ref.at[sem_id],                        # semaphore to use
)
copy.start()

# ... do other useful work here (overlap!) ...

# Wait for completion. This also resets the semaphore.
copy.wait()
```

You can also use `pltpu.async_copy` for VMEM → SMEM copies:

```python
pltpu.async_copy(
    src_ref=vmem_ref,
    dst_ref=smem_ref,
    sem=sem_ref,
).start()
```

### Remote DMA (HBM → HBM on another device via ICI)

This is what makes EP A2A possible. You write directly into another device's HBM:

```python
pltpu.async_remote_copy(
    src_ref=local_hbm_ref.at[pl.ds(src_offset, size)],
    dst_ref=remote_hbm_ref.at[pl.ds(dst_offset, size)],
    send_sem=send_sem_ref,    # signaled on SENDER when data has left
    recv_sem=recv_sem_ref,    # signaled on RECEIVER when data has arrived
    device_id=(jnp.int32(0), ep_rank, fsdp_rank, jnp.int32(0)),
    device_id_type=pltpu.DeviceIdType.MESH,
).start()
```

**CRITICAL semantics**:
- `recv_sem` is waited on by the **receiver**, not the sender
- The sender waits on `send_sem` to know when it's safe to reuse the source buffer
- Both devices must allocate the same `remote_hbm_ref` shape in their HBM scratch

The device ID tuple indexes into the mesh axis order. For mesh
`(dp=1, ep=32, fsdp=16, tp=1)` with `axis_names=("dp","ep","fsdp","tp")`,
device ID is `(dp_idx, ep_idx, fsdp_idx, tp_idx)`.

### Semaphores

```python
# DMA semaphore — for make_async_copy / async_remote_copy
pltpu.SemaphoreType.DMA    # declare in scratch_shapes

# Barrier semaphore — global sync across all EP devices
barrier_sem = pltpu.get_barrier_semaphore()
# Signal all other EP devices:
for i in range(num_devices):
    pltpu.semaphore_signal(
        barrier_sem, inc=1,
        device_id=get_mesh_device_id(i),
        device_id_type=pltpu.DeviceIdType.MESH,
    )
# Wait until all devices have signaled me:
pltpu.semaphore_wait(barrier_sem, num_devices)
```

### Double buffering pattern

The standard pattern for overlapping DMA and compute:

```python
# Prefetch next tile while processing current tile
buf_id = bt_id % 2       # current buffer
next_buf_id = 1 - buf_id  # prefetch target

# Start prefetch for bt_id+1
start_prefetch(bt_id + 1, next_buf_id)

# Process current buffer (overlaps with prefetch above)
compute(vmem_buf.at[buf_id])

# Wait for prefetch to complete before next iteration
wait_prefetch(next_buf_id)
```

---

## Part 4: The EP MoE Problem

### Setup

```
Config: DSv3, EP=32, FSDP=16, TP=1, 4×4×16 bodaborg
  T_fsdp   = 65,536    # tokens per device (GBS=256 × S=4096 / FSDP=16)
  D        = 7,168     # hidden dim
  E        = 256       # global experts
  E_local  = 8         # experts per EP device (E / EP = 256 / 32)
  F_shard  = 128       # intermediate dim shard (F=2048 / FSDP=16)
  K        = 8         # top-K routing

Each device owns:
  tokens:   (T_fsdp, D)           local token slice
  w1:       (E_local, 2, D, F_shard)   gate+up weights (stacked), FSDP-sharded
  w2:       (E_local, F_shard, D)      down weights, FSDP-sharded
  gating:   (T_fsdp, E)           router logit scores (pre-computed by caller)
```

### What the forward pass computes

For each token `t` routed to expert `e_k` with weight `w_k`:

```
gate_k  = token @ w1[e_k, 0, :, :]    # (D,) @ (D, F_shard) → (F_shard,)
up_k    = token @ w1[e_k, 1, :, :]    # (D,) @ (D, F_shard) → (F_shard,)
hidden  = silu(gate_k) * up_k          # SwiGLU activation
out    += weight_k * (hidden @ w2[e_k])  # (F_shard,) @ (F_shard, D) → (D,)
```

Sum over all K experts. Each device then does `psum(out, "fsdp")` to reduce
the FSDP-sharded partial results.

### Why this needs a Pallas kernel

Token `t` lives on device `d_t = t // T_fsdp`. Expert `e_k` lives on device
`d_e = e_k // E_local`. In general `d_t ≠ d_e`. You need to:

1. **Scatter**: send each token to the device that owns its assigned experts
2. **Compute**: each device runs its local experts on the received tokens
3. **Gather**: send computed results back to the token-owning devices

With JAX collectives (`all_gather`), you'd gather ALL tokens to every device
(938 MB at this config) then discard most of it — wasteful. The Pallas kernel
does a sparse A2A: each token only travels to the devices that actually need it.

---

## Part 5: Algorithm — All Five Phases

### Phase 0: Top-K Routing

For the current `bt` tokens, determine which expert each token routes to.

**Inputs**: `gating: (bt, E)` logit scores
**Outputs**:
- `t2e_routing: (bt, K)` int32 — which global expert for each (token, k)
- `expert_sizes: (E,)` int32 — how many of my tokens go to each expert
- `expert_starts: (E,)` int32 — (filled in by metadata ring, see Phase 1)

**Mosaic constraint on get_top_k**: You cannot use:
- `jnp.argmax(..., keepdims=True)` — produces unsupported shape cast
- `(bt, E)` bool masks — large bool 2-D arrays cause relayout errors at scale
- `.at[i, :].set(v)` on a JAX array — lowers to `scatter_p`, unsupported in TC

Use `lax.fori_loop` over rows with **int32 row-selector masks**:

```python
def get_top_k(gating, top_k):
    # gating: (bt, E) bfloat16
    bt_local, E = gating.shape
    row_iota_e = lax.broadcasted_iota(jnp.int32, gating.shape, 0)  # (bt, E)
    expert_iota = lax.broadcasted_iota(jnp.int32, (E,), 0)          # (E,)
    t2e = jnp.zeros(gating.shape, jnp.int32)
    routing = jnp.zeros((bt_local, top_k), jnp.int32)

    for k in range(top_k):
        def pick_row(i, carry):
            routing_k, hit_mask, inp = carry
            # Row selector: 1 where row_iota == i, else 0 (NO bool arrays)
            row_sel = 1 - jnp.minimum(jnp.abs(row_iota_e - i), 1)  # (bt, E)
            # Extract row i as a 1-D f32 vector
            row_f32 = jnp.sum(
                inp.astype(jnp.float32) * row_sel.astype(jnp.float32), axis=0
            )  # (E,)
            best_expert = jnp.argmax(row_f32).astype(jnp.int32)  # scalar
            # Update routing for row i (row-selector, no scatter_p)
            row_sel_k = 1 - jnp.minimum(
                jnp.abs(lax.broadcasted_iota(jnp.int32, (bt_local, top_k), 0) - i), 1
            )
            routing_k = routing_k * (1 - row_sel_k) + row_sel_k * best_expert
            # Mark this expert as hit for row i
            expert_hit = (expert_iota == best_expert).astype(jnp.int32)  # (E,)
            hit_mask = hit_mask + row_sel * expert_hit   # (bt,E)*(E,)→(bt,E)
            return routing_k, hit_mask, inp

        routing, hit_mask, gating = lax.fori_loop(
            0, bt_local, pick_row,
            (routing, jnp.zeros_like(t2e), gating)
        )
        t2e = t2e + hit_mask
        # Suppress selected experts for next k iteration
        _neg_inf = jnp.finfo(gating.dtype).min
        gating = gating + hit_mask.astype(gating.dtype) * _neg_inf

    expert_sizes = jnp.sum(t2e, axis=0)  # (E,)
    return routing, expert_sizes
```

### Phase 1: Metadata Ring-Reduce

Before scattering tokens, every device needs to know globally how many tokens
are coming to each of its experts (to pre-size the A2A receive buffer) and
where to write incoming tokens (offset within the buffer).

This is a ring-reduce over the EP axis using ICI:

```python
def all_reduce_metadata(my_expert_sizes, my_id, num_devices, right_id):
    # d2e_count_vmem: (EP * E,) — row my_id holds my local counts
    # After ring: every device knows total tokens per expert across all EP devices

    row_id = my_id
    reduced_sizes  = my_expert_sizes           # starts with local counts
    reduced_starts = jnp.zeros_like(my_expert_sizes)  # cumulative offset

    def ring_step(step, state):
        row_id, sizes, starts = state

        # All devices must have written their row before we read it
        sync_barrier()

        # Send my row to right neighbor; they write into the same row slot
        # (they receive into d2e_count_vmem[row_id * E : (row_id+1) * E])
        pltpu.async_remote_copy(
            src_ref=d2e_count_vmem.at[pl.ds(row_id * E, E)],
            dst_ref=d2e_count_vmem.at[pl.ds(row_id * E, E)],
            send_sem=send_sem,
            recv_sem=recv_sem,
            device_id=get_mesh_device_id(right_id),
            device_id_type=pltpu.DeviceIdType.MESH,
        ).wait()

        # Move to previous row (the one we just sent; right neighbor sent to us)
        row_id = (row_id + num_devices - 1) % num_devices
        new_sizes = d2e_count_vmem[pl.ds(row_id * E, E)]

        sizes  = sizes + new_sizes
        starts = starts + lax.select(
            my_id > step,
            new_sizes,
            jnp.zeros_like(new_sizes)
        )
        return row_id, sizes, starts

    _, reduced_sizes, reduced_starts = lax.fori_loop(
        0, num_devices - 1, ring_step,
        (row_id, reduced_sizes, reduced_starts)
    )
    return reduced_sizes, reduced_starts
    # reduced_sizes[e]  = total tokens routed to expert e across all EP devices
    # reduced_starts[e] = this device's write offset in the A2A receive buffer
```

After this phase: load `reduced_starts → expert_starts_smem` and
`reduced_sizes → expert_sizes_smem` so the SparseCore can index into them.

### Phase 2: A2A Scatter

For each expert `e`, send tokens routed to `e` to the device that owns `e`.

```python
def a2a_scatter(bt_id, e_sem_id):
    # e_sem_id alternates 0/1 per expert (double-buffered scatter buffers)
    send_sz = jnp.int32(0)

    def scatter_one_expert(e, send_sz):
        local_e_id = e % E_local
        owner_ep_rank = e // E_local

        # How many of my tokens go to expert e?
        sz = expert_sizes_smem[e]         # from routing (Phase 0)
        offset = expert_offsets_smem[e]   # running offset in scatter buffer

        is_local = (owner_ep_rank == my_id)
        local_sz  = lax.select(is_local, sz, jnp.int32(0))
        remote_sz = lax.select(is_local, jnp.int32(0), sz)

        # Compute scatter destination: reduced_starts[e] + my offset
        start_off = expert_starts_smem[e] + offset
        t_id = bt * bt_id + bt_t_id  # which token row in HBM

        # Local copy (same device): HBM → a2a_s_x2_vmem[e_sem_id]
        pltpu.make_async_copy(
            src_ref=tokens_hbm.at[pl.ds(t_id, local_sz)],
            dst_ref=a2a_s_x2_vmem.at[e_sem_id, pl.ds(start_off, local_sz)],
            sem=recv_sems.at[e_sem_id],
        ).start()

        # Remote copy: send to owner device
        pltpu.async_remote_copy(
            src_ref=tokens_hbm.at[pl.ds(t_id, remote_sz)],
            dst_ref=a2a_s_x2_vmem.at[e_sem_id, pl.ds(start_off, remote_sz)],
            send_sem=send_sems.at[e_sem_id],
            recv_sem=recv_sems.at[e_sem_id],
            device_id=get_mesh_device_id(owner_ep_rank),
            device_id_type=pltpu.DeviceIdType.MESH,
        ).start()

        return send_sz + remote_sz

    send_sz = lax.fori_loop(0, E, scatter_one_expert, send_sz)
    a2a_s_sends_smem[e_sem_id] = send_sz
```

### Phase 3: GEMM (Local Expert Computation)

Wait for scattered tokens to arrive, then run each local expert's FFN:

```python
def compute_local_experts(e_sem_id):
    # Wait for all incoming tokens (scatter phase completed)
    pltpu.semaphore_wait(recv_sems.at[e_sem_id], count=total_incoming)

    for e_local in range(E_local):
        e_global = my_id * E_local + e_local
        start = expert_starts_smem[e_global]      # where my expert's tokens start
        size  = expert_sizes_smem[e_global]        # how many tokens for this expert

        tokens_e = a2a_s_x2_vmem[e_sem_id, pl.ds(start, max_tpe)]  # (max_tpe, D)

        # Load weights for this expert (double-buffered: prefetch next while computing)
        w1_e = b_w1_x2_vmem[buf_id]   # (D, F_shard) — gate weights
        w3_e = b_w3_x2_vmem[buf_id]   # (D, F_shard) — up weights
        w2_e = b_w2_x2_vmem[buf_id]   # (F_shard, D) — down weights

        # Gate + up projection using float32 accumulator
        gate = jnp.zeros((max_tpe, F_shard), jnp.float32)
        up   = jnp.zeros((max_tpe, F_shard), jnp.float32)
        for f_tile in range(num_f_tiles):
            gate = gate + tokens_e @ w1_e[pl.ds(f_tile * bf, bf)]
            up   = up   + tokens_e @ w3_e[pl.ds(f_tile * bf, bf)]

        # SwiGLU activation
        hidden = jax.nn.silu(gate) * up  # (max_tpe, F_shard)

        # Down projection
        out = jnp.zeros((max_tpe, D), jnp.float32)
        for d_tile in range(num_d_tiles):
            out = out + hidden @ w2_e[pl.ds(d_tile * bd, bd)]

        # Store to A2A gather HBM buffer to be sent back
        a2a_g_hbm.at[e_global, pl.ds(0, max_tpe)].set(out.astype(jnp.bfloat16))
```

**Important**: zero-initialize all VMEM accumulators explicitly. Do not rely on
the buffer being clean — XLA reuses freed HBM and the contents may be garbage.

### Phase 4: A2A Gather

Send computed expert outputs back to the token-owning devices:

```python
def a2a_gather():
    def gather_one_expert(e, start):
        local_e_id = e % E_local
        owner_ep_rank = e // E_local
        sz = expert_sizes_smem[e]

        # Send expert output from owner device back to token-owning devices
        # (This runs on the owner device; sz tokens go back to their sources)
        pltpu.async_remote_copy(
            src_ref=a2a_g_hbm.at[local_e_id, pl.ds(0, sz)],
            dst_ref=a2a_g_hbm.at[e, pl.ds(0, sz)],   # at receiving device
            send_sem=send_sems.at[e_sem_id],
            recv_sem=a2a_gather_sem,
            device_id=get_mesh_device_id(token_owner_ep_rank),
            device_id_type=pltpu.DeviceIdType.MESH,
        ).start()
        return start + sz

    lax.fori_loop(0, E_local, gather_one_expert, jnp.int32(0))
```

### Phase 5: Accumulate Output

Once all gather results arrive, apply the routing weights and sum:

```python
def accumulate_output(bt_id):
    # a2a_g_acc_vmem: (K, bt, D) — one slot per top-k expert per token
    pltpu.semaphore_wait(a2a_gather_sem, count=expected_tokens)

    out = jnp.zeros((bt, D), jnp.float32)
    for k in range(K):
        # Routing weight for this k-slot
        weight_k = top_k_weights_smem[pl.ds(bt_id * bt * K + k * bt, bt)]  # (bt,)
        result_k = a2a_g_acc_vmem[k]  # (bt, D)

        # Weighted accumulation — no [:, None] (Mosaic constraint!)
        # Build weight matrix explicitly, row by row
        row_iota = lax.broadcasted_iota(jnp.int32, (bt, D), 0)
        w_mat = jnp.zeros((bt, D), jnp.float32)
        def weight_row(i, w):
            row_eq = 1 - jnp.minimum(jnp.abs(row_iota - i), 1)  # (bt, D)
            wi = jnp.sum(weight_k * (row_iota[:, 0] == i).astype(jnp.float32))
            return w + row_eq.astype(jnp.float32) * wi
        w_mat = lax.fori_loop(0, bt, weight_row, w_mat)
        out = out + w_mat * result_k.astype(jnp.float32)

    # Write to output HBM
    pltpu.make_async_copy(
        src_ref=b_output_vmem,   # VMEM holding out
        dst_ref=output_hbm.at[pl.ds(bt_id * bt, bt)],
        sem=local_sems.at[...],
    ).start()
```

---

## Part 6: Full Kernel Skeleton

```python
def _moe_kernel(
    tp_rank_scalar,         # scalar_prefetch: (1,) int32
    # HBM inputs
    tokens_hbm,             # (T_local, D) bfloat16 — packed as (T, pack, D//pack)
    w1_hbm,                 # (E_local, 2, D, F_shard) bfloat16
    w2_hbm,                 # (E_local, F_shard, D) bfloat16
    gating_hbm,             # (T_local, E) bfloat16
    a2a_g_hbm,              # (E, bt, D) bfloat16 — HBM scratch for A2A gather
    # HBM output
    output_hbm,             # (T_local, D) bfloat16
    # SMEM (ALL 1-D — mandatory for E=256)
    t2e_routing_smem,       # (bt * K_padded,) int32
    d2e_count_smem,         # (EP * E_padded,) int32
    expert_offsets_smem,    # (2 * E_padded,) int32
    expert_starts_smem,     # (E_padded,) int32
    expert_sizes_smem,      # (E_padded,) int32
    a2a_s_sends_smem,       # (2,) int32
    # VMEM scratch
    a2a_s_x2_vmem,          # (2, bt*EP, D) — double-buffered A2A scatter receive
    a2a_g_acc_vmem,         # (K, bt, D)    — gather accumulator
    b_gating_x2_vmem,       # (2, bt, E)    — double-buffered gating prefetch
    b_output_x2_vmem,       # (2, bt, D)    — double-buffered output
    b_w1_x2_vmem,           # (2, D, F_shard) — double-buffered w1
    b_w2_x2_vmem,           # (2, F_shard, D) — double-buffered w2
    b_acc_vmem,             # (2, bt*EP, F_shard) float32 — GEMM accumulator
    local_sems,             # DMA(2, 5)
    send_sems,              # DMA(2,)
    recv_sems,              # DMA(2,)
    a2a_gather_sem,         # DMA
    *,
    top_k: int,
    ep_axis_name: str,
    bt: int,
    bf: int,
):
    bt_id      = pl.program_id(0)
    my_id      = lax.axis_index(ep_axis_name)
    num_devices = lax.axis_size(ep_axis_name)
    right_id   = (my_id + 1) % num_devices

    def get_mesh_device_id(ep_rank):
        fsdp_rank = lax.axis_index("fsdp")
        return (jnp.int32(0), ep_rank, fsdp_rank, tp_rank_scalar[0])

    def sync_barrier():
        barrier_sem = pltpu.get_barrier_semaphore()
        def signal_one(i, _):
            pltpu.semaphore_signal(
                barrier_sem, device_id=get_mesh_device_id(i),
                device_id_type=pltpu.DeviceIdType.MESH)
            return _
        lax.fori_loop(0, num_devices, signal_one, None)
        pltpu.semaphore_wait(barrier_sem, num_devices)

    # ----- Phase 0: top-k routing -----
    # Load gating[bt_id * bt : (bt_id+1) * bt] from gating_hbm (via b_gating_x2_vmem)
    # compute t2e_routing, expert_sizes, expert_starts
    # store into SMEM

    # ----- Phase 1: metadata ring-reduce -----
    # ring over EP to get global expert sizes and starts

    # ----- Phase 2: A2A scatter -----
    # for each expert, send tokens to owner device via async_remote_copy

    # ----- Phase 3: GEMM -----
    # wait for incoming tokens, run w1/activation/w2 for each local expert

    # ----- Phase 4: A2A gather -----
    # send computed results back to token-owner devices

    # ----- Phase 5: accumulate output -----
    # apply routing weights and sum into output_hbm
```

---

## Part 7: Wrapper (pallas_call setup)

```python
def fused_ep_moe_fwd(
    tokens,      # (T_local, D) inside shard_map — already EP+FSDP sharded
    w1,          # (E_local, 2, D, F_shard)
    w2,          # (E_local, F_shard, D)
    gating,      # (T_local, E)
    top_k: int,
    ep_axis_name: str,
    tp_rank: int = 0,       # static: 0 for TP=1
    collective_id: int = 0,
):
    T_local, D = tokens.shape
    E_local, _, _, F_shard = w1.shape
    E = gating.shape[1]
    EP = lax.axis_size(ep_axis_name)

    bt  = 8       # token tile size (tune based on VMEM budget)
    bf  = 128     # F tile size
    num_bt = T_local // bt

    K_padded = ((top_k + 127) // 128) * 128
    E_padded = ((E + 127) // 128) * 128

    # tp_rank_arr: (num_bt,) — same value broadcast to all grid steps
    tp_rank_arr = jnp.full((num_bt,), tp_rank, dtype=jnp.int32)

    return pl.pallas_call(
        functools.partial(
            _moe_kernel,
            top_k=top_k,
            ep_axis_name=ep_axis_name,
            bt=bt,
            bf=bf,
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            grid=(num_bt,),
            in_specs=[
                pl.BlockSpec((bt, D), lambda i: (i * bt, 0)),    # tokens
                pl.BlockSpec((None, None, D, F_shard), lambda i: (0, 0, 0, 0)),  # w1
                pl.BlockSpec((None, F_shard, D), lambda i: (0, 0, 0)),           # w2
                pl.BlockSpec((bt, None), lambda i: (i * bt, 0)),  # gating
                pl.BlockSpec((None, bt, D), lambda i: (0, 0, 0)), # a2a_g_hbm scratch
            ],
            out_specs=pl.BlockSpec((None, D), lambda i: (0, 0)),  # full output
            scratch_shapes=[
                # SMEM — all 1-D
                pltpu.SMEM((bt * K_padded,), jnp.int32),       # t2e_routing
                pltpu.SMEM((EP * E_padded,), jnp.int32),       # d2e_count
                pltpu.SMEM((2 * E_padded,), jnp.int32),        # expert_offsets
                pltpu.SMEM((E_padded,), jnp.int32),            # expert_starts
                pltpu.SMEM((E_padded,), jnp.int32),            # expert_sizes
                pltpu.SMEM((2,), jnp.int32),                   # a2a_s_sends
                # VMEM — no trailing size-1 dims
                pltpu.VMEM((2, bt * EP, D), jnp.bfloat16),    # a2a_s_x2
                pltpu.VMEM((top_k, bt, D), jnp.bfloat16),     # a2a_g_acc
                pltpu.VMEM((2, bt, E_padded), jnp.bfloat16),  # b_gating_x2
                pltpu.VMEM((2, bt, D), jnp.bfloat16),         # b_output_x2
                pltpu.VMEM((2, D, F_shard), jnp.bfloat16),    # b_w1_x2
                pltpu.VMEM((2, F_shard, D), jnp.bfloat16),    # b_w2_x2
                pltpu.VMEM((2, bt * EP, F_shard), jnp.float32), # b_acc (f32!)
                # Semaphores
                pltpu.SemaphoreType.DMA,   # local_sems (allocate as needed)
                pltpu.SemaphoreType.DMA,   # send_sems
                pltpu.SemaphoreType.DMA,   # recv_sems
                pltpu.SemaphoreType.DMA,   # a2a_gather_sem
            ],
        ),
        out_shape=jax.ShapeDtypeStruct(tokens.shape, tokens.dtype),
        compiler_params=pltpu.CompilerParams(collective_id=collective_id),
    )(tp_rank_arr, tokens, w1, w2, gating,
      jnp.zeros((E, bt, D), jnp.bfloat16))  # a2a_g_hbm scratch
```

---

## Part 8: Calling from shard_map

```python
import functools
from jax.experimental.shard_map import shard_map
from jax.sharding import PartitionSpec as P

def moe_forward(tokens, w0, w1, wo, gating, mesh, K):
    @functools.partial(
        shard_map, mesh=mesh,
        in_specs=(
            P("fsdp", None),        # tokens: (T_fsdp, D) per device
            P("ep", None, "fsdp"),  # w0 gate: (E_local, D, F_shard)
            P("ep", None, "fsdp"),  # w1 up:   (E_local, D, F_shard)
            P("ep", "fsdp", None),  # wo down: (E_local, F_shard, D)
            P("fsdp", None),        # gating:  (T_fsdp, E)
        ),
        out_specs=P("fsdp", None),
        check_rep=False,
    )
    def _fn(tok, w0_, w1_, wo_, gating_):
        w1_stk = jnp.stack([w0_, w1_], axis=1)  # (E_local, 2, D, F_shard)
        out = fused_ep_moe_fwd(
            tok, w1_stk, wo_, gating_,
            top_k=K,
            ep_axis_name="ep",
            tp_rank=0,              # TP=1 → always 0
            collective_id=0,
        )
        return jax.lax.psum(out, "fsdp")  # reduce FSDP partial sums

    return _fn(tokens, w0, w1, wo, gating)
```

---

## Part 9: Mosaic Constraints — The Complete List

These are non-negotiable. Each one has caused a crash in this kernel's history.

### 1. All SMEM arrays must be 1-D

```python
# WRONG — fails SC bitpacking at E=256
pltpu.SMEM((EP, E), jnp.int32)

# CORRECT — flatten and use arithmetic indexing
pltpu.SMEM((EP * E,), jnp.int32)
flat_idx = ep_rank * E + expert_id
```

### 2. No trailing size-1 VMEM dimensions

```python
# WRONG — size-1 dim maps to sublanes; downstream reshape crashes
pltpu.VMEM((2, N, 1, F), jnp.float32)

# CORRECT
pltpu.VMEM((2, N, F), jnp.float32)
```

### 3. No bool arrays in the kernel body

```python
# WRONG — produces i1 dtype; reshape or where will crash
mask = (row_iota == i)   # bool

# CORRECT — use int32 arithmetic
row_eq = 1 - jnp.minimum(jnp.abs(row_iota - i), 1)  # int32, 1 where equal
```

### 4. No `[:, None]` or any reshape to a trailing-1 dimension

```python
# WRONG — vector<N> → vector<Nx1> shape cast rejected by Mosaic
weights[:, None] * matrix    # expands (N,) to (N,1)

# CORRECT — build the 2-D weight matrix row-by-row with a fori_loop
```

### 5. No `.at[i, :].set(v)` on JAX arrays inside the kernel

```python
# WRONG — lowers to scatter_p, unsupported on TC path
result = result.at[i, :].set(new_row)

# CORRECT — use jnp.where with a row-iota mask (or operate on a VMEM Ref directly)
row_eq = 1 - jnp.minimum(jnp.abs(row_iota - i), 1)
result = result * (1 - row_eq) + row_eq * new_row
```

Note: `.at[].set()` on a **VMEM Ref** (not a JAX array) IS fine:
```python
vmem_ref.at[pl.ds(offset, size)].set(value)   # OK — in-place VMEM write
```

### 6. No `@jax.jit` on any function called inside `shard_map`

Remove all `@jax.jit` decorators from helper functions. Pallas compiles inline
inside `shard_map`. Adding `@jax.jit` causes 10×+ compilation slowdown.

### 7. `lax.axis_index` inside Pallas

- `lax.axis_index("ep")` inside the kernel body: **works** — Mosaic propagates
  the EP axis context correctly because the kernel runs inside `shard_map(ep)`.
- `lax.axis_index("tp")` inside the kernel body: **does not work** — Mosaic
  does not propagate axes that come from a dummy `P("tp")` shard_map input.
  Pass tp_rank via scalar_prefetch instead (or use TP=1 and hardcode 0).

### 8. SC BF16 row gather limit: ≤ 65,536 source rows

SparseCore's BF16 indexed gather uses 16-bit row indices. If `T_fsdp > 65,536`,
chunk the computation dynamically:

```python
SC_MAX_ROWS = 65536
n_chunks = 1
while T_fsdp // n_chunks > SC_MAX_ROWS:
    n_chunks *= 2
```

At T_fsdp=65,536 you are exactly at the limit. GBS=512 pushes over it.

### 9. Zero-initialize all VMEM accumulators explicitly

```python
# At the start of each tile iteration:
b_acc_vmem[buf_id, ...] = jnp.zeros_like(b_acc_vmem[buf_id])
```

XLA reuses freed HBM. Stale values may be NaN. Do not rely on `0.0 * stale`
to produce 0 — `0.0 * NaN = NaN`.

### 10. No leading-1 reshape either

```python
# WRONG — (E,) → (1, E) also fails
expert_hit = jnp.expand_dims(expert_hit, 0)   # shape (1, E)
result = result * expert_hit                    # broadcast fails
```

### 11. lax.fori_loop with unroll=False for variable-trip-count loops

```python
# Explicitly disable unrolling for loops with large/dynamic trip counts
lax.fori_loop(0, num_devices - 1, ring_step, init, unroll=False)
```

---

## Part 10: Test Ladder

Never skip levels. Each level tests a different code path.

### Level 1: AOT compile check (no hardware, ~3 min)

```python
from jax.experimental import topologies
import numpy as np

topo = topologies.get_topology_desc("tpu7x:4x4x4", platform="tpu")
devs = np.array(topo.devices).reshape(1, 4, 4, 1)  # dp=1,ep=4,fsdp=4,tp=1
mesh = jax.sharding.Mesh(devs, ("dp", "ep", "fsdp", "tp"))

tokens_abs = jax.ShapeDtypeStruct((T // FSDP, D), jnp.bfloat16)
w1_abs     = jax.ShapeDtypeStruct((E // EP, 2, D, F // FSDP), jnp.bfloat16)
# ... etc

fn = shard_map(moe_forward_fn, mesh=mesh, in_specs=..., out_specs=..., check_rep=False)

with jax.default_device(topo.devices[0]):
    lowered = jax.jit(fn).lower(tokens_abs, w1_abs, ...)
    lowered.compile()   # Mosaic runs for real; raises on any shape error

print("AOT PASS")
```

This catches all Mosaic shape/relayout errors locally. Run this before every
cluster submission.

### Level 2: EP=1 execution (single chip, correctness)

```python
# Build tiny mesh: dp=1, ep=1, fsdp=1, tp=1 (4 devices for v7x)
# Run forward kernel
# Compare against NumPy/JAX reference implementation
assert max_diff < 1e-2, f"Forward incorrect: max_diff={max_diff}"
```

At EP=1 there is no ICI A2A. This tests: routing, GEMM, activation, accumulation.

### Level 3: EP=4 execution (ICI A2A path)

```python
# mesh: dp=1, ep=4, fsdp=1, tp=1
# Run forward kernel
# Compare against JAX EP=4 reference
```

This tests the ring-reduce, scatter, and gather over ICI. If EP=1 passes but
EP=4 fails, the bug is in `get_mesh_device_id` or the A2A DMA logic.

### Level 4: Full cluster (EP=32, FSDP=16)

Submit a k8s JobSet. Pass = finite, decreasing loss for 3 steps with no crashes
on any of the 64 pods.

---

## Part 11: Debugging Tips

### Print from inside the kernel

```python
# Use default (NOT ordered=True — that fails on multi-device)
jax.debug.print(
    "ep={ep} bt={bt} expert_sizes={sz}",
    ep=lax.axis_index("ep"),
    bt=bt_id,
    sz=expert_sizes_smem[:8],
)
```

On multi-device runs, filter logs: `grep "ep=0 " /tmp/run.log`.

### Verify metadata ring-reduce

After Phase 1, print `reduced_sizes` and verify:
```
sum(reduced_sizes) ≈ T_fsdp * K  (total token-expert assignments)
reduced_sizes[e] ≈ T_fsdp * K / E  (uniform distribution)
```

### Verify device IDs

Print `get_mesh_device_id(right_id)` for ep=0,1,...,EP-1 and check they form
a valid ring: device for ep=0 sends to ep=1, ep=1 to ep=2, ..., ep=EP-1 to ep=0.

### Verify scatter/gather sizes

`a2a_s_sends_smem[e_sem_id]` = total remote tokens sent. Should sum to
`T_fsdp * K * (EP-1) / EP` (all tokens except those routing to local experts).

### Common failure modes

| Symptom | Likely cause |
|---|---|
| BoundsCheck 21 (dma.hbm_to_vmem OOB) | `start_off` wrong → metadata ring-reduce bug or wrong `get_mesh_device_id` |
| NaN from step 1 | VMEM accumulator not zero-initialized, or routing weights are NaN |
| Hang (no output) | Missing `semaphore_wait` or mismatched send/recv semaphores |
| AOT PASS, EP=1 crashes | Shape mismatch between BlockSpec and actual tensor dims |
| EP=1 PASS, EP=4 crashes | Wrong device ID in `async_remote_copy` |
| Correct at FSDP=1, wrong at FSDP=16 | F_shard slicing bug (weights use full F when FSDP=1) |

---

## Part 12: Implementation Order

Start simple and add one new mechanism at a time:

```
Step 1: kernel skeleton that copies tokens_hbm → output_hbm unchanged
        → confirms pallas_call wiring, BlockSpec, grid

Step 2: add local GEMM (no A2A, all tokens local, EP=1 only)
        → confirms w1/activation/w2 compute is correct

Step 3: add get_top_k routing
        → confirms routing indices and weights

Step 4: add metadata ring-reduce (EP=4)
        → confirms ICI semaphores work, ring topology correct

Step 5: add A2A scatter + gather (EP=4)
        → confirms tokens arrive at right device and results return correctly

Step 6: add double-buffering
        → performance optimization, correctness should be unchanged

Step 7: full pipeline with overlap
        → tune bt, bf for VMEM budget
```

Each step should pass the AOT compile check + corresponding test level before
proceeding. A bug at step 4 is definitively a ring-reduce bug, not a GEMM bug.
