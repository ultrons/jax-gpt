"""Mini reproduction of v301's chunked MoE body for fast XLA-pipelining iteration.

Mirrors mini_dsv3/model.py::_expert_mlp_gmm_ag_body but at toy scale on 4 v4 cores.
Goal: confirm that the 2-chunk Python-unrolled loop produces an HLO where chunk 0's
psum-scatter overlaps chunk 1's ragged-dot. We inspect the optimized HLO directly.

Run with:
  source ~/xdb/.xprof/bin/activate
  XLA_FLAGS="--xla_dump_to=/tmp/mini_chunk_hlo --xla_dump_hlo_pass_re=.*" \\
      python mini_chunked_body.py
Then:
  ls /tmp/mini_chunk_hlo/*after_optimizations*.txt
"""

import os
import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, PartitionSpec as P
import numpy as np

print("devices:", jax.devices())
assert len(jax.devices()) == 4, f"expected 4 devices, got {len(jax.devices())}"

# Mesh: dp=1, fsdp=2, ep=2, tp=1 — matches v301's axis names.
devs = np.array(jax.devices()).reshape(1, 2, 2, 1)
mesh = Mesh(devs, ("dp", "fsdp", "ep", "tp"))
print("mesh:", mesh)

# Toy dims
T_local = 512        # per-EP-shard token count after sort
D = 512              # hidden
F_moe = 256          # MoE feed-forward
E = 8                # experts global
EP = 2
FSDP = 2
E_local = E // EP    # = 4
F_shard = F_moe // FSDP  # = 128
K = 4                # top-k


def _body(x_local, wi_0, wi_1, wo, indices_local, weights_local, n_chunks=2):
    """Mirror v301: AG once, then per-chunk sort/gather/ragged_dot/scatter/psum_scatter."""

    # AG weights along F (FSDP) — full F_moe
    wi_0_f = lax.all_gather(wi_0, "fsdp", axis=1, tiled=True)  # (E_local, F_moe, D)
    wi_1_f = lax.all_gather(wi_1, "fsdp", axis=1, tiled=True)
    wo_f   = lax.all_gather(wo,   "fsdp", axis=2, tiled=True)  # (E_local, D, F_moe)? — adjust below

    # AG tokens / indices / weights along EP
    all_x       = lax.all_gather(x_local,       "ep", axis=0, tiled=True)  # (T_all, D)
    all_indices = lax.all_gather(indices_local, "ep", axis=0, tiled=True)  # (T_all, K)
    all_weights = lax.all_gather(weights_local, "ep", axis=0, tiled=True)

    T_all = all_x.shape[0]
    chunk_size = T_all // n_chunks

    # Local expert window
    my_ep = lax.axis_index("ep")
    local_start = my_ep * E_local

    def _process_chunk(c):
        with jax.named_scope(f"chunk{c}"):
            cs = c * chunk_size
            chunk_x       = lax.dynamic_slice(all_x,       [cs, 0], [chunk_size, D])
            chunk_indices = lax.dynamic_slice(all_indices, [cs, 0], [chunk_size, K])
            chunk_weights = lax.dynamic_slice(all_weights, [cs, 0], [chunk_size, K])

            # Flatten (chunk, K) -> chunk*K
            flat_idx = chunk_indices.reshape(-1)        # (chunk*K,)
            flat_w   = chunk_weights.reshape(-1)
            # Tile x along K
            flat_x = jnp.repeat(chunk_x, K, axis=0)     # (chunk*K, D)

            # Mask non-local experts
            local_mask = (flat_idx >= local_start) & (flat_idx < local_start + E_local)
            local_e    = jnp.where(local_mask, flat_idx - local_start, 0)

            # Sort by expert (toy: just use argsort)
            order = jnp.argsort(local_e)
            sorted_x  = flat_x[order]
            sorted_e  = local_e[order]
            sorted_w  = flat_w[order]
            sorted_m  = local_mask[order].astype(jnp.float32)

            # Group sizes (one-hot sum)
            one_hot = jax.nn.one_hot(sorted_e, E_local, dtype=jnp.int32)
            group_sizes = (one_hot * sorted_m.astype(jnp.int32)[:, None]).sum(axis=0)

            # ragged_dot(x @ wi_0): (T,D) x (E_local,F,D) -> (T, F)
            h0 = jax.lax.ragged_dot(sorted_x, wi_0_f.transpose(0, 2, 1), group_sizes)  # (T, F_moe)
            h1 = jax.lax.ragged_dot(sorted_x, wi_1_f.transpose(0, 2, 1), group_sizes)
            h  = jax.nn.silu(h0) * h1
            out = jax.lax.ragged_dot(h, wo_f.transpose(0, 2, 1), group_sizes)  # (T, D)

            out = out * sorted_w[:, None] * sorted_m[:, None]

            # Unsort
            inv_order = jnp.argsort(order)
            unsorted = out[inv_order]                           # (chunk*K, D)
            chunk_out = unsorted.reshape(chunk_size, K, D).sum(axis=1)  # (chunk, D)

            # psum_scatter back to per-EP-shard
            return lax.psum_scatter(chunk_out, "ep",
                                    scatter_dimension=0, tiled=True)

    chunks = [_process_chunk(c) for c in range(n_chunks)]
    return jnp.concatenate(chunks, axis=0)


# wo shape: we want (E_local, D, F_moe) so the AG axis=2 is F. Then transpose to (E,F,D) for ragged_dot weight orient.
# Adjust: ragged_dot expects weights (E, in_dim, out_dim). h0 = sorted_x (T,D) @ (E, D, F).
# So wi_0 should have shape (E_local, D, F_moe). After AG along F (axis=2)? No — we shard wi_0 along F.
# Let me rewrite the shapes correctly.

def _body_v2(x_local, wi_0, wi_1, wo, indices_local, weights_local, n_chunks=2,
             use_barriers=True):
    """v302 pattern: per-chunk AG (slice LOCAL first, then AG) + optimization_barrier
    between chunks. use_barriers=False reproduces v301 layout for A/B comparison."""
    # Weight AGs hoisted (FSDP axis, shared across chunks — no benefit to chunking).
    wi_0_f = lax.all_gather(wi_0, "fsdp", axis=2, tiled=True)
    wi_1_f = lax.all_gather(wi_1, "fsdp", axis=2, tiled=True)
    wo_f   = lax.all_gather(wo,   "fsdp", axis=1, tiled=True)

    T_local = x_local.shape[0]
    sz_local = T_local // n_chunks
    chunk_size = sz_local * EP   # tokens per chunk after AG

    my_ep = lax.axis_index("ep")
    local_start = my_ep * E_local

    # Pre-slice LOCAL inputs (before AG).
    local_inputs = []
    for c in range(n_chunks):
        cs_local = c * sz_local
        local_inputs.append((
            lax.dynamic_slice(x_local,       [cs_local, 0], [sz_local, D]),
            lax.dynamic_slice(indices_local, [cs_local, 0], [sz_local, K]),
            lax.dynamic_slice(weights_local, [cs_local, 0], [sz_local, K]),
        ))

    def _process_chunk(c, inp):
        chunk_x_local, chunk_indices_local, chunk_weights_local = inp
        with jax.named_scope(f"chunk{c}"):
            if use_barriers:
                chunk_x_local, chunk_indices_local, chunk_weights_local = (
                    lax.optimization_barrier(
                        (chunk_x_local, chunk_indices_local, chunk_weights_local)))
            with jax.named_scope("ep_token_gather"):
                chunk_x       = lax.all_gather(chunk_x_local,       "ep", axis=0, tiled=True)
                chunk_indices = lax.all_gather(chunk_indices_local, "ep", axis=0, tiled=True)
                chunk_weights = lax.all_gather(chunk_weights_local, "ep", axis=0, tiled=True)
            if use_barriers:
                chunk_x, chunk_indices, chunk_weights = lax.optimization_barrier(
                    (chunk_x, chunk_indices, chunk_weights))

            flat_idx = chunk_indices.reshape(-1)
            flat_w   = chunk_weights.reshape(-1)
            flat_x   = jnp.repeat(chunk_x, K, axis=0)

            local_mask = (flat_idx >= local_start) & (flat_idx < local_start + E_local)
            local_e    = jnp.where(local_mask, flat_idx - local_start, 0)

            order = jnp.argsort(local_e)
            sorted_x = flat_x[order]
            sorted_e = local_e[order]
            sorted_w = flat_w[order]
            sorted_m = local_mask[order].astype(jnp.float32)

            one_hot = jax.nn.one_hot(sorted_e, E_local, dtype=jnp.int32)
            group_sizes = (one_hot * sorted_m.astype(jnp.int32)[:, None]).sum(axis=0)

            h0 = jax.lax.ragged_dot(sorted_x, wi_0_f, group_sizes)
            h1 = jax.lax.ragged_dot(sorted_x, wi_1_f, group_sizes)
            h  = jax.nn.silu(h0) * h1
            out = jax.lax.ragged_dot(h, wo_f, group_sizes)
            out = out * sorted_w[:, None] * sorted_m[:, None]

            if use_barriers:
                out = lax.optimization_barrier(out)

            inv_order = jnp.argsort(order)
            unsorted = out[inv_order]
            chunk_out = unsorted.reshape(chunk_size, K, D).sum(axis=1)
            result_c = lax.psum_scatter(chunk_out, "ep",
                                        scatter_dimension=0, tiled=True)
            if use_barriers:
                result_c = lax.optimization_barrier(result_c)
        return result_c

    # Sibling chunks (no cross-chunk barrier). Intra-chunk barriers keep AG
    # and scatter as atomic blocks so XLA can schedule chunks on different engines.
    chunks = [_process_chunk(c, local_inputs[c]) for c in range(n_chunks)]
    return jnp.concatenate(chunks, axis=0)


def make_inputs(seed=0):
    rng = np.random.default_rng(seed)
    # Per-shard sizes
    x        = rng.standard_normal((T_local * EP, D)).astype(np.float32)  # global tokens
    wi_0     = rng.standard_normal((E, D, F_moe)).astype(np.float32) * 0.02
    wi_1     = rng.standard_normal((E, D, F_moe)).astype(np.float32) * 0.02
    wo       = rng.standard_normal((E, F_moe, D)).astype(np.float32) * 0.02
    indices  = rng.integers(0, E, size=(T_local * EP, K), dtype=np.int32)
    weights  = rng.standard_normal((T_local * EP, K)).astype(np.float32)
    return x, wi_0, wi_1, wo, indices, weights


def run(n_chunks, profile_dir=None, use_barriers=True):
    x, wi_0, wi_1, wo, indices, weights = make_inputs()

    # Place inputs with the expected shardings.
    x_s        = jax.device_put(x,       jax.sharding.NamedSharding(mesh, P("ep", None)))
    indices_s  = jax.device_put(indices, jax.sharding.NamedSharding(mesh, P("ep", None)))
    weights_s  = jax.device_put(weights, jax.sharding.NamedSharding(mesh, P("ep", None)))
    wi_0_s     = jax.device_put(wi_0,    jax.sharding.NamedSharding(mesh, P("ep", None, "fsdp")))
    wi_1_s     = jax.device_put(wi_1,    jax.sharding.NamedSharding(mesh, P("ep", None, "fsdp")))
    wo_s       = jax.device_put(wo,      jax.sharding.NamedSharding(mesh, P("ep", "fsdp", None)))

    fn = shard_map(
        lambda x, w0, w1, wo, idx, w: _body_v2(x, w0, w1, wo, idx, w,
                                                n_chunks=n_chunks,
                                                use_barriers=use_barriers),
        mesh=mesh,
        in_specs=(P("ep", None), P("ep", None, "fsdp"), P("ep", None, "fsdp"),
                  P("ep", "fsdp", None), P("ep", None), P("ep", None)),
        out_specs=P("ep", None),
        check_rep=False,
    )
    jit = jax.jit(fn)

    # Warmup
    out = jit(x_s, wi_0_s, wi_1_s, wo_s, indices_s, weights_s)
    out.block_until_ready()
    print(f"n_chunks={n_chunks}: out shape={out.shape}, sum={float(jnp.abs(out).sum()):.4f}")

    # Timed runs
    import time
    N_REPEAT = 50
    t0 = time.perf_counter()
    for _ in range(N_REPEAT):
        out = jit(x_s, wi_0_s, wi_1_s, wo_s, indices_s, weights_s)
    out.block_until_ready()
    dt = (time.perf_counter() - t0) / N_REPEAT * 1e6
    print(f"n_chunks={n_chunks}: avg per call = {dt:.1f} us  (over {N_REPEAT} reps)")

    # Profile
    if profile_dir:
        sub = os.path.join(profile_dir, f"chunks{n_chunks}")
        os.makedirs(sub, exist_ok=True)
        jax.profiler.start_trace(sub)
        for _ in range(5):
            out = jit(x_s, wi_0_s, wi_1_s, wo_s, indices_s, weights_s)
        out.block_until_ready()
        jax.profiler.stop_trace()
        print(f"  profile -> {sub}")

    return jit, (x_s, wi_0_s, wi_1_s, wo_s, indices_s, weights_s)


if __name__ == "__main__":
    profile_dir = os.environ.get("PROFILE_DIR")  # e.g. /tmp/mini_chunk_prof
    print("=== n_chunks=1 (baseline) ===")
    run(1, profile_dir=(profile_dir + "/n1") if profile_dir else None,
        use_barriers=False)
    print("=== n_chunks=2 NO barriers (v301-style) ===")
    run(2, profile_dir=(profile_dir + "/n2_nobar") if profile_dir else None,
        use_barriers=False)
    print("=== n_chunks=2 WITH barriers (v302-style) ===")
    run(2, profile_dir=(profile_dir + "/n2_bar") if profile_dir else None,
        use_barriers=True)

    # Inspect HLO if dump dir set.
    dump = os.environ.get("XLA_FLAGS", "")
    if "xla_dump_to" in dump:
        path = dump.split("xla_dump_to=")[1].split()[0].rstrip(",")
        print(f"\nHLO dumped to: {path}")
        print("Look for *.after_optimizations.txt and grep for psum-scatter / ragged-dot interleave.")
