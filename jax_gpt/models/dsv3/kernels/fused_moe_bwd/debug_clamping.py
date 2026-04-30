"""Check if dynamic_slice clamping in _expert_mlp_ep_body_ep_sharded causes the mismatch.

With max_tpe=TK and E_local=2:
  expert_starts[1] ≈ n_e0 > 0
  dynamic_slice(sorted_tids, (n_e0,), (TK,)) → JAX clamps start to min(n_e0, TK-TK)=0!
  Expert 1 gets the SAME tokens as expert 0 (wrong).

Fix: pad sorted_tids to size TK+max_tpe so no clamping ever occurs.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "mini_dsv3"))

import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from jax import lax

D     = 32
D_moe = 8
E     = 4
K     = 2
T     = 16
SEED  = 0

def make_mesh(fsdp, ep):
    devs = jax.devices()
    arr  = np.array(devs[:fsdp*ep]).reshape(fsdp, ep)
    return Mesh(arr, ("fsdp", "ep"))

def make_sharded(data, mesh, spec):
    sharding = NamedSharding(mesh, spec)
    idx_map  = sharding.addressable_devices_indices_map(data.shape)
    shards   = [jax.device_put(data[idx_map[d]], d) for d in idx_map]
    return jax.make_array_from_single_device_arrays(data.shape, sharding, shards)

def main():
    # Use single device (no shard_map) to test the forward body directly
    # Simulate what FSDP stripe 1, EP0 device sees

    # FSDP stripe 1 (fsdp=1), EP0 device (ep=0):
    # - local tokens = global [4,8) from ep=0, global [12,16) from ep=1
    # - After EP all_gather: local 0..3 = global 4..7, local 4..7 = global 12..15
    fsdp, ep = 2, 2
    mesh = make_mesh(fsdp, ep)
    act_spec = P(("ep","fsdp"), None)
    ep_axis  = "ep"

    T_per_dev = T // fsdp
    E_local   = E // ep
    avg_tpe   = T_per_dev * K // E_local
    max_tpe   = min(T_per_dev * K, max(4, 2 * avg_tpe))
    TK_local  = T_per_dev * K
    print(f"T_per_dev={T_per_dev}  E_local={E_local}  max_tpe={max_tpe}  TK_local={TK_local}")
    print(f"Clamping risk: max_tpe={max_tpe} == TK_local={TK_local} → YES!\n")

    # Manually compute what expert_starts are for FSDP stripe 1, EP0:
    key = jax.random.PRNGKey(SEED)
    rng = np.random.default_rng(SEED)
    fi_np = np.argsort(rng.random((T, E)), axis=-1)[:, :K].astype(np.int32)
    fw_np = np.ones((T, K), dtype=np.float32) / K

    # FSDP stripe 1 sees: global [4..8) from ep=0, global [12..16) from ep=1
    fi_local = np.concatenate([fi_np[4:8], fi_np[12:16]], axis=0)  # (8, K)
    print("FSDP stripe 1 fi_local (local indices 0..7 = global 4,5,6,7,12,13,14,15):")
    print(fi_local)

    # EP0's experts = {0,1}, offset=0
    expert_offset = 0
    E_local_val = 2
    fi_flat = fi_local.reshape(-1)  # TK=16
    tok_ids = np.repeat(np.arange(8), K)  # [0,0,1,1,...,7,7]

    is_local = (fi_flat >= expert_offset) & (fi_flat < expert_offset + E_local_val)
    local_exp = np.where(is_local, fi_flat - expert_offset, E_local_val)  # 0,1,or 2(sentinel)

    sort_order = np.argsort(local_exp, stable=True)
    sorted_exp = local_exp[sort_order]
    sorted_tok = tok_ids[sort_order]

    # searchsorted for forward
    from bisect import bisect_left, bisect_right
    expert_starts_fwd = [bisect_left(sorted_exp.tolist(), e) for e in range(E_local_val)]
    expert_ends_fwd   = [bisect_right(sorted_exp.tolist(), e) for e in range(E_local_val)]

    print(f"\nexpert_starts (forward searchsorted): {expert_starts_fwd}")
    print(f"expert_ends (forward searchsorted):   {expert_ends_fwd}")
    print(f"Expert 0: {expert_ends_fwd[0]-expert_starts_fwd[0]} tokens, "
          f"Expert 1: {expert_ends_fwd[1]-expert_starts_fwd[1]} tokens")

    print(f"\nsorted_tok (first 10): {sorted_tok[:10].tolist()}")

    print(f"\nForward dynamic_slice behavior for expert 1:")
    s1 = expert_starts_fwd[1]
    clamped_s1 = max(0, min(s1, len(sorted_tok) - max_tpe))
    print(f"  expert_starts[1] = {s1}, max_tpe = {max_tpe}, TK = {len(sorted_tok)}")
    print(f"  JAX clamps to: min({s1}, {len(sorted_tok)}-{max_tpe}) = min({s1}, {len(sorted_tok)-max_tpe}) = {clamped_s1}")
    if clamped_s1 != s1:
        print(f"  CLAMPING OCCURS! Getting tokens from position {clamped_s1} instead of {s1}")
        print(f"  Forward expert 1 sees tokens: {sorted_tok[clamped_s1:clamped_s1+max_tpe][:expert_ends_fwd[1]-expert_starts_fwd[1]].tolist()}")
        print(f"  Correct expert 1 tokens:      {sorted_tok[s1:s1+expert_ends_fwd[1]-expert_starts_fwd[1]].tolist()}")
    else:
        print(f"  No clamping.")

    # Fix: pad sorted_tok to size TK+max_tpe
    padded = np.concatenate([sorted_tok, np.zeros(max_tpe, dtype=np.int32)])
    clamped_s1_padded = max(0, min(s1, len(padded) - max_tpe))
    print(f"\nWith padding (size {len(padded)}): JAX clamps to min({s1}, {len(padded)-max_tpe}) = {clamped_s1_padded}")
    print(f"No clamping with padding! ✓")

    # Verify: fixing the forward by padding → does it fix the mismatch?
    print("\n---- Testing with fixed forward (padded sorted_tids) ----")
    # We'll apply the patch to model.py and rerun

if __name__ == "__main__":
    main()
