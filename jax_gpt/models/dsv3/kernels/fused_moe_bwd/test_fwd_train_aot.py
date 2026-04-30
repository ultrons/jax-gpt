"""AOT compile check for fwd_kernel_train.py — no physical TPU required.

Uses jax.experimental.topologies to get virtual v7x devices and Mosaic-compile
the training forward kernel with the same shapes/sharding as the cluster run:
  EP=16, FSDP=16, TP=2 on 4x4x16 (512 devices = 256 chips)

v204: collective_id=1 (matching _moe_pallas_fwd_jax_bwd_fwd which uses id=1 so
that the VJP fwd kernel uses a different ICI channel from the primal eval kernel
which uses collective_id=0).  This prevents XLA async collective fusion from
colliding the two ring-reduce transfers on the same ICI hardware channel.

lax.axis_index("tp") at the JAX/shard_map level (outside Pallas) gives the
correct partition_id and is used for both instances.

This catches every Mosaic shape-cast / relayout / SC-gather error locally,
in ~2-3 min, before spending a slot on the cluster.

Usage:
    source ~/xdb/.xprof/bin/activate
    cd ~/ml-experiments/dsv3
    python fused_moe_bwd/test_fwd_train_aot.py
"""

import functools
import sys
import os

import numpy as np

# Resolve dsv3 root so imports work whether run from dsv3/ or elsewhere.
_dsv3_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _dsv3_root)

import jax
import jax.numpy as jnp
from jax.experimental import topologies
from jax.experimental.shard_map import shard_map
from jax.sharding import PartitionSpec as P

from .fwd_kernel_train import fused_ep_moe_fwd_train_v1

# ---------------------------------------------------------------------------
# Cluster config: 4x4x16, EP=16 FSDP=16 TP=2, GBS=256
# ---------------------------------------------------------------------------
DP    = 1
EP    = 16
FSDP  = 16
TP    = 2
GBS   = 256
SEQ   = 4096
D     = 7168
F     = 2048
E     = 256   # global experts
K     = 8     # top-k

TOTAL_DEVICES = DP * EP * FSDP * TP    # = 512 — matches 4x4x16

# ---------------------------------------------------------------------------
# Virtual topology — no physical TPU needed
# ---------------------------------------------------------------------------
print(f"[AOT] Loading tpu7x:4x4x16 virtual topology ({TOTAL_DEVICES} devices)...")
topo = topologies.get_topology_desc("tpu7x:4x4x16", platform="tpu")
assert len(topo.devices) == TOTAL_DEVICES, (
    f"Expected {TOTAL_DEVICES} devices, got {len(topo.devices)}")

# Mesh layout: (dp=1, ep=16, fsdp=16, tp=2)
devs = np.array(topo.devices).reshape(DP, EP, FSDP, TP)
mesh = jax.sharding.Mesh(devs, ("dp", "ep", "fsdp", "tp"))
print(f"[AOT] Mesh: {dict(mesh.shape)}")

# ---------------------------------------------------------------------------
# Global shapes (what shard_map receives — BEFORE sharding)
#
# Match model.py exactly:
#   act_spec  = P("fsdp", None)
#   w0/w1 (wi_0, wi_1): P("ep", None, "fsdp")  → (E_local, D, F_shard) inside
#   wo (down proj):     P("ep", "fsdp", None)   → (E_local, F_shard, D) inside
#
# Inside shard_map (what pallas_call receives):
#   tokens : (T_fsdp=65536, D=7168)
#   w0/w1  : (E_local=16, D=7168, F_shard=128)  — stacked → (16, 2, 7168, 128)
#   wo     : (E_local=16, F_shard=128, D=7168)
#   gating : (T_fsdp=65536, E=256)
# ---------------------------------------------------------------------------
T_total = GBS * SEQ              # 1048576 total tokens
T_fsdp  = T_total // FSDP        # 65536 per FSDP shard = T_local inside kernel

act_spec = P("fsdp", None)

# Global shapes (pre-sharding):
tokens_abs = jax.ShapeDtypeStruct((T_total, D), jnp.bfloat16)       # (1048576, 7168)
w0_abs     = jax.ShapeDtypeStruct((E, D, F), jnp.bfloat16)          # (256, 7168, 2048)
w1_abs     = jax.ShapeDtypeStruct((E, D, F), jnp.bfloat16)          # (256, 7168, 2048)
w2_abs     = jax.ShapeDtypeStruct((E, F, D), jnp.bfloat16)          # (256, 2048, 7168)
gating_abs = jax.ShapeDtypeStruct((T_total, E), jnp.bfloat16)       # (1048576, 256)

print(f"[AOT] T_total={T_total}, T_fsdp={T_fsdp}")
print(f"[AOT] Global: tokens={tokens_abs.shape}, w0={w0_abs.shape}, w2={w2_abs.shape}")
print(f"[AOT] Inside shard_map: tokens=(65536, 7168), w0=(16, 7168, 128), w2=(16, 128, 7168)")

# ---------------------------------------------------------------------------
# Kernel fn — mirrors model.py's shard_map body for fused_ep_moe_v4 backend
# ---------------------------------------------------------------------------
tp_iota = jnp.arange(TP, dtype=jnp.int32)


def _kernel_fn(tokens, w0, w1, w2, gating, _tp_dummy):
    # w0, w1: (E_local, D, F_shard); stack → (E_local, 2, D, F_shard)
    w1_stk = jnp.stack([w0, w1], axis=1)
    # lax.axis_index("tp") at the JAX/shard_map level lowers to XLA partition_id —
    # a hardware primitive that SPMD never folds to a constant.
    tp_rank = jax.lax.axis_index("tp").astype(jnp.int32)  # 0 or 1, per device
    tp_rank_arr_local = jnp.full((1,), tp_rank, dtype=jnp.int32)  # shape (1,)
    out = fused_ep_moe_fwd_train_v1(
        tokens, w1_stk, w2, gating, K,
        ep_axis_name="ep",
        act_fn="silu",
        scoring_fn="identity",
        renormalize_topk_logits=True,
        non_ep_axis_name="fsdp",
        non_ep_first=False,
        extra_device_id_prefix=(0,),
        extra_device_id_suffix=(),
        tp_axis_name=None,
        tp_rank_arr=tp_rank_arr_local,
        # v204: collective_id=1 matches _moe_pallas_fwd_jax_bwd_fwd (VJP fwd path).
        # The primal eval path uses collective_id=0. Different IDs → different ICI
        # channels → no async collision between the two HLO instances.
        collective_id=1,
    )
    # Each FSDP device contributes F_shard of the D-dim output; sum across fsdp.
    return jax.lax.psum(out, "fsdp")


sharded_fn = functools.partial(
    shard_map,
    f=_kernel_fn,
    mesh=mesh,
    in_specs=(
        act_spec,              # tokens: P("fsdp", None) → (T_fsdp, D) inside
        P("ep", None, "fsdp"),  # w0: (E//EP, D, F//FSDP) = (16, 7168, 128) inside
        P("ep", None, "fsdp"),  # w1: same
        P("ep", "fsdp", None),  # w2: (E//EP, F//FSDP, D) = (16, 128, 7168) inside
        act_spec,              # gating: (T_fsdp, E) inside
        P("tp"),               # tp_dummy: brings tp into shard_map scope
    ),
    out_specs=act_spec,
    check_rep=False,
)()

# ---------------------------------------------------------------------------
# AOT lower + compile (Mosaic runs for real on v7x backend)
# ---------------------------------------------------------------------------
print("[AOT] Lowering...")
with jax.default_device(topo.devices[0]):
    lowered = jax.jit(sharded_fn).lower(
        tokens_abs, w0_abs, w1_abs, w2_abs, gating_abs,
        jax.ShapeDtypeStruct((TP,), jnp.int32))

print("[AOT] Compiling (Mosaic)... this takes ~3-5 min")
try:
    compiled = lowered.compile()
    print("[AOT] PASS — forward kernel (v204: collective_id=1 for VJP fwd path, lax.axis_index tp_rank) compiled.")
    print(f"[AOT] Compiled object: {type(compiled)}")
    sys.exit(0)
except Exception as e:
    print(f"[AOT] FAIL — Mosaic compile error:\n{e}")
    sys.exit(1)
