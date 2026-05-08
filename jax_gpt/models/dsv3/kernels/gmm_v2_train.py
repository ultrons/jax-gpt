"""custom_vjp wrapper around gmm_v2 for training.

Stage A.2: forward = gmm_v2 (Pallas, fast); backward = JAX upstream gmm + tgmm
(both Pallas kernels; tgmm is the per-group outer product needed for d_rhs).

For plain (no fuse_act):
    fwd: out = gmm_v2(lhs, rhs, gs)
    bwd: d_lhs = gmm(d_out, rhs_T, gs)
         d_rhs = tgmm(lhs.T, d_out, gs)

For fused-silu (gate+up+silu+multiply):
    fwd: out = silu(g) * u, where g = lhs@wi_0, u = lhs@wi_1
         (computed by gmm_v2 with fuse_act='silu')
    bwd: residuals = (lhs, wi_0, wi_1, g, u)  ← MUST save g and u to avoid recompute
         d_g = d_out * u * silu_prime(g)
         d_u = d_out * silu(g)
         d_lhs = gmm(d_g, wi_0_T, gs) + gmm(d_u, wi_1_T, gs)
         d_wi_0 = tgmm(lhs.T, d_g, gs)
         d_wi_1 = tgmm(lhs.T, d_u, gs)
"""
from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
from jax.experimental.pallas.ops.tpu.megablox import gmm as megablox_gmm
from jax.experimental.pallas.ops.tpu.megablox.gmm import tgmm as megablox_tgmm

from .gmm_v2 import gmm_v2


def _gmm_tiles(M: int, K: int, N: int) -> tuple[int, int, int]:
    """Tile sizes for megablox.gmm. Defaults (128,128,128) → ~50× too many tile
    iterations for our DSv3 shapes. Use tokamax-tuned values where shape matches;
    otherwise reasonable bf16 defaults.
    """
    # Tokamax-tuned for tpu7x bf16 (from tokamax/data/autotuning/tpu7x/...).
    # Our chunked shapes after gmm_v2 fwd:
    #   gate/up bwd of d_lhs: gmm(d_g[M,N=2048], wi_0_T[E,N=2048,K=7168]) → [M, K=7168]
    #   wo bwd of d_lhs:      gmm(d_out[M,N=7168], wo_T[E,N=7168,K=2048]) → [M, K=2048]
    if (M, K, N) == (131072, 2048, 7168):  # gate/up d_lhs
        return (512, 2048, 1792)  # tokamax: 8.71 ms
    if (M, K, N) == (131072, 7168, 2048):  # wo d_lhs / fwd
        return (128, 7168, 1024)  # tokamax adapted: 7.63 ms
    if (M, K, N) == (131072, 7168, 1024):  # tokamax exact
        return (128, 7168, 1024)  # 3.84 ms
    if (M, K, N) == (131072, 1024, 7168):  # tokamax exact
        return (512, 1024, 1024)  # 6.40 ms
    # Generic bf16 fallback: bigger than (128,128,128) default.
    return (128, min(K, 2048), min(N, 1024))


def _tgmm_tiles(K: int, M: int, N: int) -> tuple[int, int, int]:
    """Tile sizes for megablox.tgmm. Same default issue.
    For tgmm: lhs[K, M], rhs[M, N]; M is the contraction axis.
    Big tile_m → fewer iterations.

    Heuristic-tuned for v7x BF16 production shapes (autoperf iter-5,
    sized at 3,026 ms/step in iter-2b xplane via xla_shell list_fusions
    on 6× tgmm.12-17 fusions). Production runs with
    `LIBTPU_INIT_ARGS=--xla_tpu_scoped_vmem_limit_kib=65536` (64 MB
    scoped VMEM, see manifests/jobset.yaml.j2:112), giving headroom
    over megablox.tgmm's default 32 MB cap.

    AOT-validated at 64 MB scoped VMEM (tpu7x:2x2x1):
      d_wi_0/d_wi_1 (gate/up bwd): tgmm(K=7168, M=131072, N=2048).
        Baseline (2048,1024,1024): 64×7×2 = 896 tile iters.
        New      (4096,1024,1024): 32×7×2 = 448 iters (2× reduction).
      d_wo (down bwd): tgmm(K=2048, M=131072, N=7168).
        Baseline (2048,1024,1024): 64×2×7 = 896 iters.
        New      (4096,1024,1024): 32×2×7 = 448 iters (2× reduction).
    Note: at the default 32 MB cap, these tiles would FAIL AOT — they
    REQUIRE the production --xla_tpu_scoped_vmem_limit_kib=65536 flag.
    """
    if (K, M, N) == (7168, 131072, 2048):  # gate/up d_wi (bwd)
        return (4096, 1024, 1024)
    if (K, M, N) == (2048, 131072, 7168):  # down d_wo (bwd)
        return (4096, 1024, 1024)
    return (min(M, 2048), min(K, 1024), min(N, 1024))


@functools.partial(jax.custom_vjp, nondiff_argnums=(3,))
def gmm_v2_train(lhs, rhs, group_sizes, vmem_limit_bytes: int = 0):
    """Plain ragged dot via gmm_v2 forward, gmm + tgmm backward.

    lhs:         [M, K]
    rhs:         [E, K, N]
    group_sizes: int32[E]
    """
    return _gmm_v2_train_fwd(lhs, rhs, group_sizes, vmem_limit_bytes)[0]


def _gmm_v2_train_fwd(lhs, rhs, group_sizes, vmem_limit_bytes):
    kw = {} if vmem_limit_bytes == 0 else {"vmem_limit_bytes": vmem_limit_bytes}
    out = gmm_v2(lhs, rhs, group_sizes, **kw)
    return out, (lhs, rhs, group_sizes)


def _gmm_v2_train_bwd(vmem_limit_bytes, res, d_out):
    del vmem_limit_bytes
    lhs, rhs, group_sizes = res
    M, K = lhs.shape
    _, _, N = rhs.shape  # rhs is [E, K, N]
    # d_lhs[m, k] = sum_n d_out[m, n] * rhs[g(m), k, n]
    # gmm(d_out[M,N], rhs.T[E,N,K]) → [M, K]
    rhs_T = rhs.transpose(0, 2, 1)
    d_lhs = megablox_gmm(d_out, rhs_T, group_sizes,
                         preferred_element_type=lhs.dtype,
                         tiling=_gmm_tiles(M, N, K))
    # d_rhs[g, k, n] = sum_{m in group g} lhs[m, k] * d_out[m, n]
    # tgmm(lhs.T[K,M], d_out[M,N]) → [E, K, N]
    d_rhs = megablox_tgmm(lhs.T, d_out, group_sizes,
                          preferred_element_type=rhs.dtype,
                          tiling=_tgmm_tiles(K, M, N))
    return d_lhs, d_rhs, jnp.zeros_like(group_sizes)


gmm_v2_train.defvjp(_gmm_v2_train_fwd, _gmm_v2_train_bwd)


@functools.partial(jax.custom_vjp, nondiff_argnums=(4,))
def gmm_v2_fused_silu_train(lhs, wi_0, wi_1, group_sizes, vmem_limit_bytes: int = 0):
    """Fused gate+up+silu via gmm_v2(fuse_act='silu').

    lhs:         [M, K]
    wi_0:        [E, K, N]            (gate weights)
    wi_1:        [E, K, N]            (up weights)
    group_sizes: int32[E]

    fwd: out = silu(lhs @ wi_0) * (lhs @ wi_1)
       implemented as gmm_v2(lhs, FusedWeightsRef(wi_0, wi_1), gs, fuse_act='silu')
       → on the wire: rhs = jnp.concatenate([wi_0, wi_1], axis=-1) of shape (E, K, 2N)
    bwd: chain rule via jax.vjp on the explicit silu(.) * (.) reference.
    """
    return _gmm_v2_fused_silu_fwd(lhs, wi_0, wi_1, group_sizes, vmem_limit_bytes)[0]


def _gmm_v2_fused_silu_fwd(lhs, wi_0, wi_1, group_sizes, vmem_limit_bytes):
    # Compute g and u explicitly (no recompute in bwd).
    # Earlier attempt recomputed g, u in bwd — produced NaN at cluster scale
    # (likely fused/unfused gmm_v2 numerics drift at M=131072). Saving them
    # adds ~1 GB / chunk to residuals (g, u each (M, N_per_rhs) bf16) but
    # guarantees bwd uses the EXACT g, u that fwd accumulated.
    g = gmm_v2(lhs, wi_0, group_sizes)
    u = gmm_v2(lhs, wi_1, group_sizes)
    out = jax.nn.silu(g) * u
    return out, (lhs, wi_0, wi_1, group_sizes, g, u)


def _gmm_v2_fused_silu_bwd(vmem_limit_bytes, res, d_out):
    del vmem_limit_bytes
    lhs, wi_0, wi_1, group_sizes, g, u = res

    # JAX-autograd silu*u (stable across edge cases) using saved g, u from fwd.
    def _act(g_, u_):
        return jax.nn.silu(g_) * u_

    _, act_vjp = jax.vjp(_act, g, u)
    d_g, d_u = act_vjp(d_out)

    # Matmul backward via gmm/tgmm with tuned tiles.
    M, K = lhs.shape
    _, _, N = wi_0.shape
    wi_0_T = wi_0.transpose(0, 2, 1)
    wi_1_T = wi_1.transpose(0, 2, 1)
    # gmm(d_g[M,N], wi_0_T[E,N,K]) → [M, K]
    gmm_tile = _gmm_tiles(M, N, K)
    d_lhs = (megablox_gmm(d_g, wi_0_T, group_sizes,
                          preferred_element_type=lhs.dtype, tiling=gmm_tile) +
             megablox_gmm(d_u, wi_1_T, group_sizes,
                          preferred_element_type=lhs.dtype, tiling=gmm_tile))
    # tgmm(lhs.T[K,M], d_g[M,N]) → [E, K, N]
    tgmm_tile = _tgmm_tiles(K, M, N)
    d_wi_0 = megablox_tgmm(lhs.T, d_g, group_sizes,
                           preferred_element_type=wi_0.dtype, tiling=tgmm_tile)
    d_wi_1 = megablox_tgmm(lhs.T, d_u, group_sizes,
                           preferred_element_type=wi_1.dtype, tiling=tgmm_tile)

    return d_lhs, d_wi_0, d_wi_1, jnp.zeros_like(group_sizes)


gmm_v2_fused_silu_train.defvjp(_gmm_v2_fused_silu_fwd, _gmm_v2_fused_silu_bwd)
