---
slug: phase-e4-through-f1
intent: results
status: snapshot 2026-05-15
sources:
  - build/v_inside/moe_block_ep_megatron.py (E.4)
  - build/v_inside/moe_block_ep_megatron_vjp.py (E.5)
  - build/v_outside/expert_ffn_d_tiled.py (D.6)
  - build/v_inside/moe_block_ep_megatron_scatter.py (E.6 step 1)
  - build/v_outside/tests/bench_f1_variants.py (F.1)
local mesh: 4-core tpu7x:2x2x1
cluster mesh: 4x4x4 = 128 cores (rbq, capacity-contended at time of writing)
---

# Phase E.4 → F.1 — Megatron path, full D, perf table

## E.4 — Megatron column+row parallel (fwd)

Real W-side HBM win for v_inside. F-sharded W kept across the FFN;
`lax.psum` across fsdp reduces the row-parallel partials.

Local bit-exact PASS at EP=2/FSDP=1, EP=2/FSDP=2, EP=1/FSDP=4
(see `phase_d6_e3_cluster.md` §E.4 for the detailed write-up).

**Cluster gate PASSED** on rbq 4x4x4 (128 cores) — `e4-megatron-3`:

```
[fwd]            mesh=(1,4,32,1) T_global=4096  max_abs=1.56e-2 bad_rows=1/4096   PASS
[v_inside-fwd]   FSDP=32  max_abs=0  max_rel=0                                    PASS  (bit-exact)
[bwd-pallas]     d_W_gate / d_W_d bit-exact; d_x_in 3.9e-3 / d_W1 3.1e-2          PASS
[large-shape]    E=64 D=2048 F=128 K=4 EP=4                                       PASS
[d-push]         E=32 D=3840 F=128 K=4 EP=4                                       PASS
[megatron]       FSDP=32  T_global=1024  max_abs=1.95e-3  max_rel=8.0e-4          PASS  ← E.4 cluster gate
[run_g3_cluster] ALL PASS
```

The `[megatron]` test compares E.4 Megatron (F-sharded W + lax.psum
across fsdp) against v_outside (full-F W replicated on fsdp) at
mesh=(1,4,32,1). Numerics match within G3 tolerance — confirming the
Megatron pattern composes correctly across 32-way fsdp ICI.

Stall note: first cluster attempt (`e4-megatron-2`) was Kueue-admitted
in 40 min but stuck in `ImagePullBackOff` for 29 min — image tag
`cde-8ce07e7` from a different SHA was never built. After deleting,
rebuilding (`cde-614afc2`), and resubmitting, `e4-megatron-3`
admitted+ran in ~5 min and all gates passed.

## E.5 — Megatron bwd VJP

Wraps the E.4 forward with `custom_vjp`. Forward uses the Pallas
kernel; backward does `jax.vjp` on a JAX-only mirror of the same math
(no Pallas in the bwd path).

`lax.psum`'s adjoint is identity-broadcast: each fsdp peer receives
the same `d_out`, runs its own local vjp on the F-shard FFN, and
produces a peer-specific `(d_W1_local, d_W_d_local)` plus a partial
`d_tok_local`. shard_map's input-cotangent reconciliation psums
`d_tok_local` across fsdp (since `x_in` has spec `P("ep", None)` —
replicated on fsdp — its cotangent must also be replicated).

Local results (`test_g3_megatron_bwd.py`):

```
fsdp=1   d_x, d_W_gate, d_W1, d_W_d  ALL  max_abs=0   PASS
fsdp=2   d_x_in  7.8e-3                others  max_abs=0    PASS
fsdp=4   d_x_in  7.8e-3                others  max_abs=0    PASS
```

Small `d_x_in` delta at fsdp>1 is bf16 rounding through the
psum-broadcast bwd; well within G3 tolerance (5e-2). 

## D.6 — true D-tiling for full DSv3 D=7168

The standard kernel holds the full per-device `W1 (E_local, D, 2F)`
in VMEM as a BlockSpec window — that's 15.7 MB at E_local=8 D=3840
F=128 (the prior cluster ceiling) and ~29 MB at D=7168 (OOMs).

D.6 changes the grid from `(num_bt,)` to `(num_bt, E_local, num_d_out)`:
1. **E-tile**: each grid step holds ONE expert's W1 (block size
   `(1, D, 2F)` instead of `(E_local, D, 2F)`).
2. **Output D-tile**: out + out_acc are sized `(bt, D_tile)` instead
   of `(bt, D)`. Default D_tile = 1024.

Inner axis = `d_out` so W1[e] stays cached in VMEM across d steps. An
`act_scratch (bt, F)` f32 is computed once per (bt, e) pair at d=0
and re-used — avoids `num_d_out`× redundant up-matmul work. Output
uses RMW across the E_local axis: e=0 initializes from zero, e>0
accumulates.

Per-grid-step VMEM at D=7168, F=128, bt=128, D_tile=1024:

```
W1 block       (1, 7168, 256) bf16   ≈ 3.7 MB
W_d block      (1, 128, 1024) bf16   ≈ 0.25 MB
tok block      (128, 7168) bf16      ≈ 1.8 MB
out block      (128, 1024) f32       ≈ 0.5 MB  (×2 buf ≈ 1 MB)
act_scratch    (128, 128) f32        ≈ 64 KB
                                total ≈ 7-9 MB  (<16 MB)
```

`test_d6_d_tiled.py` results (local 4-core):

```
small (E=4 D=256 F=64)     bit-exact vs std kernel (max_abs=0)
mid   (E=2 D=1024 F=128)   bit-exact vs std kernel
d2048 (E=4 D=2048 F=128)   bit-exact vs std kernel
ceiling bisect:
  D=3840 PASS  ←  prior std-kernel ceiling
  D=4096 PASS
  D=4608 PASS
  D=5120 PASS
  D=5632 PASS
  D=6144 PASS
  D=6656 PASS
  D=7168 PASS  ←  FULL DSv3 production D
```

`expert_ffn_v_outside` auto-impl now switches to the D-tiled kernel
when W1 > 12 MB (giving headroom for tok/out/buffering). For DSv3
production (E_local=8 D=7168 F=128, W1=28 MB) it's selected
automatically.

## E.6 step 1 — Megatron with psum-scatter + all-gather

Replaces `lax.psum(out_partial, fsdp)` with the canonical
reduce-scatter + all-gather pair:

```
E.4:  out (M, D)        = lax.psum(out_partial, fsdp)
E.6:  out_d_local (M, D/fsdp) = lax.psum_scatter(out_partial, fsdp, dim=1)
      out (M, D)              = lax.all_gather(out_d_local, fsdp, dim=1)
```

Same comm volume; same downstream interface (full M, D on every
device). Validates the scatter API path so a future fused
streaming-psum-scatter kernel can drop in as a replacement for the
`psum_scatter` call without changing the wrapper structure.

`test_g3_megatron_scatter.py` results (local 4-core):

```
EP=2 FSDP=1   vs v_outside  max_abs=0    PASS
              vs E.4 psum   max_abs=0    PASS
EP=2 FSDP=2   vs v_outside  max_abs=0    PASS
              vs E.4 psum   max_abs=0    PASS
EP=1 FSDP=4   vs v_outside  max_abs=0    PASS
              vs E.4 psum   max_abs=0    PASS
```

**E.6 step 2 (deferred)**: fused down-matmul + streaming scatter for
real comm-compute overlap. The E.3 standalone reference validates the
streaming primitive in isolation; step 2 would integrate the
per-D-chunk DMA pattern INTO the expert FFN kernel so each chunk's
DMA fires while the next chunk's matmul runs.

## F.1 — variant perf table

Local 4-core (`bench_f1_variants.py`, E=8 D=512 F=128 K=2 T=128):

```
variant            fwd_ms        bwd_ms
v_outside          1.88 ± 0.04   1.98 ± 0.05
megatron_psum      1.71 ± 0.02   1.93 ± 0.04
megatron_scatter   1.80 ± 0.04   (fwd-only — no custom_vjp wrapper)
```

(At local 4-core the two variants use different meshes — v_outside
EP=4 FSDP=1 vs Megatron EP=1 FSDP=4 — so this is not strictly
apples-to-apples; it's a sanity check that the wrappers work.)

**Cluster** on rbq 4x4x4 (`f1-bench-3`, 128 cores, E=32 D=2048 F=128
K=4 EP=4 FSDP=32, T_global=4096):

```
variant            fwd_ms          bwd_ms
v_outside          2.45 ± 0.16     3.08 ± 0.20
v_inside_optB      2.47 ± 0.21     (fwd-only)
megatron_psum      4.96 ± 0.39    12.98 ± 0.60
megatron_scatter   5.25 ± 0.11     (fwd-only)
```

What the cluster numbers say:

1. **v_inside Option β ≈ v_outside** (2.47 vs 2.45 ms fwd, within
   noise). The auto-AG of F-sharded W at the shard_map boundary
   reconstructs full-F W per device — same per-device math, same
   per-device HBM footprint, so identical perf. As expected from the
   E.2 design: Option β trades the HBM-peak win for API compatibility.

2. **Megatron variants are ~2× slower fwd, ~4× slower bwd than
   v_outside.** This is the cost of Megatron's parallelism shape on
   this mesh:
   - v_outside `x_spec = P(("ep","fsdp"), None)` → x sharded on both
     axes → T_local = 4096/128 = 32 tokens per device.
   - Megatron `x_spec = P("ep", None)` → x replicated on fsdp →
     T_local = 4096/4 = 1024 tokens per device.
   So each fsdp peer in Megatron redundantly processes 32× more
   tokens than a v_outside device. The HBM win Megatron buys (W
   F-sharded → 32× smaller per-device weight footprint) costs that
   32× extra compute per device.
   In production this is balanced against DP (which v_outside also
   needs to scale beyond a single 4x4x4 slice) — Megatron with DP
   scales like Megatron + DP × FSDP, where the redundancy is
   amortized across data parallelism. The 2-4× per-device gap is
   the architectural trade, not a kernel bug.

3. **scatter ~6% slower than psum** (5.25 vs 4.96 ms fwd). E.6
   step 1 does `psum_scatter` + `all_gather`, which is two ops vs
   one for `lax.psum`. The overhead is the extra ICI scheduling, not
   comm volume (both have 2(N-1)/N volume). E.6 step 2 (fused
   streaming kernel) is where the comm-compute overlap would
   recover this gap and ideally beat plain `psum`.

4. **bwd 4× slower than fwd for Megatron**, but 1.26× for
   v_outside. Megatron's bwd runs the full JAX-only mirror through
   `jax.vjp` (E.5 design), so it includes the JAX-only FFN loop on
   1024 tokens per device. v_outside's bwd uses the D.1 Pallas-bwd
   kernel. A Pallas-Megatron-bwd kernel would bring the bwd back to
   ~fwd ratio.

## Status summary across this session

| Phase | Status | Where |
|---|---|---|
| E.4 Megatron fwd | bit-exact local + cluster PASS at FSDP=32 | `build/v_inside/moe_block_ep_megatron.py` |
| E.5 Megatron bwd VJP | bit-exact local at fsdp=1, ≤1e-2 at fsdp=2/4 | `build/v_inside/moe_block_ep_megatron_vjp.py` |
| D.6 full D=7168 | bit-exact local; ceiling reached | `build/v_outside/expert_ffn_d_tiled.py` |
| E.6 step 1 (scatter) | bit-exact local vs psum + v_outside | `build/v_inside/moe_block_ep_megatron_scatter.py` |
| F.1 perf table | local + cluster numbers in this doc | `build/v_outside/tests/bench_f1_variants.py` |

## What's left

1. **E.5 Pallas Megatron bwd** — currently the bwd path uses a
   pure-JAX mirror through `jax.vjp`; on cluster this shows up as a
   4× fwd→bwd ratio (vs 1.26× for v_outside which uses the D.1
   Pallas bwd kernel). A Pallas Megatron bwd would recover that gap.
2. **E.6 step 2** — fused down-matmul + streaming scatter kernel
   for comm-compute overlap (real perf win on Megatron — needs to
   beat both psum and psum_scatter+all_gather).
3. **D.6 cluster validation at D=7168** — local already runs;
   needs a cluster smoke with full DSv3 shape (E=32 D=7168) to
   confirm cross-host A2A interaction with the D-tiled kernel.
4. **Custom_vjp wrappers for v_inside Option β + Megatron-scatter**
   — both currently fwd-only; bwd would let F.1 produce a complete
   bwd column.
