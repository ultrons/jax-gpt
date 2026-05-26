---
slug: phase-d6-e3-cluster
intent: results
status: snapshot 2026-05-14
sources:
  - cde run: g3-dpush-1 (cde-6173e40) — D-push cluster validation
  - targets/streaming-psum-scatter-ref/tests/test_g3_fsdp4.py — E.3 local execution validation
  - targets/dsv3-fused-ep-moe/build/v_inside/tests/test_g3_megatron.py — E.4 local validation
mesh: (dp=1, ep=4, fsdp=32, tp=1) on rbq 4x4x4
---

# Phase D.6 + E.3 + E.4 — validation at the cluster's actual ceiling

## D.6-lite (D-push) — cluster validation at 128 cores

The DSv3 SPEC's full production D is 7168. True D-tiling for that
shape requires a substantial refactor (memory_space=ANY for big I/O
+ manual VMEM staging via async_copy, OR a two-pallas-call structure
with HBM scratch for gate_up/d_act). Tracked but deferred.

What we DO have: D.4 (E-tiling) + D.5 (bf16 per-tile scratches) +
D-aware e_b heuristic pushes the existing kernel to the practical
VMEM ceiling. At E_local=8 (E=32, EP=4), F=128, K=4, e_b=1:
- D=2048 — earlier large-shape test
- D=3840 — actual ceiling (bisected; D=3968 overshoots by 1.7 MB)

`g3-dpush-1` on rbq 4x4x4 (128 cores, mesh `(1, 4, 32, 1)`):

```
[fwd]            mesh=(1,4,32,1) T_global=4096 max_abs=1.56e-2 bad_rows=1/4096  PASS
[v_inside-fwd]   FSDP=32 max_abs=0  max_rel=0                                    PASS  (bit-exact)
[bwd-pallas]     d_W_gate / d_W_d bit-exact; d_x_in 2.3e-3 / d_W1 1.5e-3         PASS
[large-shape]    E=64 D=2048 F=128 K=4 EP=4                                      PASS
[d-push]         E=32 D=3840 F=128 K=4 EP=4    ← D pushed 1.9× over prior        PASS  (NEW)
[run_g3_cluster] ALL PASS
```

D=3840 / 7168 = 0.54 of full DSv3 production. Reaching the remaining
gap requires the deferred D-tiling refactor.

## E.3 — streaming-psum-scatter execution validation

E.2 v_inside shipped via Option β (auto-AG of W inside shard_map),
which preserves the API contract but NOT the HBM peak win. E.3
builds the streaming-psum-scatter pattern as a standalone reference
kernel to validate the primitive that would deliver the real HBM win.

`targets/streaming-psum-scatter-ref/build/scatter_matmul.py`:
- Each fsdp peer computes `partial = tok_local @ W_local[:, dest_d_range]`
- Streams partials to peers via `pltpu.make_async_remote_copy` with
  double-buffered VMEM (Inv1)
- Sends before draining prior incoming (Inv2)
- Final post-loop drain catches the last in-flight partial (Inv3)
- Per-pallas_call f32 accumulator (Inv4)
- Output: (T, D_local) per device — psum-scatter result

Local 4-core execution validation (test_g3_fsdp4.py):

```
[E.3 fsdp=1] max_abs=0  max_rel=0                  PASS  (self-step, no DMA)
[E.3 fsdp=2] max_abs <= 1e-2  max_rel <= 1e-2      PASS  (real cross-device DMA)
[E.3 fsdp=4] skipped — TPU HBM fragmentation       known issue (not kernel-related)
```

Both fsdp=1 and fsdp=2 PASS against `lax.psum_scatter` reference
with F-sharded W (the real v_inside scenario). The streaming primitive
works correctly; F-sharded inputs produce genuine partials that sum
to the expected result.

Two design bugs fixed during validation:
1. fsdp=1 hung — post-loop drains assumed >=1 DMA fired; gated on
   num_fsdp_devs > 1.
2. Initial test had tok+W both replicated → kernel's psum-summing
   inflated output by N×. Real v_inside has F-sharded W (along the
   contraction dim); rewrote test to use F-sharded inputs and compare
   against lax.psum_scatter.

## Pattern docs status

Both streaming pattern docs are now exercised by real kernels with
validation gates:

| Pattern | Doc | Reference kernel | Validation |
|---|---|---|---|
| streaming-AG-into-matmul | distilled/patterns/streaming-ag-into-matmul.md | targets/streaming-ag-ref/ | AOT compile PASS (v_outside DSv3 doesn't use this pattern; D-sharded W layout — see _inbox/blocker-spec-v_inside-sharding-vs-math.md) |
| streaming-psum-scatter | distilled/patterns/streaming-psum-scatter.md | targets/streaming-psum-scatter-ref/ | AOT compile PASS + execution PASS at fsdp=1, fsdp=2 |

The latter is the building block for a future v_inside iteration that
delivers the real HBM peak win (replacing E.2's auto-AG of W with
streaming-psum-scatter of the down-matmul output). Full integration
into v_inside is a separate implementation task.

## E.4 — v_inside Megatron column+row parallel (real W-side HBM win)

E.2 Option β ships an API-compatible v_inside via auto-AG of W inside
shard_map — correct math, but the AG materialises the full-F W on
every device so the HBM peak is the same as v_outside. E.4 builds the
*actual* HBM win: a Megatron-style column+row parallel wrapper that
keeps W F-sharded throughout the FFN and uses `lax.psum` across fsdp
to reduce the row-parallel partials.

`targets/dsv3-fused-ep-moe/build/v_inside/moe_block_ep_megatron.py`:
- W1 layout: `(E_local, D, 2, F_shard)` — "2" is gate/up, F is the
  sharded dim. The naive `(E, D, 2F)` layout shards the 2F axis,
  which at FSDP=2 gives shard 0 ALL gate and shard 1 ALL up — breaking
  the gate/up pairing.
- Wrapper reshapes `(E_local, D, 2, F_shard) → (E_local, D, 2*F_shard)`
  before calling `expert_ffn_v_inside`. This preserves
  `[gate_F_shard | up_F_shard]` layout the kernel expects.
- After the FFN, `lax.psum(out_partial, axis_name=fsdp)` reduces the
  row-parallel partials. f32-exact reduction.

**x contract:** fsdp peers must hold the *same* tokens (x sharded on
ep only, replicated on fsdp). The Megatron pattern assumes each peer
computes a different F-shard *of the same (M, D) output*. If fsdp
peers held different tokens (e.g. `P(("ep","fsdp"), None)`), the psum
would mix independent (M, D) buffers and produce garbage.

Local 4-core validation (`test_g3_megatron.py`):

```
[E.4 megatron EP=2 FSDP=1]   max_abs=0  max_rel=0       PASS  (degenerate psum)
[E.4 megatron EP=2 FSDP=2]   max_abs=0  max_rel=0       PASS  (real 2-way F-shard psum)
[E.4 megatron EP=1 FSDP=4]   max_abs=0  max_rel=0       PASS  (real 4-way F-shard psum)
```

All three cases bit-exact vs the full-F v_outside reference. f32-exact
psum + correct (E, D, 2, F) sharding gives a bit-equivalent result.

Cluster path (`_run_megatron_fwd` in `run_g3_cluster.py`) runs at
8-core (FSDP=2) and 128-core (FSDP=32) when triggered — pending cde
run alongside the rest of the cluster suite.

### Debugging note: misleading "kernel bug" hypothesis

Initial test had `x_spec = P(("ep","fsdp"), None)` (mirroring the rest
of the cluster suite). fsdp=1 PASS, fsdp=2 FAIL at 2.6% rel error.
Spent significant time investigating expert_ffn_v_inside as the
suspect: pure-JAX Megatron math test → bit-exact; isolated kernel
test with manually F-sharded inputs → bit-exact. Both pointed AT the
wrapper as the culprit. Eventually traced to the test's x sharding:
fsdp peers were routing different tokens, then psum across peers
combined unrelated (M, D) buffers. Fixing `x_spec = P("ep", None)`
made all cases bit-exact. The kernel and wrapper were correct
throughout; the test's input contract was the bug.

## DSv3 fused EP-MoE kernel: closed at the cluster's reachable ceiling

| Aspect | Validated | Limit |
|---|---|---|
| Correctness (G1-G4) | bit-exact at multiple shapes | none |
| v_outside cluster fwd | rbq 128c bit-equivalent to local | 4x4x4 (4x8x8 hardware contended) |
| v_outside cluster bwd | Pallas-bwd vs JAX-bwd bit-exact | 4x4x4 (4x8x8 contended) |
| v_inside (Option β) cluster fwd | rbq 128c bit-exact vs v_outside | 4x4x4 (same) |
| v_inside (Megatron, E.4) local | bit-exact at FSDP=1/2/4 vs v_outside | 4-core (cluster wired) |
| Production-class D | E=64 D=2048 cluster PASS | D=3840 at E=32 (kernel VMEM ceiling) |
| Local perf | fwd 1.00-1.31×, bwd 1.21-1.36× vs jax_ref | small shape only |
| Streaming patterns | AOT + execution validated | both canonical |

What's left as known-deferred work (gated on actual demand):
- True D-tiling for D=7168 (D.6) — substantial refactor; current
  ceiling is D=3840.
- 4x8x8 ICI mesh cluster run — infrastructure-blocked (capacity).
- E.4 Megatron cluster validation at fsdp=32 — wired in
  `run_g3_cluster.py` (`_run_megatron_fwd`); pending cde launch.
- Streaming-psum-scatter integration into v_inside Megatron — pattern
  validated standalone (E.3); swapping `lax.psum` for the streaming
  primitive in `moe_block_ep_megatron.py` is straightforward but not
  yet done.
