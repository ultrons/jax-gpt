---
slug: kernel-agent-945964d-feedback
intent: evaluation
status: snapshot 2026-05-13
sources:
  - ~/kernel-agent @ 945964d  (snapshotted to research/dsv3/kernel-agent-snapshot-945964d/)
  - jax-gpt/jax_gpt/models/dsv3/model.py (_expert_mlp_gmm_ag_body, lines 1685-1958)
  - research/dsv3/kernel_agent_aot_check.py + /tmp/kernel_agent_aot.log
related:
  - autoperf/iter_log.md  (current autoperf state, BASELINE = v304 + iter-16 attn_proj_out SAVE)
  - autoperf/lever_queue.md (next lever priorities)
---

# kernel-agent @ 945964d — usefulness assessment for jax-gpt DSv3 training

Question asked: is the fused MoE kernel built by kernel-agent at commit
`945964d` useful for our autoperf DSv3-671B training workload?

Short answer: **not yet, but on a promising trajectory.** At the snapshot
commit the kernel passes only at a sub-production shape (E=64, D=2048,
K=4); the actual jax-gpt training shape (E=256, D=7168, K=8) hits a
`RESOURCE_EXHAUSTED` at AOT compile because the kernel does not yet tile
the D dimension. The kernel-agent SPEC explicitly tracks this gap as
"D.6 — true D-tiling" and the commit message says so. A drop-in trial
into the `_expert_mlp_gmm_ag_body` slot is **blocked on D.6** plus three
other adapter-layer issues listed below.

A snapshot of the kernel at this commit lives at
`research/dsv3/kernel-agent-snapshot-945964d/` so future iters can refer
to the exact code reviewed here, even as the upstream `~/kernel-agent`
repo continues evolving. To revisit the upstream state at this commit:
`(cd ~/kernel-agent && git show 945964d)`.

---

## 1. What the kernel actually is at 945964d

A two-variant Pallas implementation of a full MoE block (router →
A2A scatter → per-expert FFN → A2A gather → segment_sum unsort+combine):

- `v_outside`: weights are FSDP-AG'd by JAX **outside** the kernel; the
  kernel sees full-F weights and does only the local matmul.
- `v_inside`: caller passes F-sharded weights; an `auto-AG` at the
  `shard_map` boundary brings them to full-F inside the body. At 945964d
  this is **functionally identical to v_outside in HBM peak** — the
  promised streaming-AG inside the matmul is parked in Phase E.3
  (`distilled/patterns/streaming-ag-into-matmul.md`).
  An actual streaming-AG reference kernel landed one commit later
  (63e87e2, `targets/streaming-ag-ref/`) but is a standalone matmul
  reference, not wired into the MoE block.

Code map (snapshot):
```
build/v_outside/router.py            Phase 1: gate matmul + iterative-argmax top-K   (Pallas, TC)
build/v_outside/expert_ffn.py        Phase 3: per-expert FFN, Python for-loop over E   (Pallas, TC)
build/v_outside/expert_ffn_bwd.py    Phase 3 backward, custom_vjp                      (Pallas, TC)
build/v_outside/moe_block_ep.py      Composed fwd block: ragged_all_to_all + permute   (JAX)
build/v_outside/moe_block_vjp.py     custom_vjp for the whole block                    (JAX)
build/v_outside/a2a_helpers.py       compute send/recv offsets for ragged_all_to_all   (JAX)
build/v_inside/{expert_ffn,moe_block_ep}.py  v_inside variant (auto-AG at present)
build/tools/aot_check.py             reusable AOT compile gate
jax_ref.py                            pure-JAX reference (math contract + perf lower bound)
```

### What's bit-exact / what's measured (per `results/phase_d5_cluster_prod_shape.md`)

| Gate | Shape | Mesh | Status |
|---|---|---|---|
| G2 fwd vs jax_ref | E=8, D=64-256 | 1 device | PASS |
| G2 bwd vs jax.vjp | E up to 64, D up to 2048 | 1 device | PASS (D.5 0.06 % rel @ D=2048) |
| G3 fwd vs jax_ref | E=32, D=512 | rbq 128 cores | PASS (1/4096 row at noise floor) |
| G3 bwd (Pallas vs JAX peer) | E=32, D=512 | rbq 128 cores | PASS (d_W_gate / d_W_d bit-exact) |
| **G3 prod-class fwd** | **E=64, D=2048, K=4** | **rbq 128 cores** | **PASS (finite, no NaN)** |
| G5 perf (single device sweep T=16-1024) | small | local 4 cores | fwd 1.00-1.31× vs jax_ref, bwd 1.21-1.36× |

The 1.21-1.36× backward speedup is measured **against a pure-JAX
autodiff baseline** (`jax_ref.moe_grads`), NOT against the gmm_v2 +
ragged_dot path we actually train with today. See §3 below.

---

## 2. How it maps onto jax-gpt's `_expert_mlp_gmm_ag_body`

jax-gpt's production training path (model.py:1685) and the kernel-agent
kernel solve the same problem but use **different EP-communication
idioms**:

| Concern | jax-gpt today (`gmm_ag`, BASELINE = v304 iter-16) | kernel-agent 945964d |
|---|---|---|
| EP token movement | `lax.all_gather` across `ep_axis` + local sort + slice | `lax.ragged_all_to_all` + local permute |
| FSDP weight movement | `lax.all_gather` across `fsdp_axis` (one-shot, outside loop) | (v_outside): JAX outside; (v_inside): auto-AG inside `shard_map` |
| Expert FFN math | `gmm_v2` Pallas (fused gate+up+silu) over a ragged group | Python for-loop over E_local; per-expert dense `(M,D)×(D,2F)` matmul with eid mask |
| Token chunking | `n_chunks=2` chunked AG+compute+scatter+RS (overlap-driven) | one-shot per layer |
| Scatter back to per-token rows | HBM `.at[idx].add(...)` then `psum_scatter` across EP | `segment_sum` after second `ragged_all_to_all` |
| fp8 weight path | yes (cfg.moe_use_fp8_weights) | no |
| Cross-layer FSDP weight prefetch | yes (cfg.moe_xlayer_prefetch) | no |

This means **swapping `_expert_mlp_gmm_ag_body` for `moe_block_ep_fwd`
would not be a drop-in patch**: it would change the whole EP
communication shape (AG-dispatch → A2A-dispatch) for the moe block, the
weight AG pattern (one-shot → kernel-internal at v_inside maturity), and
remove the chunked overlap structure jax-gpt iter-2b relies on. Those
are real architectural differences, not just a kernel substitution.

### Per-expert for-loop and the E=256 wall

`build/v_outside/expert_ffn.py:74` runs a Python `for e in range(E_local)`
inside the Pallas body and applies an integer-arithmetic eid mask to
zero out rows that don't belong to expert `e`. The `(M, 2F) = tok @
W1[e]` matmul is computed for **every token, every expert**, then masked.
This is the "dense per-expert pass" antipattern that ragged-dot
(`jax.lax.ragged_dot`) and `gmm_v2_train` exist to avoid.

At jax-gpt production E_local=64, this for-loop:
- unrolls into 64 `bt × D × 2F` matmuls per Pallas tile
- multiplies real-FLOPs by `E_local / K_per_token` (each token visits ~K
  experts, but the kernel computes all E_local)
- explodes compile time (the jax_ref guards explicitly at `MAX_REFERENCE_E
  = 32` because trace time at E=256 is "~30 min")

The kernel will still produce correct results if it AOT-compiles, but the
compute pattern is fundamentally less efficient than `gmm_v2` at large
E_local. This is the second big gap, behind D-tiling.

---

## 3. The bwd speedup number does not transfer

The 1.21-1.36× bwd speedup in `results/phase_d_full_bench.md` is:

```
v_outside  fwd+bwd (Pallas custom_vjp)
  vs
jax_ref    fwd + jax.grad(jax_ref) bwd  ← pure JAX, NO Pallas anywhere
```

This is exactly what `SPEC §8.1` defines: "kernel must be ≥ pure-JAX
baseline." A 1.3× lead over pure JAX is the **lower bound** the kernel
must clear to earn its complexity — it's the floor, not the ceiling.

The real ceiling for jax-gpt is the gmm_v2 + ragged_dot path on a
production-bound MoE block. From `autoperf/iter_log.md` iter-4, the
current per-step time at v304 is ~16,656 ms for `moe_experts/moe_gmm_ag`
(48 % of step) decomposed as fwd 5,436 ms (gmm_v2 kernel = 1,845 ms +
scatter = 1,685 ms + dispatch AG = 998 ms) and bwd 11,219 ms (transpose
7,913 ms + jvp 3,306 ms). The kernel-agent's measured numbers are at a
toy shape (E=16 D=256) on local 4-chip hardware against an autodiff
baseline that doesn't use ragged_dot. The benchmark does not predict
behavior at production.

To know whether kernel-agent's kernel is faster than `gmm_v2`-based
`_expert_mlp_gmm_ag_body`, we would need a same-shape, same-mesh head-
to-head — and that bench does not exist at 945964d.

---

## 4. AOT probe at jax-gpt production shapes

`research/dsv3/kernel_agent_aot_check.py` runs the kernel through the
kernel-agent AOT harness at three shapes (log: `/tmp/kernel_agent_aot.log`).

| Shape | Topo | Mesh (dp,ep,fsdp,tp) | Verdict |
|---|---|---|---|
| validated@D5  (E=64  D=2048 F=128  K=4 EP=4 FSDP=2)  | tpu7x:2x2x1 | (1,4,2,1)   | **PASS** (22.1 s) |
| mid           (E=64  D=2048 F=128  K=8 EP=4 FSDP=2)  | tpu7x:2x2x1 | (1,4,2,1)   | **PASS** (25.2 s) |
| prod@dsv3     (E=256 D=7168 F=2048 K=8 EP=4 FSDP=128) | tpu7x:4x8x8 | (1,4,128,1) | **FAIL** (140 s) |

The production-shape failure is exactly the D.6 gap the SPEC documents
(`results/phase_d5_cluster_prod_shape.md` "What's still open"):

```
RESOURCE_EXHAUSTED: Allocation (size=3758096384) would exceed memory (size=67108864)
shape = u8[3758096384]{0}, space=vmem, scoped, tag = 'input window allocation for
operator input 2. The window shape is bf16[64,7168,4096], while the full shape is
bf16[64,7168,4096]. ... This allocation is single buffered.'
```

Translation: the kernel tries to bring the full per-device `W1[E_local=64,
D=7168, 2F=4096] = 3.5 GB` into VMEM in a single window. VMEM is 64 MB.
The grid windows the M-dim only, not the D-dim, so D=7168 puts the
window above any plausible VMEM ceiling. D.6 is what the kernel-agent
roadmap calls the fix: a grid over D-chunks with per-d_chunk persistent
VMEM scratches. **Until D.6 lands, this kernel cannot lower at jax-gpt
production shape.**

Note that K=8 alone is fine — the (E=64, K=8) intermediate point AOT-
PASSes at 25.2 s, so K is just a config flip as the SPEC promises.
The wall is D.

---

## 5. Gaps that would have to close before adoption

In rough order of impact:

1. **D.6 true D-tiling** (blocker). Without it, no compile at
   D=7168. Tracked as upstream kernel-agent task #52.
2. **gmm_v2-class FFN math** (efficiency blocker). The Python for-loop
   over `E_local=64` with mask-and-zero is wasteful vs ragged_dot;
   without this the kernel can compile at production but won't beat
   `gmm_v2_train` in TPS/chip.
3. **AG-dispatch vs A2A-dispatch architectural choice**. jax-gpt is
   committed to AG-dispatch for v304 (the chunked AG+RS path), and we
   have empirical evidence the AG path beats A2A on v7x (see iter-14
   `aot_collective_fusion_check.py` finding: `ragged_all_to_all` not in
   the SC-offload flag set on production manifest; v7x prefers RS).
   The kernel-agent's `ragged_all_to_all` is a different design point.
   We would need an apples-to-apples bench at production shape before
   committing to the swap.
4. **No chunked overlap structure** (perf blocker). v304's
   `_expert_mlp_gmm_ag_body` overlaps `chunk0 RS / chunk1 AG / chunk1
   compute` — without that, the kernel pays the full RS exposure cost.
5. **fp8 weight path**. jax-gpt supports `moe_use_fp8_weights` (halves
   the AG'd weight allocation 7 GB → 3.5 GB). Kernel-agent v0 is bf16-
   only by SPEC §1.
6. **Cross-layer FSDP weight prefetch**. jax-gpt has
   `moe_xlayer_prefetch` for hiding W AG behind prior-layer compute.
   The kernel does not expose a hook for this.
7. **Production-mesh validation**. SPEC §5.4 production target is
   4x8x8 (EP=4, FSDP=128), but the kernel has only ever cluster-run on
   rbq 4x4x4 — Kueue does not expose a 4x8x8 partition label on the
   available hardware. Even with D.6 the kernel has not been tested at
   the mesh shape we actually use.

---

## 6. Bottom-line recommendation

For autoperf's purposes:

- **Do not** put this kernel in iter-N's lever queue yet — it can't
  compile at our shape, and at the validated shape we'd be measuring at
  a non-production point.
- **Do** track upstream kernel-agent for the D.6 milestone. Once D.6
  lands and the kernel compiles at (E=256, D=7168, K=8, EP=4, FSDP=128),
  the right experiment is a same-mesh head-to-head against the gmm_v2
  + ragged_dot `_expert_mlp_gmm_ag_body` we use today. Until then, the
  comparison is "Pallas vs pure-JAX-autodiff at a toy shape," which
  doesn't generalize.
- **Possibly useful even before D.6:**
  - The `iterative-argmax-topk.md` pattern is a clean reference for
    bf16-incompatible top-K cases (we use `jax.lax.top_k` today on the
    f32 router output, which is fine, but the pattern is worth
    bookmarking).
  - The `streaming-ag-into-matmul.md` parked design (and the
    standalone reference kernel in 63e87e2 `targets/streaming-ag-ref/`)
    is the closest substrate work to what would unlock the iter-7
    OFFLOAD-class path we got NaN on (jax-gpt#2): if you control the
    AG-restore yourself, you sidestep the silent CSE / async DMA-race
    failure modes XLA's offloader hits.
  - `build/tools/aot_check.py` is a clean, reusable virtual-topology
    AOT harness; we already have an ad-hoc copy of this pattern
    (`research/dsv3/aot_collective_fusion_check.py`), so this is just
    a nicer template.

If the kernel-agent project gets to bench-against-`gmm_v2`-at-production-
mesh and beats it by a meaningful margin (>5% TPS/chip), THAT is when
autoperf should consider swapping. Until then this kernel is a parallel
investigation, not a candidate for our hot path.

---

## 7. Files I produced for this evaluation

- `research/dsv3/kernel-agent-snapshot-945964d/` — pinned copy of
  `~/kernel-agent/targets/dsv3-fused-ep-moe` at 945964d, including
  `SPEC.md`, `jax_ref.py`, the build/ tree (~700 KB).
- `research/dsv3/kernel_agent_aot_check.py` — AOT probe script that
  runs the kernel through the kernel-agent harness at the three shape
  points in §4.
- `/tmp/kernel_agent_aot.log` — captured probe output.
- `research/dsv3/kernel_agent_945964d_feedback.md` — this document.
