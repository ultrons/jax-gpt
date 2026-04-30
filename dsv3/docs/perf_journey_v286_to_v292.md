# DSv3 671B Perf Journey: v286 → v292

Topology: 4×8×8 bodaborg slice (256 chips × 2 cores = 512 devices)
Mesh: dp=1, fsdp=128, ep=4, tp=1 (EP on X, FSDP on Y·Z·C)
Config: GBS=2048, S=4096, K=8, E=256, gmm_ag MoE backend, splash attention

## Standing best: v292 = 1834 TPS/chip @ 29.7% MFU (17.9 s/step)

## Result table

| Run | Change | TPS/chip | MFU |
|---|---|---|---|
| v286 | EP=16 + CP+A2A+Splash (4×4×16) | 654 | 10.6% |
| v287 | EP=4 FSDP=128, no CP, AG dispatch (new mesh, gmm_ag backend) | 1400 | 24% |
| v289 | + drop post-AG dummy-expert pad concat | 1527 | 26% |
| v290 | + scheduling_group + optimization_barrier on AGs | regressed | — |
| v291 | + Splash use_fused_bwd_kernel + block_q=2048 | 1772 | 28.5% |
| **v292** | **revert v291's scheduling_group hints (kept Splash)** | **1834** | **29.7%** |

Compare to v264 baseline (no EP, FSDP=256 TP=2): 1740 TPS/chip — v292 is +5.4% past it.

## Changes that worked (cumulative)

### Mesh + dispatch (v287)
- **Topology-aware mesh**: greedy axis assignment. EP gets smallest physical axes (here X=4, 1D ring). FSDP gets the rest (Y·Z·C = 8·8·2, 2D torus).
- **AG dispatch (gmm_ag backend)**: AllGather tokens on EP + AllGather F-dim weights on FSDP. Wins over A2A when EP < K+1 (here 4 < 9).
- **No CP + joint batch sharding**: `P((batch, ep), None, None)` for the layer carry — B sharded on fsdp×ep so per-device tokens stay at 16384.

Code: `model.py:_carry_spec`, `model.py:create_mesh` greedy mapping, `model.py:_expert_mlp_gmm_ag_body`.

### Drop post-AG dummy concat (v289)
- The post-AG `concat([wi_0_t, zero_wi], axis=0)` materialized the full AG result in HBM before ragged_dot could start, defeating XLA's chunk-by-chunk pipelining of the AG ring with the MXU.
- **Fix**: route invalid slots to expert 0 with weight 0 (instead of dummy expert E_local). ragged_dot reads AG output directly; expert 0 does negligible extra work on zero-weighted rows.
- Gain: +9% TPS (1400 → 1527).

### Splash fused bwd kernel + bigger blocks (v291)
- `use_fused_bwd_kernel=True` produces dQ + dKV in one Q/K/V-reading pass instead of two separate kernels.
- `block_q = block_kv = 2048` (was 512). 2 Q-blocks at S=4096 instead of 8 → less per-tile overhead, better MXU utilization.
- Splash combined fwd+bwd time: 99 s → ~45 s in the profile.
- Gain: +12% TPS (1527 → 1772 first; then +3% more in v292 after reverting scheduling_group).
- Source: `model.py:_splash_attention`, `model.py:_splash_cp_attention`. Pattern from MaxText `init_splash_kernel`.

## Dead ends (also informative)

### v290: scheduling_group + optimization_barrier on AGs (REGRESSED)
- Wrapped EP token AGs (group 2) and FSDP weight AGs (group 1) with `set_xla_metadata(_scheduling_group_id=N)` + `optimization_barrier`.
- Hypothesis: XLA's collective scheduler would interleave them on disjoint ICI links.
- Result: per-AG times unchanged from v289, but the SparseCore offload that v289 had organically on one weight AG was lost. Net regression.
- Lesson: scheduling_group hints are a no-op or worse for our pattern. XLA already knew about the dependency graph; the optimization_barrier displaced the SC offload.

### v293a: sort-then-scatter preflight (REGRESSED)
- Pre-sort `local_tids` before the scatter add.
- Hypothesis: XLA's `scatter_custom_fusion` would respond to sorted indices with a faster lowering.
- Result: -8% TPS. The argsort overhead exceeded any scatter benefit. Confirms scatter's 1.8% HBM-eff ceiling is structural.

### v293b: vendored MaxText SC gather-reduce kernel (REGRESSED)
- Vendored `gather_reduce_sc.py` + `sort_activations.py` from MaxText.
- Padded `out_local (max_local, D)` into `(TK, D)` = 7.5 GB intermediate, then ran the SC kernel for combine.
- Result: -11% TPS. The pad+gather HBM traffic (~17 GB/call) exceeded the original scatter's 2.8 GB/call.
- The SC kernel's combine assumes consecutive-K layout (MaxText's F-sharded weights). Our E-sharded layout requires materializing that layout, paying the pad cost.

### v294 / v294b: cross-layer weight prefetch (REGRESSED)
- v294: 3-tuple scan carry `(x, aux, ws_ag)` with shifted-weights pattern. OOM at compile (109 GB carry-stash for `d_ws_ag` stacked across 58 iterations).
- v294b: scan `unroll=2` to let XLA pipeline cross-iteration AG. Result: -7% TPS. Bigger HLO body without cross-iter pipelining benefit.
- Lesson: cross-layer prefetch needs MaxText's `pcast`/`reduced` AG pattern (next attempt v295) to avoid the d_ws_ag stash.

## v292 profile breakdown (where remaining time goes)

Total: 285.9 s for 5 steps (multi-chip merged).

| Category | Time | % | Notes |
|---|---|---|---|
| ragged_dot (MoE FFN) | ~76 s | 26.6% | 11 fusions @ 47% MXU |
| Splash MHA | ~45 s | 15.6% | post-fused-bwd; was 99 s in v289 |
| **scatter (combine)** | **26.7 s** | **9.4%** | 1.8% HBM eff — known unaddressed |
| **EP collectives exposed** | **~26 s** | **9.1%** | token AG 14.5s + scatter 5.9s + RS 5.7s |
| **FSDP weight AG exposed** | **~19 s** | **6.7%** | 4 AGs at ~4.5–4.9s each |
| Other (sort, embed, control) | ~93 s | ~32% | distributed overhead |

Hardware utilization: 21.5% MXU, 14.6% HBM (whole-step weighted). Headroom both ways.

## What's left on the table (priority order)

| Lever | Est. gain | Effort | Status |
|---|---|---|---|
| MaxText pcast/reduced AG (v295) | +3–5% | 1 day | next attempt |
| Microbatching + staggered_call | +5–8% | 3–5 days | future |
| Custom_vjp avoiding d_ws_ag stash | +2–4% | 1–2 days | extension of v295 |
| Pallas TC sorted-segment-sum kernel for scatter | +5–7% | 2 days | parked (needs careful kernel work) |
| Activation offload (mlpwi_0/1 to pinned host) | +3–5% | 1 day | unstudied |

Combined ceiling (if all land): ~+15-20% → ~2100-2200 TPS/chip on this slice.

## Key files

- `model.py:_expert_mlp_gmm_ag_body` — main MoE compute body (gmm_ag backend)
- `model.py:_splash_attention`, `_splash_cp_attention` — Splash with fused_bwd
- `model.py:create_mesh` — topology-aware greedy mesh construction
- `model.py:_carry_spec` — carry sharding (CP-aware)
- `mini_dsv3/kernels/` — vendored MaxText SC kernels (unused in v292; integration parked)
- `k8s/dsv3-train-4x8x8-v292-noschedgrp.yaml` — last winning YAML

## Profile path

`gs://max-experiments/dsv3/profiles/v292-nosched-ep4-fsdp128-4x8x8/`

Re-analyze with: `cd ~/ml-experiments/timing && python -m xla_shell -c "read_xplane gs://... --multi-chip; report_timing --top 25; list_collectives --overlap"`
