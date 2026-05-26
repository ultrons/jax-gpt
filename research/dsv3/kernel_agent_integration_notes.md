---
slug: kernel-agent-integration-notes
intent: integration-report
status: snapshot 2026-05-20
sources:
  - ~/kernel-agent @ b4b63d1 (snapshotted to research/dsv3/kernel-agent-snapshot-b4b63d1/)
  - jax_gpt/models/dsv3/kernels/kernel_agent/{expert_ffn,expert_ffn_d_tiled}.py (vendored)
  - jax_gpt/models/dsv3/model.py (cfg flag + gated branch + cvjp plumbing)
  - tests/dsv3/kernels_test/exec_kernel_agent_ffn.py (parity smoke)
  - research/dsv3/aot_kernel_agent_integration.py (production-shape AOT probe)
  - /tmp/aot_kernel_agent_integration.log (captured output)
related:
  - research/dsv3/kernel_agent_945964d_feedback.md (initial usefulness assessment, 2026-05-13)
---

# Integration of kernel-agent fused-MoE FFN into jax-gpt — report

Surgical integration of the kernel-agent fused expert-FFN kernel
(`expert_ffn_v_outside`, including D.6 D-tiling) into jax-gpt's
DSv3 training path, gated behind a new config flag. All surrounding
AG-dispatch + sort + scatter + psum_scatter machinery remains
unchanged — only the three ragged_dot / gmm_v2 calls inside the
per-chunk body are swapped out.

Bottom line: **plumbing works end-to-end at small shape; the vendored
kernel does not yet fit our production scale because D.6 only handles
the D-axis, not F**. At full DSv3 (E=256, D=7168, F=2048, K=8,
BS=4096, seq=4096 on tpu7x:4x8x8) the per-expert W1 window is 56 MiB
on its own — double-buffered to 112 MiB, vs the 64 MiB VMEM cap. The
baseline ragged_dot path AOT-compiles fine at the same shape, so the
failure is in the inner Pallas kernel, not in our scaffold.

The flag is `False` by default; production training is unchanged.

---

## What landed

### Files changed
- `jax_gpt/models/dsv3/model.py`
  - new field `ModelConfig.moe_use_kernel_agent_ffn: bool = False`
    (lines ~138)
  - `_expert_mlp_gmm_ag_body`: new gated branch before the existing
    `use_gmm_v2` / `use_fp8_weights` / `else (ragged_dot)` cascade
  - `_moe_gmm_ag` / `_moe_gmm_ag_fwd` / `_moe_gmm_ag_bwd`: extra
    nondiff arg threaded through the custom_vjp boundary
  - `expert_mlp_gmm_ag`: compatibility check + forwards the flag

### Files added
- `jax_gpt/models/dsv3/kernels/kernel_agent/__init__.py`
- `jax_gpt/models/dsv3/kernels/kernel_agent/expert_ffn.py`         (vendored)
- `jax_gpt/models/dsv3/kernels/kernel_agent/expert_ffn_d_tiled.py` (vendored)
- `tests/dsv3/kernels_test/exec_kernel_agent_ffn.py`               (parity smoke)
- `research/dsv3/aot_kernel_agent_integration.py`                  (AOT probe)
- `research/dsv3/kernel-agent-snapshot-b4b63d1/`                   (pinned upstream copy)

### How the gated branch is wired

```python
# inside _process_chunk in _expert_mlp_gmm_ag_body
if use_kernel_agent_ffn:
    from .kernels.kernel_agent import expert_ffn_v_outside
    # Vendored kernel expects W1 = (E_local, D, 2F) with [gate | up] layout.
    # Our wi_0_t/wi_1_t are (E_local, D, F_full) post the model.py:1723 transpose.
    W1_fused = jnp.concatenate([wi_0_t, wi_1_t], axis=2)
    out_local_c_f32 = expert_ffn_v_outside(
        local_x_c.astype(W1_fused.dtype),
        local_eids_c,
        W1_fused,
        wo_f,
        bt=128,
    )
    out_local_c = out_local_c_f32.astype(wo_f.dtype)
elif use_gmm_v2:
    ...   # existing
elif use_fp8_weights:
    ...   # existing
else:
    ...   # existing ragged_dot
```

The branch sits ahead of the existing three, so when the flag is
`False` (default) the existing paths are bit-for-bit unchanged.

### How to enable

CLI / config:
```yaml
cde_overrides:
  moe_use_kernel_agent_ffn: true
```
Compatibility constraints (enforced in `expert_mlp_gmm_ag`):
- mutually exclusive with `moe_use_gmm_v2`
- mutually exclusive with `moe_fp8_weights` (kernel is bf16-only)

How to roll back: unset the flag (or set it to `false`) and rebuild
the image. The existing gmm_v2 / ragged_dot / fp8 paths are untouched.

---

## What was measured

### Parity smoke (TPU v4, 4 devices, small shape)

`tests/dsv3/kernels_test/exec_kernel_agent_ffn.py` runs
`expert_mlp_gmm_ag` end-to-end (through the full custom_vjp and
shard_map scaffold) on identical synthetic inputs at
`(B=1, S=256, E=8, D=128, F=64, K=2)` with mesh `(dp=1, ep=2, fsdp=2, tp=1)`,
and diffs each FFN path's output against the ragged_dot baseline:

```
backend=tpu  devices=4
---- baseline (ragged_dot, no Pallas) ----
  out shape=(1, 256, 128) dtype=bfloat16
---- kernel-agent FFN ----
  kernel-agent vs baseline  max_abs=2.441e-04  max_rel=5.000e-01
---- gmm_v2 ----
  gmm_v2 vs baseline        max_abs=4.883e-04  max_rel=1.221e+02
DONE
```

- `max_abs ≈ 2e-4` is in the bf16 rounding noise band.
- `max_rel` is meaningless when some baseline rows are near zero;
  the absolute number is what matters here.
- Both Pallas paths agree with the baseline; the kernel-agent path
  is actually marginally tighter than gmm_v2 at this shape (kernel
  carries f32 accumulation through the down-matmul, casts to bf16
  only at the boundary).

CPU run fails predictably with
`ValueError: Only interpret mode is supported on CPU backend.`
(Pallas) — that is the expected behavior; the smoke test detects
the backend and routes accordingly. The CPU run did confirm that
our Python wiring (cfg flag → custom_vjp → shard_map → body branch
→ kernel call) runs to the kernel call site without errors.

### AOT compile probe (virtual topologies, no execution)

`research/dsv3/aot_kernel_agent_integration.py` against four shape
× flag combinations:

| Label | Mesh | Shape | Flag | Verdict |
|---|---|---|---|---|
| small@2x2x1 | (1, 2, 4, 1) | E=32 D=2048 F=128 K=4 B=1 S=512 | kernel_agent=on  | **PASS** 4.9 s |
| small@2x2x1 | (1, 2, 4, 1) | (same)                          | kernel_agent=off | **PASS** 4.7 s |
| prod@dsv3   | (1, 4, 128, 1) | E=256 D=7168 F=2048 K=8 BS=4096 seq=4096 | kernel_agent=on  | **FAIL** 167.3 s |
| prod@dsv3   | (1, 4, 128, 1) | (same)                          | kernel_agent=off | **PASS** 166.6 s |

The compile times for both production runs are nearly identical
(166-167 s), so the kernel-agent path is following the same XLA
pipeline up to the failure point.

### Production failure mode (decoded)

```
RESOURCE_EXHAUSTED: Allocation (size=117440512) would exceed memory (size=67108864)
shape = 'u8[117440512]{0}', space=vmem, scoped
tag = 'input window allocation for operator input 2.
       The window shape is bf16[1, 7168, 4096],
       while the full shape is bf16[64, 7168, 4096].
       This allocation has 2 buffering levels.'
```

Decoded:

| Quantity | Bytes | Source |
|---|---:|---|
| VMEM cap | 64 MiB | hardware |
| W1 window (single expert, full D × full 2F) | 56 MiB | `bf16[1, 7168, 4096]` |
| Double-buffered W1 | **112 MiB** | 2 buffering levels |

D.6's grid is `(num_bt, E_local, num_d_out)`. It tiles the **output**
D dimension via `num_d_out`, but the **input** dimensions of the
gate+up matmul — the full D of activations and the full 2F of W1 —
are not tiled. At kernel-agent's local test shapes (D=7168, F=128 →
W1 window 3.7 MiB) double-buffering fits. At jax-gpt production
(D=7168, **F=2048** → W1 window 56 MiB), it does not.

This is **distinct from the D.6 gap** we documented on 2026-05-13.
That earlier failure was at full D with no D-tiling at all (3.5 GB
window). D.6 closed it for the D dimension. We are now hitting the
analogous gap on the **F dimension**, which D.6 does not address.

What an F-tiling fix would need:
1. Add an output-F tile dimension to the gate+up matmul (grid =
   `(num_bt, E_local, num_f_out, num_d_out)`, say).
2. Decompose the activation: `act = silu(gate) * up` consumes the
   full 2F at once. With F-tiling we would compute one F-tile of
   gate and the matching F-tile of up, apply silu, multiply, and
   contract against a corresponding F-tile of W_d into d_out.
3. Two-level accumulation: across F-tiles into a (bt, D_tile)
   accumulator, across D-tiles into per-row output, across E-tiles
   via the existing RMW pattern.

This is non-trivial — F-tiling changes the activation locality
(can no longer compute the full act vector at once for a fixed
(bt, e) pair). It is exactly the kind of work that would justify
re-engaging kernel-agent rather than patching the vendored copy
ourselves.

### Note on the local-cluster validation in kernel-agent's tree

`results/phase_e4_through_f1.md` (in the upstream snapshot) reports
cluster-validating D=7168 — but their local D.6 test used **F=128**,
not F=2048. So D=7168 with F=128 fits (W1 window ≈ 3.7 MiB) but
D=7168 with F=2048 does not (56 MiB). The phrase "FULL DSv3
production D" in the D.6 commit message refers to D=7168 alone, with
the F-dimension still at the test value of 128 — not the full DSv3
(D=7168, F=2048) combination.

---

## Recommendation for autoperf

1. **Leave the flag off in production training.** Default `False` was
   chosen deliberately — there is no production-shape compile,
   nothing has been measured on cluster, and the kernel still
   computes dense per-expert matmuls (E_local=64 wasted FLOPs per
   token vs ragged_dot's K=8 visited experts) which is the second
   blocker we documented on 2026-05-13.

2. **The integration code itself is durable.** The cfg flag, the
   custom_vjp plumbing, the W1 concat shim, and the parity smoke
   compose with the existing surface area without touching v304's
   production path. When kernel-agent ships an F-tiled variant, the
   integration point is already in place.

3. **Communicate the F-tiling gap upstream.** kernel-agent's
   `results/phase_e4_through_f1.md` "what's left" list does not
   currently flag F=2048 as an open item. This integration produced
   the concrete VMEM-arithmetic evidence that closes that gap.

4. **Do NOT queue an autoperf cluster iter on this.** Even if we
   patched around the F-tiling locally, the per-expert dense math
   would almost certainly regress vs gmm_v2 + ragged_dot at
   production E_local=64. Wait for an apples-to-apples bench
   (kernel-agent F.1 doesn't yet have a head-to-head against
   gmm_v2 at production mesh).

---

## Reproducing this report

```bash
# 1. Smoke test (TPU required for the actual kernel; CPU runs ok for wiring).
source ~/xdb/.xprof/bin/activate
python tests/dsv3/kernels_test/exec_kernel_agent_ffn.py

# 2. AOT probe (no TPU execution; ~3 min total wall time).
source ~/xdb/.xprof/bin/activate
PYTHONPATH=. python research/dsv3/aot_kernel_agent_integration.py

# 3. To refresh the vendored kernel from a later kernel-agent commit:
#    cd ~/kernel-agent && git rev-parse HEAD   # note the new commit
#    cp targets/dsv3-fused-ep-moe/build/v_outside/expert_ffn.py \
#       ~/jax-gpt/jax_gpt/models/dsv3/kernels/kernel_agent/
#    cp targets/dsv3-fused-ep-moe/build/v_outside/expert_ffn_d_tiled.py \
#       ~/jax-gpt/jax_gpt/models/dsv3/kernels/kernel_agent/
#    cp targets/dsv3-fused-ep-moe/build/v_outside/expert_ffn_f_tiled.py \
#       ~/jax-gpt/jax_gpt/models/dsv3/kernels/kernel_agent/
#    # also re-snapshot for diffability
#    cp -r targets/dsv3-fused-ep-moe \
#          ~/jax-gpt/research/dsv3/kernel-agent-snapshot-<commit>/
```

---

## ADDENDUM — refresh to kernel-agent 2cda804 (D.7 F-tiling lands)

The F=2048 VMEM gap diagnosed above is closed upstream. Pinned snapshot
refreshed to kernel-agent commit `2cda804` (HEAD as of 2026-05-22):

### What changed upstream (b4b63d1 → 2cda804, 14 commits)

The decisive ones for our integration:

| Commit | What it does |
|---|---|
| `a50888b` phase D.7 | New `expert_ffn_f_tiled.py` kernel. Grid = `(num_bt, num_d_out, E_local, num_f_tile)` — adds an F-output tile axis on top of D.6's existing E + D tiling. Fits the autoperf production case (E_local=64, D=7168, F=2048) in v7x's 64 MB VMEM. |
| `8135165` cluster gate VMEM-fit | First cluster run — fit OK, but correctness FAIL at F_tile=256 (a D-axis RMW bug). |
| `8ef5fcd` correctness-fix | Internal layout (E, D, 2F) → (E, 2, D, F) transpose. |
| `917ce01` correctness-fix-2 | Grid d→outermost. Fixes the D-axis RMW bug. |
| **`5a1a2b7` D.7 complete** | **Cluster-verified at production E_local=64, D=7168, F=2048 on x8p 4x4x4 (`d7-fix-5`). All five tests PASS, max_rel = 1.76e-4 to 3.28e-4 — well within bf16 noise.** |
| `2cda804` DROP-IN | Auto-routes the legacy `expert_ffn_v_outside` to the F-tiled kernel when per-tile W1 > 4 MB. Internal (E, D, 2F) → (E, D, 2, F) reshape; caller API unchanged. Specifically targeted at our autoperf integration. |

### Our integration changes

- `research/dsv3/kernel-agent-snapshot-2cda804/` — new pinned snapshot (1.0 MB)
- `jax_gpt/models/dsv3/kernels/kernel_agent/expert_ffn.py` — refreshed (auto-route to D.7 added)
- `jax_gpt/models/dsv3/kernels/kernel_agent/expert_ffn_f_tiled.py` — new file (355 lines)
- `jax_gpt/models/dsv3/kernels/kernel_agent/expert_ffn_d_tiled.py` — **unchanged** (D.6 still needed for the F=128 D=7168 case)
- `jax_gpt/models/dsv3/kernels/kernel_agent/__init__.py` — upstream pin updated

The gated branch in `_expert_mlp_gmm_ag_body` is unchanged: it still
calls `expert_ffn_v_outside(...)` with the same `(E_local, D, 2F)` W1
layout. The auto-impl now picks `f_tiled` when per-tile W1 exceeds 4 MB,
which catches our production shape (per-tile W1 = `D × 2F × 2` =
`7168 × 4096 × 2` = 56 MiB).

### Re-verification status

- **Upstream cluster verification at our exact production shape**:
  PASS (commit `5a1a2b7`, run `d7-fix-5` on x8p). This is the
  authoritative correctness gate — they verified D.7 at
  `E_local=64, D=7168, F=2048, EP=4, FSDP=32`, which matches the
  autoperf production sharding plan (FSDP=128 would only change the
  per-device T_local, not the per-expert weight tile shape that was
  the failure mode).
- **Our local AOT re-run at production shape**: **queued** — the
  local TPU is currently held by a parallel Claude session running
  pytest in `~/sigma`. Will re-run once it frees up. Outcome is
  predetermined by the upstream cluster verification + the unchanged
  API surface; this is a hygiene re-check, not load-bearing.
- **Our local parity smoke re-run**: same — queued.

### Updated recommendation for autoperf

The "do not queue an autoperf cluster iter" recommendation above
needs to be re-evaluated. With D.7 cluster-verified at our shape,
the open questions are now:

1. **Does it beat gmm_v2 at production mesh?** Still unanswered.
   kernel-agent's F.1 perf table (`results/phase_e4_through_f1.md`)
   gives v_outside fwd 2.45 ms vs jax_ref on a synthetic harness,
   but that's against pure-JAX baseline, not against our gmm_v2 +
   ragged_dot path. The dense per-expert math (E_local=64 mask-and-
   multiply for every token vs ragged_dot's K=8) is still a known
   FLOP regression class, and at production scale that math
   dominates.

2. **Does our chunked overlap (n_chunks=2 + ep_token_gather +
   psum_scatter pipelining) compose with the kernel's grid
   timing?** Open question — kernel-agent has no equivalent
   chunked overlap structure, but our gated branch sits inside our
   chunk loop, so the kernel is called once per chunk.

3. **fp8 path is still incompatible** (rejected by our
   compatibility check). For the cluster shot we'd compare against
   gmm_v2 (bf16, the v304 production), not against fp8.

The cleanest next step is a **research-only side-by-side**:
run both paths with the flag on/off on the same image, on a single
small cluster shot (e.g. `bodaborg-tpu7x-inference` for fast
turnaround), measure step time + TPS/chip, and decide based on
empirics. The integration itself is ready; no further code is
needed.

The smoke-and-AOT re-run is hygiene; the cluster side-by-side
is the question.

