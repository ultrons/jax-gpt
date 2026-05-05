# v7x_KNOWLEDGE.md — what we know about v7x + JAX/XLA

The autoperf agent's anti-hallucination ledger. **Read this at session start.**
**Append to this when you learn something new.** Do not delete entries; mark
them stale or contradicted with a follow-up entry.

This file complements (does not duplicate):
- **perfsim** — encodes the *numerical* TPU model (HBM BW, MXU shape, kernel-only ceilings)
- **`~/.claude/CLAUDE.md`** (global) — JAX/TPU/Pallas rules that apply across all projects
- **`~/jax-gpt/CLAUDE.md`** (repo) — model file paths, build commands

This file holds the *operational, narrative* knowledge that lets a fresh agent
make sensible experiments without re-discovering known-broken paths.

---

## 1. Hardware

**v7x physical:**
- 2 cores per chip. (Common footgun: GKE machine names like `tpu7x-standard-4t`
  count CHIPS = 4. JAX/vLLM see CORES = 8. A single 4t host is TP=8 single-host.)
- MXU: **256×256** per core. (NOT 128 — corrected 2026-05-04. perfsim and any
  tile-tuning code must respect this.)
- Per-core HBM: **101.733 GB** runtime, **94.75 GB** compile-time conservative limit
- Per-core HBM BW: **3.65 TB/s** (= shared-per-chip 7.3 TB/s ÷ 2 cores)
- ICI: w-axis is **7-13× faster** for D2D than other axes — use it for bandwidth-heavy
  comm dimensions (typically EP for MoE)

**Production cluster topology**: `4×8×8 = 256 chips = 512 devices`.
- Mesh assignment used by v304: `X=4 Y=8 Z=8 C=2` → `EP=X(4), FSDP=Y·Z·C(128), TP=—`
- 512 devices total (256 chips × 2 cores)

**Cluster context**: `gke_cloud-tpu-multipod-dev_us-central1_bodaborg-super-rbq`
- Namespace: `poc-dev`
- Priority class: `poc-dev-priority`
- Other contexts (don't use unless explicitly required):
  - `gke_cloud-tpu-multipod-dev_us-central1_ninja-v7x-64-spot` — older spot cluster
  - `gke_cloud-tpu-multipod-dev_us-central1_tpu7x-inference-cluster` — vLLM inference (no JobSet CRD)
  - `gke_cloud-tpu-multipod-dev_europe-west4_mlperf-rl-v5p-128` — v5p, eval only

---

## 2. Software stack pin (as of 2026-05-05)

```
jax     0.10.0
jaxlib  0.10.0
libtpu  0.0.41.dev20260417+nightly   ← reverted from dev20260504 due to Bug 1
```

**Why pinned**: `dev20260504` introduced a step-1 NaN at the production baseline
(`full + grad_accum=1`). See Bug 1 below. Don't bump libtpu without:
1. running `full + grad_accum=1` on a known-good config and verifying step-1 finite
2. updating this section + Bug-table date

**Image (known-good baseline)**: `gcr.io/tpu-vm-gke-testing/jax-gpt-dsv3:cde-9ea30df`
- builds from commit `b178a67` (libtpu reverted)
- v304-postrefactor reproducible: full + ga=1 + n_chunks=2 + gbs=4096 → step-1 loss 415.491

---

## 3. Known active bugs (workaround in place)

| ID | One-line | Workaround | Filed | Discovered |
|---|---|---|---|---|
| Bug 1 | NEW libtpu (dev20260504) NaN at production baseline | Pinned libtpu to dev20260417 | (none — fixed in-tree) | 2026-05-05 |
| Bug 2 | scan grad_accum > 1 + remat + vjp produces NaN cotangents | Use `--grad_accum=1` | task #43 | 2026-05-05 |
| Bug 3 | n_chunks=4 MoE body produces NaN cotangents in bwd (any libtpu) | Use `--moe_n_chunks=2` | task #44 | 2026-05-05 |
| Bug 4 | PDBS=1 (gbs=512 at fsdp=128 ep=4) produces NaN at mini config | Use `--gbs ≥ 1024` (PDBS ≥ 2) | task #45 | 2026-05-05 |

**Implication for autoperf**: the levers `moe_n_chunks` and `grad_accum` are
**off-limits** (locked at 2 and 1 respectively). Don't propose changes to these
until the corresponding bug is closed. The workload yamls bake these in.

---

## 4. JAX/XLA operational pitfalls (learned the hard way)

### `jax.debug.print` perturbs XLA scheduling
- Even with `ordered=False`, the side-effect token can flip XLA's scheduler
  and expose latent NaN-producing patterns
- Observed 2026-05-04 in `_expert_mlp_gmm_ag_body` (added prints → fwd NaN at n_chunks=4)
- Observed 2026-05-05 in `_vocab_ce_bwd` (always-on prints → step-1 NaN at full+ga=1)
- **Rule**: gate every `jax.debug.print` behind an env var. Default OFF in checked-in code.
- Pattern:
  ```python
  if os.environ.get("MY_FINITE_CHECK", "").lower() in ("1", "true", "yes"):
      jax.debug.print(...)
  ```

### scan + value_and_grad inside body + remat = NaN
- `jax.lax.scan(_accum_body, ...)` where `_accum_body` calls
  `jax.value_and_grad(compute_loss with checkpointed layers)` produces NaN
  cotangents at random init
- Restructuring (vjp outside scan, MaxText-style params-in-carry) doesn't help
- Python for-loop avoids it (Bug 2 mini repro confirmed)
- **Conclusion**: the toxic combo is `scan + checkpoint + outer-vjp` at any
  vjp nesting. Workaround: use grad_accum=1.

### argsort / fancy indexing in vjp path
- `arr[idx]` for integer idx → bwd via `scatter_p`. On TPU, scatter_p has
  known correctness/performance issues at certain shapes.
- MaxText hand-writes a custom_vjp on `_sort_activations_custom` (uses
  `argsort(idx)` to invert) specifically to avoid this.
- Our routing (`moe_routing` in dsv3) uses `sort_key_val` (sort op, not gather)
  to dodge the SC packed-element-gather error at TP=1. Different bug, same
  family of TPU-fancy-indexing landmines.

### `dataclasses.replace()` doesn't free HBM
- The old object stays live until ref count hits 0 AND JAX deallocates.
- To actually free a JAX array: `del x; gc.collect(); jax.effects_barrier()`
- Critical during staged setup (weight loading, cache transitions). See
  `~/.claude/CLAUDE.md` global JAX HBM section.

### `donate_argnums` on benchmark JIT breaks multi-run loops
- The donated buffer is consumed; subsequent calls receive a deleted buffer.
- For benchmarks: separate non-donating warmup + donating hot-loop.

### `RuntimeProgramAllocationFailure`
- Program binary (~15 GB for DSv3) needs CONTIGUOUS HBM at runtime
- Triage: `jax.live_arrays()` to find what's still live; `del` + `gc.collect()`
  before compilation step
- HLO temps scale with `page_B = batch_size / micro_batches` (paged-decode case)

---

## 5. MoE / DSv3-specific operational knowledge

### Sharding that works (v304 production baseline)
- `dp=1, fsdp=128, ep=4, tp=1` on 4×8×8 v7x
- `--config=full --gbs=4096 --grad_accum=1 --moe_n_chunks=2 --gradient_checkpoint --no_cp`
- `--optimizer=sgd --grad_clip=1.0` (AdamW now also gets grad_clip via `_maybe_clip_grads`)
- Throughput: **1770 TPS/chip @ 28.6% MFU** (v304-postrefactor)

### Aux loss
- `cfg.moe_aux_loss_coeff = 1e-4` by default
- Initial aux at random init: `coeff × E (=256) × sum(f·P) ≈ 7` per MoE layer
- Full config (58 MoE layers): initial aux ≈ 403, total initial loss ≈ 415
- Mini config (1 MoE layer): initial aux ≈ 7, total ≈ 19
- To disable: `--moe_aux_loss_coeff=0`. NOT `--aux_loss_weight=0` (that flag was
  dead and was removed in commit eb30b24)
- IMPORTANT: with coeff=0, aux=NaN at step 1+ is a SYMPTOM (`0 * NaN = NaN`),
  not a cause — the underlying NaN is in scores

### MaxText's aux-loss formula differs slightly
- Ours: `coeff × E × sum_e (avg_b f × avg_b P)` — average over batch first
- MaxText: `coeff × E² × mean_b mean_e (f_b × P_b)` — per-batch
- NOT bit-equivalent (Jensen's inequality). Forward magnitudes are similar
  (both ~7 per layer for mini, ~400 for full); bwd gradients through the
  routing softmax differ.

### MoE backend choices
- `gmm_ag` (production) — AllGather tokens on EP + AllGather F-dim weights on FSDP
- `jax` — does NOT work for DSv3 mini config (D=7168, D_moe=2048 contracting-dim mismatch)
- `fused_ep_moe_v4`, `fused_ep_moe_v4_jax_fwd`, `pfwd_jbwd` — Pallas v4 backends; test before relying

### `n_chunks` constraint
- Default `cfg.moe_n_chunks = 2` (per `model.py`)
- `n_chunks=4` triggers Bug 3 (bwd NaN). Don't propose without resolving the bug.
- `n_chunks=8` untested as of 2026-05-05.

---

## 6. cde / kubectl operational knowledge

### Image tags drift
- `cde build` computes the image tag from a hash of the build context
- Untracked files (PDFs, PNGs) in working tree can poison the hash → cde tag drift
- Fixed via `.dockerignore` + `.gitignore` patterns for `*.pdf *.png *.pptx`
  (commit 7dbd659)

### `cde profile` paths
- `cde profile path <run_id>` returns `gs://...` URI of pulled profile
- Localize: `gsutil -m cp -r <uri>/* <local-dir>/`
- xplane.pb dirs typically nested under `plugins/profile/`

### `cde history --status running` for parallel-job count
- Use this to enforce max-parallel-jobs constraint
- Example: `[ $(cde history --status running 2>/dev/null | wc -l) -ge 2 ]`

### Eviction vs failure (cde-side signals)
- `Finished=True (Succeeded)` → ok
- `Finished=True (Failed)` + `Requeued=True | NodeFailures` → eviction (retry)
- `Finished=True (Failed)` without those → real failure (read logs, halt)

### Image pull races
- Pod pull failures often racing with cde tag updates
- Workaround: rebuild explicitly (`cde build`), then resubmit with new tag

---

## 7. Filing tool issues (vs working around)

When a tool failure feels like a missing feature or a bug rather than a config
mistake, file an issue against the upstream repo. Repos and labels:

| tool | repo | issue label |
|---|---|---|
| perfsim | `ultrons/perfsim` | `autoperf-blocking` |
| cde | `ultrons/cde` | `autoperf-blocking` |
| xla-shell | `ultrons/xla-shell` | `autoperf-blocking` |

Always `gh issue list --repo <r> --search "<keyword>"` first to dedupe. Always
include a copy-pasteable repro and "definition of done" criterion.

(See AGENT.md §5 for the exact issue body template.)

---

## 8. How to use this file (the meta-rule)

- **Read at session start** — every autoperf session, first thing.
- **Append when you learn something** — a new pitfall, a fix that landed, a
  perfsim preset that's now reliable. Add a row or section.
- **Don't delete entries** — mark them stale with a date if superseded.
- **Cross-link, don't duplicate** — for things in `~/.claude/CLAUDE.md` global
  (Mosaic constraints, vLLM weight-loading via runai-streamer), reference rather
  than copy. This file is for *operational knowledge an autoperf agent needs to
  not repeat known-bad experiments*.

When a tool agent (cde/perfsim/xla-shell) closes one of your blocking issues,
update the corresponding Bug row above (status `active` → `fixed in <repo>#<pr>`)
and the stack-pin section if the fix changes a version requirement.
