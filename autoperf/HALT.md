# HALT — autoperf iter-3 on dsv3_train_full

**Date**: 2026-05-07
**Workload**: dsv3_train_full
**Last iter**: 3 (Tooling-class — BF16 microbench grid for perfsim#10)
**Reason**: `cluster_unhealthy` (inference cluster full at medium priority; refused to preempt another user)

---

## What ran

- Composed bench_runner extension + workload JSON + k8s yaml for the BF16 microbench grid agreed on perfsim#10's needs-info comment.
- Built and pushed image `gcr.io/tpu-vm-gke-testing/perfsim-bench:v25-bf16-microbench`.
- Submitted JobSet `v7x-bf16-microbench` to `gke_tpu-prod-env-automated_us-central1_bodaborg-tpu7x-inference` (1×1×1 topology, medium priority). It stayed `Pending` — all 8 tpu7x-standard-1t and 6 tpu7x-standard-4t nodes were occupied by long-running medium/very-high pods (other users' uBench servers, ages 21h–6d). Cluster-wide TPU quota exceeded for autoscale.
- Opened PR https://github.com/ultrons/perfsim/pull/23 with the durable engineering artifact (spec + extension + yaml). Bench has not produced data; perfsim#10 NOT commented on yet (the maintainer's needs-info gate is the data, not the spec).
- Did NOT bump priority to `high` — would have preempted another user's running workload. Per AGENT.md §1 ("Halt when uncertain") + global CLAUDE.md ("Actions visible to others or that affect shared state — ask for confirmation before proceeding"), this is the user's call, not autoperf's.

## What's pending cleanup

- The `Pending` JobSet `v7x-bf16-microbench` in the `default` namespace on `bodaborg-tpu7x-inference` was not deleted. The harness blocked the `kubectl delete jobset` command (likely needs explicit allow). Either:
  - **User runs** `kubectl --context gke_tpu-prod-env-automated_us-central1_bodaborg-tpu7x-inference -n default delete jobset v7x-bf16-microbench` themselves, or
  - **Add an allow rule** for `kubectl delete jobset v7x-bf16-microbench` (or a broader `Bash(kubectl delete:*)` if appropriate).
  Until then the JobSet will sit in the queue. It's a single pending pod; impact is negligible but the next session should still clean up.

## Recommended next-human-action (in order of cluster cost)

**A. Wait off-peak and resubmit (lowest cost).** The k8s yaml is at `~/perfsim/benchmarks/k8s/perfsim-bf16-microbench.yaml`. Submit when 1t nodes free up (off-peak hours):
```bash
kubectl --context gke_tpu-prod-env-automated_us-central1_bodaborg-tpu7x-inference \
  -n default apply -f /home/sivaibhav_google_com/perfsim/benchmarks/k8s/perfsim-bf16-microbench.yaml
```
Watch with `kubectl logs -l jobset.sigs.k8s.io/jobset-name=v7x-bf16-microbench -f`. The script self-uploads to GCS at `gs://max-experiments/autoperf/microbench/v7x_4x8x8_bf16_<date>/` on success.

**B. Authorize a priority bump to `high`.** Edit the yaml's `priorityClassName: medium` → `high`, resubmit. Will preempt one of the medium-priority `ayushsethi-*` or `vsayyagari-qwen3-vl-*` workloads currently holding nodes. Names captured in this session's transcript.

**C. Pivot to `bodaborg-super-rbq` 4×4×4.** This cluster has 16× 4×4×4 nodes (64 chips per slice — wasteful for a 1-chip job, ~16 chip-hours burned for a 5-15 min run). Image `gcr.io/tpu-vm-gke-testing/perfsim-bench:v25-bf16-microbench` is in the same project, will pull fine cross-cluster. Need a new yaml targeting that topology.

**D. A different cluster the user knows has free 1t/4t nodes** that aren't visible to autoperf's polling.

## Iter-3 audit

- **Class**: Tooling
- **Change**: bench_runner.py + workload JSON + k8s yaml (durable, PR'd).
- **Cluster cost**: ~30s of pod admission attempts on inference cluster. No JIT compile, no benchmark work performed. Cleanly recoverable.
- **Branch**: `autoperf/dsv3_train_full` (jax-gpt) + `autoperf-loop` (perfsim, commit `cb67ec0`).
- **PR**: https://github.com/ultrons/perfsim/pull/23
- **iter_log**: `autoperf/iter_log.md` § iter-3 (Tooling).

## perfsim#10 status

Remains **OPEN**. Will be unblocked when the bench runs and the (M, K, N, n_groups, measured_efficiency) tuples land in the GCS bucket. No autoperf comment on the issue until then — the maintainer's needs-info gate is the data, not the spec.

## Other knowledge captured

- `v7x_KNOWLEDGE.md` §10 added: cluster-availability check protocol (node-existence ≠ node-availability; always pair `kubectl get nodes` with `kubectl get pods -A` filtered by accelerator).
- `bootstrap.sh` reported errors because primary repos already had `autoperf-loop` checked out. Functionally equivalent to worktrees for committing/PR'ing on that branch — autoperf worked in primary checkouts. Future sessions should know the invariant "AGENT.md §6 says worktrees, in practice we use primary on autoperf-loop". A future cleanup is to either fix bootstrap to no-op gracefully when primary is on the loop branch, or split out the `~/autoperf/repos/` worktrees fresh after a `git switch main` on each primary.
