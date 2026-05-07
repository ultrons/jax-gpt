# HALT — autoperf iter-3 on dsv3_train_full (RESOLVED 2026-05-07 evening)

**Status**: ❌ Halt was premature. Bench actually completed in the Kueue queue at 07:56Z while I was off pivoting to the broken `tpu7x-inference-cluster`. Iter-3 closed successfully — see `iter_log.md` § "iter-3 RESUMED" for the recovery write-up.

**Original HALT body kept below for audit (the diagnostic context still useful for the next session if a similar cluster-availability situation arises).**

---

# HALT — autoperf iter-3 on dsv3_train_full (original 06:34Z, since RESOLVED)

**Date**: 2026-05-07
**Workload**: dsv3_train_full
**Last iter**: 3 (Tooling-class — BF16 microbench grid for perfsim#10)
**Reason**: `cluster_unhealthy` (inference cluster full at medium priority; refused to preempt another user)

## What ran (original write-up)

- Composed bench_runner extension + workload JSON + k8s yaml for the BF16 microbench grid agreed on perfsim#10's needs-info comment.
- Built and pushed image `gcr.io/tpu-vm-gke-testing/perfsim-bench:v25-bf16-microbench`.
- Submitted JobSet `v7x-bf16-microbench` to `gke_tpu-prod-env-automated_us-central1_bodaborg-tpu7x-inference` (1×1×1 topology, medium priority). It stayed `Pending` AT THE TIME OF THE HALT WRITE-UP — all 8 tpu7x-standard-1t and 6 tpu7x-standard-4t nodes were occupied by long-running medium/very-high pods (other users' uBench servers, ages 21h–6d).
- Opened PR https://github.com/ultrons/perfsim/pull/23 with the durable engineering artifact (spec + extension + yaml).

## What actually happened

The Kueue queue admitted the job ~50 minutes later when an `ayushsethi-*` medium pod released. The bench ran end-to-end and Completed at 07:56Z. I missed the success because I'd already pivoted to `tpu7x-inference-cluster` (which turned out to have a stale-reservation MIG bug, completely unrelated to the original halt). Recovered via `kubectl logs` of the Completed pod 12h later.

## Lessons for future sessions

1. **Halt declarations should re-poll the jobset state when a session resumes**, not assume the previous halt's pending state still holds. Kueue admits jobs asynchronously; the cluster occupancy that drove the halt can change in minutes.
2. **Standalone v7x VMs are not provisioned anywhere we have access to** (verified by scan across `tpu-vm-gke-testing`, `cloud-tpu-multipod-dev`, `tpu-prod-env-automated` × multiple zones). Cluster path is forced.
3. **`tpu7x-inference-cluster`** in `cloud-tpu-multipod-dev` has been broken for days (`vllm-tpu` pod Pending 4d19h) due to the MIG template referencing a stale reservation `cloudtpu-20251017124413-573252602`. Owner needs to recreate the node pool against the current valid reservation `cloudtpu-20260317203000-769538580`. Don't waste cycles on this cluster until that's fixed. Captured in v7x_KNOWLEDGE.md.
4. **`bodaborg-tpu7x-inference`'s docker image lacks `gsutil`/`gcloud storage`** — the wrapper script's GCS upload was a no-op. Workaround: marker-delimited JSON in `kubectl logs` is the actual durable channel; manual upload from local. Image fix is a follow-up perfsim agent should sequence.
5. **Cluster-availability check protocol** (added to v7x_KNOWLEDGE.md §10): pair `kubectl get nodes` with `kubectl get pods -A` filtered by accelerator before submitting, to distinguish "nodes exist" from "nodes have free TPU slots."
