# Running Qwen3.5-397B HumanEval on GKE TPU 4×4×4 v5p

## Architecture

```
4×4×4 v5p pod = 64 chips = 8 hosts × 8 chips
Each Kubernetes pod  = 1 host (8 chips, google.com/tpu: 8)
Sharding:  TP=8 within host │ DP=8 across hosts
HBM total: 64 × 96 GB = 6,144 GB  ▸  model: ~794 GB BF16  (fits with headroom)
Model weights disk: oss-model-dev (1 TB, europe-west4-b, project tpu-vm-gke-testing)
```

---

## One-time setup

### 1. Install the JobSet CRD

```bash
kubectl apply --server-side -f \
  https://github.com/kubernetes-sigs/jobset/releases/download/v0.7.1/manifests.yaml
```

### 2. Create the TPU node pool

The node pool must be in the same zone as the disk (`europe-west4-b`) and use
compact placement (required for multi-host v5p slices):

```bash
gcloud container node-pools create v5p-4x4x4 \
  --cluster=sivaibhav-exp \
  --zone=europe-west4-b \
  --machine-type=ct5p-hightpu-8t \
  --num-nodes=8 \
  --placement-type=COMPACT \
  --tpu-topology=4x4x4
```

### 3. Build and push the Docker image

```bash
cd /home/sivaibhav_google_com/jax-gpt

docker build -f k8s/Dockerfile.tpu \
  -t gcr.io/tpu-vm-gke-testing/jax-gpt-tpu:latest .

docker push gcr.io/tpu-vm-gke-testing/jax-gpt-tpu:latest
```

---

## Disk handoff workflow

The `oss-model-dev` disk holds the 397B weights at `/mnt/disks/tpu_data`.
GCE Persistent Disks cannot be mounted READ_WRITE on one VM and READ_ONLY
on another simultaneously. Follow this sequence:

### Before submitting the GKE job

```bash
# On this VM: unmount and detach the data disk
sudo umount /mnt/disks/tpu_data

gcloud compute instances detach-disk t1v-n-23714477-w-0 \
  --disk=oss-model-dev \
  --zone=europe-west4-b
```

### Submit the job (see next section)

The PV/PVC in the YAML reference `oss-model-dev` as ReadOnlyMany.
GKE will attach it to all 8 TPU host VMs simultaneously (read-only is fine
for multiple nodes).

### After the job finishes

```bash
# Re-attach the disk to this VM for continued development
gcloud compute instances attach-disk t1v-n-23714477-w-0 \
  --disk=oss-model-dev \
  --zone=europe-west4-b \
  --mode=rw

# Re-mount
sudo mount /dev/nvme0n2 /mnt/disks/tpu_data
```

---

## Submitting the eval job

```bash
# Apply everything: PV, PVC, output PVC, JobSet
kubectl apply -f k8s/qwen35_eval_jobset.yaml
```

Monitor:

```bash
# JobSet status
kubectl get jobset qwen35-humaneval

# All pods
kubectl get pods -l jobset.sigs.k8s.io/jobset-name=qwen35-humaneval

# Logs from all pods (streaming)
kubectl logs -l jobset.sigs.k8s.io/jobset-name=qwen35-humaneval --prefix=true -f

# Logs from rank 0 only (has final summary)
kubectl logs -l \
  "jobset.sigs.k8s.io/jobset-name=qwen35-humaneval,batch.kubernetes.io/job-completion-index=0" -f
```

Collect results:

```bash
# Print summary from rank-0 pod
RANK0_POD=$(kubectl get pods \
  -l "jobset.sigs.k8s.io/jobset-name=qwen35-humaneval,batch.kubernetes.io/job-completion-index=0" \
  -o jsonpath='{.items[0].metadata.name}')

kubectl exec $RANK0_POD -- cat /mnt/output/summary.json
```

Clean up:

```bash
kubectl delete jobset qwen35-humaneval
kubectl delete pvc eval-output-pvc     # frees the output disk
# Keep oss-model-dev-pvc / oss-model-dev-pv for next run (Retain policy)
```

---

## Testing locally on this VM (4 chips, offload mode)

```bash
python scripts/test_correctness_offload.py \
  --model-dir /mnt/disks/tpu_data/qwen3.5-397b \
  --skip-hf \
  --n-devices 4
```

## Testing the eval script locally (mini config, no real weights)

```bash
python scripts/eval_humaneval.py \
  --model-dir /mnt/disks/tpu_data/qwen3.5-397b \
  --output-dir /tmp/humaneval_out \
  --n-problems 5 \
  --tp 4
```

---

## Expected performance on 4×4×4 pod

With full model in HBM (no offloading bottleneck):
| Stage    | Estimate                   |
|----------|----------------------------|
| Load     | ~5–10 min (disk → HBM)     |
| Prefill  | ~500 tok/s                 |
| Decode   | ~10–20 tok/s               |
| 164 problems × 256 new tokens | ~3–5 hours total |

---

## JobSet DNS note

JobSet creates a headless Service automatically named:
`<jobset-name>-<replicatedjob-name>`  →  `qwen35-humaneval-slice`

Pod hostnames follow the pattern:
`<jobset-name>-<replicatedjob-name>-<replica>-<pod-index>.<service>.<ns>.svc.cluster.local`

So the coordinator (rank 0) is:
`qwen35-humaneval-slice-0-0.qwen35-humaneval-slice.default.svc.cluster.local`

This is hardcoded in `JAX_COORDINATOR_ADDRESS` in the YAML.
Update the namespace portion if you deploy into a non-default namespace.
