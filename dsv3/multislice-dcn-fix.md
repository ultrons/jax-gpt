# Multislice DCN fix on `bodaborg-super-rbq` — what changed

## Problem

Any multislice (`--num-slices >= 2`) workload on `bodaborg-super-rbq` would hang
during the first cross-slice byte transfer. MaxText runs hung for ~5 min then
SIGABRTed via the megascale runtime; pure-JAX tests hung indefinitely at
`jax.device_put` of a `dcn`-sharded array. Single-slice runs and bare
`jax.distributed.initialize() + jax.device_count()` (no actual cross-slice data
movement) were unaffected.

## Root cause

The cluster was provisioned outside `xpk cluster create`. As a result the TPU
node pools had **single-NIC** networking (only the cluster's primary VPC,
default MTU 1460), whereas xpk's standard recipe attaches **four additional
VPCs at jumbo MTU (8244)** to each TPU node. Cross-slice DCN traffic worked
for small control messages over the single NIC but stalled on bulk transfers.

## Fix

Two phases.

### Phase 1 — additive infrastructure (zero risk to existing pools)

Created four extra VPCs, each with its own subnet and internal firewall rule:

```bash
PROJECT=cloud-tpu-multipod-dev
CLUSTER=bodaborg-super-rbq
REGION=us-central1

for i in 1 2 3 4; do
  gcloud compute networks create ${CLUSTER}-net-${i} \
    --project=${PROJECT} --subnet-mode=custom --mtu=8244

  gcloud compute networks subnets create ${CLUSTER}-${REGION}-sub-${i} \
    --network=${CLUSTER}-net-${i} --region=${REGION} \
    --range=192.168.${i}.0/24 --project=${PROJECT}

  gcloud compute firewall-rules create ${CLUSTER}-internal-${i} \
    --network=${CLUSTER}-net-${i} --action=ALLOW \
    --rules=tcp:0-65535,udp:0-65535,icmp \
    --source-ranges=192.168.0.0/16 --project=${PROJECT}
done
```

Result: four jumbo-MTU VPCs ready for additional NICs.

### Phase 2 — recreate TPU node pools with multi-NIC

GKE doesn't allow modifying `additionalNodeNetworks` on an existing node pool,
so each TPU pool must be deleted and recreated. The reservation sub-block the
old pool was bound to is freed when the pool is deleted and reclaimed by the
new pool, with a brief gap.

For each TPU pool to upgrade:

```bash
POOL=<ghostfish-name>
SUB=<the pool's reservation sub-block, e.g. subblock-0042>
RES=ghostfish-rbq1c2ca7whsw

# 1. delete the existing pool (frees the reservation slot)
gcloud container node-pools delete ${POOL} \
  --cluster=${CLUSTER} --region=${REGION} --project=${PROJECT} --quiet

# 2. recreate with the same sub-block and multi-NIC attached
gcloud container node-pools create ${POOL} \
  --cluster=${CLUSTER} --region=${REGION} --project=${PROJECT} \
  --machine-type=tpu7x-standard-4t \
  --num-nodes=16 \
  --node-locations=us-central1-ai1a \
  --tpu-topology=4x4x4 \
  --placement-policy=ss-policy \
  --reservation-affinity=specific \
  --reservation=${RES}/reservationBlocks/${RES}-block-0001/reservationSubBlocks/${RES}-block-0001-subblock-${SUB} \
  --no-enable-autoupgrade --no-enable-autorepair \
  --additional-node-network=network=${CLUSTER}-net-1,subnetwork=${CLUSTER}-${REGION}-sub-1 \
  --additional-node-network=network=${CLUSTER}-net-2,subnetwork=${CLUSTER}-${REGION}-sub-2 \
  --additional-node-network=network=${CLUSTER}-net-3,subnetwork=${CLUSTER}-${REGION}-sub-3 \
  --additional-node-network=network=${CLUSTER}-net-4,subnetwork=${CLUSTER}-${REGION}-sub-4
```

After recreation each TPU node has **5 NICs**:

| NIC  | Network                   | MTU  | Purpose                          |
|------|---------------------------|------|----------------------------------|
| nic0 | `bodaborg-super-rbq-network`   | 1460 | primary, kubelet/control-plane   |
| nic1 | `bodaborg-super-rbq-net-1`     | 8244 | DCN (cross-slice TPU traffic)    |
| nic2 | `bodaborg-super-rbq-net-2`     | 8244 | DCN                              |
| nic3 | `bodaborg-super-rbq-net-3`     | 8244 | DCN                              |
| nic4 | `bodaborg-super-rbq-net-4`     | 8244 | DCN                              |

`--no-enable-autoupgrade --no-enable-autorepair` is required because the
cluster has `enable_kubernetes_alpha=true`.

## Verification

Pure-JAX cross-slice all-reduce on two recreated pools (`mn-test-0064`,
`mn-test-0066`):

- 32/32 ranks completed `jax.device_put` with cross-slice sharding (was
  previously hanging here).
- 32/32 ranks dispatched and completed a `psum` over the `dcn` mesh axis.
- All ranks printed `DONE in 0.09s local_sample=2.0 expected=2.0`.
- JobSet TerminalState `Completed`.

## Rollback

If a recreated pool needs to revert to single-NIC: delete and recreate without
the four `--additional-node-network` flags. The four jumbo VPCs can be left in
place (cost is just IPv4 subnet allocation) or removed:

```bash
for i in 1 2 3 4; do
  gcloud compute firewall-rules delete bodaborg-super-rbq-internal-${i} --quiet
  gcloud compute networks subnets delete bodaborg-super-rbq-${REGION}-sub-${i} \
    --region=${REGION} --quiet
  gcloud compute networks delete bodaborg-super-rbq-net-${i} --quiet
done
```
