#!/bin/bash
# Live status of one or more GKE node pools (default: bodaborg-super-rbq mn-test-*).
# Refreshes every 10 s. Shows pool STATUS + each pool's nodes (Ready/NotReady).
#
# Usage:
#   ./watch-pools.sh                          # default: mn-test on bodaborg-super-rbq
#   ./watch-pools.sh "ghostfish-rbq"          # any pool name substring
#   ./watch-pools.sh "mn-test" 5              # custom interval (seconds)
#   CLUSTER=other-cluster ./watch-pools.sh    # different cluster
set -euo pipefail

PATTERN="${1:-mn-test}"
INTERVAL="${2:-10}"
CLUSTER="${CLUSTER:-bodaborg-super-rbq}"
REGION="${REGION:-us-central1}"
PROJECT="${PROJECT:-cloud-tpu-multipod-dev}"
CTX="gke_${PROJECT}_${REGION}_${CLUSTER}"

CMD=$(cat <<EOF
echo "=== node pools matching '${PATTERN}' on ${CLUSTER} ==="
gcloud container node-pools list \\
  --cluster=${CLUSTER} --region=${REGION} --project=${PROJECT} \\
  --filter="name~${PATTERN}" \\
  --format="table(name,status,initialNodeCount)" 2>&1
echo
echo "=== nodes (label selector cloud.google.com/gke-nodepool ~ ${PATTERN}) ==="
POOLS=\$(gcloud container node-pools list --cluster=${CLUSTER} --region=${REGION} \\
  --project=${PROJECT} --filter="name~${PATTERN}" --format="value(name)" 2>/dev/null \\
  | tr '\\n' ',' | sed 's/,$//')
if [ -n "\$POOLS" ]; then
  kubectl --context=${CTX} get nodes -l "cloud.google.com/gke-nodepool in (\$POOLS)" 2>&1 | head -50
  echo
  echo "=== summary ==="
  for P in \$(echo "\$POOLS" | tr ',' ' '); do
    N=\$(kubectl --context=${CTX} get nodes -l cloud.google.com/gke-nodepool=\$P --no-headers 2>/dev/null | wc -l)
    R=\$(kubectl --context=${CTX} get nodes -l cloud.google.com/gke-nodepool=\$P --no-headers 2>/dev/null | awk '\$2=="Ready"' | wc -l)
    printf "  %-25s nodes=%s ready=%s\\n" "\$P" "\$N" "\$R"
  done
else
  echo "(no pools matched yet)"
fi
EOF
)

echo "Watching pools matching '${PATTERN}' on ${CLUSTER} (refresh every ${INTERVAL}s, Ctrl-C to stop)..."
exec watch -n "${INTERVAL}" -c "$CMD"
