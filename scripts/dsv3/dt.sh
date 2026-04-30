#!/bin/bash
# Tail logs of the first pod whose name contains $1 on bodaborg-super-rbq.
# Searches the PoC namespaces first (fast), then default, then -A as a fallback.
# Usage: ./dt.sh <pod-name-substring> [container]
set -euo pipefail
CTX=gke_cloud-tpu-multipod-dev_us-central1_bodaborg-super-rbq

if [ $# -lt 1 ]; then
  echo "Usage: $0 <pod-name-substring> [container]" >&2
  exit 2
fi

PAT="$1"
CONTAINER="${2:-}"   # if unset, let kubectl pick (works for single-container pods)

find_pod() {
  local ns
  for ns in poc-ml-perf poc-dev poc-gsc poc-nightly poc-scale-test default; do
    local pod
    pod=$(kubectl --context "$CTX" -n "$ns" get pods --no-headers 2>/dev/null \
            | grep -m1 "$PAT" | awk '{print $1}')
    if [ -n "$pod" ]; then
      echo "$ns $pod"
      return 0
    fi
  done
  # Fallback: scan everything (~20s on this cluster)
  kubectl --context "$CTX" get pods -A --no-headers 2>/dev/null \
      | grep -m1 "$PAT" | awk '{print $1, $2}'
}

read -r NS POD < <(find_pod || true)
if [ -z "${POD:-}" ]; then
  echo "No pod found matching '$PAT' on $CTX" >&2
  exit 1
fi

if [ -n "$CONTAINER" ]; then
  echo "Tailing $NS/$POD ($CONTAINER) ..." >&2
  exec kubectl --context "$CTX" -n "$NS" logs -f "$POD" -c "$CONTAINER"
else
  echo "Tailing $NS/$POD (default container) ..." >&2
  exec kubectl --context "$CTX" -n "$NS" logs -f "$POD"
fi
