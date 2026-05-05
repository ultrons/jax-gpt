#!/usr/bin/env bash
# iter.sh — one autoperf iteration.
#
# Assumes the orchestrator has ALREADY:
#  - applied the code change (Edit / Write)
#  - committed it ("autoperf-iter<N>: <change> on <workload>")
#
# This script:
#  1. budget_check
#  2. cde build (or use --skip-build for image-override workflows)
#  3. cde run --tag <run_id> with --profile=on, args from <workload>.yaml
#  4. cde-wait
#  5. cde profile path → pull
#  6. headroom_to_json → emit JSON to <out-json>
#
# Usage:
#   iter.sh \
#     --workload <yaml> \
#     --iteration N \
#     --run-id <run-tag> \
#     --out-json <path>
#
# Workload YAML schema (see workloads/qwen35_decode.yaml):
#   model: qwen3_coder_480b_a35b
#   hardware: v7x_4x8x8
#   regime: inference_decode
#   parallelism: {tp: 8, ep: 1, dp: 1}
#   batch: 64
#   ctx: 1536
#   prompt: 1024
#   weight_dtype: fp8
#   kv_dtype: fp8
#   cluster_context: gke_cloud-tpu-multipod-dev_us-central1_bodaborg-super-rbq
#   cde_overrides:
#     config: full
#     gbs: 4096
#     ...
#
# Exit codes:
#   0 → success, JSON at --out-json is valid
#   1 → cluster job failed
#   2 → cluster job evicted (orchestrator should retry)
#   3 → budget exhausted
#   4 → setup error (bad workload yaml, missing tools)

set -euo pipefail

WORKLOAD=""; ITERATION=""; RUN_ID=""; OUT_JSON=""; SKIP_BUILD=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --workload) WORKLOAD="$2"; shift 2;;
        --iteration) ITERATION="$2"; shift 2;;
        --run-id) RUN_ID="$2"; shift 2;;
        --out-json) OUT_JSON="$2"; shift 2;;
        --skip-build) SKIP_BUILD=true; shift;;
        -h|--help) grep "^#" "$0" | sed 's|^# \?||'; exit 0;;
        *) echo "ERR: unknown arg $1" >&2; exit 4;;
    esac
done

for v in WORKLOAD ITERATION RUN_ID OUT_JSON; do
    [[ -z "${!v}" ]] && { echo "ERR: --${v,,} required" >&2; exit 4; }
done

[[ -f "$WORKLOAD" ]] || { echo "ERR: workload yaml not found: $WORKLOAD" >&2; exit 4; }

REPO=/home/sivaibhav_google_com/jax-gpt
BIN=$REPO/autoperf/bin
WORKLOAD_NAME=$(basename "$WORKLOAD" .yaml)

# 1. Budget check
"$BIN/budget_check.py" --workload "$WORKLOAD_NAME" || exit 3

# 2. Read workload YAML — yq if available, else python
read_yaml() {
    python3 -c "
import sys, yaml
y = yaml.safe_load(open('$WORKLOAD'))
def walk(d, k):
    for kk in k.split('.'):
        d = d.get(kk, '')
    return d
print(walk(y, '$1'))
"
}

CTX=$(read_yaml cluster_context)
[[ -z "$CTX" ]] && { echo "ERR: cluster_context missing in $WORKLOAD" >&2; exit 4; }

# 3. Build (unless skipped)
if [[ "$SKIP_BUILD" == "false" ]]; then
    cd "$REPO"
    cde build 2>&1 | tail -3
fi

# 4. Construct cde run command from cde_overrides
SET_ARGS=$(python3 -c "
import yaml
y = yaml.safe_load(open('$WORKLOAD'))
ovr = y.get('cde_overrides', {}) or {}
print(' '.join(f'--set {k}={v}' for k, v in ovr.items()))
")

cd "$REPO"
echo "→ cde run --tag $RUN_ID --context $CTX $SET_ARGS --profile"
cde run --tag "$RUN_ID" --context "$CTX" --profile $SET_ARGS \
    --note "autoperf iter${ITERATION} on ${WORKLOAD_NAME}" 2>&1 | tail -3

# 5. Wait for completion
echo "→ waiting on $RUN_ID..."
"$BIN/cde-wait.sh" "$RUN_ID" --timeout 3600 || {
    rc=$?
    echo "ERR: cde-wait exited $rc" >&2
    exit "$rc"
}

# 6. Pull profile path (cde profile path returns the gs:// URI)
PROFILE_URI=$(cde profile path "$RUN_ID" 2>&1 | tail -1)
[[ -z "$PROFILE_URI" ]] && { echo "ERR: no profile_uri for $RUN_ID" >&2; exit 1; }

# Localize the profile via gsutil
LOCAL_PROFILE_DIR=$REPO/autoperf/profiles/${RUN_ID}
mkdir -p "$LOCAL_PROFILE_DIR"
echo "→ gsutil -m cp -r ${PROFILE_URI}/* ${LOCAL_PROFILE_DIR}/"
gsutil -m cp -r "${PROFILE_URI}/*" "${LOCAL_PROFILE_DIR}/" 2>&1 | tail -5

# Find the xplane.pb directory under the pulled profile.
XPLANE_DIR=$(find "$LOCAL_PROFILE_DIR" -type d -name "profile" | head -1)
[[ -z "$XPLANE_DIR" ]] && XPLANE_DIR="$LOCAL_PROFILE_DIR"

# 7. headroom_to_json — pass workload settings as forward args
python3 -c "
import yaml
y = yaml.safe_load(open('$WORKLOAD'))
par = y.get('parallelism', {}) or {}
fields = [
    f'--model={y[\"model\"]}',
    f'--hardware={y[\"hardware\"]}',
    f'--tp={par.get(\"tp\", 1)}',
    f'--ep={par.get(\"ep\", 1)}',
    f'--dp={par.get(\"dp\", 1)}',
    f'--batch={y.get(\"batch\", 1)}',
    f'--ctx={y.get(\"ctx\", 1024)}',
    f'--prompt={y.get(\"prompt\", 1024)}',
    f'--weight-dtype={y.get(\"weight_dtype\", \"bf16\")}',
    f'--kv-dtype={y.get(\"kv_dtype\", \"bf16\")}',
    f'--xplane=$XPLANE_DIR',
]
print(' '.join(fields))
" > /tmp/_iter_forward_args
FORWARD=$(cat /tmp/_iter_forward_args)

GIT_SHA=$(git -C "$REPO" rev-parse --short HEAD)
"$BIN/headroom_to_json.py" \
    --workload "$WORKLOAD_NAME" \
    --iteration "$ITERATION" \
    --git-commit "$GIT_SHA" \
    --output "$OUT_JSON" \
    --keep-md "${OUT_JSON%.json}.md" \
    -- $FORWARD

echo "OK: iter $ITERATION → $OUT_JSON"
