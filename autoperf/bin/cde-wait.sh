#!/usr/bin/env bash
# cde-wait.sh — block until a cde run reaches a terminal state.
#
# Usage:
#   cde-wait.sh <run_id> [--timeout SEC] [--poll SEC]
#
# Exit codes:
#   0  → run completed successfully (status==ok in cde history)
#   1  → run failed (status==failed)
#   2  → run was evicted (status==evicted; harness should retry)
#   3  → timeout exceeded
#   4  → invalid args / cde error
#
# Polls `cde status <run_id> --json` and inspects:
#   - workload Finished + Succeeded → exit 0
#   - workload Finished + Failed (ReachedMaxRestarts, etc.) → exit 1
#   - workload evicted (Requeued + NodeFailures, or pod deletion cascade) → exit 2
#   - else → wait poll seconds and try again
#
# Designed to be called from iter.sh after `cde run`.

set -euo pipefail

RUN_ID=""
TIMEOUT_SEC=3600   # 1 hr default; full-config compile + 5 steps is ~10 min, so 1 hr is generous
POLL_SEC=15

while [[ $# -gt 0 ]]; do
    case "$1" in
        --timeout) TIMEOUT_SEC="$2"; shift 2;;
        --poll) POLL_SEC="$2"; shift 2;;
        -h|--help)
            grep "^#" "$0" | sed 's|^# \?||'
            exit 0;;
        *)
            if [[ -z "$RUN_ID" ]]; then
                RUN_ID="$1"; shift
            else
                echo "ERR: unexpected arg '$1'" >&2
                exit 4
            fi;;
    esac
done

if [[ -z "$RUN_ID" ]]; then
    echo "ERR: run_id is required" >&2
    exit 4
fi

start_ts=$(date +%s)

while true; do
    elapsed=$(( $(date +%s) - start_ts ))
    if (( elapsed >= TIMEOUT_SEC )); then
        echo "TIMEOUT after ${elapsed}s waiting on $RUN_ID" >&2
        exit 3
    fi

    # cde status returns text; --json gives machine output but field set is limited.
    # We grep the textual output for the key signals — fewer surprises than JSON
    # parsing if cde adds fields.
    status_out=$(cde status "$RUN_ID" 2>&1 || true)

    if echo "$status_out" | grep -q "Finished=True (Succeeded)"; then
        echo "OK: $RUN_ID succeeded after ${elapsed}s"
        exit 0
    fi

    if echo "$status_out" | grep -q "Finished=True (Failed)"; then
        # Distinguish eviction (retryable) from real failure (terminal)
        if echo "$status_out" | grep -qE "Requeued=True|NodeFailures|trying and failing to pull image"; then
            echo "EVICTED: $RUN_ID (node failure or image pull race) after ${elapsed}s" >&2
            exit 2
        fi
        echo "FAILED: $RUN_ID after ${elapsed}s" >&2
        # Last-30-lines tail for diagnosis
        cde logs "$RUN_ID" --no-follow 2>&1 | tail -30 >&2 || true
        exit 1
    fi

    sleep "$POLL_SEC"
done
