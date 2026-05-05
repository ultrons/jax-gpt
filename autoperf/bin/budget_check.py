#!/usr/bin/env python3
"""budget_check.py — hard cap on cluster spend.

Enforces two limits per workload, per UTC day:
  - max-jobs           : default 12
  - max-compile-min    : default 90  (sum of compile time across iterations)

State is read from autoperf/state/<workload>_budget.json which the
orchestrator updates after each iteration. This script:

  - exit 0 → budget OK, can proceed
  - exit 1 → budget exhausted, do NOT submit new job
  - exit 2 → state file invalid

Usage:
  budget_check.py --workload <name> [--max-jobs N] [--max-compile-min M]
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

STATE_DIR = Path("/home/sivaibhav_google_com/jax-gpt/autoperf/state")
DEFAULT_MAX_JOBS = 12
DEFAULT_MAX_COMPILE_MIN = 90


def _today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def main() -> int:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--workload", required=True)
    parser.add_argument("--max-jobs", type=int, default=DEFAULT_MAX_JOBS)
    parser.add_argument("--max-compile-min", type=int, default=DEFAULT_MAX_COMPILE_MIN)
    args = parser.parse_args()

    STATE_DIR.mkdir(parents=True, exist_ok=True)
    state_path = STATE_DIR / f"{args.workload}_budget.json"

    if not state_path.exists():
        # Fresh — initialize and allow.
        state_path.write_text(json.dumps({_today(): {"jobs": 0, "compile_min": 0.0}}))
        print(f"OK budget fresh — 0/{args.max_jobs} jobs, 0/{args.max_compile_min}m compile")
        return 0

    try:
        state = json.loads(state_path.read_text())
    except json.JSONDecodeError as e:
        print(f"ERR state file invalid: {e}", file=sys.stderr)
        return 2

    today = _today()
    today_state = state.get(today, {"jobs": 0, "compile_min": 0.0})

    if today_state["jobs"] >= args.max_jobs:
        print(
            f"BUDGET-EXHAUSTED: {today_state['jobs']}/{args.max_jobs} jobs today "
            f"({today})",
            file=sys.stderr,
        )
        return 1
    if today_state["compile_min"] >= args.max_compile_min:
        print(
            f"BUDGET-EXHAUSTED: {today_state['compile_min']:.1f}/"
            f"{args.max_compile_min}m compile today ({today})",
            file=sys.stderr,
        )
        return 1

    print(
        f"OK budget {today_state['jobs']}/{args.max_jobs} jobs, "
        f"{today_state['compile_min']:.1f}/{args.max_compile_min}m compile today"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
