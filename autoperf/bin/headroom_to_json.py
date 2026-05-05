#!/usr/bin/env python3
"""headroom_to_json.py — wrap headroom_report and emit a top-N JSON.

The agent reads JSON better than markdown. This script:
  1. Invokes perfsim's headroom_report (writes its markdown to a tmp file).
  2. Parses the per-leaf table to extract (leaf, predicted_us, measured_us,
     ratio, headroom_total_ms).
  3. Emits a JSON payload with the top-N leaves by headroom_total_ms,
     plus a few summary fields (total measured time, total predicted, top
     leaf identity, etc.).

Output schema (printed to stdout, also written to --output if given):
{
  "iteration": int (echoed from --iteration),
  "workload": str (echoed from --workload),
  "git_commit": str (HEAD short SHA at run-time),
  "leaves": [
    {"leaf": str, "predicted_us": float, "measured_us": float,
     "ratio": float, "headroom_total_ms": float},
    ...
  ],
  "top_n": [<same schema, top-N entries>],
  "totals": {
    "predicted_us_per_fwd": float,
    "measured_us_per_fwd": float,
    "headroom_total_ms": float,
  },
  "halt_signals": {
    "top_headroom_below_floor": bool,   # < 0.5 ms total
    "all_leaves_at_ceiling": bool,      # no leaf > 5% with headroom > floor
  }
}

Usage:
    headroom_to_json.py --workload <name> --iteration <N> --git-commit <SHA> \\
        --output <file.json> -- <headroom_report args>

Everything after `--` is forwarded verbatim to headroom_report.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

HEADROOM_REPORT = (
    "/home/sivaibhav_google_com/ml-experiments-perfsim/perfsim/inference/scripts/"
    "headroom_report.py"
)

HEADROOM_FLOOR_MS = 0.5  # below this → top_headroom_below_floor = True
LEAF_SHARE_PCT_FLOOR = 5.0  # leaves below this share are ignored for ceiling check


def _parse_table(md_text: str) -> list[dict]:
    """Parse the per-leaf headroom markdown table.

    Table format (from headroom_report._format_report):
      | leaf | predicted us/fwd | measured us/fwd | meas/pred | headroom ms (total) |
      |---|---|---|---|---|
      | <name> | <num> | <num> | <ratio>x | <num> |
    """
    leaves = []
    table_started = False
    for line in md_text.splitlines():
        if "headroom ms (total)" in line.lower():
            table_started = True
            continue
        if not table_started:
            continue
        if line.strip().startswith("|---"):
            continue
        if not line.strip().startswith("|"):
            # Table ended — anything that's not a row marker stops parsing.
            if leaves:
                break
            continue
        parts = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(parts) != 5:
            continue
        leaf, pred_s, meas_s, ratio_s, hr_s = parts
        try:
            predicted_us = float(pred_s)
            measured_us = float(meas_s)
            ratio = float("inf") if ratio_s == "∞" else float(ratio_s.rstrip("x"))
            headroom_total_ms = float(hr_s)
        except ValueError:
            # Header row ("predicted us/fwd" etc.) or stray formatting.
            continue
        leaves.append({
            "leaf": leaf,
            "predicted_us": predicted_us,
            "measured_us": measured_us,
            "ratio": ratio,
            "headroom_total_ms": headroom_total_ms,
        })
    return leaves


def _git_short_sha(repo_dir: str = "/home/sivaibhav_google_com/jax-gpt") -> str:
    try:
        out = subprocess.check_output(
            ["git", "-C", repo_dir, "rev-parse", "--short", "HEAD"],
            text=True,
        ).strip()
        return out
    except subprocess.CalledProcessError:
        return "unknown"


def main() -> int:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--workload", required=True)
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--git-commit", default=None,
                        help="Override; default = current HEAD short SHA")
    parser.add_argument("--top-n", type=int, default=5,
                        help="How many top leaves to surface in top_n field")
    parser.add_argument("--output", default=None,
                        help="Also write JSON to this file (in addition to stdout)")
    parser.add_argument("--keep-md",
                        help="Path to keep the upstream markdown report at")
    parser.add_argument("forward_args", nargs=argparse.REMAINDER,
                        help="Args after `--` are forwarded to headroom_report")
    args = parser.parse_args()

    if args.forward_args and args.forward_args[0] == "--":
        forward_args = args.forward_args[1:]
    else:
        forward_args = args.forward_args

    if not forward_args:
        print("ERR: no headroom_report args after `--`", file=sys.stderr)
        return 1

    md_out = (Path(args.keep_md) if args.keep_md
              else Path(tempfile.mkstemp(suffix=".md")[1]))

    # Inject our --output if not already present in forward_args
    if "--output" not in forward_args:
        forward_args = forward_args + ["--output", str(md_out)]
    else:
        idx = forward_args.index("--output")
        md_out = Path(forward_args[idx + 1])

    cmd = ["python", HEADROOM_REPORT] + forward_args
    print(f"# invoking: {' '.join(cmd)}", file=sys.stderr)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        return proc.returncode

    md_text = md_out.read_text()
    leaves = _parse_table(md_text)
    leaves.sort(key=lambda r: -r["headroom_total_ms"])

    total_pred = sum(r["predicted_us"] for r in leaves)
    total_meas = sum(r["measured_us"] for r in leaves)
    total_hr = sum(r["headroom_total_ms"] for r in leaves)

    # Halt-signal checks
    top_headroom = leaves[0]["headroom_total_ms"] if leaves else 0.0
    top_below_floor = top_headroom < HEADROOM_FLOOR_MS

    # All leaves at ceiling: no leaf with both > LEAF_SHARE% of measured time
    # AND headroom > floor_ms remains.
    big_leaves_with_headroom = [
        r for r in leaves
        if (total_meas > 0 and (r["measured_us"] / total_meas * 100) > LEAF_SHARE_PCT_FLOOR
            and r["headroom_total_ms"] > HEADROOM_FLOOR_MS)
    ]
    all_at_ceiling = (len(big_leaves_with_headroom) == 0)

    payload = {
        "iteration": args.iteration,
        "workload": args.workload,
        "git_commit": args.git_commit or _git_short_sha(),
        "leaves": leaves,
        "top_n": leaves[:args.top_n],
        "totals": {
            "predicted_us_per_fwd": total_pred,
            "measured_us_per_fwd": total_meas,
            "headroom_total_ms": total_hr,
        },
        "halt_signals": {
            "top_headroom_below_floor": top_below_floor,
            "all_leaves_at_ceiling": all_at_ceiling,
        },
        "report_md_path": str(md_out),
    }

    json_text = json.dumps(payload, indent=2)
    print(json_text)
    if args.output:
        Path(args.output).write_text(json_text)

    return 0


if __name__ == "__main__":
    sys.exit(main())
