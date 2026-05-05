#!/usr/bin/env python3
"""autoperf.py — main loop orchestrator.

Drives N iterations of:
  1. Load workload + state.
  2. If iteration > 0, get latest headroom JSON for context.
  3. Spawn Claude with the iter prompt → returns either a CHANGE proposal or HALT.
  4. If HALT → record reason, exit.
  5. Else apply the change (Claude does Edit/Write/git-commit itself in step 3),
     then run autoperf/bin/iter.sh (build → cde run → wait → pull → headroom).
  6. Compare the new metric to the prior one → outcome (improved/regressed/neutral).
  7. record_iter; check stop conditions (3 regressions, halt, budget, top-leaf-floor).

Claude integration:
  - In dry-run mode, we don't spawn Claude; instead we invoke a stub that prints
    what it WOULD propose. This validates the orchestrator structure without
    needing a long Claude session.
  - In real mode, we shell out to `claude --print` with the iter prompt baked
    in, and parse the structured response.

Designed to be RESUMABLE: state is persisted after each step, so killing and
re-running picks up at the same iteration.

Usage:
  autoperf.py --workload <yaml-path> --iters N [--dry-run] [--from-iter N]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# Allow running as a script
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

import state as state_mod  # noqa: E402

REPO = Path("/home/sivaibhav_google_com/jax-gpt")
AUTOPERF = REPO / "autoperf"
BIN = AUTOPERF / "bin"
DIARY_DIR = AUTOPERF / "diary"
REPORTS_DIR = AUTOPERF / "reports"

ITER_PROMPT_TEMPLATE = AUTOPERF / "agent" / "iter_prompt.md"

REGRESSION_LIMIT = 3
HEADROOM_FLOOR_MS = 0.5


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git_short_sha() -> str:
    return subprocess.check_output(
        ["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], text=True
    ).strip()


def _classify_outcome(metric_before: Optional[float],
                       metric_after: Optional[float]) -> str:
    if metric_before is None or metric_after is None:
        return "neutral"
    if metric_after < metric_before * 0.99:  # > 1% improvement on time-to-X
        return "improved"
    if metric_after > metric_before * 1.01:  # > 1% regression
        return "regressed"
    return "neutral"


def _spawn_claude(iter_prompt: str) -> dict:
    """Invoke Claude with the iter prompt, parse structured response.

    Expected response format (Claude must emit this as the LAST block):
      ```json
      {"action": "CHANGE", "summary": "...", "git_commit_msg": "..."}
      ```
    or
      ```json
      {"action": "HALT", "reason": "..."}
      ```

    Claude is responsible for actually making the edit + committing BEFORE
    emitting the JSON. We just parse the verdict.
    """
    # Real spawn would shell to `claude --print` here. For Day 3 we keep it
    # stubbed so the orchestrator can dry-run end-to-end.
    proc = subprocess.run(
        ["claude", "--print", "--output-format", "json"],
        input=iter_prompt,
        capture_output=True,
        text=True,
        timeout=1800,  # 30 min for Claude to think + edit + commit
    )
    if proc.returncode != 0:
        raise RuntimeError(f"Claude exited {proc.returncode}: {proc.stderr[:500]}")

    # Claude --print --output-format=json wraps the assistant text in a JSON
    # envelope. Extract the assistant message and find our action JSON inside.
    try:
        env = json.loads(proc.stdout)
        text = env.get("result", env.get("message", ""))
        if isinstance(text, dict):
            text = text.get("content", "")
    except json.JSONDecodeError:
        text = proc.stdout

    # Find a fenced JSON block at the end.
    import re
    blocks = re.findall(r"```json\s*\n(.+?)\n```", text, re.DOTALL)
    if not blocks:
        raise RuntimeError(
            f"Claude response had no ```json block. Tail:\n{text[-1000:]}"
        )
    return json.loads(blocks[-1])


def _build_iter_prompt(workload_path: Path, state: dict,
                       prev_headroom_json: Optional[dict]) -> str:
    """Render the iteration prompt for Claude.

    Pulls last 5 diary entries + last headroom JSON + workload spec +
    list of exhausted levers. Claude has full repo access via its tools;
    the prompt just tells it what context exists and what to produce.
    """
    template = ITER_PROMPT_TEMPLATE.read_text()

    diary_tail = state["diary"][-5:] if state["diary"] else []
    diary_md = "\n".join(
        f"- iter {e['iter']} ({e['git_sha']}): {e['change']} → "
        f"{e['outcome']} (top_leaf={e.get('top_leaf','?')}, "
        f"headroom={e.get('top_headroom_ms', 0):.1f}ms)"
        for e in diary_tail
    ) or "_(no prior iterations)_"

    exhausted_md = ", ".join(state["exhausted_levers"]) or "_(none)_"

    headroom_md = (
        json.dumps(prev_headroom_json["top_n"], indent=2)
        if prev_headroom_json else "_(no prior headroom report)_"
    )

    return template.format(
        workload_path=str(workload_path),
        workload_name=workload_path.stem,
        iteration=state["iteration"] + 1,
        git_sha=_git_short_sha(),
        diary=diary_md,
        exhausted_levers=exhausted_md,
        latest_headroom=headroom_md,
        regression_count=state["consecutive_regressions"],
    )


def _stub_claude(workload_path: Path, state: dict) -> dict:
    """Dry-run stand-in for Claude.

    Always proposes a no-op (touches a marker file in autoperf/diary/) and
    commits. Lets us validate the orchestrator end-to-end without burning
    cluster cycles or Claude session time.
    """
    iter_n = state["iteration"] + 1
    marker = DIARY_DIR / f"{state['workload']}_dryrun_marker.txt"
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(f"dry-run iter {iter_n} at {_now_iso()}\n")
    subprocess.run(["git", "-C", str(REPO), "add", str(marker)], check=True)
    subprocess.run(
        ["git", "-C", str(REPO), "commit", "-m",
         f"autoperf-dryrun iter{iter_n}: marker (no-op) on {state['workload']}"],
        check=True,
    )
    return {
        "action": "CHANGE",
        "summary": f"dry-run no-op marker iter{iter_n}",
        "git_commit_msg": f"autoperf-dryrun iter{iter_n}",
    }


def main() -> int:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--workload", required=True,
                        help="Path to workload YAML")
    parser.add_argument("--iters", type=int, default=12,
                        help="Max iterations (default 12)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Don't spawn Claude or launch cluster jobs")
    parser.add_argument("--from-iter", type=int, default=None,
                        help="Resume from iteration N (default = state's saved iter)")
    args = parser.parse_args()

    workload_path = Path(args.workload).resolve()
    if not workload_path.exists():
        print(f"ERR: workload not found: {workload_path}", file=sys.stderr)
        return 4

    workload_name = workload_path.stem
    state = state_mod.load(workload_name)

    if args.from_iter is not None:
        state["iteration"] = args.from_iter
        state_mod.save(state)

    DIARY_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"# autoperf orchestrator")
    print(f"# workload: {workload_name}")
    print(f"# starting at iter {state['iteration']+1}, max {args.iters}")
    print(f"# dry_run={args.dry_run}")

    prev_headroom = None
    last_metric = state_mod.latest_metric(state)

    for i in range(args.iters):
        iter_n = state["iteration"] + 1
        print(f"\n=== iter {iter_n} ===")

        # Stop check 1: 3 consecutive regressions
        if state["consecutive_regressions"] >= REGRESSION_LIMIT:
            print(f"HALT: {REGRESSION_LIMIT} consecutive regressions")
            state_mod.halt(state, f"{REGRESSION_LIMIT}_regressions")
            break

        # 1. Build prompt + spawn Claude
        if args.dry_run:
            verdict = _stub_claude(workload_path, state)
        else:
            iter_prompt = _build_iter_prompt(workload_path, state, prev_headroom)
            verdict = _spawn_claude(iter_prompt)

        print(f"# verdict: {verdict.get('action')}: "
              f"{verdict.get('summary', verdict.get('reason', ''))}")

        if verdict.get("action") == "HALT":
            state_mod.halt(state, verdict.get("reason", "claude_halt"))
            print(f"HALT: {verdict.get('reason')}")
            break

        if verdict.get("action") != "CHANGE":
            print(f"ERR: unknown action {verdict.get('action')}", file=sys.stderr)
            state_mod.halt(state, "claude_invalid_action")
            return 1

        # 2. Run iter.sh
        run_id = f"autoperf-{workload_name[:6]}-i{iter_n}"
        out_json = REPORTS_DIR / f"{workload_name}_iter{iter_n}.json"
        change_summary = verdict.get("summary", "(no summary)")

        if args.dry_run:
            print(f"# dry-run: skipping iter.sh; would have invoked with run_id={run_id}")
            outcome = "neutral"
            top_leaf = None
            top_headroom = 0.0
            new_metric = last_metric
        else:
            try:
                rc = subprocess.run(
                    [str(BIN / "iter.sh"),
                     "--workload", str(workload_path),
                     "--iteration", str(iter_n),
                     "--run-id", run_id,
                     "--out-json", str(out_json)],
                    check=False,
                ).returncode
            except KeyboardInterrupt:
                print("interrupted")
                return 130

            if rc == 2:
                # eviction → don't count as regression, retry next iter
                print("# evicted; recording and continuing")
                state_mod.record_iter(
                    state, _git_short_sha(), change_summary,
                    last_metric, last_metric, None, 0.0, "evicted",
                )
                continue
            if rc == 3:
                state_mod.halt(state, "budget_exhausted")
                print("HALT: budget exhausted")
                break
            if rc != 0:
                state_mod.halt(state, f"iter_sh_rc_{rc}")
                print(f"HALT: iter.sh failed rc={rc}")
                return rc

            prev_headroom = json.loads(out_json.read_text())
            new_metric = prev_headroom["totals"]["measured_us_per_fwd"]
            top_leaf = (prev_headroom["top_n"][0]["leaf"]
                        if prev_headroom["top_n"] else None)
            top_headroom = (prev_headroom["top_n"][0]["headroom_total_ms"]
                            if prev_headroom["top_n"] else 0.0)
            outcome = _classify_outcome(last_metric, new_metric)

            # Stop check 2: top-leaf below floor
            if prev_headroom["halt_signals"]["top_headroom_below_floor"]:
                state_mod.record_iter(state, _git_short_sha(), change_summary,
                                      last_metric, new_metric, top_leaf,
                                      top_headroom, outcome)
                state_mod.halt(state, "top_headroom_below_floor")
                print(f"HALT: top headroom {top_headroom:.2f}ms < floor")
                break

            # Stop check 3: workload at predicted ceiling
            if prev_headroom["halt_signals"]["all_leaves_at_ceiling"]:
                state_mod.record_iter(state, _git_short_sha(), change_summary,
                                      last_metric, new_metric, top_leaf,
                                      top_headroom, outcome)
                state_mod.halt(state, "workload_at_ceiling")
                print("HALT: workload at predicted ceiling")
                break

        state_mod.record_iter(state, _git_short_sha(), change_summary,
                              last_metric, new_metric, top_leaf, top_headroom,
                              outcome)
        last_metric = new_metric

        # 1-line diary
        diary_line = (
            f"iter{iter_n} {_git_short_sha()}: {change_summary} | "
            f"metric {last_metric or 0:.1f} | top_leaf={top_leaf} "
            f"hr={top_headroom:.1f}ms | {outcome}"
        )
        (DIARY_DIR / f"{workload_name}.log").open("a").write(diary_line + "\n")
        print(f"# {diary_line}")

    # Final summary
    summary_path = DIARY_DIR / f"{workload_name}_summary.md"
    _write_summary(state, summary_path)
    print(f"\n# summary written to {summary_path}")
    return 0


def _write_summary(state: dict, out: Path) -> None:
    lines = [f"# autoperf summary — {state['workload']}",
             "",
             f"started: {state['started_at']}",
             f"last iter: {state.get('last_iter_at') or '_never_'}",
             f"iterations: {state['iteration']}",
             f"halt_reason: {state.get('halt_reason') or '_running_'}",
             f"exhausted levers: {', '.join(state['exhausted_levers']) or '_(none)_'}",
             "",
             "## diary",
             "",
             "| iter | git_sha | change | metric_before | metric_after | top_leaf | headroom_ms | outcome |",
             "|---|---|---|---|---|---|---|---|"]
    for e in state["diary"]:
        lines.append(
            f"| {e['iter']} | {e['git_sha']} | {e['change'][:60]} | "
            f"{e.get('metric_before','-')} | {e.get('metric_after','-')} | "
            f"{e.get('top_leaf','-')} | {e.get('top_headroom_ms', 0):.1f} | "
            f"{e['outcome']} |"
        )
    out.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    sys.exit(main())
