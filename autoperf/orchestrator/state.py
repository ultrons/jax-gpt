"""state.py — per-workload runtime state for autoperf orchestrator.

Persisted in autoperf/state/<workload>.json with this schema:

{
  "workload": str,
  "iteration": int,
  "started_at": ISO timestamp,
  "last_iter_at": ISO timestamp,
  "diary": [
    {"iter": int, "git_sha": str, "change": str, "metric_before": float,
     "metric_after": float, "top_leaf": str, "top_headroom_ms": float,
     "outcome": "improved" | "regressed" | "neutral" | "halted" | "evicted"},
    ...
  ],
  "exhausted_levers": [str, ...],   # leaves where Claude decided "no lever to pull"
  "consecutive_regressions": int,
  "halt_reason": str | null,
  "budget": {
    "<YYYY-MM-DD>": {"jobs": int, "compile_min": float}
  }
}

Functions:
  - load(workload) → State
  - save(state)
  - record_iter(state, ...) — append diary entry, increment counters, persist
  - mark_lever_exhausted(state, leaf)
  - bump_budget(state, jobs=0, compile_min=0.0)
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

STATE_DIR = Path("/home/sivaibhav_google_com/jax-gpt/autoperf/state")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _path_for(workload: str) -> Path:
    return STATE_DIR / f"{workload}.json"


def load(workload: str) -> dict:
    """Load state for workload. Returns fresh state if none exists."""
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    p = _path_for(workload)
    if not p.exists():
        return {
            "workload": workload,
            "iteration": 0,
            "started_at": _now_iso(),
            "last_iter_at": None,
            "diary": [],
            "exhausted_levers": [],
            "consecutive_regressions": 0,
            "halt_reason": None,
            "budget": {},
        }
    return json.loads(p.read_text())


def save(state: dict) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    p = _path_for(state["workload"])
    p.write_text(json.dumps(state, indent=2))


def record_iter(
    state: dict,
    git_sha: str,
    change: str,
    metric_before: Optional[float],
    metric_after: Optional[float],
    top_leaf: Optional[str],
    top_headroom_ms: Optional[float],
    outcome: str,
) -> None:
    """Append a diary entry for the iteration just completed."""
    state["iteration"] += 1
    state["last_iter_at"] = _now_iso()
    entry = {
        "iter": state["iteration"],
        "git_sha": git_sha,
        "change": change,
        "metric_before": metric_before,
        "metric_after": metric_after,
        "top_leaf": top_leaf,
        "top_headroom_ms": top_headroom_ms,
        "outcome": outcome,
    }
    state["diary"].append(entry)

    # consecutive_regressions counter
    if outcome == "regressed":
        state["consecutive_regressions"] += 1
    elif outcome in ("improved", "neutral"):
        state["consecutive_regressions"] = 0

    save(state)


def mark_lever_exhausted(state: dict, leaf: str) -> None:
    if leaf not in state["exhausted_levers"]:
        state["exhausted_levers"].append(leaf)
        save(state)


def bump_budget(state: dict, jobs: int = 0, compile_min: float = 0.0) -> None:
    today = _today()
    cur = state["budget"].get(today, {"jobs": 0, "compile_min": 0.0})
    cur["jobs"] += jobs
    cur["compile_min"] += compile_min
    state["budget"][today] = cur
    save(state)


def halt(state: dict, reason: str) -> None:
    state["halt_reason"] = reason
    save(state)


def latest_metric(state: dict) -> Optional[float]:
    """Return the most recent metric_after we've recorded, for trend tracking."""
    for e in reversed(state["diary"]):
        if e.get("metric_after") is not None:
            return e["metric_after"]
    return None
