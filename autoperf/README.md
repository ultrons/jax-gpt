## autoperf — auto-optimization agent harness

Long-running orchestrator that drives a perfsim-guided perf-optimization loop
over one workload at a time. Each iteration applies ONE code change, builds,
launches via `cde` with profiling on, waits for completion, pulls the profile,
runs `perfsim.headroom_report`, and picks the next change from the heuristic
table in `~/ml-experiments-perfsim/perfsim/docs/auto-perf-guide.md`.

### Layout

```
bin/                  shell + python utilities (one job each)
  cde-wait.sh         poll cde status until terminal state, exit 0/1
  headroom_to_json.py wrap headroom_report → top-N JSON
  budget_check.py     hard cap on jobs/day + compile minutes
  iter.sh             one-iteration glue: build → launch → wait → pull → report

orchestrator/
  autoperf.py         main loop: spawns claude per iter, manages state
  state.py            iter counter, diary, exhausted-levers tracking

agent/
  iter_prompt.md      template for per-iter Claude prompt
  halt_prompt.md      structured halt-signal contract

workloads/
  *.yaml              workload spec: model, hw, parallelism, regime, baseline cmd

state/                gitignored; per-workload runtime state
diary/                commit-tracked iteration diaries (audit trail)
```

### Quick start

```bash
# Dry-run (proposes changes but doesn't apply or launch)
python orchestrator/autoperf.py --workload workloads/qwen35_decode.yaml --dry-run --iters 1

# Real run, 12 iterations max, hard budget cap
python orchestrator/autoperf.py --workload workloads/qwen35_decode.yaml --iters 12
```

### Stop conditions (hard-coded; don't override unless documented)

1. Top-leaf headroom < 0.5 ms total (nothing left to harvest).
2. 3 iterations in a row regress the metric (digging a hole).
3. Daily budget cap hit (default: 12 cluster jobs/day).
4. Claude signals `HALT` in iter prompt (novel situation, needs human).
5. Predicted ≈ measured for all leaves > 5% step share (workload at ceiling).

### Audit trail

Every iteration produces:
- A commit on `auto-perf/<workload>` branch with the change.
- A diary entry in `diary/<workload>/iter<N>.md`.
- An updated state JSON in `state/<workload>.json`.

Resume any sprint by re-invoking with the same workload — orchestrator picks
up from the saved state.
