# Canonical autoperf-blocking issue body

Used when the autoperf agent files an issue against `cde`, `perfsim`, or
`xla-shell`. Mirror this in each tool repo's `.github/ISSUE_TEMPLATE/`
(verbatim) so `gh issue create --template autoperf-blocking` pre-populates it.

```markdown
**Tool**: <cde | perfsim | xla-shell>
**Filed by**: autoperf agent (jax-gpt commit <SHA>)
**Workload**: <name from autoperf/workloads/*.yaml>
**Iteration**: <N>

## Context

- Model preset: `<key from MODEL_PRESETS>`
- Hardware preset: `<key from HW_PRESETS>`
- Parallelism: `tp=<X> ep=<Y> dp=<Z> fsdp=<W>`
- Workload params:
  - batch=<B>, ctx=<C>, prompt=<P>
  - weight_dtype=<dt>, kv_dtype=<dt>
- Profile path: `<gs:// URI from cde profile path>` (or local: `<absolute path>`)
- Top-leaf the autoperf agent was trying to optimize: `<leaf name>`

## What I tried

```bash
<exact, copy-pasteable command>
```

## Expected (per docs/contract)

<what the docs/perfsim-protocol.md / docs/auto-perf-guide.md / cde --help / xla-shell README say should happen>

## Got

```
<actual stdout/stderr/output, paste verbatim>
```

## Repro (minimum, copy-pasteable)

```bash
<the smallest sequence of commands that reproduces the bug from a clean state>
```

## Definition of done

<one concrete observable that would let the autoperf agent confirm the fix landed>

For example:
- "After fix, `headroom_report --model X --hardware Y --xplane <path>` reports
  leaf `Z` with `predicted_us` between A and B (currently reports <wrong-value>)."
- "After fix, `cde profile path <run_id>` returns the gs:// URI within 30s
  (currently times out after 5 min)."
- "After fix, `xla_shell read_xprof <dir>` produces a fusion record named
  `Expert_gate_up` for the FFN-up matmul (currently named `<wrong>`)."

## Workaround

<what the autoperf agent did instead, if any — including "halted iteration N">
```

## Why this template

- **Tool/Filed-by/Workload/Iteration** in the header → reproduce the exact
  agent state.
- **Profile path** → tool agents need a real `xplane.pb` to repro.
- **Expected vs Got** with refs to docs → makes "is this a real bug or
  protocol violation?" a 30-second judgment.
- **Repro** is verbatim → the maintainer agent runs it without rephrasing.
- **Definition of done** is the single most important field → without a
  testable success criterion, agents loop on "is it fixed yet?".
