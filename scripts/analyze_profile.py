#!/usr/bin/env python3
"""Analyze XProf profiles for Qwen3.5-397B decode/prefill performance.

Parses xplane.pb files using the xprof Python API and produces:
  1. Per-layer-type breakdown (GQA vs DeltaNet vs MoE vs other)
  2. Per-op-category breakdown (matmul, KV cache, collectives, etc.)
  3. Top-N ops by self-time
  4. A/B comparison between two profile versions

Usage:
    # Single profile analysis
    python scripts/analyze_profile.py /tmp/profiles/v47/host0.xplane.pb

    # A/B comparison
    python scripts/analyze_profile.py \
        --a /tmp/profiles/v46/host0.xplane.pb --label-a v46-manual-sdpa \
        --b /tmp/profiles/v47/host0.xplane.pb --label-b v47-dot-product-attn

    # Download from GCS first
    python scripts/analyze_profile.py \
        --a gs://max-experiments/qwen-profiles/v7x/decode/.../host0.xplane.pb \
        --b gs://max-experiments/qwen-profiles/v7x/decode/.../host0.xplane.pb

    # Filter to decode-only or prefill-only ops
    python scripts/analyze_profile.py /path/to/xplane.pb --filter decode
    python scripts/analyze_profile.py /path/to/xplane.pb --filter prefill

Requires: pip install tensorboard-plugin-profile
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from collections import defaultdict
from dataclasses import dataclass, field


def _ensure_xprof():
    try:
        from xprof.convert import raw_to_tool_data  # noqa: F401
    except ImportError:
        print("ERROR: xprof not found. Install with: pip install tensorboard-plugin-profile")
        sys.exit(1)


def _resolve_path(path: str) -> str:
    """Download from GCS if needed, return local path."""
    if not path.startswith("gs://"):
        return path
    local = os.path.join(tempfile.mkdtemp(), os.path.basename(path))
    subprocess.run(["gcloud", "storage", "cp", path, local],
                   check=True, capture_output=True)
    return local


# ---------------------------------------------------------------------------
# Op classification
# ---------------------------------------------------------------------------

def _classify_layer(op_name: str) -> str:
    """Classify an op into a layer type based on its name scope."""
    if "gqa_attn" in op_name:
        return "GQA-Attention"
    if "gqa_moe" in op_name:
        return "GQA-MoE"
    if "deltanet_attn" in op_name:
        return "DeltaNet-Attention"
    if "deltanet_moe" in op_name:
        return "DeltaNet-MoE"
    if "moe_routing" in op_name:
        return "MoE-Routing"
    if "moe_experts" in op_name:
        return "MoE-Experts"
    if "embed" in op_name or "lm_head" in op_name:
        return "Embed/LMHead"
    if "rms_norm" in op_name or "norm" in op_name:
        return "Norm"
    return "Other"


def _classify_op_category(op_type: str, op_name: str) -> str:
    """Classify an op into a compute category."""
    op_type_lower = op_type.lower()
    op_name_lower = op_name.lower()

    # Collectives
    if any(c in op_type_lower for c in
           ["all-reduce", "all-gather", "reduce-scatter", "all-to-all",
            "collective", "send", "recv"]):
        return "Collectives"
    if any(c in op_name_lower for c in
           ["reduce-scatter", "all-reduce", "all-gather"]):
        return "Collectives"

    # KV cache ops
    if op_type_lower in ("dynamic_update_slice", "dynamic_slice"):
        if "cache" in op_name_lower or "gqa_attn" in op_name_lower:
            return "KV-Cache"
        return "DynSlice-Other"

    # Matmul / dot
    if op_type_lower in ("dot_general", "dot", "conv"):
        return "Matmul"
    if "ragged-dot" in op_type_lower or "ragged_dot" in op_name_lower:
        return "Matmul-RaggedDot"

    # Type conversion (FP8 dequant)
    if op_type_lower == "convert_element_type":
        return "FP8-Dequant"

    # Attention-specific
    if any(x in op_name_lower for x in ["softmax", "reduce_max", "reduce_sum"]):
        if "gqa" in op_name_lower or "attn" in op_name_lower:
            return "Attention-Softmax"
    if "dot_product_attention" in op_name_lower:
        return "Attention-SDPA"

    # MoE routing
    if any(x in op_name_lower for x in ["top_k", "argsort", "sort"]):
        return "MoE-Sort/TopK"
    if "scatter" in op_type_lower:
        return "Scatter"
    if "gather" in op_type_lower:
        return "Gather"

    # Element-wise
    if op_type_lower in ("multiply", "add", "subtract", "tanh", "exp",
                          "rsqrt", "negate", "maximum", "minimum"):
        return "Elementwise"

    # Reshape/transpose (usually free)
    if op_type_lower in ("reshape", "transpose", "bitcast", "broadcast",
                          "slice", "concatenate", "pad", "squeeze", "iota"):
        return "Layout"

    return "Other"


def _op_filter(op_name: str, filter_mode: str | None) -> bool:
    """Return True if op should be included based on filter."""
    if filter_mode is None:
        return True
    if filter_mode == "decode":
        return "decode_step" in op_name
    if filter_mode == "prefill":
        return "prefill_fn" in op_name
    return True


# ---------------------------------------------------------------------------
# Profile parsing
# ---------------------------------------------------------------------------

@dataclass
class OpRecord:
    name: str
    op_type: str
    occurrences: int
    self_time_us: float
    layer_type: str
    op_category: str
    flop_rate: float = 0.0
    memory_bw: float = 0.0
    bound_by: str = ""


@dataclass
class ProfileSummary:
    label: str
    ops: list[OpRecord] = field(default_factory=list)
    total_device_time_us: float = 0.0

    # Aggregated views
    by_layer: dict[str, float] = field(default_factory=dict)
    by_category: dict[str, float] = field(default_factory=dict)


def parse_profile(xplane_path: str, label: str,
                  filter_mode: str | None = None) -> ProfileSummary:
    """Parse an xplane.pb file and return a ProfileSummary."""
    from xprof.convert import raw_to_tool_data as convert

    local_path = _resolve_path(xplane_path)
    data, _ = convert.xspace_to_tool_data([local_path], "framework_op_stats", {})
    if not data:
        print(f"ERROR: No data from {xplane_path}")
        sys.exit(1)

    stats = json.loads(data)
    rows = stats[0]["rows"]

    summary = ProfileSummary(label=label)
    layer_times: dict[str, float] = defaultdict(float)
    category_times: dict[str, float] = defaultdict(float)

    for row in rows:
        c = row["c"]
        host_dev = c[1]["v"]
        if host_dev != "Device":
            continue

        op_type = c[2]["v"]
        op_name = c[3]["v"]
        occurrences = int(c[4]["v"])
        self_time_us = c[7]["v"]

        if not _op_filter(op_name, filter_mode):
            continue

        layer_type = _classify_layer(op_name)
        op_category = _classify_op_category(op_type, op_name)

        # Extract optional fields
        flop_rate = c[14]["v"] if c[14]["v"] else 0.0
        memory_bw = c[15]["v"] if c[15]["v"] else 0.0
        bound_by = c[17]["v"] if len(c) > 17 else ""

        rec = OpRecord(
            name=op_name, op_type=op_type, occurrences=occurrences,
            self_time_us=self_time_us, layer_type=layer_type,
            op_category=op_category, flop_rate=flop_rate,
            memory_bw=memory_bw, bound_by=bound_by,
        )
        summary.ops.append(rec)
        summary.total_device_time_us += self_time_us
        layer_times[layer_type] += self_time_us
        category_times[op_category] += self_time_us

    summary.by_layer = dict(sorted(layer_times.items(),
                                    key=lambda x: -x[1]))
    summary.by_category = dict(sorted(category_times.items(),
                                       key=lambda x: -x[1]))
    # Sort ops by self-time descending
    summary.ops.sort(key=lambda r: -r.self_time_us)
    return summary


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def _fmt_time(us: float) -> str:
    if us >= 1_000_000:
        return f"{us / 1_000_000:.2f}s"
    if us >= 1_000:
        return f"{us / 1_000:.1f}ms"
    return f"{us:.0f}us"


def _pct(part: float, total: float) -> str:
    if total == 0:
        return "  0.0%"
    return f"{100 * part / total:5.1f}%"


def print_summary(s: ProfileSummary, top_n: int = 25):
    """Print a single profile summary."""
    print(f"\n{'=' * 80}")
    print(f"  Profile: {s.label}")
    print(f"  Total device self-time: {_fmt_time(s.total_device_time_us)}")
    print(f"{'=' * 80}")

    # Layer breakdown
    print(f"\n  Layer Type Breakdown")
    print(f"  {'Layer Type':<25} {'Self-time':>12} {'%':>7}")
    print(f"  {'-' * 46}")
    for layer, time_us in s.by_layer.items():
        print(f"  {layer:<25} {_fmt_time(time_us):>12} {_pct(time_us, s.total_device_time_us)}")

    # Category breakdown
    print(f"\n  Op Category Breakdown")
    print(f"  {'Category':<25} {'Self-time':>12} {'%':>7}")
    print(f"  {'-' * 46}")
    for cat, time_us in s.by_category.items():
        print(f"  {cat:<25} {_fmt_time(time_us):>12} {_pct(time_us, s.total_device_time_us)}")

    # Top ops
    print(f"\n  Top {top_n} Ops by Self-time")
    print(f"  {'#':>3} {'Self-time':>10} {'%':>6} {'Cumul%':>7} "
          f"{'#Occ':>6} {'Bound':>6}  Op Name")
    print(f"  {'-' * 100}")
    cumul = 0.0
    for i, op in enumerate(s.ops[:top_n]):
        cumul += op.self_time_us
        # Shorten name for display
        name = op.name.replace("jit(decode_step)/while/body/", "D/")
        name = name.replace("jit(prefill_fn)/while/body/", "P/")
        name = name.replace("closed_call/", "")
        name = name.replace("while/body/closed_call/", "")
        if len(name) > 70:
            name = name[:67] + "..."
        print(f"  {i+1:>3} {_fmt_time(op.self_time_us):>10} "
              f"{_pct(op.self_time_us, s.total_device_time_us)} "
              f"{_pct(cumul, s.total_device_time_us)} "
              f"{op.occurrences:>6} {op.bound_by:>6}  {name}")


def print_comparison(a: ProfileSummary, b: ProfileSummary, top_n: int = 25):
    """Print side-by-side comparison of two profiles."""
    print(f"\n{'=' * 90}")
    print(f"  A/B Comparison: {a.label} vs {b.label}")
    print(f"{'=' * 90}")
    print(f"  Total device time:  A={_fmt_time(a.total_device_time_us)}"
          f"  B={_fmt_time(b.total_device_time_us)}"
          f"  delta={_pct(b.total_device_time_us - a.total_device_time_us, a.total_device_time_us)}")

    # Layer comparison
    all_layers = list(dict.fromkeys(list(a.by_layer.keys()) + list(b.by_layer.keys())))
    print(f"\n  Layer Type Comparison")
    print(f"  {'Layer Type':<25} {'A':>12} {'A%':>6} {'B':>12} {'B%':>6} {'Delta':>8}")
    print(f"  {'-' * 73}")
    for layer in all_layers:
        ta = a.by_layer.get(layer, 0.0)
        tb = b.by_layer.get(layer, 0.0)
        delta = tb - ta
        delta_str = f"{'+' if delta >= 0 else ''}{_pct(delta, ta) if ta > 0 else 'NEW'}"
        print(f"  {layer:<25} {_fmt_time(ta):>12} {_pct(ta, a.total_device_time_us)} "
              f"{_fmt_time(tb):>12} {_pct(tb, b.total_device_time_us)} {delta_str:>8}")

    # Category comparison
    all_cats = list(dict.fromkeys(list(a.by_category.keys()) + list(b.by_category.keys())))
    print(f"\n  Op Category Comparison")
    print(f"  {'Category':<25} {'A':>12} {'A%':>6} {'B':>12} {'B%':>6} {'Delta':>8}")
    print(f"  {'-' * 73}")
    for cat in all_cats:
        ta = a.by_category.get(cat, 0.0)
        tb = b.by_category.get(cat, 0.0)
        delta = tb - ta
        delta_str = f"{'+' if delta >= 0 else ''}{_pct(delta, ta) if ta > 0 else 'NEW'}"
        print(f"  {cat:<25} {_fmt_time(ta):>12} {_pct(ta, a.total_device_time_us)} "
              f"{_fmt_time(tb):>12} {_pct(tb, b.total_device_time_us)} {delta_str:>8}")

    # Top movers (biggest absolute delta)
    # Build op lookup by (op_type, short_name) for matching
    def _op_key(name: str) -> str:
        return name.replace("jit(decode_step)/while/body/", "").replace(
            "jit(prefill_fn)/while/body/", "")

    a_by_key = {_op_key(op.name): op for op in a.ops}
    b_by_key = {_op_key(op.name): op for op in b.ops}
    all_keys = set(a_by_key.keys()) | set(b_by_key.keys())

    movers = []
    for key in all_keys:
        ta = a_by_key[key].self_time_us if key in a_by_key else 0
        tb = b_by_key[key].self_time_us if key in b_by_key else 0
        movers.append((key, ta, tb, tb - ta))
    movers.sort(key=lambda x: -abs(x[3]))

    print(f"\n  Top {top_n} Movers (biggest absolute self-time change)")
    print(f"  {'#':>3} {'A-time':>10} {'B-time':>10} {'Delta':>10} {'Δ%':>7}  Op")
    print(f"  {'-' * 90}")
    for i, (key, ta, tb, delta) in enumerate(movers[:top_n]):
        name = key
        if len(name) > 60:
            name = name[:57] + "..."
        dpct = f"{100 * delta / ta:+.1f}%" if ta > 0 else "NEW"
        print(f"  {i+1:>3} {_fmt_time(ta):>10} {_fmt_time(tb):>10} "
              f"{'+' if delta >= 0 else ''}{_fmt_time(abs(delta)):>9} {dpct:>7}  {name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Analyze XProf xplane.pb profiles for Qwen3.5-397B",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("profile", nargs="?",
                        help="Single xplane.pb path (local or gs://)")
    parser.add_argument("--a", help="Profile A for A/B comparison")
    parser.add_argument("--b", help="Profile B for A/B comparison")
    parser.add_argument("--label-a", default="A", help="Label for profile A")
    parser.add_argument("--label-b", default="B", help="Label for profile B")
    parser.add_argument("--filter", choices=["decode", "prefill"],
                        help="Filter to decode-only or prefill-only ops")
    parser.add_argument("--top-n", type=int, default=25,
                        help="Number of top ops to show (default: 25)")
    args = parser.parse_args()

    _ensure_xprof()

    if args.a and args.b:
        # A/B comparison mode
        print(f"Parsing profile A: {args.a}")
        sa = parse_profile(args.a, args.label_a, args.filter)
        print(f"Parsing profile B: {args.b}")
        sb = parse_profile(args.b, args.label_b, args.filter)
        print_summary(sa, args.top_n)
        print_summary(sb, args.top_n)
        print_comparison(sa, sb, args.top_n)
    elif args.profile:
        # Single profile mode
        print(f"Parsing profile: {args.profile}")
        s = parse_profile(args.profile, args.profile, args.filter)
        print_summary(s, args.top_n)
    else:
        parser.error("Provide either a single profile path or --a/--b for comparison")


if __name__ == "__main__":
    main()
