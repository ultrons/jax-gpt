"""DSv3 Fused EP-MoE — bench harness (SPEC §7).

This is the **stub** at Phase A.5; Phase C will fill in the real attention
glue and wire the actual Pallas kernels. For now the structural skeleton is
runnable end-to-end using `jax_ref.moe_forward` as the MoE-block placeholder,
so we can verify the harness layout (mesh, sharding, layer loop, timing,
fwd+bwd) before Phase B kernel code lands.

Per SPEC §7, the bench tests the kernel inside a realistic transformer block:

    for each MoE layer:
      x = LayerNorm(x)
      q, k, v = attention_qkv_proj(x)
      attn_out = attention(q, k, v)
      x = x + attn_out
      x = LayerNorm(x)
      x = moe_block(x, W_gate, W1, W_d)         # the kernel under test

Configurable:
  - mesh_preset: "iteration" (EP=2 FSDP=8) or "production" (EP=4 FSDP=128)
  - moe_impl:    "jax_ref" (pure-JAX baseline) | "v_outside" | "v_inside"
                 — Phase B kernel slots; "jax_ref" is the only one wired now
  - num_layers:  default 3 — measure steady-state, not first-layer warmup

Frontmatter:
  slug: dsv3-fused-ep-moe-bench
  intent: bench-harness
  status: STUB v0 — Phase A.5 prereq; jax_ref-only; Phase C fills attention + real kernel
  sources:
    - targets/dsv3-fused-ep-moe/SPEC.md (v0.4 §7)
    - targets/dsv3-fused-ep-moe/jax_ref.py
  related: targets/dsv3-fused-ep-moe/build/tools/aot_check.py
"""
from __future__ import annotations

import argparse
import functools
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# Import jax_ref from the target dir.
_TARGET_ROOT = Path(__file__).resolve().parent
if str(_TARGET_ROOT) not in sys.path:
    sys.path.insert(0, str(_TARGET_ROOT))


@dataclass(frozen=True)
class BenchConfig:
    """Bench-time config (separate from SPEC's MoEConfig — adds attention/layer dims)."""
    # MoE shapes (must match jax_ref.MoEConfig).
    E: int = 8
    D: int = 64
    F: int = 32
    K: int = 2
    # Attention shapes.
    n_heads: int = 4
    head_dim: int = 16
    # Bench shapes.
    T: int = 16
    num_layers: int = 3
    # Steady-state measurement.
    num_warmup: int = 2
    num_timed: int = 5


# Mesh presets per SPEC v0.4 §5.4 + AOT preset table. Stub uses "iteration"
# scaled-down to fit local TPU (4 chips = 8 cores).
MESH_PRESETS = {
    "local":      {"axes": (1, 2, 4, 1),  "topo": "tpu7x:2x2x1"},   # 8 cores; this VM
    "iteration":  {"axes": (1, 2, 8, 1),  "topo": "tpu7x:2x2x2"},   # 16 cores; bodaborg
    "production": {"axes": (1, 4, 128, 1), "topo": "tpu7x:4x8x8"},  # 512 cores; ninja-class
}


# -----------------------------------------------------------------------------
# Multi-head attention — qkv-proj + scaled-dot-product + output-proj.
# Representative compute volume; not optimised (no flash/splash) — good
# enough for the bench harness's "before-MoE" stage.
# -----------------------------------------------------------------------------

def multihead_attention(x, W_qkv, W_o, n_heads: int):
    """Multi-head attention.

    x:     (T, D) bf16
    W_qkv: (D, 3*D) bf16
    W_o:   (D, D)   bf16
    n_heads: D must be divisible by n_heads

    Returns: (T, D) bf16
    """
    import jax.numpy as jnp
    T, D = x.shape
    head_dim = D // n_heads
    # f32 internally to match jax_ref's routing precision philosophy.
    xf = x.astype(jnp.float32)
    qkv = xf @ W_qkv.astype(jnp.float32)                          # (T, 3D)
    q, k, v = jnp.split(qkv, 3, axis=-1)                          # (T, D) each
    q = q.reshape(T, n_heads, head_dim)
    k = k.reshape(T, n_heads, head_dim)
    v = v.reshape(T, n_heads, head_dim)
    scale = jnp.float32(1.0) / jnp.sqrt(jnp.float32(head_dim))
    # Per-head scaled dot-product: (n_heads, T, T)
    scores = jnp.einsum("thd,uhd->htu", q, k) * scale
    attn = jax.nn.softmax(scores, axis=-1)
    # (T, n_heads, head_dim) ← (n_heads, T, T) × (T, n_heads, head_dim)
    out = jnp.einsum("htu,uhd->thd", attn, v).reshape(T, D)
    out = out @ W_o.astype(jnp.float32)
    return out.astype(x.dtype)


# Back-compat alias — earlier code referenced attention_stub.
def attention_stub(x, W_qkv, W_o):
    """Legacy entrypoint. Calls multihead_attention with n_heads=4 (matches
    BenchConfig default). Kept so existing import sites don't break."""
    return multihead_attention(x, W_qkv, W_o, n_heads=4)


# -----------------------------------------------------------------------------
# LayerNorm — bias-free, scale-free RMSNorm-ish stub.
# -----------------------------------------------------------------------------

def layernorm_stub(x, eps: float = 1e-5):
    import jax.numpy as jnp
    var = (x.astype(jnp.float32) ** 2).mean(axis=-1, keepdims=True)
    return (x.astype(jnp.float32) / jnp.sqrt(var + eps)).astype(x.dtype)


# -----------------------------------------------------------------------------
# MoE-block dispatch: jax_ref now; v_outside / v_inside in Phase B.
# -----------------------------------------------------------------------------

def get_moe_block(impl: str, cfg: BenchConfig, bwd_impl: str = "jax"):
    """Return (moe_block_fn, is_kernel) where moe_block_fn(x, Wg, W1, Wd) -> y.

    is_kernel=True means it's the kernel under test; False means baseline (jax_ref)."""
    from jax_ref import MoEConfig, moe_forward
    moe_cfg = MoEConfig(E=cfg.E, D=cfg.D, F=cfg.F, K=cfg.K)

    if impl == "jax_ref":
        def fn(x, Wg, W1, Wd):
            return moe_forward(x, Wg, W1, Wd, moe_cfg)
        return fn, False

    if impl == "v_outside":
        from build.v_outside.moe_block import MoEBlockConfig
        from build.v_outside.moe_block_vjp import make_moe_block
        # bt_ffn heuristic: ~M/4 (4 grid tiles) but capped at 1024 so the bwd
        # kernel's per-tile VMEM scratches stay safe. At small M (<128) falls
        # through to the "tile" path automatically via impl="auto".
        M = cfg.T * cfg.K
        bt_ffn = min(1024, max(8, M // 4))
        # Round down to nearest multiple of 128 if ≥128 (Mosaic rank-1
        # BlockSpec lane-count rule); else keep as-is.
        if bt_ffn >= 128:
            bt_ffn = (bt_ffn // 128) * 128
            # Ensure M is divisible by bt_ffn.
            while bt_ffn > 128 and M % bt_ffn != 0:
                bt_ffn -= 128
        block_cfg = MoEBlockConfig(
            E=cfg.E, D=cfg.D, F=cfg.F, K=cfg.K,
            bt_router=cfg.T,
            bt_ffn=bt_ffn,
        )
        # bwd_impl: "jax" (default; safe at any M) or "pallas" (D.1+D.3;
        # grid kernel kicks in when bt_ffn >= 128).
        moe_block = make_moe_block(block_cfg, bwd_impl=bwd_impl)
        return moe_block, True

    if impl == "v_inside":
        raise NotImplementedError(
            "v_inside not wired yet — gated on v_outside G3 + 2 remaining "
            "[2B-PENDING] pattern docs (streaming-ag-into-matmul, "
            "streaming-psum-scatter)")

    raise ValueError(f"unknown moe_impl: {impl}")


# -----------------------------------------------------------------------------
# One transformer block (per SPEC §7)
# -----------------------------------------------------------------------------

def transformer_block(x, params, moe_block_fn, n_heads: int):
    """x: (T, D) bf16. params: dict per layer. moe_block_fn: (x, Wg, W1, Wd) -> y.

    Returns: (T, D) bf16 — output after attention + MoE residuals.
    """
    # Attention sub-block
    h = layernorm_stub(x)
    attn_out = multihead_attention(h, params["W_qkv"], params["W_o"], n_heads)
    x = x + attn_out

    # MoE sub-block. The MoE kernel itself owns its residual add per SPEC §3
    # step 4.4, so moe_block_fn returns x_in + moe_contribution directly.
    h = layernorm_stub(x)
    x = moe_block_fn(h, params["W_gate"], params["W1"], params["W_d"])
    return x


def stack_layers(x, all_params, moe_block_fn, num_layers, n_heads):
    for layer_idx in range(num_layers):
        x = transformer_block(x, all_params[layer_idx], moe_block_fn, n_heads)
    return x


# -----------------------------------------------------------------------------
# Param init (synthetic) and one-step loss (for jax.grad timing).
# -----------------------------------------------------------------------------

def init_params(cfg: BenchConfig, seed: int = 0):
    import jax
    import jax.numpy as jnp

    keys = jax.random.split(jax.random.PRNGKey(seed), cfg.num_layers * 5)
    params_per_layer = []
    for L in range(cfg.num_layers):
        k_qkv, k_o, k_g, k_w1, k_wd = keys[L*5:(L+1)*5]
        layer = {
            "W_qkv":  (jax.random.normal(k_qkv, (cfg.D, 3 * cfg.D))         * 0.05).astype(jnp.bfloat16),
            "W_o":    (jax.random.normal(k_o,   (cfg.D, cfg.D))             * 0.05).astype(jnp.bfloat16),
            "W_gate": (jax.random.normal(k_g,   (cfg.E, cfg.D))             * 0.1 ).astype(jnp.bfloat16),
            "W1":     (jax.random.normal(k_w1,  (cfg.E, cfg.D, 2 * cfg.F))  * 0.05).astype(jnp.bfloat16),
            "W_d":    (jax.random.normal(k_wd,  (cfg.E, cfg.F, cfg.D))      * 0.05).astype(jnp.bfloat16),
        }
        params_per_layer.append(layer)
    return params_per_layer


def loss_fn(x_in, all_params, moe_block_fn, num_layers, n_heads):
    return stack_layers(x_in, all_params, moe_block_fn, num_layers, n_heads).sum()


# -----------------------------------------------------------------------------
# Bench loop — fwd-only and fwd+bwd, with warmup + timed steady-state.
# -----------------------------------------------------------------------------

@dataclass
class BenchResult:
    impl: str
    mesh_preset: str
    num_layers: int
    fwd_ms_mean: float
    fwd_ms_std: float
    bwd_ms_mean: float
    bwd_ms_std: float
    timings_fwd_ms: list = field(default_factory=list)
    timings_bwd_ms: list = field(default_factory=list)


def run_bench(impl: str, mesh_preset: str, cfg: BenchConfig,
              bwd_impl: str = "jax") -> BenchResult:
    import jax
    import jax.numpy as jnp

    moe_block_fn, _is_kernel = get_moe_block(impl, cfg, bwd_impl=bwd_impl)

    print(f"[bench] impl={impl}, mesh={mesh_preset}, cfg={cfg}")
    print(f"[bench] devices: {jax.devices()}")

    x_in = (jax.random.normal(jax.random.PRNGKey(42), (cfg.T, cfg.D)) * 0.5).astype(jnp.bfloat16)
    params = init_params(cfg, seed=1)

    fwd = jax.jit(lambda x, p: stack_layers(x, p, moe_block_fn, cfg.num_layers, cfg.n_heads))
    bwd = jax.jit(jax.grad(loss_fn, argnums=(0, 1)),
                  static_argnames=("moe_block_fn", "num_layers", "n_heads"))

    # Warmup
    for _ in range(cfg.num_warmup):
        y = fwd(x_in, params)
        y.block_until_ready()

    # Timed forward
    fwd_ms = []
    for _ in range(cfg.num_timed):
        t0 = time.perf_counter()
        y = fwd(x_in, params)
        y.block_until_ready()
        fwd_ms.append((time.perf_counter() - t0) * 1000)

    # Timed backward (fwd + bwd; we report bwd by subtracting? simpler: time
    # full grad call which includes a forward pass plus backward).
    bwd_ms = []
    for _ in range(cfg.num_warmup):
        g_x, g_p = bwd(x_in, params, moe_block_fn=moe_block_fn,
                       num_layers=cfg.num_layers, n_heads=cfg.n_heads)
        jax.block_until_ready((g_x, g_p))
    for _ in range(cfg.num_timed):
        t0 = time.perf_counter()
        g_x, g_p = bwd(x_in, params, moe_block_fn=moe_block_fn,
                       num_layers=cfg.num_layers, n_heads=cfg.n_heads)
        jax.block_until_ready((g_x, g_p))
        bwd_ms.append((time.perf_counter() - t0) * 1000)

    return BenchResult(
        impl=impl,
        mesh_preset=mesh_preset,
        num_layers=cfg.num_layers,
        fwd_ms_mean=float(np.mean(fwd_ms)),
        fwd_ms_std=float(np.std(fwd_ms)),
        bwd_ms_mean=float(np.mean(bwd_ms)),
        bwd_ms_std=float(np.std(bwd_ms)),
        timings_fwd_ms=fwd_ms,
        timings_bwd_ms=bwd_ms,
    )


def print_result(r: BenchResult) -> None:
    print()
    print(f"[bench] === {r.impl} on {r.mesh_preset}, {r.num_layers} layers ===")
    print(f"[bench] fwd  : {r.fwd_ms_mean:6.2f} ± {r.fwd_ms_std:5.2f} ms (n={len(r.timings_fwd_ms)})")
    print(f"[bench] fwd+bwd: {r.bwd_ms_mean:6.2f} ± {r.bwd_ms_std:5.2f} ms")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="DSv3 fused EP-MoE bench harness.")
    p.add_argument("--impl", choices=("jax_ref", "v_outside", "v_inside"),
                   default="jax_ref",
                   help="Which MoE implementation to bench")
    p.add_argument("--mesh", choices=tuple(MESH_PRESETS.keys()), default="local",
                   help="Mesh preset (local TPU / bodaborg iteration / ninja production)")
    p.add_argument("--num-layers", type=int, default=3)
    p.add_argument("--num-timed", type=int, default=5)
    p.add_argument("--T", type=int, default=16, help="Token count per device")
    p.add_argument("--sweep", action="store_true",
                   help="Run a sweep over T values and impls; ignore --T/--impl")
    p.add_argument("--shape", choices=("tiny", "small"), default="tiny",
                   help="Model-shape preset for --sweep. tiny: E=8 D=64 F=32 K=2 (default; dispatch-dominated). "
                        "small: E=16 D=256 F=128 K=4 (matmul-dominated; better G5 signal).")
    p.add_argument("--bwd-impl", choices=("jax", "pallas"), default="jax",
                   help="Bwd path for v_outside. 'jax' (default; safe) or 'pallas' "
                        "(D.1+D.3; grid kernel when bt_ffn>=128). No effect for jax_ref.")
    args = p.parse_args(argv)

    if args.sweep:
        return _sweep(args)

    cfg = BenchConfig(num_layers=args.num_layers, num_timed=args.num_timed, T=args.T)
    print("=" * 72)
    print(f"DSv3 fused EP-MoE bench — impl={args.impl}, mesh={args.mesh}")
    print("=" * 72)

    if args.mesh != "local":
        print(f"[bench] WARNING: mesh={args.mesh} requires {MESH_PRESETS[args.mesh]['topo']}; "
              f"local TPU has {len(__import__('jax').devices())} devices. "
              f"Cluster mesh wiring lands later; stub runs single-device.")

    try:
        r = run_bench(args.impl, args.mesh, cfg, bwd_impl=args.bwd_impl)
    except NotImplementedError as e:
        print(f"[bench] STUB: {e}")
        return 0

    print_result(r)
    return 0


def _sweep(args) -> int:
    """G5 perf signal: side-by-side jax_ref vs v_outside across T values."""
    Ts = [16, 64, 128, 256, 512, 1024]
    impls = ["jax_ref", "v_outside"]
    SHAPES = {
        "tiny":  dict(E=8,  D=64,  F=32,  K=2, n_heads=4,  head_dim=16),
        "small": dict(E=16, D=256, F=128, K=4, n_heads=4,  head_dim=64),
    }
    shape = SHAPES[args.shape]
    print("=" * 80)
    print(f"DSv3 fused EP-MoE bench — shape={args.shape} {shape}")
    print(f"sweep over T={Ts}, layers={args.num_layers}, impls={impls}, bwd_impl={args.bwd_impl}")
    print("=" * 80)

    rows = []
    for T in Ts:
        for impl in impls:
            cfg = BenchConfig(num_layers=args.num_layers,
                              num_timed=args.num_timed, T=T, **shape)
            try:
                r = run_bench(impl, "local", cfg, bwd_impl=args.bwd_impl)
                rows.append((T, impl, r.fwd_ms_mean, r.fwd_ms_std,
                             r.bwd_ms_mean, r.bwd_ms_std))
            except Exception as e:
                print(f"[bench] {impl} @ T={T}: ERROR {e}")
                rows.append((T, impl, None, None, None, None))

    # Side-by-side table
    print()
    print(f"{'T':>5} | {'impl':<10} | {'fwd ms':>10} | {'fwd+bwd ms':>12}")
    print(f"{'-'*5} | {'-'*10} | {'-'*10} | {'-'*12}")
    for T, impl, fm, fs, bm, bs in rows:
        if fm is None:
            print(f"{T:>5} | {impl:<10} | {'ERR':>10} | {'ERR':>12}")
        else:
            print(f"{T:>5} | {impl:<10} | {fm:>6.2f}±{fs:>4.2f} | {bm:>7.2f}±{bs:>4.2f}")

    # Speedup column
    print()
    print(f"{'T':>5} | jax_ref fwd | v_outside fwd | speedup (fwd) | jax_ref bwd | v_outside bwd | speedup (bwd)")
    by_T = {}
    for T, impl, fm, _, bm, _ in rows:
        if fm is None:
            continue
        by_T.setdefault(T, {})[impl] = (fm, bm)
    for T in Ts:
        d = by_T.get(T, {})
        if "jax_ref" in d and "v_outside" in d:
            jr_f, jr_b = d["jax_ref"]
            vo_f, vo_b = d["v_outside"]
            sp_f = jr_f / vo_f if vo_f else float("nan")
            sp_b = jr_b / vo_b if vo_b else float("nan")
            print(f"{T:>5} | {jr_f:>10.2f} | {vo_f:>12.2f} | {sp_f:>12.2f}× | "
                  f"{jr_b:>10.2f} | {vo_b:>12.2f} | {sp_b:>12.2f}×")
    return 0


# Need to import jax at module level for attention_stub's softmax.
import jax


if __name__ == "__main__":
    sys.exit(main())
