#!/usr/bin/env python3
"""Mini DeepSeek-v3: Training loop and CLI.

Pure JAX implementation with pluggable components (MLA, GMM, routing).
Every component has jax.named_scope for xla_shell profile analysis.

Usage:
    # Smoke test on local TPU (v4-4, 4 cores)
    python mini_dsv3/train.py --config tiny

    # Full mini model on 4x4x4
    python mini_dsv3/train.py --config mini --mesh 4x4x4

    # With profiling
    python mini_dsv3/train.py --config tiny --profile --profile_dir /tmp/mini_profile

    # Use pure JAX (no Pallas kernels)
    python mini_dsv3/train.py --config tiny --moe_backend jax --attn_backend jax

    # Use ragged_dot GMM
    python mini_dsv3/train.py --config tiny --moe_backend ragged_dot

    # Use Megablox GMM (TPU only)
    python mini_dsv3/train.py --config tiny --moe_backend megablox

    # With AdamW optimizer
    python mini_dsv3/train.py --config tiny --optimizer adamw

    # With gradient checkpointing
    python mini_dsv3/train.py --config tiny --gradient_checkpoint
"""

from __future__ import annotations

import argparse
import gc
import os
import time

import jax
import jax.numpy as jnp
from jax import random

from .model import (
    ModelConfig, ShardConfig, CONFIGS,
    init_params, compute_loss, full_671b_config,
)


# ============================================================================
# Optimizer
# ============================================================================

def init_adam_state(params):
    """Initialize AdamW optimizer state."""
    return {
        "m": jax.tree.map(jnp.zeros_like, params),
        "v": jax.tree.map(jnp.zeros_like, params),
        "step": jnp.array(0, dtype=jnp.int32),
    }


def adam_step(params, grads, opt_state, lr=1e-4,
             beta1=0.9, beta2=0.999, eps=1e-8, wd=0.01):
    """One AdamW update step."""
    step = opt_state["step"] + 1
    m = jax.tree.map(lambda m, g: beta1 * m + (1 - beta1) * g,
                     opt_state["m"], grads)
    v = jax.tree.map(lambda v, g: beta2 * v + (1 - beta2) * g ** 2,
                     opt_state["v"], grads)
    bc = 1 - beta1 ** step
    bc2 = 1 - beta2 ** step
    params = jax.tree.map(
        lambda p, m_, v_: p - lr * (m_ / bc / (jnp.sqrt(v_ / bc2) + eps) + wd * p),
        params, m, v)
    return params, {"m": m, "v": v, "step": step}


# ============================================================================
# FLOP estimation
# ============================================================================

def count_params(params) -> int:
    """Count total parameters."""
    leaves = jax.tree.leaves(params)
    return sum(x.size for x in leaves if hasattr(x, 'size'))


def estimate_flops(cfg: ModelConfig, B: int) -> float:
    """Estimate FLOPs for one forward+backward pass (global cluster total).

    B is the global batch size (all sequences processed across the whole cluster).
    Returns total cluster FLOPs — divide by n_chips to get per-chip FLOPs.

    Backward FLOPs = 2× forward: each weight matmul has dX + dW, each at the
    same cost as the forward matmul, so bwd = 2× fwd and total = 3× fwd.
    With gradient checkpointing, add one extra forward recompute: total = 4× fwd.
    """
    D, H, S = cfg.D, cfg.H, cfg.S
    R_q, R_kv = cfg.R_q, cfg.R_kv
    qk_dim, d_v = cfg.qk_dim, cfg.d_v

    # MLA forward per layer (all projections + attention + output)
    mla_fwd = (2*B*D*R_q + 2*B*R_q*H*qk_dim +           # Q LoRA
               2*B*D*(R_kv + cfg.d_rope) +                 # KV LoRA
               2*B*R_kv*H*(cfg.d_nope + d_v) +            # KV proj_b
               2*B*H*S*qk_dim + 2*B*H*S*d_v +            # QK^T + AV
               2*B*H*d_v*D)                                # out proj

    # Dense MLP forward: gate + up + down (SwiGLU)
    dense_fwd = 2*B*D*cfg.D_mlp * 3

    # MoE forward: expert multiplier depends on backend
    # "jax" and "ring" compute all E experts; ragged_dot/megablox compute only K
    if cfg.moe_backend in ("jax", "fused_ep_moe_v4_jax_fwd", "pfwd_jbwd"):
        expert_mult = cfg.E  # all-expert dispatch (einsum or JAX path)
    else:
        expert_mult = cfg.K  # ragged_dot / Pallas: only top-K active experts
    moe_fwd = (2*B*D*cfg.E +                              # gate logits
               2*B*expert_mult*D*cfg.D_moe * 3 +          # routed: wi0 + wi1 + wo
               2*B*D*cfg.D_moe * 3)                       # shared expert

    # Output head
    output_fwd = 2*B*D*cfg.V

    total_fwd = (mla_fwd * cfg.L + dense_fwd * cfg.L_dense +
                 moe_fwd * cfg.L_moe + output_fwd)
    # bwd = 2x fwd: each weight matmul has dX + dW, both at the same cost as fwd.
    # With gradient checkpointing: add 1 extra forward recompute pass.
    bwd_mult = 2.0
    recompute = 1.0 if getattr(cfg, 'gradient_checkpoint', False) else 0.0
    return total_fwd * (1 + bwd_mult + recompute)


# ============================================================================
# Training loop
# ============================================================================

def train(cfg: ModelConfig, shard_cfg: ShardConfig, args):
    """Main training loop."""
    print(f"{'='*70}")
    print(f"Mini DeepSeek-v3 Trainer: {cfg.name}")
    print(f"Devices: {jax.device_count()} x {jax.devices()[0].device_kind}")
    print(f"Config: D={cfg.D}, L={cfg.L} ({cfg.L_dense} dense + {cfg.L_moe} MoE), "
          f"H={cfg.H}, E={cfg.E}, K={cfg.K}")
    print(f"Backends: moe={cfg.moe_backend}, attn={cfg.attn_backend}")
    print(f"Optimizer: {args.optimizer}")
    print(f"GBS={args.gbs}, S={cfg.S}, dtype={cfg.dtype}")
    if cfg.gradient_checkpoint:
        print(f"Gradient checkpointing: ON")
    print(f"{'='*70}")

    # Create mesh and shard params
    mesh = shard_cfg.create_mesh()
    cfg.mesh = mesh
    print(f"Mesh: dp={shard_cfg.dp}, fsdp={shard_cfg.fsdp}, ep={shard_cfg.ep}, tp={shard_cfg.tp}")

    with mesh:
        # Init — create a dummy to get the sharding spec, then init sharded
        key = random.PRNGKey(42)
        key, init_key, data_key = random.split(key, 3)

        print("Initializing parameters (sharded)...")
        params = init_params(cfg, init_key, mesh=mesh)
        jax.block_until_ready(params)
        n_params = count_params(params)
        print(f"Parameters: {n_params:,} ({n_params * 2 / 1e9:.2f} GB in bf16)")

        # Optimizer state
        opt_state = None
        if args.optimizer == "adamw":
            opt_state = init_adam_state(params)
            print(f"Optimizer state: ~{n_params * 2 * 2 / 1e9:.2f} GB (m + v)")

        # Estimate FLOPs
        B = args.gbs                    # global batch size (sequences across all devices)
        tokens_per_step = B * cfg.S
        est_flops = estimate_flops(cfg, tokens_per_step)
        # With dp=1 FSDP, all devices process the same global batch together.
        # est_flops is total cluster FLOPs for tokens_per_step tokens.
        n_devices_now = jax.device_count()
        devices_per_chip_now = 2 if "7x" in jax.devices()[0].device_kind.lower() else 1
        n_chips_now = n_devices_now // devices_per_chip_now
        pdbs_now = B // n_devices_now   # per-device batch size (sequences per JAX device)
        print(f"Global batch: {tokens_per_step} tokens ({B} seq x {cfg.S} len)"
              f" | PDBS={pdbs_now} seq/device")
        print(f"Cluster: {n_devices_now} devices = {n_chips_now} chips")
        print(f"Estimated FLOPs/step (global): {est_flops/1e12:.2f} TFLOP"
              f" = {est_flops/n_chips_now/1e12:.3f} TFLOP/chip")

        # JIT compile the training step
        n_accum = args.grad_accum
        if n_accum > 1:
            print(f"Gradient accumulation: {n_accum} micro-batches")

        # compute_loss now returns (total, {'lm_loss', 'aux_loss'}) — pass
        # has_aux=True through value_and_grad and propagate the aux dict so
        # train_step reports LM and MoE-balance terms separately.
        # MaxText-style grad_accum: params threaded through scan CARRY, not
        # captured via closure. value_and_grad with explicit argnums.
        # See maxtext/utils/gradient_accumulation.py:25 for the reference
        # pattern. Working hypothesis: closure-capturing sharded params
        # confuses scan-vjp's residual-stacking, leading to NaN cotangents.
        # Putting params in the carry forces JAX to thread them explicitly.
        def _accum_body(carry, micro_tokens):
            acc_grads = carry["grads"]
            acc_loss = carry["loss"]
            acc_aux = carry["aux"]
            params_ = carry["params"]
            # value_and_grad with explicit argnums=0 (mirrors MaxText argnums=4)
            (loss_i, aux_i), grads_i = jax.value_and_grad(
                compute_loss, has_aux=True, argnums=0)(params_, micro_tokens, cfg)
            new_carry = {
                "params": params_,  # unchanged across iters
                "grads": jax.tree.map(jnp.add, acc_grads, grads_i),
                "loss": acc_loss + loss_i,
                "aux": jax.tree.map(jnp.add, acc_aux, aux_i),
            }
            return new_carry, None

        def _grad_accum_scan(params, tokens):
            micros = tokens.reshape(n_accum, B // n_accum, cfg.S)
            init = {
                "params": params,
                "grads": jax.tree.map(jnp.zeros_like, params),
                "loss": jnp.float32(0.0),
                "aux": {"lm_loss": jnp.float32(0.0),
                        "aux_loss": jnp.float32(0.0)},
            }
            final, _ = jax.lax.scan(_accum_body, init, micros, length=n_accum)
            return (final["loss"] / n_accum,
                    jax.tree.map(lambda x: x / n_accum, final["aux"]),
                    jax.tree.map(lambda g: g / n_accum, final["grads"]))

        # Shared grad-clip helper. Applies on BOTH optimizer paths so AdamW
        # gets the same clipping as SGD (was missing on AdamW path).
        def _maybe_clip_grads(grads):
            if args.grad_clip is None:
                return grads
            grad_norm = jnp.sqrt(sum(
                jnp.sum(g.astype(jnp.float32) ** 2)
                for g in jax.tree.leaves(grads)))
            jax.debug.print("grad_norm: {x}", x=grad_norm)
            # Per-top-level-key NaN check to localize the source sub-tree.
            # Set BWD_GRAD_FINITE_CHECK=1 to enable.
            if os.environ.get("BWD_GRAD_FINITE_CHECK", "").lower() in ("1", "true", "yes"):
                if isinstance(grads, dict):
                    for _k in sorted(grads.keys()):
                        _sub_leaves = jax.tree.leaves(grads[_k])
                        _any_nan = jnp.any(jnp.stack([
                            jnp.any(jnp.isnan(_l.astype(jnp.float32)))
                            for _l in _sub_leaves
                        ]))
                        _max_abs = jnp.max(jnp.stack([
                            jnp.max(jnp.abs(_l.astype(jnp.float32)))
                            for _l in _sub_leaves
                        ]))
                        jax.debug.print(
                            "[grad-check] " + _k + ": nan={n} max_abs={m}",
                            n=_any_nan, m=_max_abs)
            scale = jnp.minimum(1.0, args.grad_clip / (grad_norm + 1e-6))
            return jax.tree.map(lambda g: g * scale, grads)

        if args.optimizer == "adamw":
            @jax.jit
            def train_step(params, tokens, opt_state):
                if n_accum > 1:
                    loss, aux, grads = _grad_accum_scan(params, tokens)
                else:
                    (loss, aux), grads = jax.value_and_grad(
                        compute_loss, has_aux=True)(params, tokens, cfg)
                grads = _maybe_clip_grads(grads)
                params, opt_state = adam_step(
                    params, grads, opt_state, lr=args.lr)
                return params, loss, aux, opt_state
        else:
            @jax.jit
            def train_step(params, tokens):
                if n_accum > 1:
                    loss, aux, grads = _grad_accum_scan(params, tokens)
                else:
                    (loss, aux), grads = jax.value_and_grad(
                        compute_loss, has_aux=True)(params, tokens, cfg)
                grads = _maybe_clip_grads(grads)
                params = jax.tree.map(
                    lambda p, g: (p.astype(jnp.float32) - args.lr * g.astype(jnp.float32)).astype(p.dtype),
                    params, grads)
                return params, loss, aux

        # Synthetic data
        def make_batch(key):
            return random.randint(key, (B, cfg.S), 0, cfg.V)

        # Free any stale HBM from parameter init before loading the compiled binary.
        # Without this, fragmentation can block the program's contiguous HBM reservation.
        gc.collect()
        jax.effects_barrier()

        # Warmup (compilation)
        print("Compiling (first step)...")
        tokens = make_batch(data_key)
        t0 = time.time()
        print("  [DBG] calling train_step (XLA compilation starts)", flush=True)
        try:
            if args.optimizer == "adamw":
                params, loss, aux, opt_state = train_step(params, tokens, opt_state)
            else:
                params, loss, aux = train_step(params, tokens)
            print("  [DBG] train_step returned (compilation done, awaiting exec)", flush=True)
            loss.block_until_ready()
            print("  [DBG] block_until_ready done", flush=True)
        except Exception as e:
            import traceback
            print(f"  [DBG] EXCEPTION: {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
            raise
        compile_time = time.time() - t0
        print(f"Compilation: {compile_time:.1f}s, initial loss: {loss:.3f} "
              f"(lm={float(aux['lm_loss']):.3f}, aux={float(aux['aux_loss']):.4f})")

        # Profiling setup
        profile_active = False
        if args.profile:
            print(f"Will profile steps {args.profile_skip+1}-"
                  f"{args.profile_skip+args.profile_steps} → {args.profile_dir}",
                  flush=True)

        # Training
        print(f"\nTraining for {args.steps} steps...")
        for step in range(args.steps):
            step_key = random.fold_in(data_key, step + 1)
            tokens = make_batch(step_key)

            # Start profiling (rank 0 only to avoid 64× profile overhead on cluster).
            # Pass gs:// path directly; stop_trace will flush to GCS in-line.
            if args.profile and step == args.profile_skip and jax.process_index() == 0:
                jax.profiler.start_trace(args.profile_dir)
                profile_active = True

            t0 = time.time()
            if args.optimizer == "adamw":
                params, loss, aux, opt_state = train_step(params, tokens, opt_state)
            else:
                params, loss, aux = train_step(params, tokens)
            loss.block_until_ready()
            step_time = time.time() - t0

            # Stop profiling — flushes directly to GCS (~10-30s, one-time cost).
            if profile_active and step == args.profile_skip + args.profile_steps:
                print(f"  Flushing profile to {args.profile_dir} ...", flush=True)
                jax.profiler.stop_trace()
                profile_active = False
                print(f"  Profile flushed to {args.profile_dir}", flush=True)

            # Logging
            # est_flops and tokens_per_step are GLOBAL (whole cluster) metrics,
            # since dp=1 means one batch processed across all FSDP devices.
            # v7x has 2 JAX devices per physical chip; device_kind helps identify.
            n_devices = jax.device_count()
            device_kind = jax.devices()[0].device_kind
            # v7x (device_kind="TPU7x"): 2 JAX devices per chip; others default to 1
            devices_per_chip = 2 if "7x" in device_kind.lower() else 1
            n_chips = n_devices // devices_per_chip

            cluster_tflop_s = est_flops / step_time / 1e12  # total cluster
            cluster_tps = tokens_per_step / step_time         # total cluster tokens/s
            chip_tflop_s = cluster_tflop_s / n_chips
            chip_tps = cluster_tps / n_chips

            # MFU: chip_tflop_s / peak_per_chip.  v7x peak = 2307 TFLOP/s bf16.
            PEAK_PER_CHIP = {"7x": 2307.0, "v6e": 918.0, "v5e": 393.0, "v4": 275.0}
            peak = next((v for k, v in PEAK_PER_CHIP.items() if k in device_kind.lower()), None)
            mfu_str = f"{chip_tflop_s / peak * 100:.1f}%" if peak else "?"

            print(f"completed step: {step+1}, seconds: {step_time:.3f}, "
                  f"TFLOP/s/chip: {chip_tflop_s:.1f}, "
                  f"cluster_TFLOP/s: {cluster_tflop_s:.1f}, "
                  f"MFU: {mfu_str}, "
                  f"TPS/chip: {chip_tps:.1f}, "
                  f"cluster_TPS: {cluster_tps:.0f}, "
                  f"loss: {loss:.3f} "
                  f"(lm={float(aux['lm_loss']):.3f}, aux={float(aux['aux_loss']):.4f})")

        # Final summary
        print(f"\n{'='*70}")
        print(f"Total TFLOPs: {est_flops * args.steps / 1e12:.2f}")
        if profile_active:
            # Fallback: profile_steps wasn't reached (steps < profile_skip+profile_steps)
            print(f"  Flushing partial profile to {args.profile_dir} ...", flush=True)
            jax.profiler.stop_trace()
            profile_active = False
            print(f"  Profile flushed to {args.profile_dir}", flush=True)
        print(f"{'='*70}")


# ============================================================================
# Roofline estimation
# ============================================================================

def run_roofline(cfg: ModelConfig, args):
    """Run JAX experimental roofline analysis."""
    print(f"Running roofline analysis for {cfg.name}...")

    try:
        from jax.experimental import roofline as jax_roofline
    except ImportError:
        print("jax.experimental.roofline not available in this JAX version")
        return

    B = args.gbs
    tokens = jnp.ones((B, cfg.S), dtype=jnp.int32)

    key = random.PRNGKey(0)
    params = init_params(cfg, key)

    print("Computing roofline for forward pass...")
    try:
        result = jax_roofline.roofline(
            compute_loss,
            in_specs=(None, None, None),
        )(params, tokens, cfg)
        print(f"Roofline result: {result}")
    except Exception as e:
        print(f"Roofline API error: {e}")
        print("Falling back to manual FLOP estimation...")
        est = estimate_flops(cfg, B * cfg.S)
        print(f"Manual estimate: {est/1e12:.2f} TFLOP per step (fwd+bwd)")


# ============================================================================
# Main
# ============================================================================

def main():
    # JAX numerical-debug flags MUST be set before any JIT compilation.
    # Read straight from env so users can pass via env injection too.
    import os as _os
    if _os.environ.get("JAX_DEBUG_NANS", "").lower() in ("1", "true", "yes"):
        jax.config.update("jax_debug_nans", True)
        print("⚠️  jax_debug_nans=True — JIT'd ops will re-execute in de-optimized "
              "mode on first NaN to pinpoint the producing op. Slow.", flush=True)
    if _os.environ.get("JAX_DEBUG_INFS", "").lower() in ("1", "true", "yes"):
        jax.config.update("jax_debug_infs", True)
        print("⚠️  jax_debug_infs=True", flush=True)

    # Multi-host init (no-op on single host)
    jax.distributed.initialize(initialization_timeout=600)  # 10 min for large slices

    parser = argparse.ArgumentParser(description="Mini DeepSeek-v3 trainer")
    parser.add_argument("--config", default="full", choices=CONFIGS.keys())
    parser.add_argument("--gbs", type=int, default=1,
                        help="Global batch size (total sequences across all devices)")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--mesh", default=None,
                        help="Mesh topology e.g. 4x4x4")
    parser.add_argument("--fsdp", type=int, default=1)
    parser.add_argument("--ep", type=int, default=1)
    parser.add_argument("--tp", type=int, default=1,
                        help="Tensor parallel size (2 = both TCs per v7x chip)")
    parser.add_argument("--moe_backend", default=None,
                        choices=["jax", "ragged_dot", "gmm", "gmm_ag", "fused_ep_moe_v4", "fused_ep_moe_v4_jax_fwd", "pfwd_jbwd"])
    parser.add_argument("--attn_backend", default=None,
                        choices=["jax", "splash"])
    parser.add_argument("--optimizer", default="sgd",
                        choices=["sgd", "adamw"])
    parser.add_argument("--gradient_checkpoint", action="store_true")
    parser.add_argument("--grad_clip", type=float, default=None,
                        help="Global gradient norm clip (e.g. 1.0)")
    parser.add_argument("--grad_accum", type=int, default=1,
                        help="Gradient accumulation micro-batches (GBS must be divisible)")
    parser.add_argument("--aux_loss_weight", type=float, default=None)
    parser.add_argument("--moe_aux_loss_coeff", type=float, default=None,
                        help="Override cfg.moe_aux_loss_coeff (DSv3 default: 1e-4). "
                             "Pass 0 to disable the load-balance aux loss for "
                             "bwd-NaN bisection.")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--profile_dir", default="/tmp/mini_dsv3_profile")
    parser.add_argument("--profile_skip", type=int, default=2,
                        help="Steps before profiling")
    parser.add_argument("--profile_steps", type=int, default=1)
    parser.add_argument("--roofline", action="store_true",
                        help="Run JAX roofline analysis")
    parser.add_argument("--dtype", default=None,
                        choices=["bfloat16", "float32"])
    parser.add_argument("--no_cp", action="store_true",
                        help="Disable context parallelism (no sequence sharding on EP).")
    parser.add_argument("--moe_xlayer_prefetch", action="store_true",
                        help="Cross-layer FSDP weight AG prefetch (gmm_ag backend only, "
                             "uses pcast/reduced semantics for clean bwd).")
    parser.add_argument("--moe_use_sc_scatter", action="store_true",
                        help="Use SC gather-reduce kernel for the per-chunk scatter "
                             "(v305 — slower in practice ~5%% TPS regression at our "
                             "chunk freq, kept for experimentation).")
    parser.add_argument("--moe_use_gmm_v2", action="store_true",
                        help="Use Pallas gmm_v2 (with fused gate+up+silu) for the 3 "
                             "ragged-dots per chunk. Forward via gmm_v2, backward via "
                             "jax.vjp on jax.lax.ragged_dot reference (Stage A.1).")
    parser.add_argument("--moe_n_chunks", type=int, default=2,
                        help="Token chunking inside MoE body. 1=no chunking; 2=default. "
                             "With faster gmm_v2 forward, n=1 may pipeline AG/RS better "
                             "via cross-layer overlap rather than per-chunk overlap.")
    parser.add_argument("--moe_shard_e_with_fsdp", action="store_true",
                        help="Shard MoE weight E axis across (ep,fsdp); AG along axis 0 "
                             "instead of along F (axis 2 for wi, axis 1 for wo). Test if "
                             "AG-along-leading-dim avoids the layout-copy multiplication.")
    parser.add_argument("--moe_shard_d_with_fsdp", action="store_true",
                        help="Shard MoE weight D=7168 axis with FSDP (instead of F=2048). "
                             "Wider per-chip stripe (14 vs 4 at FSDP=512); same AG peak. "
                             "Compute pattern unchanged.")
    parser.add_argument("--moe_fp8_weights", action="store_true",
                        help="Cast wi_0/wi_1/wo to fp8_e4m3fn before the FSDP all-gather "
                             "in the gmm_ag MoE body. Halves the per-layer 7 GB AG "
                             "allocation. Activations stay bf16; ragged_dot uses "
                             "preferred_element_type=bf16. No per-channel scale "
                             "(memory experiment first; expect some loss elevation).")
    parser.add_argument("--moe_no_weight_ag", action="store_true",
                        help="EP>1+TP>1 only. Use colrow+A2A body that keeps weights "
                             "F-fsdp-sharded throughout — psum on TP after wi, psum on "
                             "FSDP after wo. Removes the per-layer weight AllGather "
                             "transient (~1.4 GB at EP=8/TP=2). Pairs with --moe_backend=gmm.")
    parser.add_argument("--moe_debug_nans", action="store_true",
                        help="Insert jax.debug.print finite-checks at strategic points "
                             "in the gmm_ag MoE body (post-AG, post-sort, post-ragged_dot×3, "
                             "post-scatter, post-psum_scatter). Logs NaN/Inf/max-abs per "
                             "chunk with ordered=True so the first non-finite tensor is "
                             "easy to find in kubectl logs. Slow due to host-device sync.")
    args = parser.parse_args()

    cfg = CONFIGS[args.config]()

    if args.moe_backend:
        cfg.moe_backend = args.moe_backend
    if args.attn_backend:
        cfg.attn_backend = args.attn_backend
    if args.dtype:
        cfg.dtype = args.dtype
    if args.aux_loss_weight is not None:
        cfg.aux_loss_weight = args.aux_loss_weight
    if args.moe_aux_loss_coeff is not None:
        cfg.moe_aux_loss_coeff = args.moe_aux_loss_coeff
    if args.gradient_checkpoint:
        cfg.gradient_checkpoint = True
    if args.no_cp:
        cfg.use_cp = False
    if args.moe_xlayer_prefetch:
        cfg.moe_xlayer_prefetch = True
    if args.moe_use_sc_scatter:
        cfg.moe_use_sc_scatter = True
    if args.moe_use_gmm_v2:
        cfg.moe_use_gmm_v2 = True
    cfg.moe_n_chunks = args.moe_n_chunks
    if args.moe_shard_e_with_fsdp:
        cfg.moe_shard_e_with_fsdp = True
        from . import model as _model
        _model._SHARD_E_WITH_FSDP = True
    if args.moe_shard_d_with_fsdp:
        from . import model as _model
        _model._SHARD_D_WITH_FSDP = True
    if args.moe_fp8_weights:
        cfg.moe_fp8_weights = True
    if args.moe_no_weight_ag:
        cfg.moe_no_weight_ag = True
    if args.moe_debug_nans:
        cfg.moe_debug_nans = True
        from . import model as _model
        _model._MOE_NO_WEIGHT_AG = True

    shard_cfg = ShardConfig(fsdp=args.fsdp, ep=args.ep, tp=args.tp,
                            explicit_axes=False)  # AG path uses an inline Explicit mesh

    if args.roofline:
        run_roofline(cfg, args)
    else:
        train(cfg, shard_cfg, args)


if __name__ == "__main__":
    main()
