# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""fused_ep_moe backward pass — custom_vjp wrapper.

Development stages:
  Stage A (this file):  Pure JAX backward math, ref_moe_with_residuals forward.
                        No Pallas, no EP. Correctness oracle — all 5 tests PASS.
  Stage B (v2 below):   fused_ep_moe (Pallas) forward + JAX backward.
                        Residuals still from ref_moe_with_residuals. EP=1 only.
  Stage C (TODO):       Store h_gate/h_up/sorted_tokens as HBM residuals inside
                        fused_ep_moe; replace JAX backward with Pallas kernels;
                        add EP A2A in backward. Requires kernel modification.

Usage:
  from backward import make_fused_ep_moe_train
  fn = make_fused_ep_moe_train(top_k=8, scoring_fn="sigmoid",
                                renormalize_topk_logits=True)
  out = fn(tokens, w1, w2, gating_output)
  loss = out.sum()
  grads = jax.grad(lambda t, w1, w2, g: fn(t, w1, w2, g).sum())(...)
"""

import functools

import jax
import jax.numpy as jnp
from jax import lax

# ---------------------------------------------------------------------------
# Activation helpers
# ---------------------------------------------------------------------------

def silu(x):
    return jax.nn.silu(x)


def silu_grad(x):
    """d/dx silu(x) = sigmoid(x) * (1 + x*(1 - sigmoid(x)))."""
    sig = jax.nn.sigmoid(x)
    return sig * (1.0 + x * (1.0 - sig))


def apply_scoring_fn(scoring_fn: str, x):
    if scoring_fn == "softmax":
        return jax.nn.softmax(x, axis=-1)
    elif scoring_fn == "sigmoid":
        return jax.nn.sigmoid(x)
    else:
        raise NotImplementedError(f"Unknown scoring_fn: {scoring_fn}")


def apply_scoring_fn_grad(scoring_fn: str, x, d_scores):
    """Backward through apply_scoring_fn.

    Returns d_gating_output (same shape as x).
    """
    if scoring_fn == "softmax":
        # d_x = softmax(x) * (d_scores - sum(d_scores * softmax(x)))
        s = jax.nn.softmax(x, axis=-1)
        return s * (d_scores - jnp.sum(d_scores * s, axis=-1, keepdims=True))
    elif scoring_fn == "sigmoid":
        sig = jax.nn.sigmoid(x)
        return d_scores * sig * (1.0 - sig)
    else:
        raise NotImplementedError(f"Unknown scoring_fn: {scoring_fn}")


# ---------------------------------------------------------------------------
# ref_moe_with_residuals
# Runs the reference forward and returns intermediates needed for the backward.
# ---------------------------------------------------------------------------

def ref_moe_with_residuals(
        tokens,         # (T, D)  float32
        w1,             # (E, 2, D, F)   F = intermediate_size = D_moe//2
        w2,             # (E, F, D)
        gating_output,  # (T, E)
        top_k: int,
        *,
        renormalize_topk_logits: bool = False,
        scoring_fn: str = "softmax",
        act_fn: str = "silu",
):
    """Forward pass returning (output, residuals).

    residuals = (
        h_gate,        # (T, K, F)  gate projection outputs (pre-activation)
        h_up,          # (T, K, F)  up projection outputs
        expert_outs,   # (T, K, D)  per-(token,expert) FFN outputs
        top_k_indices, # (T, K)     selected expert ids
        top_k_logits,  # (T, K)     gating weights (post renorm)
        gating_scores, # (T, E)     post-scoring logits (pre top-k)
    )
    """
    T, D = tokens.shape
    E, _, _, F = w1.shape  # E experts, 2 (gate/up), D hidden, F intermediate

    # Scoring
    gating_scores = apply_scoring_fn(scoring_fn, gating_output)  # (T, E)

    # Top-K selection
    top_k_logits, top_k_indices = lax.top_k(gating_scores, top_k)  # (T,K),(T,K)

    if renormalize_topk_logits:
        top_k_logits = top_k_logits / jnp.sum(
            top_k_logits, axis=-1, keepdims=True)

    # Vectorised forward: for each (token, expert-slot) pair compute FFN output.
    # Flatten to (T*K,) pairs, gather weights, vmap over pairs.
    expert_ids_flat = top_k_indices.reshape(-1)              # (T*K,)
    tokens_flat = jnp.tile(
        tokens[:, None, :], (1, top_k, 1)).reshape(T * top_k, D)  # (T*K, D)

    w1_gate_pairs = w1[expert_ids_flat, 0]  # (T*K, D, F)
    w1_up_pairs   = w1[expert_ids_flat, 1]  # (T*K, D, F)
    w2_pairs      = w2[expert_ids_flat]     # (T*K, F, D)

    def ffn_one(token, wg, wu, wd):
        """Forward for a single (token, expert) pair. Returns (h_gate, h_up, out)."""
        h_g = token @ wg          # (F,)
        h_u = token @ wu          # (F,)
        if act_fn == "silu":
            act = silu(h_g) * h_u
        elif act_fn == "gelu":
            act = jax.nn.gelu(h_g) * h_u
        else:
            raise NotImplementedError(act_fn)
        out = act @ wd             # (D,)
        return h_g, h_u, out

    # vmap over the T*K pairs
    h_gate_flat, h_up_flat, expert_outs_flat = jax.vmap(ffn_one)(
        tokens_flat, w1_gate_pairs, w1_up_pairs, w2_pairs)
    # shapes: (T*K, F), (T*K, F), (T*K, D)

    # Reshape residuals to (T, K, ...)
    h_gate      = h_gate_flat.reshape(T, top_k, F)       # (T, K, F)
    h_up        = h_up_flat.reshape(T, top_k, F)         # (T, K, F)
    expert_outs = expert_outs_flat.reshape(T, top_k, D)  # (T, K, D)

    # Weighted combine: output[t] = sum_k( weight[t,k] * expert_out[t,k] )
    output = jnp.sum(
        top_k_logits[:, :, None] * expert_outs, axis=1)  # (T, D)
    output = output.astype(tokens.dtype)

    residuals = (h_gate, h_up, expert_outs, top_k_indices,
                 top_k_logits, gating_scores)
    return output, residuals


# ---------------------------------------------------------------------------
# Backward math — pure JAX
# ---------------------------------------------------------------------------

def moe_bwd_combine(d_output, top_k_logits, expert_outs):
    """Backward through the weighted combine step.

    Forward:  output[t] = sum_k( w[t,k] * expert_out[t,k] )

    Returns:
      d_expert_outs  (T, K, D)
      d_top_k_logits (T, K)
    """
    # d_expert_outs[t,k,:] = d_output[t,:] * top_k_logits[t,k]
    d_expert_outs = d_output[:, None, :] * top_k_logits[:, :, None]  # (T,K,D)
    # d_top_k_logits[t,k] = dot(d_output[t], expert_outs[t,k])
    d_top_k_logits = jnp.sum(d_output[:, None, :] * expert_outs, axis=-1)  # (T,K)
    return d_expert_outs, d_top_k_logits


def moe_bwd_ffn(d_expert_outs, tokens, w1, w2, top_k_indices, h_gate, h_up,
                act_fn: str = "silu"):
    """Backward through the expert FFN for all (token, expert) pairs.

    Returns:
      d_tokens  (T, D)
      d_w1      (E, 2, D, F)   stacked [d_w1_gate, d_w1_up]
      d_w2      (E, F, D)
    """
    T, top_k, D = d_expert_outs.shape
    E, _, _, F = w1.shape

    expert_ids_flat = top_k_indices.reshape(-1)          # (T*K,)
    tokens_flat = jnp.tile(
        tokens[:, None, :], (1, top_k, 1)).reshape(T * top_k, D)

    d_expert_outs_flat = d_expert_outs.reshape(T * top_k, D)
    h_gate_flat = h_gate.reshape(T * top_k, F)
    h_up_flat   = h_up.reshape(T * top_k, F)

    w1_gate_pairs = w1[expert_ids_flat, 0]  # (T*K, D, F)
    w1_up_pairs   = w1[expert_ids_flat, 1]  # (T*K, D, F)
    w2_pairs      = w2[expert_ids_flat]     # (T*K, F, D)

    def bwd_one(token, wg, wu, wd, hg, hu, d_out):
        """Backward for one (token, expert) pair."""
        # hidden = act(h_gate) * h_up
        if act_fn == "silu":
            hidden = silu(hg) * hu                   # (F,)
            d_hidden = d_out @ wd.T                  # (F,)
            d_hu   = d_hidden * silu(hg)             # (F,)
            d_hg   = d_hidden * hu * silu_grad(hg)  # (F,)
        elif act_fn == "gelu":
            hidden = jax.nn.gelu(hg) * hu
            d_hidden = d_out @ wd.T
            d_hu   = d_hidden * jax.nn.gelu(hg)
            # gelu_grad via JAX autodiff of scalar
            d_hg   = d_hidden * hu * jax.grad(jax.nn.gelu)(hg)
        else:
            raise NotImplementedError(act_fn)

        # d_W2 = hidden.T @ d_out  → outer product for 1-D vectors
        d_wd = hidden[:, None] * d_out[None, :]      # (F, D)
        # d_W1_gate, d_W1_up
        d_wg = token[:, None] * d_hg[None, :]        # (D, F)
        d_wu = token[:, None] * d_hu[None, :]        # (D, F)
        # d_token
        d_tok = d_hg @ wg.T + d_hu @ wu.T           # (D,)
        return d_tok, d_wg, d_wu, d_wd

    d_tok_flat, d_wg_flat, d_wu_flat, d_wd_flat = jax.vmap(bwd_one)(
        tokens_flat, w1_gate_pairs, w1_up_pairs, w2_pairs,
        h_gate_flat, h_up_flat, d_expert_outs_flat)
    # shapes: (T*K, D), (T*K, D, F), (T*K, D, F), (T*K, F, D)

    # Accumulate token gradients: sum over K slots per token
    d_tokens = d_tok_flat.reshape(T, top_k, D).sum(axis=1)  # (T, D)

    # Accumulate weight gradients: segment_sum over expert ids
    d_w1_gate = jax.ops.segment_sum(
        d_wg_flat, expert_ids_flat, num_segments=E)  # (E, D, F)
    d_w1_up = jax.ops.segment_sum(
        d_wu_flat, expert_ids_flat, num_segments=E)  # (E, D, F)
    d_w2 = jax.ops.segment_sum(
        d_wd_flat, expert_ids_flat, num_segments=E)  # (E, F, D)

    d_w1 = jnp.stack([d_w1_gate, d_w1_up], axis=1)  # (E, 2, D, F)
    return d_tokens, d_w1, d_w2


def moe_bwd_routing(d_top_k_logits, top_k_indices, top_k_logits,
                     gating_scores, gating_output,
                     scoring_fn: str = "softmax",
                     renormalize_topk_logits: bool = False):
    """Backward through routing: top_k + optional renorm + scoring.

    Returns d_gating_output (T, E).
    """
    T, K = top_k_indices.shape
    E = gating_output.shape[-1]

    d_logits = d_top_k_logits  # (T, K) — grad wrt top_k_logits

    # --- Backward through renormalization ---
    if renormalize_topk_logits:
        # Forward: logits_renorm = logits / sum(logits)
        # This is equivalent to softmax without exp — just L1-normalize.
        # Let s = sum(raw_logits), then renorm[k] = raw[k] / s
        # d_raw[k] = d_renorm[k] / s - raw[k] / s^2 * sum(d_renorm)
        #           = (d_renorm[k] - renorm[k] * sum(d_renorm)) / s
        # We recover raw from renorm * sum = renorm * s, but we stored top_k_logits
        # as the renormalized values. We need the pre-renorm sum.
        # Since renorm = raw/sum(raw) and gating_scores are the raw values:
        raw_logits = gating_scores[
            jnp.arange(T)[:, None],
            top_k_indices]  # (T, K) — pre-renorm top-k scores
        s = jnp.sum(raw_logits, axis=-1, keepdims=True)  # (T, 1)
        d_logits = (d_logits - jnp.sum(d_logits * top_k_logits, axis=-1,
                                       keepdims=True)) / s  # (T, K)

    # --- Backward through top_k (scatter from (T,K) to (T,E)) ---
    # top_k is not differentiable through indices, but we pass grads to the
    # selected positions.
    d_gating_scores = jnp.zeros((T, E), dtype=d_logits.dtype)
    # scatter-add: d_gating_scores[t, top_k_indices[t, k]] += d_logits[t, k]
    t_idx = jnp.arange(T)[:, None] * jnp.ones((1, K), dtype=jnp.int32)
    d_gating_scores = d_gating_scores.at[
        t_idx.reshape(-1), top_k_indices.reshape(-1)
    ].add(d_logits.reshape(-1))  # (T, E)

    # --- Backward through scoring function ---
    d_gating_output = apply_scoring_fn_grad(
        scoring_fn, gating_output, d_gating_scores)  # (T, E)

    return d_gating_output


# ---------------------------------------------------------------------------
# custom_vjp wrapper
# ---------------------------------------------------------------------------

def make_fused_ep_moe_train(
        top_k: int,
        *,
        renormalize_topk_logits: bool = False,
        scoring_fn: str = "softmax",
        act_fn: str = "silu",
        # Stage B/C: mesh and ep_axis_name for EP sharding
        mesh=None,
        ep_axis_name: str = "ep",
):
    """Return a custom_vjp-wrapped MoE forward.

    Static config (top_k, scoring_fn, etc.) is captured via closure so that
    jax.custom_vjp only sees differentiable array inputs.

    Args:
      top_k: Number of experts per token.
      renormalize_topk_logits: Whether to L1-normalize top-k weights.
      scoring_fn: "softmax" or "sigmoid".
      act_fn: "silu" or "gelu".
      mesh: JAX Mesh for EP sharding (Stage B+, None for single-device).
      ep_axis_name: Mesh axis name for expert parallelism.

    Returns:
      fused_ep_moe_train(tokens, w1, w2, gating_output) -> output
    """
    # Capture config in closure
    cfg = dict(
        top_k=top_k,
        renormalize_topk_logits=renormalize_topk_logits,
        scoring_fn=scoring_fn,
        act_fn=act_fn,
    )

    @jax.custom_vjp
    def fused_ep_moe_train(tokens, w1, w2, gating_output):
        """MoE forward. Differentiable w.r.t. tokens, w1, w2, gating_output."""
        if mesh is None:
            # Stage A: single device, use ref_moe_with_residuals forward only
            out, _ = ref_moe_with_residuals(
                tokens.astype(jnp.float32), w1, w2, gating_output,
                **cfg)
            return out.astype(tokens.dtype)
        else:
            # Stage B+: call fused_ep_moe (to be implemented)
            raise NotImplementedError("EP mode not yet implemented in Stage A")

    def _fwd(tokens, w1, w2, gating_output):
        tokens_f32 = tokens.astype(jnp.float32)
        w1_f32 = w1.astype(jnp.float32)
        w2_f32 = w2.astype(jnp.float32)
        gating_f32 = gating_output.astype(jnp.float32)

        out, residuals = ref_moe_with_residuals(
            tokens_f32, w1_f32, w2_f32, gating_f32, **cfg)
        out = out.astype(tokens.dtype)

        # Store original dtype tensors alongside residuals for backward
        saved = (tokens_f32, w1_f32, w2_f32, gating_f32) + residuals
        return out, saved

    def _bwd(saved, d_out):
        (tokens_f32, w1_f32, w2_f32, gating_f32,
         h_gate, h_up, expert_outs, top_k_indices,
         top_k_logits, gating_scores) = saved

        d_out_f32 = d_out.astype(jnp.float32)

        # 1. Backward through combine
        d_expert_outs, d_top_k_logits = moe_bwd_combine(
            d_out_f32, top_k_logits, expert_outs)

        # 2. Backward through expert FFNs
        d_tokens, d_w1, d_w2 = moe_bwd_ffn(
            d_expert_outs, tokens_f32, w1_f32, w2_f32,
            top_k_indices, h_gate, h_up,
            act_fn=cfg["act_fn"])

        # 3. Backward through routing
        d_gating = moe_bwd_routing(
            d_top_k_logits, top_k_indices, top_k_logits,
            gating_scores, gating_f32,
            scoring_fn=cfg["scoring_fn"],
            renormalize_topk_logits=cfg["renormalize_topk_logits"])

        # Cast back to original dtype
        d_tokens = d_tokens.astype(tokens_f32.dtype)

        return (d_tokens, d_w1, d_w2, d_gating)

    fused_ep_moe_train.defvjp(_fwd, _bwd)
    return fused_ep_moe_train


# ---------------------------------------------------------------------------
# Stage C: fused_ep_moe (Pallas) forward + Pallas backward (EP=1)
# ---------------------------------------------------------------------------

def make_fused_ep_moe_train_v3(
        mesh,
        top_k: int,
        *,
        renormalize_topk_logits: bool = False,
        scoring_fn: str = "softmax",
        act_fn: str = "silu",
        ep_axis_name: str = "model",
        bte: int | None = None,
        tile_D: int | None = None,
        vmem_limit_bytes: int = 16 * 1024 * 1024,
):
    """Stage C/D: Pallas forward + Pallas backward (EP=1 and EP>1).

    Forward uses fused_ep_moe (same as Stage B).
    Backward uses fused_ep_moe_bwd (Pallas kernel):
      - Pre-sorts tokens by expert in JAX (O(T*K) preprocessing)
      - Per-expert batched matmuls in VMEM (no vmap over T*K pairs)
      - Activation checkpointing: h_gate/h_up recomputed in backward
      - EP>1: each device processes its local experts inside shard_map;
        d_tokens and d_gating are reduced via lax.psum across the EP axis.

    Args:
      mesh: 2D JAX mesh (same as Stage B).
      bte: Token block size for backward kernel (default 128).
    """
    from jax.sharding import PartitionSpec as P
    shard_map = jax.shard_map
    from tpu_inference.kernels.fused_moe.v1.kernel import fused_ep_moe
    from backward_kernel import fused_ep_moe_bwd

    EP = mesh.shape[ep_axis_name]

    @jax.custom_vjp
    def fused_ep_moe_train(tokens, w1, w2, gating_output):
        return fused_ep_moe(
            mesh, tokens, w1, w2, gating_output,
            top_k=top_k,
            renormalize_topk_logits=renormalize_topk_logits,
            scoring_fn=scoring_fn,
            act_fn=act_fn,
            ep_axis_name=ep_axis_name,
        )

    def _fwd(tokens, w1, w2, gating_output):
        tokens_f32 = tokens.astype(jnp.float32)
        w1_f32     = w1.astype(jnp.float32)
        w2_f32     = w2.astype(jnp.float32)
        gating_f32 = gating_output.astype(jnp.float32)

        out = fused_ep_moe(
            mesh, tokens_f32, w1_f32, w2_f32, gating_f32,
            top_k=top_k,
            renormalize_topk_logits=renormalize_topk_logits,
            scoring_fn=scoring_fn,
            act_fn=act_fn,
            ep_axis_name=ep_axis_name,
        )
        out = out.astype(tokens.dtype)
        saved = (tokens_f32, w1_f32, w2_f32, gating_f32)
        return out, saved

    def _bwd(saved, d_out):
        tokens_f32, w1_f32, w2_f32, gating_f32 = saved
        d_out_f32 = d_out.astype(jnp.float32)

        bwd_kwargs = dict(
            top_k=top_k,
            scoring_fn=scoring_fn,
            renormalize_topk_logits=renormalize_topk_logits,
            act_fn=act_fn,
            ep_axis_name=ep_axis_name,
            bte=bte,
            tile_D=tile_D,
            vmem_limit_bytes=vmem_limit_bytes,
        )

        if EP == 1:
            d_tokens, d_w1, d_w2, d_gating = fused_ep_moe_bwd(
                d_out_f32, tokens_f32, w1_f32, w2_f32, gating_f32, **bwd_kwargs)
        else:
            def _local_bwd(d_out_rep, tokens_rep, w1_local, w2_local, gating_rep):
                d_tok_partial, d_w1_local, d_w2_local, d_gate_partial = fused_ep_moe_bwd(
                    d_out_rep, tokens_rep, w1_local, w2_local, gating_rep, **bwd_kwargs)
                # All-reduce partial contributions across EP devices.
                d_tok   = lax.psum(d_tok_partial,   axis_name=ep_axis_name)
                d_gate  = lax.psum(d_gate_partial,  axis_name=ep_axis_name)
                return d_tok, d_w1_local, d_w2_local, d_gate

            d_tokens, d_w1, d_w2, d_gating = shard_map(
                _local_bwd,
                mesh=mesh,
                in_specs=(
                    P(),              # d_out   — replicated
                    P(),              # tokens  — replicated
                    P(ep_axis_name),  # w1      — sharded on expert axis
                    P(ep_axis_name),  # w2      — sharded on expert axis
                    P(),              # gating  — replicated
                ),
                out_specs=(
                    P(),              # d_tokens  — replicated (after psum)
                    P(ep_axis_name),  # d_w1      — sharded (each device owns its shard)
                    P(ep_axis_name),  # d_w2      — sharded
                    P(),              # d_gating  — replicated (after psum)
                ),
                check_vma=False,
            )(d_out_f32, tokens_f32, w1_f32, w2_f32, gating_f32)

        d_tokens = d_tokens.astype(tokens_f32.dtype)
        return (d_tokens, d_w1, d_w2, d_gating)

    fused_ep_moe_train.defvjp(_fwd, _bwd)
    return fused_ep_moe_train


# ---------------------------------------------------------------------------
# Stage 1: Pallas forward + per-expert streaming JAX backward
# ---------------------------------------------------------------------------


def make_fused_ep_moe_train_v4(
        mesh,
        top_k: int,
        *,
        renormalize_topk_logits: bool = False,
        scoring_fn: str = "softmax",
        act_fn: str = "silu",
        ep_axis_name: str = "model",
        max_tpe: int | None = None,
        fsdp_axis_name: str | None = None,
):
    """Stage 1: Pallas forward + per-expert streaming JAX backward.

    Avoids T*K*D materialization — scales to full DSv3 671B
    (T=524K, K=8, EP=8, D=7168).

    Forward uses fused_ep_moe (same Pallas kernel as v3).
    Backward uses fused_ep_moe_bwd_streaming:
      - Keeps only 1D bin-index arrays (~33 MB total).
      - Loops over E_local experts; per expert gathers tokens on-the-fly
        from original HBM (~470 MB/expert, never all at once).
      - Eliminates the bins_tokens (120 GB) and bins_d_exp (120 GB) buffers
        and the vmap weight gather (247 TB) that OOM in fused_ep_moe_bwd.
      - EP>1: same shard_map structure as v3; d_tokens and d_gating are
        psum'd across the EP axis by the caller.
      - FSDP>1 (fsdp_axis_name set): correct VJP of forward FSDP psum via
        psum(g, "fsdp") before EP all_gather; weights accepted FSDP-sharded.

    Args:
      mesh: JAX Mesh with ep_axis_name axis for expert parallelism.
      top_k: Number of experts per token.
      renormalize_topk_logits: L1-normalize top-k weights.
      scoring_fn: "softmax" or "sigmoid".
      act_fn: "silu" or "gelu".
      ep_axis_name: Mesh axis name for EP (default "model").
      max_tpe: Override the static max-tokens-per-expert upper bound.
               Useful for testing; defaults to auto-computed value.
      fsdp_axis_name: Mesh axis name for FSDP weight sharding.
               When set, activations use P((ep_axis_name, fsdp_axis_name), None)
               and weights use P(ep_axis_name, None, None, fsdp_axis_name) /
               P(ep_axis_name, fsdp_axis_name, None). The backward applies
               psum(g, fsdp_axis_name) before the EP all_gather — the correct
               VJP of the forward's psum(fsdp) reduction.
    """
    from jax.sharding import PartitionSpec as P
    shard_map = jax.shard_map
    from tpu_inference.kernels.fused_moe.v1.kernel import fused_ep_moe
    from backward_kernel import fused_ep_moe_bwd_streaming

    EP = mesh.shape[ep_axis_name]
    fsdp_size = mesh.shape.get(fsdp_axis_name, 1) if fsdp_axis_name else 1

    @jax.custom_vjp
    def fused_ep_moe_train(tokens, w1, w2, gating_output):
        return fused_ep_moe(
            mesh, tokens, w1, w2, gating_output,
            top_k=top_k,
            renormalize_topk_logits=renormalize_topk_logits,
            scoring_fn=scoring_fn,
            act_fn=act_fn,
            ep_axis_name=ep_axis_name,
        )

    def _fwd(tokens, w1, w2, gating_output):
        tokens_f32 = tokens.astype(jnp.float32)
        w1_f32     = w1.astype(jnp.float32)
        w2_f32     = w2.astype(jnp.float32)
        gating_f32 = gating_output.astype(jnp.float32)

        out = fused_ep_moe(
            mesh, tokens_f32, w1_f32, w2_f32, gating_f32,
            top_k=top_k,
            renormalize_topk_logits=renormalize_topk_logits,
            scoring_fn=scoring_fn,
            act_fn=act_fn,
            ep_axis_name=ep_axis_name,
        )
        out = out.astype(tokens.dtype)
        saved = (tokens_f32, w1_f32, w2_f32, gating_f32)
        return out, saved

    def _bwd(saved, d_out):
        tokens_f32, w1_f32, w2_f32, gating_f32 = saved
        d_out_f32 = d_out.astype(jnp.float32)

        bwd_kwargs = dict(
            top_k=top_k,
            scoring_fn=scoring_fn,
            renormalize_topk_logits=renormalize_topk_logits,
            act_fn=act_fn,
            ep_axis_name=ep_axis_name,
            max_tpe=max_tpe,
        )

        if EP == 1 and fsdp_size == 1:
            d_tokens, d_w1, d_w2, d_gating = fused_ep_moe_bwd_streaming(
                d_out_f32, tokens_f32, w1_f32, w2_f32, gating_f32, **bwd_kwargs)

        elif fsdp_size > 1:
            # EP+FSDP backward: correct VJP of forward FSDP psum.
            #
            # Forward collective sequence (mirrors _expert_mlp_ep_body_ep_sharded):
            #   all_gather(x, "ep") → expert MLP → psum_scatter("ep") → psum(fsdp)
            #
            # Backward VJP (in reverse):
            #   psum(g, fsdp)        — VJP of forward psum(fsdp): AllReduce VJP = AllReduce
            #   all_gather(g, "ep")  — VJP of forward psum_scatter("ep")
            #   streaming kernel     — per-expert d_tokens (partial) and d_w (F_local shard)
            #   psum(d_tok, "ep")    — VJP of forward all_gather("ep")
            #   dynamic_slice        — recover EP-local token slice
            #
            # d_w is NOT reduced: each device already holds its (E_local, F_local) weight shard
            # and the corresponding gradient shard — no further EP or FSDP reduction needed.
            E_global = w1_f32.shape[0]
            act_spec = P((ep_axis_name, fsdp_axis_name), None)

            def _local_bwd_fsdp(d_out_l, tokens_l, w1_l, w2_l, gating_l):
                D = d_out_l.shape[1]

                # VJP of forward psum(fsdp_axis): psum g across FSDP devices.
                # Without this, d_w is 1/FSDP of correct (missing FSDP contribution factor).
                d_out_sum = lax.psum(d_out_l, fsdp_axis_name)  # (T/(EP*FSDP), D)

                # EP all_gather: reconstruct the T/FSDP token view.
                d_out_full  = lax.all_gather(d_out_sum, ep_axis_name, axis=0, tiled=True)
                tokens_full = lax.all_gather(tokens_l,  ep_axis_name, axis=0, tiled=True)
                gating_full = lax.all_gather(gating_l,  ep_axis_name, axis=0, tiled=True)

                # Streaming backward with FSDP-local weights (F_local slice).
                # w1_l: (E_local, 2, D, F_local), w2_l: (E_local, F_local, D)
                d_tok_partial, d_w1_l, d_w2_l, d_gate_partial = fused_ep_moe_bwd_streaming(
                    d_out_full, tokens_full, w1_l, w2_l, gating_full,
                    E_global_override=E_global,
                    **bwd_kwargs)

                # EP psum: sum partial d_tokens and d_gating from all EP devices.
                # No psum(fsdp) here: FSDP factor already incorporated via d_out_sum.
                d_tok_full  = lax.psum(d_tok_partial,  ep_axis_name)   # (T/FSDP, D)
                d_gate_full = lax.psum(d_gate_partial, ep_axis_name)   # (T/FSDP, E_global)

                # Slice back to this device's EP-local token portion.
                T_ep = d_out_l.shape[0]
                dev_ep = lax.axis_index(ep_axis_name)
                d_tok_l  = lax.dynamic_slice(d_tok_full,  (dev_ep * T_ep, 0), (T_ep, D))
                d_gate_l = lax.dynamic_slice(d_gate_full, (dev_ep * T_ep, 0), (T_ep, E_global))
                return d_tok_l, d_w1_l, d_w2_l, d_gate_l

            d_tokens, d_w1, d_w2, d_gating = shard_map(
                _local_bwd_fsdp,
                mesh=mesh,
                in_specs=(
                    act_spec,                                            # d_out
                    act_spec,                                            # tokens
                    P(ep_axis_name, None, None, fsdp_axis_name),        # w1 (E,2,D,F) F-shard
                    P(ep_axis_name, fsdp_axis_name, None),              # w2 (E,F,D) F-shard
                    act_spec,                                            # gating
                ),
                out_specs=(
                    act_spec,                                            # d_tokens
                    P(ep_axis_name, None, None, fsdp_axis_name),        # d_w1
                    P(ep_axis_name, fsdp_axis_name, None),              # d_w2
                    act_spec,                                            # d_gating
                ),
                check_vma=False,
            )(d_out_f32, tokens_f32, w1_f32, w2_f32, gating_f32)

        else:
            # EP>1, FSDP=1: original path — tokens replicated, weights EP-sharded.
            def _local_bwd(d_out_rep, tokens_rep, w1_local, w2_local, gating_rep):
                d_tok_partial, d_w1_local, d_w2_local, d_gate_partial = fused_ep_moe_bwd_streaming(
                    d_out_rep, tokens_rep, w1_local, w2_local, gating_rep, **bwd_kwargs)
                # All-reduce partial contributions across EP devices.
                d_tok  = lax.psum(d_tok_partial,  axis_name=ep_axis_name)
                d_gate = lax.psum(d_gate_partial, axis_name=ep_axis_name)
                return d_tok, d_w1_local, d_w2_local, d_gate

            d_tokens, d_w1, d_w2, d_gating = shard_map(
                _local_bwd,
                mesh=mesh,
                in_specs=(
                    P(),              # d_out   — replicated
                    P(),              # tokens  — replicated
                    P(ep_axis_name),  # w1      — sharded on expert axis
                    P(ep_axis_name),  # w2      — sharded on expert axis
                    P(),              # gating  — replicated
                ),
                out_specs=(
                    P(),              # d_tokens  — replicated (after psum)
                    P(ep_axis_name),  # d_w1      — sharded
                    P(ep_axis_name),  # d_w2      — sharded
                    P(),              # d_gating  — replicated (after psum)
                ),
                check_vma=False,
            )(d_out_f32, tokens_f32, w1_f32, w2_f32, gating_f32)

        d_tokens = d_tokens.astype(tokens_f32.dtype)
        return (d_tokens, d_w1, d_w2, d_gating)

    fused_ep_moe_train.defvjp(_fwd, _bwd)
    return fused_ep_moe_train


# ---------------------------------------------------------------------------
# Stage B: Pallas forward + JAX backward
# ---------------------------------------------------------------------------


def make_fused_ep_moe_train_v2(
        mesh,
        top_k: int,
        *,
        renormalize_topk_logits: bool = False,
        scoring_fn: str = "softmax",
        act_fn: str = "silu",
        ep_axis_name: str = "model",
):
    """Stage B: fused_ep_moe (Pallas) forward + JAX backward, EP=1.

    Uses the actual fused_ep_moe Pallas kernel for the forward output value.
    Residuals for the backward are computed via ref_moe_with_residuals (JAX
    reference). Supports EP=1 only; EP>1 backward requires Stage C.

    Args:
      mesh: 2D JAX mesh. The ep_axis_name axis drives expert parallelism;
            all other axes must have size 1. For a single device:
              Mesh(np.array([[device]]), ("model", "fsdp"))
      top_k, scoring_fn, renormalize_topk_logits, act_fn: same as v1.
      ep_axis_name: Mesh axis name for EP (default "model").

    Dimension constraints imposed by fused_ep_moe:
      hidden_size % 128 == 0
      intermediate_size % 128 == 0
      num_tokens % ep_size == 0
      num_experts % ep_size == 0
    """
    from tpu_inference.kernels.fused_moe.v1.kernel import fused_ep_moe

    cfg = dict(
        top_k=top_k,
        renormalize_topk_logits=renormalize_topk_logits,
        scoring_fn=scoring_fn,
        act_fn=act_fn,
    )

    @jax.custom_vjp
    def fused_ep_moe_train(tokens, w1, w2, gating_output):
        return fused_ep_moe(
            mesh, tokens, w1, w2, gating_output,
            top_k=top_k,
            renormalize_topk_logits=renormalize_topk_logits,
            scoring_fn=scoring_fn,
            act_fn=act_fn,
            ep_axis_name=ep_axis_name,
        )

    def _fwd(tokens, w1, w2, gating_output):
        tokens_f32 = tokens.astype(jnp.float32)
        w1_f32     = w1.astype(jnp.float32)
        w2_f32     = w2.astype(jnp.float32)
        gating_f32 = gating_output.astype(jnp.float32)

        # Fast Pallas forward — this is the value used downstream.
        out = fused_ep_moe(
            mesh, tokens_f32, w1_f32, w2_f32, gating_f32,
            top_k=top_k,
            renormalize_topk_logits=renormalize_topk_logits,
            scoring_fn=scoring_fn,
            act_fn=act_fn,
            ep_axis_name=ep_axis_name,
        )
        out = out.astype(tokens.dtype)

        # Residuals via JAX reference.  Assumes EP=1 (local == global tensors).
        # For EP>1 this would need an all-gather of tokens/weights first (Stage C).
        _, residuals = ref_moe_with_residuals(
            tokens_f32, w1_f32, w2_f32, gating_f32, **cfg)

        saved = (tokens_f32, w1_f32, w2_f32, gating_f32) + residuals
        return out, saved

    def _bwd(saved, d_out):
        (tokens_f32, w1_f32, w2_f32, gating_f32,
         h_gate, h_up, expert_outs, top_k_indices,
         top_k_logits, gating_scores) = saved

        d_out_f32 = d_out.astype(jnp.float32)

        d_expert_outs, d_top_k_logits = moe_bwd_combine(
            d_out_f32, top_k_logits, expert_outs)
        d_tokens, d_w1, d_w2 = moe_bwd_ffn(
            d_expert_outs, tokens_f32, w1_f32, w2_f32,
            top_k_indices, h_gate, h_up,
            act_fn=act_fn)
        d_gating = moe_bwd_routing(
            d_top_k_logits, top_k_indices, top_k_logits,
            gating_scores, gating_f32,
            scoring_fn=scoring_fn,
            renormalize_topk_logits=renormalize_topk_logits)

        d_tokens = d_tokens.astype(tokens_f32.dtype)
        return (d_tokens, d_w1, d_w2, d_gating)

    fused_ep_moe_train.defvjp(_fwd, _bwd)
    return fused_ep_moe_train
