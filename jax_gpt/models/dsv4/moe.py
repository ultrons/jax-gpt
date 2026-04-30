"""DSv4 MoE — 384 routed experts + 1 shared, top-6, sqrtsoftplus + noaux_tc.

Differences vs qwen35 MoE:
  - num experts: 384 (vs 128-512 in qwen35).
  - top-k:       6   (vs 10).
  - scoring_func='sqrtsoftplus' applied to router logits before top-k.
  - topk_method='noaux_tc' — no auxiliary load-balance loss; instead the
    router has a per-expert bias that is updated online toward balance.
  - routed_scaling_factor=2.5 multiplies post-norm gate weights.
  - SwiGLU activation is **clipped at swiglu_limit=10.0** (clip after the
    SiLU(gate) * up product, before down_proj — TBD: confirm exact location).
  - MoE expert weights stored in FP4 with block-scale dequant; everything
    else is FP8 e4m3.
  - Single shared expert is added to every token's output (post-routing).

Reuses jax_gpt's gmm_v2 Pallas GMM kernel for expert dispatch (see
qwen35/megablox/) — that kernel takes per-expert chunks of sorted tokens
and runs grouped matmul. Will need an FP4 dequant variant.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from jax_gpt.models.dsv4.config import DSv4Config


def sqrtsoftplus(x: jax.Array) -> jax.Array:
    """Router scoring: f(x) = sqrt(softplus(x)). Strictly positive, smoother than ReLU.

    softplus(x) = log(1 + exp(x)); we use jax.nn.softplus for numerical stability.
    """
    return jnp.sqrt(jax.nn.softplus(x))


def noaux_tc_topk(
    logits: jax.Array,           # (T, n_routed_experts)
    expert_bias: jax.Array,      # (n_routed_experts,) — online-updated balance bias
    top_k: int,
) -> tuple[jax.Array, jax.Array]:
    """Top-k expert selection with per-expert bias (no aux loss).

    Returns (topk_indices, topk_scores). The bias is added BEFORE top-k for
    selection but the gate weight uses the un-biased (post-sqrtsoftplus) score.
    """
    score = sqrtsoftplus(logits)              # (T, E)
    biased = score + expert_bias[None, :]
    _topk_biased, idx = jax.lax.top_k(biased, top_k)
    # Gate values: gather un-biased scores at the selected indices.
    gate = jnp.take_along_axis(score, idx, axis=-1)
    return idx.astype(jnp.int32), gate


def moe_block(
    x: jax.Array,
    params: dict,
    cfg: DSv4Config,
) -> jax.Array:
    """Routed top-6 experts + 1 shared expert. STUB."""
    raise NotImplementedError('Wire to gmm_v2 with FP4 dequant; see module docstring.')
