"""Tier-3 correctness probe: LM cross-entropy on the MaxText DSv3 checkpoint.

Loads gs://mlperf-6-submission/ckpt0424-fsdp/0/items via orbax (MaxText's
mid-training resume point — first-step llm_loss=5.337 in MaxText's run),
maps the param tree into our DSv3 dict layout, runs forward + LM CE on a
fixed pre-tokenized English snippet, reports the result.

Token IDs are pre-computed locally from the released HF deepseek-ai/DeepSeek-V3
tokenizer over the same snippet eval_lm_loss.py uses, so the pod doesn't
need a tokenizer at runtime (the tokenizer files aren't on this cluster).

Acceptance criteria (rough):
  - LM CE around 2-4 → forward path is correct against the MaxText reference.
    (Won't exactly match 5.337 because the MaxText baseline uses a different
    tokenizer (Llama-3 tiktoken) and a c4 batch, not this hardcoded snippet.)
  - LM CE >= 8 → real bug somewhere — investigate param mapping next.
  - LM CE = NaN → numerical issue or shape/layout mismatch.
"""

from __future__ import annotations

import argparse
import time

import jax
import jax.numpy as jnp

from .model import (
    ModelConfig, full_671b_config, ShardConfig, forward,
    _vocab_ce,
)
from .load_maxtext_ckpt import load_maxtext_dsv3


# Pre-tokenized via HF deepseek-ai/DeepSeek-V3 tokenizer.
# Source text:
#   "The capital of France is Paris. France is a country in Western Europe
#    known for its art, culture, cuisine, and history. Paris, the capital,
#    is home to many world-famous landmarks including the Eiffel Tower, the
#    Louvre Museum, and the Notre-Dame Cathedral. The Seine River runs
#    through the heart of the city, dividing it into two banks: the Right
#    Bank and the Left Bank, each with its own distinct character."
DEFAULT_IDS = [
    671, 79666, 2154, 51725, 278, 51119, 7812, 13382, 278, 439, 870, 744,
    261, 67576, 29207, 7825, 2251, 1303, 521, 24351, 7395, 14, 15483, 36545,
    40280, 41069, 5497, 32357, 55295, 79666, 14, 994, 3527, 316, 1631, 29616,
    2410, 27604, 1831, 22144, 17473, 1805, 39, 4280, 317, 54, 1344, 55295, 46,
    34983, 266, 47, 50587, 40280, 1805, 6343, 266, 6897, 691, 38499, 29569,
    14170, 4374, 560, 102293, 15721, 121935, 6051, 263, 37838, 39499, 32842,
    481, 32244, 1841, 288, 279, 650, 347, 89, 924, 5268, 28, 1805, 16697,
    39196, 458, 1805, 21242, 39196, 14, 38065, 6135, 1303, 359, 289, 435,
    6149, 32363, 16,
]


def main():
    import os
    if os.environ.get("MEGASCALE_COORDINATOR_ADDRESS"):
        jax.distributed.initialize()

    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="gs://mlperf-6-submission/ckpt0424-fsdp/0/items")
    parser.add_argument("--fsdp", type=int, default=128)
    parser.add_argument("--ep", type=int, default=4)
    parser.add_argument("--tp", type=int, default=1)
    args, unknown = parser.parse_known_args()
    if unknown:
        print(f"(ignoring inherited training-only flags: {unknown})")

    # MUST set vocab=129280 BEFORE init/load — output_head & embed shapes depend on it.
    cfg = full_671b_config()
    cfg.V = 129280                      # MaxText DSv3 vocab
    # gmm_ag matches v304 training and avoids the _moe_jax_ep code path,
    # whose `from backward_kernel import sc_gather_rows` (model.py:735) fails
    # in our pod (PYTHONPATH=/app, but kernel lives at .kernels.fused_moe_bwd).
    cfg.moe_backend = "gmm_ag"
    cfg.gradient_checkpoint = False     # inference only
    # MaxText-trained DSv3 671B uses routed_scaling_factor=2.5 (its config
    # configs/models/deepseek3-671b.yml). Without this, MoE outputs are
    # ~2.5x under-weighted in the residual stream → high LM CE.
    cfg.routed_scaling_factor = 2.5
    # DSv3 671B group-limited routing: 256 experts split into 8 groups, top-4
    # groups selected per token. Without this, our flat top-K picks different
    # experts than the model was trained for → very high aux + bad LM CE.
    cfg.n_routing_groups = 8
    cfg.topk_routing_group = 4
    # YaRN-modified attention softmax scale (MaxText's MLA: rope_factor=40,
    # mscale=1.0). Effective scale ≈ 0.1352 vs vanilla 0.0722 → ~1.87x
    # sharper softmax. The model was trained with this; without it, attention
    # is too flat → high LM CE.
    cfg.use_yarn_scale = True
    cfg.attn_mscale = 1.0
    cfg.rope_factor = 40.0

    shard_cfg = ShardConfig(fsdp=args.fsdp, ep=args.ep, tp=args.tp)
    mesh = shard_cfg.create_mesh()
    cfg.mesh = mesh

    print(f"Devices: {jax.device_count()} x {jax.devices()[0].device_kind}")
    print(f"Mesh: dp={shard_cfg.dp}, fsdp={shard_cfg.fsdp}, "
          f"ep={shard_cfg.ep}, tp={shard_cfg.tp}")
    print(f"cfg.V (vocab) = {cfg.V}")

    # ── Load MaxText checkpoint ───────────────────────────────────────────
    t0 = time.time()
    params = load_maxtext_dsv3(args.ckpt, cfg, mesh=mesh)
    jax.block_until_ready(params)
    print(f"\nWeights loaded in {time.time() - t0:.0f}s")
    print(f"  embed shape:       {params['embed'].shape}")
    print(f"  output_head shape: {params['output_head'].shape}")
    print(f"  final_norm shape:  {params['final_norm'].shape}")

    # ── Tokens (pre-tokenized; pad to cfg.S) ──────────────────────────────
    # Prepend BoS (id=0 in HF DSv3 tokenizer). MaxText trains with BoS at
    # position 0; without it the model sees "real text" at pos 0 which is
    # an unusual distribution → first few token CEs are inflated.
    ids = [0] + list(DEFAULT_IDS)  # BoS = 0
    real_len = len(ids)
    pad_id = 0
    while len(ids) < cfg.S:
        ids.append(pad_id)
    if len(ids) > cfg.S:
        ids = ids[:cfg.S]

    # Activation sharding constraints in our forward use P('fsdp', 'ep', None)
    # for (B, S, D) hidden states. B=1 can't be sharded across fsdp=128 (t5
    # IndivisibleError). Tile to B = fsdp*ep so every replica sees the same
    # data — final per-token CE is identical, just computed redundantly.
    B = args.fsdp * args.ep
    tokens = jnp.tile(jnp.array([ids], dtype=jnp.int32), (B, 1))  # (B, S)
    print(f"\nReal-text tokens: {real_len} / {cfg.S} (padded with id={pad_id})")
    print(f"  Tiled batch to B={B} to match fsdp*ep activation sharding.")
    print(f"  max id: {max(DEFAULT_IDS)}, min id: {min(DEFAULT_IDS)}")

    # ── Forward + LM CE on the real-text positions only ──────────────────
    print("\nRunning forward pass...")
    t0 = time.time()
    x_final, aux_loss = forward(params, tokens, cfg, return_final_x=True)
    # Use just batch row 0 — all rows are identical.
    x_pred = x_final[:1, :-1]
    real_S = max(1, real_len - 1)
    x_pred_real = x_pred[:, :real_S, :]
    targets_real = tokens[:1, 1:1 + real_S].astype(jnp.int32)

    V_CHUNK = 4096
    n_chunks = cfg.V // V_CHUNK     # 31 for V=129280; trailing 2304 dims unused (test ids < 126976)
    lm_loss = _vocab_ce(x_pred_real, params["output_head"],
                        targets_real, n_chunks, V_CHUNK)

    jax.block_until_ready(lm_loss)
    print(f"Forward + CE took {time.time() - t0:.1f}s")

    print("\n" + "=" * 70)
    print(f"  LM cross-entropy (MaxText DSv3 ckpt, real text {real_S} positions): "
          f"{float(lm_loss):.4f}")
    print(f"  MoE aux_loss (small coefficient ~1e-4):                              "
          f"{float(aux_loss):.6f}")
    print("=" * 70)
    print()
    print("Acceptance:")
    print("  ~ 2-4    : forward correct, in expected DSv3-base range.")
    print("  ~ 5-7    : forward likely correct (MaxText reports 5.337 mid-training).")
    print("  ~ 8-15   : possible param-mapping bug. Diff our params vs orbax tree.")
    print("  >= 20    : likely shape/transpose error somewhere in load_maxtext_ckpt.")


if __name__ == "__main__":
    main()
