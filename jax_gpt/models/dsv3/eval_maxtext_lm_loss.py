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
    ModelConfig, full_671b_maxtext_config, ShardConfig, forward,
    _vocab_ce,
)
from .load_maxtext_ckpt import load_maxtext_dsv3


# Source text used to seed DEFAULT_IDS — kept verbatim for the runtime
# tokenizer round-trip assertion (verify_tokens). The smoking gun in the
# t-series was that DEFAULT_IDS were NOT actually produced by the claimed
# tokenizer; only token 0 matched. Now any drift fails fast at startup.
SOURCE_TEXT = (
    "The capital of France is Paris. France is a country in Western Europe "
    "known for its art, culture, cuisine, and history. Paris, the capital, "
    "is home to many world-famous landmarks including the Eiffel Tower, "
    "the Louvre Museum, and the Notre-Dame Cathedral. The Seine River runs "
    "through the heart of the city, dividing it into two banks: the Right "
    "Bank and the Left Bank, each with its own distinct character."
)
DEFAULT_IDS = [
    # Re-tokenized 2026-05-02 via HF deepseek-ai/DeepSeek-V3 tokenizer
    # (tokenizer.encode(text, add_special_tokens=False)). The previous
    # 100-token list was wrong tokenizer (only token 0 matched correct
    # encoding) — model saw nonsense tokens, explaining CE≈10 vs MaxText 5.337.
    671, 6102, 294, 8760, 344, 11111, 16, 8760, 344, 260, 3924, 295,
    10734, 4174, 3459, 362, 1009, 2783, 14, 5785, 14, 41506, 14, 305,
    3980, 16, 11111, 14, 270, 6102, 14, 344, 2680, 304, 1623, 2058,
    2410, 27604, 63554, 2622, 270, 446, 4280, 317, 36788, 14, 270, 125089,
    13924, 14, 305, 270, 53893, 6897, 691, 54860, 16, 455, 96727, 9875,
    12122, 1407, 270, 4082, 294, 270, 4593, 14, 26843, 436, 1055, 1234,
    14664, 28, 270, 15759, 9063, 305, 270, 27925, 9063, 14, 1660, 418,
    1009, 1956, 8250, 3053, 16,
]


def verify_default_ids(strict: bool = True) -> None:
    """Re-tokenize SOURCE_TEXT and assert DEFAULT_IDS matches.

    Catches the bug class where hardcoded token IDs drift from what the
    tokenizer actually produces (the t8-t14 trap: only the first token
    matched, model received nonsense, CE was stuck near random).

    Skipped silently if `transformers` isn't installed in the pod.
    """
    try:
        from transformers import AutoTokenizer
    except ImportError:
        print("verify_default_ids: transformers not installed; skipping.")
        return
    tk = AutoTokenizer.from_pretrained(
        "deepseek-ai/DeepSeek-V3", trust_remote_code=True)
    actual = tk.encode(SOURCE_TEXT, add_special_tokens=False)
    if actual == DEFAULT_IDS:
        print(f"verify_default_ids: ✓ {len(actual)} tokens match HF DSv3 tokenizer.")
        return
    # Find first divergence for diagnostic
    n = min(len(actual), len(DEFAULT_IDS))
    first_diff = next((i for i in range(n) if actual[i] != DEFAULT_IDS[i]), n)
    msg = (
        f"DEFAULT_IDS DO NOT MATCH HF DSv3 tokenizer encoding.\n"
        f"  expected (encode): len={len(actual)}, first 10 = {actual[:10]}\n"
        f"  got (DEFAULT_IDS):  len={len(DEFAULT_IDS)}, first 10 = {DEFAULT_IDS[:10]}\n"
        f"  first divergence at index {first_diff}.\n"
        f"  → re-tokenize SOURCE_TEXT and update DEFAULT_IDS before proceeding."
    )
    if strict:
        raise RuntimeError(msg)
    print("WARN " + msg)


def main():
    import os
    if os.environ.get("MEGASCALE_COORDINATOR_ADDRESS"):
        jax.distributed.initialize()
    # Tokenizer round-trip guard — fail fast on hallucinated DEFAULT_IDS.
    if jax.process_index() == 0:
        verify_default_ids(strict=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="gs://mlperf-6-submission/ckpt0424-fsdp/0/items")
    parser.add_argument("--fsdp", type=int, default=128)
    parser.add_argument("--ep", type=int, default=4)
    parser.add_argument("--tp", type=int, default=1)
    args, unknown = parser.parse_known_args()
    if unknown:
        print(f"(ignoring inherited training-only flags: {unknown})")

    # All DSv3-MaxText architectural knobs in one place (V=129280,
    # routed_scaling_factor=2.5, n_routing_groups=8, topk_routing_group=4,
    # use_yarn_scale=True, attn_mscale=1.0, rope_factor=40). Each is
    # required to match MaxText's trained behavior — see model.py.
    cfg = full_671b_maxtext_config()
    cfg.moe_backend = "gmm_ag"           # avoid _moe_jax_ep import path
    cfg.gradient_checkpoint = False      # inference only

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

    # ── Top-K diagnostics at semantic positions ───────────────────────────
    # With BoS at pos 0 + correct HF DSv3 tokens:
    #   pos 1 = "The"      → predict pos 2 = " capital" (6102)
    #   pos 5 = " is"      → predict pos 6 = " Paris"   (11111)  ★ key one
    #   pos 6 = " Paris"   → predict pos 7 = "."        (16)
    #   pos 11 = " in"     → predict pos 12 = " Western" (10734)
    #   pos 12 = " Western"→ predict pos 13 = " Europe" (4174)
    # If "Paris" shows up high-rank, attention/RoPE/routing all work and
    # remaining LM CE gap is calibration. If "Paris" is rank ~thousand,
    # there is a fundamental forward-pass bug.
    diag_positions = [1, 5, 6, 11, 12]
    expected_ids   = [6102, 11111, 16, 10734, 4174]
    print("\n──── Top-K diagnostics ────")
    # x_pred is (1, S-1, D); take diag positions, all of vocab.
    x_diag = x_pred[0, jnp.array(diag_positions), :].astype(jnp.float32)  # (P, D)
    logits_diag = (x_diag @ params["output_head"].astype(jnp.float32))    # (P, V)
    # Top-5 token IDs and their logits per position.
    top_v, top_i = jax.lax.top_k(logits_diag, k=5)
    jax.block_until_ready(top_v)
    for i, (pos, exp_id) in enumerate(zip(diag_positions, expected_ids)):
        exp_logit = float(logits_diag[i, exp_id])
        # Rank of expected (count of tokens with strictly higher logit + 1)
        rank = int(jnp.sum(logits_diag[i] > logits_diag[i, exp_id])) + 1
        top5 = list(zip([int(x) for x in top_i[i]], [float(x) for x in top_v[i]]))
        print(f"  pos {pos:>2d} → expected id={exp_id:>6d} "
              f"(logit={exp_logit:+.3f}, rank={rank}/{cfg.V})")
        for tid, lg in top5:
            mark = " ★" if tid == exp_id else ""
            print(f"      top: id={tid:>6d}  logit={lg:+.3f}{mark}")

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
