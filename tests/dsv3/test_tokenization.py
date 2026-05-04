"""Tokenizer round-trip guard for hardcoded DEFAULT_IDS lists.

The Tier 3 t8-t14 arc burned ~10 cluster iterations chasing architectural
fixes when the actual bug was that DEFAULT_IDS in eval_maxtext_lm_loss.py
were not what the HF DSv3 tokenizer produces — only token 0 matched.
This test would have caught it in 30 seconds.

Skipped if `transformers` is not installed.
"""
from __future__ import annotations
import pytest

pytest.importorskip("transformers")

from transformers import AutoTokenizer  # noqa: E402

from jax_gpt.models.dsv3.eval_maxtext_lm_loss import (  # noqa: E402
    SOURCE_TEXT, DEFAULT_IDS,
)


def test_default_ids_match_hf_dsv3_tokenizer():
    tk = AutoTokenizer.from_pretrained(
        "deepseek-ai/DeepSeek-V3", trust_remote_code=True)
    actual = tk.encode(SOURCE_TEXT, add_special_tokens=False)
    assert actual == DEFAULT_IDS, (
        f"len: encoded={len(actual)} hardcoded={len(DEFAULT_IDS)}\n"
        f"first 10 encoded   = {actual[:10]}\n"
        f"first 10 hardcoded = {DEFAULT_IDS[:10]}\n"
        f"Re-tokenize SOURCE_TEXT and update DEFAULT_IDS."
    )
