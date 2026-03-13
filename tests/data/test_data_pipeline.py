"""Tests for jax_gpt.data.pipeline.DataPipeline."""

import numpy as np
import pytest

grain = pytest.importorskip('grain', reason='grain not installed')

from jax_gpt.data.pipeline import DataPipeline
from jax_gpt.trainer.config import TrainConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_temp_bin(tmp_path, n_tokens: int = 10_000) -> str:
    """Write a sequential uint16 token file and return its path."""
    data = np.arange(n_tokens, dtype=np.uint16)
    path = tmp_path / "train.bin"
    data.tofile(path)
    return str(path)


def make_config(tmp_path, micro_batch_size: int = 4, seq_len: int = 16) -> TrainConfig:
    return TrainConfig(
        data_source='local',
        local_data_path=str(tmp_path),
        micro_batch_size=micro_batch_size,
        seq_len=seq_len,
        num_workers=0,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_get_batch_shapes(tmp_path):
    """x and y must have shape (micro_batch_size, seq_len)."""
    micro_batch_size = 4
    seq_len = 16
    make_temp_bin(tmp_path)
    config = make_config(tmp_path, micro_batch_size=micro_batch_size, seq_len=seq_len)

    pipeline = DataPipeline(config, split='train', _shard_index=0, _shard_count=1)
    x, y = pipeline.get_batch()

    assert x.shape == (micro_batch_size, seq_len), f"x.shape={x.shape}"
    assert y.shape == (micro_batch_size, seq_len), f"y.shape={y.shape}"


def test_targets_are_inputs_shifted(tmp_path):
    """y must equal x shifted left by one position (causal LM targets)."""
    make_temp_bin(tmp_path)
    config = make_config(tmp_path)

    pipeline = DataPipeline(config, split='train', _shard_index=0, _shard_count=1)
    x, y = pipeline.get_batch()

    # x[:, 1:] should equal y[:, :-1]
    assert np.array_equal(x[:, 1:], y[:, :-1]), (
        "Targets are not a one-step shifted version of inputs."
    )


def test_multi_shard_no_overlap(tmp_path):
    """Two shards must return different (non-overlapping) sequences."""
    make_temp_bin(tmp_path, n_tokens=10_000)
    config = make_config(tmp_path, micro_batch_size=4, seq_len=16)

    pipeline0 = DataPipeline(config, split='train', _shard_index=0, _shard_count=2)
    pipeline1 = DataPipeline(config, split='train', _shard_index=1, _shard_count=2)

    x0, _ = pipeline0.get_batch()
    x1, _ = pipeline1.get_batch()

    assert not np.array_equal(x0, x1), (
        "Shard 0 and shard 1 returned identical batches — sharding is broken."
    )
