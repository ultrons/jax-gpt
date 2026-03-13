"""Data pipeline for GPT-2 training using grain.

Supports two data sources:
  - 'local': a flat uint16 numpy memmap .bin file
  - 'gcs':   grain ArrayRecordDataSource pointing at a GCS path

The pipeline is multi-host aware: each jax.process_index() automatically
receives a unique shard of the dataset.
"""

import os

import grain.python as grain
import jax
import numpy as np

from jax_gpt.trainer.config import TrainConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _local_path(config: TrainConfig, split: str) -> str:
    return os.path.join(config.local_data_path, f"{split}.bin")


def _gcs_path(config: TrainConfig, split: str) -> str:
    return f"gs://{config.gcs_bucket}/{config.gcs_dataset_path}/{split}"


# ---------------------------------------------------------------------------
# grain DataSource for flat memmap token files
# ---------------------------------------------------------------------------

class _MemMapSource:
    """grain DataSource wrapping a flat numpy memmap of uint16 tokens."""

    def __init__(self, path: str, seq_len: int):
        data = np.memmap(path, dtype=np.uint16, mode='r')
        # Number of complete windows of (seq_len + 1) tokens
        self._windows = (len(data) - 1) // seq_len
        self._data = data
        self._seq_len = seq_len

    def __len__(self) -> int:
        return self._windows

    def __getitem__(self, idx: int) -> np.ndarray:
        start = idx * self._seq_len
        return self._data[start : start + self._seq_len + 1].astype(np.int32)


# ---------------------------------------------------------------------------
# Public DataPipeline
# ---------------------------------------------------------------------------

class DataPipeline:
    """Iterable data pipeline that yields (x, y) mini-batches.

    Parameters
    ----------
    config:
        Training configuration.  Relevant fields: data_source, gcs_bucket,
        gcs_dataset_path, local_data_path, num_workers, micro_batch_size,
        seq_len.
    split:
        One of 'train' or 'val'.
    _shard_index:
        Override the shard index (used in tests to simulate multiple hosts
        without actually spawning processes).
    _shard_count:
        Override the shard count (used in tests).
    """

    def __init__(
        self,
        config: TrainConfig,
        split: str = 'train',
        *,
        _shard_index: int | None = None,
        _shard_count: int | None = None,
    ):
        self._batch_size = config.micro_batch_size
        self._seq_len = config.seq_len

        # ------------------------------------------------------------------
        # Build data source
        # ------------------------------------------------------------------
        if config.data_source == 'local':
            path = _local_path(config, split)
            source: grain.RandomAccessDataSource = _MemMapSource(path, config.seq_len)
        elif config.data_source == 'gcs':
            # Import lazily so the module is importable on platforms where
            # GCS support may be absent (e.g. Apple M1 dev machines).
            from grain.experimental import ArrayRecordDataSource  # type: ignore
            path = _gcs_path(config, split)
            source = ArrayRecordDataSource(path)
        else:
            raise ValueError(
                f"Unknown data_source '{config.data_source}'. "
                "Expected 'local' or 'gcs'."
            )

        # ------------------------------------------------------------------
        # Multi-host sharding
        # ------------------------------------------------------------------
        shard_index = _shard_index if _shard_index is not None else jax.process_index()
        shard_count = _shard_count if _shard_count is not None else jax.process_count()

        shard_options = grain.ShardOptions(
            shard_index=shard_index,
            shard_count=shard_count,
        )

        # ------------------------------------------------------------------
        # Build loader
        # ------------------------------------------------------------------
        self._loader = grain.DataLoader(
            data_source=source,
            sampler=grain.SequentialSampler(
                num_records=len(source),
                shard_options=shard_options,
            ),
            worker_count=config.num_workers,
            read_options=grain.ReadOptions(prefetch_buffer_size=128),
        )
        self._iter = iter(self._loader)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_batch(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (x, y) each of shape (micro_batch_size, seq_len)."""
        sequences = [next(self._iter) for _ in range(self._batch_size)]
        batch = np.stack(sequences)   # (micro_batch_size, seq_len + 1)
        x = batch[:, :-1]            # (micro_batch_size, seq_len)
        y = batch[:, 1:]             # (micro_batch_size, seq_len)
        return x, y

    def skip_to_step(self, step: int, grad_accum_steps: int) -> None:
        """Fast-forward the iterator to resume from a checkpoint step.

        Each training step consumes ``grad_accum_steps * micro_batch_size``
        sequences (one mini-batch per accumulation micro-step).
        """
        n_to_skip = step * grad_accum_steps * self._batch_size
        for _ in range(n_to_skip):
            next(self._iter)
