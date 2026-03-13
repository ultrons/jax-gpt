#!/usr/bin/env python3
"""
Offline tokenization script. Run once to prepare data for training.

Reads raw text from a directory of .txt files or a single JSONL file
(one {"text": "..."} per line), tokenizes with the GPT-2 BPE tokenizer
(tiktoken), and writes flat uint16 binary files (train.bin / val.bin)
either locally or to a GCS bucket.

Usage (local, directory of .txt files):
  python -m jax_gpt.data.tokenize_to_gcs --input data/raw/ --output data/

Usage (local, JSONL):
  python -m jax_gpt.data.tokenize_to_gcs --input data/raw.jsonl --output data/

Usage (GCS):
  python -m jax_gpt.data.tokenize_to_gcs --input data/raw/ \\
    --gcs_bucket my-bucket --gcs_path datasets/openwebtext
"""

import argparse
import io
import json
import os
import sys

import numpy as np
import tiktoken

# GPT-2 end-of-text token id
EOT_TOKEN: int = 50256


# ---------------------------------------------------------------------------
# Text iteration helpers
# ---------------------------------------------------------------------------

def _iter_texts_from_dir(directory: str):
    """Yield raw strings from every .txt file in *directory*."""
    for fname in sorted(os.listdir(directory)):
        if fname.endswith('.txt'):
            fpath = os.path.join(directory, fname)
            with open(fpath, 'r', encoding='utf-8') as fh:
                yield fh.read()


def _iter_texts_from_jsonl(path: str):
    """Yield the 'text' field from each line of a JSONL file."""
    with open(path, 'r', encoding='utf-8') as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            yield obj['text']


def _iter_texts(input_path: str):
    """Auto-detect whether *input_path* is a JSONL file or a directory."""
    if os.path.isfile(input_path):
        yield from _iter_texts_from_jsonl(input_path)
    elif os.path.isdir(input_path):
        yield from _iter_texts_from_dir(input_path)
    else:
        raise FileNotFoundError(f"Input path not found: {input_path}")


# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------

def tokenize(input_path: str, val_fraction: float = 0.05):
    """Tokenize all texts and return (train_tokens, val_tokens) as uint16 arrays."""
    enc = tiktoken.get_encoding('gpt2')

    all_ids: list[int] = []
    for text in _iter_texts(input_path):
        ids = enc.encode_ordinary(text)
        ids.append(EOT_TOKEN)
        all_ids.extend(ids)

    arr = np.array(all_ids, dtype=np.uint16)
    n_val = max(1, int(len(arr) * val_fraction))
    train_arr = arr[n_val:]
    val_arr = arr[:n_val]
    return train_arr, val_arr


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _write_local(arr: np.ndarray, output_dir: str, split: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{split}.bin")
    arr.tofile(path)
    print(f"Wrote {len(arr):,} tokens -> {path}")
    return path


def _write_gcs(arr: np.ndarray, bucket: str, gcs_path: str, split: str) -> None:
    try:
        from google.cloud import storage  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "google-cloud-storage is required for GCS upload. "
            "Install with: pip install google-cloud-storage"
        ) from exc

    blob_name = f"{gcs_path.strip('/')}/{split}.bin"
    client = storage.Client()
    bkt = client.bucket(bucket)
    blob = bkt.blob(blob_name)

    buf = io.BytesIO(arr.tobytes())
    blob.upload_from_file(buf, content_type='application/octet-stream')
    print(f"Uploaded {len(arr):,} tokens -> gs://{bucket}/{blob_name}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='Tokenize raw text to uint16 binary files for GPT-2 training.',
    )
    parser.add_argument(
        '--input', required=True,
        help='Path to a directory of .txt files or a single JSONL file.',
    )
    parser.add_argument(
        '--output', default='',
        help='Local output directory for train.bin / val.bin (required when not using GCS).',
    )
    parser.add_argument(
        '--gcs_bucket', default='',
        help='GCS bucket name (triggers GCS upload when set).',
    )
    parser.add_argument(
        '--gcs_path', default='',
        help='GCS object path prefix, e.g. datasets/openwebtext.',
    )
    parser.add_argument(
        '--val_fraction', type=float, default=0.05,
        help='Fraction of tokens reserved for validation (default: 0.05).',
    )
    parser.add_argument(
        '--format', dest='fmt', choices=['bin'], default='bin',
        help='Output format (only "bin" is supported for now).',
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    use_gcs = bool(args.gcs_bucket)
    if not use_gcs and not args.output:
        print("ERROR: Provide --output for local storage or --gcs_bucket for GCS.", file=sys.stderr)
        sys.exit(1)

    print(f"Tokenizing: {args.input}")
    train_arr, val_arr = tokenize(args.input, val_fraction=args.val_fraction)
    print(f"Total tokens: {len(train_arr) + len(val_arr):,}  "
          f"(train={len(train_arr):,}, val={len(val_arr):,})")

    if use_gcs:
        _write_gcs(train_arr, args.gcs_bucket, args.gcs_path, 'train')
        _write_gcs(val_arr, args.gcs_bucket, args.gcs_path, 'val')
    else:
        _write_local(train_arr, args.output, 'train')
        _write_local(val_arr, args.output, 'val')

    # Also write locally when both output and GCS are specified
    if use_gcs and args.output:
        _write_local(train_arr, args.output, 'train')
        _write_local(val_arr, args.output, 'val')


if __name__ == '__main__':
    main()
