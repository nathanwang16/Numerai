"""Download + cache the Numerai v5.2 dataset locally.

Usage:
    python scripts/download_data.py [--version v5.2] [--cache-dir data]
"""
from __future__ import annotations

import argparse

from numerai_stack import DEFAULT_DATA_VERSION
from numerai_stack.data import DataLoader


FILES = [
    "features.json",
    "train.parquet",
    "validation.parquet",
    "live.parquet",
    "meta_model.parquet",
    "train_benchmark_models.parquet",
    "validation_benchmark_models.parquet",
    "live_benchmark_models.parquet",
    "validation_example_preds.parquet",
    "live_example_preds.parquet",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default=DEFAULT_DATA_VERSION)
    ap.add_argument("--cache-dir", default="data")
    ap.add_argument("--files", nargs="*", default=FILES)
    args = ap.parse_args()

    loader = DataLoader(cache_dir=args.cache_dir, version=args.version)
    for f in args.files:
        try:
            path = loader.ensure(f)
            print(f"[ok] {f} -> {path}")
        except Exception as e:
            print(f"[fail] {f}: {e}")


if __name__ == "__main__":
    main()
