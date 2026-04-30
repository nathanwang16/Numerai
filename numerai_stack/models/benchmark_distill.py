"""Benchmark-model distillation.

Numerai ships ``train_benchmark_models.parquet`` + ``validation_benchmark_models.parquet``
containing per-row predictions from many Numerai benchmark models. We can feed
these predictions as *extra features* to a student model -- a form of soft
label distillation that captures what sophisticated benchmarks "see" without
needing to re-train them.

Example:
    train = loader.load_train(features=feats, targets=['target'])
    tb    = loader.load_benchmark_models('train')
    train_x = pd.concat([train[feats], tb.add_prefix('bench_')], axis=1)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class BenchmarkFeatureBuilder:
    """Attach benchmark-model predictions as features.

    Column names get the ``bench_`` prefix. NaNs are filled with 0.5 to reflect
    neutral signal in uniform [0,1] space.
    """

    prefix: str = "bench_"

    def attach(self, X: pd.DataFrame, bench_df: pd.DataFrame) -> pd.DataFrame:
        bench = bench_df.copy()
        bench.columns = [f"{self.prefix}{c}" for c in bench.columns]
        bench = bench.fillna(0.5)
        return pd.concat([X, bench.loc[X.index]], axis=1)

    def column_names(self, bench_df: pd.DataFrame) -> list[str]:
        return [f"{self.prefix}{c}" for c in bench_df.columns]


__all__ = ["BenchmarkFeatureBuilder"]
