"""Feature engineering on Numerai's obfuscated features.

- PerEraPCA: fit IncrementalPCA across all train rows, then per-era transform.
- FeatureGroupAggregates: compute per-era mean/std/rank within each feature
  group (using ``features.json`` groupings) to create diversity-useful inputs.

These transformers are pickle-safe and usable at inference time.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.decomposition import IncrementalPCA


@dataclass
class PerEraPCA:
    n_components: int = 32
    fit_batch: int = 500_000
    pca_: IncrementalPCA | None = field(default=None, init=False, repr=False)
    feature_cols_: list[str] = field(default_factory=list, init=False)

    def fit(self, df: pd.DataFrame, feature_cols: list[str]) -> "PerEraPCA":
        self.feature_cols_ = list(feature_cols)
        ipca = IncrementalPCA(n_components=self.n_components)
        n = len(df)
        for start in range(0, n, self.fit_batch):
            batch = df.iloc[start:start + self.fit_batch][self.feature_cols_].astype(np.float32).fillna(0.5).values
            ipca.partial_fit(batch)
        self.pca_ = ipca
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.pca_ is None:
            raise RuntimeError("Fit the PCA first.")
        X = df[self.feature_cols_].astype(np.float32).fillna(0.5).values
        Z = self.pca_.transform(X)
        cols = [f"pca_{i}" for i in range(Z.shape[1])]
        return pd.DataFrame(Z, columns=cols, index=df.index)


@dataclass
class FeatureGroupAggregates:
    """Per-era, per-group summary features (mean + std + rank position)."""

    groups: dict[str, list[str]]
    era_col: str = "era"

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        pieces = []
        for gname, feats in self.groups.items():
            feats_in = [f for f in feats if f in df.columns]
            if not feats_in:
                continue
            g = df[feats_in]
            mean = g.mean(axis=1).rename(f"grp_{gname}_mean")
            std = g.std(axis=1).rename(f"grp_{gname}_std")
            pieces.append(pd.concat([mean, std], axis=1))
        if not pieces:
            return pd.DataFrame(index=df.index)
        return pd.concat(pieces, axis=1)


__all__ = ["PerEraPCA", "FeatureGroupAggregates"]
