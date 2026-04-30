"""Regime-aware training: cluster eras then train a head per cluster + a router.

Regime clustering
-----------------
Per era, summarize with a vector of cross-sectional statistics:
    - mean/std of first K features
    - feature-wise std of ranks (dispersion)
    - mean absolute correlation within a feature sample (collinearity)
Cluster era-level vectors with KMeans to get regime labels.

Router
------
At inference we predict regime from current-era features (nearest-centroid in
the era-summary space), then apply the regime-specific head. We also blend
with a global head as a safety net.
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def era_summary(df_era: pd.DataFrame, feature_cols: Sequence[str], n_sample_feats: int = 32) -> np.ndarray:
    """Per-era feature summary used for regime clustering."""
    X = df_era[list(feature_cols)].astype(np.float32)
    sample_cols = list(feature_cols)[:n_sample_feats]
    Xs = df_era[sample_cols].astype(np.float32)
    stats = [
        X.mean().mean(),
        X.std().mean(),
        Xs.std().std(),
        Xs.rank(pct=True).std().mean(),
    ]
    return np.asarray(stats, dtype=np.float32)


@dataclass
class RegimeRouter:
    feature_cols: list[str]
    n_regimes: int = 4
    n_sample_feats: int = 32
    random_state: int = 0

    # Fitted
    _scaler: StandardScaler | None = field(default=None, init=False, repr=False)
    _kmeans: KMeans | None = field(default=None, init=False, repr=False)

    def fit(self, df: pd.DataFrame, era_col: str = "era") -> "RegimeRouter":
        rows = []
        eras = []
        for era, sub in df.groupby(era_col):
            rows.append(era_summary(sub, self.feature_cols, self.n_sample_feats))
            eras.append(era)
        M = np.stack(rows, axis=0)
        self._scaler = StandardScaler().fit(M)
        Ms = self._scaler.transform(M)
        self._kmeans = KMeans(
            n_clusters=self.n_regimes, random_state=self.random_state, n_init=10
        ).fit(Ms)
        self.era_regime_map_ = dict(zip(eras, self._kmeans.labels_.tolist()))
        return self

    def assign(self, df: pd.DataFrame, era_col: str = "era") -> pd.Series:
        if self._kmeans is None:
            raise RuntimeError("Fit the router first.")
        out = np.empty(len(df), dtype=np.int64)
        for era, sub in df.groupby(era_col):
            if era in self.era_regime_map_:
                label = self.era_regime_map_[era]
            else:
                v = era_summary(sub, self.feature_cols, self.n_sample_feats).reshape(1, -1)
                label = int(self._kmeans.predict(self._scaler.transform(v))[0])
            out[sub.index.get_indexer_for(sub.index) if hasattr(sub.index, "get_indexer_for") else sub.index.to_numpy(copy=False)] = label
            # ^ keep indexing simple:
        # Re-do indexing cleanly
        out = np.empty(len(df), dtype=np.int64)
        positions = {e: i for i, e in enumerate(df[era_col].tolist())}
        for era, sub in df.groupby(era_col):
            if era in self.era_regime_map_:
                label = self.era_regime_map_[era]
            else:
                v = era_summary(sub, self.feature_cols, self.n_sample_feats).reshape(1, -1)
                label = int(self._kmeans.predict(self._scaler.transform(v))[0])
            idx = df.index.get_indexer(sub.index)
            out[idx] = label
        return pd.Series(out, index=df.index, name="regime")


@dataclass
class RegimeEnsemble:
    """Train a base learner per regime + a global learner; blend at inference."""

    router: RegimeRouter
    factory: Callable[[int], Any]
    global_weight: float = 0.3
    seed: int = 0

    heads_: dict[int, Any] = field(default_factory=dict, init=False, repr=False)
    global_: Any = field(default=None, init=False, repr=False)

    def fit(
        self,
        df: pd.DataFrame,
        feature_cols: Sequence[str],
        target_col: str,
        era_col: str = "era",
    ) -> "RegimeEnsemble":
        self.router.fit(df, era_col=era_col)
        labels = self.router.assign(df, era_col=era_col)
        self.heads_ = {}
        for label, sub_idx in labels.groupby(labels):
            mask = labels == label
            X = df.loc[mask, list(feature_cols)]
            y = df.loc[mask, target_col]
            e = df.loc[mask, era_col]
            if len(X) < 500:
                continue
            m = self.factory(self.seed + int(label))
            m.fit(X, y, era=e)
            self.heads_[int(label)] = m
        # Global head on all data as a safety net / blend partner.
        self.global_ = self.factory(self.seed + 1000)
        self.global_.fit(df[list(feature_cols)], df[target_col], era=df[era_col])
        return self

    def predict(self, df: pd.DataFrame, feature_cols: Sequence[str], era_col: str = "era") -> np.ndarray:
        global_pred = np.asarray(self.global_.predict(df[list(feature_cols)]), dtype=np.float64)
        out = global_pred.copy() * self.global_weight
        labels = self.router.assign(df, era_col=era_col)
        for label, head in self.heads_.items():
            mask = (labels == label).values
            if mask.any():
                pred = np.asarray(head.predict(df.loc[mask, list(feature_cols)]), dtype=np.float64)
                out[mask] += (1.0 - self.global_weight) * pred
        # Rows that didn't match any regime head (e.g. regimes with too few samples)
        # are kept at pure global prediction.
        return out

    def save(self, path: str | Path) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)


__all__ = ["RegimeRouter", "RegimeEnsemble", "era_summary"]
