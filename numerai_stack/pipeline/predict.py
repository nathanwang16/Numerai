"""The canonical prediction callable used inside the deployed pickle.

``PredictPipeline`` is intentionally a plain, cloudpickle-friendly class.
It holds:
    - a list of (name, trained_model) tuples for base learners
    - a list of target names (one per base model), used for logging only
    - a ``feature_cols`` list (the features we actually consume)
    - an optional stacker (e.g. linear weights in rank-gauss space)
    - optional neutralization config (neutralizer features + proportion)

Contract required by Numerai Compute:

    live_features: pd.DataFrame  # has "era" + feature columns
    -> pd.Series indexed by the live-features index with values in [0, 1]
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import pandas as pd

from ..cv.metrics import feature_neutralize
from ..models.rank_gauss import rank_gauss_blend, rank_gauss_series


@dataclass
class PredictPipeline:
    base_models: list[tuple[str, Any]] = field(default_factory=list)
    feature_cols: list[str] = field(default_factory=list)
    stacker_weights: list[float] | None = None
    # Features to neutralize against at inference (e.g. the 'medium' set).
    neutralize_features: list[str] | None = None
    neutralize_proportion: float = 1.0
    # If True, final output is rank-scaled to [0, 1] (what Numerai expects).
    final_rank_uniform: bool = True

    def __call__(self, live_features: pd.DataFrame) -> pd.Series:
        return self.predict(live_features)

    def predict(self, live_features: pd.DataFrame) -> pd.Series:
        X = live_features[self.feature_cols]
        era = live_features["era"] if "era" in live_features.columns else pd.Series(
            "live", index=live_features.index
        )

        # Score each base model.
        scored: dict[str, np.ndarray] = {}
        for name, model in self.base_models:
            scored[name] = np.asarray(model.predict(X), dtype=np.float64)

        blend_df = pd.DataFrame(scored, index=live_features.index)
        blend_df["__era"] = era.values

        # Rank-gauss blend with stacker weights (or equal weights).
        cols = [name for name, _ in self.base_models]
        weights = (
            list(self.stacker_weights)
            if self.stacker_weights is not None
            else [1.0 / len(cols)] * len(cols)
        )
        blended = rank_gauss_blend(
            blend_df, columns=cols, weights=weights, era_col="__era",
            gaussianize_output=True,
        )

        # Optional feature neutralization against a chosen feature set.
        if self.neutralize_features:
            neut_df = pd.DataFrame(
                {"__p": blended.values, "__era": era.values},
                index=live_features.index,
            )
            for c in self.neutralize_features:
                neut_df[c] = live_features[c].values
            neut = feature_neutralize(
                neut_df,
                columns=["__p"],
                neutralizers=list(self.neutralize_features),
                proportion=self.neutralize_proportion,
                era_col="__era",
            )["__p"]
            blended = neut

        if self.final_rank_uniform:
            result = pd.Series(blended.values, index=live_features.index).groupby(era).transform(
                lambda s: (s.rank(method="first") - 0.5) / len(s)
            )
        else:
            result = pd.Series(blended.values, index=live_features.index)
        return result.rename("prediction")


__all__ = ["PredictPipeline"]
