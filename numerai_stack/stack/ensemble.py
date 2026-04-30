"""Final ensemble post-processing pipeline (per-era).

Steps (mirrors Numerai's own ensemble recipe):
    1. rank-gauss each base prediction per era
    2. weighted dot -> blend
    3. rank-gauss the blend
    4. optionally feature-neutralize to a chosen feature set
    5. rank-scale to [0, 1] for submission
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd

from ..cv.metrics import feature_neutralize
from ..models.rank_gauss import rank_gauss_blend


@dataclass
class EnsemblePostProcess:
    """Wrap rank-gauss blend + neutralization + uniform rank-scale."""

    weights: dict[str, float]
    era_col: str = "era"
    neutralize_features: list[str] | None = None
    neutralize_proportion: float = 1.0
    final_rank_uniform: bool = True

    def __call__(
        self,
        df: pd.DataFrame,
        neutralizer_df: pd.DataFrame | None = None,
    ) -> pd.Series:
        return self.apply(df, neutralizer_df=neutralizer_df)

    def apply(
        self,
        df: pd.DataFrame,
        neutralizer_df: pd.DataFrame | None = None,
    ) -> pd.Series:
        cols = list(self.weights.keys())
        w = [self.weights[c] for c in cols]
        blended = rank_gauss_blend(
            df, columns=cols, weights=w, era_col=self.era_col, gaussianize_output=True,
        )
        result = blended
        if self.neutralize_features:
            if neutralizer_df is None:
                raise ValueError("neutralize_features set but neutralizer_df is None")
            tmp = pd.DataFrame({"__p": result.values}, index=df.index)
            for c in self.neutralize_features:
                tmp[c] = neutralizer_df[c].values
            tmp[self.era_col] = df[self.era_col].values
            neut = feature_neutralize(
                tmp,
                columns=["__p"],
                neutralizers=list(self.neutralize_features),
                proportion=self.neutralize_proportion,
                era_col=self.era_col,
            )["__p"]
            result = neut
        if self.final_rank_uniform:
            result = pd.Series(result.values, index=df.index).groupby(df[self.era_col]).transform(
                lambda s: (s.rank(method="first") - 0.5) / len(s)
            )
        return result.rename("prediction")


__all__ = ["EnsemblePostProcess"]
