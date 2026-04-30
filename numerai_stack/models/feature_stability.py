"""Adversarial feature-stability filter.

Given (early_eras_df, late_eras_df) with the same features, train a classifier
to distinguish early from late rows. Features with high permutation importance
(or classifier weight) are the ones that shifted most; drop them.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


@dataclass
class StabilitySelector:
    top_fraction_to_drop: float = 0.1
    penalty: str = "l2"
    C: float = 1.0
    random_state: int = 0

    def fit(
        self,
        early_df: pd.DataFrame,
        late_df: pd.DataFrame,
        feature_cols: list[str],
    ) -> "StabilitySelector":
        X = pd.concat(
            [
                early_df[feature_cols].astype(np.float32),
                late_df[feature_cols].astype(np.float32),
            ],
            ignore_index=True,
        ).fillna(0.5).values
        y = np.concatenate([
            np.zeros(len(early_df), dtype=np.int64),
            np.ones(len(late_df), dtype=np.int64),
        ])
        clf = LogisticRegression(
            penalty=self.penalty, C=self.C, max_iter=2000, solver="liblinear",
            random_state=self.random_state,
        ).fit(X, y)
        coefs = np.abs(clf.coef_).ravel()
        self.feature_cols_ = list(feature_cols)
        self.scores_ = pd.Series(coefs, index=feature_cols, name="shift_score").sort_values(
            ascending=False
        )
        return self

    def drop_columns(self) -> list[str]:
        n_drop = max(1, int(len(self.feature_cols_) * self.top_fraction_to_drop))
        return list(self.scores_.head(n_drop).index)

    def keep_columns(self) -> list[str]:
        drop = set(self.drop_columns())
        return [c for c in self.feature_cols_ if c not in drop]


__all__ = ["StabilitySelector"]
