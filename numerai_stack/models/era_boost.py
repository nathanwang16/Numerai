"""Era boosting: upweight the worst-performing eras during training.

Adapted from Numerai forum post "Era Boosted Models":
- Train base model on all eras
- Predict on train, compute per-era correlation with target
- Keep the worst-half eras, train another stage on just those eras
- Iterate

This produces models that trade mean CORR for Sharpe across eras.
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from ..cv.metrics import per_era_corr


@dataclass
class EraBoosted:
    """Iteratively train N_ITERS boosters, each on the worst half of eras.

    The final prediction is the sum of the per-iteration predictions.
    """

    factory: Callable[[int], Any]  # (seed) -> Trainer-compatible model
    n_iters: int = 5
    proportion_worst: float = 0.5
    seed: int = 0
    stages: list[Any] = field(default_factory=list, init=False)

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        era: pd.Series,
    ) -> "EraBoosted":
        if era is None:
            raise ValueError("EraBoosted requires `era` (a pd.Series aligned with X/y).")
        self.stages = []
        current_pred = np.zeros(len(y), dtype=np.float64)

        for it in range(self.n_iters):
            if it == 0:
                mask = np.ones(len(X), dtype=bool)
            else:
                # Per-era correlation of the current additive model vs target.
                s_pred = pd.Series(current_pred, index=y.index)
                corr = per_era_corr(s_pred, y, era, method="spearman")
                threshold = corr.quantile(self.proportion_worst)
                worst = set(corr[corr <= threshold].index.tolist())
                mask = era.isin(worst).values

            model = self.factory(self.seed + it)
            y_iter = y - current_pred if it > 0 else y
            model.fit(X[mask], y_iter[mask], era=era[mask])
            self.stages.append(model)

            # Update current_pred for next round on *all* rows.
            step_pred = model.predict(X)
            # Shrink additive updates to avoid fast divergence.
            current_pred = current_pred + 0.5 * step_pred

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        total = np.zeros(X.shape[0], dtype=np.float64)
        for i, m in enumerate(self.stages):
            lr = 1.0 if i == 0 else 0.5
            total += lr * m.predict(X)
        return total

    def save(self, path: str | Path) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str | Path) -> "EraBoosted":
        with open(path, "rb") as f:
            return pickle.load(f)


__all__ = ["EraBoosted"]
