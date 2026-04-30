"""Meta-model residual learner.

Train a model on ``target - alpha * meta_model_preds`` (after per-era
rank-gauss) so it explicitly captures the component of target that the crowd
misses. Drop-in replacement that returns a Trainer-compatible object.
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from ..cv.metrics import rank_gauss_pow1


@dataclass
class MetaResidualTrainer:
    """Wraps a base trainer; learns on ``target - alpha * meta_model``.

    Parameters
    ----------
    factory : callable (seed) -> Trainer
        Base model factory.
    alpha : float
        Weight of meta-model subtraction (after per-era rank-gauss).
    seed : int
    """

    factory: Callable[[int], Any]
    alpha: float = 1.0
    seed: int = 0
    _model: Any = field(default=None, init=False, repr=False)

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        era: pd.Series,
        meta_model: pd.Series,
    ) -> "MetaResidualTrainer":
        # Per-era rank-gauss on target and meta_model, then subtract.
        y_rg = (
            pd.DataFrame({"y": y.values, "era": era.values}, index=y.index)
            .groupby("era", group_keys=False)["y"]
            .transform(rank_gauss_pow1)
        )
        m_rg = (
            pd.DataFrame({"m": meta_model.values, "era": era.values}, index=meta_model.index)
            .groupby("era", group_keys=False)["m"]
            .transform(rank_gauss_pow1)
        )
        y_res = y_rg - self.alpha * m_rg
        # Re-rank target to [0,1] uniform for GBM RMSE stability
        y_res = (y_res.rank(method="first") - 0.5) / len(y_res)

        self._model = self.factory(self.seed)
        self._model.fit(X, y_res, era=era)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self._model.predict(X)

    def save(self, path: str | Path) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)


__all__ = ["MetaResidualTrainer"]
