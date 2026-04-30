"""Knowledge distillation: compress a heavy ensemble into a small student.

We use the full ensemble's OOF predictions as soft labels and train a compact
student (small LightGBM by default). This cuts pickle size and CPU inference
time at the cost of a small bit of signal.

Usage:

    teacher_oof = <pd.Series from running the heavy ensemble over train/validation>
    student = DistilledStudent(trainer_factory=lambda s: LightGBMTrainer(...))
    student.fit(train_df, features, teacher_oof, era=train_df['era'])
    student.predict(live_df[features])
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd


@dataclass
class DistilledStudent:
    trainer_factory: Callable[[int], Any]
    seed: int = 0
    _model: Any = field(default=None, init=False, repr=False)

    def fit(
        self,
        df: pd.DataFrame,
        feature_cols: Sequence[str],
        teacher_preds: pd.Series,
        era: pd.Series | None = None,
    ) -> "DistilledStudent":
        # Use teacher predictions as soft labels; rank-percent to bound in [0, 1]
        soft = teacher_preds.rank(pct=True)
        soft = soft.reindex(df.index)
        self._model = self.trainer_factory(self.seed)
        self._model.fit(df[list(feature_cols)], soft, era=era if era is not None else df.get("era"))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self._model.predict(X)

    def save(self, path: str | Path) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str | Path) -> "DistilledStudent":
        with open(path, "rb") as f:
            return pickle.load(f)


__all__ = ["DistilledStudent"]
