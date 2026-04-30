"""Gradient-boosted tree trainers for Numerai.

Exposes a uniform ``Trainer`` protocol:
    - fit(X, y, era=None, eval_set=None)
    - predict(X) -> np.ndarray
    - save(path), load(path)

Plus a ``SeedAveraged`` wrapper that trains the same config under N seeds
and averages predictions at inference time.
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Standard Numerai benchmark params (from https://docs.numer.ai/.../models)
# ---------------------------------------------------------------------------

STANDARD_LGBM_PARAMS: dict[str, Any] = dict(
    n_estimators=2000,
    learning_rate=0.01,
    max_depth=5,
    num_leaves=2 ** 5,
    colsample_bytree=0.1,
    min_data_in_leaf=10000,
    verbose=-1,
)

DEEP_LGBM_PARAMS: dict[str, Any] = dict(
    n_estimators=30000,
    learning_rate=0.001,
    max_depth=10,
    num_leaves=1024,
    colsample_bytree=0.1,
    min_data_in_leaf=10000,
    verbose=-1,
)


class Trainer(Protocol):
    def fit(self, X: pd.DataFrame, y: pd.Series, era: pd.Series | None = None) -> "Trainer": ...
    def predict(self, X: pd.DataFrame) -> np.ndarray: ...
    def save(self, path: str | Path) -> None: ...


# ---------------------------------------------------------------------------
# LightGBM
# ---------------------------------------------------------------------------

@dataclass
class LightGBMTrainer:
    params: dict = field(default_factory=lambda: dict(STANDARD_LGBM_PARAMS))
    seed: int = 0
    device: str = "cpu"  # "cpu" or "cuda"
    categorical_features: list[str] | None = None
    _model: Any = field(default=None, init=False, repr=False)

    def _merged_params(self) -> dict:
        p = dict(self.params)
        p.setdefault("random_state", self.seed)
        p.setdefault("seed", self.seed)
        # LightGBM CUDA device name
        if self.device == "cuda":
            p.setdefault("device", "cuda")
            p.setdefault("gpu_use_dp", False)
        return p

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        era: pd.Series | None = None,
        eval_set: tuple[pd.DataFrame, pd.Series] | None = None,
        sample_weight: np.ndarray | pd.Series | None = None,
        callbacks: list | None = None,
    ) -> "LightGBMTrainer":
        import lightgbm as lgb

        params = self._merged_params()
        n_estimators = params.pop("n_estimators")
        learning_rate = params.pop("learning_rate", 0.01)

        # Drop rows with NaN target (can happen with auxiliary targets)
        mask = ~y.isna()
        if mask.sum() < len(y):
            X = X[mask.values]
            y = y[mask.values]
            if sample_weight is not None:
                sample_weight = np.asarray(sample_weight)[mask.values]

        params["learning_rate"] = learning_rate
        self._model = lgb.LGBMRegressor(n_estimators=n_estimators, **params)
        fit_kwargs: dict[str, Any] = {}
        if eval_set is not None:
            fit_kwargs["eval_set"] = [eval_set]
        if callbacks:
            fit_kwargs["callbacks"] = callbacks
        if sample_weight is not None:
            fit_kwargs["sample_weight"] = np.asarray(sample_weight, dtype=np.float64)
        self._model.fit(X.values.astype(np.float32), y.values.astype(np.float32), **fit_kwargs)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self._model.predict(X.values.astype(np.float32))

    def feature_importance(self) -> np.ndarray:
        return np.asarray(self._model.booster_.feature_importance(importance_type="gain"))

    def save(self, path: str | Path) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str | Path) -> "LightGBMTrainer":
        with open(path, "rb") as f:
            return pickle.load(f)


# ---------------------------------------------------------------------------
# XGBoost
# ---------------------------------------------------------------------------

@dataclass
class XGBoostTrainer:
    params: dict = field(
        default_factory=lambda: dict(
            n_estimators=2000,
            learning_rate=0.01,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.1,
            min_child_weight=100,
            reg_lambda=1.0,
            objective="reg:squarederror",
            tree_method="hist",
        )
    )
    seed: int = 0
    device: str = "cpu"  # "cpu" or "cuda"
    _model: Any = field(default=None, init=False, repr=False)

    def _merged_params(self) -> dict:
        p = dict(self.params)
        p.setdefault("random_state", self.seed)
        if self.device == "cuda":
            p["device"] = "cuda"
        return p

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        era: pd.Series | None = None,
        eval_set: tuple[pd.DataFrame, pd.Series] | None = None,
        sample_weight: np.ndarray | pd.Series | None = None,
    ) -> "XGBoostTrainer":
        import xgboost as xgb

        params = self._merged_params()
        n_estimators = params.pop("n_estimators")

        mask = ~y.isna()
        if mask.sum() < len(y):
            X = X[mask.values]
            y = y[mask.values]
            if sample_weight is not None:
                sample_weight = np.asarray(sample_weight)[mask.values]

        self._model = xgb.XGBRegressor(n_estimators=n_estimators, **params)
        fit_kwargs: dict[str, Any] = {}
        if eval_set is not None:
            fit_kwargs["eval_set"] = [eval_set]
            fit_kwargs["verbose"] = False
        if sample_weight is not None:
            fit_kwargs["sample_weight"] = np.asarray(sample_weight, dtype=np.float64)
        self._model.fit(X.values.astype(np.float32), y.values.astype(np.float32), **fit_kwargs)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self._model.predict(X.values.astype(np.float32))

    def save(self, path: str | Path) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str | Path) -> "XGBoostTrainer":
        with open(path, "rb") as f:
            return pickle.load(f)


# ---------------------------------------------------------------------------
# CatBoost
# ---------------------------------------------------------------------------

@dataclass
class CatBoostTrainer:
    params: dict = field(
        default_factory=lambda: dict(
            iterations=2000,
            learning_rate=0.03,
            depth=6,
            l2_leaf_reg=3.0,
            loss_function="RMSE",
            verbose=False,
        )
    )
    seed: int = 0
    device: str = "cpu"  # "cpu" or "cuda"
    _model: Any = field(default=None, init=False, repr=False)

    def _merged_params(self) -> dict:
        p = dict(self.params)
        p.setdefault("random_seed", self.seed)
        if self.device == "cuda":
            p["task_type"] = "GPU"
            p["devices"] = "0"
        return p

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        era: pd.Series | None = None,
        eval_set: tuple[pd.DataFrame, pd.Series] | None = None,
        sample_weight: np.ndarray | pd.Series | None = None,
    ) -> "CatBoostTrainer":
        from catboost import CatBoostRegressor, Pool

        mask = ~y.isna()
        if mask.sum() < len(y):
            X = X[mask.values]
            y = y[mask.values]
            if sample_weight is not None:
                sample_weight = np.asarray(sample_weight)[mask.values]

        params = self._merged_params()
        self._model = CatBoostRegressor(**params)
        train_pool = Pool(
            X.values.astype(np.float32),
            label=y.values.astype(np.float32),
            weight=None if sample_weight is None else np.asarray(sample_weight, dtype=np.float64),
        )
        eval_pool = None
        if eval_set is not None:
            eval_pool = Pool(
                eval_set[0].values.astype(np.float32),
                label=eval_set[1].values.astype(np.float32),
            )
        self._model.fit(train_pool, eval_set=eval_pool, verbose=False)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self._model.predict(X.values.astype(np.float32))

    def save(self, path: str | Path) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str | Path) -> "CatBoostTrainer":
        with open(path, "rb") as f:
            return pickle.load(f)


# ---------------------------------------------------------------------------
# Seed averaging
# ---------------------------------------------------------------------------

@dataclass
class SeedAveraged:
    """Train the same trainer class under multiple seeds; average predictions."""

    factory: Any  # callable(seed: int) -> trainer instance
    seeds: list[int] = field(default_factory=lambda: [0, 1, 2, 3])
    models: list[Any] = field(default_factory=list, init=False)

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        era: pd.Series | None = None,
        eval_set: tuple[pd.DataFrame, pd.Series] | None = None,
        sample_weight: np.ndarray | pd.Series | None = None,
    ) -> "SeedAveraged":
        self.models = []
        for s in self.seeds:
            m = self.factory(s)
            m.fit(X, y, era=era, eval_set=eval_set, sample_weight=sample_weight)
            self.models.append(m)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        preds = np.stack([m.predict(X) for m in self.models], axis=0)
        return preds.mean(axis=0)

    def save(self, path: str | Path) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str | Path) -> "SeedAveraged":
        with open(path, "rb") as f:
            return pickle.load(f)


__all__ = [
    "STANDARD_LGBM_PARAMS",
    "DEEP_LGBM_PARAMS",
    "LightGBMTrainer",
    "XGBoostTrainer",
    "CatBoostTrainer",
    "SeedAveraged",
]
