"""Smoke tests for base learners + rank-gauss + neutralize helpers."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from numerai_stack.models import (
    DEEP_LGBM_PARAMS,
    LightGBMTrainer,
    SeedAveraged,
    XGBoostTrainer,
    neutralize_per_era,
    rank_gauss_blend,
    rank_gauss_series,
)
from numerai_stack.cv.metrics import per_era_corr


def _synth(n_eras=30, rows=200, seed=0):
    rng = np.random.default_rng(seed)
    rows_list = []
    for i in range(n_eras):
        era = f"{i + 1:04d}"
        X = rng.normal(size=(rows, 8)).astype(np.float32)
        beta = rng.normal(size=8) / 2
        y = X @ beta + rng.normal(size=rows) * 0.5
        df = pd.DataFrame(X, columns=[f"feature_{j}" for j in range(8)])
        df["era"] = era
        df["target"] = pd.Series(y).rank(pct=True).values
        rows_list.append(df)
    return pd.concat(rows_list, ignore_index=True)


def test_lightgbm_trainer_fits_and_predicts():
    df = _synth()
    feats = [c for c in df.columns if c.startswith("feature_")]
    params = dict(n_estimators=100, learning_rate=0.1, max_depth=4,
                  num_leaves=16, colsample_bytree=0.8, min_data_in_leaf=10,
                  verbose=-1)
    m = LightGBMTrainer(params=params, seed=0).fit(df[feats], df["target"], era=df["era"])
    preds = m.predict(df[feats])
    assert preds.shape == (len(df),)
    corr = per_era_corr(pd.Series(preds), df["target"], df["era"]).mean()
    assert corr > 0.3


def test_seed_averaged_lgbm():
    df = _synth(n_eras=10)
    feats = [c for c in df.columns if c.startswith("feature_")]
    params = dict(n_estimators=50, learning_rate=0.1, max_depth=3,
                  num_leaves=8, colsample_bytree=1.0, min_data_in_leaf=10,
                  verbose=-1)
    m = SeedAveraged(
        factory=lambda s: LightGBMTrainer(params=params, seed=s),
        seeds=[0, 1, 2],
    ).fit(df[feats], df["target"], era=df["era"])
    preds = m.predict(df[feats])
    assert len(m.models) == 3
    assert preds.shape == (len(df),)


def test_xgboost_trainer():
    df = _synth(n_eras=10)
    feats = [c for c in df.columns if c.startswith("feature_")]
    params = dict(n_estimators=50, learning_rate=0.1, max_depth=4,
                  subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
                  objective="reg:squarederror", tree_method="hist")
    m = XGBoostTrainer(params=params, seed=0).fit(df[feats], df["target"], era=df["era"])
    preds = m.predict(df[feats])
    assert preds.shape == (len(df),)


def test_rank_gauss_blend_per_era_properties():
    df = _synth(n_eras=5)
    df["a"] = np.random.default_rng(0).normal(size=len(df))
    df["b"] = np.random.default_rng(1).normal(size=len(df))
    blend = rank_gauss_blend(df, columns=["a", "b"], weights=[0.5, 0.5])
    # Per-era std of a Gaussian-ized series should be ~1.
    for era, sub in df.groupby("era"):
        std = blend.loc[sub.index].std(ddof=0)
        assert abs(std - 1.0) < 0.1


def test_neutralize_per_era_shape():
    df = _synth(n_eras=4, rows=80)
    df["__p"] = df["feature_0"] + df["feature_1"]
    feats = [c for c in df.columns if c.startswith("feature_")]
    out = neutralize_per_era(df, columns=["__p"], neutralizers=feats, era_col="era")
    assert out.shape[0] == len(df)
    assert "__p" in out.columns
