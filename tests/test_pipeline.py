"""Smoke test for walk-forward OOF orchestration + predict pipeline."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from numerai_stack.models import LightGBMTrainer, SeedAveraged
from numerai_stack.pipeline import (
    PredictPipeline,
    train_final_model,
    walk_forward_oof,
)
from numerai_stack.cv.metrics import mean_corr


def _synth_large(n_eras=320, rows=80, seed=0):
    rng = np.random.default_rng(seed)
    # Stable beta across eras so walk-forward actually has signal.
    beta = rng.normal(size=6) / 2
    out = []
    for i in range(n_eras):
        era = f"{i + 1:04d}"
        X = rng.normal(size=(rows, 6)).astype(np.float32)
        y = X @ beta + rng.normal(size=rows) * 0.3
        df = pd.DataFrame(X, columns=[f"feature_{j}" for j in range(6)])
        df["era"] = era
        df["target"] = pd.Series(y).rank(pct=True).values
        out.append(df)
    return pd.concat(out, ignore_index=True)


def test_walk_forward_oof_produces_valid_oof():
    df = _synth_large(n_eras=320)
    feats = [c for c in df.columns if c.startswith("feature_")]
    params = dict(
        n_estimators=40, learning_rate=0.1, max_depth=3, num_leaves=8,
        colsample_bytree=0.8, min_data_in_leaf=10, verbose=-1,
    )

    def factory():
        return LightGBMTrainer(params=params, seed=0)

    res = walk_forward_oof(
        df, feats, "target",
        trainer_factory=factory,
        chunk_size=156, embargo=8, min_train_eras=52,
        verbose=False,
    )
    # Expect at least one fold that covers ~156 eras of test rows
    assert len(res.splits) >= 1
    assert res.oof.notna().any()
    corr = mean_corr(res.oof.dropna(), df["target"].loc[res.oof.dropna().index], df["era"].loc[res.oof.dropna().index])
    # With our synthetic signal this should be clearly positive
    assert corr > 0.1


def test_predict_pipeline_end_to_end():
    df = _synth_large(n_eras=60)
    feats = [c for c in df.columns if c.startswith("feature_")]
    params = dict(
        n_estimators=30, learning_rate=0.1, max_depth=3, num_leaves=8,
        colsample_bytree=0.8, min_data_in_leaf=10, verbose=-1,
    )
    model_a = train_final_model(df, feats, "target", lambda: LightGBMTrainer(params=params, seed=0))
    model_b = train_final_model(df, feats, "target", lambda: LightGBMTrainer(params=params, seed=1))

    pipe = PredictPipeline(
        base_models=[("lgbm_a", model_a), ("lgbm_b", model_b)],
        feature_cols=feats,
        stacker_weights=[0.5, 0.5],
        neutralize_features=None,
    )
    out = pipe(df)
    assert len(out) == len(df)
    assert out.between(0, 1).all()


def test_predict_pipeline_with_neutralization():
    df = _synth_large(n_eras=40, rows=200)
    feats = [c for c in df.columns if c.startswith("feature_")]
    params = dict(
        n_estimators=30, learning_rate=0.1, max_depth=3, num_leaves=8,
        colsample_bytree=0.8, min_data_in_leaf=10, verbose=-1,
    )
    model = train_final_model(df, feats, "target", lambda: LightGBMTrainer(params=params, seed=0))
    pipe = PredictPipeline(
        base_models=[("lgbm", model)],
        feature_cols=feats,
        stacker_weights=[1.0],
        neutralize_features=feats[:3],
        neutralize_proportion=1.0,
    )
    out = pipe(df)
    assert len(out) == len(df)
    assert out.between(0, 1).all()
