"""Integration test: train multiple base models on synthetic eras, fit the
MMC-aware stacker on walk-forward OOF, build a pickle, and run the smoke test.

Exercises all the phase boundaries at once.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from numerai_stack.compute import (
    build_predict_function,
    smoke_test_pickle,
    write_pickle,
)
from numerai_stack.cv.metrics import full_metrics_report
from numerai_stack.models import LightGBMTrainer
from numerai_stack.pipeline import PredictPipeline, train_final_model
from numerai_stack.pipeline.train import walk_forward_oof
from numerai_stack.stack import MMCAwareStacker


def _synth(n_eras=320, rows=80, n_features=8, seed=0):
    rng = np.random.default_rng(seed)
    beta = rng.normal(size=n_features) / np.sqrt(n_features)
    rows_list = []
    for i in range(n_eras):
        era = f"{i + 1:04d}"
        X = rng.normal(size=(rows, n_features)).astype(np.float32)
        y = X @ beta + rng.normal(size=rows) * 0.3
        meta = X @ beta + rng.normal(size=rows) * 0.4
        df = pd.DataFrame(X, columns=[f"feature_{j}" for j in range(n_features)])
        df["era"] = era
        df["target"] = pd.Series(y).rank(pct=True).values
        df["meta"] = pd.Series(meta).rank(pct=True).values
        rows_list.append(df)
    return pd.concat(rows_list, ignore_index=True)


def _fast_lgbm(seed: int):
    return LightGBMTrainer(
        params=dict(n_estimators=40, learning_rate=0.1, max_depth=3, num_leaves=8,
                    colsample_bytree=0.8, min_data_in_leaf=10, verbose=-1),
        seed=seed,
    )


def test_full_pipeline_synthetic():
    df = _synth(n_eras=320)
    feats = [c for c in df.columns if c.startswith("feature_")]
    targets = ["target"]

    # Walk-forward OOF for two different "views" of the problem
    res_a = walk_forward_oof(
        df, feats, "target", trainer_factory=lambda: _fast_lgbm(0),
        chunk_size=156, embargo=8, min_train_eras=52, verbose=False,
    )
    res_b = walk_forward_oof(
        df, feats, "target", trainer_factory=lambda: _fast_lgbm(7),
        chunk_size=156, embargo=8, min_train_eras=52, verbose=False,
    )
    oof_matrix = pd.DataFrame({"a": res_a.oof, "b": res_b.oof}).dropna()
    aligned = df.loc[oof_matrix.index]

    # Fit the MMC-aware stacker
    stacker = MMCAwareStacker(lambda_mmc=2.0, diversity_penalty=0.0)
    stacker.fit(
        oof_matrix[["a", "b"]], aligned["target"], aligned["meta"], aligned["era"],
    )
    blend = stacker.predict(oof_matrix[["a", "b"]], aligned["era"])
    report = full_metrics_report(
        blend, aligned["target"], aligned["era"], meta_model=aligned["meta"],
    )
    assert report["mean_corr"] > 0.0
    # Stacker weights sum to 1
    assert abs(stacker.weights_.sum() - 1.0) < 1e-6

    # Fit final models on full train and build + smoke-test the pickle
    final_a = train_final_model(df, feats, "target", lambda: _fast_lgbm(0))
    final_b = train_final_model(df, feats, "target", lambda: _fast_lgbm(7))
    pipe = PredictPipeline(
        base_models=[("a", final_a), ("b", final_b)],
        feature_cols=feats,
        stacker_weights=stacker.weights_.tolist(),
    )
    predict_fn = build_predict_function(pipe)
    with tempfile.TemporaryDirectory() as tmp:
        path = write_pickle(predict_fn, Path(tmp) / "m.pkl")
        out = smoke_test_pickle(path, live_features=df)
        assert out.shape == (len(df), 1)
        assert out["prediction"].between(0, 1).all()
