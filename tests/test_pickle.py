"""End-to-end pickle build + smoke test."""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from numerai_stack.compute import (
    DistilledStudent,
    build_predict_function,
    load_pickle,
    smoke_test_pickle,
    write_pickle,
)
from numerai_stack.models import LightGBMTrainer
from numerai_stack.pipeline import PredictPipeline, train_final_model


def _synth(n_eras=10, rows=200, n_features=8, seed=0):
    rng = np.random.default_rng(seed)
    beta = rng.normal(size=n_features) / np.sqrt(n_features)
    out = []
    for i in range(n_eras):
        era = f"{i + 1:04d}"
        X = rng.normal(size=(rows, n_features)).astype(np.float32)
        y = X @ beta + rng.normal(size=rows) * 0.3
        df = pd.DataFrame(X, columns=[f"feature_{j}" for j in range(n_features)])
        df["era"] = era
        df["target"] = pd.Series(y).rank(pct=True).values
        out.append(df)
    return pd.concat(out, ignore_index=True)


def _lgbm(seed: int):
    return LightGBMTrainer(
        params=dict(n_estimators=40, learning_rate=0.1, max_depth=3, num_leaves=8,
                    colsample_bytree=0.8, min_data_in_leaf=10, verbose=-1),
        seed=seed,
    )


def test_pickle_roundtrip_valid_output():
    df = _synth()
    feats = [c for c in df.columns if c.startswith("feature_")]
    model_a = train_final_model(df, feats, "target", lambda: _lgbm(0))
    model_b = train_final_model(df, feats, "target", lambda: _lgbm(1))
    pipe = PredictPipeline(
        base_models=[("a", model_a), ("b", model_b)],
        feature_cols=feats,
        stacker_weights=[0.5, 0.5],
    )
    predict_fn = build_predict_function(pipe)
    with tempfile.TemporaryDirectory() as tmp:
        path = write_pickle(predict_fn, Path(tmp) / "m.pkl")
        out = smoke_test_pickle(path, live_features=df)
        assert list(out.columns) == ["prediction"]
        assert out["prediction"].between(0, 1).all()
        assert out.index.equals(df.index)


def test_distilled_student_roundtrip():
    df = _synth()
    feats = [c for c in df.columns if c.startswith("feature_")]
    teacher = train_final_model(df, feats, "target", lambda: _lgbm(0))
    teacher_preds = pd.Series(teacher.predict(df[feats]), index=df.index)

    student = DistilledStudent(trainer_factory=_lgbm, seed=42)
    student.fit(df, feats, teacher_preds, era=df["era"])
    preds = student.predict(df[feats])
    assert preds.shape == (len(df),)
