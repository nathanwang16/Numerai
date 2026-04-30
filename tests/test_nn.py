"""Smoke test for the tabular NN branch (GPU if available, CPU inference)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from numerai_stack.models.nn import TabularNNTrainer


def _synth(n_eras=8, rows=400, n_features=16, seed=0):
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


def test_mlp_trains_and_predicts_cpu_safe():
    df = _synth()
    feats = [c for c in df.columns if c.startswith("feature_")]
    m = TabularNNTrainer(
        arch="mlp", arch_kwargs=dict(hidden=(64, 32), dropout=0.1),
        epochs=6, batch_size=256, lr=3e-3, mixup_alpha=0.0, era_balanced=False,
        device="cpu", verbose=False,
    )
    m.fit(df[feats], df["target"], era=df["era"])
    preds = m.predict(df[feats])
    assert preds.shape == (len(df),)
    # Should have positive train corr
    from numerai_stack.cv.metrics import mean_corr
    assert mean_corr(pd.Series(preds), df["target"], df["era"]) > 0.2


def test_resnet_trains_cpu():
    df = _synth(n_eras=4, rows=200, n_features=8)
    feats = [c for c in df.columns if c.startswith("feature_")]
    m = TabularNNTrainer(
        arch="resnet", arch_kwargs=dict(hidden=32, n_blocks=2, dropout=0.0),
        epochs=4, batch_size=128, lr=3e-3, mixup_alpha=0.0, era_balanced=False,
        device="cpu", verbose=False,
    )
    m.fit(df[feats], df["target"], era=df["era"])
    preds = m.predict(df[feats])
    assert preds.shape == (len(df),)


def test_ft_transformer_trains_cpu_small():
    df = _synth(n_eras=3, rows=120, n_features=8)
    feats = [c for c in df.columns if c.startswith("feature_")]
    m = TabularNNTrainer(
        arch="ft", arch_kwargs=dict(d_token=16, n_heads=2, n_layers=2, dropout=0.0),
        epochs=3, batch_size=64, lr=3e-3, mixup_alpha=0.0, era_balanced=False,
        device="cpu", verbose=False,
    )
    m.fit(df[feats], df["target"], era=df["era"])
    preds = m.predict(df[feats])
    assert preds.shape == (len(df),)
