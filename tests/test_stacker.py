"""Smoke tests for MMC-aware stacker and ensemble post-process."""
from __future__ import annotations

import numpy as np
import pandas as pd

from numerai_stack.stack import EnsemblePostProcess, MMCAwareStacker, stacker_objective_scores


def _synth_oof(n_eras=20, rows=200, n_models=4, seed=0):
    rng = np.random.default_rng(seed)
    rows_list = []
    era_labels = []
    target_all = []
    meta_all = []
    model_cols = [f"model_{i}" for i in range(n_models)]

    signal_dim = 1
    for i in range(n_eras):
        era = f"{i + 1:04d}"
        signal = rng.normal(size=rows)
        target = pd.Series(signal + rng.normal(size=rows) * 0.5).rank(pct=True).values
        meta = signal + rng.normal(size=rows) * 0.7
        noises = rng.normal(size=(rows, n_models))
        # First model is the good one; others are noisier
        preds = np.column_stack([
            signal + 0.2 * noises[:, 0],
            signal + 0.5 * noises[:, 1],
            signal + 1.0 * noises[:, 2],
            noises[:, 3],
        ])
        df = pd.DataFrame(preds, columns=model_cols)
        df["era"] = era
        df["target"] = target
        df["meta"] = meta
        rows_list.append(df)
    return pd.concat(rows_list, ignore_index=True)


def test_mmc_stacker_prefers_strong_diverse_model():
    df = _synth_oof()
    oof = df[[c for c in df.columns if c.startswith("model_")]]
    stacker = MMCAwareStacker(lambda_mmc=2.0, diversity_penalty=0.1)
    stacker.fit(oof, df["target"], df["meta"], df["era"])
    w = stacker.weights_summary()
    assert w.sum() == pytest_approx(1.0)
    # Strongest signal model should have the largest weight
    assert w.index[0] == "model_0"


def test_mmc_stacker_predict_shape():
    df = _synth_oof()
    oof = df[[c for c in df.columns if c.startswith("model_")]]
    stacker = MMCAwareStacker(lambda_mmc=2.0, diversity_penalty=0.0)
    stacker.fit(oof, df["target"], df["meta"], df["era"])
    preds = stacker.predict(oof, df["era"])
    assert preds.shape == (len(df),)


def test_stacker_objective_scores_finite():
    df = _synth_oof()
    oof = df[[c for c in df.columns if c.startswith("model_")]].values
    from numerai_stack.stack.mmc_stacker import _era_groups, _per_era_rank_gauss

    era_idx = _era_groups(df["era"])
    oof_rg = _per_era_rank_gauss(oof, era_idx)
    blend = oof_rg.mean(axis=1)
    corr, mmc, payout = stacker_objective_scores(
        blend, df["target"].values, df["meta"].values, era_idx
    )
    for v in (corr, mmc, payout):
        assert np.isfinite(v)


def test_ensemble_post_process_applies_neutralization():
    df = _synth_oof()
    feats = ["model_2", "model_3"]  # pretend these are neutralizer features
    ep = EnsemblePostProcess(
        weights={"model_0": 0.7, "model_1": 0.3},
        neutralize_features=feats, neutralize_proportion=1.0,
    )
    out = ep(df, neutralizer_df=df)
    assert out.shape[0] == len(df)
    assert out.between(0, 1).all()


# Helper -- pytest.approx but as a value
def pytest_approx(value, tol=1e-6):
    class _Approx:
        def __eq__(self, other):
            return abs(other - value) < tol

    return _Approx()
