"""Smoke tests for the CV harness and per-era metrics.

These tests are synthetic: they don't hit the network and don't need any
parquet file downloaded. They verify the shapes, invariants, and expected
leakage-free behavior of the CV splitter.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from numerai_stack.cv import (
    combinatorial_purged_splits,
    feature_neutral_correlation,
    feature_neutralize,
    full_metrics_report,
    mean_corr,
    per_era_corr,
    proxy_mmc,
    purged_walk_forward_splits,
    sharpe,
)
from numerai_stack.cv.metrics import gaussianize, rank_gauss_pow1


def _make_synth(n_eras=400, rows_per_era=50, n_features=20, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_eras):
        era = f"{i + 1:04d}"
        X = rng.normal(size=(rows_per_era, n_features))
        beta = rng.normal(size=n_features) / np.sqrt(n_features)
        noise = rng.normal(size=rows_per_era)
        y = X @ beta + 0.5 * noise
        df = pd.DataFrame(X, columns=[f"feature_{j}" for j in range(n_features)])
        df["era"] = era
        df["target"] = pd.Series(y).rank(pct=True).values  # in [0, 1]
        rows.append(df)
    return pd.concat(rows, ignore_index=True)


def test_walk_forward_no_leak():
    eras = [f"{i:04d}" for i in range(1, 501)]
    splits = purged_walk_forward_splits(eras, chunk_size=156, embargo=8)
    assert len(splits) >= 2
    for s in splits:
        assert not set(s.train_eras) & set(s.test_eras)
        assert not set(s.train_eras) & set(s.embargo_eras)
        # embargo eras sit between train and test
        if s.embargo_eras and s.train_eras and s.test_eras:
            assert max(s.train_eras) < min(s.embargo_eras)
            assert max(s.embargo_eras) < min(s.test_eras)


def test_combinatorial_purged_splits():
    eras = [f"{i:04d}" for i in range(1, 361)]
    splits = list(combinatorial_purged_splits(eras, n_groups=6, n_test_groups=2, embargo=8))
    assert len(splits) == 15  # C(6,2)
    for s in splits:
        assert not set(s.train_eras) & set(s.test_eras)
        assert not set(s.train_eras) & set(s.embargo_eras)


def test_gaussianize_zero_mean_unit_std():
    s = pd.Series(np.random.default_rng(0).normal(size=5000))
    g = gaussianize(s)
    assert abs(g.mean()) < 0.05
    assert abs(g.std(ddof=0) - 1.0) < 0.05
    g2 = rank_gauss_pow1(s)
    assert abs(g2.std(ddof=0) - 1.0) < 1e-3


def test_per_era_metrics_on_synth():
    df = _make_synth(n_eras=60)
    # "preds" = noisy target -> high per-era corr
    preds = df["target"] + np.random.default_rng(1).normal(scale=0.2, size=len(df))
    corr = per_era_corr(pd.Series(preds), df["target"], df["era"])
    assert corr.shape[0] == 60
    assert corr.mean() > 0.5
    assert mean_corr(pd.Series(preds), df["target"], df["era"]) > 0.5


def test_proxy_mmc_orthogonal_case():
    df = _make_synth(n_eras=30)
    rng = np.random.default_rng(7)
    meta = df["target"] + rng.normal(scale=0.5, size=len(df))
    preds = df["target"] + rng.normal(scale=0.5, size=len(df))
    mmc = proxy_mmc(pd.Series(preds), df["target"], pd.Series(meta), df["era"])
    assert mmc.shape[0] == 30
    # Hard to assert sign; verify it's finite and per-era
    assert np.isfinite(mmc.values).all()


def test_feature_neutralization_reduces_exposure():
    df = _make_synth(n_eras=6, rows_per_era=300, n_features=20)
    feature_cols = [c for c in df.columns if c.startswith("feature_")]
    preds = df["feature_0"]
    neut_df = df.copy()
    neut_df["__p"] = preds.values
    out = feature_neutralize(
        neut_df, columns=["__p"], neutralizers=feature_cols, proportion=1.0
    )["__p"]
    # Neutralization should collapse the ~1.0 exposure on feature_0
    # to something much smaller per era.
    for era, sub in df.groupby("era"):
        before = abs(np.corrcoef(preds.loc[sub.index].values, sub["feature_0"].values)[0, 1])
        after = abs(np.corrcoef(out.loc[sub.index].values, sub["feature_0"].values)[0, 1])
        assert before > 0.9, f"sanity check failed, before={before}"
        assert after < 0.2, f"era {era}: exposure after neutralization = {after}"


def test_full_metrics_report_contains_expected_keys():
    df = _make_synth(n_eras=20)
    preds = df["target"] + np.random.default_rng(0).normal(scale=0.3, size=len(df))
    meta = df["target"] + np.random.default_rng(1).normal(scale=0.3, size=len(df))
    feats = df[[c for c in df.columns if c.startswith("feature_")]].iloc[:, :5]
    report = full_metrics_report(
        pd.Series(preds), df["target"], df["era"],
        meta_model=pd.Series(meta), features=feats,
    )
    for k in ["mean_corr", "sharpe", "mean_mmc", "payout_mean", "mean_fnc"]:
        assert k in report
