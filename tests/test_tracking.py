"""Smoke tests for experiment tracking + staking optimizer."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from numerai_stack.tracking import RunIndex, optimize_stakes, save_run


def test_run_index_round_trip(tmp_path):
    idx = RunIndex(root=tmp_path / "runs")
    run = idx.new_run(config={"target": "cyrusd_20", "lr": 0.001}, tags=["baseline"])
    run.metrics = {"mean_corr": 0.025, "mean_mmc": 0.008, "sharpe": 1.2}
    oof = pd.Series(np.random.default_rng(0).normal(size=100))
    save_run(idx, run, oof=oof)
    df = idx.load_index()
    assert len(df) == 1
    assert "metric.mean_corr" in df.columns
    assert df.loc[0, "metric.mean_corr"] == pytest.approx(0.025)


def test_stake_portfolio_prefers_better_model():
    rng = np.random.default_rng(0)
    # Model A has higher mean payout with similar vol
    rounds = 50
    a = rng.normal(0.03, 0.02, size=rounds)
    b = rng.normal(0.01, 0.02, size=rounds)
    c = rng.normal(-0.005, 0.02, size=rounds)
    payouts = pd.DataFrame({"A": a, "B": b, "C": c})
    port = optimize_stakes(payouts, mode="mean_variance", risk_aversion=3.0, shrinkage=0.1)
    assert abs(port.weights.sum() - 1.0) < 1e-6
    assert port.weights["A"] > port.weights["C"]


def test_max_sharpe_mode():
    rng = np.random.default_rng(1)
    payouts = pd.DataFrame({
        "A": rng.normal(0.02, 0.01, size=40),
        "B": rng.normal(0.01, 0.03, size=40),
    })
    port = optimize_stakes(payouts, mode="max_sharpe")
    assert abs(port.weights.sum() - 1.0) < 1e-6
    assert port.stats["sharpe"] > 0
