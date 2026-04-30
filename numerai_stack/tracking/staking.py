"""Multi-model staking portfolio optimizer.

Given a set of live models with historical per-round payout histories
(``CORR + 2 * MMC`` preferred), allocate stake weights via mean-variance
optimization to maximize expected payout at a target volatility.

We support two modes:
    - "mean_variance" : classical Markowitz with a risk-aversion parameter.
    - "max_sharpe"    : unconstrained (up to simplex) sharpe maximizer.

This is *portfolio-level* over models, not over assets.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy import optimize


@dataclass
class StakePortfolio:
    weights: pd.Series
    stats: dict = field(default_factory=dict)


def _project_simplex(v: np.ndarray) -> np.ndarray:
    """Project a vector onto the probability simplex (sum=1, nonneg)."""
    n = len(v)
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - 1.0
    rho = np.where(u - cssv / (np.arange(n) + 1) > 0)[0]
    if rho.size == 0:
        return np.ones(n) / n
    rho = rho[-1]
    theta = cssv[rho] / (rho + 1.0)
    return np.maximum(v - theta, 0.0)


def optimize_stakes(
    payouts: pd.DataFrame,
    mode: str = "mean_variance",
    risk_aversion: float = 5.0,
    shrinkage: float = 0.2,
) -> StakePortfolio:
    """Optimize stake weights over models given a payout history.

    Parameters
    ----------
    payouts : DataFrame
        Rows = rounds, columns = model names, values = per-round payout
        (e.g. ``CORR + 2 * MMC``).
    mode : {"mean_variance", "max_sharpe", "equal_weight"}
    risk_aversion : float
        Higher = more conservative mean-variance allocation.
    shrinkage : float in [0, 1]
        Shrinkage of the sample covariance toward a scaled identity to avoid
        over-fitting to few rounds.
    """
    mu = payouts.mean(axis=0).values.astype(np.float64)
    S = payouts.cov().values.astype(np.float64)
    if shrinkage > 0:
        n = S.shape[0]
        lam = shrinkage
        target = np.trace(S) / n * np.eye(n)
        S = (1 - lam) * S + lam * target

    if mode == "equal_weight":
        w = np.ones(len(mu)) / len(mu)
    elif mode == "mean_variance":
        def neg_util(z):
            w = _project_simplex(z)
            return -(w @ mu - 0.5 * risk_aversion * w @ S @ w)

        z0 = np.ones(len(mu)) / len(mu)
        res = optimize.minimize(neg_util, z0, method="Nelder-Mead", options=dict(maxiter=5000))
        w = _project_simplex(res.x)
    elif mode == "max_sharpe":
        def neg_sharpe(z):
            w = _project_simplex(z)
            r = w @ mu
            v = float(np.sqrt(w @ S @ w))
            return -(r / (v + 1e-9))

        z0 = np.ones(len(mu)) / len(mu)
        res = optimize.minimize(neg_sharpe, z0, method="Nelder-Mead", options=dict(maxiter=5000))
        w = _project_simplex(res.x)
    else:
        raise ValueError(f"Unknown mode {mode!r}")

    weights = pd.Series(w, index=payouts.columns, name="stake_weight")
    stats = {
        "expected_payout": float(weights.values @ mu),
        "volatility": float(np.sqrt(weights.values @ S @ weights.values)),
        "n_rounds": int(len(payouts)),
        "mode": mode,
    }
    stats["sharpe"] = stats["expected_payout"] / (stats["volatility"] + 1e-9)
    return StakePortfolio(weights=weights, stats=stats)


__all__ = ["optimize_stakes", "StakePortfolio"]
