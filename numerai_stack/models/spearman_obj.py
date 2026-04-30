"""Differentiable Spearman-surrogate loss for gradient boosting frameworks.

We approximate ranks with ``sigmoid((p_i - p_j) / tau)`` pairwise counts,
computed per-era. Exact soft-rank is O(n^2) per era, which is fine for
Numerai's ~5k rows/era.

Usage (LightGBM custom objective):

    params = {...}
    obj = SpearmanSurrogateObjective(era_series=train_era_series, tau=0.5)
    model = lgb.train(params, dtrain, fobj=obj, ...)

We return ``(grad, hess)`` arrays for LightGBM/XGBoost native APIs.
Because ``LGBMRegressor`` and ``XGBRegressor`` don't easily support custom
objectives + early stopping, prefer the native train API if you want this.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def _era_groups(era: pd.Series) -> list[np.ndarray]:
    idx = np.arange(len(era))
    groups: dict[object, list[int]] = {}
    for i, e in enumerate(era.values):
        groups.setdefault(e, []).append(i)
    return [np.array(v, dtype=np.int64) for v in groups.values()]


@dataclass
class SpearmanSurrogateObjective:
    """Negative soft-Spearman between predictions and target, per era averaged.

    ``tau`` controls softness; smaller = harder rank but noisier gradients.
    """

    era: pd.Series
    tau: float = 0.5

    def __post_init__(self) -> None:
        self._groups = _era_groups(self.era)

    def __call__(self, preds: np.ndarray, labels: np.ndarray):
        # LightGBM passes (preds, dataset). We handle both calling conventions.
        if hasattr(labels, "get_label"):
            y = labels.get_label()
        else:
            y = np.asarray(labels, dtype=np.float64)
        p = np.asarray(preds, dtype=np.float64)
        grad = np.zeros_like(p)
        hess = np.ones_like(p) * 1e-6
        tau = self.tau
        for idx in self._groups:
            if len(idx) < 2:
                continue
            pi = p[idx]
            yi = y[idx]
            # Soft-rank of predictions: r_i = sum_j sigmoid((pi-pj)/tau)
            diff = (pi[:, None] - pi[None, :]) / tau
            sig = 1.0 / (1.0 + np.exp(-diff))
            # Centered soft-rank and target.
            rp = sig.sum(axis=1)
            rp -= rp.mean()
            yc = yi - yi.mean()
            # Objective per era: -cov(rp, yc) / (std(rp)*std(yc))
            # For gradient tractability we use -rp @ yc / N as surrogate (= -cov).
            # d/dp_i of -sum_k rp_k * yc_k = - sum_k (drp_k/dp_i) * yc_k
            # drp_k/dp_i = sig_ki * (1-sig_ki) / tau * (delta(k==i) - 1/N)? -> complex.
            # Simpler: drp_i/dp_i = sum_{j != i} sig_ij (1 - sig_ij) / tau (diag dominant).
            sig_d = sig * (1.0 - sig) / tau
            dri = sig_d.sum(axis=1) - np.diag(sig_d)
            g_era = -yc * dri
            h_era = np.abs(yc) * dri  # crude PSD-ish approximation
            grad[idx] = g_era
            hess[idx] = np.maximum(h_era, 1e-4)
        return grad, hess


__all__ = ["SpearmanSurrogateObjective"]
