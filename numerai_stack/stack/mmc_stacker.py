"""MMC-aware stacker.

Given per-era OOF predictions from N base models + the Meta Model on the
validation eras + validation targets, learn non-negative weights ``w`` that
maximize the proxy payout ``CORR + 2 * MMC`` while encouraging diversity
from the Meta Model.

We parameterize weights via softmax over a free vector so they stay
non-negative and sum to 1, and optimize with L-BFGS via scipy. That lets us
handle ~20 base models with a stable numerical objective.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import optimize

from ..cv.metrics import gaussianize, per_era_corr, proxy_mmc


def _softmax(x: np.ndarray) -> np.ndarray:
    z = x - x.max()
    e = np.exp(z)
    return e / e.sum()


def _era_groups(era: pd.Series) -> dict[object, np.ndarray]:
    out: dict[object, list[int]] = {}
    for i, e in enumerate(era.values):
        out.setdefault(e, []).append(i)
    return {k: np.asarray(v, dtype=np.int64) for k, v in out.items()}


def _per_era_rank_gauss(M: np.ndarray, era_idx: dict[object, np.ndarray]) -> np.ndarray:
    """Per-era rank-gauss along axis=0. M: (N_rows, N_models)."""
    out = np.empty_like(M, dtype=np.float64)
    for _, idx in era_idx.items():
        sub = M[idx]
        # Tie-kept rank
        ranked = np.argsort(np.argsort(sub, axis=0, kind="stable"), axis=0, kind="stable") + 1
        u = (ranked - 0.5) / sub.shape[0]
        # Inverse normal via scipy
        from scipy import stats

        g = stats.norm.ppf(np.clip(u, 1e-6, 1 - 1e-6))
        std = g.std(axis=0, ddof=0)
        std = np.where(std > 0, std, 1.0)
        out[idx] = g / std
    return out


def stacker_objective_scores(
    blend: np.ndarray,
    targets: np.ndarray,
    meta: np.ndarray,
    era_idx: dict[object, np.ndarray],
) -> tuple[float, float, float]:
    """Return (mean_corr, mean_mmc, payout) for a blended prediction vector.

    ``blend``, ``targets``, ``meta`` are 1D arrays aligned on row index.
    """
    corr_per_era = []
    mmc_per_era = []
    for _, idx in era_idx.items():
        b = blend[idx]
        t = targets[idx]
        m = meta[idx]
        # Rank-gauss on b and m already assumed? We recompute to be safe.
        from scipy import stats

        def rg(x):
            r = np.argsort(np.argsort(x, kind="stable"), kind="stable") + 1
            u = (r - 0.5) / len(x)
            g = stats.norm.ppf(np.clip(u, 1e-6, 1 - 1e-6))
            s = g.std(ddof=0)
            return g / s if s > 0 else g

        bn = rg(b)
        mn = rg(m)
        tc = t - t.mean()
        denom = float(mn @ mn)
        beta = float(bn @ mn) / denom if denom > 0 else 0.0
        neutral = bn - beta * mn
        corr_per_era.append(float((bn * tc).mean()))
        mmc_per_era.append(float((neutral * tc).mean()))
    mean_corr = float(np.mean(corr_per_era))
    mean_mmc = float(np.mean(mmc_per_era))
    return mean_corr, mean_mmc, mean_corr + 2.0 * mean_mmc


@dataclass
class MMCAwareStacker:
    """Learns weights over base models by maximizing CORR + lambda_mmc * MMC."""

    lambda_mmc: float = 2.0
    l2: float = 1e-4
    # Penalty on correlation of final blend with meta_model (encourages
    # diversity). 0.0 disables.
    diversity_penalty: float = 0.5
    max_iter: int = 200

    model_names: list[str] = field(default_factory=list, init=False)
    weights_: np.ndarray | None = field(default=None, init=False)

    def fit(
        self,
        oof_matrix: pd.DataFrame,
        targets: pd.Series,
        meta_model: pd.Series,
        eras: pd.Series,
    ) -> "MMCAwareStacker":
        self.model_names = list(oof_matrix.columns)
        M = oof_matrix.values.astype(np.float64)
        y = targets.values.astype(np.float64)
        m = meta_model.values.astype(np.float64)
        era_idx = _era_groups(eras)

        # Pre-normalize base model predictions per-era (rank-gauss).
        M_rg = _per_era_rank_gauss(M, era_idx)
        m_rg = _per_era_rank_gauss(m.reshape(-1, 1), era_idx).ravel()
        y_centered = np.zeros_like(y)
        for _, idx in era_idx.items():
            y_centered[idx] = y[idx] - y[idx].mean()

        lambda_mmc = self.lambda_mmc
        diversity = self.diversity_penalty
        l2 = self.l2

        def payout_loss(z):
            w = _softmax(z)
            blend = M_rg @ w  # (N_rows,)
            # Gaussianize per era again for consistency
            blend_rg = _per_era_rank_gauss(blend.reshape(-1, 1), era_idx).ravel()
            total = 0.0
            for _, idx in era_idx.items():
                b = blend_rg[idx]
                mm = m_rg[idx]
                tc = y_centered[idx]
                denom = float(mm @ mm)
                beta = float(b @ mm) / denom if denom > 0 else 0.0
                neutral = b - beta * mm
                corr = float((b * tc).mean())
                mmc = float((neutral * tc).mean())
                total += corr + lambda_mmc * mmc
            total = total / len(era_idx)
            # Diversity penalty: per-era correlation of blend with meta
            if diversity > 0:
                corrs = []
                for _, idx in era_idx.items():
                    b = blend_rg[idx]
                    mm = m_rg[idx]
                    num = float(b @ mm)
                    den = float(np.sqrt((b @ b) * (mm @ mm)) + 1e-9)
                    corrs.append(num / den)
                total -= diversity * float(np.mean(np.abs(corrs)))
            # L2 regularization on raw z to keep weights smooth
            total -= l2 * float(z @ z)
            return -total  # scipy minimizes

        z0 = np.zeros(M.shape[1], dtype=np.float64)
        res = optimize.minimize(
            payout_loss, z0, method="L-BFGS-B",
            options=dict(maxiter=self.max_iter),
        )
        self.weights_ = _softmax(res.x)
        return self

    def predict(self, base_preds: pd.DataFrame, eras: pd.Series) -> pd.Series:
        if self.weights_ is None:
            raise RuntimeError("Call fit() before predict().")
        cols = [c for c in self.model_names if c in base_preds.columns]
        missing = [c for c in self.model_names if c not in base_preds.columns]
        if missing:
            raise KeyError(f"base_preds missing stacker columns: {missing}")
        M = base_preds[cols].values.astype(np.float64)
        era_idx = _era_groups(eras)
        M_rg = _per_era_rank_gauss(M, era_idx)
        blend = M_rg @ self.weights_
        # Final per-era gaussianize
        blend_rg = _per_era_rank_gauss(blend.reshape(-1, 1), era_idx).ravel()
        return pd.Series(blend_rg, index=base_preds.index, name="stack_blend")

    def weights_summary(self) -> pd.Series:
        if self.weights_ is None:
            raise RuntimeError("Call fit() before reading weights.")
        return pd.Series(self.weights_, index=self.model_names, name="weight").sort_values(
            ascending=False
        )


__all__ = ["MMCAwareStacker", "stacker_objective_scores"]
