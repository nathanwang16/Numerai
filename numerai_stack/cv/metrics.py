"""Per-era metrics for Numerai Classic.

All metrics are computed per-era and then aggregated. Eras are the unit of
analysis; row-level correlations conflate eras and are misleading.
"""
from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# Normalization primitives
# ---------------------------------------------------------------------------

def _rank01(s: pd.Series) -> pd.Series:
    """Uniform ranks in (0, 1), ties-kept."""
    r = s.rank(method="first")
    return (r - 0.5) / len(s)


def gaussianize(s: pd.Series) -> pd.Series:
    """Rank-gauss: uniform ranks -> std normal ppf."""
    u = _rank01(s).clip(1e-6, 1 - 1e-6)
    return pd.Series(stats.norm.ppf(u.values), index=s.index)


def rank_gauss_pow1(s: pd.Series) -> pd.Series:
    """Numerai's rank-gauss with std-normalization to 1 (matches their code)."""
    g = gaussianize(s)
    std = g.std(ddof=0)
    return g / std if std > 0 else g


# ---------------------------------------------------------------------------
# Correlation metrics
# ---------------------------------------------------------------------------

def per_era_corr(
    preds: pd.Series,
    targets: pd.Series,
    eras: pd.Series,
    method: str = "spearman",
) -> pd.Series:
    """Per-era correlation between preds and targets.

    Returns a Series indexed by era (empty float64 Series if inputs are empty).
    """
    if len(preds) == 0:
        return pd.Series([], dtype=np.float64, name=None)
    df = pd.DataFrame({"p": preds.values, "t": targets.values, "e": eras.values})
    grouped = df.groupby("e", group_keys=True)
    if method == "spearman":
        out = grouped.apply(
            lambda d: d["p"].corr(d["t"], method="spearman"), include_groups=False
        )
    elif method == "pearson":
        out = grouped.apply(
            lambda d: d["p"].corr(d["t"], method="pearson"), include_groups=False
        )
    elif method == "numerai":
        # Numerai's official CORR: rank both, then Pearson on rank-gauss.
        def _c(d):
            p = rank_gauss_pow1(d["p"])
            t = d["t"] - d["t"].mean()
            return float((p.values * t.values).mean())

        out = grouped.apply(_c, include_groups=False)
    else:
        raise ValueError(f"Unknown method {method!r}")

    # groupby.apply on empty / single-column DataFrames can return a DataFrame
    # (not a Series). Coerce to a 1-D float Series.
    if isinstance(out, pd.DataFrame):
        if out.shape[1] == 0 or len(out) == 0:
            out = pd.Series([], dtype=np.float64)
        else:
            out = out.iloc[:, 0]
    return out.astype(np.float64)


def mean_corr(preds, targets, eras, method: str = "spearman") -> float:
    return float(per_era_corr(preds, targets, eras, method=method).mean())


def sharpe(series: pd.Series) -> float:
    """Era-level Sharpe ratio (mean / std)."""
    mu, sd = series.mean(), series.std(ddof=0)
    return float(mu / sd) if sd > 0 else 0.0


def hit_rate(series: pd.Series) -> float:
    return float((series > 0).mean())


def max_drawdown(series: pd.Series) -> float:
    """Max drawdown over cumulative era returns (series of per-era correlations)."""
    cum = series.cumsum()
    peak = cum.cummax()
    return float((cum - peak).min())


# ---------------------------------------------------------------------------
# Feature neutralization
# ---------------------------------------------------------------------------

def feature_neutralize(
    df: pd.DataFrame,
    columns: Sequence[str],
    neutralizers: Sequence[str],
    proportion: float = 1.0,
    era_col: str = "era",
) -> pd.DataFrame:
    """Per-era linear neutralization of ``columns`` against ``neutralizers``.

    Returns a DataFrame with the same index and ``columns`` as input,
    std-normalized per-era.

    This mirrors Numerai's reference implementation.
    """
    out_chunks = []
    unique_eras = df[era_col].unique()
    for era in unique_eras:
        df_era = df[df[era_col] == era]
        if df_era.empty:
            continue
        # Rank-gauss the scores
        scores = df_era[columns].values.astype(np.float64)
        scores2 = np.empty_like(scores)
        for j in range(scores.shape[1]):
            x = pd.Series(scores[:, j])
            r = (x.rank(method="first") - 0.5) / len(x.dropna())
            scores2[:, j] = stats.norm.ppf(r.clip(1e-6, 1 - 1e-6).values)
        scores = scores2

        exposures = (
            df_era[neutralizers]
            .astype(np.float64)
            .fillna(df_era[neutralizers].median(numeric_only=True))
            .fillna(0.5)
            .values
        )
        pinv = np.linalg.pinv(exposures.astype(np.float32), rcond=1e-6)
        scores = scores - proportion * exposures @ (pinv @ scores.astype(np.float32))
        std = np.nanstd(scores, axis=0, ddof=0)
        std = np.where(std > 0, std, 1.0)
        scores = scores / std
        out_chunks.append(
            pd.DataFrame(scores, columns=list(columns), index=df_era.index)
        )
    return pd.concat(out_chunks).reindex(df.index)


def feature_neutral_correlation(
    preds: pd.Series,
    targets: pd.Series,
    features: pd.DataFrame,
    eras: pd.Series,
    proportion: float = 1.0,
) -> pd.Series:
    """FNC per-era: corr(target, feature-neutral preds) per era."""
    df = features.copy()
    df["__p"] = preds.values
    df["__era"] = eras.values
    neut = feature_neutralize(
        df, columns=["__p"], neutralizers=list(features.columns),
        proportion=proportion, era_col="__era",
    )["__p"]
    return per_era_corr(neut, targets, eras, method="numerai")


# ---------------------------------------------------------------------------
# Meta Model Contribution (proxy)
# ---------------------------------------------------------------------------

def proxy_mmc(
    preds: pd.Series,
    targets: pd.Series,
    meta_model: pd.Series,
    eras: pd.Series,
) -> pd.Series:
    """Per-era proxy of MMC on validation eras.

    Implements Numerai's published calculation:
        1. tie-kept rank + gaussianize both preds and meta_model
        2. orthogonalize preds w.r.t. meta_model (per era)
        3. center target
        4. mmc_era = mean( orthogonalized_preds * centered_target )
    """
    df = pd.DataFrame(
        {"p": preds.values, "m": meta_model.values, "t": targets.values, "e": eras.values}
    ).dropna()
    out = {}
    for era, sub in df.groupby("e"):
        p = gaussianize(sub["p"]).values
        m = gaussianize(sub["m"]).values
        # Orthogonalize p w.r.t. m (per-era OLS, no intercept, rank-space).
        denom = float(m @ m)
        beta = float(p @ m) / denom if denom > 0 else 0.0
        neutral = p - beta * m
        t = sub["t"].values - sub["t"].values.mean()
        out[era] = float((neutral * t).mean())
    return pd.Series(out).sort_index()


# ---------------------------------------------------------------------------
# Full reporter
# ---------------------------------------------------------------------------

def full_metrics_report(
    preds: pd.Series,
    targets: pd.Series,
    eras: pd.Series,
    meta_model: pd.Series | None = None,
    features: pd.DataFrame | None = None,
    fnc_proportion: float = 1.0,
) -> dict:
    """Compute the canonical suite of per-era metrics."""
    corr = per_era_corr(preds, targets, eras, method="numerai")
    n_eras = int(corr.shape[0])
    if n_eras == 0:
        return {
            "mean_corr": float("nan"), "std_corr": float("nan"),
            "sharpe": 0.0, "hit_rate": 0.0, "max_drawdown": 0.0,
            "n_eras": 0,
        }
    report = {
        "mean_corr": float(corr.mean()),
        "std_corr": float(corr.std(ddof=0)),
        "sharpe": sharpe(corr),
        "hit_rate": hit_rate(corr),
        "max_drawdown": max_drawdown(corr),
        "n_eras": n_eras,
    }
    if meta_model is not None:
        mmc = proxy_mmc(preds, targets, meta_model, eras)
        report["mean_mmc"] = float(mmc.mean())
        report["std_mmc"] = float(mmc.std(ddof=0))
        report["sharpe_mmc"] = sharpe(mmc)
        report["payout_mean"] = report["mean_corr"] + 2.0 * report["mean_mmc"]
    if features is not None:
        fnc = feature_neutral_correlation(
            preds, targets, features=features, eras=eras, proportion=fnc_proportion
        )
        report["mean_fnc"] = float(fnc.mean())
        report["sharpe_fnc"] = sharpe(fnc)
    return report


__all__ = [
    "gaussianize",
    "rank_gauss_pow1",
    "per_era_corr",
    "mean_corr",
    "sharpe",
    "hit_rate",
    "max_drawdown",
    "feature_neutralize",
    "feature_neutral_correlation",
    "proxy_mmc",
    "full_metrics_report",
]
