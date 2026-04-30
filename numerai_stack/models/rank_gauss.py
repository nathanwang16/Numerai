"""Rank-gauss normalization + ensemble helper.

These are the exact per-era post-processing primitives Numerai recommends for
ensembling multiple predictions (see the Models doc).
"""
from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy import stats


def rank_gauss_series(s: pd.Series, clip: float = 1e-6) -> pd.Series:
    """Tie-kept rank -> uniform -> inverse-normal -> std-normalize to 1."""
    r = s.rank(method="first")
    u = (r - 0.5) / len(s)
    g = stats.norm.ppf(u.clip(clip, 1 - clip).values)
    std = g.std(ddof=0)
    if std > 0:
        g = g / std
    return pd.Series(g, index=s.index, name=s.name)


def _group_rank_gauss(df: pd.DataFrame, columns: Sequence[str], era_col: str) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        out[col] = (
            out.groupby(era_col, group_keys=False)[col]
            .transform(rank_gauss_series)
        )
    return out


def rank_gauss_blend(
    df: pd.DataFrame,
    columns: Sequence[str],
    weights: Sequence[float] | None = None,
    era_col: str = "era",
    gaussianize_output: bool = True,
) -> pd.Series:
    """Per-era rank-gauss each column, weighted dot, optional final gaussianize.

    Returns a Series with the blended predictions, indexed like ``df``.
    """
    weights = list(weights) if weights is not None else [1.0 / len(columns)] * len(columns)
    if len(weights) != len(columns):
        raise ValueError("weights/columns length mismatch")
    norm = _group_rank_gauss(df, list(columns), era_col=era_col)
    blended = norm[list(columns)].values @ np.asarray(weights, dtype=np.float64)
    s = pd.Series(blended, index=df.index)
    if gaussianize_output:
        s = (
            pd.DataFrame({"__p": s.values, era_col: df[era_col].values}, index=df.index)
            .groupby(era_col, group_keys=False)["__p"]
            .transform(rank_gauss_series)
        )
    return s.rename("blend")


__all__ = ["rank_gauss_series", "rank_gauss_blend"]
