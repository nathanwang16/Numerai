"""Feature + meta-model neutralization helpers (re-exported from cv.metrics
for convenience in the models/ layer).
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from ..cv.metrics import feature_neutralize, gaussianize


def neutralize_per_era(
    df: pd.DataFrame,
    columns: Sequence[str],
    neutralizers: Sequence[str],
    proportion: float = 1.0,
    era_col: str = "era",
) -> pd.DataFrame:
    """Proxy for ``cv.metrics.feature_neutralize`` with the standard interface."""
    return feature_neutralize(
        df, columns=columns, neutralizers=neutralizers,
        proportion=proportion, era_col=era_col,
    )


def orthogonalize_to(
    preds: pd.Series,
    reference: pd.Series,
    eras: pd.Series,
) -> pd.Series:
    """Per-era orthogonalize ``preds`` against ``reference`` in rank-gauss space.

    Used to build MMC-friendly blends: remove the component of preds aligned
    with the stake-weighted Meta Model (or any benchmark).
    """
    df = pd.DataFrame({"p": preds.values, "r": reference.values, "e": eras.values})
    out = pd.Series(index=preds.index, dtype=float)
    for era, sub in df.groupby("e"):
        p = gaussianize(sub["p"]).values
        r = gaussianize(sub["r"]).values
        denom = float(r @ r)
        beta = float(p @ r) / denom if denom > 0 else 0.0
        out.iloc[sub.index.values if isinstance(sub.index, pd.RangeIndex) else sub.index] = p - beta * r
    out.index = preds.index
    # Build correctly using alignment since index handling above is brittle.
    result = pd.Series(index=preds.index, dtype=float)
    for era, sub in df.groupby("e"):
        mask = df["e"] == era
        p = gaussianize(preds[mask.values]).values
        r = gaussianize(reference[mask.values]).values
        denom = float(r @ r)
        beta = float(p @ r) / denom if denom > 0 else 0.0
        neutral = p - beta * r
        result.loc[preds.index[mask.values]] = neutral
    return result


__all__ = ["neutralize_per_era", "orthogonalize_to"]
