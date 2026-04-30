"""Validation-time diagnostics gate.

Two complementary functions:

1. ``validate_pickle_on_validation`` (local): runs a built pickle on the cached
   validation.parquet and asserts the key metrics don't regress below a
   configured baseline. Cheap, local, deterministic.

2. ``run_diagnostics`` (Numerai-hosted): uploads a prediction file to Numerai's
   diagnostics endpoint, which runs their own scoring suite over validation.
   Mirrors the ``run_diagnostics`` MCP tool.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..cv.metrics import full_metrics_report
from ..data.loader import DataLoader
from .credentials import get_credentials


@dataclass
class DiagnosticsResult:
    metrics: dict
    passed: bool = True
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def validate_pickle_on_validation(
    pickle_path: str | Path,
    loader: DataLoader,
    feature_cols: list[str],
    baseline: dict | None = None,
    fnc_feature_set: str | None = "medium",
    require_mmc_positive: bool = False,
    require_corr_sharpe_above: float = 0.3,
) -> DiagnosticsResult:
    """Run the pickle over the cached validation split and score it.

    ``baseline`` optionally carries a previous prod pickle's metrics dict; if
    any of our metrics drop below (baseline * tolerance) we mark the run as
    failed.
    """
    import cloudpickle

    with open(pickle_path, "rb") as f:
        predict_fn = cloudpickle.load(f)

    # Include all columns needed: features + era (target is separate).
    val = loader.load_validation(features=feature_cols, targets=("target",))
    val = val.dropna(subset=["target"])
    preds_df = predict_fn(val)
    if isinstance(preds_df, pd.DataFrame):
        preds = preds_df.iloc[:, 0]
    else:
        preds = preds_df
    preds = preds.astype(float)

    meta = loader.load_meta_model().reindex(val.index)

    feats_df = None
    if fnc_feature_set:
        fn_set = loader.feature_set(fnc_feature_set)
        feats_df = val[[c for c in fn_set if c in val.columns]]

    metrics = full_metrics_report(
        preds=preds,
        targets=val["target"],
        eras=val["era"],
        meta_model=meta if meta.notna().any() else None,
        features=feats_df,
    )

    reasons: list[str] = []
    if metrics.get("sharpe", 0.0) < require_corr_sharpe_above:
        reasons.append(
            f"corr sharpe {metrics['sharpe']:.3f} < gate {require_corr_sharpe_above}"
        )
    if require_mmc_positive and metrics.get("mean_mmc", 0.0) <= 0:
        reasons.append(f"mean_mmc {metrics.get('mean_mmc')} <= 0")
    if baseline:
        for key in ("mean_corr", "sharpe", "mean_mmc"):
            if key in baseline and key in metrics:
                if metrics[key] < baseline[key] * 0.9:
                    reasons.append(
                        f"{key} regressed: {metrics[key]:.4f} < 0.9 * {baseline[key]:.4f}"
                    )
    return DiagnosticsResult(metrics=metrics, passed=len(reasons) == 0, reasons=reasons)


def run_diagnostics(
    predictions: pd.DataFrame,
    model_id: str,
    tournament: int = 8,
    public_id: str | None = None,
    secret_key: str | None = None,
    poll_seconds: int = 15,
    timeout_seconds: int = 1800,
) -> dict[str, Any]:
    """Upload a prediction file to Numerai's diagnostics endpoint.

    ``predictions`` must be a DataFrame indexed by id with a single
    ``prediction`` column in [0, 1].
    """
    import tempfile

    from numerapi import NumerAPI

    creds = get_credentials(public_id, secret_key)
    napi = NumerAPI(public_id=creds.public_id, secret_key=creds.secret_key)

    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp:
        predictions.to_parquet(tmp.name)
        diag_id = napi.upload_diagnostics(
            file_path=tmp.name,
            model_id=model_id,
            tournament=tournament,
        )

    # Poll until done
    start = time.time()
    while time.time() - start < timeout_seconds:
        info = napi.diagnostics(model_id=model_id, diagnostics_id=diag_id)
        status = (info or {}).get("status")
        if status and status != "PENDING":
            return info
        time.sleep(poll_seconds)
    raise TimeoutError(f"Diagnostics {diag_id} timed out")


__all__ = ["validate_pickle_on_validation", "run_diagnostics", "DiagnosticsResult"]
