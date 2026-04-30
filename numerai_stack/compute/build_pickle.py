"""Package a PredictPipeline into a CPU-safe cloudpickle for Numerai Compute.

Numerai Compute runs the pickle in a fixed Docker image with CPU-only access.
The pickle must be a callable with the signature:

    predict(live_features: pd.DataFrame) -> pd.DataFrame

where the output has a single column of per-row predictions indexed by the
same row ids as the input.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Callable

import cloudpickle
import numpy as np
import pandas as pd

from ..pipeline.predict import PredictPipeline


def build_predict_function(pipeline: PredictPipeline) -> Callable[[pd.DataFrame], pd.DataFrame]:
    """Wrap ``PredictPipeline`` into the single-argument callable Numerai expects."""

    def predict(live_features: pd.DataFrame) -> pd.DataFrame:
        # Numerai Compute passes the full live DataFrame (including the "era"
        # column and features). The pipeline already selects what it needs.
        preds = pipeline.predict(live_features)
        return preds.to_frame("prediction")

    return predict


def _register_local_packages_by_value() -> None:
    """Make cloudpickle inline our own packages instead of importing them.

    Numerai Compute runs the pickle in a Python container with the standard
    data-stack (numpy / pandas / scikit-learn / lightgbm / xgboost / etc.) but
    NOT ``numerai_stack``. Without by-value registration the pickle stores
    ``numerai_stack`` references and load fails server-side with
    ``ModuleNotFoundError`` -- which manifests as
    ``triggerStatus=model_failed: "Model failed to generate a live output!"``.
    """
    import numerai_stack
    import numerai_stack.compute
    import numerai_stack.models
    import numerai_stack.pipeline
    import numerai_stack.stack

    for mod in (
        numerai_stack,
        numerai_stack.compute,
        numerai_stack.models,
        numerai_stack.pipeline,
        numerai_stack.stack,
    ):
        try:
            cloudpickle.register_pickle_by_value(mod)
        except Exception:
            pass


def write_pickle(predict_fn: Callable, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _register_local_packages_by_value()
    with open(path, "wb") as f:
        cloudpickle.dump(predict_fn, f)
    return path


def load_pickle(path: str | Path):
    with open(path, "rb") as f:
        return cloudpickle.load(f)


def smoke_test_pickle(
    path: str | Path,
    live_features: pd.DataFrame,
    expected_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Load and run the pickle, returning its output.

    Raises if:
    - the pickle isn't callable
    - output isn't a DataFrame with one column
    - output contains NaNs or values outside [0, 1]
    - output index doesn't match live_features
    """
    fn = load_pickle(path)
    if not callable(fn):
        raise TypeError(f"Pickle at {path} is not callable (got {type(fn)!r})")
    out = fn(live_features)
    if not isinstance(out, pd.DataFrame):
        raise TypeError(f"Pickle must return a DataFrame, got {type(out)!r}")
    if out.shape[1] != 1:
        raise ValueError(f"Pickle output must have exactly 1 column, got shape {out.shape}")
    col = out.iloc[:, 0]
    if col.isna().any():
        raise ValueError(f"Pickle produced {int(col.isna().sum())} NaN predictions")
    if (col < 0).any() or (col > 1).any():
        raise ValueError("Pickle output contains values outside [0, 1]")
    if not out.index.equals(live_features.index):
        raise ValueError("Pickle output index does not match live_features index")
    return out


__all__ = ["build_predict_function", "write_pickle", "load_pickle", "smoke_test_pickle"]
