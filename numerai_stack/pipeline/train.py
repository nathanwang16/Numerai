"""Training orchestration: walk-forward OOF + final full-fit.

Given a ``trainer_factory(seed) -> Trainer`` and a dataframe with ``era`` +
features + target, produces:
    - OOF predictions (one row per input row, filled only for walk-forward
      test folds)
    - A dict of per-fold full-fit Trainer objects keyed by fold index
    - Optionally, a final trainer fit on all data (for deployment).
"""
from __future__ import annotations

import gc
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd

from ..cv import WalkForwardSplit, purged_walk_forward_splits


@dataclass
class WalkForwardResult:
    oof: pd.Series
    per_fold_models: dict[int, Any]
    splits: list[WalkForwardSplit]
    fit_seconds: float
    meta: dict = field(default_factory=dict)


def walk_forward_oof(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    target_col: str,
    trainer_factory: Callable[[], Any],
    chunk_size: int = 156,
    embargo: int = 8,
    era_col: str = "era",
    min_train_eras: int = 52,
    save_models: bool = False,
    verbose: bool = True,
) -> WalkForwardResult:
    """Run walk-forward CV, returning OOF predictions.

    ``trainer_factory`` is a zero-arg callable that returns a fresh trainer.
    Use ``lambda: SeedAveraged(factory=..., seeds=[0,1,2,3])`` for seed
    averaging.
    """
    eras = sorted(df[era_col].unique().tolist())
    splits = purged_walk_forward_splits(
        eras, chunk_size=chunk_size, embargo=embargo, min_train_eras=min_train_eras,
    )
    if not splits:
        raise ValueError(
            f"No CV folds generated from {len(eras)} eras with chunk={chunk_size}, "
            f"embargo={embargo}, min_train={min_train_eras}. "
            "Check data coverage (Numerai train alone may not span >{min_train}+chunk eras)."
        )

    oof = pd.Series(np.nan, index=df.index, dtype=np.float64)
    per_fold: dict[int, Any] = {}
    start = time.time()
    for split in splits:
        train_mask = df[era_col].isin(split.train_eras)
        test_mask = df[era_col].isin(split.test_eras)
        if train_mask.sum() == 0 or test_mask.sum() == 0:
            continue
        X_tr = df.loc[train_mask, list(feature_cols)]
        y_tr = df.loc[train_mask, target_col]
        X_te = df.loc[test_mask, list(feature_cols)]
        era_tr = df.loc[train_mask, era_col]

        if verbose:
            print(
                f"[fold {split.fold}] train={train_mask.sum():,} rows "
                f"({len(split.train_eras)} eras), test={test_mask.sum():,} rows "
                f"({len(split.test_eras)} eras)"
            )

        model = trainer_factory()
        model.fit(X_tr, y_tr, era=era_tr)
        preds = model.predict(X_te)
        oof.loc[df.index[test_mask]] = preds

        if save_models:
            per_fold[split.fold] = model
        else:
            del model
        gc.collect()

    elapsed = time.time() - start
    return WalkForwardResult(
        oof=oof,
        per_fold_models=per_fold,
        splits=splits,
        fit_seconds=elapsed,
        meta=dict(
            target=target_col,
            n_features=len(feature_cols),
            n_splits=len(splits),
        ),
    )


def train_final_model(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    target_col: str,
    trainer_factory: Callable[[], Any],
    era_col: str = "era",
    holdout_eras: int = 0,
) -> Any:
    """Fit a single model on the full dataset (minus an optional tail holdout).

    ``holdout_eras`` > 0 leaves the most recent N eras out, which is useful if
    you want a small validation signal on the freshest data before deploying.
    """
    df_sorted_eras = sorted(df[era_col].unique().tolist())
    if holdout_eras > 0 and holdout_eras < len(df_sorted_eras):
        train_eras = set(df_sorted_eras[:-holdout_eras])
        train_mask = df[era_col].isin(train_eras)
    else:
        train_mask = pd.Series(True, index=df.index)

    model = trainer_factory()
    model.fit(
        df.loc[train_mask, list(feature_cols)],
        df.loc[train_mask, target_col],
        era=df.loc[train_mask, era_col],
    )
    return model


__all__ = ["WalkForwardResult", "walk_forward_oof", "train_final_model"]
