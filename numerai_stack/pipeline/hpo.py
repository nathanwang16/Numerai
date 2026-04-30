"""Optuna-based hyperparameter search with the MMC payout as the objective.

Uses combinatorial purged CV for honest evaluation on overlapping labels.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd

from ..cv.metrics import proxy_mmc, per_era_corr
from ..cv import combinatorial_purged_splits


@dataclass
class HpoResult:
    best_params: dict
    best_value: float
    history: list[dict] = field(default_factory=list)


def run_optuna(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    target_col: str,
    meta_model: pd.Series,
    trainer_factory: Callable[[dict, int], Any],
    search_space: Callable[["optuna.trial.Trial"], dict],
    era_col: str = "era",
    n_trials: int = 30,
    n_groups: int = 6,
    n_test_groups: int = 2,
    embargo: int = 8,
    lambda_mmc: float = 2.0,
    seeds: Sequence[int] = (0,),
    direction: str = "maximize",
) -> HpoResult:
    import optuna

    eras = sorted(df[era_col].unique())
    all_splits = list(
        combinatorial_purged_splits(eras, n_groups=n_groups, n_test_groups=n_test_groups, embargo=embargo)
    )

    def objective(trial: "optuna.trial.Trial") -> float:
        params = search_space(trial)
        payouts: list[float] = []
        for split in all_splits[:4]:  # cap folds for speed
            train_mask = df[era_col].isin(split.train_eras)
            test_mask = df[era_col].isin(split.test_eras)
            if train_mask.sum() == 0 or test_mask.sum() == 0:
                continue
            preds_accum = np.zeros(int(test_mask.sum()), dtype=np.float64)
            for s in seeds:
                m = trainer_factory(params, s)
                m.fit(df.loc[train_mask, list(feature_cols)], df.loc[train_mask, target_col])
                preds_accum += m.predict(df.loc[test_mask, list(feature_cols)])
            preds = preds_accum / len(seeds)
            p = pd.Series(preds, index=df.index[test_mask])
            t = df.loc[test_mask, target_col]
            e = df.loc[test_mask, era_col]
            mm = meta_model.reindex(df.index[test_mask])
            if mm.isna().any():
                # Only validation split has meta_model -- fall back to CORR
                corr = per_era_corr(p, t, e, method="numerai").mean()
                payouts.append(float(corr))
            else:
                corr = per_era_corr(p, t, e, method="numerai").mean()
                mmc = proxy_mmc(p, t, mm, e).mean()
                payouts.append(float(corr + lambda_mmc * mmc))
        return float(np.mean(payouts)) if payouts else 0.0

    study = optuna.create_study(direction=direction)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return HpoResult(
        best_params=study.best_params,
        best_value=study.best_value,
        history=[{"params": t.params, "value": t.value} for t in study.trials],
    )


__all__ = ["run_optuna", "HpoResult"]
