"""Build a deployable cloudpickle from trained base models + stacker config.

Input: a YAML config that lists:
    - feature_set (small|medium|all)
    - base_models: list of {name, path_to_pickle}
    - stacker_weights: optional list (defaults to equal)
    - neutralize_feature_set: optional (e.g. medium)
    - neutralize_proportion: float

Output: runs/<run_id>/predict.pkl + a smoke-test report.
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import pandas as pd
import yaml

from numerai_stack import DEFAULT_DATA_VERSION
from numerai_stack.compute import (
    build_predict_function,
    smoke_test_pickle,
    write_pickle,
)
from numerai_stack.data import DataLoader
from numerai_stack.pipeline import PredictPipeline


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache-dir", default="data")
    ap.add_argument("--version", default=DEFAULT_DATA_VERSION)
    ap.add_argument("--skip-smoke", action="store_true")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    loader = DataLoader(cache_dir=args.cache_dir, version=args.version)
    feature_set_name = cfg.get("feature_set", "medium")
    feature_cols = loader.feature_set(feature_set_name)

    base_models: list[tuple[str, object]] = []
    for spec in cfg["base_models"]:
        with open(spec["path"], "rb") as f:
            model = pickle.load(f)
        base_models.append((spec["name"], model))

    neutralize_features = None
    if cfg.get("neutralize_feature_set"):
        neutralize_features = loader.feature_set(cfg["neutralize_feature_set"])

    pipeline = PredictPipeline(
        base_models=base_models,
        feature_cols=feature_cols,
        stacker_weights=cfg.get("stacker_weights"),
        neutralize_features=neutralize_features,
        neutralize_proportion=cfg.get("neutralize_proportion", 1.0),
    )
    predict_fn = build_predict_function(pipeline)

    out = write_pickle(predict_fn, args.out)
    print(f"wrote pickle -> {out}")

    if not args.skip_smoke:
        live = loader.load_live(features=feature_cols)
        out_df = smoke_test_pickle(out, live_features=live)
        print(f"smoke test passed: {len(out_df)} rows, preds in [{out_df.iloc[:, 0].min():.4f}, "
              f"{out_df.iloc[:, 0].max():.4f}]")


if __name__ == "__main__":
    main()
