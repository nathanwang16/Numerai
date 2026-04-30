"""M2 milestone: train the GPU-LightGBM baseline (deep params, seed-averaged)
on ``target``, ``cyrusd_20``, ``teager2b_20`` and score on validation.

Outputs go to ``runs/<run_id>/``: oof.parquet, metrics.json, config.yaml.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from numerai_stack import DEFAULT_DATA_VERSION
from numerai_stack.cv.metrics import full_metrics_report
from numerai_stack.data import DataLoader
from numerai_stack.models import (
    DEEP_LGBM_PARAMS,
    STANDARD_LGBM_PARAMS,
    LightGBMTrainer,
    SeedAveraged,
)
from numerai_stack.pipeline import train_final_model
from numerai_stack.pipeline.train import walk_forward_oof
from numerai_stack.tracking import RunIndex, save_run


def build_trainer_factory(params: dict, device: str, seeds: list[int]):
    def factory():
        return SeedAveraged(
            factory=lambda s: LightGBMTrainer(params=dict(params), seed=s, device=device),
            seeds=list(seeds),
        )

    return factory


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default=DEFAULT_DATA_VERSION)
    ap.add_argument("--cache-dir", default="data")
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--feature-set", default="medium",
                    choices=["small", "medium", "all"])
    ap.add_argument("--targets", nargs="+",
                    default=["target", "target_cyrusd_20", "target_teager2b_20"])
    ap.add_argument("--params", default="deep", choices=["deep", "standard"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3])
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--every-nth-era", type=int, default=1,
                    help="Subsample training eras (use 4 for fast iteration)")
    ap.add_argument("--tags", nargs="*", default=["baseline", "phase2"])
    ap.add_argument("--dry-run", action="store_true", help="Use synthetic data")
    args = ap.parse_args()

    params = DEEP_LGBM_PARAMS if args.params == "deep" else STANDARD_LGBM_PARAMS
    index = RunIndex(root=args.runs_dir)

    loader = DataLoader(cache_dir=args.cache_dir, version=args.version)

    for target in args.targets:
        print(f"\n==== training target = {target} ====")
        config = {
            "target": target, "feature_set": args.feature_set,
            "params": args.params, "device": args.device,
            "seeds": args.seeds, "every_nth_era": args.every_nth_era,
        }
        run = index.new_run(config=config, tags=[*args.tags, f"target={target}"])

        if args.dry_run:
            from tests.test_pipeline import _synth_large  # type: ignore

            train = _synth_large(n_eras=160, rows=100)
            val = _synth_large(n_eras=40, rows=100, seed=1)
            feats = [c for c in train.columns if c.startswith("feature_")]
            train_df = train
            train_df[target] = train_df["target"]
            val_df = val
            val_df[target] = val_df["target"]
            meta = None
        else:
            feats = loader.feature_set(args.feature_set)
            print(f"features={len(feats)}, target={target}")
            train_df = loader.load_train(
                features=feats, targets=[target], every_nth_era=args.every_nth_era,
            )
            val_df = loader.load_validation(features=feats, targets=[target])
            try:
                meta = loader.load_meta_model()
            except Exception as e:
                print(f"[warn] meta_model unavailable: {e}")
                meta = None

        factory = build_trainer_factory(params, args.device, args.seeds)

        print("walk-forward OOF on train ...")
        oof_res = walk_forward_oof(
            train_df, feature_cols=feats, target_col=target,
            trainer_factory=factory, chunk_size=156, embargo=8, min_train_eras=52,
            verbose=True,
        )
        oof = oof_res.oof

        # Score OOF (only on rows that got predicted)
        mask = oof.notna()
        oof_metrics = full_metrics_report(
            preds=oof[mask],
            targets=train_df.loc[mask, target],
            eras=train_df.loc[mask, "era"],
        )
        print(f"OOF metrics: {oof_metrics}")

        # Final model on full train, eval on validation
        print("fitting final model on full train ...")
        final_model = train_final_model(
            train_df, feats, target, factory,
        )
        val_pred = pd.Series(final_model.predict(val_df[feats]), index=val_df.index)
        val_target = val_df[target] if target in val_df.columns else val_df["target"]
        val_mask = val_target.notna()
        fnc_feats = None
        if not args.dry_run:
            medium = loader.feature_set("medium")
            fnc_cols = [c for c in medium if c in val_df.columns][:50]
            if fnc_cols:
                fnc_feats = val_df.loc[val_mask, fnc_cols]
        val_metrics = full_metrics_report(
            preds=val_pred[val_mask],
            targets=val_target[val_mask],
            eras=val_df.loc[val_mask, "era"],
            meta_model=meta.reindex(val_df.index)[val_mask] if meta is not None else None,
            features=fnc_feats,
        )
        print(f"VAL metrics: {val_metrics}")

        run.metrics = {
            **{f"oof.{k}": v for k, v in oof_metrics.items()},
            **{f"val.{k}": v for k, v in val_metrics.items()},
            "fit_seconds": oof_res.fit_seconds,
        }
        save_run(index, run, oof=oof)
        # Store the final model too
        with open(Path(index.path(run)) / "final_model.pkl", "wb") as f:
            import pickle
            pickle.dump(final_model, f)
        print(f"saved run {run.run_id}")


if __name__ == "__main__":
    main()
