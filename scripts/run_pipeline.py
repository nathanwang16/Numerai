"""End-to-end driver: trains N base models, collects OOF, fits the MMC-aware
stacker, fits final models on full train, writes a prod config + cloudpickle,
and runs local validation diagnostics.

This replaces the "fit the stacker manually / write configs/prod.yaml / run
build_pickle" operator steps with a single reproducible command.
"""
from __future__ import annotations

import argparse
import gc
import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from numerai_stack import DEFAULT_DATA_VERSION
from numerai_stack.compute import (
    build_predict_function,
    smoke_test_pickle,
    validate_pickle_on_validation,
    write_pickle,
)
from numerai_stack.cv.metrics import full_metrics_report
from numerai_stack.data import DataLoader
from numerai_stack.models import (
    DEEP_LGBM_PARAMS,
    STANDARD_LGBM_PARAMS,
    LightGBMTrainer,
)
from numerai_stack.pipeline import PredictPipeline, train_final_model
from numerai_stack.pipeline.train import walk_forward_oof
from numerai_stack.stack import MMCAwareStacker
from numerai_stack.tracking import RunIndex, save_run


def _trainer_factory(params: dict, seed: int, device: str):
    def f():
        return LightGBMTrainer(params=dict(params), seed=seed, device=device)

    return f


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default=DEFAULT_DATA_VERSION)
    ap.add_argument("--cache-dir", default="data")
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--feature-set", default="medium",
                    choices=["small", "medium", "all"])
    ap.add_argument("--target", default="target")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--every-nth-era", type=int, default=2,
                    help="Downsample eras for faster dev iteration")
    ap.add_argument("--chunk-size", type=int, default=156)
    ap.add_argument("--embargo", type=int, default=8)
    ap.add_argument("--min-train-eras", type=int, default=52)
    ap.add_argument("--auto-scale-cv", action="store_true",
                    help="If the eras-after-downsample are too few to fit one "
                         "CV fold at the requested chunk/embargo/min_train, "
                         "shrink them automatically (smoke-test convenience).")
    ap.add_argument("--pickle-out", default="artifacts/predict.pkl")
    ap.add_argument("--config-out", default="configs/prod.yaml")
    ap.add_argument("--profile", default="fast",
                    choices=["fast", "standard", "deep"],
                    help="fast = 2 lightweight LGBM variants; "
                         "standard = 2 STANDARD_LGBM seeds; "
                         "deep = STANDARD + DEEP (slow on CPU)")
    args = ap.parse_args()

    loader = DataLoader(cache_dir=args.cache_dir, version=args.version)
    feats = loader.feature_set(args.feature_set)

    print(f"loading train (target={args.target}, features={len(feats)}, "
          f"every_nth_era={args.every_nth_era}) ...")
    train_df = loader.load_train(
        features=feats, targets=[args.target], every_nth_era=args.every_nth_era,
    )
    train_df = train_df.dropna(subset=[args.target])
    n_train_eras = train_df["era"].nunique()
    print(f"train rows={len(train_df):,} eras={n_train_eras}")

    needed = args.min_train_eras + args.embargo + args.chunk_size
    if n_train_eras < needed:
        if args.auto_scale_cv:
            # Reserve ~25% of eras for the test chunk, ~5% for embargo, the
            # rest for min train. Floor to small but workable values.
            chunk = max(2, n_train_eras // 4)
            embargo = max(1, n_train_eras // 20)
            min_train = max(2, n_train_eras - chunk - embargo - 1)
            print(
                f"[auto-scale-cv] {n_train_eras} eras < required {needed}; "
                f"shrinking chunk={args.chunk_size}->{chunk}, "
                f"embargo={args.embargo}->{embargo}, "
                f"min_train={args.min_train_eras}->{min_train}"
            )
            args.chunk_size, args.embargo, args.min_train_eras = chunk, embargo, min_train
        else:
            print(
                f"[warn] {n_train_eras} eras < {needed} (chunk+embargo+min_train); "
                "CV split will fail. Re-run with --auto-scale-cv to shrink "
                "the CV params for smoke testing, or pass smaller "
                "--chunk-size/--embargo/--min-train-eras explicitly."
            )

    print("loading validation ...")
    val_df = loader.load_validation(features=feats, targets=[args.target])
    val_df = val_df.dropna(subset=[args.target])
    print(f"validation rows={len(val_df):,} eras={val_df['era'].nunique()}")

    try:
        meta = loader.load_meta_model()
        print(f"meta_model rows={len(meta):,}")
    except Exception as e:
        print(f"[warn] meta_model unavailable: {e}")
        meta = None

    # Two base models whose shape depends on --profile.
    fast_a = dict(n_estimators=200, learning_rate=0.05, max_depth=5,
                  num_leaves=32, colsample_bytree=0.3, min_data_in_leaf=2000,
                  verbose=-1)
    fast_b = dict(n_estimators=200, learning_rate=0.05, max_depth=6,
                  num_leaves=64, colsample_bytree=0.3, min_data_in_leaf=2000,
                  verbose=-1)
    if args.profile == "fast":
        base_specs = [
            {"name": "lgbm_fast_a", "params": fast_a, "seed": 0},
            {"name": "lgbm_fast_b", "params": fast_b, "seed": 1},
        ]
    elif args.profile == "standard":
        base_specs = [
            {"name": "lgbm_std_s0", "params": STANDARD_LGBM_PARAMS, "seed": 0},
            {"name": "lgbm_std_s1", "params": STANDARD_LGBM_PARAMS, "seed": 1},
        ]
    else:  # deep
        base_specs = [
            {"name": "lgbm_standard", "params": STANDARD_LGBM_PARAMS, "seed": 0},
            {"name": "lgbm_deep",     "params": DEEP_LGBM_PARAMS,     "seed": 0},
        ]

    index = RunIndex(root=args.runs_dir)
    run = index.new_run(
        config={
            "target": args.target, "feature_set": args.feature_set,
            "device": args.device, "every_nth_era": args.every_nth_era,
            "chunk_size": args.chunk_size, "base_models": [s["name"] for s in base_specs],
        },
        tags=["pipeline", args.target],
    )
    run_dir = index.path(run)
    artifacts_dir = Path(args.pickle_out).parent
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    Path(args.config_out).parent.mkdir(parents=True, exist_ok=True)

    # ----- walk-forward OOF per base model -----
    oof_frames = {}
    fold_idx_ref = None
    per_base_metrics = {}
    for spec in base_specs:
        print(f"\n=== OOF for {spec['name']} ===")
        t0 = time.time()
        res = walk_forward_oof(
            train_df, feature_cols=feats, target_col=args.target,
            trainer_factory=_trainer_factory(spec["params"], spec["seed"], args.device),
            chunk_size=args.chunk_size, embargo=args.embargo,
            min_train_eras=args.min_train_eras, verbose=True,
        )
        oof_frames[spec["name"]] = res.oof
        elapsed = time.time() - t0
        mask = res.oof.notna()
        metrics = full_metrics_report(
            preds=res.oof[mask],
            targets=train_df.loc[mask, args.target],
            eras=train_df.loc[mask, "era"],
        )
        per_base_metrics[spec["name"]] = metrics
        print(f"[{spec['name']}] OOF eras={metrics['n_eras']} "
              f"mean_corr={metrics['mean_corr']:.4f} sharpe={metrics['sharpe']:.3f} "
              f"time={elapsed:.1f}s")
        if fold_idx_ref is None:
            fold_idx_ref = mask

    # Align OOF matrix -- only rows where all base models produced predictions
    oof_matrix = pd.DataFrame(oof_frames).dropna()
    aligned = train_df.loc[oof_matrix.index]
    print(f"\naligned OOF matrix: rows={len(oof_matrix)} cols={list(oof_matrix.columns)}")

    # ----- MMC-aware stacker -----
    print("\nfitting MMC-aware stacker ...")
    if meta is not None:
        aligned_meta = meta.reindex(aligned.index)
        # Any missing rows get the per-era mean as a fallback so the stacker can see them.
        aligned_meta = aligned_meta.fillna(aligned_meta.mean())
    else:
        aligned_meta = pd.Series(0.5, index=aligned.index)

    stacker = MMCAwareStacker(lambda_mmc=2.0, diversity_penalty=0.0)
    stacker.fit(
        oof_matrix=oof_matrix,
        targets=aligned[args.target],
        meta_model=aligned_meta,
        eras=aligned["era"],
    )
    weights = {n: float(w) for n, w in zip(oof_matrix.columns, stacker.weights_)}
    print(f"stacker weights: {weights}")

    blended = stacker.predict(oof_matrix, aligned["era"])
    oof_metrics = full_metrics_report(
        preds=blended,
        targets=aligned[args.target],
        eras=aligned["era"],
        meta_model=aligned_meta,
    )
    print(f"OOF (stacked) metrics: {oof_metrics}")

    # ----- fit final models on full train -----
    print("\nfitting final models on full train ...")
    final_models = []
    for spec in base_specs:
        t0 = time.time()
        m = train_final_model(
            train_df, feats, args.target,
            _trainer_factory(spec["params"], spec["seed"], args.device),
        )
        final_models.append((spec["name"], m))
        final_path = run_dir / f"final_{spec['name']}.pkl"
        with open(final_path, "wb") as f:
            pickle.dump(m, f)
        print(f"[{spec['name']}] final fit {time.time() - t0:.1f}s -> {final_path}")
        gc.collect()

    # ----- build pickle via PredictPipeline -----
    pipe = PredictPipeline(
        base_models=final_models,
        feature_cols=feats,
        stacker_weights=[weights[n] for n, _ in final_models],
    )
    predict_fn = build_predict_function(pipe)
    pickle_path = write_pickle(predict_fn, args.pickle_out)
    print(f"wrote pickle -> {pickle_path}")

    # Write config (records the exact artifacts used).
    cfg = {
        "feature_set": args.feature_set,
        "target": args.target,
        "stacker_weights": [weights[n] for n, _ in final_models],
        "base_models": [
            {"name": n, "path": str(run_dir / f"final_{n}.pkl")}
            for n, _ in final_models
        ],
    }
    with open(args.config_out, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    print(f"wrote prod config -> {args.config_out}")

    # Smoke-test the pickle on a small slice of live (or the first 2 eras of val).
    print("\nsmoke-testing pickle on live (or validation fallback) ...")
    try:
        live = loader.load_live(features=feats)
        smoke_input = live
    except Exception as e:
        print(f"[warn] live.parquet load failed: {e} - using validation head")
        smoke_input = val_df.head(2000)
    out = smoke_test_pickle(pickle_path, live_features=smoke_input)
    print(f"smoke OK: rows={len(out)} "
          f"pred_range=[{out.iloc[:, 0].min():.4f}, {out.iloc[:, 0].max():.4f}]")

    # ----- validation diagnostics -----
    print("\nrunning local diagnostics on validation ...")
    diag = validate_pickle_on_validation(
        pickle_path=pickle_path, loader=loader, feature_cols=feats,
        fnc_feature_set="medium",
        require_corr_sharpe_above=0.0,  # don't gate on sharpe for the demo run
    )
    print(f"diagnostics passed={diag.passed} reasons={diag.reasons}")
    print(f"validation metrics: {diag.metrics}")

    # ----- persist run -----
    run.metrics = {
        **{f"base.{n}.{k}": v for n, m in per_base_metrics.items() for k, v in m.items()},
        **{f"stack_oof.{k}": v for k, v in oof_metrics.items()},
        **{f"val.{k}": v for k, v in diag.metrics.items()},
        "diagnostics_passed": int(diag.passed),
    }
    run.config["stacker_weights"] = weights
    run.artifacts["pickle"] = str(pickle_path)
    run.artifacts["config"] = args.config_out
    save_run(index, run, oof=blended)
    print(f"\nsaved run {run.run_id} -> {run_dir}")


if __name__ == "__main__":
    main()
