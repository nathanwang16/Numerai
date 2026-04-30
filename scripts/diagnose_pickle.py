"""Cheap standalone diagnostics: score an already-built pickle on validation.

Uses a configurable small FNC feature cap so it completes in minutes even for
the medium feature set.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from numerai_stack import DEFAULT_DATA_VERSION
from numerai_stack.compute import load_pickle, validate_pickle_on_validation
from numerai_stack.cv.metrics import full_metrics_report
from numerai_stack.data import DataLoader


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pickle", required=True)
    ap.add_argument("--cache-dir", default="data")
    ap.add_argument("--version", default=DEFAULT_DATA_VERSION)
    ap.add_argument("--feature-set", default="medium")
    ap.add_argument("--fnc-features", type=int, default=30,
                    help="Cap number of features used for FNC (per-era "
                         "pseudo-inverse). Set 0 to skip FNC.")
    ap.add_argument("--max-eras", type=int, default=0,
                    help="Cap number of validation eras scored. 0 = all.")
    ap.add_argument("--target", default="target")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    loader = DataLoader(cache_dir=args.cache_dir, version=args.version)
    feats = loader.feature_set(args.feature_set)
    print(f"loading validation (features={len(feats)}) ...")
    val = loader.load_validation(features=feats, targets=[args.target])
    val = val.dropna(subset=[args.target])

    if args.max_eras > 0:
        eras_keep = sorted(val["era"].unique())[: args.max_eras]
        val = val[val["era"].isin(eras_keep)]
    print(f"validation rows={len(val):,} eras={val['era'].nunique()}")

    print("running predict ...")
    t0 = time.time()
    fn = load_pickle(args.pickle)
    preds_df = fn(val)
    pred_t = time.time() - t0
    print(f"predict took {pred_t:.1f}s; preds shape={preds_df.shape}")

    preds = preds_df.iloc[:, 0]

    try:
        meta = loader.load_meta_model().reindex(val.index)
        meta_ok = meta.notna().any()
    except Exception as e:
        print(f"[warn] meta_model unavailable: {e}")
        meta = None
        meta_ok = False

    fnc_feats = None
    if args.fnc_features > 0:
        cols = [c for c in feats if c in val.columns][: args.fnc_features]
        fnc_feats = val[cols]
        print(f"FNC using {len(cols)} features")
    else:
        print("FNC skipped")

    t0 = time.time()
    metrics = full_metrics_report(
        preds=preds, targets=val[args.target], eras=val["era"],
        meta_model=meta if meta_ok else None, features=fnc_feats,
    )
    print(f"metrics computed in {time.time() - t0:.1f}s")
    print(json.dumps(metrics, indent=2, default=float))

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(metrics, indent=2, default=float))
        print(f"wrote -> {args.out}")


if __name__ == "__main__":
    main()
