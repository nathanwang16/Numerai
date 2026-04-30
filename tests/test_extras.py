"""Smoke tests for Phase-4 extras."""
from __future__ import annotations

import numpy as np
import pandas as pd

from numerai_stack.models import (
    BenchmarkFeatureBuilder,
    EraBoosted,
    FeatureGroupAggregates,
    LightGBMTrainer,
    MetaResidualTrainer,
    PerEraPCA,
    RegimeEnsemble,
    RegimeRouter,
    StabilitySelector,
)


def _synth(n_eras=50, rows=120, n_features=10, seed=0):
    rng = np.random.default_rng(seed)
    beta = rng.normal(size=n_features) / np.sqrt(n_features)
    out = []
    for i in range(n_eras):
        era = f"{i + 1:04d}"
        X = rng.normal(size=(rows, n_features)).astype(np.float32)
        y = X @ beta + rng.normal(size=rows) * 0.3
        meta = X @ beta + rng.normal(size=rows) * 0.5
        df = pd.DataFrame(X, columns=[f"feature_{j}" for j in range(n_features)])
        df["era"] = era
        df["target"] = pd.Series(y).rank(pct=True).values
        df["meta"] = pd.Series(meta).rank(pct=True).values
        out.append(df)
    return pd.concat(out, ignore_index=True)


def _lgbm_factory(seed: int, **overrides):
    params = dict(
        n_estimators=40, learning_rate=0.1, max_depth=3, num_leaves=8,
        colsample_bytree=0.8, min_data_in_leaf=10, verbose=-1,
    )
    params.update(overrides)
    return LightGBMTrainer(params=params, seed=seed)


def test_era_boosted_fits_and_predicts():
    df = _synth()
    feats = [c for c in df.columns if c.startswith("feature_")]
    m = EraBoosted(factory=_lgbm_factory, n_iters=3, proportion_worst=0.5, seed=0)
    m.fit(df[feats], df["target"], era=df["era"])
    preds = m.predict(df[feats])
    assert preds.shape == (len(df),)
    assert len(m.stages) == 3


def test_meta_residual_trainer():
    df = _synth()
    feats = [c for c in df.columns if c.startswith("feature_")]
    m = MetaResidualTrainer(factory=_lgbm_factory, alpha=1.0, seed=0)
    m.fit(df[feats], df["target"], era=df["era"], meta_model=df["meta"])
    preds = m.predict(df[feats])
    assert preds.shape == (len(df),)


def test_regime_router_and_ensemble():
    df = _synth(n_eras=40, rows=50)
    feats = [c for c in df.columns if c.startswith("feature_")]
    router = RegimeRouter(feature_cols=feats, n_regimes=3, n_sample_feats=5)
    ensemble = RegimeEnsemble(router=router, factory=_lgbm_factory, global_weight=0.5)
    ensemble.fit(df, feature_cols=feats, target_col="target")
    preds = ensemble.predict(df, feature_cols=feats)
    assert preds.shape == (len(df),)
    # There should be at least 1 regime head (some may be skipped due to min size)
    assert len(ensemble.heads_) >= 1


def test_stability_selector_identifies_drift():
    df = _synth()
    feats = [c for c in df.columns if c.startswith("feature_")]
    early = df[df["era"] <= "0020"]
    late = df[df["era"] > "0020"].copy()
    # Inject artificial drift into feature_0
    late["feature_0"] = late["feature_0"] + 3.0
    sel = StabilitySelector(top_fraction_to_drop=0.2).fit(early, late, feats)
    dropped = sel.drop_columns()
    assert "feature_0" in dropped


def test_per_era_pca():
    df = _synth()
    feats = [c for c in df.columns if c.startswith("feature_")]
    pca = PerEraPCA(n_components=3).fit(df, feats)
    Z = pca.transform(df)
    assert Z.shape == (len(df), 3)


def test_feature_group_aggregates():
    df = _synth()
    feats = [c for c in df.columns if c.startswith("feature_")]
    groups = {"a": feats[:3], "b": feats[3:6]}
    agg = FeatureGroupAggregates(groups=groups).transform(df)
    assert "grp_a_mean" in agg.columns
    assert "grp_b_std" in agg.columns


def test_benchmark_feature_builder_attaches():
    df = _synth()
    feats = [c for c in df.columns if c.startswith("feature_")]
    bench = pd.DataFrame(
        {"model_x": np.random.default_rng(0).uniform(size=len(df)),
         "model_y": np.random.default_rng(1).uniform(size=len(df))},
        index=df.index,
    )
    bfb = BenchmarkFeatureBuilder()
    X2 = bfb.attach(df[feats], bench)
    cols = bfb.column_names(bench)
    assert all(c in X2.columns for c in cols)
