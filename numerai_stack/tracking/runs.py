"""Experiment run index.

Each run writes into ``runs/<run_id>/`` with:
    config.yaml      - the full config used for reproducibility
    metrics.json     - validation metrics (full_metrics_report)
    oof.parquet      - out-of-fold predictions
    predict.pkl      - the canonical cloudpickle (if produced)

A top-level ``runs/index.parquet`` gets one row per run so that the user can
diff runs and answer "did this actually help?" trivially.
"""
from __future__ import annotations

import json
import os
import subprocess
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "nogit"


@dataclass
class Run:
    run_id: str
    created_at: float
    git_sha: str
    config: dict
    metrics: dict = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    artifacts: dict = field(default_factory=dict)  # relative paths

    def row(self) -> dict:
        row = {"run_id": self.run_id, "created_at": self.created_at, "git_sha": self.git_sha}
        for k, v in self.metrics.items():
            row[f"metric.{k}"] = v
        for tag in self.tags:
            row[f"tag.{tag}"] = True
        for k, v in self.artifacts.items():
            row[f"artifact.{k}"] = v
        # Flatten one level of config
        for k, v in self.config.items():
            if isinstance(v, (str, int, float, bool)) or v is None:
                row[f"cfg.{k}"] = v
        return row


@dataclass
class RunIndex:
    root: Path

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        self.root.mkdir(parents=True, exist_ok=True)

    @property
    def index_path(self) -> Path:
        return self.root / "index.parquet"

    def new_run(self, config: dict, tags: list[str] | None = None) -> Run:
        run_id = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
        run = Run(
            run_id=run_id,
            created_at=time.time(),
            git_sha=_git_sha(),
            config=config,
            tags=list(tags or []),
        )
        (self.root / run_id).mkdir(parents=True, exist_ok=True)
        return run

    def path(self, run: Run) -> Path:
        return self.root / run.run_id

    def save(self, run: Run) -> None:
        run_dir = self.path(run)
        run_dir.mkdir(parents=True, exist_ok=True)
        with open(run_dir / "config.yaml", "w") as f:
            yaml.safe_dump(run.config, f)
        with open(run_dir / "metrics.json", "w") as f:
            json.dump(run.metrics, f, indent=2, default=float)
        with open(run_dir / "run.json", "w") as f:
            json.dump(asdict(run), f, indent=2, default=str)
        # Update the top-level index
        self._append_index(run)

    def _append_index(self, run: Run) -> None:
        row = run.row()
        df_new = pd.DataFrame([row])
        if self.index_path.exists():
            df = pd.read_parquet(self.index_path)
            df = pd.concat([df, df_new], ignore_index=True)
        else:
            df = df_new
        df.to_parquet(self.index_path)

    def load_index(self) -> pd.DataFrame:
        if not self.index_path.exists():
            return pd.DataFrame()
        return pd.read_parquet(self.index_path)


def save_run(
    index: RunIndex,
    run: Run,
    oof: pd.Series | None = None,
    predict_pickle: bytes | None = None,
) -> Run:
    run_dir = index.path(run)
    if oof is not None:
        run.artifacts["oof"] = "oof.parquet"
        oof.to_frame("prediction").to_parquet(run_dir / "oof.parquet")
    if predict_pickle is not None:
        run.artifacts["predict_pkl"] = "predict.pkl"
        with open(run_dir / "predict.pkl", "wb") as f:
            f.write(predict_pickle)
    index.save(run)
    return run


__all__ = ["Run", "RunIndex", "save_run"]
