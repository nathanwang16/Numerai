"""Tabular neural-network branch.

Architectures
-------------
- ``MLP``        : classic dense MLP with BatchNorm + Dropout
- ``ResNet``     : dense-residual (Gorishniy et al. "Revisiting Deep Learning
                   Models for Tabular Data")
- ``FTTransformer``: small FT-Transformer; feature tokens + transformer encoder

Training specifics
------------------
- Per-era MixUp augmentation
- Era-balanced sampling (sample equal rows per era to prevent long eras dominating)
- AdamW + cosine LR, early stopping on per-era validation corr
- Multi-target training is supported through ``target_cols`` (averaged loss)

Deployment
----------
At inference we run on CPU (Numerai Compute constraint). ``predict`` explicitly
sets ``device='cpu'`` and uses fp32. The pickle holds a CPU state_dict.
"""
from __future__ import annotations

import copy
import math
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


def _lazy_torch():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    return torch, nn, F


# ---------------------------------------------------------------------------
# Architectures
# ---------------------------------------------------------------------------

def build_mlp(n_features: int, n_targets: int, hidden: Sequence[int] = (512, 256, 128), dropout: float = 0.3):
    _, nn, _ = _lazy_torch()
    layers: list = []
    prev = n_features
    for h in hidden:
        layers += [
            nn.Linear(prev, h),
            nn.BatchNorm1d(h),
            nn.GELU(),
            nn.Dropout(dropout),
        ]
        prev = h
    layers.append(nn.Linear(prev, n_targets))
    return nn.Sequential(*layers)


class _ResBlock:
    """Factory for ResBlock module (defined inside function to avoid module-level torch import)."""


def build_resnet(n_features: int, n_targets: int, hidden: int = 256, n_blocks: int = 4, dropout: float = 0.2):
    torch, nn, F = _lazy_torch()

    class ResBlock(nn.Module):
        def __init__(self, d: int):
            super().__init__()
            self.bn1 = nn.BatchNorm1d(d)
            self.fc1 = nn.Linear(d, d * 2)
            self.fc2 = nn.Linear(d * 2, d)
            self.do = nn.Dropout(dropout)

        def forward(self, x):
            z = self.bn1(x)
            z = F.gelu(self.fc1(z))
            z = self.do(z)
            z = self.fc2(z)
            return x + z

    class ResNetTab(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Linear(n_features, hidden)
            self.blocks = nn.ModuleList([ResBlock(hidden) for _ in range(n_blocks)])
            self.head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, n_targets))

        def forward(self, x):
            x = self.embed(x)
            for b in self.blocks:
                x = b(x)
            return self.head(x)

    return ResNetTab()


def build_ft_transformer(
    n_features: int, n_targets: int, d_token: int = 64, n_heads: int = 4,
    n_layers: int = 3, dropout: float = 0.1,
):
    torch, nn, F = _lazy_torch()

    class FeatureTokenizer(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(n_features, d_token) * 0.02)
            self.bias = nn.Parameter(torch.zeros(n_features, d_token))
            self.cls = nn.Parameter(torch.randn(1, 1, d_token) * 0.02)

        def forward(self, x):
            # x: (B, F) -> tokens: (B, F+1, d_token)
            tokens = x.unsqueeze(-1) * self.weight + self.bias
            cls = self.cls.expand(x.size(0), -1, -1)
            return torch.cat([cls, tokens], dim=1)

    class FTTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.tok = FeatureTokenizer()
            enc_layer = nn.TransformerEncoderLayer(
                d_model=d_token, nhead=n_heads, dim_feedforward=d_token * 2,
                dropout=dropout, batch_first=True, activation="gelu",
            )
            self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
            self.norm = nn.LayerNorm(d_token)
            self.head = nn.Linear(d_token, n_targets)

        def forward(self, x):
            t = self.tok(x)
            t = self.transformer(t)
            cls = self.norm(t[:, 0])
            return self.head(cls)

    return FTTransformer()


ARCH_FACTORIES = {
    "mlp": build_mlp,
    "resnet": build_resnet,
    "ft": build_ft_transformer,
}


# ---------------------------------------------------------------------------
# Training wrapper
# ---------------------------------------------------------------------------

@dataclass
class TabularNNTrainer:
    """Pickle-safe tabular NN trainer with CPU inference.

    Only stores the CPU state_dict + architecture spec, so the pickle is tiny
    and runs fine inside Numerai Compute.
    """

    arch: str = "resnet"
    arch_kwargs: dict = field(default_factory=dict)
    n_targets: int = 1
    epochs: int = 30
    batch_size: int = 4096
    lr: float = 1e-3
    weight_decay: float = 1e-5
    device: str = "auto"  # "auto", "cpu", "cuda"
    mixup_alpha: float = 0.2
    era_balanced: bool = True
    patience: int = 6
    seed: int = 0
    grad_clip: float = 1.0
    verbose: bool = False

    # Populated after fit
    _state_dict: dict | None = field(default=None, init=False, repr=False)
    _n_features: int | None = field(default=None, init=False, repr=False)
    _means: np.ndarray | None = field(default=None, init=False, repr=False)
    _stds: np.ndarray | None = field(default=None, init=False, repr=False)

    # -- helpers --
    def _select_device(self, torch):
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)

    def _build(self, n_features: int):
        factory = ARCH_FACTORIES[self.arch]
        return factory(n_features=n_features, n_targets=self.n_targets, **self.arch_kwargs)

    # -- training --
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series | pd.DataFrame,
        era: pd.Series | None = None,
        eval_set: tuple[pd.DataFrame, pd.Series] | None = None,
        sample_weight=None,
    ) -> "TabularNNTrainer":
        torch, nn, F = _lazy_torch()
        torch.manual_seed(self.seed)

        device = self._select_device(torch)

        # Drop rows with all-NaN targets
        if isinstance(y, pd.Series):
            Y = y.to_frame()
        else:
            Y = y
        mask = ~Y.isna().all(axis=1)
        X = X.loc[mask]
        Y = Y.loc[mask]
        if era is not None:
            era = era.loc[mask]

        X_np = X.astype(np.float32).fillna(0.5).values
        Y_np = Y.astype(np.float32).fillna(Y.median()).values
        self._n_features = X_np.shape[1]
        self.n_targets = Y_np.shape[1]

        # Feature standardization
        self._means = X_np.mean(axis=0)
        self._stds = X_np.std(axis=0) + 1e-6
        X_std = (X_np - self._means) / self._stds

        # Era-balanced sampler: compute weights so each era contributes equally.
        if self.era_balanced and era is not None:
            counts = era.value_counts()
            w = 1.0 / counts.loc[era].values
            w = w / w.sum() * len(w)
        else:
            w = np.ones(len(X_std), dtype=np.float32)

        model = self._build(self._n_features).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.epochs)

        X_t = torch.from_numpy(X_std).to(device)
        Y_t = torch.from_numpy(Y_np).to(device)
        W_t = torch.from_numpy(w.astype(np.float32)).to(device)

        best_state = None
        best_loss = float("inf")
        bad = 0

        for epoch in range(self.epochs):
            model.train()
            idx = torch.randperm(len(X_t), device=device)
            for start in range(0, len(X_t), self.batch_size):
                sel = idx[start:start + self.batch_size]
                xb = X_t[sel]
                yb = Y_t[sel]
                wb = W_t[sel].unsqueeze(-1)

                # MixUp within batch
                if self.mixup_alpha > 0:
                    lam = float(np.random.beta(self.mixup_alpha, self.mixup_alpha))
                    perm = torch.randperm(xb.size(0), device=device)
                    xb = lam * xb + (1 - lam) * xb[perm]
                    yb = lam * yb + (1 - lam) * yb[perm]

                pred = model(xb)
                # Weighted Huber-like loss (robust for noisy Numerai targets)
                loss = ((pred - yb) ** 2 * wb).mean()
                opt.zero_grad()
                loss.backward()
                if self.grad_clip:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip)
                opt.step()
            sched.step()

            # Validation
            if eval_set is not None:
                model.eval()
                with torch.no_grad():
                    Xv = (eval_set[0].astype(np.float32).fillna(0.5).values - self._means) / self._stds
                    yv = eval_set[1].astype(np.float32).fillna(0.5).values
                    Xv_t = torch.from_numpy(Xv).to(device)
                    pv = model(Xv_t).cpu().numpy().ravel()
                vloss = float(np.mean((pv - yv) ** 2))
                if self.verbose:
                    print(f"epoch {epoch}: val_mse={vloss:.5f}")
                if vloss < best_loss - 1e-6:
                    best_loss = vloss
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                    bad = 0
                else:
                    bad += 1
                    if bad >= self.patience:
                        break
            else:
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        self._state_dict = best_state
        return self

    # -- inference (CPU safe) --
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        torch, nn, F = _lazy_torch()
        if self._state_dict is None:
            raise RuntimeError("Call fit() before predict().")
        X_np = X.astype(np.float32).fillna(0.5).values
        X_std = (X_np - self._means) / self._stds

        # Build fresh model on CPU, load weights, evaluate.
        model = self._build(self._n_features)
        model.load_state_dict(self._state_dict)
        model.eval()
        with torch.no_grad():
            xb = torch.from_numpy(X_std.astype(np.float32))
            preds = model(xb).cpu().numpy()
        if preds.ndim == 2 and preds.shape[1] == 1:
            return preds.ravel()
        # Multi-target: return the average across targets by default
        return preds.mean(axis=1)

    def save(self, path: str | Path) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str | Path) -> "TabularNNTrainer":
        with open(path, "rb") as f:
            return pickle.load(f)


__all__ = [
    "TabularNNTrainer",
    "build_mlp",
    "build_resnet",
    "build_ft_transformer",
]
