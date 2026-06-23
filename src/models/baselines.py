"""Baseline models — see Section 4.3.

The ablation study must beat each of these convincingly. All four expose the
same predict/evaluate contract for fair comparison.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score


class MidPriceBaseline:
    """Always predicts the most common class in the training set."""

    def __init__(self) -> None:
        self.majority_class: int = 1  # default STATIONARY

    def fit(self, y: np.ndarray) -> "MidPriceBaseline":
        # y in {0,1,2}
        flat = y.ravel() if y.ndim > 1 else y
        self.majority_class = int(np.bincount(flat).argmax())
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        n = X.shape[0]
        return np.full(n, self.majority_class, dtype=np.int64)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict[str, float]:
        y_true = y[:, 0] if y.ndim > 1 else y
        y_pred = self.predict(X)
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        }


class SpreadMeanReversionBaseline:
    """If current mid > ewma(mid, 20): predict DOWN. Else UP."""

    def __init__(self, ewma_index: int = 0, mid_index: int = 0) -> None:
        # X is expected to contain a normalized ewma feature and a normalized mid_price
        # In normalized space we just compare two features. The actual column indices
        # must be passed in by the caller (see ablation notebook).
        self.ewma_index = ewma_index
        self.mid_index = mid_index

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "SpreadMeanReversionBaseline":
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        # X shape: (N, F) or (N, T, F) — use last timestep if 3D
        if X.ndim == 3:
            X_last = X[:, -1, :]
        else:
            X_last = X
        # signs interpreted in standardized feature space
        diff = X_last[:, self.mid_index] - X_last[:, self.ewma_index]
        # >0 means above ewma -> mean reversion says DOWN (class 0)
        # <0 means below ewma -> UP (class 2)
        return np.where(diff > 0, 0, 2).astype(np.int64)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict[str, float]:
        y_true = y[:, 0] if y.ndim > 1 else y
        y_pred = self.predict(X)
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        }


class ImbalanceBaseline:
    """Logistic regression on the L1 bid/ask imbalance feature only."""

    def __init__(self, imbalance_index: int = 0) -> None:
        self.imbalance_index = imbalance_index
        self.model = LogisticRegression(max_iter=2000)

    def _flatten_features(self, X: np.ndarray) -> np.ndarray:
        if X.ndim == 3:
            X = X[:, -1, :]
        return X[:, self.imbalance_index : self.imbalance_index + 1]

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ImbalanceBaseline":
        y_true = y[:, 0] if y.ndim > 1 else y
        self.model.fit(self._flatten_features(X), y_true)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(self._flatten_features(X)).astype(np.int64)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict[str, float]:
        y_true = y[:, 0] if y.ndim > 1 else y
        y_pred = self.predict(X)
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        }


class LSTMBaseline(nn.Module):
    """Single-layer LSTM with separate output head per horizon."""

    def __init__(
        self,
        n_features: int,
        n_classes: int = 3,
        hidden_size: int = 64,
        horizons: Iterable[int] = (10, 50, 100),
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.horizons = tuple(horizons)
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.heads = nn.ModuleDict()
        for h in self.horizons:
            self.heads[f"horizon_{h}"] = nn.Sequential(
                nn.Linear(hidden_size, 64),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(64, n_classes),
            )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        out, (h_n, _) = self.lstm(x)
        last = self.dropout(h_n[-1])
        return {name: head(last) for name, head in self.heads.items()}

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


__all__ = [
    "MidPriceBaseline",
    "SpreadMeanReversionBaseline",
    "ImbalanceBaseline",
    "LSTMBaseline",
]
