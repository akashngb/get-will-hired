"""Tests for baseline models."""

from __future__ import annotations

import numpy as np
import torch

from src.models.baselines import (
    ImbalanceBaseline,
    LSTMBaseline,
    MidPriceBaseline,
    SpreadMeanReversionBaseline,
)


def test_mid_price_baseline_picks_majority():
    y = np.array([[2], [2], [2], [1], [0]])
    X = np.zeros((5, 4), dtype=np.float32)
    b = MidPriceBaseline().fit(y)
    assert b.majority_class == 2
    preds = b.predict(X)
    assert (preds == 2).all()


def test_spread_mean_reversion():
    X = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    b = SpreadMeanReversionBaseline(ewma_index=1, mid_index=0)
    # row 0: mid > ewma -> DOWN (0); row 1: mid < ewma -> UP (2)
    np.testing.assert_array_equal(b.predict(X), np.array([0, 2], dtype=np.int64))


def test_imbalance_baseline_runs():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((200, 10)).astype(np.float32)
    # crude relationship: high imbalance -> class 2
    y = np.where(X[:, 0] > 0, 2, 0).reshape(-1, 1)
    b = ImbalanceBaseline(imbalance_index=0).fit(X, y)
    metrics = b.evaluate(X, y)
    assert metrics["accuracy"] > 0.8


def test_lstm_baseline_forward():
    m = LSTMBaseline(n_features=8, horizons=(10, 50, 100))
    x = torch.randn(4, 24, 8)
    out = m(x)
    assert set(out.keys()) == {"horizon_10", "horizon_50", "horizon_100"}
    for v in out.values():
        assert v.shape == (4, 3)
