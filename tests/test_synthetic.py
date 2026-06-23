"""Tests for SyntheticLOBGenerator — see Section 3.4."""

from __future__ import annotations

import numpy as np
import pytest

from src.data.synthetic import MARKET_OPEN_SECONDS, SyntheticLOBGenerator


@pytest.fixture(scope="module")
def small_dataset():
    gen = SyntheticLOBGenerator(n_events=5_000, n_levels=10, seed=123)
    return gen.generate()


def test_shape(small_dataset):
    ob_df, msg_df = small_dataset
    assert len(ob_df) == 5_000
    assert len(msg_df) == 5_000
    # 4 columns per level + time = 41 for 10 levels
    assert ob_df.shape[1] == 1 + 4 * 10
    assert set(["time", "event_type", "order_id", "event_size", "event_price", "direction"]) <= set(
        msg_df.columns
    )


def test_spread_positive(small_dataset):
    ob_df, _ = small_dataset
    assert (ob_df["ask_price_1"] > ob_df["bid_price_1"]).all()


def test_price_monotonicity(small_dataset):
    ob_df, _ = small_dataset
    for i in range(1, 10):
        assert (ob_df[f"ask_price_{i + 1}"] >= ob_df[f"ask_price_{i}"]).all()
        assert (ob_df[f"bid_price_{i + 1}"] <= ob_df[f"bid_price_{i}"]).all()


def test_no_negative_values(small_dataset):
    ob_df, msg_df = small_dataset
    for col in ob_df.columns:
        if col == "time":
            continue
        assert (ob_df[col] > 0).all(), f"{col} contains non-positive values"
    assert (msg_df["event_size"] > 0).all()
    assert (msg_df["event_price"] > 0).all()


def test_reproducibility():
    a = SyntheticLOBGenerator(n_events=1_000, seed=7).generate()
    b = SyntheticLOBGenerator(n_events=1_000, seed=7).generate()
    np.testing.assert_array_equal(a[0].values, b[0].values)
    np.testing.assert_array_equal(a[1].values, b[1].values)


def test_timestamps_monotonic(small_dataset):
    _, msg_df = small_dataset
    assert (np.diff(msg_df["time"].values) >= 0).all()
    assert msg_df["time"].iloc[0] >= MARKET_OPEN_SECONDS


def test_midprice_statistics():
    gen = SyntheticLOBGenerator(n_events=20_000, sigma=0.0002, seed=11)
    ob, _ = gen.generate()
    mid = (ob["ask_price_1"].values + ob["bid_price_1"].values) / 2.0
    log_ret = np.diff(np.log(mid))
    # cumulative log return drift should be near zero (no drift in our ABM)
    assert abs(np.mean(log_ret)) < 5e-4
    # realized vol should be within 50% of sigma (loose bound; quantization noise)
    realized = np.std(log_ret)
    assert realized < 0.0006
