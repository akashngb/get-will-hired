"""Tests for LobsterLoader and the synthetic fallback path."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.lobster_loader import DataValidationError, LobsterLoader, PRICE_SCALE


def test_fallback_to_synthetic(tmp_path: Path):
    loader = LobsterLoader(data_dir=tmp_path, ticker="AAPL", date="2012-06-21", levels=10)
    ob, msg = loader.load()
    assert loader.is_synthetic is True
    assert len(ob) > 0 and len(msg) > 0
    assert "ask_price_1" in ob.columns and "bid_price_1" in ob.columns


def test_real_data_path_roundtrip(tmp_path: Path):
    """Write a tiny synthetic LOBSTER-format CSV pair and re-read it."""
    levels = 2
    n = 100
    # Build raw integer-style CSV in LOBSTER schema
    ob_cols = []
    for i in range(1, levels + 1):
        ob_cols.extend([f"ask_p{i}", f"ask_s{i}", f"bid_p{i}", f"bid_s{i}"])
    raw = pd.DataFrame(
        {
            "ask_p1": np.full(n, 1000100),
            "ask_s1": np.full(n, 10),
            "bid_p1": np.full(n, 999900),
            "bid_s1": np.full(n, 12),
            "ask_p2": np.full(n, 1000200),
            "ask_s2": np.full(n, 8),
            "bid_p2": np.full(n, 999800),
            "bid_s2": np.full(n, 11),
        }
    )
    ticker = "TEST"
    date = "2020-01-01"
    ob_path = tmp_path / f"{ticker}_{date}_34200000_57600000_orderbook_{levels}.csv"
    msg_path = tmp_path / f"{ticker}_{date}_34200000_57600000_message_{levels}.csv"
    raw[ob_cols].to_csv(ob_path, header=False, index=False)
    msg = pd.DataFrame(
        {
            "time": np.linspace(34200.0, 34250.0, n),
            "event_type": np.ones(n, dtype=int),
            "order_id": np.arange(n),
            "event_size": np.full(n, 5),
            "event_price": np.full(n, 1000000),
            "direction": np.ones(n, dtype=int),
        }
    )
    msg.to_csv(msg_path, header=False, index=False)

    loader = LobsterLoader(data_dir=tmp_path, ticker=ticker, date=date, levels=levels)
    ob, msg = loader.load()
    assert loader.is_synthetic is False
    assert len(ob) == n
    # Prices converted from integer LOBSTER format
    assert abs(ob["ask_price_1"].iloc[0] - 100.01) < 1e-9
    assert abs(ob["bid_price_1"].iloc[0] - 99.99) < 1e-9


def test_validation_catches_crossed_book(tmp_path: Path):
    levels = 1
    n = 50
    ticker = "BAD"
    date = "2020-01-02"
    ob_path = tmp_path / f"{ticker}_{date}_34200000_57600000_orderbook_{levels}.csv"
    msg_path = tmp_path / f"{ticker}_{date}_34200000_57600000_message_{levels}.csv"
    # ask < bid -> crossed
    pd.DataFrame(
        {
            "a_p": np.full(n, 999000),
            "a_s": np.full(n, 5),
            "b_p": np.full(n, 1001000),
            "b_s": np.full(n, 5),
        }
    ).to_csv(ob_path, header=False, index=False)
    pd.DataFrame(
        {
            "time": np.linspace(34200.0, 34250.0, n),
            "event_type": np.ones(n, dtype=int),
            "order_id": np.arange(n),
            "event_size": np.full(n, 1),
            "event_price": np.full(n, 1000000),
            "direction": np.ones(n, dtype=int),
        }
    ).to_csv(msg_path, header=False, index=False)

    loader = LobsterLoader(data_dir=tmp_path, ticker=ticker, date=date, levels=levels)
    with pytest.raises(DataValidationError):
        loader.load()


def test_to_snapshots(tmp_path: Path):
    loader = LobsterLoader(data_dir=tmp_path, ticker="MSFT", date="2012-06-21", levels=10)
    snaps = loader.to_snapshots()
    assert "ask_price_1" in snaps.columns
    assert "event_type" in snaps.columns
    assert "direction" in snaps.columns
