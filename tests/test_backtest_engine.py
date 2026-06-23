"""Tests for backtest strategy, engine, and metrics."""

from __future__ import annotations

import numpy as np
import pytest

from src.backtest.engine import BacktestEngine
from src.backtest.metrics import (
    full_report,
    hit_rate,
    max_drawdown,
    profit_factor,
    sharpe_ratio,
    sortino_ratio,
)
from src.backtest.strategy import SignalStrategy


def test_signal_strategy_confidence_gating():
    preds = np.array([0, 1, 2, 2, 0])
    probs = np.array(
        [
            [0.8, 0.1, 0.1],
            [0.4, 0.4, 0.2],
            [0.7, 0.2, 0.1],  # max=0.7 but class is 2 -> not consistent in our encoding
            [0.2, 0.1, 0.7],
            [0.5, 0.2, 0.3],
        ]
    )
    strat = SignalStrategy(horizon=5, cost_bps=0.5, confidence_threshold=0.6)
    sig = strat.generate_signals(preds, probs)
    # row 0: pred=DOWN with confidence 0.8 -> -1
    # row 1: pred=STATIONARY -> 0 anyway
    # row 2: pred=UP but probs max corresponds to class 0 (0.7) — we still emit because
    #        we threshold on max prob alone; the encoding is: emit UP if pred==2 and max>=t.
    # row 3: pred=UP with confidence 0.7 -> +1
    # row 4: pred=DOWN but max=0.5 < threshold -> 0
    assert sig.tolist() == [-1, 0, 1, 1, 0]


def test_sharpe_zero_std():
    assert sharpe_ratio(np.zeros(10)) == 0.0


def test_max_drawdown_simple():
    cum = np.array([1.0, 1.2, 1.5, 0.9, 1.1])
    # peak 1.5, trough 0.9 -> dd = 0.4
    mdd = max_drawdown(cum)
    assert abs(mdd - 0.4) < 1e-9


def test_hit_rate_basic():
    signals = np.array([1, 0, -1, 1])
    actual = np.array([1, -1, -1, -1])
    # nonzero signals: idx 0 (sig=1, act=1, correct), idx 2 (sig=-1 act=-1, correct),
    # idx 3 (sig=1 act=-1, wrong)
    assert hit_rate(signals, actual) == pytest.approx(2 / 3)


def test_profit_factor_inf():
    assert profit_factor(5.0, 0.0) == float("inf")


def test_backtest_engine_long_trade():
    # Price rises from 100 to 110 over 10 events. Signal +1 at t=0. Costs 0.5bps.
    prices = np.linspace(100.0, 110.0, 11)
    signals = np.zeros(11, dtype=np.int64)
    signals[0] = 1
    strat = SignalStrategy(horizon=10, cost_bps=0.5, confidence_threshold=0.0)
    eng = BacktestEngine(strat, initial_capital=1000.0)
    ledger = eng.run(prices, signals)
    report = eng.summary()
    assert ledger["action"].iloc[0].startswith("ENTRY")
    # Final capital should be higher than initial (price went up 10% with tiny costs)
    assert report["final_capital"] > 1000.0
    assert report["n_trades"] >= 1


def test_full_report_keys():
    r = np.array([0.001, -0.002, 0.003])
    s = np.array([1, -1, 1])
    a = np.array([1, -1, 1])
    bench = np.array([0.0005, -0.001, 0.001])
    rep = full_report(r, s, a, bench)
    for key in [
        "sharpe",
        "sortino",
        "max_drawdown",
        "calmar",
        "hit_rate",
        "profit_factor",
        "alpha",
        "beta",
        "total_return",
        "n_trades",
    ]:
        assert key in rep
