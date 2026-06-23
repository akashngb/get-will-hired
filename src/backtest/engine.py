"""Event-driven backtest engine."""

from __future__ import annotations

import logging
from typing import Iterable

import numpy as np
import pandas as pd

from src.backtest.metrics import full_report
from src.backtest.strategy import SignalStrategy

logger = logging.getLogger(__name__)


class BacktestEngine:
    """Walks an aligned price + signal series and records the ledger.

    Signals at time t are decided from predictions made at t and executed at t+1
    to eliminate look-ahead. Position is fixed-size ±1 unit; held for horizon
    events then forced flat.
    """

    def __init__(
        self,
        strategy: SignalStrategy,
        initial_capital: float = 1_000_000.0,
        spread: float | np.ndarray | None = None,
    ) -> None:
        self.strategy = strategy
        self.initial_capital = initial_capital
        self.spread = spread
        self.ledger: pd.DataFrame | None = None
        self._summary: dict | None = None

    def run(self, prices: np.ndarray, signals: np.ndarray) -> pd.DataFrame:
        prices = np.asarray(prices, dtype=np.float64)
        signals = np.asarray(signals, dtype=np.int64)
        n = min(len(prices), len(signals))
        prices, signals = prices[:n], signals[:n]

        cost_frac = self.strategy.cost_bps / 10_000.0
        horizon = self.strategy.horizon

        # Determine spread for slippage
        if self.spread is None:
            half_spread = np.zeros(n)
        elif np.isscalar(self.spread):
            half_spread = np.full(n, float(self.spread) / 2.0)
        else:
            sp = np.asarray(self.spread, dtype=np.float64)[:n]
            half_spread = sp / 2.0

        position = 0
        entry_price = 0.0
        time_in_position = 0
        events = []
        cumulative_pnl = 0.0
        capital = self.initial_capital

        for t in range(n - 1):
            sig = int(signals[t])
            execute_price = prices[t + 1]
            action = "HOLD"
            pnl = 0.0

            # Forced exit at end of horizon
            if position != 0 and time_in_position >= horizon:
                exit_price = execute_price - position * half_spread[t + 1]
                pnl_pct = position * (exit_price - entry_price) / entry_price
                pnl = pnl_pct - cost_frac  # exit cost
                cumulative_pnl += pnl
                capital *= 1.0 + pnl
                events.append(
                    {
                        "event_idx": t + 1,
                        "action": f"EXIT_{position:+d}",
                        "price": exit_price,
                        "position": 0,
                        "pnl": pnl,
                        "cumulative_pnl": cumulative_pnl,
                        "capital": capital,
                    }
                )
                position = 0
                entry_price = 0.0
                time_in_position = 0
                action = "FLAT"

            # Open a new position only when flat and signal is decisive
            if position == 0 and sig != 0:
                entry_price = execute_price + sig * half_spread[t + 1]
                position = sig
                time_in_position = 0
                pnl = -cost_frac  # entry cost
                cumulative_pnl += pnl
                capital *= 1.0 + pnl
                events.append(
                    {
                        "event_idx": t + 1,
                        "action": f"ENTRY_{position:+d}",
                        "price": entry_price,
                        "position": position,
                        "pnl": pnl,
                        "cumulative_pnl": cumulative_pnl,
                        "capital": capital,
                    }
                )
                continue  # next event

            if position != 0:
                time_in_position += 1

            if action == "HOLD" and position != 0:
                # mark-to-market step PnL
                mtm = position * (execute_price - prices[t]) / prices[t]
                cumulative_pnl += mtm
                capital *= 1.0 + mtm
                events.append(
                    {
                        "event_idx": t + 1,
                        "action": "MTM",
                        "price": execute_price,
                        "position": position,
                        "pnl": mtm,
                        "cumulative_pnl": cumulative_pnl,
                        "capital": capital,
                    }
                )

        self.ledger = pd.DataFrame(events)
        logger.info(
            "Backtest done. n_events=%d trades=%d final_capital=%.2f",
            n,
            self.ledger["action"].str.startswith("ENTRY").sum() if not self.ledger.empty else 0,
            capital,
        )
        return self.ledger

    def summary(
        self,
        actual_directions: np.ndarray | None = None,
        benchmark_returns: np.ndarray | None = None,
        periods_per_year: int = 252 * int(6.5 * 3600),
    ) -> dict:
        if self.ledger is None:
            raise RuntimeError("run() must be called before summary()")
        per_event_pnl = self.ledger["pnl"].to_numpy() if not self.ledger.empty else np.array([])
        positions = (
            self.ledger["position"].to_numpy()
            if not self.ledger.empty
            else np.array([], dtype=np.int64)
        )
        # Hit rate is reported against the user's per-window truth — keep separate
        # signal series for alignment.
        if actual_directions is None:
            hit_signals = positions
            hit_truth = np.zeros_like(positions)
        else:
            # Use entry events' positions as directional signals, aligned to truth
            entry_mask = (
                self.ledger["action"].str.startswith("ENTRY").to_numpy()
                if not self.ledger.empty
                else np.array([], dtype=bool)
            )
            entry_idx = (
                self.ledger.loc[entry_mask, "event_idx"].to_numpy()
                if not self.ledger.empty
                else np.array([], dtype=np.int64)
            )
            entry_idx = entry_idx[entry_idx < len(actual_directions)]
            hit_signals = (
                self.ledger.loc[entry_mask, "position"].to_numpy()[: len(entry_idx)]
                if entry_idx.size
                else np.array([], dtype=np.int64)
            )
            hit_truth = actual_directions[entry_idx] if entry_idx.size else np.array([])
        if benchmark_returns is None:
            benchmark_returns = np.zeros_like(per_event_pnl)
        report = full_report(
            per_event_pnl, hit_signals, hit_truth, benchmark_returns, periods_per_year
        )
        report["final_capital"] = (
            float(self.ledger["capital"].iloc[-1]) if not self.ledger.empty else self.initial_capital
        )
        report["initial_capital"] = self.initial_capital
        self._summary = report
        return report
