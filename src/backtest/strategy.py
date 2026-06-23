"""Signal generation: convert model predictions into +1/0/-1 trade signals.

See LOB_TCN_DESIGN.md Section 5.1.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Position:
    direction: int  # +1 long, -1 short, 0 flat
    entry_price: float
    entry_time: int
    horizon: int


class SignalStrategy:
    """Confidence-gated, single-unit directional strategy with forced exit at +k."""

    def __init__(
        self,
        horizon: int = 10,
        cost_bps: float = 0.5,
        confidence_threshold: float = 0.55,
    ) -> None:
        if horizon < 1:
            raise ValueError("horizon must be >= 1")
        self.horizon = horizon
        self.cost_bps = cost_bps
        self.confidence_threshold = confidence_threshold

    def generate_signals(
        self,
        predictions: np.ndarray,
        probabilities: np.ndarray,
    ) -> np.ndarray:
        """Map class predictions {0,1,2} -> {-1, 0, +1} signals.

        Class encoding: 0 = DOWN, 1 = STATIONARY, 2 = UP.
        Only emits a signal if max probability exceeds confidence_threshold.
        """
        if predictions.shape[0] != probabilities.shape[0]:
            raise ValueError("predictions and probabilities length mismatch")
        max_p = probabilities.max(axis=1)
        confident = max_p >= self.confidence_threshold
        signal = np.zeros_like(predictions, dtype=np.int64)
        signal[(predictions == 2) & confident] = 1
        signal[(predictions == 0) & confident] = -1
        return signal

    def apply_costs(self, returns: np.ndarray, signals: np.ndarray) -> np.ndarray:
        """Subtract per-trade transaction cost when the signal flips position."""
        cost_frac = self.cost_bps / 10_000.0
        # detect entries (where position changes from 0 -> +/-1) and exits
        position_changes = np.diff(np.concatenate([[0], signals])) != 0
        costs = np.where(position_changes, cost_frac, 0.0)
        return returns - costs
