"""Synthetic LOB data generator using Poisson process simulation.

Market microstructure model:
- Mid-price follows arithmetic Brownian motion: dS = sigma * dW
- Bid-ask spread follows mean-reverting Ornstein-Uhlenbeck process
- Order arrivals: independent Poisson(lambda_buy) and Poisson(lambda_sell) streams
- Order sizes: log-normal distribution
- 10 price levels per side, quantities decay geometrically with depth

Output format matches LobsterLoader.to_snapshots():
- orderbook columns: bid_price_i, bid_size_i, ask_price_i, ask_size_i (i = 1..levels)
- message columns: time, event_type, order_id, event_size, event_price, direction
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


MARKET_OPEN_SECONDS = 34_200.0  # 9:30 a.m. NY time


class SyntheticLOBGenerator:
    """Generate synthetic Level-2 order book snapshots and matching message tape."""

    def __init__(
        self,
        n_events: int = 500_000,
        n_levels: int = 10,
        sigma: float = 0.0001,
        lambda_arrival: float = 10.0,
        spread_mean: float = 0.01,
        spread_kappa: float = 5.0,
        spread_sigma: float = 0.002,
        tick_size: float = 0.01,
        initial_price: float = 100.0,
        seed: int = 42,
    ) -> None:
        if n_events < 100:
            raise ValueError("n_events must be >= 100 for a useful dataset")
        if n_levels < 1:
            raise ValueError("n_levels must be >= 1")
        self.n_events = n_events
        self.n_levels = n_levels
        self.sigma = sigma
        self.lambda_arrival = lambda_arrival
        self.spread_mean = spread_mean
        self.spread_kappa = spread_kappa
        self.spread_sigma = spread_sigma
        self.tick_size = tick_size
        self.initial_price = initial_price
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def generate(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Return (orderbook_df, message_df) in the same format as LobsterLoader."""
        logger.info(
            "Generating synthetic LOB: n_events=%d n_levels=%d seed=%d",
            self.n_events,
            self.n_levels,
            self.seed,
        )
        mid = self._simulate_midprice()
        spread = self._simulate_spread()
        ob_df = self._build_book_levels(mid, spread)
        msg_df = self._build_messages(mid)
        ob_df.insert(0, "time", msg_df["time"].values)
        logger.info("Synthetic LOB generated. ob_shape=%s msg_shape=%s", ob_df.shape, msg_df.shape)
        return ob_df, msg_df

    def _simulate_midprice(self) -> np.ndarray:
        """Arithmetic Brownian motion in price space, snapped to the tick grid."""
        dW = self.rng.normal(0.0, 1.0, size=self.n_events) * self.sigma * self.initial_price
        mid = np.cumsum(dW) + self.initial_price
        # snap to half-tick grid so that bid/ask sit on tick boundaries
        mid = np.round(mid / (self.tick_size / 2.0)) * (self.tick_size / 2.0)
        # guard against negative or zero prices (extremely unlikely but defensive)
        mid = np.clip(mid, self.initial_price * 0.5, self.initial_price * 2.0)
        return mid

    def _simulate_spread(self) -> np.ndarray:
        """OU mean-reverting positive spread process."""
        spread = np.empty(self.n_events)
        spread[0] = self.spread_mean
        dt = 1.0
        noise = self.rng.normal(0.0, 1.0, size=self.n_events)
        for t in range(1, self.n_events):
            ds = (
                self.spread_kappa * (self.spread_mean - spread[t - 1]) * dt
                + self.spread_sigma * np.sqrt(dt) * noise[t]
            )
            spread[t] = max(spread[t - 1] + ds, self.tick_size)
        # quantize to tick grid
        spread = np.maximum(np.round(spread / self.tick_size) * self.tick_size, self.tick_size)
        return spread

    def _build_book_levels(self, mid: np.ndarray, spread: np.ndarray) -> pd.DataFrame:
        """Construct full Level-2 book by stacking geometrically decaying sizes."""
        n = len(mid)
        half = spread / 2.0
        best_bid = mid - half
        best_ask = mid + half

        # size profile: level i has expected size lambda_arrival * decay^(i-1)
        decay = 0.85
        base_log_mean = np.log(self.lambda_arrival * 5.0)
        base_log_sigma = 0.4

        cols: dict[str, np.ndarray] = {}
        # cumulative level offsets are sums of geometric integer gaps (>=1 tick each)
        bid_offset = np.zeros(n)
        ask_offset = np.zeros(n)
        for i in range(1, self.n_levels + 1):
            if i > 1:
                gap_bid = (
                    self.rng.geometric(p=0.6, size=n).astype(float) * self.tick_size
                )
                gap_ask = (
                    self.rng.geometric(p=0.6, size=n).astype(float) * self.tick_size
                )
                bid_offset = bid_offset + gap_bid
                ask_offset = ask_offset + gap_ask
            bid_price = best_bid - bid_offset
            ask_price = best_ask + ask_offset
            log_mean = base_log_mean + np.log(decay) * (i - 1)
            bid_size = np.maximum(
                np.round(self.rng.lognormal(log_mean, base_log_sigma, size=n)),
                1.0,
            )
            ask_size = np.maximum(
                np.round(self.rng.lognormal(log_mean, base_log_sigma, size=n)),
                1.0,
            )
            cols[f"bid_price_{i}"] = bid_price
            cols[f"bid_size_{i}"] = bid_size
            cols[f"ask_price_{i}"] = ask_price
            cols[f"ask_size_{i}"] = ask_size

        df = pd.DataFrame(cols)
        return df

    def _build_messages(self, mid: np.ndarray) -> pd.DataFrame:
        """Generate the message tape — arrivals, types, sides, sizes, timestamps."""
        n = self.n_events
        # interarrival times from exponential(1/lambda)
        interarrivals = self.rng.exponential(1.0 / self.lambda_arrival, size=n)
        timestamps = MARKET_OPEN_SECONDS + np.cumsum(interarrivals)

        # event_type: 1=new limit, 2=partial cancel, 3=delete, 4=visible exec
        # distribution roughly matches empirical LOBSTER frequencies
        event_type = self.rng.choice([1, 2, 3, 4], size=n, p=[0.55, 0.15, 0.20, 0.10])
        direction = self.rng.choice([1, -1], size=n)  # +1=buy, -1=sell
        order_id = np.arange(1, n + 1, dtype=np.int64)
        event_size = np.maximum(
            np.round(self.rng.lognormal(np.log(self.lambda_arrival), 0.5, size=n)),
            1.0,
        )
        event_price = mid + direction * (self.tick_size * self.rng.integers(0, 5, size=n))

        msg_df = pd.DataFrame(
            {
                "time": timestamps,
                "event_type": event_type.astype(np.int64),
                "order_id": order_id,
                "event_size": event_size,
                "event_price": event_price,
                "direction": direction.astype(np.int64),
            }
        )
        return msg_df
