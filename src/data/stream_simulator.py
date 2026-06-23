"""Kafka-style streaming event simulator for real-time inference demos."""

from __future__ import annotations

import logging
import time
from collections import deque
from typing import Generator

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class StreamSimulator:
    """Replay an event tape at a configurable speed multiplier.

    speed_multiplier > 1.0 means faster than real time. speed_multiplier == 0
    means yield without any sleep (used for benchmarking).
    """

    def __init__(self, data_source: pd.DataFrame, speed_multiplier: float = 100.0) -> None:
        if "time" not in data_source.columns:
            raise ValueError("data_source must contain a 'time' column")
        self.data_source = data_source.reset_index(drop=True)
        self.speed_multiplier = speed_multiplier
        self._cursor = 0

    def reset(self) -> None:
        self._cursor = 0

    def stream(self) -> Generator[dict, None, None]:
        timestamps = self.data_source["time"].values
        cols = [c for c in self.data_source.columns if c not in {"time"}]
        n = len(self.data_source)
        start_wall = time.perf_counter()
        start_sim = float(timestamps[self._cursor]) if n else 0.0

        for i in range(self._cursor, n):
            self._cursor = i
            sim_now = float(timestamps[i])
            row = self.data_source.iloc[i]
            event = {
                "timestamp": sim_now,
                "sequence_num": int(i),
                "orderbook": row[
                    [c for c in cols if c.startswith(("ask_", "bid_"))]
                ].to_numpy(dtype=np.float64),
                "message": {
                    "event_type": int(row["event_type"]) if "event_type" in row else 0,
                    "order_id": int(row["order_id"]) if "order_id" in row else 0,
                    "event_size": float(row["event_size"]) if "event_size" in row else 0.0,
                    "event_price": float(row["event_price"]) if "event_price" in row else 0.0,
                    "direction": int(row["direction"]) if "direction" in row else 0,
                },
            }
            if self.speed_multiplier > 0.0:
                target_wall = start_wall + (sim_now - start_sim) / self.speed_multiplier
                delay = target_wall - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
            yield event


class StreamBuffer:
    """Sliding-window buffer of fixed length, primed for model inference."""

    def __init__(self, maxlen: int = 200, n_features: int | None = None) -> None:
        if maxlen < 1:
            raise ValueError("maxlen must be >= 1")
        self.maxlen = maxlen
        self.n_features = n_features
        self._buf: deque[np.ndarray] = deque(maxlen=maxlen)

    def push(self, feature_vec: np.ndarray) -> None:
        arr = np.asarray(feature_vec, dtype=np.float32).ravel()
        if self.n_features is None:
            self.n_features = arr.shape[0]
        elif arr.shape[0] != self.n_features:
            raise ValueError(
                f"feature vector length {arr.shape[0]} != expected {self.n_features}"
            )
        self._buf.append(arr)

    def is_ready(self) -> bool:
        return len(self._buf) == self.maxlen

    def get_feature_window(self) -> np.ndarray | None:
        if not self.is_ready():
            return None
        return np.stack(list(self._buf), axis=0)

    def clear(self) -> None:
        self._buf.clear()

    def __len__(self) -> int:
        return len(self._buf)
