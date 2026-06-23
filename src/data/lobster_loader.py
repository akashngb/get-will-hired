"""LOBSTER data loader — reads raw CSV pairs and returns unified DataFrames.

See LOB_TCN_DESIGN.md Section 3.3 for the column contract.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class DataValidationError(ValueError):
    """Raised when the LOBSTER files fail integrity checks."""


PRICE_SCALE = 10_000.0  # LOBSTER raw integer prices are USD * 10,000


class LobsterLoader:
    """Read LOBSTER-formatted CSVs into the project's canonical schema.

    Falls back to SyntheticLOBGenerator when the raw files are not available.
    Sets ``self.is_synthetic = True`` in that case so downstream code can react.
    """

    def __init__(
        self,
        data_dir: str | Path,
        ticker: str,
        date: str,
        levels: int = 10,
        start_time: int = 34_200_000,
        end_time: int = 57_600_000,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.ticker = ticker.upper()
        self.date = date
        self.levels = levels
        self.start_time = start_time
        self.end_time = end_time
        self.is_synthetic = False

    # ------------------------------------------------------------------ API

    def load(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Return (orderbook_df, message_df) — falls back to synthetic if not found."""
        ob_path, msg_path = self._resolve_paths()
        if ob_path is None or msg_path is None:
            logger.warning(
                "LOBSTER files for %s on %s not found in %s — using synthetic fallback. "
                "See https://lobsterdata.com/info/DataAccess.php to download real data.",
                self.ticker,
                self.date,
                self.data_dir,
            )
            return self._synthetic_fallback()

        logger.info("Loading LOBSTER files: %s and %s", ob_path, msg_path)
        ob_df = self._read_orderbook(ob_path)
        msg_df = self._read_messages(msg_path)
        # align lengths (LOBSTER rows are 1:1 by event index)
        n = min(len(ob_df), len(msg_df))
        ob_df = ob_df.iloc[:n].reset_index(drop=True)
        msg_df = msg_df.iloc[:n].reset_index(drop=True)
        ob_df.insert(0, "time", msg_df["time"].values)
        self._validate(ob_df, msg_df)
        return ob_df, msg_df

    def to_snapshots(self) -> pd.DataFrame:
        """Convenience: merged event-aligned snapshot DataFrame."""
        ob, msg = self.load()
        merged = ob.copy()
        for col in ("event_type", "order_id", "event_size", "event_price", "direction"):
            if col in msg.columns:
                merged[col] = msg[col].values
        return merged

    # ------------------------------------------------------------------ Internals

    def _resolve_paths(self) -> tuple[Path | None, Path | None]:
        if not self.data_dir.exists():
            return None, None
        ob_pattern = f"{self.ticker}_{self.date}_*_orderbook_{self.levels}.csv"
        msg_pattern = f"{self.ticker}_{self.date}_*_message_{self.levels}.csv"
        ob_matches = sorted(self.data_dir.glob(ob_pattern))
        msg_matches = sorted(self.data_dir.glob(msg_pattern))
        if not ob_matches or not msg_matches:
            return None, None
        return ob_matches[0], msg_matches[0]

    def _read_orderbook(self, path: Path) -> pd.DataFrame:
        columns: list[str] = []
        for i in range(1, self.levels + 1):
            columns.extend(
                [f"ask_price_{i}", f"ask_size_{i}", f"bid_price_{i}", f"bid_size_{i}"]
            )
        df = pd.read_csv(path, header=None, names=columns)
        df = self._convert_prices(df)
        return df

    def _read_messages(self, path: Path) -> pd.DataFrame:
        df = pd.read_csv(
            path,
            header=None,
            names=["time", "event_type", "order_id", "event_size", "event_price", "direction"],
        )
        df["event_price"] = df["event_price"] / PRICE_SCALE
        return df

    def _convert_prices(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in df.columns:
            if col.endswith("_price_") or "_price_" in col:
                df[col] = df[col] / PRICE_SCALE
        return df

    def _validate(self, ob_df: pd.DataFrame, msg_df: pd.DataFrame) -> None:
        if len(ob_df) != len(msg_df):
            raise DataValidationError(
                f"orderbook ({len(ob_df)}) and messages ({len(msg_df)}) length mismatch"
            )
        if ob_df.isna().any().any():
            raise DataValidationError("orderbook contains NaNs in critical columns")
        # ask_price_1 > bid_price_1
        if not (ob_df["ask_price_1"] > ob_df["bid_price_1"]).all():
            raise DataValidationError("crossed market detected: ask_price_1 <= bid_price_1")
        # monotonicity per level
        for i in range(1, self.levels):
            if not (ob_df[f"ask_price_{i + 1}"] >= ob_df[f"ask_price_{i}"]).all():
                raise DataValidationError(f"ask prices not monotone at level {i}")
            if not (ob_df[f"bid_price_{i + 1}"] <= ob_df[f"bid_price_{i}"]).all():
                raise DataValidationError(f"bid prices not monotone at level {i}")

    def _synthetic_fallback(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        from src.data.synthetic import SyntheticLOBGenerator  # local import to avoid cycle

        self.is_synthetic = True
        gen = SyntheticLOBGenerator(
            n_events=200_000,
            n_levels=self.levels,
            seed=abs(hash((self.ticker, self.date))) % (2**31),
        )
        return gen.generate()


__all__ = ["LobsterLoader", "DataValidationError"]
