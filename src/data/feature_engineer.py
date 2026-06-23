"""Feature engineering from raw LOB snapshots — see Section 3.5.

All features are causal: the value at time t depends only on rows <= t.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


PRICE_RETURN_LAGS: tuple[int, ...] = (1, 5, 10, 20, 50)
ROLLING_WINDOWS: tuple[int, ...] = (20, 50, 100)
EWMA_ALPHAS: tuple[float, ...] = (0.1, 0.3, 0.5)
FLOW_WINDOWS: tuple[int, ...] = (10, 50, 100)


class FeatureEngineer:
    """Compute causal LOB features and price-direction labels."""

    def __init__(
        self,
        levels: int = 10,
        horizons: Iterable[int] = (10, 50, 100),
        stationary_threshold_bps: float = 0.05,
    ) -> None:
        self.levels = levels
        self.horizons = tuple(horizons)
        self.stationary_threshold = stationary_threshold_bps / 10_000.0
        self.feature_stats: dict[str, dict[str, float]] = {}
        self._feature_names: list[str] = []
        self._fitted = False

    # ------------------------------------------------------------------ API

    def fit_transform(self, ob_df: pd.DataFrame, msg_df: pd.DataFrame) -> pd.DataFrame:
        """Compute features + labels, fit normalization stats, return normalized frame."""
        feats = self._compute_all_features(ob_df, msg_df)
        labels = self._compute_labels(feats)
        out = pd.concat([feats, labels], axis=1)
        out = out.dropna().reset_index(drop=True)

        self._feature_names = [c for c in feats.columns if c != "timestamp"]
        self.feature_stats = {}
        for col in self._feature_names:
            mean = float(out[col].mean())
            std = float(out[col].std(ddof=0))
            self.feature_stats[col] = {"mean": mean, "std": std if std > 1e-9 else 1.0}
        self._fitted = True

        normalized = out.copy()
        for col in self._feature_names:
            stats = self.feature_stats[col]
            normalized[col] = (out[col] - stats["mean"]) / stats["std"]
        return normalized

    def transform(self, ob_df: pd.DataFrame, msg_df: pd.DataFrame) -> pd.DataFrame:
        """Compute features + labels using already-fit stats."""
        if not self._fitted:
            raise RuntimeError("FeatureEngineer.transform() called before fit_transform()")
        feats = self._compute_all_features(ob_df, msg_df)
        labels = self._compute_labels(feats)
        out = pd.concat([feats, labels], axis=1)
        out = out.dropna().reset_index(drop=True)
        for col in self._feature_names:
            stats = self.feature_stats.get(col, {"mean": 0.0, "std": 1.0})
            out[col] = (out[col] - stats["mean"]) / stats["std"]
        return out

    def get_feature_names(self) -> list[str]:
        return list(self._feature_names)

    def save_stats(self, path: str | Path) -> None:
        """Persist feature stats and metadata for inference-time normalization."""
        payload = {
            "feature_names": self._feature_names,
            "feature_stats": self.feature_stats,
            "horizons": list(self.horizons),
            "levels": self.levels,
            "stationary_threshold": self.stationary_threshold,
        }
        Path(path).write_text(json.dumps(payload, indent=2))

    def load_stats(self, path: str | Path) -> None:
        payload = json.loads(Path(path).read_text())
        self._feature_names = payload["feature_names"]
        self.feature_stats = payload["feature_stats"]
        self.horizons = tuple(payload["horizons"])
        self.levels = payload["levels"]
        self.stationary_threshold = payload["stationary_threshold"]
        self._fitted = True

    # -------------------------------------------------------------- Computers

    def _compute_all_features(
        self, ob_df: pd.DataFrame, msg_df: pd.DataFrame
    ) -> pd.DataFrame:
        # Align indexes
        ob = ob_df.reset_index(drop=True).copy()
        msg = msg_df.reset_index(drop=True).copy()
        if "time" not in ob.columns and "time" in msg.columns:
            ob["time"] = msg["time"].values

        price = self._compute_price_features(ob)
        imbalance = self._compute_imbalance_features(ob)
        pressure = self._compute_pressure_features(ob)
        flow = self._compute_flow_features(price, msg)
        rolling = self._compute_rolling_features(price)

        feats = pd.concat(
            [
                price[["timestamp", "mid_price", "spread", "spread_bps"]],
                price.drop(columns=["timestamp", "mid_price", "spread", "spread_bps"]),
                imbalance,
                pressure,
                flow,
                rolling,
            ],
            axis=1,
        )
        # remove duplicate columns if any
        feats = feats.loc[:, ~feats.columns.duplicated()]
        return feats

    def _compute_price_features(self, ob: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame()
        out["timestamp"] = ob["time"].values if "time" in ob.columns else np.arange(len(ob))
        out["mid_price"] = (ob["ask_price_1"].values + ob["bid_price_1"].values) / 2.0
        out["spread"] = ob["ask_price_1"].values - ob["bid_price_1"].values
        out["spread_bps"] = out["spread"] / out["mid_price"] * 10_000.0
        for lag in PRICE_RETURN_LAGS:
            out[f"log_return_{lag}"] = np.log(out["mid_price"] / out["mid_price"].shift(lag))
        return out

    def _compute_imbalance_features(self, ob: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame()
        for i in range(1, min(self.levels, 5) + 1):
            bid = ob[f"bid_size_{i}"].values
            ask = ob[f"ask_size_{i}"].values
            denom = bid + ask
            denom = np.where(denom == 0, 1.0, denom)
            out[f"bid_ask_imbalance_l{i}"] = (bid - ask) / denom

        # depth-weighted (1/i)
        weighted = np.zeros(len(ob))
        for i in range(1, self.levels + 1):
            w = 1.0 / i
            weighted += w * (ob[f"bid_size_{i}"].values - ob[f"ask_size_{i}"].values)
        out["volume_imbalance_weighted"] = weighted

        out["total_bid_volume"] = sum(
            ob[f"bid_size_{i}"].values for i in range(1, self.levels + 1)
        )
        out["total_ask_volume"] = sum(
            ob[f"ask_size_{i}"].values for i in range(1, self.levels + 1)
        )
        return out

    def _compute_pressure_features(self, ob: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame()
        out["bid_price_range"] = (
            ob["bid_price_1"].values - ob[f"bid_price_{self.levels}"].values
        )
        out["ask_price_range"] = (
            ob[f"ask_price_{self.levels}"].values - ob["ask_price_1"].values
        )
        bid_pressure = np.zeros(len(ob))
        ask_pressure = np.zeros(len(ob))
        for i in range(1, self.levels + 1):
            bid_pressure += ob[f"bid_size_{i}"].values / np.maximum(
                ob[f"bid_price_{i}"].values, 1e-6
            )
            ask_pressure += ob[f"ask_size_{i}"].values / np.maximum(
                ob[f"ask_price_{i}"].values, 1e-6
            )
        out["price_pressure_bid"] = bid_pressure
        out["price_pressure_ask"] = ask_pressure
        return out

    def _compute_flow_features(
        self, price: pd.DataFrame, msg: pd.DataFrame
    ) -> pd.DataFrame:
        out = pd.DataFrame(index=price.index)
        # Identify trade events (type 4 visible, 5 hidden) and signed flow
        is_trade = msg["event_type"].isin([4, 5]).values if "event_type" in msg.columns else np.zeros(len(msg), dtype=bool)
        direction = msg["direction"].values if "direction" in msg.columns else np.zeros(len(msg))
        size = msg["event_size"].values if "event_size" in msg.columns else np.zeros(len(msg))
        signed_flow = np.where(is_trade, direction * size, 0.0)
        abs_flow = np.abs(signed_flow)

        signed_series = pd.Series(signed_flow, index=price.index)
        abs_series = pd.Series(abs_flow, index=price.index)

        for w in FLOW_WINDOWS:
            num = signed_series.rolling(w, min_periods=w).sum()
            den = abs_series.rolling(w, min_periods=w).sum().replace(0, np.nan)
            out[f"trade_flow_imbalance_{w}"] = (num / den).fillna(0.0)

        # Order arrival rate per second
        if "time" in msg.columns:
            dt = msg["time"].diff().fillna(0.0).values
            with np.errstate(divide="ignore", invalid="ignore"):
                inv_dt = np.where(dt > 0, 1.0 / dt, 0.0)
            out["order_arrival_rate"] = (
                pd.Series(inv_dt, index=price.index).rolling(50, min_periods=50).mean()
            )
        else:
            out["order_arrival_rate"] = 0.0

        # cancellation rate (types 2, 3)
        is_cancel = (
            msg["event_type"].isin([2, 3]).values
            if "event_type" in msg.columns
            else np.zeros(len(msg))
        )
        cancel_series = pd.Series(is_cancel.astype(float), index=price.index)
        out["cancellation_rate"] = cancel_series.rolling(50, min_periods=50).mean()

        # Kyle's lambda: rolling OLS slope of |delta_price| on volume
        delta_p = price["mid_price"].diff().abs().fillna(0.0)
        vol = pd.Series(abs_flow, index=price.index)
        w = 100
        # use rolling cov / var
        cov = delta_p.rolling(w, min_periods=w).cov(vol)
        var = vol.rolling(w, min_periods=w).var()
        kyle = (cov / var.replace(0, np.nan)).fillna(0.0)
        out["kyle_lambda"] = kyle

        return out

    def _compute_rolling_features(self, price: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=price.index)
        log_ret = price["log_return_1"]
        for w in ROLLING_WINDOWS:
            out[f"rolling_volatility_{w}"] = log_ret.rolling(w, min_periods=w).std()
        out["rolling_autocorr_1"] = log_ret.rolling(50, min_periods=50).apply(
            lambda x: pd.Series(x).autocorr(lag=1), raw=False
        )
        for alpha in EWMA_ALPHAS:
            out[f"ewma_mid_{alpha}"] = price["mid_price"].ewm(alpha=alpha, adjust=False).mean()
        return out

    def _compute_labels(self, price: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=price.index)
        mid = price["mid_price"]
        for k in self.horizons:
            future_mid = mid.shift(-k)
            out[f"future_mid_price_{k}"] = future_mid
            change = (future_mid - mid) / mid
            direction = np.where(
                change > self.stationary_threshold,
                1,
                np.where(change < -self.stationary_threshold, -1, 0),
            )
            out[f"label_direction_{k}"] = direction.astype(np.int64)
            # smoothed: average of next k future mid prices
            smooth_mean = (
                mid.rolling(k, min_periods=k).mean().shift(-k)
            )
            change_smooth = (smooth_mean - mid) / mid
            out[f"label_smooth_{k}"] = np.where(
                change_smooth > self.stationary_threshold,
                1,
                np.where(change_smooth < -self.stationary_threshold, -1, 0),
            ).astype(np.int64)
        return out

    # ---------------------------------------------------------- Utilities

    @staticmethod
    def split_x_y(
        df: pd.DataFrame, horizons: Iterable[int]
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """Split a featurized dataframe into feature matrix X and label matrix y.

        Labels are mapped from {-1, 0, 1} to {0, 1, 2} for classification.
        """
        horizons = list(horizons)
        label_cols = [f"label_direction_{k}" for k in horizons]
        meta_cols = (
            ["timestamp"]
            + [f"future_mid_price_{k}" for k in horizons]
            + [f"label_smooth_{k}" for k in horizons]
        )
        drop_cols = label_cols + meta_cols
        feature_cols = [c for c in df.columns if c not in drop_cols]
        X = df[feature_cols].values.astype(np.float32)
        y_signed = df[label_cols].values.astype(np.int64)
        y = y_signed + 1  # remap -1,0,1 -> 0,1,2
        return X, y, feature_cols
