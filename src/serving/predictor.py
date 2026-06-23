"""Model wrapper for inference — see Section 6.2.

Mirrors the FeatureEngineer.transform() pipeline so train/serve are aligned.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml

from src.data.feature_engineer import FeatureEngineer
from src.models.tcn import TCNModel

logger = logging.getLogger(__name__)

DIRECTION_MAP = {0: "DOWN", 1: "STATIONARY", 2: "UP"}


class Predictor:
    """End-to-end inference: raw events -> features -> tensor -> logits -> decoded."""

    MODEL_VERSION = "v0.1.0"

    def __init__(
        self,
        checkpoint_path: str | Path,
        feature_stats_path: str | Path,
        config_path: str | Path | None = None,
        device: str | torch.device = "cpu",
        seq_len: int = 100,
        n_classes: int = 3,
        horizons: tuple[int, ...] | None = None,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self.feature_stats_path = Path(feature_stats_path)
        self.config_path = Path(config_path) if config_path else None
        self.device = torch.device(device)
        self.seq_len = seq_len
        self.n_classes = n_classes
        self.horizons = horizons
        self.model: TCNModel | None = None
        self.feature_engineer: FeatureEngineer | None = None
        self.checkpoint_loaded = False

    def load_model(self) -> None:
        """Load checkpoint and feature stats; fall back to random init if missing."""
        # Load feature stats — required
        if not self.feature_stats_path.exists():
            raise FileNotFoundError(
                f"feature_stats missing at {self.feature_stats_path}. "
                "Run scripts/build_dataset.py first."
            )
        stats_payload = json.loads(self.feature_stats_path.read_text())
        self.horizons = tuple(stats_payload.get("horizons", self.horizons or (10, 50, 100)))
        n_features = len(stats_payload["feature_names"])

        # Model architecture
        if self.config_path and self.config_path.exists():
            cfg = yaml.safe_load(self.config_path.read_text())
            mcfg = cfg["model"]
            self.seq_len = cfg.get("training", {}).get("seq_len", self.seq_len)
        else:
            mcfg = {"n_levels": 3, "n_channels": 32, "kernel_size": 3, "dropout": 0.1}

        self.model = TCNModel(
            n_features=n_features,
            n_classes=self.n_classes,
            n_levels=mcfg.get("n_levels", 4),
            n_channels=mcfg.get("n_channels", 64),
            kernel_size=mcfg.get("kernel_size", 3),
            dropout=mcfg.get("dropout", 0.2),
            horizons=self.horizons,
        )

        if self.checkpoint_path.exists():
            ckpt = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(ckpt["model_state"])
            self.checkpoint_loaded = True
            logger.info("Loaded checkpoint from %s", self.checkpoint_path)
        else:
            logger.warning(
                "Checkpoint missing at %s — using random weights. Predictions are meaningless.",
                self.checkpoint_path,
            )

        self.model = self.model.to(self.device).eval()
        # torch.compile if available
        try:
            self.model = torch.compile(self.model, mode="reduce-overhead")
            logger.info("torch.compile enabled")
        except Exception as exc:  # pragma: no cover
            logger.warning("torch.compile unavailable (%s); using eager mode", exc)

        # Feature engineer with loaded stats
        self.feature_engineer = FeatureEngineer(
            levels=stats_payload.get("levels", 10), horizons=self.horizons
        )
        self.feature_engineer.load_stats(self.feature_stats_path)

    def warmup(self, n_warmup: int = 5) -> None:
        if self.model is None or self.feature_engineer is None:
            raise RuntimeError("call load_model() first")
        n_features = len(self.feature_engineer.get_feature_names())
        x = torch.zeros(1, self.seq_len, n_features, device=self.device)
        with torch.no_grad():
            for _ in range(n_warmup):
                _ = self.model(x)

    # -------------------------------------------------------- Core prediction

    def predict_from_window(self, window: np.ndarray) -> dict[str, Any]:
        """Run inference on a pre-featurized, pre-normalized window of shape (seq_len, F)."""
        if self.model is None:
            raise RuntimeError("call load_model() first")
        if window.ndim == 2:
            window = window[None, ...]
        x = torch.from_numpy(np.ascontiguousarray(window, dtype=np.float32)).to(self.device)
        t0 = time.perf_counter()
        with torch.no_grad():
            outputs = self.model(x)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return self._decode(outputs, elapsed_ms)

    def predict(self, raw_snapshot_sequence: list[dict]) -> dict[str, Any]:
        """Featurize a raw sequence of events then run inference.

        ``raw_snapshot_sequence`` is a list of dicts containing both the
        orderbook columns (bid_price_i, bid_size_i, ...) and any available
        message columns (event_type, direction, ...). Length must be >= seq_len.
        """
        if self.feature_engineer is None:
            raise RuntimeError("call load_model() first")
        if len(raw_snapshot_sequence) < self.seq_len + 100:
            # we need warmup rows for rolling features
            raise ValueError(
                f"need >= {self.seq_len + 100} events; got {len(raw_snapshot_sequence)}"
            )
        df = pd.DataFrame(raw_snapshot_sequence)
        ob_cols = [
            c
            for c in df.columns
            if c.startswith(("bid_price_", "bid_size_", "ask_price_", "ask_size_"))
        ]
        msg_cols = [
            c
            for c in df.columns
            if c in {"time", "event_type", "order_id", "event_size", "event_price", "direction"}
        ]
        if "time" not in df.columns:
            df["time"] = np.arange(len(df), dtype=np.float64)
            msg_cols.append("time")
        ob_df = df[["time"] + ob_cols].copy()
        msg_df = df[msg_cols].copy()
        feats = self.feature_engineer.transform(ob_df, msg_df)
        feature_names = self.feature_engineer.get_feature_names()
        window = feats.iloc[-self.seq_len :][feature_names].to_numpy()
        return self.predict_from_window(window)

    def predict_batch(self, windows: list[np.ndarray]) -> list[dict[str, Any]]:
        return [self.predict_from_window(w) for w in windows]

    # ---------------------------------------------------------------- Helpers

    def _decode(self, outputs: dict[str, torch.Tensor], elapsed_ms: float) -> dict[str, Any]:
        predictions: dict[str, dict[str, Any]] = {}
        for name, logits in outputs.items():
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]
            pred_idx = int(probs.argmax())
            predictions[name] = {
                "direction": DIRECTION_MAP[pred_idx],
                "probability": float(probs[pred_idx]),
                "logits": logits.detach().cpu().numpy()[0].tolist(),
                "probabilities": probs.tolist(),
            }
        return {
            "predictions": predictions,
            "inference_time_ms": float(elapsed_ms),
            "model_version": self.MODEL_VERSION,
            "checkpoint_loaded": self.checkpoint_loaded,
        }
