"""Run the model over the test set and backtest the resulting signals.

Usage:
    python scripts/run_backtest.py --checkpoint checkpoints/best_model.pt
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import click
import numpy as np
import torch
import torch.nn.functional as F
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.backtest.engine import BacktestEngine  # noqa: E402
from src.backtest.strategy import SignalStrategy  # noqa: E402
from src.models.tcn import TCNModel  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s :: %(message)s",
)
logger = logging.getLogger("backtest")


def model_from_config(config_path: Path, n_features: int, horizons: list[int]) -> TCNModel:
    cfg = yaml.safe_load(config_path.read_text())
    mcfg = cfg["model"]
    return TCNModel(
        n_features=n_features,
        n_classes=3,
        n_levels=mcfg.get("n_levels", 4),
        n_channels=mcfg.get("n_channels", 64),
        kernel_size=mcfg.get("kernel_size", 3),
        dropout=mcfg.get("dropout", 0.2),
        horizons=horizons,
    )


@click.command()
@click.option("--checkpoint", default="checkpoints/best_model.pt", show_default=True)
@click.option("--config", default="configs/tcn_small.yaml", show_default=True)
@click.option("--data-dir", default="data/splits", show_default=True)
@click.option("--horizon", default=10, show_default=True, type=int)
@click.option("--seq-len", default=64, show_default=True, type=int)
@click.option("--cost-bps", default=0.5, show_default=True, type=float)
@click.option("--confidence", default=0.5, show_default=True, type=float)
@click.option("--initial-capital", default=1_000_000.0, type=float)
@click.option("--out", default="data/backtest_results.json", show_default=True)
def main(
    checkpoint: str,
    config: str,
    data_dir: str,
    horizon: int,
    seq_len: int,
    cost_bps: float,
    confidence: float,
    initial_capital: float,
    out: str,
) -> None:
    data_path = Path(data_dir)
    metadata = json.loads((data_path / "metadata.json").read_text())
    horizons = metadata["horizons"]

    X_test = np.load(data_path / "X_test.npy")
    y_test = np.load(data_path / "y_test.npy")
    mid_test = np.load(data_path / "mid_price_test.npy")
    n_features = metadata["n_features"]

    logger.info("Loaded test set: X=%s mid=%s", X_test.shape, mid_test.shape)

    model = model_from_config(Path(config), n_features=n_features, horizons=horizons)
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    logger.info("Loaded checkpoint %s", checkpoint)

    # Build sliding windows
    n = len(X_test)
    n_windows = n - seq_len
    if n_windows <= 0:
        raise RuntimeError(f"X_test too short ({n}) for seq_len={seq_len}")
    windows = np.lib.stride_tricks.sliding_window_view(X_test, (seq_len, n_features))[: n_windows + 1, 0]
    logger.info("Built %d windows of shape %s", windows.shape[0], windows.shape[1:])

    head_name = f"horizon_{horizon}"
    h_idx = horizons.index(horizon)
    batch_size = 1024
    all_probs = []
    all_preds = []
    with torch.no_grad():
        for i in range(0, len(windows), batch_size):
            x = torch.from_numpy(windows[i : i + batch_size]).float()
            outputs = model(x)
            logits = outputs[head_name]
            probs = F.softmax(logits, dim=1).cpu().numpy()
            preds = logits.argmax(dim=1).cpu().numpy()
            all_probs.append(probs)
            all_preds.append(preds)
    probs = np.concatenate(all_probs)
    preds = np.concatenate(all_preds)
    logger.info("Generated predictions: %s", preds.shape)

    # The prediction at window i forecasts the direction at index (seq_len-1+i) -> (seq_len-1+i+horizon)
    # We align signals to the price series so prices[seq_len-1:] aligns with preds.
    strategy = SignalStrategy(horizon=horizon, cost_bps=cost_bps, confidence_threshold=confidence)
    signals = strategy.generate_signals(preds, probs)
    aligned_prices = mid_test[seq_len - 1 : seq_len - 1 + len(signals)]
    aligned_truth = y_test[seq_len - 1 : seq_len - 1 + len(signals), h_idx] - 1  # -1,0,1

    engine = BacktestEngine(
        strategy=strategy,
        initial_capital=initial_capital,
        spread=None,
    )
    ledger = engine.run(aligned_prices, signals)
    benchmark_returns = np.diff(aligned_prices) / aligned_prices[:-1]
    report = engine.summary(
        actual_directions=aligned_truth,
        benchmark_returns=benchmark_returns,
    )

    pred_dist = dict(zip(*np.unique(preds, return_counts=True)))
    sig_dist = dict(zip(*np.unique(signals, return_counts=True)))
    out_payload = {
        "horizon": horizon,
        "confidence_threshold": confidence,
        "cost_bps": cost_bps,
        "n_windows": int(len(windows)),
        "prediction_distribution": {int(k): int(v) for k, v in pred_dist.items()},
        "signal_distribution": {int(k): int(v) for k, v in sig_dist.items()},
        "report": {
            k: (float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v)
            for k, v in report.items()
        },
    }
    Path(out).write_text(json.dumps(out_payload, indent=2))
    logger.info("Backtest report:")
    for k, v in report.items():
        logger.info("  %s: %s", k, v)


if __name__ == "__main__":
    main()
