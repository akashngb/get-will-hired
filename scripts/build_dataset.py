"""Build the training/val/test arrays from raw LOB data.

Usage:
    python scripts/build_dataset.py --synthetic --n-events 200000
    python scripts/build_dataset.py --ticker AAPL --date 2012-06-21
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import click
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.feature_engineer import FeatureEngineer  # noqa: E402
from src.data.lobster_loader import LobsterLoader  # noqa: E402
from src.data.synthetic import SyntheticLOBGenerator  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s :: %(message)s",
)
logger = logging.getLogger("build_dataset")


@click.command()
@click.option("--ticker", default="AAPL", show_default=True)
@click.option("--date", default="2012-06-21", show_default=True)
@click.option("--levels", default=10, show_default=True, type=int)
@click.option("--n-events", default=200_000, show_default=True, type=int)
@click.option("--synthetic/--no-synthetic", default=True, show_default=True)
@click.option("--data-dir", default="data/raw", show_default=True)
@click.option("--out-dir", default="data/splits", show_default=True)
@click.option("--horizons", default="10,50,100", show_default=True)
@click.option("--seed", default=42, show_default=True, type=int)
def main(
    ticker: str,
    date: str,
    levels: int,
    n_events: int,
    synthetic: bool,
    data_dir: str,
    out_dir: str,
    horizons: str,
    seed: int,
) -> None:
    horizon_list = [int(h) for h in horizons.split(",")]
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    if synthetic:
        logger.info("Generating %d synthetic events (seed=%d)", n_events, seed)
        gen = SyntheticLOBGenerator(n_events=n_events, n_levels=levels, seed=seed)
        ob_df, msg_df = gen.generate()
    else:
        logger.info("Loading LOBSTER data: %s %s levels=%d", ticker, date, levels)
        loader = LobsterLoader(data_dir=data_dir, ticker=ticker, date=date, levels=levels)
        ob_df, msg_df = loader.load()
        if loader.is_synthetic:
            logger.warning("LOBSTER files missing — fell back to synthetic data")

    logger.info("Featurizing...")
    fe = FeatureEngineer(levels=levels, horizons=horizon_list)
    df = fe.fit_transform(ob_df, msg_df)
    X, y, feat_cols = FeatureEngineer.split_x_y(df, horizon_list)
    logger.info("Feature matrix: X=%s y=%s features=%d", X.shape, y.shape, len(feat_cols))

    # Temporal split — NEVER shuffle
    n = len(X)
    train_end = int(0.70 * n)
    val_end = int(0.85 * n)
    splits = {
        "train": (X[:train_end], y[:train_end]),
        "val": (X[train_end:val_end], y[train_end:val_end]),
        "test": (X[val_end:], y[val_end:]),
    }

    for name, (Xi, yi) in splits.items():
        np.save(out_path / f"X_{name}.npy", Xi)
        np.save(out_path / f"y_{name}.npy", yi)
        logger.info("Saved %s: X=%s y=%s", name, Xi.shape, yi.shape)

    # also save mid-price series for backtest
    mid_price = df["mid_price"].values.astype(np.float64)
    np.save(out_path / "mid_price.npy", mid_price)
    np.save(out_path / "mid_price_test.npy", mid_price[val_end:])

    # save feature stats and metadata
    fe.save_stats(out_path / "feature_stats.json")
    (out_path / "metadata.json").write_text(
        json.dumps(
            {
                "feature_columns": feat_cols,
                "horizons": horizon_list,
                "n_features": int(X.shape[1]),
                "levels": levels,
                "n_total": int(n),
                "train_end": int(train_end),
                "val_end": int(val_end),
                "is_synthetic": synthetic,
                "ticker": ticker,
                "date": date,
            },
            indent=2,
        )
    )
    logger.info("Done. Outputs in %s", out_path.resolve())
    logger.info(
        "Class balance (horizon_10): %s",
        dict(zip(*np.unique(y[:, 0], return_counts=True))),
    )


if __name__ == "__main__":
    main()
