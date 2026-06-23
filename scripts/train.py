"""CLI training entrypoint.

Usage:
    python scripts/train.py --config configs/base.yaml
    python scripts/train.py --config configs/tcn_small.yaml --epochs 3
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import click
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.baselines import LSTMBaseline  # noqa: E402
from src.models.tcn import TCNModel  # noqa: E402
from src.models.train import (  # noqa: E402
    LOBDataset,
    Trainer,
    TrainerConfig,
    collate_with_horizons,
    compute_class_weights,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s :: %(message)s",
)
logger = logging.getLogger("train")


def load_config(path: str | Path) -> dict:
    return yaml.safe_load(Path(path).read_text())


def build_model(model_name: str, n_features: int, mcfg: dict, horizons: list[int]):
    if model_name == "tcn":
        return TCNModel(
            n_features=n_features,
            n_classes=3,
            n_levels=mcfg.get("n_levels", 4),
            n_channels=mcfg.get("n_channels", 64),
            kernel_size=mcfg.get("kernel_size", 3),
            dropout=mcfg.get("dropout", 0.2),
            horizons=horizons,
        )
    if model_name == "lstm":
        return LSTMBaseline(
            n_features=n_features,
            hidden_size=mcfg.get("hidden_size", 64),
            horizons=horizons,
            dropout=mcfg.get("dropout", 0.2),
        )
    raise ValueError(f"unknown model {model_name}")


@click.command()
@click.option("--config", default="configs/base.yaml", show_default=True)
@click.option("--data-dir", default="data/splits", show_default=True)
@click.option("--checkpoint-dir", default="./checkpoints", show_default=True)
@click.option("--epochs", default=None, type=int, help="Override config n_epochs")
@click.option("--device", default="auto", show_default=True)
@click.option("--limit-train", default=None, type=int, help="cap training samples for quick smoke")
def main(
    config: str,
    data_dir: str,
    checkpoint_dir: str,
    epochs: int | None,
    device: str,
    limit_train: int | None,
) -> None:
    cfg = load_config(config)
    horizons = cfg["data"]["horizons"]
    seq_len = cfg["training"]["seq_len"]
    model_name = cfg.get("model_name", "tcn")

    data_path = Path(data_dir)
    X_train = np.load(data_path / "X_train.npy")
    y_train = np.load(data_path / "y_train.npy")
    X_val = np.load(data_path / "X_val.npy")
    y_val = np.load(data_path / "y_val.npy")
    metadata = json.loads((data_path / "metadata.json").read_text())

    if limit_train is not None:
        X_train = X_train[:limit_train]
        y_train = y_train[:limit_train]

    logger.info(
        "Loaded data: X_train=%s X_val=%s features=%d",
        X_train.shape,
        X_val.shape,
        metadata["n_features"],
    )

    train_ds = LOBDataset(X_train, y_train, seq_len=seq_len, horizons=horizons)
    val_ds = LOBDataset(X_val, y_val, seq_len=seq_len, horizons=horizons)
    logger.info("Datasets: train_windows=%d val_windows=%d", len(train_ds), len(val_ds))

    batch_size = cfg["training"]["batch_size"]
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_with_horizons,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_with_horizons,
        drop_last=False,
    )

    model = build_model(model_name, metadata["n_features"], cfg["model"], horizons)
    logger.info(
        "Built %s: params=%d receptive_field=%s",
        model_name,
        sum(p.numel() for p in model.parameters() if p.requires_grad),
        getattr(model, "receptive_field", lambda: "n/a")()
        if hasattr(model, "receptive_field")
        else "n/a",
    )

    class_weights = compute_class_weights(y_train[:, 0]) if cfg["training"].get(
        "class_weighted", True
    ) else None

    trainer_cfg = TrainerConfig(
        n_epochs=epochs or cfg["training"]["n_epochs"],
        batch_size=batch_size,
        learning_rate=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
        patience=cfg["training"]["patience"],
        seq_len=seq_len,
        horizons=tuple(horizons),
        scheduler=cfg["training"].get("scheduler", "cosine"),
        class_weights=class_weights,
        use_wandb=False,
        use_mlflow=False,
        metadata=metadata,
    )

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    trainer = Trainer(model, trainer_cfg, device=device)

    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    summary = trainer.train(train_loader, val_loader, checkpoint_dir=checkpoint_dir)
    logger.info("Best val_loss=%.4f", summary["best_val_loss"])


if __name__ == "__main__":
    main()
