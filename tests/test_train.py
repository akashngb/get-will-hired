"""Tests for LOBDataset and Trainer smoke run."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.models.tcn import TCNModel
from src.models.train import (
    LOBDataset,
    Trainer,
    TrainerConfig,
    collate_with_horizons,
    compute_class_weights,
)


def test_lob_dataset_shapes():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((300, 8)).astype(np.float32)
    y = rng.integers(0, 3, size=(300, 3)).astype(np.int64)
    ds = LOBDataset(X, y, seq_len=20, horizons=(10, 50, 100))
    # n - seq_len - max_horizon + 1 = 300 - 20 - 100 + 1 = 181
    assert len(ds) == 181
    x_win, labels = ds[0]
    assert x_win.shape == (20, 8)
    assert set(labels.keys()) == {"horizon_10", "horizon_50", "horizon_100"}
    assert all(v.shape == () for v in labels.values())


def test_compute_class_weights_uniform():
    y = np.array([0, 0, 1, 1, 2, 2, 2])
    w = compute_class_weights(y)
    # most frequent class (2) should have the smallest weight
    assert w[2] < w[0] and w[2] < w[1]


def test_trainer_one_epoch_smoke():
    torch.manual_seed(0)
    rng = np.random.default_rng(1)
    X = rng.standard_normal((400, 6)).astype(np.float32)
    y = rng.integers(0, 3, size=(400, 3)).astype(np.int64)
    ds = LOBDataset(X, y, seq_len=16, horizons=(10, 50, 100))
    loader = DataLoader(ds, batch_size=8, collate_fn=collate_with_horizons, drop_last=True)

    model = TCNModel(n_features=6, n_levels=2, n_channels=8, dropout=0.0)
    cfg = TrainerConfig(
        n_epochs=1,
        batch_size=8,
        learning_rate=1e-3,
        weight_decay=0.0,
        patience=2,
        seq_len=16,
        horizons=(10, 50, 100),
        scheduler="cosine",
        class_weights=None,
        use_wandb=False,
        use_mlflow=False,
    )
    trainer = Trainer(model, cfg, device="cpu")
    train_metrics = trainer._train_epoch(loader, epoch=1)
    val_metrics = trainer._val_epoch(loader, epoch=1)
    assert "train_loss" in train_metrics and train_metrics["train_loss"] > 0
    assert "val_loss" in val_metrics and val_metrics["val_loss"] > 0
