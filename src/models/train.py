"""Training loop for the TCN (and any other multi-horizon model with the same contract).

See LOB_TCN_DESIGN.md Section 4.4.
"""

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


class LOBDataset(Dataset):
    """Sliding-window dataset of LOB feature sequences with per-horizon labels."""

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        seq_len: int = 100,
        horizons: Iterable[int] = (10, 50, 100),
    ) -> None:
        if X.ndim != 2:
            raise ValueError(f"X must be 2D (n_events, n_features); got {X.shape}")
        if y.ndim != 2 or y.shape[1] != len(tuple(horizons)):
            raise ValueError(
                f"y must be 2D with one column per horizon; got {y.shape} for horizons {horizons}"
            )
        self.X = X.astype(np.float32)
        self.y = y.astype(np.int64)
        self.seq_len = seq_len
        self.horizons = tuple(horizons)
        self.max_horizon = max(self.horizons)

    def __len__(self) -> int:
        return max(0, len(self.X) - self.seq_len - self.max_horizon + 1)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        end = idx + self.seq_len
        x_window = self.X[idx:end]  # (seq_len, n_features)
        # label index aligns with the last input timestep; future targets already
        # encoded into y rows at that index in build_dataset.py
        label_row = self.y[end - 1]
        labels = {
            f"horizon_{h}": torch.tensor(label_row[i], dtype=torch.long)
            for i, h in enumerate(self.horizons)
        }
        return torch.from_numpy(x_window), labels


def collate_with_horizons(batch):
    xs = torch.stack([b[0] for b in batch], dim=0)
    keys = batch[0][1].keys()
    labels = {k: torch.stack([b[1][k] for b in batch], dim=0) for k in keys}
    return xs, labels


@dataclass
class TrainerConfig:
    n_epochs: int = 20
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 5
    seq_len: int = 100
    horizons: tuple[int, ...] = (10, 50, 100)
    scheduler: str = "cosine"
    grad_clip: float = 1.0
    log_every: int = 100
    use_wandb: bool = False
    wandb_project: str = "lob-tcn"
    wandb_entity: str | None = None
    use_mlflow: bool = False
    mlflow_uri: str = "./mlruns"
    n_classes: int = 3
    class_weights: list[float] | None = None
    metadata: dict = field(default_factory=dict)


class Trainer:
    """Generic multi-horizon classification trainer."""

    def __init__(
        self,
        model: nn.Module,
        config: TrainerConfig,
        device: str | torch.device = "cpu",
    ) -> None:
        self.model = model.to(device)
        self.config = config
        self.device = torch.device(device)
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.scheduler: torch.optim.lr_scheduler._LRScheduler | None = None  # type: ignore
        self.best_val_loss = math.inf
        self.epochs_since_improvement = 0
        self.history: list[dict] = []
        self._setup_logging()

    def _setup_logging(self) -> None:
        self.wandb_run = None
        if self.config.use_wandb:
            try:
                import wandb  # type: ignore

                self.wandb_run = wandb.init(
                    project=self.config.wandb_project,
                    entity=self.config.wandb_entity,
                    reinit=True,
                )
            except Exception as exc:  # pragma: no cover - optional
                logger.warning("W&B init failed (%s); continuing without it", exc)
                self.wandb_run = None

        self.mlflow = None
        if self.config.use_mlflow:
            try:
                import mlflow  # type: ignore

                mlflow.set_tracking_uri(self.config.mlflow_uri)
                self.mlflow = mlflow
                self.mlflow.start_run()
            except Exception as exc:  # pragma: no cover - optional
                logger.warning("MLflow init failed (%s); continuing without it", exc)
                self.mlflow = None

    # ------------------------------------------------------------------ Train

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        n_epochs: int | None = None,
        checkpoint_dir: str | Path = "./checkpoints",
    ) -> dict:
        n_epochs = n_epochs or self.config.n_epochs
        if self.config.scheduler == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=n_epochs
            )

        ckpt_dir = Path(checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        best_ckpt = ckpt_dir / "best_model.pt"

        for epoch in range(1, n_epochs + 1):
            t0 = time.time()
            train_metrics = self._train_epoch(train_loader, epoch)
            val_metrics = self._val_epoch(val_loader, epoch)
            elapsed = time.time() - t0

            if self.scheduler is not None:
                self.scheduler.step()

            entry = {"epoch": epoch, **train_metrics, **val_metrics, "elapsed_s": elapsed}
            self.history.append(entry)
            self._log_metrics(entry, step=epoch)
            logger.info(
                "epoch=%d train_loss=%.4f val_loss=%.4f val_acc_h10=%.3f elapsed=%.1fs",
                epoch,
                train_metrics["train_loss"],
                val_metrics["val_loss"],
                val_metrics.get("val_acc_horizon_10", float("nan")),
                elapsed,
            )

            if val_metrics["val_loss"] < self.best_val_loss - 1e-4:
                self.best_val_loss = val_metrics["val_loss"]
                self.epochs_since_improvement = 0
                self.save_checkpoint(best_ckpt, is_best=True)
            else:
                self.epochs_since_improvement += 1
                if self.early_stopping(val_metrics["val_loss"]):
                    logger.info("Early stopping at epoch %d", epoch)
                    break

        self.save_checkpoint(ckpt_dir / "last_model.pt", is_best=False)
        (ckpt_dir / "training_history.json").write_text(json.dumps(self.history, indent=2))
        if self.wandb_run is not None:
            self.wandb_run.finish()
        if self.mlflow is not None:
            self.mlflow.end_run()
        return {"history": self.history, "best_val_loss": self.best_val_loss}

    def _train_epoch(self, loader: DataLoader, epoch: int) -> dict[str, float]:
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        all_preds: dict[str, list[int]] = {f"horizon_{h}": [] for h in self.config.horizons}
        all_labels: dict[str, list[int]] = {f"horizon_{h}": [] for h in self.config.horizons}

        for batch_idx, (x, labels) in enumerate(loader):
            x = x.to(self.device, non_blocking=True)
            labels = {k: v.to(self.device, non_blocking=True) for k, v in labels.items()}
            self.optimizer.zero_grad()
            outputs = self.model(x)
            loss = self._compute_loss(outputs, labels)
            loss.backward()
            if self.config.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1
            for k, v in outputs.items():
                all_preds[k].extend(v.argmax(dim=1).detach().cpu().tolist())
                all_labels[k].extend(labels[k].detach().cpu().tolist())

        out = {"train_loss": total_loss / max(1, n_batches)}
        for k in all_preds:
            out[f"train_acc_{k}"] = float(accuracy_score(all_labels[k], all_preds[k]))
        return out

    @torch.no_grad()
    def _val_epoch(self, loader: DataLoader, epoch: int) -> dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        n_batches = 0
        all_preds: dict[str, list[int]] = {f"horizon_{h}": [] for h in self.config.horizons}
        all_labels: dict[str, list[int]] = {f"horizon_{h}": [] for h in self.config.horizons}

        for x, labels in loader:
            x = x.to(self.device, non_blocking=True)
            labels = {k: v.to(self.device, non_blocking=True) for k, v in labels.items()}
            outputs = self.model(x)
            loss = self._compute_loss(outputs, labels)
            total_loss += loss.item()
            n_batches += 1
            for k, v in outputs.items():
                all_preds[k].extend(v.argmax(dim=1).cpu().tolist())
                all_labels[k].extend(labels[k].cpu().tolist())

        out = {"val_loss": total_loss / max(1, n_batches)}
        for k in all_preds:
            out[f"val_acc_{k}"] = float(accuracy_score(all_labels[k], all_preds[k]))
            out[f"val_f1_macro_{k}"] = float(
                f1_score(all_labels[k], all_preds[k], average="macro", zero_division=0)
            )
            out[f"val_f1_weighted_{k}"] = float(
                f1_score(all_labels[k], all_preds[k], average="weighted", zero_division=0)
            )
        return out

    def _compute_loss(
        self, outputs: dict[str, torch.Tensor], labels: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        weight = None
        if self.config.class_weights is not None:
            weight = torch.tensor(self.config.class_weights, device=self.device, dtype=torch.float32)
        loss = torch.zeros((), device=self.device)
        for name, logits in outputs.items():
            target = labels[name]
            loss = loss + F.cross_entropy(logits, target, weight=weight)
        return loss

    def _log_metrics(self, metrics: dict, step: int) -> None:
        if self.wandb_run is not None:
            try:
                self.wandb_run.log(metrics, step=step)
            except Exception:  # pragma: no cover - defensive
                pass
        if self.mlflow is not None:
            try:
                for k, v in metrics.items():
                    if isinstance(v, (int, float)):
                        self.mlflow.log_metric(k, float(v), step=step)
            except Exception:  # pragma: no cover - defensive
                pass

    def save_checkpoint(self, path: str | Path, is_best: bool = False) -> None:
        ckpt = {
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "history": self.history,
            "best_val_loss": self.best_val_loss,
            "config": self.config.__dict__,
        }
        torch.save(ckpt, path)
        logger.info("Saved checkpoint: %s (best=%s)", path, is_best)

    def load_checkpoint(self, path: str | Path) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state"])
        if "optimizer_state" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer_state"])

    def early_stopping(self, val_loss: float) -> bool:
        return self.epochs_since_improvement >= self.config.patience


def compute_class_weights(y: np.ndarray, n_classes: int = 3) -> list[float]:
    """w_c = N / (n_classes * count_c). Used for imbalance compensation."""
    flat = y.ravel() if y.ndim > 1 else y
    counts = np.bincount(flat, minlength=n_classes).astype(np.float64)
    total = counts.sum()
    weights = total / (n_classes * np.maximum(counts, 1.0))
    return weights.tolist()
