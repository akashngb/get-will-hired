"""Run the ablation experiments described in Section 8.1.

Each experiment trains for a small number of epochs and reports test-set accuracy
per horizon. Results are appended to data/ablation_results.json and rendered as a
markdown table in data/ablation_results.md.

This script is deliberately small-scale by default so it can run on CPU in
minutes. Bump --epochs and --max-train for a longer, more discriminating run.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

import click
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.baselines import ImbalanceBaseline, LSTMBaseline, MidPriceBaseline  # noqa: E402
from src.models.tcn import TCNBlock, TCNModel  # noqa: E402
from src.models.train import (  # noqa: E402
    LOBDataset,
    Trainer,
    TrainerConfig,
    collate_with_horizons,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s :: %(message)s",
)
logger = logging.getLogger("ablation")


@dataclass
class AblationConfig:
    name: str
    description: str
    n_levels: int = 3
    n_channels: int = 32
    kernel_size: int = 3
    dropout: float = 0.1
    seq_len: int = 64
    use_lstm: bool = False
    feature_mask: list[str] | None = None
    uniform_dilation: bool = False


def build_model(cfg: AblationConfig, n_features: int, horizons: list[int]):
    if cfg.use_lstm:
        return LSTMBaseline(
            n_features=n_features, hidden_size=cfg.n_channels, horizons=horizons, dropout=cfg.dropout
        )
    model = TCNModel(
        n_features=n_features,
        n_levels=cfg.n_levels,
        n_channels=cfg.n_channels,
        kernel_size=cfg.kernel_size,
        dropout=cfg.dropout,
        horizons=horizons,
    )
    if cfg.uniform_dilation:
        # Rebuild blocks with dilation=1 to test the contribution of dilated convs
        new_blocks = torch.nn.ModuleList()
        in_ch = n_features
        for i in range(cfg.n_levels):
            new_blocks.append(
                TCNBlock(
                    in_channels=in_ch,
                    out_channels=cfg.n_channels,
                    kernel_size=cfg.kernel_size,
                    dilation=1,
                    dropout=cfg.dropout,
                )
            )
            in_ch = cfg.n_channels
        model.blocks = new_blocks
    return model


def filter_features(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    feature_cols: list[str],
    keep_predicate: Callable[[str], bool],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    mask = [keep_predicate(c) for c in feature_cols]
    idxs = [i for i, k in enumerate(mask) if k]
    cols = [feature_cols[i] for i in idxs]
    return X_train[:, idxs], X_val[:, idxs], X_test[:, idxs], cols


@torch.no_grad()
def evaluate(model, loader, device, horizons):
    model.eval()
    preds = {f"horizon_{h}": [] for h in horizons}
    truths = {f"horizon_{h}": [] for h in horizons}
    for x, labels in loader:
        x = x.to(device)
        out = model(x)
        for k, logits in out.items():
            preds[k].extend(logits.argmax(dim=1).cpu().tolist())
            truths[k].extend(labels[k].tolist())
    return {k: float(accuracy_score(truths[k], preds[k])) for k in preds}


def run_experiment(
    cfg: AblationConfig,
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    horizons: list[int],
    epochs: int,
    device: str,
    batch_size: int,
) -> dict:
    torch.manual_seed(0)
    n_features = X_train.shape[1]
    model = build_model(cfg, n_features, horizons)
    train_ds = LOBDataset(X_train, y_train, seq_len=cfg.seq_len, horizons=horizons)
    val_ds = LOBDataset(X_val, y_val, seq_len=cfg.seq_len, horizons=horizons)
    test_ds = LOBDataset(X_test, y_test, seq_len=cfg.seq_len, horizons=horizons)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_with_horizons, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_with_horizons
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_with_horizons
    )
    trainer_cfg = TrainerConfig(
        n_epochs=epochs,
        batch_size=batch_size,
        learning_rate=1e-3,
        weight_decay=1e-4,
        patience=epochs + 1,  # disable early stopping for ablation comparability
        seq_len=cfg.seq_len,
        horizons=tuple(horizons),
        scheduler="cosine",
        class_weights=None,
        use_wandb=False,
        use_mlflow=False,
    )
    trainer = Trainer(model, trainer_cfg, device=device)
    t0 = time.time()
    trainer.train(train_loader, val_loader, n_epochs=epochs, checkpoint_dir=Path(f"checkpoints/ablation_{cfg.name}"))
    elapsed = time.time() - t0
    test_acc = evaluate(trainer.model, test_loader, torch.device(device), horizons)
    n_params = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    return {
        "name": cfg.name,
        "description": cfg.description,
        "params": int(n_params),
        "train_time_s": float(elapsed),
        "test_accuracy": test_acc,
        "config": asdict(cfg),
    }


@click.command()
@click.option("--data-dir", default="data/splits", show_default=True)
@click.option("--epochs", default=2, show_default=True, type=int)
@click.option("--batch-size", default=256, show_default=True, type=int)
@click.option("--max-train", default=20000, show_default=True, type=int)
@click.option("--device", default="cpu", show_default=True)
@click.option("--out-json", default="data/ablation_results.json", show_default=True)
@click.option("--out-md", default="data/ablation_results.md", show_default=True)
def main(data_dir, epochs, batch_size, max_train, device, out_json, out_md):
    data_path = Path(data_dir)
    metadata = json.loads((data_path / "metadata.json").read_text())
    horizons = metadata["horizons"]
    feature_cols = metadata["feature_columns"]

    X_train = np.load(data_path / "X_train.npy")[:max_train]
    y_train = np.load(data_path / "y_train.npy")[:max_train]
    X_val = np.load(data_path / "X_val.npy")
    y_val = np.load(data_path / "y_val.npy")
    X_test = np.load(data_path / "X_test.npy")
    y_test = np.load(data_path / "y_test.npy")

    # ---- Define experiments ------------------------------------------------
    experiments: list[tuple[AblationConfig, dict | None]] = []
    experiments.append((AblationConfig(name="baseline", description="TCN n=3 c=32 k=3"), None))
    experiments.append(
        (AblationConfig(name="lstm", description="LSTM hidden=32 (LSTM vs TCN)", use_lstm=True), None)
    )
    experiments.append(
        (
            AblationConfig(
                name="uniform_dilation",
                description="TCN with dilation [1,1,1] (no dilation skip)",
                uniform_dilation=True,
            ),
            None,
        )
    )
    experiments.append(
        (AblationConfig(name="n_levels_2", description="TCN n_levels=2", n_levels=2), None)
    )
    experiments.append(
        (AblationConfig(name="n_levels_5", description="TCN n_levels=5", n_levels=5), None)
    )
    experiments.append(
        (AblationConfig(name="no_imbalance", description="Drop imbalance features"), {"drop": "imbalance"})
    )
    experiments.append(
        (AblationConfig(name="no_flow", description="Drop flow features"), {"drop": "flow"})
    )

    def keep(c: str, drop_group: str) -> bool:
        if drop_group == "imbalance":
            return "imbalance" not in c
        if drop_group == "flow":
            return not any(p in c for p in ("kyle", "trade_flow", "order_arrival", "cancellation"))
        return True

    results = []
    for cfg, modifier in experiments:
        logger.info("===== %s =====", cfg.name)
        if modifier:
            xt, xv, xs, cols = filter_features(
                X_train, X_val, X_test, feature_cols, lambda c: keep(c, modifier["drop"])
            )
        else:
            xt, xv, xs, cols = X_train, X_val, X_test, feature_cols
        try:
            r = run_experiment(
                cfg, xt, y_train, xv, y_val, xs, y_test, horizons, epochs, device, batch_size
            )
            r["n_features"] = len(cols)
        except Exception as exc:
            logger.exception("experiment %s failed: %s", cfg.name, exc)
            r = {"name": cfg.name, "error": str(exc)}
        results.append(r)

    # Add a non-learning baseline for context (majority class)
    baseline = MidPriceBaseline().fit(y_train[:, 0])
    base_preds = baseline.predict(np.zeros(len(y_test)))
    acc_base = {f"horizon_{h}": float((base_preds == y_test[: len(base_preds), i]).mean()) for i, h in enumerate(horizons)}
    results.append(
        {
            "name": "majority_class",
            "description": "Always predict the majority class",
            "params": 0,
            "train_time_s": 0.0,
            "test_accuracy": acc_base,
            "n_features": 0,
        }
    )

    Path(out_json).write_text(json.dumps(results, indent=2))
    render_markdown(results, horizons, Path(out_md))
    logger.info("Ablation results written to %s and %s", out_json, out_md)


def render_markdown(results: list[dict], horizons: list[int], out_path: Path) -> None:
    header = "| Experiment | " + " | ".join(f"H{h} Acc" for h in horizons) + " | Params | Train (s) |\n"
    sep = "|" + "|".join(["---"] * (len(horizons) + 3)) + "|\n"
    lines = ["# Ablation Results\n\n", header, sep]
    for r in results:
        if "error" in r:
            lines.append(f"| {r['name']} | (failed) | (failed) | (failed) | - | - |\n")
            continue
        accs = " | ".join(
            f"{r['test_accuracy'].get(f'horizon_{h}', float('nan')):.3f}" for h in horizons
        )
        lines.append(
            f"| {r['name']} | {accs} | {r.get('params', '-')} | {r.get('train_time_s', 0.0):.1f} |\n"
        )
    out_path.write_text("".join(lines))


if __name__ == "__main__":
    main()
