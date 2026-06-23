"""Streamlit monitoring dashboard — see Section 7.2.

Run with:
    streamlit run src/monitoring/dashboard.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from src.monitoring.drift import DriftMonitor, PredictionDriftMonitor


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SPLITS_DIR = PROJECT_ROOT / "data" / "splits"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"


def _load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception as exc:  # pragma: no cover
        st.warning(f"Could not parse {path}: {exc}")
        return None


def _load_npy(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    try:
        return np.load(path)
    except Exception:  # pragma: no cover
        return None


def page_live_predictions() -> None:
    st.header("Live Predictions")
    backtest = _load_json(PROJECT_ROOT / "data" / "backtest_results.json")
    if backtest is None:
        st.info("No backtest_results.json yet — run scripts/run_backtest.py to populate.")
        return
    pred_dist = backtest.get("prediction_distribution", {})
    sig_dist = backtest.get("signal_distribution", {})
    st.write("Prediction distribution (test set)", pred_dist)
    st.write("Signal distribution", sig_dist)

    history = _load_json(CHECKPOINTS_DIR / "training_history.json")
    if history is not None and len(history):
        df = pd.DataFrame(history)
        st.subheader("Rolling validation accuracy")
        cols = [c for c in df.columns if c.startswith("val_acc_horizon_")]
        if cols:
            st.line_chart(df[["epoch"] + cols].set_index("epoch"))


def page_feature_health() -> None:
    st.header("Feature Health")
    stats = _load_json(SPLITS_DIR / "feature_stats.json")
    if stats is None:
        st.info("feature_stats.json not found.")
        return

    X_train = _load_npy(SPLITS_DIR / "X_train.npy")
    X_test = _load_npy(SPLITS_DIR / "X_test.npy")
    metadata = _load_json(SPLITS_DIR / "metadata.json") or {}
    if X_train is None or X_test is None:
        st.info("X_train / X_test arrays missing — run scripts/build_dataset.py")
        return
    feature_cols = metadata.get("feature_columns", [f"f_{i}" for i in range(X_train.shape[1])])

    monitor = DriftMonitor(X_train, feature_names=feature_cols, alert_threshold=0.2)
    result = monitor.check_drift(X_test)
    df = pd.DataFrame(
        sorted(result["psi_scores"].items(), key=lambda kv: -kv[1]),
        columns=["feature", "psi"],
    )
    st.subheader("PSI scores (train vs test)")
    st.bar_chart(df.set_index("feature"))
    st.subheader("Top 10 most-drifted features")
    st.table(df.head(10))
    st.metric("Drift alert", "YES" if result["alert"] else "OK")


def page_backtest_summary() -> None:
    st.header("Backtest Summary")
    backtest = _load_json(PROJECT_ROOT / "data" / "backtest_results.json")
    if backtest is None:
        st.info("Run scripts/run_backtest.py to populate this page.")
        return
    report = backtest.get("report", {})
    cols = st.columns(4)
    metric_names = ["sharpe", "sortino", "max_drawdown", "hit_rate", "profit_factor", "n_trades"]
    for i, name in enumerate(metric_names):
        if name in report:
            cols[i % 4].metric(name.replace("_", " ").title(), f"{report[name]:.3f}" if isinstance(report[name], float) else str(report[name]))
    st.json(report)


def page_model_info() -> None:
    st.header("Model Info")
    history = _load_json(CHECKPOINTS_DIR / "training_history.json")
    if history is None:
        st.info("No training history yet — run scripts/train.py")
        return
    df = pd.DataFrame(history)
    st.subheader("Training curves")
    loss_cols = [c for c in df.columns if "loss" in c]
    if loss_cols:
        st.line_chart(df[["epoch"] + loss_cols].set_index("epoch"))
    acc_cols = [c for c in df.columns if "acc" in c]
    if acc_cols:
        st.line_chart(df[["epoch"] + acc_cols].set_index("epoch"))
    st.write("Total epochs:", len(history))
    st.write("Best val loss:", min(h["val_loss"] for h in history))


def main() -> None:
    st.set_page_config(page_title="LOB-TCN Monitor", layout="wide")
    st.sidebar.title("LOB-TCN Monitor")
    page = st.sidebar.radio(
        "Page",
        ["Live Predictions", "Feature Health", "Backtest Summary", "Model Info"],
    )
    st.sidebar.markdown(f"_Updated: {time.strftime('%Y-%m-%d %H:%M:%S')}_")
    if page == "Live Predictions":
        page_live_predictions()
    elif page == "Feature Health":
        page_feature_health()
    elif page == "Backtest Summary":
        page_backtest_summary()
    elif page == "Model Info":
        page_model_info()


if __name__ == "__main__":
    main()
