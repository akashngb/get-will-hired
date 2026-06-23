"""Tests for the FastAPI inference server."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.data.feature_engineer import FeatureEngineer
from src.data.synthetic import SyntheticLOBGenerator


REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    # Build a tiny dataset and stats so the API can boot
    out_dir = tmp_path_factory.mktemp("splits")
    ckpt_dir = tmp_path_factory.mktemp("ckpts")
    ob, msg = SyntheticLOBGenerator(n_events=2_500, seed=11).generate()
    fe = FeatureEngineer(levels=10, horizons=(10, 50, 100))
    df = fe.fit_transform(ob, msg)
    feat_names = fe.get_feature_names()
    fe.save_stats(out_dir / "feature_stats.json")
    np.save(out_dir / "X_train.npy", df[feat_names].to_numpy().astype(np.float32))

    os.environ["MODEL_CHECKPOINT_PATH"] = str(ckpt_dir / "missing.pt")
    os.environ["FEATURE_STATS_PATH"] = str(out_dir / "feature_stats.json")
    os.environ["MODEL_CONFIG_PATH"] = str(REPO_ROOT / "configs/tcn_small.yaml")
    os.environ["INFERENCE_SEQ_LEN"] = "32"

    from src.serving.api import app  # late import so env vars are set first

    with TestClient(app) as client:
        yield client


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["model_loaded"] is True
    # checkpoint intentionally missing -> degraded model still loaded with random weights
    assert body["checkpoint_loaded"] is False


def test_metrics_endpoint(client):
    r = client.get("/metrics")
    assert r.status_code == 200
    assert "lob_tcn_requests_total" in r.text


def test_predict_invalid_input_rows(client):
    r = client.post(
        "/predict",
        json={"orderbook_snapshot": [[0.0] * 40], "n_levels": 10, "sequence_length": 32},
    )
    assert r.status_code == 422


def test_predict_invalid_row_width(client):
    rows = [[0.0] * 36 for _ in range(200)]  # wrong column count
    r = client.post(
        "/predict",
        json={"orderbook_snapshot": rows, "n_levels": 10, "sequence_length": 32},
    )
    assert r.status_code == 422


def test_predict_happy_path(client):
    # Synthesize a realistic small book of ~250 rows
    ob, msg = SyntheticLOBGenerator(n_events=250, seed=42).generate()
    rows = []
    for i in range(len(ob)):
        row = []
        for lvl in range(1, 11):
            row.extend(
                [
                    float(ob[f"ask_price_{lvl}"].iloc[i]),
                    float(ob[f"ask_size_{lvl}"].iloc[i]),
                    float(ob[f"bid_price_{lvl}"].iloc[i]),
                    float(ob[f"bid_size_{lvl}"].iloc[i]),
                ]
            )
        rows.append(row)
    message_tape = [
        {
            "time": float(msg["time"].iloc[i]),
            "event_type": int(msg["event_type"].iloc[i]),
            "order_id": int(msg["order_id"].iloc[i]),
            "event_size": float(msg["event_size"].iloc[i]),
            "event_price": float(msg["event_price"].iloc[i]),
            "direction": int(msg["direction"].iloc[i]),
        }
        for i in range(len(msg))
    ]
    payload = {
        "orderbook_snapshot": rows,
        "n_levels": 10,
        "sequence_length": 32,
        "message_tape": message_tape,
    }
    r = client.post("/predict", json=payload)
    assert r.status_code == 200, r.text
    body = r.json()
    assert "predictions" in body
    assert "X-Inference-Time-Ms" in r.headers
    for horizon in ("horizon_10", "horizon_50", "horizon_100"):
        assert horizon in body["predictions"]
        assert body["predictions"][horizon]["direction"] in {"UP", "DOWN", "STATIONARY"}
