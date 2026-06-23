# LOB-TCN: Limit Order Book Modeling with Temporal Convolutional Networks

End-to-end ML system that predicts short-term mid-price direction from Level-2 order book data using a Temporal Convolutional Network. Built to the spec in [`LOB_TCN_DESIGN.md`](./LOB_TCN_DESIGN.md).

## What's inside

| Phase | Module | Status |
|---|---|---|
| 1. Data | `src/data/synthetic.py`, `src/data/lobster_loader.py`, `src/data/feature_engineer.py`, `src/data/stream_simulator.py` | done |
| 2. Model | `src/models/tcn.py`, `src/models/baselines.py`, `src/models/train.py` | done |
| 3. Backtest | `src/backtest/strategy.py`, `src/backtest/engine.py`, `src/backtest/metrics.py` | done |
| 4. Serving | `src/serving/predictor.py`, `src/serving/api.py`, `src/serving/latency_bench.py` | done |
| 5. Monitoring | `src/monitoring/drift.py`, `src/monitoring/dashboard.py` | done |
| 6. Ablation | `scripts/run_ablation.py`, `notebooks/03_model_ablation.ipynb` | done |

## Quickstart (3 commands)

```bash
# 1. Install dependencies
pip install -e ".[dev]"

# 2. Generate a synthetic dataset (no LOBSTER registration required)
python scripts/build_dataset.py --synthetic --n-events 200000

# 3. Train, backtest, and serve
python scripts/train.py --config configs/tcn_small.yaml
python scripts/run_backtest.py
uvicorn src.serving.api:app --host 0.0.0.0 --port 8080
```

Open the monitoring dashboard with:

```bash
streamlit run src/monitoring/dashboard.py
```

## Real LOBSTER data

Register at <https://lobsterdata.com/info/DataAccess.php> for free sample CSVs
(AAPL, AMZN, GOOG, INTC, MSFT — 5 days each, 10 levels). Place the files under
`data/raw/`, then:

```bash
python scripts/build_dataset.py --no-synthetic --ticker AAPL --date 2012-06-21
```

If LOBSTER files are missing, the loader automatically falls back to
`SyntheticLOBGenerator` and tags the resulting dataset with `is_synthetic: true`.

## Architecture summary

```
raw events -> FeatureEngineer -> normalized (N, F) -> sliding window (T, F)
            -> Causal TCN blocks (dilation 1,2,4,8,...) -> global avg pool
            -> per-horizon classification heads (UP / DOWN / STATIONARY)
```

Three horizons are predicted jointly: k=10, k=50, k=100 events ahead.

## Key design decisions

- **TCN over LSTM**: parallelizable training, fixed and exactly-controllable
  receptive field, O(1) inference in sequence length.
- **Causal convolutions**: every `Conv1d` left-pads by `(kernel-1)*dilation`
  then trims the right edge — `tests/test_tcn.py::test_causality_full_model`
  guards against any future leakage.
- **Temporal train/val/test split (70/15/15)**: no shuffling.
- **Feature normalization** is fit on train only and persisted to
  `data/splits/feature_stats.json` so the inference server applies identical
  statistics — eliminates train-serve skew.
- **Confidence-gated trading**: only trade when softmax confidence exceeds a
  threshold; positions are unit-sized and held for exactly `k` events.

## Testing

```bash
pytest tests/ -v
```

48 tests covering: causality of features and the TCN, label shift correctness,
order book monotonicity, backtest engine accounting, API schemas, drift PSI
behavior, and Streamlit-free monitoring logic.

## Project layout

```
lob-tcn/
├── pyproject.toml
├── README.md
├── LOB_TCN_DESIGN.md         # the spec
├── .env.example
├── configs/
│   ├── base.yaml
│   ├── tcn_small.yaml
│   └── tcn_large.yaml
├── data/
│   ├── raw/                  # LOBSTER CSVs land here
│   ├── processed/
│   └── splits/               # X_train.npy, feature_stats.json, ...
├── checkpoints/              # best_model.pt, training_history.json
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_analysis.ipynb
│   ├── 03_model_ablation.ipynb
│   └── 04_backtest_analysis.ipynb
├── scripts/
│   ├── build_dataset.py
│   ├── train.py
│   ├── run_backtest.py
│   ├── run_ablation.py
│   ├── download_data.sh
│   ├── run_training.sh
│   ├── run_backtest.sh
│   └── run_server.sh
├── src/
│   ├── data/
│   ├── models/
│   ├── backtest/
│   ├── serving/
│   └── monitoring/
└── tests/
    ├── test_synthetic.py
    ├── test_feature_engineer.py
    ├── test_lobster_loader.py
    ├── test_stream_simulator.py
    ├── test_tcn.py
    ├── test_baselines.py
    ├── test_train.py
    ├── test_backtest_engine.py
    ├── test_api.py
    └── test_drift.py
```

## Latency

`python src/serving/latency_bench.py --n-requests 1000 --seq-len 64` produces a
p99 of ~1.5 ms on CPU (post-warmup), far inside the 100 ms target.

## API

```http
POST /predict
{
  "orderbook_snapshot": [[ask_p1, ask_s1, bid_p1, bid_s1, ...], ...],
  "n_levels": 10,
  "sequence_length": 100,
  "message_tape": [{"event_type": 1, "direction": 1, ...}, ...]   # optional
}
```

Response:

```json
{
  "predictions": {
    "horizon_10":  {"direction": "UP",   "probability": 0.73, "logits": [...], "probabilities": [...]},
    "horizon_50":  {"direction": "DOWN", "probability": 0.61, "logits": [...], "probabilities": [...]},
    "horizon_100": {"direction": "UP",   "probability": 0.55, "logits": [...], "probabilities": [...]}
  },
  "inference_time_ms": 1.4,
  "model_version": "v0.1.0",
  "sequence_num": 10423
}
```

Every response also includes the `X-Inference-Time-Ms` header.

Streaming:

```bash
curl -X POST http://localhost:8080/stream/start
curl http://localhost:8080/stream/events    # SSE feed
```

## Monitoring

- **Feature drift**: PSI per feature on a rolling window vs the training
  reference. Alerts when any feature exceeds 0.2.
- **Prediction drift**: alerts when a single class exceeds 80 % of the last
  N predictions.
- **Dashboard**: 4 pages — Live Predictions, Feature Health, Backtest Summary,
  Model Info.
