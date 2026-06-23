# Limit Order Book (LOB) Modeling with TCN — Full System Design
### For Claude Code: Plan, Architect, and Implement

---

## 0. AGENT OPERATING INSTRUCTIONS

This document is the single source of truth for building the LOB-TCN project end to end. Work through phases sequentially. Within each phase, implement all components before moving to the next. Follow these rules:

- **Never stop on a blocker.** If a dependency fails (data download, API rate limit, missing library), implement the documented fallback/placeholder and leave a `# TODO: REPLACE WITH REAL DATA` comment. Continue building.
- **Always stub before implementing.** Define all interfaces, function signatures, and class skeletons first. Fill in logic second.
- **Write tests alongside code.** Each module gets a corresponding `test_<module>.py` with at least 3 tests.
- **Log everything.** Use Python `logging` module throughout. Every pipeline stage emits structured logs.
- **Token efficiency:** Implement one file at a time. Confirm file is written before moving to next.
- **Environment:** Python 3.10+. Use `pyproject.toml` for dependency management. Virtual environment in `.venv/`.

---

## 1. PROJECT OVERVIEW

### 1.1 What We're Building

A production-grade machine learning system that:
1. **Ingests** Level 2 order book data (bid/ask depth snapshots) from LOBSTER dataset or synthetic fallback
2. **Engineers features** from raw order book state in real time
3. **Trains** a Temporal Convolutional Network (TCN) to predict short-term mid-price movement at three horizons: k=10, k=50, k=100 events ahead
4. **Serves inference** via a low-latency REST API with sub-100ms response time
5. **Backtests** predicted signals against a simple trading strategy with realistic transaction costs
6. **Monitors** live prediction drift and feature distribution shift

### 1.2 Target Metrics
- Model: >60% directional accuracy at k=10 (statistically significant over baseline)
- Inference: <100ms p99 latency on REST endpoint
- Pipeline: Processes 10,000 order book events/second in streaming simulation
- Backtest: Positive Sharpe ratio after transaction costs (0.5 bps per side)

### 1.3 Repository Layout

```
lob-tcn/
├── pyproject.toml
├── README.md
├── .env.example
├── data/
│   ├── raw/                    # Raw LOBSTER files or synthetic fallback
│   ├── processed/              # Featurized numpy arrays
│   └── splits/                 # Train/val/test splits (temporal)
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── lobster_loader.py   # LOBSTER data ingestion
│   │   ├── synthetic.py        # Synthetic LOB data generator (fallback)
│   │   ├── stream_simulator.py # Kafka-style event stream simulator
│   │   └── feature_engineer.py # LOB feature extraction
│   ├── models/
│   │   ├── __init__.py
│   │   ├── tcn.py              # TCN architecture
│   │   ├── baselines.py        # Naive baselines (mid-price, LSTM comparison)
│   │   └── train.py            # Training loop with W&B logging
│   ├── backtest/
│   │   ├── __init__.py
│   │   ├── strategy.py         # Signal → position logic
│   │   ├── engine.py           # Backtest engine
│   │   └── metrics.py          # Sharpe, max drawdown, alpha, beta
│   ├── serving/
│   │   ├── __init__.py
│   │   ├── api.py              # FastAPI inference server
│   │   ├── predictor.py        # Model loader + preprocessor
│   │   └── latency_bench.py    # Latency benchmarking script
│   └── monitoring/
│       ├── __init__.py
│       ├── drift.py            # Feature drift detection (PSI)
│       └── dashboard.py        # Streamlit monitoring dashboard
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_analysis.ipynb
│   ├── 03_model_ablation.ipynb
│   └── 04_backtest_analysis.ipynb
├── tests/
│   ├── test_lobster_loader.py
│   ├── test_feature_engineer.py
│   ├── test_tcn.py
│   ├── test_stream_simulator.py
│   ├── test_backtest_engine.py
│   └── test_api.py
├── configs/
│   ├── base.yaml
│   ├── tcn_small.yaml
│   └── tcn_large.yaml
└── scripts/
    ├── download_data.sh
    ├── run_training.sh
    ├── run_backtest.sh
    └── run_server.sh
```

---

## 2. DEPENDENCIES & ENVIRONMENT

### 2.1 pyproject.toml

```toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.backends.legacy:build"

[project]
name = "lob-tcn"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
    # Core ML
    "torch>=2.2.0",
    "numpy>=1.26.0",
    "pandas>=2.1.0",
    "scikit-learn>=1.4.0",

    # Data
    "requests>=2.31.0",
    "pyarrow>=14.0.0",
    "h5py>=3.10.0",

    # Serving
    "fastapi>=0.110.0",
    "uvicorn[standard]>=0.27.0",
    "pydantic>=2.6.0",

    # Experiment tracking
    "wandb>=0.16.0",
    "mlflow>=2.10.0",

    # Backtesting
    "vectorbt>=0.26.0",

    # Monitoring
    "streamlit>=1.31.0",
    "plotly>=5.18.0",

    # Utilities
    "pyyaml>=6.0.1",
    "python-dotenv>=1.0.0",
    "loguru>=0.7.2",
    "tqdm>=4.66.0",
    "click>=8.1.7",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
    "pytest-asyncio>=0.23.0",
    "pytest-cov>=4.1.0",
    "black>=24.0.0",
    "ruff>=0.2.0",
    "mypy>=1.8.0",
    "httpx>=0.26.0",   # for FastAPI test client
]
```

### 2.2 Environment Variables (.env.example)

```bash
# Data
LOBSTER_DATA_DIR=./data/raw
USE_SYNTHETIC_DATA=false      # Set true if LOBSTER unavailable

# Experiment Tracking
WANDB_PROJECT=lob-tcn
WANDB_ENTITY=your-entity
MLFLOW_TRACKING_URI=./mlruns

# Serving
API_HOST=0.0.0.0
API_PORT=8080
MODEL_CHECKPOINT_PATH=./checkpoints/best_model.pt

# Backtest
TRANSACTION_COST_BPS=0.5
INITIAL_CAPITAL=1000000
```

---

## 3. PHASE 1: DATA LAYER

### 3.1 LOBSTER Data Format

LOBSTER (https://lobsterdata.com) provides historical NASDAQ Level 2 order book data. Each stock/date pair produces two files:
- `TICKER_DATE_STARTTIME_ENDTIME_orderbook_LEVELS.csv` — order book snapshots
- `TICKER_DATE_STARTTIME_ENDTIME_message_LEVELS.csv` — event messages

**Order book CSV columns (10 levels):**
```
ASK_PRICE_1, ASK_SIZE_1, BID_PRICE_1, BID_SIZE_1,
ASK_PRICE_2, ASK_SIZE_2, BID_PRICE_2, BID_SIZE_2,
... (repeat for levels 3–10)
```
Prices are in integer format (dollars × 10000). Divide by 10000 to get USD.

**Message CSV columns:**
```
Time, Type, Order_ID, Size, Price, Direction
```
Message types: 1=New limit order, 2=Cancellation (partial), 3=Deletion, 4=Execution (visible), 5=Execution (hidden), 7=Trading halt

### 3.2 LOBSTER Data Access

**Primary path:** Register at https://lobsterdata.com/info/DataAccess.php
- Free sample data available for AAPL, AMZN, GOOG, INTC, MSFT (5 days each, 10 levels)
- Download script at `scripts/download_data.sh`

**FALLBACK (implement this first, replace later):**
If LOBSTER registration is pending or files aren't downloaded, `synthetic.py` generates statistically realistic LOB data using a Poisson arrival process. See Section 3.4.

### 3.3 src/data/lobster_loader.py

```python
"""
LOBSTER data loader. Reads raw CSV pairs and returns unified DataFrames.

IMPLEMENT:
- LobsterLoader class
    - __init__(self, data_dir: str, ticker: str, date: str, levels: int = 10)
    - load(self) -> tuple[pd.DataFrame, pd.DataFrame]
        Reads orderbook and message CSVs, validates columns, converts price units
    - to_snapshots(self) -> pd.DataFrame
        Merges orderbook + message files on index, returns event-aligned snapshot df
    - _validate(self, ob_df, msg_df) -> None
        Assert shapes, no nulls in critical columns, price monotonicity per level
    - _convert_prices(self, df: pd.DataFrame) -> pd.DataFrame
        Divide all price columns by 10000

COLUMN NAMING CONVENTION (output):
    bid_price_{i}, bid_size_{i}, ask_price_{i}, ask_size_{i}  for i in 1..levels
    time, event_type, order_id, event_size, event_price, direction

RAISES:
    FileNotFoundError — with clear message pointing to LOBSTER download instructions
    DataValidationError — custom exception, subclass of ValueError

FALLBACK BEHAVIOUR:
    If files not found, log a WARNING and return synthetic data by calling
    SyntheticLOBGenerator(ticker, date).generate() from synthetic.py
    Set a flag self.is_synthetic = True on the loader instance.
"""

import logging
from pathlib import Path
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class DataValidationError(ValueError):
    pass


class LobsterLoader:
    # TODO: Implement per docstring above
    pass
```

### 3.4 src/data/synthetic.py

**Implement this first — it's the fallback for everything.**

```python
"""
Synthetic LOB data generator using Poisson process simulation.

Market microstructure model:
- Mid-price follows arithmetic Brownian motion: dS = σ dW
- Bid-ask spread follows mean-reverting process (Ornstein-Uhlenbeck)
- Order arrivals: Poisson(λ_buy) and Poisson(λ_sell) processes
- Order sizes: log-normal distribution calibrated to realistic tick sizes
- 10 price levels on each side, quantities decay geometrically with depth

IMPLEMENT:
- SyntheticLOBGenerator class
    - __init__(self, n_events: int = 500_000, n_levels: int = 10,
               sigma: float = 0.0001, lambda_arrival: float = 10.0,
               spread_mean: float = 0.01, seed: int = 42)
    - generate(self) -> tuple[pd.DataFrame, pd.DataFrame]
        Returns (orderbook_df, message_df) in same format as LobsterLoader output
    - _simulate_midprice(self) -> np.ndarray
    - _simulate_spread(self) -> np.ndarray
    - _build_book_levels(self, mid: np.ndarray, spread: np.ndarray) -> pd.DataFrame
    - _build_messages(self) -> pd.DataFrame

REALISM REQUIREMENTS:
    - Prices must be strictly monotone: ask_price_1 < ask_price_2 < ... and
      bid_price_1 > bid_price_2 > ...
    - No negative prices or sizes
    - Spread always positive (ask_price_1 > bid_price_1 at every timestep)
    - Timestamp column in seconds from market open (9:30am = 34200.0)

UNIT TESTS (in tests/test_synthetic.py):
    1. test_price_monotonicity — assert no violations across full generated dataset
    2. test_spread_positive — assert ask_price_1 > bid_price_1 always
    3. test_shape — assert output has expected number of rows and columns
    4. test_reproducibility — same seed = same output
    5. test_midprice_statistics — check drift and volatility are within 20% of params
"""
```

### 3.5 src/data/feature_engineer.py

This is the most important data module. Features must be computed causally (no future data).

```python
"""
Feature engineering from raw LOB snapshots.

ALL FEATURES ARE CAUSAL — computed only from data available at time t.

FEATURE GROUPS:

1. PRICE FEATURES
   - mid_price: (ask_price_1 + bid_price_1) / 2
   - spread: ask_price_1 - bid_price_1
   - spread_bps: spread / mid_price * 10000
   - log_return_{n}: log(mid_price_t / mid_price_{t-n}) for n in [1, 5, 10, 20, 50]

2. VOLUME IMBALANCE FEATURES (core LOB signal)
   - bid_ask_imbalance_l1: (bid_size_1 - ask_size_1) / (bid_size_1 + ask_size_1)
   - bid_ask_imbalance_l{n}: same for levels 1-5
   - volume_imbalance_weighted: sum over levels of (bid_size_i - ask_size_i) * w_i
     where w_i = 1/i (depth-weighted, closer levels matter more)
   - total_bid_volume: sum of bid_size_1 through bid_size_10
   - total_ask_volume: sum of ask_size_1 through ask_size_10

3. PRICE LEVEL FEATURES
   - bid_price_range: bid_price_1 - bid_price_10 (depth of visible bid book)
   - ask_price_range: ask_price_10 - ask_price_1
   - price_pressure_bid: sum(bid_size_i / bid_price_i) for i in 1..10
     (volume-weighted bid pressure, proxy for latent demand)
   - price_pressure_ask: sum(ask_size_i / ask_price_i) for i in 1..10

4. FLOW FEATURES (from message file)
   - trade_flow_imbalance: (buy_volume - sell_volume) / (buy_volume + sell_volume)
     computed over rolling window of last W trades, W in [10, 50, 100]
   - order_arrival_rate: events per second in rolling 1s window
   - cancellation_rate: cancellations / total events in rolling window
   - kyle_lambda: price impact per unit volume (rolling OLS of |Δprice| on volume)

5. ROLLING STATISTICAL FEATURES
   - rolling_volatility_{w}: std of log returns over w in [20, 50, 100] events
   - rolling_autocorr_1: lag-1 autocorrelation of mid_price over 50 events
   - ewma_mid_{alpha}: exponential weighted moving average, alpha in [0.1, 0.3, 0.5]

6. LABELS (prediction targets)
   For horizon k in [10, 50, 100]:
   - future_mid_price_k: mid_price at event t+k
   - label_direction_k: +1 if future > current, -1 if future < current, 0 if equal
     (in practice 0 is rare; merge with +1 or -1 based on threshold)
   - label_smooth_k: uses average of k future mid-prices to reduce noise
     (mid_avg_{t+1..t+k} - mid_price_t) / mid_price_t, sign → label

IMPLEMENT:
- FeatureEngineer class
    - __init__(self, levels: int = 10, horizons: list[int] = [10, 50, 100])
    - fit_transform(self, ob_df: pd.DataFrame, msg_df: pd.DataFrame) -> pd.DataFrame
        Returns feature matrix with label columns appended
    - transform(self, ob_df: pd.DataFrame, msg_df: pd.DataFrame) -> pd.DataFrame
        Same as fit_transform but doesn't refit any stateful transforms (for serving)
    - _compute_price_features(self, df) -> pd.DataFrame
    - _compute_imbalance_features(self, df) -> pd.DataFrame
    - _compute_flow_features(self, df, msg_df) -> pd.DataFrame
    - _compute_rolling_features(self, df) -> pd.DataFrame
    - _compute_labels(self, df) -> pd.DataFrame
    - get_feature_names(self) -> list[str]   # for SHAP and debugging

CRITICAL IMPLEMENTATION NOTES:
    - All rolling computations must use pandas .rolling(window, min_periods=window)
      and drop NaN rows at the start. Never fill forward — prefer dropping.
    - Labels must be computed BEFORE any normalization so they reflect true direction.
    - Store column indices separately for features vs labels. Labels are NEVER
      passed into the model as inputs.
    - Save feature statistics (mean, std per feature) as self.feature_stats dict
      for use during inference normalization.

NORMALIZATION:
    - Use z-score normalization per feature: (x - mean) / std
    - Compute mean/std on training set only (fit_transform)
    - Apply same stats to val/test/live (transform)
    - DO NOT normalize label columns

OUTPUT DATAFRAME COLUMNS:
    [timestamp, all_feature_cols..., label_direction_10, label_direction_50,
     label_direction_100, label_smooth_10, label_smooth_50, label_smooth_100]
"""
```

### 3.6 src/data/stream_simulator.py

```python
"""
Kafka-style streaming event simulator for real-time inference demo.

Simulates a real-time feed of order book updates at configurable speed.
Used for the serving demo and latency benchmarking — not for training.

IMPLEMENT:
- StreamSimulator class
    - __init__(self, data_source: pd.DataFrame, speed_multiplier: float = 100.0)
        speed_multiplier: 100.0 means 100x real-time replay
    - stream(self) -> Generator[dict, None, None]
        Yields one event dict at a time, respecting simulated timestamps
        Each event: {'timestamp': float, 'orderbook': np.ndarray,
                     'message': dict, 'sequence_num': int}
    - reset(self) -> None

- StreamBuffer class
    - __init__(self, maxlen: int = 200)
        Maintains a rolling window of the last maxlen events
    - push(self, event: dict) -> None
    - get_feature_window(self) -> np.ndarray | None
        Returns (maxlen, n_features) array if buffer is full, else None
    - is_ready(self) -> bool

PERFORMANCE TARGET:
    stream() must yield events fast enough to saturate a 10k events/second consumer.
    Use time.sleep() only in real-time mode (speed_multiplier=1.0).
    In fast mode, yield without sleeping.
"""
```

### 3.7 Data Pipeline Script

```python
# scripts/build_dataset.py
"""
CLI script to run full data pipeline: raw → features → splits → saved arrays.

Usage:
    python scripts/build_dataset.py --ticker AAPL --date 2012-06-21
    python scripts/build_dataset.py --synthetic --n-events 1000000

Steps:
    1. Load raw data (LobsterLoader or SyntheticLOBGenerator)
    2. Run FeatureEngineer.fit_transform()
    3. Temporal train/val/test split: 70% / 15% / 15% by event index
       CRITICAL: NO shuffling. Order must be preserved.
    4. Save as .npy arrays to data/splits/:
       X_train.npy, X_val.npy, X_test.npy (shape: [n_events, seq_len, n_features])
       y_train.npy, y_val.npy, y_test.npy (shape: [n_events, 3] for 3 horizons)
    5. Save feature_stats.json for inference normalization
    6. Print summary statistics

IMPLEMENT this script with click CLI decorators.
"""
```

---

## 4. PHASE 2: MODEL LAYER

### 4.1 TCN Architecture — Theoretical Grounding

A Temporal Convolutional Network uses **dilated causal convolutions** to capture long-range temporal dependencies without data leakage. Key properties:

- **Causal:** output at time t depends only on inputs at times ≤ t. Enforced by left-padding.
- **Dilated:** dilation factor d^l at layer l means the convolution skips d^l-1 positions. With L layers and dilation [1, 2, 4, 8, ...], receptive field = 2^L × kernel_size.
- **Residual connections:** each block adds input to output, enabling deep networks to train.
- **No recurrence:** parallelizable across time during training (unlike LSTM/GRU).

**Why TCN over LSTM for LOB:**
- LOB data arrives at ~1000 events/second. At inference, we need to process each event in <1ms for the model computation itself. TCN inference is O(1) in sequence length (single forward pass on fixed window). LSTM is O(T) per sequence.
- Empirically, TCN matches or outperforms LSTM on LOB prediction tasks (Zhang et al., 2019).

### 4.2 src/models/tcn.py

```python
"""
Temporal Convolutional Network for LOB mid-price direction prediction.

ARCHITECTURE:

Input: (batch_size, seq_len, n_features)  → permute to (batch_size, n_features, seq_len)

TCN Block (repeated N times):
    DilatedCausalConv1d(in_channels, out_channels, kernel_size=3, dilation=2^i)
        Left-pad input by (kernel_size-1)*dilation so output length = input length
    BatchNorm1d(out_channels)
    ReLU
    Dropout(p=0.2)
    DilatedCausalConv1d(out_channels, out_channels, kernel_size=3, dilation=2^i)
    BatchNorm1d(out_channels)
    ReLU
    Dropout(p=0.2)
    Residual connection: 1x1 Conv if in_channels != out_channels, else identity

After N blocks:
    Global Average Pooling over time dimension → (batch_size, out_channels)

Output heads (one per horizon):
    Linear(out_channels, 64) → ReLU → Linear(64, 3)
    3 classes: price UP, price DOWN, price STATIONARY
    Separate head per horizon: head_10, head_50, head_100

IMPLEMENT:
- CausalConv1d(nn.Module)
    - __init__(self, in_channels, out_channels, kernel_size, dilation)
    - forward(self, x): x shape (B, C, T), output same shape
    - MUST left-pad by (kernel_size - 1) * dilation before conv, then trim right
      This ensures output[t] depends only on input[0..t]

- TCNBlock(nn.Module)
    - __init__(self, in_channels, out_channels, kernel_size, dilation, dropout)
    - forward(self, x)
    - Residual: if in_channels != out_channels, use nn.Conv1d(in, out, kernel_size=1)

- TCNModel(nn.Module)
    - __init__(self, n_features: int, n_classes: int = 3,
               n_levels: int = 4, n_channels: int = 64,
               kernel_size: int = 3, dropout: float = 0.2,
               horizons: list[int] = [10, 50, 100])
    - forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]
        Returns {'horizon_10': logits, 'horizon_50': logits, 'horizon_100': logits}
        Each logits shape: (batch_size, 3)
    - receptive_field(self) -> int
        Returns theoretical receptive field size (important: must exceed sequence length)
    - n_parameters(self) -> int

HYPERPARAMETER GRID (for ablation):
    n_levels: [3, 4, 5, 6]         → receptive field: [24, 48, 96, 192]
    n_channels: [32, 64, 128]
    kernel_size: [2, 3, 4]
    dropout: [0.1, 0.2, 0.3]

CAUSAL CHECK:
    Implement a test in tests/test_tcn.py that confirms causality:
    1. Forward pass on input x (B, T, F)
    2. Modify x[:, t:, :] for some t < T (future inputs)
    3. Re-run forward
    4. Assert outputs[:, :t] are identical — future changes must not affect past outputs

INITIALIZATION:
    - Conv layers: Kaiming normal (He initialization)
    - BatchNorm: weight=1, bias=0
    - Linear: Xavier uniform
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class CausalConv1d(nn.Module):
    # TODO: implement per docstring
    pass

class TCNBlock(nn.Module):
    # TODO: implement per docstring
    pass

class TCNModel(nn.Module):
    # TODO: implement per docstring
    pass
```

### 4.3 src/models/baselines.py

```python
"""
Baseline models to compare against TCN.

IMPLEMENT ALL — ablation study requires beating these convincingly:

1. MidPriceBaseline
   Always predicts "UP" (or "DOWN" — whichever is more frequent in training set).
   Sets the floor for directional accuracy.

2. SpreadMeanReversionBaseline
   If current mid > ewma(mid, 20): predict DOWN. Else predict UP.
   Classic naive mean-reversion. Should beat random, worse than TCN.

3. ImbalanceBaseline
   Uses only bid_ask_imbalance_l1 feature with logistic regression.
   Surprisingly competitive on short horizons — TCN must beat this.

4. LSTMBaseline(nn.Module)
   Single-layer LSTM with same output head as TCN.
   __init__(self, n_features, hidden_size=64, horizons=[10,50,100])
   forward(self, x) -> dict[str, torch.Tensor]
   Same input/output contract as TCNModel for fair comparison.

All baselines implement:
    predict(self, x: np.ndarray) -> np.ndarray
    evaluate(self, X: np.ndarray, y: np.ndarray) -> dict[str, float]
        Returns {'accuracy': float, 'f1_macro': float, 'f1_weighted': float}
"""
```

### 4.4 src/models/train.py

```python
"""
Training loop for TCN and LSTM models.

IMPLEMENT:

- LOBDataset(torch.utils.data.Dataset)
    - __init__(self, X: np.ndarray, y: np.ndarray, seq_len: int = 100)
        X shape: (n_events, n_features)
        y shape: (n_events, n_horizons)
        seq_len: number of events in each input window
    - __len__(self) -> int: n_events - seq_len - max(horizons)
    - __getitem__(self, idx) -> tuple[torch.Tensor, dict[str, torch.Tensor]]
        Returns (x_window, {label_10: tensor, label_50: tensor, label_100: tensor})
        x_window shape: (seq_len, n_features)
        CRITICAL: ensure no overlap between x_window and label timesteps

- Trainer class
    - __init__(self, model, config: dict, device: str = 'cuda')
    - train(self, train_loader, val_loader, n_epochs: int) -> dict
    - _train_epoch(self, loader) -> dict[str, float]
    - _val_epoch(self, loader) -> dict[str, float]
    - _compute_loss(self, outputs: dict, labels: dict) -> torch.Tensor
        Loss = sum over horizons of CrossEntropyLoss(logits_k, label_k)
        With optional class weighting for imbalanced UP/DOWN/STATIONARY
    - _log_metrics(self, metrics: dict, step: int) -> None
        Log to W&B and MLflow simultaneously
    - save_checkpoint(self, path: str, is_best: bool = False) -> None
    - load_checkpoint(self, path: str) -> None
    - early_stopping(self, val_loss: float) -> bool
        Returns True if training should stop (patience=10 epochs)

TRAINING CONFIG (configs/base.yaml):
    model:
      n_levels: 4
      n_channels: 64
      kernel_size: 3
      dropout: 0.2
    training:
      batch_size: 512
      learning_rate: 0.001
      weight_decay: 0.0001
      n_epochs: 100
      seq_len: 100
      patience: 10
      scheduler: cosine     # CosineAnnealingLR
    data:
      horizons: [10, 50, 100]
      train_split: 0.70
      val_split: 0.15
      test_split: 0.15

CLASS IMBALANCE HANDLING:
    LOB labels are often imbalanced (more UP events in bull runs).
    Compute class weights from training set: w_c = n_total / (n_classes * n_c)
    Pass to CrossEntropyLoss(weight=class_weights).

METRICS TO LOG (each epoch, per horizon):
    - loss (train, val)
    - accuracy (train, val)
    - f1_macro (val only)
    - f1_weighted (val only)
    - confusion_matrix (val, logged as W&B table every 5 epochs)
    - learning_rate (for scheduler debugging)
"""
```

---

## 5. PHASE 3: BACKTESTING

### 5.1 Strategy Logic

```python
# src/backtest/strategy.py
"""
Convert model predictions into trading positions.

STRATEGY: Signal-threshold mean-reversion with position sizing

Rules:
    - At each event t, receive prediction p ∈ {UP, DOWN, STATIONARY} for horizon k
    - If p == UP and current_position <= 0: enter long (buy 1 unit)
    - If p == DOWN and current_position >= 0: enter short (sell 1 unit)
    - If p == STATIONARY: hold current position
    - Maximum position size: ±1 unit (binary long/short/flat)
    - Hold position for k events, then close regardless of subsequent signals
    - Transaction cost: cost_bps basis points per trade, applied at mid-price

IMPLEMENT:
- Position dataclass
    direction: int      # +1 long, -1 short, 0 flat
    entry_price: float
    entry_time: int     # event index
    horizon: int        # events until forced exit

- SignalStrategy class
    - __init__(self, horizon: int = 10, cost_bps: float = 0.5,
               confidence_threshold: float = 0.6)
        Only trade if max(softmax(logits)) > confidence_threshold
    - generate_signals(self, predictions: np.ndarray,
                       probabilities: np.ndarray) -> np.ndarray
        Returns array of +1/0/-1 for each event
    - apply_costs(self, returns: np.ndarray, signals: np.ndarray) -> np.ndarray

IMPORTANT: Backtest must use test set predictions only.
           NEVER use training or validation set data in the backtest.
"""
```

### 5.2 Backtest Engine

```python
# src/backtest/engine.py
"""
Event-driven backtest engine.

IMPLEMENT:
- BacktestEngine class
    - __init__(self, strategy: SignalStrategy, initial_capital: float = 1_000_000)
    - run(self, prices: np.ndarray, signals: np.ndarray) -> pd.DataFrame
        Returns trade-by-trade ledger with columns:
        [event_idx, action, price, position, pnl, cumulative_pnl, capital]
    - summary(self) -> dict
        Returns all metrics from metrics.py

- Key implementation details:
    - Use mid_price series from test set as execution price
    - Slippage model: execute at mid + (spread/2) for buys, mid - (spread/2) for sells
    - Track gross PnL and net PnL (after costs) separately
    - No look-ahead: signal at t uses prediction made at t, executes at t+1
"""
```

### 5.3 Metrics

```python
# src/backtest/metrics.py
"""
IMPLEMENT these functions (all inputs are numpy arrays or floats):

- sharpe_ratio(returns: np.ndarray, periods_per_year: int = 252*6.5*3600) -> float
    Annualized Sharpe. periods_per_year for ~1-second events in a trading day.
    Handle zero std gracefully (return 0.0).

- sortino_ratio(returns: np.ndarray, periods_per_year: int) -> float
    Like Sharpe but penalizes only downside deviation.

- max_drawdown(cumulative_returns: np.ndarray) -> float
    Maximum peak-to-trough decline. Return as positive number (e.g., 0.15 for 15%).

- calmar_ratio(returns: np.ndarray, periods_per_year: int) -> float
    Annualized return / max_drawdown.

- hit_rate(signals: np.ndarray, actual_directions: np.ndarray) -> float
    Fraction of non-zero signals that were correct.

- profit_factor(gross_profits: float, gross_losses: float) -> float
    gross_profits / abs(gross_losses). > 1.0 is profitable before costs.

- alpha_beta(strategy_returns: np.ndarray,
             benchmark_returns: np.ndarray) -> tuple[float, float]
    OLS regression of strategy on benchmark. Returns (alpha_annualized, beta).

- full_report(returns, signals, actual_directions, benchmark_returns) -> dict
    Calls all above functions, returns unified dict for logging and display.
"""
```

---

## 6. PHASE 4: SERVING LAYER

### 6.1 FastAPI Inference Server

```python
# src/serving/api.py
"""
REST API for real-time LOB direction predictions.

ENDPOINTS:

POST /predict
    Input (JSON):
    {
        "orderbook_snapshot": [[ask_p1, ask_s1, bid_p1, bid_s1, ...], ...],
        "n_levels": 10,
        "sequence_length": 100   // number of historical events provided
    }
    Output:
    {
        "predictions": {
            "horizon_10": {"direction": "UP", "probability": 0.73, "logits": [...]},
            "horizon_50": {"direction": "DOWN", "probability": 0.61, "logits": [...]},
            "horizon_100": {"direction": "UP", "probability": 0.55, "logits": [...]}
        },
        "inference_time_ms": 12.4,
        "model_version": "v0.1.0",
        "sequence_num": 10423
    }

GET /health
    Returns {"status": "ok", "model_loaded": true, "uptime_seconds": 1234}

GET /metrics
    Returns Prometheus-format text metrics:
    - lob_tcn_inference_latency_ms (histogram)
    - lob_tcn_requests_total (counter)
    - lob_tcn_prediction_distribution (counter per direction/horizon)

POST /stream/start
    Starts the StreamSimulator in a background thread, begins streaming predictions
    to a Server-Sent Events (SSE) endpoint.

GET /stream/events
    SSE endpoint — yields prediction events as they arrive from simulator.

IMPLEMENT:
- Pydantic models for all request/response schemas
- Lifespan context manager for model loading on startup
- Middleware for request timing (add X-Inference-Time-Ms header to every response)
- Global exception handler returning {"error": str, "status_code": int}

LATENCY TARGETS:
    P50: <20ms
    P95: <50ms
    P99: <100ms
    Measure with latency_bench.py using 10,000 sequential requests.
"""
```

### 6.2 src/serving/predictor.py

```python
"""
Model wrapper for inference. Handles preprocessing, inference, postprocessing.

IMPLEMENT:
- Predictor class
    - __init__(self, checkpoint_path: str, feature_stats_path: str, device: str = 'cpu')
    - load_model(self) -> None
        Load TCNModel from checkpoint, set to eval mode, compile with torch.compile()
        if torch version >= 2.0
    - predict(self, raw_snapshot_sequence: list[dict]) -> dict
        Full pipeline: raw events → features → normalize → tensor → forward → decode
    - predict_batch(self, sequences: list) -> list[dict]
        Batched version for throughput benchmarking
    - warmup(self, n_warmup: int = 50) -> None
        Run n_warmup dummy predictions to prime JIT and caches

PREPROCESSING INSIDE PREDICTOR:
    Must mirror exactly what FeatureEngineer.transform() does.
    Load feature_stats.json and apply same z-score normalization.
    This is the most common source of train-serve skew — document carefully.

FALLBACK:
    If checkpoint not found, load a randomly-initialized model with a WARNING log.
    Predictions will be meaningless but server will start correctly for demo purposes.
"""
```

---

## 7. PHASE 5: MONITORING

### 7.1 src/monitoring/drift.py

```python
"""
Feature drift detection using Population Stability Index (PSI).

PSI measures how much a distribution has shifted between reference (training)
and current (live) data. PSI < 0.1: stable, 0.1-0.2: moderate shift, >0.2: alert.

PSI = sum over bins of (actual% - expected%) * ln(actual% / expected%)

IMPLEMENT:
- DriftMonitor class
    - __init__(self, reference_features: np.ndarray, n_bins: int = 10,
               alert_threshold: float = 0.2)
    - compute_psi(self, current_features: np.ndarray) -> dict[str, float]
        Returns PSI per feature column
    - check_drift(self, current_features: np.ndarray) -> dict
        Returns {'drifted_features': list, 'psi_scores': dict, 'alert': bool}
    - update_reference(self, new_reference: np.ndarray) -> None
        For online reference updates (optional)

ALSO IMPLEMENT:
- PredictionDriftMonitor class
    Tracks whether prediction distribution has shifted (e.g., always predicting UP).
    - __init__(self, window_size: int = 1000)
    - update(self, prediction: int) -> None
    - get_distribution(self) -> dict[str, float]
    - check_alert(self) -> bool
        Alert if any single class exceeds 80% of predictions in rolling window
"""
```

### 7.2 src/monitoring/dashboard.py

```python
"""
Streamlit monitoring dashboard.

PAGES:
1. Live Predictions
   - Real-time chart of predicted direction vs actual mid-price movement
   - Rolling accuracy for each horizon (last 1000 predictions)
   - Prediction distribution pie charts

2. Feature Health
   - PSI scores per feature (bar chart, red > 0.2)
   - Feature distribution histograms: training vs live (overlay)
   - Top 10 most-drifted features

3. Backtest Summary
   - Equity curve (cumulative PnL chart)
   - Key metrics table: Sharpe, Sortino, Max Drawdown, Hit Rate, Profit Factor
   - Trade log table (filterable)

4. Model Info
   - Architecture diagram (static image or mermaid)
   - Training curves (loss and accuracy per epoch, per horizon)
   - Confusion matrices per horizon

IMPLEMENT using st.session_state for state management.
Load pre-computed backtest results from data/backtest_results.json.
For live data, poll /metrics endpoint every 2 seconds.

Run with: streamlit run src/monitoring/dashboard.py
"""
```

---

## 8. PHASE 6: ABLATION STUDY

This is critical for the interview. You must be able to say: "I removed component X and performance dropped by Y%, which proves X is doing Z."

### 8.1 Ablations to Run

```python
# notebooks/03_model_ablation.ipynb
"""
ABLATION EXPERIMENTS — run all, log to W&B, visualize in notebook.

Each experiment trains from scratch with one change vs baseline (TCN, n_levels=4,
n_channels=64, seq_len=100, all features).

ARCHITECTURE ABLATIONS:
A1. TCN vs LSTM (same n_params) — proves TCN architectural choice
A2. TCN with dilation [1,1,1,1] vs [1,2,4,8] — proves dilated convs matter
A3. n_levels: 2, 3, 4, 5, 6 — receptive field vs accuracy tradeoff
A4. With vs without residual connections — proves ResNet-style connections help
A5. With vs without multi-head output (train all horizons jointly vs separately)

FEATURE ABLATIONS:
F1. Price features only (no imbalance, no flow) — baseline
F2. + imbalance features — most important individual group
F3. + flow features (Kyle's lambda, trade flow imbalance)
F4. + rolling statistical features — full model
F5. Remove only volume imbalance — tests its individual contribution
F6. Remove only Kyle's lambda — tests trade flow proxy importance

SEQUENCE LENGTH ABLATIONS:
S1. seq_len: 20, 50, 100, 200 events — how much history matters

RESULTS TABLE FORMAT:
    | Experiment | Horizon 10 Acc | Horizon 50 Acc | Horizon 100 Acc | Parameters | Train Time |
    |------------|---------------|----------------|-----------------|------------|------------|
    | Baseline   | 0.634         | 0.612          | 0.589           | 127K       | 4m32s      |
    | A1 (LSTM)  | 0.601         | 0.598          | 0.581           | 128K       | 9m14s      |
    ...

WHAT TO CONCLUDE (fill in after running):
    - "Dilated convolutions contribute X% accuracy improvement over uniform dilation"
    - "Volume imbalance is the single most important feature group: removing it
       drops accuracy by Y%"
    - "TCN trains 2x faster than LSTM with equivalent or better accuracy"
"""
```

---

## 9. TESTS

### 9.1 Critical Tests (must pass before each phase is complete)

```python
# tests/test_feature_engineer.py
"""
IMPLEMENT:
1. test_no_lookahead — for any feature at time t, modifying future rows (t+1 onward)
   must not change the feature value at t
2. test_label_shift — label_direction_10 at index i should reflect price change
   from i to i+10, not i to i-10
3. test_normalization_fit_transform — mean ≈ 0, std ≈ 1 for all features after
   fit_transform (within 0.01 tolerance)
4. test_normalization_transform — transform with saved stats produces different
   values than fit_transform on out-of-sample data (proves no re-fitting)
5. test_no_nan_in_output — after dropping warmup rows, no NaN in any column
6. test_class_balance — log class distribution; test passes always but prints warning
   if any class < 10% of total
"""

# tests/test_tcn.py
"""
IMPLEMENT:
1. test_causality — CRITICAL. See Section 4.2 for implementation.
2. test_output_shapes — forward pass produces dict with correct shapes
3. test_receptive_field — model.receptive_field() matches manual calculation
4. test_gradient_flow — after backward(), all parameter gradients are non-None
   and non-zero (checks for dead neurons or broken residual paths)
5. test_determinism — same input + same seed → same output
6. test_multi_horizon_independence — gradient of horizon_10 loss does not flow
   through horizon_50 head (separate output heads)
"""

# tests/test_api.py
"""
IMPLEMENT using FastAPI TestClient (httpx):
1. test_health_endpoint — GET /health returns 200 and model_loaded: true
2. test_predict_schema — POST /predict with valid input returns correct schema
3. test_predict_invalid_input — POST /predict with wrong shape returns 422
4. test_latency_header — X-Inference-Time-Ms header present in every response
5. test_concurrent_requests — 10 concurrent requests all return 200
   (use asyncio.gather with httpx.AsyncClient)
"""
```

---

## 10. INTERVIEW DEMO SCRIPT

The interviewer will ask you to walk through the project. Here is the exact narrative to deliver, with the artifacts to show:

### 10.1 Two-Minute Summary (memorize this)

> "I built an end-to-end system that predicts short-term price direction from Level 2 order book data. The core model is a Temporal Convolutional Network — I chose it over an LSTM because it's parallelizable, has a fixed receptive field I can control exactly, and inference is O(1) in sequence length which matters for latency. I engineered features that capture market microstructure: bid-ask imbalance across depth levels, trade flow toxicity using Kyle's lambda, and rolling volatility. I ran ablations that proved volume imbalance is the single most important feature group. I wrapped it in a FastAPI server hitting sub-100ms P99 latency, and ran a proper backtest on held-out test data with realistic transaction costs showing a positive Sharpe ratio."

### 10.2 Whiteboard Topics (be ready for any of these)

- Draw the TCN architecture: causal convolution, dilation pattern, residual block
- Derive why dilation [1,2,4,8] gives receptive field 2^L × kernel_size
- Explain bid-ask imbalance mathematically and intuitively
- Explain Kyle's lambda: what it measures, how you estimate it with OLS
- Why temporal train/test split and not random shuffle?
- What is look-ahead bias? Give a concrete example of how it could corrupt a feature.
- How would you detect if the model is degrading in production?
- What's PSI and when do you alert?
- How does the streaming pipeline work end to end?

### 10.3 Artifacts to Bring

1. Printed ablation results table (physical paper — shows rigour)
2. Laptop with Streamlit dashboard running locally
3. GitHub repo with clean commit history (one commit per phase)
4. `README.md` with a 3-command quickstart: install, generate data, run server

---

## 11. KNOWN BLOCKERS & FALLBACKS

| Blocker | Fallback |
|---|---|
| LOBSTER data registration takes >1 week | Use `SyntheticLOBGenerator` — flip `USE_SYNTHETIC_DATA=true` in `.env` |
| LOBSTER free tier only covers 5 stocks × 5 days | Fine — AAPL 5 days is ~1.5M events, more than enough |
| W&B account needed for logging | Fall back to MLflow local (`./mlruns`). Set `USE_WANDB=false` in config. |
| `vectorbt` install fails on some platforms | Implement simple backtest engine manually in `engine.py` — it's cleaner anyway |
| GPU not available | TCN is small enough to train on CPU in ~2 hours. Set `device=cpu` in config. |
| `torch.compile()` fails (Windows / older PyTorch) | Wrap in try/except, fall back to eager mode with a WARNING log |
| Port 8080 in use | Set `API_PORT=8081` in `.env` |

---

## 12. IMPLEMENTATION ORDER

Follow this sequence exactly. Do not start Phase N+1 until Phase N tests pass.

```
Phase 1a: SyntheticLOBGenerator → tests pass
Phase 1b: FeatureEngineer (on synthetic data) → tests pass
Phase 1c: LobsterLoader (with synthetic fallback) → tests pass
Phase 1d: StreamSimulator → tests pass
Phase 1e: build_dataset.py script runs end to end

Phase 2a: CausalConv1d + causality test passes
Phase 2b: TCNBlock
Phase 2c: TCNModel full forward pass → shape tests pass
Phase 2d: Baselines implemented
Phase 2e: Trainer + LOBDataset → training loop runs 1 epoch without error
Phase 2f: Full training run → checkpoint saved

Phase 3a: SignalStrategy
Phase 3b: BacktestEngine
Phase 3c: Metrics
Phase 3d: Full backtest run → results JSON saved

Phase 4a: Predictor (loads checkpoint, runs predict())
Phase 4b: FastAPI server → health endpoint
Phase 4c: /predict endpoint → latency benchmark
Phase 4d: SSE streaming endpoint

Phase 5a: DriftMonitor
Phase 5b: PredictionDriftMonitor
Phase 5c: Streamlit dashboard (all 4 pages)

Phase 6: Ablation study (parallel training runs)
Phase 7: Notebooks, README, demo polish
```
