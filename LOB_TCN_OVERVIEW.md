# What We're Building: A Plain-English Guide
### Limit Order Book Prediction with Deep Learning

---

## The One-Sentence Version

We're building a system that watches the live buy and sell orders sitting in a stock exchange, learns the patterns in how those orders move, and predicts whether the price is about to go up or down — fast enough to be useful in a real trading environment.

---

## Part 1: The Problem

### What is a stock exchange actually doing?

Most people think of a stock price as a single number. But at any moment, the exchange is actually managing two running lists:

- **The buy side (bids):** People who want to buy the stock, ranked by the highest price they're willing to pay
- **The sell side (asks):** People who want to sell the stock, ranked by the lowest price they're willing to accept

The current "price" you see quoted is just the midpoint between the best bid (highest willing buyer) and the best ask (lowest willing seller).

Here's what it looks like for a simplified example with 3 levels on each side:

```
SELL ORDERS (asks)
  Ask Level 3:  $100.06  ←  200 shares available
  Ask Level 2:  $100.04  ←  150 shares available
  Ask Level 1:  $100.02  ←  100 shares available  ← best ask (lowest seller)
  ─────────────────────────────────────────────────
  Mid-price:    $100.01
  ─────────────────────────────────────────────────
  Bid Level 1:  $100.00  ←  80 shares wanted      ← best bid (highest buyer)
  Bid Level 2:  $99.98   ←  120 shares wanted
  Bid Level 3:  $99.96   ←  300 shares wanted
BUY ORDERS (bids)
```

This full picture — all the visible buy and sell orders stacked up by price — is called the **Limit Order Book (LOB)**. It updates thousands of times per second as new orders arrive, old orders get cancelled, and trades happen.

### What moves the price?

The mid-price moves when the balance of buying and selling pressure shifts. If there are suddenly far more buyers than sellers sitting in the book, the price tends to drift up as buyers compete with each other. If there's a flood of sell orders, the price drifts down.

These shifts leave fingerprints in the order book *before* the price actually moves. That's the pattern we're trying to learn.

### Why is this hard?

Because financial markets are what researchers call **adversarial environments**. The moment a pattern becomes widely known and exploited, traders trade against it and it disappears. The signals are buried in noise, they're short-lived, and the data arrives faster than a human can process.

This makes it a perfect problem for machine learning.

---

## Part 2: The Data

### What does a single data point look like?

Every time something happens in the market — a new order placed, an order cancelled, a trade executed — we get a snapshot of the entire order book at that moment. A single snapshot for 10 levels on each side looks like this:

| Column | Value | Meaning |
|---|---|---|
| ask_price_1 | 100.02 | Cheapest price anyone is selling at |
| ask_size_1 | 100 | How many shares at that price |
| bid_price_1 | 100.00 | Highest price anyone is buying at |
| bid_size_1 | 80 | How many shares at that price |
| ask_price_2 | 100.04 | Next cheapest sell order |
| ... | ... | ... (10 levels each side) |

For a busy stock like Apple, we get roughly **300,000 of these snapshots per trading day**. A five-day sample gives us around 1.5 million data points — plenty to train a model.

### Where does the data come from?

We use **LOBSTER** (Limit Order Book System: The Efficient Reconstructor), a dataset from NASDAQ that provides exactly these order book snapshots for major US stocks. It's the standard academic dataset for this type of research.

If that data isn't available yet during setup, we also built a **synthetic data generator** that simulates a realistic order book using statistical models. It's not real market data, but it has the right mathematical properties to develop and test everything against.

---

## Part 3: The Features

Raw order book snapshots aren't fed directly into the model. We first compute **features** — derived numbers that capture meaningful signals more explicitly. Think of features as translating the raw data into a language the model can learn from more easily.

### The most important features

**Bid-Ask Imbalance**
This is the most powerful single signal in the order book. It compares how much volume is sitting on the buy side vs the sell side at the top of the book:

```
Imbalance = (buy_volume - sell_volume) / (buy_volume + sell_volume)
```

A value close to +1 means buyers heavily outnumber sellers → price likely to go up.
A value close to -1 means sellers heavily outnumber buyers → price likely to go down.
We compute this for all 10 levels, not just the top, because deeper levels provide additional signal about latent pressure.

**Kyle's Lambda (Trade Flow Toxicity)**
Named after economist Albert Kyle, this measures how much the price moves per unit of trading volume. When "informed traders" — people who actually know something — trade, they move the price more per share than random noise traders do. A high lambda means the market is sensing informed flow, which is a signal that a significant price move is coming.

**Rolling Volatility**
How much has the price been jumping around in the last 50 events? High volatility changes how the model should interpret other signals, so we provide this as context.

**Log Returns**
The percentage price change over the last N events, computed as log(price_now / price_before). We compute this for several lookback windows (last 5, 10, 20, 50 events) to give the model a sense of recent price momentum.

### The labels (what we're predicting)

For each snapshot, we look at what the price does over the next K events and classify it:

- **UP:** Average future price is meaningfully above current price
- **DOWN:** Average future price is meaningfully below current price
- **STATIONARY:** Price doesn't move much

We do this for three different prediction horizons: K=10 events ahead, K=50 events ahead, and K=100 events ahead. The model predicts all three simultaneously, which helps it share what it learns across timescales.

---

## Part 4: The Model

### What is a neural network, briefly?

A neural network is a mathematical function with millions of adjustable parameters (called weights). We feed it our features, it produces a prediction, we measure how wrong that prediction is, and we adjust all the weights slightly to make the next prediction a bit more accurate. Do this millions of times across all our training data, and the network gradually learns to extract the patterns that matter.

### Why a Temporal Convolutional Network (TCN)?

The order book is sequential data — each snapshot is connected to the ones before it. The natural tool for sequential data is a family of models called **recurrent networks** (LSTMs are the most famous). But we chose a different architecture called a TCN for three reasons:

**1. It's causal by design.**
"Causal" means the model's prediction at time T can only look at data from times T and earlier — never the future. This is trivially important: we're trying to predict the future from the past. A small implementation mistake in an LSTM can accidentally let future data leak into past predictions during training, making results look fantastic in the lab and useless in production. TCNs enforce causality structurally, at the architecture level.

**2. It has a controllable memory.**
An LSTM theoretically has infinite memory but practically struggles with very long sequences. A TCN has a precisely calculable **receptive field** — the exact number of past events it can see. We design this on purpose: with 4 layers and a dilation pattern of [1, 2, 4, 8], the model sees exactly the last 48 events. We know what we're building.

**3. Inference is fast.**
An LSTM must process a sequence step-by-step, one event at a time. A TCN processes the entire window in one parallel operation. For a system that needs to produce predictions in under 100 milliseconds, this matters.

### How the TCN works (the gist)

The key ingredient is a **dilated causal convolution**. A regular convolution looks at a small window of adjacent inputs and computes a weighted average. A dilated convolution does the same but *skips* positions — with dilation=4, it looks at every 4th event. Stack multiple layers with increasing dilation and the model can efficiently see a large window of history without having an astronomically large number of parameters.

```
Layer 1 (dilation=1): looks at events [t, t-1, t-2]
Layer 2 (dilation=2): looks at events [t, t-2, t-4]
Layer 3 (dilation=4): looks at events [t, t-4, t-8]
Layer 4 (dilation=8): looks at events [t, t-8, t-16]

Combined receptive field: 48 events of history
```

Each layer is also connected to the output by a **residual connection** (an idea from image recognition models) — this means even if a layer learns something unhelpful, it can effectively "skip itself" by learning zero weights, preventing the network from getting worse as it gets deeper.

---

## Part 5: Training

### The train/val/test split

We split our data into three non-overlapping time periods:
- **Training set (70%):** The model learns from this. Weights are updated based on errors here.
- **Validation set (15%):** We check performance here during training to tune settings. The model never trains on this data, but we do use it to make decisions about the model, so it's not fully "held out."
- **Test set (15%):** Touched exactly once, at the end, to report final performance. This is the number we report. Everything else is practice.

**Critical rule:** The split is *temporal* — training data is always earlier than validation, which is always earlier than test. We never shuffle. Shuffling would let the model learn from "future" data when predicting the "past," which is cheating that only shows up when you deploy.

### What the model is optimizing

During training, the model produces probability distributions: "I think there's a 65% chance the price goes UP, 25% DOWN, 10% STATIONARY." The loss function (cross-entropy) measures how wrong these probabilities are compared to what actually happened. The training process adjusts the model's weights to minimize this loss.

### Ablation studies

We don't just train one model. We train many versions with individual components removed or changed, so we can prove what's actually working:

- *Remove the volume imbalance features* → accuracy drops 8% → proves imbalance is the most important feature group
- *Replace TCN with LSTM* → same accuracy, but training takes twice as long → justifies architecture choice
- *Remove dilated convolutions* → accuracy drops → proves the architecture isn't arbitrary
- *Reduce sequence length from 100 to 20* → accuracy drops → proves we need a long memory window

This kind of systematic experimentation is what separates a serious ML project from a one-off experiment.

---

## Part 6: The Real-Time System

Training a model is only half the work. The other half is making it *usable*.

### The streaming pipeline

In a real trading environment, order book events arrive continuously, one after another. We simulate this with a **stream simulator** that replays our historical data at high speed, yielding one event at a time to downstream consumers.

The system maintains a rolling buffer of the last 100 events. Every time a new event arrives, it:
1. Pushes the new event into the buffer
2. Computes features for the current window
3. Feeds the feature window into the model
4. Receives UP/DOWN/STATIONARY predictions for all three horizons
5. Logs the prediction and (eventually) whether it was right

### The REST API

We wrap the model in a **FastAPI server** — a Python web server that accepts HTTP requests. Any other system (a trading system, a dashboard, a phone app) can send it an order book snapshot and receive predictions back in JSON format. We benchmark this to ensure the P99 latency (worst case for 99% of requests) stays below 100 milliseconds.

### The monitoring dashboard

A Streamlit dashboard (a simple Python web app framework) shows:
- Live rolling accuracy for each prediction horizon
- Which features have drifted away from their training distribution (using a metric called PSI — Population Stability Index)
- The equity curve from the backtest (see below)
- Model training history

---

## Part 7: The Backtest

A backtest is a simulation of how much money the strategy would have made if we had used it historically. We run it on the test set only — data the model has never seen.

**The strategy is simple:**
- If the model predicts UP with >60% confidence: buy
- If the model predicts DOWN with >60% confidence: sell
- If the model is not confident: do nothing
- Every trade costs 0.5 basis points (0.005%) in transaction costs — a realistic estimate for institutional trading

We then compute:
- **Sharpe Ratio:** Risk-adjusted return (higher is better; >1.0 is considered good)
- **Maximum Drawdown:** The worst peak-to-trough loss during the period
- **Hit Rate:** What fraction of trades were in the right direction

A positive Sharpe ratio after transaction costs means the strategy produces genuine signal — it's not just memorizing noise.

---

## The Full Picture

Here's how every component connects:

```
Raw Order Book Data (LOBSTER)
          ↓
    Feature Engineer
  (imbalance, flow,       ← the "language" we teach the model
   volatility, returns)
          ↓
     TCN Model            ← learns patterns in sequences of features
  (causal, dilated,
   residual blocks)
          ↓
  UP / DOWN / STATIONARY  ← one prediction per horizon (10, 50, 100 events)
  predictions
       ↙         ↘
Backtest Engine    FastAPI Server
(did it make       (serves live
 money?)            predictions)
                        ↓
                 Streamlit Dashboard
                 (monitors drift,
                  accuracy, PnL)
```

Every component is tested, every design decision has a reason, and every claim is backed by a number from the ablation study. That's the project.
