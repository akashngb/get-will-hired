"""Latency benchmark for the predictor — see Section 6.1 latency targets."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.serving.predictor import Predictor  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s :: %(message)s",
)
logger = logging.getLogger("latency_bench")


def benchmark(predictor: Predictor, n_requests: int, seq_len: int) -> dict:
    n_features = len(predictor.feature_engineer.get_feature_names())  # type: ignore[union-attr]
    rng = np.random.default_rng(0)
    samples = [
        rng.standard_normal((seq_len, n_features)).astype(np.float32) for _ in range(n_requests)
    ]
    predictor.warmup(n_warmup=10)

    latencies = []
    for sample in samples:
        t0 = time.perf_counter()
        predictor.predict_from_window(sample)
        latencies.append((time.perf_counter() - t0) * 1000.0)
    arr = np.array(latencies)
    summary = {
        "n_requests": n_requests,
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),
        "mean_ms": float(arr.mean()),
        "max_ms": float(arr.max()),
        "min_ms": float(arr.min()),
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pt")
    parser.add_argument("--feature-stats", default="data/splits/feature_stats.json")
    parser.add_argument("--config", default="configs/tcn_small.yaml")
    parser.add_argument("--n-requests", type=int, default=1000)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", default="data/latency_benchmark.json")
    args = parser.parse_args()

    predictor = Predictor(
        checkpoint_path=args.checkpoint,
        feature_stats_path=args.feature_stats,
        config_path=args.config,
        device=args.device,
        seq_len=args.seq_len,
    )
    predictor.load_model()

    summary = benchmark(predictor, args.n_requests, args.seq_len)
    Path(args.out).write_text(json.dumps(summary, indent=2))
    logger.info("Latency summary:")
    for k, v in summary.items():
        logger.info("  %s: %s", k, v)


if __name__ == "__main__":
    main()
