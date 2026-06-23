"""FastAPI inference server — see Section 6.1.

Endpoints:
    POST /predict
    GET  /health
    GET  /metrics
    POST /stream/start  (background SSE stream)
    GET  /stream/events
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from collections import Counter, deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import PlainTextResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

from src.serving.predictor import Predictor

logger = logging.getLogger(__name__)


# ----- Pydantic schemas ------------------------------------------------------


class PredictRequest(BaseModel):
    orderbook_snapshot: list[list[float]] = Field(
        ..., description="List of LOBSTER-style rows: [ask_p1, ask_s1, bid_p1, bid_s1, ...]"
    )
    n_levels: int = 10
    sequence_length: int = 100
    message_tape: list[dict] | None = None


class PredictionEntry(BaseModel):
    direction: str
    probability: float
    logits: list[float]
    probabilities: list[float]


class PredictResponse(BaseModel):
    predictions: dict[str, PredictionEntry]
    inference_time_ms: float
    model_version: str
    sequence_num: int


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    checkpoint_loaded: bool
    uptime_seconds: float


# ----- App state --------------------------------------------------------------


class AppState:
    predictor: Predictor | None = None
    started_at: float = time.time()
    request_counter: int = 0
    latency_history: deque = deque(maxlen=10_000)
    prediction_counts: Counter = Counter()
    stream_task: asyncio.Task | None = None
    stream_queue: asyncio.Queue | None = None


state = AppState()


# ----- Lifespan ---------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    checkpoint = os.getenv("MODEL_CHECKPOINT_PATH", "./checkpoints/best_model.pt")
    feature_stats = os.getenv("FEATURE_STATS_PATH", "./data/splits/feature_stats.json")
    config = os.getenv("MODEL_CONFIG_PATH", "./configs/tcn_small.yaml")
    device = os.getenv("INFERENCE_DEVICE", "cpu")
    seq_len = int(os.getenv("INFERENCE_SEQ_LEN", "64"))

    predictor = Predictor(
        checkpoint_path=checkpoint,
        feature_stats_path=feature_stats,
        config_path=config,
        device=device,
        seq_len=seq_len,
    )
    try:
        predictor.load_model()
        predictor.warmup(n_warmup=2)
        state.predictor = predictor
        logger.info("Predictor ready. checkpoint_loaded=%s", predictor.checkpoint_loaded)
    except FileNotFoundError as exc:
        logger.error("Predictor init failed: %s", exc)
        state.predictor = None
    yield
    state.predictor = None


app = FastAPI(title="LOB-TCN Inference Server", version=Predictor.MODEL_VERSION, lifespan=lifespan)


# ----- Middleware -------------------------------------------------------------


@app.middleware("http")
async def add_timing_header(request: Request, call_next):
    t0 = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    response.headers["X-Inference-Time-Ms"] = f"{elapsed_ms:.3f}"
    state.latency_history.append(elapsed_ms)
    return response


@app.exception_handler(Exception)
async def global_handler(request: Request, exc: Exception):
    logger.exception("unhandled error in %s", request.url.path)
    status = getattr(exc, "status_code", 500)
    return Response(
        content=json.dumps({"error": str(exc), "status_code": status}),
        status_code=status,
        media_type="application/json",
    )


# ----- Routes -----------------------------------------------------------------


@app.get("/health", response_model=HealthResponse)
async def health() -> dict[str, Any]:
    return {
        "status": "ok" if state.predictor is not None else "degraded",
        "model_loaded": state.predictor is not None,
        "checkpoint_loaded": getattr(state.predictor, "checkpoint_loaded", False),
        "uptime_seconds": time.time() - state.started_at,
    }


@app.post("/predict", response_model=PredictResponse)
async def predict(body: PredictRequest) -> dict[str, Any]:
    if state.predictor is None:
        raise HTTPException(status_code=503, detail="predictor not loaded")
    rows = body.orderbook_snapshot
    if len(rows) < body.sequence_length:
        raise HTTPException(
            status_code=422,
            detail=(
                f"orderbook_snapshot has {len(rows)} rows; "
                f"sequence_length requires at least {body.sequence_length}"
            ),
        )
    n_cols_expected = body.n_levels * 4
    if any(len(r) != n_cols_expected for r in rows):
        raise HTTPException(
            status_code=422,
            detail=f"each row must contain {n_cols_expected} values",
        )

    events = _rows_to_events(rows, body.n_levels, body.message_tape)
    try:
        result = state.predictor.predict(events)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    state.request_counter += 1
    for name, entry in result["predictions"].items():
        state.prediction_counts[(name, entry["direction"])] += 1
    result["sequence_num"] = state.request_counter
    return result


@app.get("/metrics", response_class=PlainTextResponse)
async def metrics() -> str:
    lines: list[str] = []
    lines.append("# HELP lob_tcn_requests_total Total /predict requests")
    lines.append("# TYPE lob_tcn_requests_total counter")
    lines.append(f"lob_tcn_requests_total {state.request_counter}")
    lines.append("# HELP lob_tcn_inference_latency_ms Per-request inference latency, ms")
    lines.append("# TYPE lob_tcn_inference_latency_ms summary")
    if state.latency_history:
        arr = np.array(state.latency_history)
        for q, name in [(50, "p50"), (95, "p95"), (99, "p99")]:
            lines.append(
                f'lob_tcn_inference_latency_ms{{quantile="{q / 100:.2f}"}} '
                f"{float(np.percentile(arr, q)):.3f}"
            )
    lines.append("# HELP lob_tcn_prediction_distribution Predictions per direction/horizon")
    lines.append("# TYPE lob_tcn_prediction_distribution counter")
    for (horizon, direction), count in state.prediction_counts.items():
        lines.append(
            f'lob_tcn_prediction_distribution{{horizon="{horizon}",direction="{direction}"}} '
            f"{count}"
        )
    return "\n".join(lines) + "\n"


@app.post("/stream/start")
async def stream_start(request: Request) -> dict[str, str]:
    if state.predictor is None:
        raise HTTPException(status_code=503, detail="predictor not loaded")
    if state.stream_task and not state.stream_task.done():
        return {"status": "already_running"}

    state.stream_queue = asyncio.Queue(maxsize=100)
    state.stream_task = asyncio.create_task(_simulate_stream())
    return {"status": "started"}


@app.get("/stream/events")
async def stream_events() -> StreamingResponse:
    if state.stream_queue is None:
        raise HTTPException(status_code=400, detail="stream not started")
    queue = state.stream_queue

    async def gen():
        while True:
            try:
                event = await asyncio.wait_for(queue.get(), timeout=5.0)
            except asyncio.TimeoutError:
                yield "event: heartbeat\ndata: {}\n\n"
                continue
            if event is None:
                break
            yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")


# ----- Utilities --------------------------------------------------------------


async def _simulate_stream() -> None:
    """Replay synthetic events through the predictor at a high speed multiplier."""
    from src.data.stream_simulator import StreamSimulator
    from src.data.synthetic import SyntheticLOBGenerator

    if state.predictor is None or state.stream_queue is None:
        return
    gen = SyntheticLOBGenerator(n_events=2_000, seed=int(time.time()) % 10000)
    ob, msg = gen.generate()
    merged = ob.copy()
    for col in ("event_type", "order_id", "event_size", "event_price", "direction"):
        merged[col] = msg[col].values
    sim = StreamSimulator(merged, speed_multiplier=500.0)

    buffer: list[dict] = []
    for event in sim.stream():
        n_levels = (len(event["orderbook"]) // 4) if event["orderbook"].size else 10
        row = {"time": event["timestamp"]}
        for i in range(n_levels):
            base = i * 4
            row[f"ask_price_{i + 1}"] = event["orderbook"][base]
            row[f"ask_size_{i + 1}"] = event["orderbook"][base + 1]
            row[f"bid_price_{i + 1}"] = event["orderbook"][base + 2]
            row[f"bid_size_{i + 1}"] = event["orderbook"][base + 3]
        row.update(event["message"])
        buffer.append(row)
        if len(buffer) >= state.predictor.seq_len + 200:
            try:
                result = state.predictor.predict(buffer[-state.predictor.seq_len - 150 :])
                result["timestamp"] = event["timestamp"]
                await state.stream_queue.put(result)
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("stream prediction failed: %s", exc)
            buffer = buffer[-state.predictor.seq_len - 200 :]
        await asyncio.sleep(0)
    await state.stream_queue.put(None)


def _rows_to_events(
    rows: list[list[float]],
    n_levels: int,
    message_tape: list[dict] | None,
) -> list[dict]:
    """LOBSTER-style flat rows -> per-event dicts the FeatureEngineer can ingest."""
    events: list[dict] = []
    for i, row in enumerate(rows):
        event = {}
        for level in range(n_levels):
            base = level * 4
            event[f"ask_price_{level + 1}"] = row[base]
            event[f"ask_size_{level + 1}"] = row[base + 1]
            event[f"bid_price_{level + 1}"] = row[base + 2]
            event[f"bid_size_{level + 1}"] = row[base + 3]
        if message_tape and i < len(message_tape):
            event.update(message_tape[i])
        events.append(event)
    return events


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run(
        "src.serving.api:app",
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", "8080")),
    )
