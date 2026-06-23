"""Tests for StreamSimulator and StreamBuffer."""

from __future__ import annotations

import time

import numpy as np
import pytest

from src.data.stream_simulator import StreamBuffer, StreamSimulator
from src.data.synthetic import SyntheticLOBGenerator


@pytest.fixture(scope="module")
def event_frame():
    ob, msg = SyntheticLOBGenerator(n_events=500, seed=4).generate()
    df = ob.copy()
    for col in ("event_type", "order_id", "event_size", "event_price", "direction"):
        df[col] = msg[col].values
    return df


def test_stream_yields_all_events(event_frame):
    sim = StreamSimulator(event_frame, speed_multiplier=0.0)
    out = list(sim.stream())
    assert len(out) == len(event_frame)
    assert out[0]["sequence_num"] == 0
    assert out[-1]["sequence_num"] == len(event_frame) - 1
    assert "orderbook" in out[0] and "message" in out[0]


def test_stream_buffer_window():
    buf = StreamBuffer(maxlen=10)
    for i in range(9):
        buf.push(np.arange(5, dtype=np.float32) + i)
        assert not buf.is_ready()
    buf.push(np.arange(5, dtype=np.float32) + 9)
    assert buf.is_ready()
    win = buf.get_feature_window()
    assert win.shape == (10, 5)
    # newest sample (offset +9) should sit at the end
    assert win[-1, 0] == 9.0


def test_stream_buffer_clear_and_overflow():
    buf = StreamBuffer(maxlen=3)
    for i in range(5):
        buf.push(np.full(2, i, dtype=np.float32))
    assert len(buf) == 3
    win = buf.get_feature_window()
    # oldest two should have been evicted
    assert win[0, 0] == 2.0
    assert win[-1, 0] == 4.0
    buf.clear()
    assert len(buf) == 0


def test_stream_buffer_rejects_wrong_shape():
    buf = StreamBuffer(maxlen=4, n_features=3)
    buf.push(np.zeros(3, dtype=np.float32))
    with pytest.raises(ValueError):
        buf.push(np.zeros(4, dtype=np.float32))


def test_speed_multiplier_runs_fast(event_frame):
    """speed_multiplier=0 should be near-instant even for many events."""
    sim = StreamSimulator(event_frame.iloc[:200].copy(), speed_multiplier=0.0)
    start = time.perf_counter()
    count = sum(1 for _ in sim.stream())
    elapsed = time.perf_counter() - start
    assert count == 200
    assert elapsed < 1.0, f"streaming was too slow: {elapsed:.3f}s"
