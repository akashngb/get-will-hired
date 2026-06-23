"""Tests for drift monitors."""

from __future__ import annotations

import numpy as np
import pytest

from src.monitoring.drift import (
    DriftMonitor,
    PredictionDriftMonitor,
    population_stability_index,
)


def test_psi_same_distribution_is_small():
    rng = np.random.default_rng(0)
    ref = rng.standard_normal(2000)
    cur = rng.standard_normal(2000)
    assert population_stability_index(ref, cur) < 0.1


def test_psi_shifted_distribution_is_large():
    rng = np.random.default_rng(0)
    ref = rng.standard_normal(2000)
    cur = rng.standard_normal(2000) + 1.5  # mean shift
    psi = population_stability_index(ref, cur)
    assert psi > 0.2


def test_drift_monitor_alert_threshold():
    rng = np.random.default_rng(1)
    ref = rng.standard_normal((1000, 4))
    cur = rng.standard_normal((1000, 4))
    cur[:, 2] += 2.0  # shift one feature
    monitor = DriftMonitor(ref, feature_names=[f"f{i}" for i in range(4)], alert_threshold=0.2)
    result = monitor.check_drift(cur)
    assert result["alert"] is True
    assert "f2" in result["drifted_features"]
    # other features should not all be drifted
    assert len(result["drifted_features"]) < 4


def test_prediction_drift_monitor():
    monitor = PredictionDriftMonitor(window_size=10, alert_concentration=0.7)
    for _ in range(10):
        monitor.update(2)
    assert monitor.check_alert() is True
    dist = monitor.get_distribution()
    assert dist["class_2"] == 1.0
