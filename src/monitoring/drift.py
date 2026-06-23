"""Feature drift detection using PSI — see Section 7.1."""

from __future__ import annotations

import logging
from collections import Counter, deque

import numpy as np

logger = logging.getLogger(__name__)


def population_stability_index(
    reference: np.ndarray,
    current: np.ndarray,
    n_bins: int = 10,
    eps: float = 1e-6,
) -> float:
    """PSI for a single feature column."""
    ref = np.asarray(reference, dtype=np.float64)
    cur = np.asarray(current, dtype=np.float64)
    ref = ref[np.isfinite(ref)]
    cur = cur[np.isfinite(cur)]
    if ref.size == 0 or cur.size == 0:
        return 0.0

    # Use quantile bins from the reference so PSI is well-defined for skewed data
    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.unique(np.quantile(ref, quantiles))
    if edges.size < 3:  # all reference points collapsed
        return 0.0
    edges[0] = -np.inf
    edges[-1] = np.inf

    ref_hist, _ = np.histogram(ref, bins=edges)
    cur_hist, _ = np.histogram(cur, bins=edges)

    ref_frac = ref_hist / ref_hist.sum()
    cur_frac = cur_hist / cur_hist.sum()
    ref_frac = np.where(ref_frac == 0, eps, ref_frac)
    cur_frac = np.where(cur_frac == 0, eps, cur_frac)
    return float(np.sum((cur_frac - ref_frac) * np.log(cur_frac / ref_frac)))


class DriftMonitor:
    """Computes PSI per feature column against a reference (training) sample."""

    def __init__(
        self,
        reference_features: np.ndarray,
        feature_names: list[str] | None = None,
        n_bins: int = 10,
        alert_threshold: float = 0.2,
    ) -> None:
        if reference_features.ndim != 2:
            raise ValueError("reference_features must be 2D (n_rows, n_features)")
        self.reference = reference_features.astype(np.float64)
        self.feature_names = feature_names or [
            f"feature_{i}" for i in range(reference_features.shape[1])
        ]
        self.n_bins = n_bins
        self.alert_threshold = alert_threshold

    def compute_psi(self, current_features: np.ndarray) -> dict[str, float]:
        if current_features.ndim != 2:
            raise ValueError("current_features must be 2D")
        if current_features.shape[1] != self.reference.shape[1]:
            raise ValueError(
                f"feature dimension mismatch: ref={self.reference.shape[1]} "
                f"cur={current_features.shape[1]}"
            )
        psi = {}
        for i, name in enumerate(self.feature_names):
            psi[name] = population_stability_index(
                self.reference[:, i], current_features[:, i], n_bins=self.n_bins
            )
        return psi

    def check_drift(self, current_features: np.ndarray) -> dict:
        psi = self.compute_psi(current_features)
        drifted = [name for name, score in psi.items() if score > self.alert_threshold]
        alert = len(drifted) > 0
        return {"drifted_features": drifted, "psi_scores": psi, "alert": alert}

    def update_reference(self, new_reference: np.ndarray) -> None:
        if new_reference.shape[1] != self.reference.shape[1]:
            raise ValueError("feature dim mismatch")
        self.reference = new_reference.astype(np.float64)


class PredictionDriftMonitor:
    """Alerts if prediction class distribution becomes too concentrated."""

    def __init__(
        self,
        window_size: int = 1000,
        n_classes: int = 3,
        alert_concentration: float = 0.8,
    ) -> None:
        if window_size < 1:
            raise ValueError("window_size must be >= 1")
        self.window_size = window_size
        self.n_classes = n_classes
        self.alert_concentration = alert_concentration
        self._buf: deque[int] = deque(maxlen=window_size)

    def update(self, prediction: int) -> None:
        self._buf.append(int(prediction))

    def get_distribution(self) -> dict[str, float]:
        if not self._buf:
            return {f"class_{c}": 0.0 for c in range(self.n_classes)}
        counts = Counter(self._buf)
        total = len(self._buf)
        return {f"class_{c}": counts.get(c, 0) / total for c in range(self.n_classes)}

    def check_alert(self) -> bool:
        dist = self.get_distribution()
        return max(dist.values(), default=0.0) >= self.alert_concentration
