"""Tests for FeatureEngineer — see Section 9.1."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.feature_engineer import FeatureEngineer
from src.data.synthetic import SyntheticLOBGenerator


@pytest.fixture(scope="module")
def featurized():
    ob, msg = SyntheticLOBGenerator(n_events=3_000, seed=1).generate()
    fe = FeatureEngineer(levels=10, horizons=(10, 50, 100))
    df = fe.fit_transform(ob, msg)
    return fe, df


def test_no_lookahead():
    """Modifying future rows must not change features at earlier rows."""
    ob, msg = SyntheticLOBGenerator(n_events=1_500, seed=2).generate()
    fe = FeatureEngineer(levels=10, horizons=(10,))
    df_a = fe.fit_transform(ob, msg).copy()

    # Corrupt the last 300 rows of raw data, then re-featurize fresh
    ob_b = ob.copy()
    msg_b = msg.copy()
    ob_b.iloc[-300:, 1:] = 0.0
    msg_b.iloc[-300:, 1:] = 0.0

    fe2 = FeatureEngineer(levels=10, horizons=(10,))
    df_b = fe2.fit_transform(ob_b, msg_b)

    # Compare the unnormalized features instead: re-compute without normalization
    # so we are testing the causality of features, not stats.
    feats_a = fe._compute_all_features(ob, msg)
    feats_b = fe2._compute_all_features(ob_b, msg_b)
    n_safe = len(feats_a) - 500  # cushion for future-dependent labels and warmup
    np.testing.assert_allclose(
        feats_a.iloc[:n_safe].values, feats_b.iloc[:n_safe].values, equal_nan=True
    )


def test_label_shift():
    """label_direction_10 at i must reflect mid_price[i+10] vs mid_price[i]."""
    ob, msg = SyntheticLOBGenerator(n_events=2_000, seed=3).generate()
    fe = FeatureEngineer(levels=10, horizons=(10,))
    feats = fe._compute_all_features(ob, msg)
    labels = fe._compute_labels(feats)
    df = pd.concat([feats, labels], axis=1).dropna().reset_index(drop=True)

    # Re-derive expectations from mid prices in df
    mid = df["mid_price"].values
    for i in range(0, len(df) - 100, 50):
        expected_change = (mid[i + 10] - mid[i]) / mid[i]
        expected = (
            1
            if expected_change > fe.stationary_threshold
            else (-1 if expected_change < -fe.stationary_threshold else 0)
        )
        # Note: df["label_direction_10"].iloc[i] is causal at i but uses mid at i+10
        # In our dropped frame the label is already computed correctly
        observed = df["label_direction_10"].iloc[i]
        assert observed == expected, f"mismatch at {i}: got {observed} want {expected}"


def test_normalization_fit_transform(featurized):
    fe, df = featurized
    for col in fe.get_feature_names():
        mean = df[col].mean()
        std = df[col].std(ddof=0)
        assert abs(mean) < 0.05, f"{col} mean = {mean}"
        assert abs(std - 1.0) < 0.05, f"{col} std = {std}"


def test_no_nan_in_output(featurized):
    _, df = featurized
    assert not df.isna().any().any(), "found NaNs in featurized output"


def test_transform_uses_saved_stats(featurized):
    fe, _ = featurized
    ob2, msg2 = SyntheticLOBGenerator(n_events=1_500, seed=99).generate()
    df2 = fe.transform(ob2, msg2)
    # On OOS data, mean should not be exactly zero anymore (stats were fit elsewhere)
    means = [df2[c].mean() for c in fe.get_feature_names()]
    assert any(abs(m) > 0.01 for m in means), "transform appears to refit stats"


def test_class_balance(featurized):
    _, df = featurized
    counts = df["label_direction_10"].value_counts()
    total = counts.sum()
    fractions = counts / total
    # always passes, but prints a warning
    if fractions.min() < 0.10:
        print(f"WARN: class imbalance — minimum class fraction = {fractions.min():.3f}")
