"""Tests for the TCN model — see Section 9.1."""

from __future__ import annotations

import torch
import pytest

from src.models.tcn import CausalConv1d, TCNBlock, TCNModel


def test_causality_module_level():
    """A single CausalConv1d output at t must not depend on inputs at >t."""
    torch.manual_seed(0)
    conv = CausalConv1d(in_channels=4, out_channels=8, kernel_size=3, dilation=2).eval()
    x = torch.randn(2, 4, 20)
    y1 = conv(x)
    x2 = x.clone()
    x2[:, :, 10:] = torch.randn(2, 4, 10)
    y2 = conv(x2)
    torch.testing.assert_close(y1[:, :, :10], y2[:, :, :10])


def test_causality_full_model():
    """Future input perturbations must not affect past outputs (single block in eval)."""
    torch.manual_seed(0)
    model = TCNModel(
        n_features=8, n_levels=3, n_channels=16, kernel_size=3, dropout=0.0
    ).eval()
    # Use a single block's representation (pre-pool) for the causality check
    x = torch.randn(2, 50, 8)
    # Forward up to pre-pool: replicate model's transposition + blocks
    z1 = x.transpose(1, 2)
    z2 = z1.clone()
    z2[:, :, 30:] = torch.randn(2, 8, 20)
    for block in model.blocks:
        z1 = block(z1)
        z2 = block(z2)
    # Use a buffer of 5 timesteps just before the perturbation for safety
    torch.testing.assert_close(z1[:, :, :25], z2[:, :, :25])


def test_output_shapes():
    model = TCNModel(n_features=12, n_levels=3, n_channels=32, horizons=(10, 50, 100))
    x = torch.randn(4, 64, 12)
    out = model(x)
    assert set(out.keys()) == {"horizon_10", "horizon_50", "horizon_100"}
    for v in out.values():
        assert v.shape == (4, 3)


def test_receptive_field_matches_formula():
    model = TCNModel(n_features=8, n_levels=4, n_channels=16, kernel_size=3)
    # 1 + 2*(k-1)*(2^L - 1) = 1 + 2*2*(15) = 61
    assert model.receptive_field() == 61


def test_gradient_flow():
    model = TCNModel(n_features=8, n_levels=3, n_channels=16, dropout=0.0)
    x = torch.randn(8, 32, 8, requires_grad=False)
    out = model(x)
    target = torch.zeros(8, dtype=torch.long)
    loss = sum(torch.nn.functional.cross_entropy(v, target) for v in out.values())
    loss.backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, f"{name} has no grad"
        assert torch.any(p.grad != 0), f"{name} grad is all zero"


def test_determinism():
    torch.manual_seed(1234)
    m1 = TCNModel(n_features=6, n_levels=3, n_channels=16, dropout=0.0).eval()
    torch.manual_seed(1234)
    m2 = TCNModel(n_features=6, n_levels=3, n_channels=16, dropout=0.0).eval()
    x = torch.randn(2, 20, 6)
    for k in m1.forward(x):
        torch.testing.assert_close(m1.forward(x)[k], m2.forward(x)[k])


def test_multi_horizon_independence():
    """Gradient of one head's loss should not flow through the other heads' linear layers."""
    model = TCNModel(n_features=6, n_levels=2, n_channels=8, horizons=(10, 50), dropout=0.0)
    x = torch.randn(2, 16, 6)
    out = model(x)
    target = torch.zeros(2, dtype=torch.long)
    loss10 = torch.nn.functional.cross_entropy(out["horizon_10"], target)
    loss10.backward()
    # head_10 params should have grads; head_50 params should be None / zero
    h10_params = list(model.heads["horizon_10"].parameters())
    h50_params = list(model.heads["horizon_50"].parameters())
    assert all(p.grad is not None and torch.any(p.grad != 0) for p in h10_params)
    assert all(p.grad is None or torch.all(p.grad == 0) for p in h50_params)
