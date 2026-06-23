"""Backtest performance metrics — see Section 5.3."""

from __future__ import annotations

from typing import Iterable

import numpy as np


# At ~1s per event, periods_per_year for crypto 24x7 trading would be much larger;
# for equities 6.5h * 3600s * 252 days ≈ 5.9M periods/yr.
DEFAULT_PERIODS_PER_YEAR = 252 * int(6.5 * 3600)


def sharpe_ratio(returns: np.ndarray, periods_per_year: int = DEFAULT_PERIODS_PER_YEAR) -> float:
    r = np.asarray(returns, dtype=np.float64)
    if r.size == 0:
        return 0.0
    sigma = r.std(ddof=1) if r.size > 1 else 0.0
    if sigma <= 1e-12:
        return 0.0
    return float(r.mean() / sigma * np.sqrt(periods_per_year))


def sortino_ratio(returns: np.ndarray, periods_per_year: int = DEFAULT_PERIODS_PER_YEAR) -> float:
    r = np.asarray(returns, dtype=np.float64)
    if r.size == 0:
        return 0.0
    downside = r[r < 0]
    if downside.size == 0:
        return 0.0
    sigma_down = downside.std(ddof=1) if downside.size > 1 else 0.0
    if sigma_down <= 1e-12:
        return 0.0
    return float(r.mean() / sigma_down * np.sqrt(periods_per_year))


def max_drawdown(cumulative_returns: np.ndarray) -> float:
    c = np.asarray(cumulative_returns, dtype=np.float64)
    if c.size == 0:
        return 0.0
    running_max = np.maximum.accumulate(c)
    drawdowns = (running_max - c) / np.where(running_max != 0, np.abs(running_max), 1.0)
    return float(drawdowns.max())


def calmar_ratio(returns: np.ndarray, periods_per_year: int = DEFAULT_PERIODS_PER_YEAR) -> float:
    r = np.asarray(returns, dtype=np.float64)
    if r.size == 0:
        return 0.0
    cum = np.cumprod(1.0 + r) - 1.0
    mdd = max_drawdown(cum + 1.0)
    if mdd <= 1e-12:
        return 0.0
    # use log return to avoid overflow on extreme period counts
    mean_log = float(np.log1p(r).mean())
    log_annualized = mean_log * periods_per_year
    if log_annualized > 700:
        return float("inf")
    annualized = float(np.expm1(log_annualized))
    return float(annualized / mdd)


def hit_rate(signals: np.ndarray, actual_directions: np.ndarray) -> float:
    s = np.asarray(signals)
    a = np.asarray(actual_directions)
    nonzero = s != 0
    if not nonzero.any():
        return 0.0
    correct = np.sign(s[nonzero]) == np.sign(a[nonzero])
    return float(correct.mean())


def profit_factor(gross_profits: float, gross_losses: float) -> float:
    if gross_losses == 0:
        return float("inf") if gross_profits > 0 else 0.0
    return float(gross_profits / abs(gross_losses))


def alpha_beta(
    strategy_returns: np.ndarray,
    benchmark_returns: np.ndarray,
    periods_per_year: int = DEFAULT_PERIODS_PER_YEAR,
) -> tuple[float, float]:
    s = np.asarray(strategy_returns, dtype=np.float64)
    b = np.asarray(benchmark_returns, dtype=np.float64)
    n = min(s.size, b.size)
    if n < 2:
        return 0.0, 0.0
    s, b = s[:n], b[:n]
    cov = np.cov(s, b, ddof=1)
    beta = cov[0, 1] / cov[1, 1] if cov[1, 1] > 1e-12 else 0.0
    alpha_per_period = s.mean() - beta * b.mean()
    alpha_ann = alpha_per_period * periods_per_year
    return float(alpha_ann), float(beta)


def full_report(
    returns: np.ndarray,
    signals: np.ndarray,
    actual_directions: np.ndarray,
    benchmark_returns: np.ndarray,
    periods_per_year: int = DEFAULT_PERIODS_PER_YEAR,
) -> dict:
    r = np.asarray(returns, dtype=np.float64)
    cum = np.cumsum(r)
    gross_profits = float(r[r > 0].sum())
    gross_losses = float(r[r < 0].sum())
    alpha, beta = alpha_beta(r, benchmark_returns, periods_per_year)
    return {
        "sharpe": sharpe_ratio(r, periods_per_year),
        "sortino": sortino_ratio(r, periods_per_year),
        "max_drawdown": max_drawdown(np.cumprod(1.0 + r)),
        "calmar": calmar_ratio(r, periods_per_year),
        "hit_rate": hit_rate(signals, actual_directions),
        "profit_factor": profit_factor(gross_profits, gross_losses),
        "alpha": alpha,
        "beta": beta,
        "total_return": float(cum[-1]) if cum.size else 0.0,
        "n_trades": int(np.sum(np.diff(np.concatenate([[0], signals])) != 0)),
        "gross_profits": gross_profits,
        "gross_losses": gross_losses,
    }
