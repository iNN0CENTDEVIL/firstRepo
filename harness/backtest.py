"""Vectorized cross-sectional long/short backtest with realistic costs.

Dollar-neutral decile portfolio: each day, long the top signal quantile and
short the bottom, each side normalized to unit gross. Positions from the signal
at day t earn `forward_returns(gap=...)` so there is no look-ahead. Costs are
charged on traded notional (change in weights) each rebalance.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .metrics import TRADING_DAYS, forward_returns


def long_short_weights(signal: pd.DataFrame, quantile: int = 10) -> pd.DataFrame:
    """Dollar-neutral weights: +1 gross long (top quantile), -1 gross short
    (bottom quantile), equal-weighted within each side. Net exposure ~ 0.
    """
    ranks = signal.rank(axis=1)
    n = signal.notna().sum(axis=1)
    edge = n / quantile

    is_long = ranks.gt(n - edge, axis=0)
    is_short = ranks.le(edge, axis=0)

    long_w = is_long.astype(float)
    short_w = is_short.astype(float)
    long_w = long_w.div(long_w.sum(axis=1).replace(0, np.nan), axis=0)
    short_w = short_w.div(short_w.sum(axis=1).replace(0, np.nan), axis=0)
    return (long_w - short_w).fillna(0.0)


def max_drawdown(returns: pd.Series) -> float:
    """Worst peak-to-trough decline of the cumulative return curve."""
    curve = (1.0 + returns).cumprod()
    return (curve / curve.cummax() - 1.0).min()


def backtest(
    signal: pd.DataFrame,
    prices: pd.DataFrame,
    quantile: int = 10,
    cost_bps: float = 5.0,
    gap: int = 1,
) -> dict:
    """Run the L/S backtest and return the net return series plus summary stats.

    `cost_bps` is a one-way cost (commission + half-spread + impact proxy)
    charged on the gross notional traded at each rebalance.
    """
    weights = long_short_weights(signal, quantile)
    fwd = forward_returns(prices, horizon=1, gap=gap)
    weights, fwd = weights.align(fwd, join="inner")

    gross = (weights * fwd).sum(axis=1)
    traded = (weights - weights.shift(1)).abs().sum(axis=1)
    cost = traded * (cost_bps / 1e4)
    net = (gross - cost).dropna()

    ann_ret = net.mean() * TRADING_DAYS
    ann_vol = net.std() * np.sqrt(TRADING_DAYS)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan

    return {
        "returns": net,
        "sharpe": sharpe,
        "ann_return": ann_ret,
        "ann_vol": ann_vol,
        "max_drawdown": max_drawdown(net),
        "avg_daily_turnover": traded.mean(),
        "n_days": len(net),
    }
