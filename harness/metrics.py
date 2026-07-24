"""Signal-quality metrics: forward returns, rank IC, quantile spreads.

All functions are cross-sectional and vectorized over a (dates x tickers) frame.
Spearman rank IC is computed as a Pearson correlation of cross-sectional ranks,
so the only dependency is pandas/numpy.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def forward_returns(prices: pd.DataFrame, horizon: int = 1, gap: int = 1) -> pd.DataFrame:
    """Return earned by a position opened `gap` days after the signal date and
    held `horizon` days.

    A signal observed at the close of day t cannot be traded until later, so
    `fwd[t] = close[t+gap+horizon] / close[t+gap] - 1`. `gap=1` models next-day
    execution and guarantees no same-bar look-ahead. Rows near the end are NaN.
    """
    return prices.shift(-(gap + horizon)) / prices.shift(-gap) - 1.0


def _cross_sectional_ranks(frame: pd.DataFrame) -> pd.DataFrame:
    """Per-day ranks in [0, 1]; NaNs preserved."""
    return frame.rank(axis=1, pct=True)


def rank_ic(signal: pd.DataFrame, fwd_ret: pd.DataFrame) -> pd.Series:
    """Daily Spearman rank IC between signal and forward return.

    Correlates cross-sectional ranks each day. Days with fewer than 2 valid
    pairs yield NaN.
    """
    signal, fwd_ret = signal.align(fwd_ret, join="inner")
    sig_r = _cross_sectional_ranks(signal.where(fwd_ret.notna()))
    ret_r = _cross_sectional_ranks(fwd_ret.where(signal.notna()))

    sig_d = sig_r.sub(sig_r.mean(axis=1), axis=0)
    ret_d = ret_r.sub(ret_r.mean(axis=1), axis=0)

    cov = (sig_d * ret_d).sum(axis=1)
    denom = np.sqrt((sig_d**2).sum(axis=1) * (ret_d**2).sum(axis=1))
    ic = cov / denom.replace(0.0, np.nan)
    valid = signal.notna() & fwd_ret.notna()
    return ic.where(valid.sum(axis=1) >= 2)


def ic_summary(ic: pd.Series) -> dict:
    """Mean IC, volatility, annualized IR, hit rate, and t-stat."""
    ic = ic.dropna()
    mean, std, n = ic.mean(), ic.std(), len(ic)
    ir = (mean / std) * np.sqrt(TRADING_DAYS) if std > 0 else np.nan
    tstat = (mean / std) * np.sqrt(n) if std > 0 and n > 0 else np.nan
    return {
        "mean_ic": mean,
        "ic_std": std,
        "ir_annualized": ir,
        "hit_rate": (ic > 0).mean(),
        "t_stat": tstat,
        "n_days": n,
    }


def quantile_returns(
    signal: pd.DataFrame, fwd_ret: pd.DataFrame, q: int = 10
) -> pd.Series:
    """Mean forward return per signal quantile, averaged over time.

    Index is the quantile label 0..q-1 (0 = lowest signal). A monotone profile
    is a stronger sign of a real effect than the top-minus-bottom spread alone.
    """
    signal, fwd_ret = signal.align(fwd_ret, join="inner")
    buckets = signal.apply(
        lambda row: pd.qcut(row, q, labels=False, duplicates="drop"), axis=1
    )
    out = {}
    for label in range(q):
        mask = buckets == label
        out[label] = fwd_ret.where(mask).mean(axis=1).mean()
    return pd.Series(out, name="mean_fwd_return")
