"""Investment simulation from Kronos (arXiv:2508.02739): a long-only top-k
portfolio scored by Annualized Excess Return (AER) and Information Ratio (IR)
against an equal-weight-universe benchmark.

Complements the dollar-neutral L/S backtest in backtest.py with the long-only,
benchmark-relative view the paper uses to judge real-world usefulness.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .metrics import TRADING_DAYS


def top_k_long_only(
    signal: pd.DataFrame,
    fwd_ret: pd.DataFrame,
    k: int | None = None,
    top_quantile: float = 0.1,
) -> dict:
    """Hold the top names by signal each day, equal-weighted and long-only.

    Selection is the top `k` names, or the top `top_quantile` fraction if `k` is
    None. The benchmark is the equal-weight return of every name with a signal
    that day, so excess return isolates the signal's stock-selection value.

    Returns the portfolio/benchmark/excess return series plus:
      - aer: annualized excess return (mean daily excess * 252)
      - ir : information ratio (annualized excess / annualized tracking error)
      - hit_rate: share of days the portfolio beat the benchmark
    """
    signal, fwd_ret = signal.align(fwd_ret, join="inner")
    ranks = signal.rank(axis=1, ascending=False)

    if k is not None:
        selected = ranks.le(k)
    else:
        n = signal.notna().sum(axis=1)
        selected = ranks.le((n * top_quantile).clip(lower=1), axis=0)

    portfolio = fwd_ret.where(selected).mean(axis=1)
    benchmark = fwd_ret.where(signal.notna()).mean(axis=1)
    excess = (portfolio - benchmark).dropna()

    aer = excess.mean() * TRADING_DAYS
    te = excess.std() * np.sqrt(TRADING_DAYS)
    ir = aer / te if te > 0 else np.nan

    return {
        "portfolio": portfolio,
        "benchmark": benchmark,
        "excess": excess,
        "aer": aer,
        "ir": ir,
        "hit_rate": (excess > 0).mean(),
        "n_days": len(excess),
    }
