"""Signals. A signal is a (dates x tickers) frame of cross-sectional scores;
only the relative ordering within each row matters to the harness.
"""

from __future__ import annotations

import pandas as pd


def momentum(prices: pd.DataFrame, lookback: int = 252, skip: int = 21) -> pd.DataFrame:
    """Classic 12-1 momentum: trailing return over `lookback` days, skipping the
    most recent `skip` days to avoid short-term reversal.

    score[t] = close[t-skip] / close[t-lookback] - 1
    """
    return prices.shift(skip) / prices.shift(lookback) - 1.0
