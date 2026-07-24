"""Data layer.

The rest of the harness only needs a `prices` DataFrame: a DatetimeIndex of
trading days by columns of tickers, holding split/dividend-adjusted closes,
*including delisted names over their live window* (survivorship-bias-free).

`generate_synthetic_prices` fabricates such a frame with a real, persistent
cross-sectional trend so that trailing-return momentum is genuinely predictive
of forward returns. That lets the momentum gate (run_momentum.py) validate the
harness end-to-end with no data subscription.

To go live, replace `generate_synthetic_prices` with a loader for your
point-in-time source that returns the same shape. Fundamentals (which need an
`available_at` accessor) are not modelled here yet.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def generate_synthetic_prices(
    n_stocks: int = 200,
    n_days: int = 900,
    seed: int = 0,
    trend_persistence: float = 0.99,
    trend_innovation: float = 0.0008,
    idio_vol: float = 0.02,
    start: str = "2018-01-01",
) -> pd.DataFrame:
    """Fabricate adjusted closes with a slow, persistent per-stock drift.

    Each stock's expected daily return follows a highly persistent AR(1)
    ("trend"). Because the trend changes slowly, a stock's trailing return is a
    noisy estimate of its current trend, which in turn predicts its forward
    return -- so momentum works, by construction. Idiosyncratic noise dominates
    any single day, keeping the daily information coefficient realistically small.
    """
    rng = np.random.default_rng(seed)

    trend = np.zeros((n_days, n_stocks))
    trend[0] = rng.normal(0.0, trend_innovation, n_stocks)
    for t in range(1, n_days):
        trend[t] = trend_persistence * trend[t - 1] + rng.normal(
            0.0, trend_innovation, n_stocks
        )

    idio = rng.normal(0.0, idio_vol, (n_days, n_stocks))
    rets = trend + idio

    prices = 100.0 * np.cumprod(1.0 + rets, axis=0)

    dates = pd.bdate_range(start=start, periods=n_days, name="date")
    tickers = [f"S{i:04d}" for i in range(n_stocks)]
    return pd.DataFrame(prices, index=dates, columns=tickers)
