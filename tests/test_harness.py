"""Harness correctness tests -- these guard against the harness silently lying.

Run: `pytest`
"""

import numpy as np
import pandas as pd

from harness.backtest import backtest, long_short_weights
from harness.data import generate_synthetic_prices
from harness.metrics import forward_returns, rank_ic
from harness.signals import momentum


def _frame(values):
    dates = pd.bdate_range("2020-01-01", periods=len(values), name="date")
    return pd.DataFrame(values, index=dates, columns=list("ABCDE"))


def test_rank_ic_perfect_positive():
    # signal identical ordering to forward return -> IC == 1 every day.
    sig = _frame([[1, 2, 3, 4, 5], [5, 4, 3, 2, 1]])
    fwd = sig.copy()
    ic = rank_ic(sig, fwd).dropna()
    assert np.allclose(ic.values, 1.0)


def test_rank_ic_perfect_negative():
    sig = _frame([[1, 2, 3, 4, 5]])
    fwd = _frame([[5, 4, 3, 2, 1]])
    ic = rank_ic(sig, fwd).dropna()
    assert np.allclose(ic.values, -1.0)


def test_forward_returns_no_lookahead():
    # gap=1: fwd[t] must depend only on prices strictly after t.
    prices = _frame(np.arange(1, 21).reshape(4, 5) * 1.0)
    fwd = forward_returns(prices, horizon=1, gap=1)
    # fwd[0] uses prices[1] and prices[2]; independent of prices[0].
    expected = prices.iloc[2] / prices.iloc[1] - 1
    assert np.allclose(fwd.iloc[0].values, expected.values)
    # last two rows cannot look ahead -> NaN.
    assert fwd.iloc[-1].isna().all()
    assert fwd.iloc[-2].isna().all()


def test_long_short_weights_dollar_neutral():
    prices = generate_synthetic_prices(n_stocks=100, n_days=400, seed=1)
    w = long_short_weights(momentum(prices), quantile=10).dropna(how="all")
    # net exposure ~ 0 and gross ~ 2 on fully-populated rows.
    populated = w[(w != 0).sum(axis=1) > 0]
    assert np.allclose(populated.sum(axis=1).values, 0.0, atol=1e-9)
    assert np.allclose(populated.abs().sum(axis=1).values, 2.0, atol=1e-9)


def test_momentum_gate_positive_ic():
    # the synthetic world has real momentum -> harness must detect it.
    prices = generate_synthetic_prices(n_stocks=200, n_days=900, seed=0)
    sig = momentum(prices)
    fwd = forward_returns(prices, horizon=1, gap=1)
    assert rank_ic(sig, fwd).mean() > 0
    assert backtest(sig, prices)["sharpe"] > 0
