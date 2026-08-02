"""Anti-overfitting layer: purged/embargoed cross-validation and the
probabilistic / deflated Sharpe ratio (Bailey & Lopez de Prado).

These are the defense from STRATEGY.md section 5.4 -- because agents let you test
thousands of hypotheses, a raw backtest Sharpe is nearly meaningless until it is
discounted for the number of trials that produced it.

No third-party stats dependency: `statistics.NormalDist` (stdlib) supplies the
normal CDF and its inverse.
"""

from __future__ import annotations

from statistics import NormalDist

import numpy as np
import pandas as pd

from .metrics import TRADING_DAYS, rank_ic

_NORM = NormalDist()
_EULER_MASCHERONI = 0.5772156649015329


# --------------------------------------------------------------------------- #
# Purged, embargoed K-fold cross-validation
# --------------------------------------------------------------------------- #
class PurgedKFold:
    """K contiguous test folds over an ordered date axis, with purging and an
    embargo to stop label leakage.

    When a label spans `horizon` days forward, a training observation whose
    label window overlaps the test block leaks future information; those are
    *purged*. Serial correlation can leak just past the test block too, so an
    `embargo` fraction of observations immediately after each test block is also
    dropped. Yields (train_idx, test_idx) integer arrays.

    For a parameter-free signal (e.g. plain momentum) the train fold is unused;
    the value is the leakage-safe test partition. When you fit or select a model,
    fit on train_idx and score on test_idx.
    """

    def __init__(self, n_splits: int = 5, horizon: int = 1, embargo: float = 0.0):
        if n_splits < 2:
            raise ValueError("n_splits must be >= 2")
        self.n_splits = n_splits
        self.horizon = horizon
        self.embargo = embargo

    def split(self, dates):
        n = len(dates)
        indices = np.arange(n)
        embargo_n = int(n * self.embargo)
        for test_idx in np.array_split(indices, self.n_splits):
            t0, t1 = int(test_idx[0]), int(test_idx[-1])
            keep = np.ones(n, dtype=bool)
            # purge the test block itself plus any prior obs whose label reaches it
            keep[max(0, t0 - self.horizon) : t1 + 1] = False
            # embargo the observations right after the test block
            keep[t1 + 1 : min(n, t1 + 1 + embargo_n)] = False
            yield indices[keep], test_idx


def cross_validated_ic(
    signal: pd.DataFrame, fwd_ret: pd.DataFrame, cv: PurgedKFold
) -> pd.Series:
    """Out-of-sample mean rank IC on each test fold."""
    signal, fwd_ret = signal.align(fwd_ret, join="inner")
    dates = signal.index
    scores = []
    for _, test_idx in cv.split(dates):
        test_dates = dates[test_idx]
        scores.append(rank_ic(signal.loc[test_dates], fwd_ret.loc[test_dates]).mean())
    return pd.Series(scores, name="oos_mean_ic")


# --------------------------------------------------------------------------- #
# Probabilistic & deflated Sharpe ratio
# --------------------------------------------------------------------------- #
def _standardized_moment(x: np.ndarray, power: int) -> float:
    m = x.mean()
    s = x.std(ddof=0)
    return float((((x - m) / s) ** power).mean())


def probabilistic_sharpe_ratio(
    returns: pd.Series,
    sr_benchmark: float = 0.0,
    periods_per_year: int = TRADING_DAYS,
) -> float:
    """P(true annualized Sharpe > `sr_benchmark`), correcting for track-record
    length and the skew/kurtosis of returns (fat tails make a high Sharpe less
    trustworthy). `sr_benchmark` is annualized.
    """
    r = np.asarray(returns.dropna(), dtype=float)
    n = len(r)
    if n < 2 or r.std(ddof=1) == 0:
        return np.nan

    sr_hat = r.mean() / r.std(ddof=1)                 # per-observation
    sr_star = sr_benchmark / np.sqrt(periods_per_year)
    skew = _standardized_moment(r, 3)
    kurt = _standardized_moment(r, 4)                 # non-excess (normal = 3)

    denom = 1.0 - skew * sr_hat + ((kurt - 1.0) / 4.0) * sr_hat**2
    if denom <= 0:
        return np.nan
    z = (sr_hat - sr_star) * np.sqrt(n - 1) / np.sqrt(denom)
    return _NORM.cdf(z)


def expected_max_sharpe(sr_trials_std: float, n_trials: int) -> float:
    """Expected maximum annualized Sharpe under the null (no skill) across
    `n_trials` independent trials, given the cross-trial std of annualized
    Sharpes. This is the bar a real signal must clear.
    """
    if n_trials <= 1 or sr_trials_std <= 0:
        return 0.0
    z1 = _NORM.inv_cdf(1.0 - 1.0 / n_trials)
    z2 = _NORM.inv_cdf(1.0 - 1.0 / (n_trials * np.e))
    return sr_trials_std * ((1.0 - _EULER_MASCHERONI) * z1 + _EULER_MASCHERONI * z2)


def deflated_sharpe_ratio(
    returns: pd.Series,
    n_trials: int,
    sr_trials_std: float,
    periods_per_year: int = TRADING_DAYS,
) -> float:
    """P(true Sharpe > expected max Sharpe from `n_trials`). A DSR near 1 means
    the result survives the multiple-testing correction; near 0.5 or below means
    it is likely a lucky draw among many trials.

    `n_trials` and `sr_trials_std` must come from your real research log -- the
    count of configurations tested and the dispersion of their Sharpes.
    """
    sr0 = expected_max_sharpe(sr_trials_std, n_trials)
    return probabilistic_sharpe_ratio(returns, sr_benchmark=sr0, periods_per_year=periods_per_year)
