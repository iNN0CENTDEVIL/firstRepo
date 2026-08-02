"""Tests for purged CV and probabilistic/deflated Sharpe."""

import numpy as np
import pandas as pd

from harness.validation import (
    PurgedKFold,
    deflated_sharpe_ratio,
    expected_max_sharpe,
    probabilistic_sharpe_ratio,
)


def test_purged_kfold_no_leakage():
    dates = pd.bdate_range("2020-01-01", periods=100)
    cv = PurgedKFold(n_splits=5, horizon=3, embargo=0.05)
    embargo_n = int(len(dates) * 0.05)
    for train, test in cv.split(dates):
        t0, t1 = int(test[0]), int(test[-1])
        assert set(train).isdisjoint(set(test))
        # nothing in the purge window [t0 - horizon, t1]
        assert not ((train >= t0 - 3) & (train <= t1)).any()
        # nothing in the embargo window right after the test block
        assert not ((train > t1) & (train <= t1 + embargo_n)).any()


def test_purged_kfold_covers_all_test_indices():
    dates = pd.bdate_range("2020-01-01", periods=53)
    cv = PurgedKFold(n_splits=5)
    covered = np.concatenate([test for _, test in cv.split(dates)])
    assert sorted(covered.tolist()) == list(range(53))


def test_expected_max_sharpe_increases_with_trials():
    a = expected_max_sharpe(sr_trials_std=0.5, n_trials=10)
    b = expected_max_sharpe(sr_trials_std=0.5, n_trials=1000)
    assert 0 < a < b


def test_psr_high_for_strong_track_record():
    rng = np.random.default_rng(0)
    strong = pd.Series(rng.normal(0.0015, 0.01, 2000))  # ann. SR ~ 2.4
    weak = pd.Series(rng.normal(0.00005, 0.01, 2000))   # ann. SR ~ 0.08
    assert probabilistic_sharpe_ratio(strong) > 0.99
    assert probabilistic_sharpe_ratio(weak) < 0.9


def test_deflated_sharpe_falls_as_trials_grow():
    rng = np.random.default_rng(1)
    returns = pd.Series(rng.normal(0.0009, 0.01, 1500))
    few = deflated_sharpe_ratio(returns, n_trials=2, sr_trials_std=0.5)
    many = deflated_sharpe_ratio(returns, n_trials=2000, sr_trials_std=0.5)
    assert few > many
