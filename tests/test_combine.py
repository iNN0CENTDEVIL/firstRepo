"""Tests for signal combination and incremental-IC attribution, using
constructed factors with known structure."""

import numpy as np
import pandas as pd

from harness.combine import combine_signals, cross_sectional_zscore, incremental_ic
from harness.metrics import rank_ic


def _panel(arr):
    idx = pd.bdate_range("2020-01-01", periods=arr.shape[0], name="date")
    cols = [f"S{i}" for i in range(arr.shape[1])]
    return pd.DataFrame(arr, index=idx, columns=cols)


def test_cross_sectional_zscore_normalizes_rows():
    rng = np.random.default_rng(0)
    z = cross_sectional_zscore(_panel(rng.normal(size=(10, 50))))
    assert np.allclose(z.mean(axis=1).values, 0.0, atol=1e-9)
    assert np.allclose(z.std(axis=1).values, 1.0, atol=1e-9)


def test_combine_orients_negative_signal():
    rng = np.random.default_rng(1)
    f = rng.normal(size=(200, 40))
    fwd = _panel(f + rng.normal(scale=0.5, size=(200, 40)))
    neg = _panel(-f)  # negatively predicts fwd
    combo = combine_signals({"neg": neg}, fwd)
    assert rank_ic(combo, fwd).mean() > 0  # oriented to point the right way


def test_incremental_ic_orthogonal_factors():
    rng = np.random.default_rng(2)
    T, N = 400, 60
    f1, f2 = rng.normal(size=(T, N)), rng.normal(size=(T, N))
    fwd = _panel(f1 + f2 + rng.normal(scale=0.5, size=(T, N)))
    signals = {
        "mom": _panel(f1 + rng.normal(scale=0.5, size=(T, N))),   # sees f1
        "text": _panel(f2 + rng.normal(scale=0.5, size=(T, N))),  # sees f2
        "kronos": _panel(rng.normal(size=(T, N))),                # sees nothing
    }
    res = incremental_ic(signals, fwd)
    t = res["attribution"]
    assert t.loc["mom", "incremental_ic"] > 0.03
    assert t.loc["text", "incremental_ic"] > 0.03
    # a useless signal adds far less than a real one
    assert t.loc["kronos", "incremental_ic"] < t.loc["mom", "incremental_ic"]
    # the blend beats any single component
    assert res["combined_ic"] > t.loc["mom", "standalone_ic"]


def test_incremental_ic_detects_redundancy():
    rng = np.random.default_rng(3)
    T, N = 400, 60
    f1, f2 = rng.normal(size=(T, N)), rng.normal(size=(T, N))
    fwd = _panel(f1 + f2 + rng.normal(scale=0.5, size=(T, N)))
    signals = {
        "a": _panel(f1 + rng.normal(scale=0.4, size=(T, N))),   # collinear with b
        "b": _panel(f1 + rng.normal(scale=0.4, size=(T, N))),   # collinear with a
        "c": _panel(f2 + rng.normal(scale=0.4, size=(T, N))),   # unique factor
    }
    t = incremental_ic(signals, fwd)["attribution"]
    # a and b cover each other, so each adds little; c is unique and adds more
    assert t.loc["c", "incremental_ic"] > t.loc["a", "incremental_ic"]
    assert t.loc["c", "incremental_ic"] > t.loc["b", "incremental_ic"]
    # yet both a and b have real standalone IC
    assert t.loc["a", "standalone_ic"] > 0.1
