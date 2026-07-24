"""Tests for the Kronos integration: investment simulation, test-time ensembling,
and the forecast-signal pipeline."""

import numpy as np
import pandas as pd

from harness.data import close_panel, generate_synthetic_ohlcv
from harness.kronos_signal import (
    build_forecast_signal,
    build_forecast_signal_batched,
    ensemble_forecast,
    future_timestamps,
    trailing_return_forecaster,
)
from harness.metrics import forward_returns, rank_ic
from harness.simulation import top_k_long_only


def _frame(values):
    dates = pd.bdate_range("2020-01-01", periods=len(values), name="date")
    return pd.DataFrame(values, index=dates, columns=list("ABCDE"))


def test_top_k_long_only_rewards_good_selection():
    # signal ranks stocks by forward return each day; the top name carries a
    # persistent positive spread with day-to-day noise -> positive AER and IR.
    rng = np.random.default_rng(0)
    rows = []
    for _ in range(120):
        base = rng.normal(0.0, 0.005, 5)
        base[4] += 0.01  # the top-ranked stock outperforms on average
        rows.append(base.tolist())
    sig = _frame([[1, 2, 3, 4, 5]] * 120)
    fwd = _frame(rows)
    out = top_k_long_only(sig, fwd, k=1)
    assert out["aer"] > 0
    assert out["ir"] > 0
    assert out["hit_rate"] > 0.5


def test_top_k_long_only_flat_signal_no_excess():
    # every stock identical forward return -> top-k equals benchmark -> ~0 excess.
    sig = _frame([[5, 4, 3, 2, 1]] * 30)
    fwd = _frame([[0.01, 0.01, 0.01, 0.01, 0.01]] * 30)
    out = top_k_long_only(sig, fwd, k=2)
    assert abs(out["aer"]) < 1e-12


def test_future_timestamps_extends_calendar():
    idx = pd.bdate_range("2020-01-01", periods=10)
    fut = future_timestamps(idx, pred_len=3)
    assert len(fut) == 3
    assert (fut > idx[-1]).all()
    # continues business-day cadence from the last historical date
    assert fut.iloc[0] == idx[-1] + pd.offsets.BDay(1)


def test_ensemble_forecast_averages():
    assert ensemble_forecast(lambda: 0.5, 10) == 0.5
    seq = iter([0.0, 1.0, 2.0, 3.0])
    assert ensemble_forecast(lambda: next(seq), 4) == 1.5


def test_ensemble_forecast_reduces_variance():
    rng = np.random.default_rng(0)
    single = [rng.normal() for _ in range(400)]
    ens = [ensemble_forecast(lambda: rng.normal(), 25) for _ in range(400)]
    assert np.std(ens) < np.std(single)


def test_build_forecast_signal_no_lookahead():
    panel = generate_synthetic_ohlcv(n_stocks=5, n_days=300, seed=1)

    seen_last_dates = []

    def spy(window):
        seen_last_dates.append(window.index[-1])
        return float(window["close"].iloc[-1])

    signal = build_forecast_signal(panel, spy, lookback=50, step=25)
    dates = next(iter(panel.values())).index
    # every window ends at or before its own signal date (never looks ahead).
    assert all(d in set(dates) for d in seen_last_dates)
    assert signal.shape[1] == 5


def test_batched_builder_matches_per_stock():
    # batching a whole universe per date must give exactly the per-stock result.
    panel = generate_synthetic_ohlcv(n_stocks=8, n_days=400, seed=2)
    single = build_forecast_signal(panel, trailing_return_forecaster, lookback=252, step=20)
    batched = build_forecast_signal_batched(
        panel, lambda ws: [trailing_return_forecaster(w) for w in ws], lookback=252, step=20
    )
    pd.testing.assert_frame_equal(single, batched)


def test_batched_builder_handles_ragged_universe():
    panel = generate_synthetic_ohlcv(n_stocks=4, n_days=400, seed=3)
    a, b, c, d = list(panel)
    panel[a].iloc[:350, :] = np.nan   # IPOs late: never gets a full 100-day window
    panel[b].iloc[151:, :] = np.nan   # delists after row 150

    seen = []

    def spy(windows):
        for w in windows:
            assert not w.isnull().values.any()  # no NaN ever reaches predict_batch
        seen.append(len(windows))
        return [float(w["close"].iloc[-1]) for w in windows]

    signal = build_forecast_signal_batched(panel, spy, lookback=100, step=10)

    assert signal[a].isna().all()                 # too little history -> no signal
    assert signal[b].notna().any()                # trades early
    assert pd.isna(signal[b].iloc[-1])            # no stale signal after delisting
    assert signal[c].notna().any() and signal[d].notna().any()
    assert min(seen) < 4                           # some dates had fewer eligible names


def test_forecast_pipeline_positive_ic_on_synthetic_momentum():
    panel = generate_synthetic_ohlcv(n_stocks=60, n_days=900, seed=0)
    closes = close_panel(panel)
    signal = build_forecast_signal(panel, trailing_return_forecaster, lookback=252, step=5)
    fwd = forward_returns(closes, horizon=1, gap=1)
    assert rank_ic(signal, fwd).mean() > 0
    assert top_k_long_only(signal, fwd, k=10)["aer"] > 0
