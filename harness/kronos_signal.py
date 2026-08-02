"""Kronos as a cross-sectional return-forecast signal (arXiv:2508.02739).

A `ReturnForecaster` maps a trailing OHLCV window to an expected forward return.
`build_forecast_signal` rolls one across dates into an availability-dated signal
frame (no look-ahead), which then feeds the same metrics / backtest / simulation
as any other signal.

`KronosForecaster` wraps the pretrained Kronos model; `ensemble_forecast`
implements the paper's test-time scaling (average N stochastic rollouts).
`trailing_return_forecaster` is a transparent offline baseline that stands in for
Kronos so the pipeline runs and self-validates without a model download.
"""

from __future__ import annotations

from typing import Callable, Protocol

import numpy as np
import pandas as pd

# A window of trailing OHLCV (index = dates, columns include 'close') -> forecast.
ReturnForecaster = Callable[[pd.DataFrame], float]
# Many windows -> many forecasts, in one call (GPU-batched at universe scale).
BatchReturnForecaster = Callable[[list], list]


class _SupportsPredict(Protocol):
    def __call__(self, ohlcv: pd.DataFrame) -> float: ...


def ensemble_forecast(sample_fn: Callable[[], float], n_samples: int) -> float:
    """Test-time scaling: average `n_samples` stochastic forecasts to reduce
    variance (Kronos Figure 7 -- IC/RankIC improve monotonically with N). Use
    this to wrap a forecaster that returns a single stochastic draw. The real
    Kronos model does the same averaging internally via its `sample_count`
    argument, so `KronosForecaster` does not need this wrapper.
    """
    if n_samples < 1:
        raise ValueError("n_samples must be >= 1")
    return float(np.mean([sample_fn() for _ in range(n_samples)]))


def future_timestamps(index: pd.DatetimeIndex, pred_len: int) -> pd.Series:
    """Generate `pred_len` future timestamps continuing `index`'s frequency --
    the `y_timestamp` argument Kronos.predict() requires. Falls back to business
    days if the frequency can't be inferred.
    """
    freq = pd.infer_freq(index) or "B"
    future = pd.date_range(start=index[-1], periods=pred_len + 1, freq=freq)[1:]
    return pd.Series(future)


def trailing_return_forecaster(
    window: pd.DataFrame, lookback: int = 252, skip: int = 21
) -> float:
    """Offline baseline standing in for Kronos: forecast the forward return as
    trailing 12-1 momentum of the window's close. Deterministic and transparent
    -- NOT a foundation-model forecast, just a stand-in to exercise the pipeline.
    """
    close = window["close"]
    if len(close) < lookback:
        return np.nan
    return close.iloc[-1 - skip] / close.iloc[-lookback] - 1.0


def build_forecast_signal(
    ohlcv_panel: dict[str, pd.DataFrame],
    forecaster: ReturnForecaster,
    lookback: int = 252,
    step: int = 1,
    min_periods: int | None = None,
) -> pd.DataFrame:
    """Roll `forecaster` across dates for every stock to build a (dates x tickers)
    signal frame. At date t the forecaster sees only `ohlcv[t-lookback+1 : t+1]`,
    so there is no look-ahead. `step` > 1 forecasts every `step`-th day and
    forward-fills between (cheaper; real use batches on a GPU).

    Handles ragged universes like the batched builder: panels are aligned to a
    union calendar, a ticker is forecast on a date only when its window has at
    least `min_periods` rows and no NaN (never feeding a NaN window to the
    forecaster), and non-trading dates are left NaN so no signal leaks across an
    IPO or delisting.
    """
    min_periods = min_periods or lookback
    panel, dates = _aligned_panel(ohlcv_panel)
    eval_positions = range(min_periods - 1, len(dates), step)

    columns = {}
    for ticker, df in panel.items():
        values = {}
        for pos in eval_positions:
            window = df.iloc[max(0, pos - lookback + 1) : pos + 1]
            if len(window) >= min_periods and not window.isnull().values.any():
                values[dates[pos]] = forecaster(window)
        columns[ticker] = pd.Series(values)

    signal = pd.DataFrame(columns).reindex(dates).ffill()
    trading = pd.DataFrame({t: panel[t]["close"] for t in panel}).notna()
    signal = signal.where(trading)
    signal.index.name = "date"
    return signal


def _aligned_panel(ohlcv_panel: dict[str, pd.DataFrame]):
    """Reindex every ticker onto the union calendar so ragged histories (IPOs,
    delistings) line up; dates a ticker didn't trade become NaN rows.
    """
    calendar = None
    for df in ohlcv_panel.values():
        calendar = df.index if calendar is None else calendar.union(df.index)
    calendar = calendar.sort_values()
    return {t: df.reindex(calendar) for t, df in ohlcv_panel.items()}, calendar


def build_forecast_signal_batched(
    ohlcv_panel: dict[str, pd.DataFrame],
    batch_forecaster: BatchReturnForecaster,
    lookback: int = 252,
    step: int = 1,
) -> pd.DataFrame:
    """Like `build_forecast_signal`, but forecasts every stock for a given date in
    a single `batch_forecaster` call -- the only practical shape for a real
    GPU-bound model (`KronosForecaster.batch`) across a large universe.

    Handles ragged universes: panels are aligned to a union calendar, and each
    date only forecasts tickers with a full, NaN-free `lookback` window (so
    Kronos.predict_batch's equal-length, no-NaN contract holds). Stocks are left
    NaN on dates they aren't trading, so no stale signal leaks past a delisting.
    No look-ahead: date t uses only rows <= t.
    """
    panel, dates = _aligned_panel(ohlcv_panel)
    tickers = list(panel)
    columns = {t: {} for t in tickers}

    for pos in range(lookback - 1, len(dates), step):
        eligible, windows = [], []
        for t in tickers:
            window = panel[t].iloc[pos - lookback + 1 : pos + 1]
            if len(window) == lookback and not window.isnull().values.any():
                eligible.append(t)
                windows.append(window)
        if not windows:
            continue
        for t, value in zip(eligible, batch_forecaster(windows)):
            columns[t][dates[pos]] = value

    signal = pd.DataFrame({t: pd.Series(columns[t]) for t in tickers}).reindex(dates).ffill()
    trading = pd.DataFrame({t: panel[t]["close"] for t in tickers}).notna()
    signal = signal.where(trading)
    signal.index.name = "date"
    return signal


class KronosForecaster:
    """Adapter: the pretrained Kronos K-line model as a `ReturnForecaster`, with
    test-time ensembling.

    Requires `torch` and the Kronos package (github.com/shiyu-coder/Kronos) plus a
    HuggingFace model download -- unavailable in registry-only sandboxes. Imports
    are lazy so this module loads without those installed. The call matches the
    published API: KronosPredictor(model, tokenizer, device=None, max_context=512)
    and predict(df, x_timestamp, y_timestamp, pred_len, T, top_k, top_p,
    sample_count, verbose). `sample_count` averages that many stochastic rollouts
    inside predict() -- this is Kronos's test-time scaling.
    """

    _COLS = ["open", "high", "low", "close", "volume", "amount"]

    def __init__(
        self,
        model_id: str = "NeoQuasar/Kronos-base",
        tokenizer_id: str = "NeoQuasar/Kronos-Tokenizer-base",
        device: str | None = None,
        pred_len: int = 1,
        sample_count: int = 20,
        max_context: int = 512,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 0.9,
    ):
        from model import Kronos, KronosTokenizer, KronosPredictor  # lazy, optional

        tokenizer = KronosTokenizer.from_pretrained(tokenizer_id)
        model = Kronos.from_pretrained(model_id)
        self.predictor = KronosPredictor(model, tokenizer, device=device, max_context=max_context)
        self.pred_len = pred_len
        self.sample_count = sample_count
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p

    def __call__(self, ohlcv: pd.DataFrame) -> float:
        cols = [c for c in self._COLS if c in ohlcv.columns]
        pred = self.predictor.predict(
            df=ohlcv[cols],
            x_timestamp=pd.Series(ohlcv.index),
            y_timestamp=future_timestamps(ohlcv.index, self.pred_len),
            pred_len=self.pred_len,
            T=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
            sample_count=self.sample_count,  # averaged rollouts (test-time scaling)
            verbose=False,
        )
        return float(pred["close"].iloc[-1] / ohlcv["close"].iloc[-1] - 1.0)

    def batch(self, windows: list[pd.DataFrame]) -> list[float]:
        """Forecast a list of equal-length OHLCV windows in one call via
        Kronos.predict_batch -- use with `build_forecast_signal_batched` to score
        a whole universe per date efficiently.
        """
        cols = [c for c in self._COLS if c in windows[0].columns]
        preds = self.predictor.predict_batch(
            df_list=[w[cols] for w in windows],
            x_timestamp_list=[pd.Series(w.index) for w in windows],
            y_timestamp_list=[future_timestamps(w.index, self.pred_len) for w in windows],
            pred_len=self.pred_len,
            T=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
            sample_count=self.sample_count,
            verbose=False,
        )
        return [
            float(p["close"].iloc[-1] / w["close"].iloc[-1] - 1.0)
            for p, w in zip(preds, windows)
        ]
