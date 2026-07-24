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


class _SupportsPredict(Protocol):
    def __call__(self, ohlcv: pd.DataFrame) -> float: ...


def ensemble_forecast(sample_fn: Callable[[], float], n_samples: int) -> float:
    """Test-time scaling: average `n_samples` stochastic forecasts to reduce
    variance (Kronos Figure 7 -- IC/RankIC improve monotonically with N).
    """
    if n_samples < 1:
        raise ValueError("n_samples must be >= 1")
    return float(np.mean([sample_fn() for _ in range(n_samples)]))


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
    """
    min_periods = min_periods or lookback
    any_df = next(iter(ohlcv_panel.values()))
    dates = any_df.index
    eval_positions = range(min_periods - 1, len(dates), step)

    columns = {}
    for ticker, df in ohlcv_panel.items():
        values = {}
        for pos in eval_positions:
            window = df.iloc[max(0, pos - lookback + 1) : pos + 1]
            values[dates[pos]] = forecaster(window)
        columns[ticker] = pd.Series(values)

    signal = pd.DataFrame(columns).reindex(dates).ffill()
    signal.index.name = "date"
    return signal


class KronosForecaster:
    """Adapter: the pretrained Kronos K-line model as a `ReturnForecaster`, with
    test-time ensembling.

    Requires `torch` and the Kronos package (github.com/shiyu-coder/Kronos) plus a
    HuggingFace model download -- unavailable in registry-only sandboxes. Imports
    are lazy so this module loads without those installed. Confirm `predict()`'s
    signature against the installed Kronos version before live use.
    """

    def __init__(
        self,
        model_id: str = "NeoQuasar/Kronos-base",
        tokenizer_id: str = "NeoQuasar/Kronos-Tokenizer-base",
        device: str = "cpu",
        pred_len: int = 1,
        sample_count: int = 20,
        max_context: int = 512,
        temperature: float = 1.0,
        top_p: float = 0.9,
    ):
        from model import Kronos, KronosTokenizer, KronosPredictor  # lazy, optional

        tokenizer = KronosTokenizer.from_pretrained(tokenizer_id)
        model = Kronos.from_pretrained(model_id)
        self.predictor = KronosPredictor(model, tokenizer, device=device, max_context=max_context)
        self.pred_len = pred_len
        self.sample_count = sample_count
        self.temperature = temperature
        self.top_p = top_p

    def __call__(self, ohlcv: pd.DataFrame) -> float:
        cols = ["open", "high", "low", "close", "volume", "amount"]
        pred = self.predictor.predict(
            df=ohlcv[cols],
            pred_len=self.pred_len,
            T=self.temperature,
            top_p=self.top_p,
            sample_count=self.sample_count,  # ensembled rollouts (test-time scaling)
        )
        return float(pred["close"].iloc[-1] / ohlcv["close"].iloc[-1] - 1.0)
