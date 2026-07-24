"""Real-data adapters behind the same `prices` DataFrame seam used everywhere
else (DatetimeIndex x tickers of adjusted closes).

Two entry points:

- `load_prices_from_csv` -- ingest a real vendor extract (Sharadar SEP, Norgate
  export, or any long-format date/ticker/price CSV). Works fully offline; this
  is the seam you feed production data through.
- `load_prices_stooq` -- fetch daily closes from Stooq (free, no API key).

SURVIVORSHIP WARNING: free feeds (Stooq, Yahoo) generally carry only
*currently listed* tickers, so a universe built from them is survivorship
biased -- delisted losers are missing and every backtest looks too good. Use
them for development only. For go/no-go decisions use a point-in-time,
survivorship-bias-free source (see STRATEGY.md section 4) delivered as CSV and
loaded via `load_prices_from_csv`.
"""

from __future__ import annotations

import io
import urllib.request
from typing import Iterable

import pandas as pd


def load_prices_from_csv(
    path: str,
    date_col: str = "date",
    ticker_col: str = "ticker",
    price_col: str = "close_adj",
) -> pd.DataFrame:
    """Pivot a long-format CSV (one row per date/ticker) into the wide `prices`
    frame. `price_col` should be split/dividend adjusted. Delisted names are
    kept over their live window (they simply go NaN after delisting), preserving
    survivorship-bias-free data if the source provides it.
    """
    df = pd.read_csv(path, parse_dates=[date_col])
    wide = df.pivot(index=date_col, columns=ticker_col, values=price_col).sort_index()
    wide.index.name = "date"
    wide.columns.name = None
    return wide


_STOOQ_URL = "https://stooq.com/q/d/l/?s={sym}&i=d&d1={d1}&d2={d2}"


def load_prices_stooq(
    tickers: Iterable[str], start: str, end: str, timeout: int = 30
) -> pd.DataFrame:
    """Fetch daily closes from Stooq for `tickers` between `start` and `end`
    (YYYY-MM-DD). US tickers get a `.us` suffix if none is given.

    Requires outbound network to stooq.com. Blocked in locked-down sandboxes
    whose egress is limited to package registries -- run this where stooq.com is
    reachable. Read the survivorship warning in this module's docstring first.
    """
    d1, d2 = start.replace("-", ""), end.replace("-", "")
    series = {}
    for ticker in tickers:
        sym = ticker.lower() if "." in ticker else f"{ticker.lower()}.us"
        url = _STOOQ_URL.format(sym=sym, d1=d1, d2=d2)
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            raw = resp.read().decode()
        col = pd.read_csv(io.StringIO(raw), parse_dates=["Date"]).set_index("Date")["Close"]
        series[ticker] = col
    wide = pd.DataFrame(series).sort_index()
    wide.index.name = "date"
    return wide
