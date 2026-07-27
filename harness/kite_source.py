"""Zerodha Kite Connect data adapter for Indian (NSE/BSE) equities, behind the
harness prices/panel seam.

`KiteOHLCVLoader` fetches daily OHLCV and assembles the same `{ticker: OHLCV}`
panel / wide close frame the rest of the harness consumes. It requires the
`kiteconnect` package plus a valid api_key/access_token and network to
kite.trade -- unavailable in registry-only sandboxes; the import is lazy so this
module loads without kiteconnect installed. The record-parsing helpers are pure
and unit-tested offline.

TWO INDIA-SPECIFIC WARNINGS baked into the workflow:
1. Kite historical candles are UNADJUSTED for corporate actions (splits, bonuses,
   rights). Apply adjustments from a corporate-actions source before trusting
   momentum/forecast signals, or you will see phantom jumps (STRATEGY.md sec. 4).
2. Retail cannot hold overnight shorts in cash equities, so the dollar-neutral
   L/S backtest is not tradable on cash; prefer the long-only top-k AER/IR
   simulation (harness/simulation.py), or take shorts via single-stock futures.
"""

from __future__ import annotations

import pandas as pd

from .data import close_panel

_OHLCV_COLS = ["open", "high", "low", "close", "volume", "amount"]


def _chunk_ranges(start, end, max_days: int = 2000):
    """Split [start, end] into contiguous chunks of at most `max_days` days --
    Kite's historical endpoint caps daily candles per request.
    """
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    chunks = []
    cur = start
    while cur <= end:
        chunk_end = min(cur + pd.Timedelta(days=max_days - 1), end)
        chunks.append((cur, chunk_end))
        cur = chunk_end + pd.Timedelta(days=1)
    return chunks


def _records_to_ohlcv(records: list[dict]) -> pd.DataFrame:
    """Turn Kite `historical_data` records into a date-indexed OHLCV frame with a
    derived `amount` column. Timestamps are reduced to tz-naive dates.
    """
    if not records:
        return pd.DataFrame(columns=_OHLCV_COLS)

    df = pd.DataFrame(records)
    ts = pd.to_datetime(df["date"])
    if getattr(ts.dt, "tz", None) is not None:
        ts = ts.dt.tz_localize(None)   # keep IST wall-clock date, drop tz
    df["date"] = ts.dt.normalize()
    df = df.set_index("date").sort_index()
    df["amount"] = df["volume"] * df["close"]
    df.index.name = "date"
    return df[_OHLCV_COLS]


class KiteOHLCVLoader:
    """Daily OHLCV for an NSE/BSE universe via Kite Connect.

    Respect Kite's rate limits (historical ~3 requests/sec) when loading a large
    universe. With real Kite data the panel index is already the exchange's
    trading sessions, so the harness's forward returns and signals use the
    correct NSE calendar automatically -- see `nse_future_sessions` for the one
    place (Kronos y_timestamp) that needs the calendar explicitly.
    """

    def __init__(self, api_key: str, access_token: str, exchange: str = "NSE"):
        from kiteconnect import KiteConnect  # lazy, optional

        self.kite = KiteConnect(api_key=api_key)
        self.kite.set_access_token(access_token)
        self.exchange = exchange
        self._token_map: dict[str, int] | None = None

    def _tokens(self, symbols) -> dict[str, int]:
        if self._token_map is None:
            self._token_map = {
                inst["tradingsymbol"]: inst["instrument_token"]
                for inst in self.kite.instruments(self.exchange)
            }
        return {s: self._token_map[s] for s in symbols if s in self._token_map}

    def fetch_ohlcv(self, symbol: str, start, end, interval: str = "day") -> pd.DataFrame:
        token = self._tokens([symbol])[symbol]
        records: list[dict] = []
        for frm, to in _chunk_ranges(start, end):
            records += self.kite.historical_data(token, frm.date(), to.date(), interval)
        return _records_to_ohlcv(records)

    def load_panel(self, symbols, start, end) -> dict[str, pd.DataFrame]:
        """`{symbol: OHLCV}` panel for `build_forecast_signal*` and `close_panel`."""
        return {s: self.fetch_ohlcv(s, start, end) for s in symbols}

    def load_prices(self, symbols, start, end) -> pd.DataFrame:
        """Wide close-price frame for momentum and the metrics/backtest layer."""
        return close_panel(self.load_panel(symbols, start, end))


def nse_future_sessions(last_date, n: int, calendar: str = "XBOM") -> pd.Series:
    """`n` future NSE trading sessions after `last_date`, for Kronos y_timestamp.

    Uses `exchange_calendars` if installed (lazy); falls back to business days
    (US-style, ignores Indian holidays) with the same shape so callers keep
    working. Install exchange_calendars for a correct NSE holiday calendar.
    """
    try:
        import exchange_calendars as xcals

        cal = xcals.get_calendar(calendar)
        start = pd.Timestamp(last_date) + pd.Timedelta(days=1)
        sessions = cal.sessions_in_range(start, start + pd.Timedelta(days=n * 3 + 15))
        return pd.Series(sessions[:n])
    except Exception:
        future = pd.bdate_range(pd.Timestamp(last_date) + pd.Timedelta(days=1), periods=n)
        return pd.Series(future)
