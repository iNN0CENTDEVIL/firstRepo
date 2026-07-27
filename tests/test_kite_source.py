"""Tests for the Zerodha Kite data adapter (offline: parsing + chunking only)."""

import pandas as pd

from harness.kite_source import _chunk_ranges, _records_to_ohlcv


def test_chunk_ranges_contiguous_and_bounded():
    chunks = _chunk_ranges("2015-01-01", "2024-12-31", max_days=2000)
    assert chunks[0][0] == pd.Timestamp("2015-01-01")
    assert chunks[-1][1] == pd.Timestamp("2024-12-31")
    for (_, a_end), (b_start, _) in zip(chunks, chunks[1:]):
        assert b_start == a_end + pd.Timedelta(days=1)   # no gaps or overlaps
    for frm, to in chunks:
        assert (to - frm).days <= 1999


def test_chunk_ranges_single_chunk_when_short():
    assert _chunk_ranges("2024-01-01", "2024-01-10") == [
        (pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-10"))
    ]


def test_records_to_ohlcv_builds_panel_frame():
    records = [
        {"date": "2024-01-02", "open": 104, "high": 106, "low": 103, "close": 105, "volume": 1200},
        {"date": "2024-01-01", "open": 100, "high": 105, "low": 99, "close": 104, "volume": 1000},
    ]
    df = _records_to_ohlcv(records)
    assert list(df.columns) == ["open", "high", "low", "close", "volume", "amount"]
    assert df.index.name == "date"
    assert df.index.is_monotonic_increasing               # sorted by date
    assert df.loc["2024-01-02", "amount"] == 1200 * 105   # amount = volume * close


def test_records_to_ohlcv_drops_timezone():
    records = [
        {"date": "2024-03-01 00:00:00+05:30", "open": 1, "high": 1, "low": 1, "close": 1, "volume": 1},
    ]
    df = _records_to_ohlcv(records)
    assert df.index.tz is None
    assert df.index[0] == pd.Timestamp("2024-03-01")      # IST date preserved, not shifted


def test_records_to_ohlcv_empty():
    df = _records_to_ohlcv([])
    assert list(df.columns) == ["open", "high", "low", "close", "volume", "amount"]
    assert df.empty


def test_module_imports_without_kiteconnect():
    import harness.kite_source as k

    assert hasattr(k, "KiteOHLCVLoader")
    assert hasattr(k, "nse_future_sessions")
