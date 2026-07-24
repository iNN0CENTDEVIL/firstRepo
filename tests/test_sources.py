"""Tests for the real-data CSV adapter (offline)."""

import pandas as pd

from harness.sources import load_prices_from_csv


def test_load_prices_from_csv_pivots_long_to_wide(tmp_path):
    csv = tmp_path / "extract.csv"
    csv.write_text(
        "date,ticker,close_adj\n"
        "2020-01-02,AAA,10.0\n"
        "2020-01-02,BBB,20.0\n"
        "2020-01-03,AAA,11.0\n"
        "2020-01-03,BBB,21.0\n"
    )
    prices = load_prices_from_csv(str(csv))

    assert list(prices.columns) == ["AAA", "BBB"]
    assert prices.index.name == "date"
    assert isinstance(prices.index, pd.DatetimeIndex)
    assert prices.loc["2020-01-03", "AAA"] == 11.0
    assert prices.shape == (2, 2)


def test_load_prices_from_csv_keeps_delisted_as_nan(tmp_path):
    # BBB stops trading after the first day -> present but NaN, not dropped.
    csv = tmp_path / "extract.csv"
    csv.write_text(
        "date,ticker,close_adj\n"
        "2020-01-02,AAA,10.0\n"
        "2020-01-02,BBB,20.0\n"
        "2020-01-03,AAA,11.0\n"
    )
    prices = load_prices_from_csv(str(csv))
    assert "BBB" in prices.columns
    assert pd.isna(prices.loc["2020-01-03", "BBB"])
