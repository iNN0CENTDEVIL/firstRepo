"""Tests for the 10-K risk-factor change signal and EDGAR extraction."""

import numpy as np
import pandas as pd

from harness.edgar import extract_risk_factors, strip_html
from harness.metrics import rank_ic
from harness.text_signal import (
    build_risk_change_signal,
    make_planted_dataset,
    risk_factor_change,
)


def test_risk_factor_change_identical_is_zero():
    text = "alpha beta gamma delta epsilon"
    out = risk_factor_change(text, text)
    assert out["jaccard_distance"] == 0.0
    assert out["new_word_fraction"] == 0.0
    assert out["word_count_ratio"] == 0.0


def test_risk_factor_change_disjoint_is_one():
    out = risk_factor_change("aaa bbb ccc", "xxx yyy zzz")
    assert out["jaccard_distance"] == 1.0
    assert out["new_word_fraction"] == 1.0


def test_new_word_fraction_is_exact_share():
    # 4 words this year, 2 of them new -> 0.5
    out = risk_factor_change("aaa bbb ppp qqq", "aaa bbb ccc ddd")
    assert out["new_word_fraction"] == 0.5


def test_extract_risk_factors_slices_item_1a_to_1b():
    doc = (
        "<html><body>Item 1. Business blah."
        "<p>Item 1A. Risk Factors: our supply chain may fail.</p>"
        "Item 1B. Unresolved Staff Comments none.</body></html>"
    )
    section = extract_risk_factors(doc)
    assert section.lower().startswith("item 1a")
    assert "supply chain may fail" in section
    assert "unresolved" not in section.lower()


def test_extract_risk_factors_missing_returns_empty():
    assert extract_risk_factors("<p>No such section here.</p>") == ""


def test_strip_html_removes_tags():
    assert strip_html("<b>a</b>  <i>b</i>") == "a b"


def test_build_signal_is_availability_dated():
    filings = pd.DataFrame(
        {
            "ticker": ["AAA", "AAA"],
            "filing_date": pd.to_datetime(["2020-03-02", "2021-03-01"]),
            "risk_text": ["aaa bbb ccc", "aaa bbb xxx"],  # one word changed
        }
    )
    calendar = pd.bdate_range("2020-01-01", "2021-06-01", name="date")
    signal = build_risk_change_signal(filings, calendar, feature="new_word_fraction")

    # no value before the second filing (first filing has no prior to diff)
    assert signal.loc[:"2021-02-26", "AAA"].isna().all()
    # value appears on/after the 2021 filing date and is forward-filled
    assert signal.loc["2021-03-01", "AAA"] > 0
    assert signal.loc["2021-05-03", "AAA"] == signal.loc["2021-03-01", "AAA"]


def test_planted_relationship_gives_negative_ic():
    filings, fwd, calendar = make_planted_dataset(n_stocks=80, n_years=4, seed=0)
    signal = build_risk_change_signal(filings, calendar, feature="new_word_fraction")
    assert rank_ic(signal, fwd).mean() < 0
