"""First AI signal: 10-K year-over-year risk-factor change ("Lazy Prices").

Firms that materially rewrite their risk-factor language tend to underperform.
The default features are deliberately *mechanical* (how much the text changed),
not model-judged sentiment -- mechanical features are far less exposed to LLM
hindsight leakage (STRATEGY.md section 4), because an LLM scoring a 2019 filing
already knows what happened next, but a token-set diff does not.

Pipeline:
    filings (ticker, filing_date, risk_text)
      -> risk_factor_change() per consecutive pair        [feature per filing]
      -> build_risk_change_signal()                        [availability-dated
                                                            daily dates x tickers frame]
      -> the same metrics/backtest/validation as any signal.

NOTE ON SIGN: larger change historically predicts *lower* forward returns, so as
a higher-is-better score you short the big changers. The builder returns the raw
change magnitude; let rank IC confirm the (negative) sign rather than hardcoding
a belief. `make_planted_dataset` fabricates a coupled world where this holds, so
the wiring is self-validating like the momentum gate.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

_TOKEN = re.compile(r"[a-z]{3,}")

_FEATURES = ("jaccard_distance", "new_word_fraction", "word_count_ratio")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def risk_factor_change(text_now: str, text_prior: str) -> dict:
    """Mechanical year-over-year change features between two risk-factor texts.

    - jaccard_distance : 1 - |shared vocab| / |combined vocab|  (0 = identical)
    - new_word_fraction: share of this year's vocab that is new vs last year
    - word_count_ratio : signed change in length relative to last year
    """
    now, prior = tokenize(text_now), tokenize(text_prior)
    set_now, set_prior = set(now), set(prior)
    union = set_now | set_prior

    jaccard = 1.0 - len(set_now & set_prior) / len(union) if union else 0.0
    new_frac = len(set_now - set_prior) / len(set_now) if set_now else 0.0
    wc_ratio = (len(now) - len(prior)) / len(prior) if prior else 0.0
    return {"jaccard_distance": jaccard, "new_word_fraction": new_frac, "word_count_ratio": wc_ratio}


def build_risk_change_signal(
    filings: pd.DataFrame, calendar: pd.DatetimeIndex, feature: str = "new_word_fraction"
) -> pd.DataFrame:
    """Turn a long filings frame into an availability-dated daily signal.

    `filings` columns: ticker, filing_date (datetime), risk_text. For each
    ticker the change vs its previous filing becomes available on the current
    filing_date and is forward-filled until the next filing -- so no value is
    ever used before it was public (point-in-time). Output: `calendar` x tickers.
    """
    if feature not in _FEATURES:
        raise ValueError(f"feature must be one of {_FEATURES}")

    columns = {}
    for ticker, grp in filings.sort_values("filing_date").groupby("ticker"):
        dates = list(grp["filing_date"])
        texts = list(grp["risk_text"])
        values = {
            dates[i]: risk_factor_change(texts[i], texts[i - 1])[feature]
            for i in range(1, len(texts))
        }
        if values:
            columns[ticker] = pd.Series(values)

    signal = pd.DataFrame(columns)
    if signal.empty:
        return pd.DataFrame(index=calendar)
    full = calendar.union(signal.index)
    signal = signal.reindex(full).sort_index().ffill().reindex(calendar)
    signal.index.name = "date"
    return signal


# --------------------------------------------------------------------------- #
# Coupled synthetic world for self-validation (no data subscription needed)
# --------------------------------------------------------------------------- #
def make_planted_dataset(
    n_stocks: int = 80,
    n_years: int = 4,
    seed: int = 0,
    base_len: int = 200,
    start: str = "2016-06-15",
):
    """Fabricate (filings, forward_returns, calendar) with a planted relationship:
    each stock has a latent `deterioration`; higher deterioration -> more
    risk-text rewritten AND lower forward returns. So the change signal should
    show a *negative* rank IC, validating end-to-end detection.

    Base and new words use disjoint alphabets, so `new_word_fraction` tracks the
    injected mutation rate exactly.
    """
    rng = np.random.default_rng(seed)
    tickers = [f"T{i:03d}" for i in range(n_stocks)]
    deterioration = rng.normal(size=n_stocks)              # latent, per stock
    mutation_rate = 0.05 + 0.5 / (1.0 + np.exp(-deterioration))  # ~[0.05, 0.55]

    base_alpha = list("abcdefghijklm")
    new_alpha = list("nopqrstuvwxyz")

    def word(alpha, length):
        return "".join(rng.choice(alpha, length))

    filing_dates = pd.Timestamp(start) + pd.to_timedelta(
        [365 * y for y in range(n_years)], unit="D"
    )

    rows = []
    for i, ticker in enumerate(tickers):
        tokens = [word(base_alpha, 4) for _ in range(base_len)]
        for y, fdate in enumerate(filing_dates):
            if y > 0:
                k = int(base_len * mutation_rate[i])
                idx = rng.choice(base_len, k, replace=False)
                for j in idx:
                    tokens[j] = word(new_alpha, 6)
            rows.append({"ticker": ticker, "filing_date": fdate, "risk_text": " ".join(tokens)})

    filings = pd.DataFrame(rows)

    calendar = pd.bdate_range(start, filing_dates[-1] + pd.Timedelta(days=30), name="date")
    # forward returns: mean decreases with deterioration -> big changers underperform
    means = -deterioration * 0.002
    fwd = pd.DataFrame(
        rng.normal(loc=means, scale=0.02, size=(len(calendar), n_stocks)),
        index=calendar,
        columns=tickers,
    )
    return filings, fwd, calendar
