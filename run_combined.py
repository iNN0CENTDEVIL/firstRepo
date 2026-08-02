"""Combine momentum + 10-K risk-factor change + Kronos forecast, and attribute
each signal's value with leave-one-out incremental IC.

The universe is a single synthetic world with two latent drivers: a persistent
price trend (which momentum and the Kronos stand-in read from prices) and a risk
factor (which the 10-K change reads from filings). The forward-return truth
blends both, with the risk component treated as fresh information not yet in
trailing prices -- so the text signal can add value momentum cannot.

The Kronos slot uses the offline `trailing_return_forecaster` stand-in (the real
model can't run here); since that stand-in is essentially momentum, the
attribution should show it as largely redundant with momentum -- a correct and
instructive result: incremental IC catches redundancy that standalone IC hides.

Run: `python run_combined.py`
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from harness.combine import combine_signals, incremental_ic
from harness.kronos_signal import build_forecast_signal, trailing_return_forecaster
from harness.metrics import forward_returns  # noqa: F401  (available for real-return use)
from harness.signals import momentum
from harness.simulation import top_k_long_only
from harness.text_signal import build_risk_change_signal


def make_combined_dataset(n_stocks=80, n_years=6, seed=0, start="2015-01-01"):
    rng = np.random.default_rng(seed)
    n_days = 252 * n_years
    dates = pd.bdate_range(start, periods=n_days, name="date")
    tickers = [f"C{i:03d}" for i in range(n_stocks)]

    # persistent price trend -> momentum / Kronos stand-in read this from prices
    trend = np.zeros((n_days, n_stocks))
    trend[0] = rng.normal(0, 0.0008, n_stocks)
    for t in range(1, n_days):
        trend[t] = 0.99 * trend[t - 1] + rng.normal(0, 0.0008, n_stocks)
    idio = rng.normal(0, 0.02, (n_days, n_stocks))
    closes = pd.DataFrame(100 * np.cumprod(1 + trend + idio, axis=0), index=dates, columns=tickers)

    panel = {}
    for t in tickers:
        c = closes[t]
        pc = c.shift(1).fillna(c.iloc[0])
        o = pc * (1 + rng.normal(0, 0.002, n_days))
        span = np.abs(rng.normal(0, 0.01, n_days)) * c
        v = rng.lognormal(12, 0.4, n_days)
        panel[t] = pd.DataFrame(
            {"open": o, "high": np.maximum(o, c) + span, "low": np.minimum(o, c) - span,
             "close": c, "volume": v, "amount": v * c},
            index=dates,
        )

    # risk deterioration -> 10-K text change (disjoint alphabets: change tracks mutation)
    d = rng.normal(size=n_stocks)
    mutation = 0.05 + 0.5 / (1 + np.exp(-d))
    base_alpha, new_alpha = list("abcdefghijklm"), list("nopqrstuvwxyz")
    fdates = pd.Timestamp(start) + pd.to_timedelta([365 * y for y in range(n_years)], unit="D")
    rows = []
    for j, t in enumerate(tickers):
        toks = ["".join(rng.choice(base_alpha, 4)) for _ in range(200)]
        for y, fd in enumerate(fdates):
            if y > 0:
                for jj in rng.choice(200, int(200 * mutation[j]), replace=False):
                    toks[jj] = "".join(rng.choice(new_alpha, 6))
            rows.append({"ticker": t, "filing_date": fd, "risk_text": " ".join(toks)})
    filings = pd.DataFrame(rows)

    # forward-return truth: trend component (price-readable) + risk component
    # (fresh info, not in trailing prices) + noise
    ztrend = pd.DataFrame(trend, index=dates, columns=tickers)
    ztrend = ztrend.sub(ztrend.mean(axis=1), axis=0).div(ztrend.std(axis=1), axis=0)
    zrisk = pd.Series(-d, index=tickers)
    zrisk = (zrisk - zrisk.mean()) / zrisk.std()
    fwd = 0.01 * (0.5 * ztrend + 0.5 * zrisk) + pd.DataFrame(
        rng.normal(0, 0.02, (n_days, n_stocks)), index=dates, columns=tickers
    )
    return panel, closes, filings, fwd, dates


def main() -> None:
    panel, closes, filings, fwd, dates = make_combined_dataset()

    signals = {
        "momentum": momentum(closes),
        "tenk_change": build_risk_change_signal(filings, dates, feature="new_word_fraction"),
        "kronos": build_forecast_signal(panel, trailing_return_forecaster, lookback=252, step=5),
    }

    res = incremental_ic(signals, fwd)
    combo = combine_signals(signals, fwd)
    sim = top_k_long_only(combo, fwd, k=10)

    print("=== Combined signal — incremental-IC attribution ===\n")
    print(f"{'signal':<14}{'standalone IC':>16}{'incremental IC':>18}")
    for name, row in res["attribution"].iterrows():
        print(f"{name:<14}{row['standalone_ic']:>+16.4f}{row['incremental_ic']:>+18.4f}")
    print(f"\ncombined IC : {res['combined_ic']:+.4f}")
    print("\nInvestment simulation on the combined signal (long-only top-10)")
    print(f"  AER : {sim['aer']:+.2%}")
    print(f"  IR  : {sim['ir']:+.2f}   (n={sim['n_days']})\n")

    attr = res["attribution"]["incremental_ic"]
    passed = res["combined_ic"] > 0 and attr["tenk_change"] > 0
    print(f"WIRING CHECK: {'PASS' if passed else 'FAIL'} "
          "(combined IC > 0 and the orthogonal 10-K signal adds incremental IC)")
    print("Note: the Kronos stand-in is ~momentum here, so its incremental IC is "
          "small -- incremental IC correctly flags that redundancy.")


if __name__ == "__main__":
    main()
