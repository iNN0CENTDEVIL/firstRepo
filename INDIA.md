# Running on the Indian market (NSE/BSE) with live data

The harness is market-agnostic at the `prices` / OHLCV seam, so going live on
Indian equities means writing one data adapter, not changing the engine. This
repo ships a Zerodha Kite Connect adapter (`harness/kite_source.py`).

## 1. Get data via Kite Connect

```python
from harness.kite_source import KiteOHLCVLoader

loader = KiteOHLCVLoader(api_key="...", access_token="...")   # from your Kite app
universe = ["RELIANCE", "TCS", "HDFCBANK", ...]               # e.g. Nifty 500 symbols
panel  = loader.load_panel(universe, "2015-01-01", "2025-01-01")  # {symbol: OHLCV}
closes = loader.load_prices(universe, "2015-01-01", "2025-01-01") # wide close frame
```

`kiteconnect` is an optional dependency (`pip install kiteconnect`); the module
imports without it. Respect Kite's historical rate limit (~3 requests/sec) across
a large universe. Live fetch cannot run in a registry-only sandbox — run this in
your own environment.

## 2. Run the existing pipeline

```python
from harness.signals import momentum
from harness.kronos_signal import build_forecast_signal_batched, KronosForecaster
from harness.metrics import forward_returns, rank_ic
from harness.simulation import top_k_long_only

fwd    = forward_returns(closes, horizon=1, gap=1)
mom    = momentum(closes)
kronos = build_forecast_signal_batched(panel, KronosForecaster().batch, lookback=252)

print(rank_ic(mom, fwd).mean())
print(top_k_long_only(mom, fwd, k=50))       # long-only: India-safe (see below)
```

## 3. India-specific things that change the analysis

1. **Corporate actions — adjust prices first.** Kite historical candles are
   *unadjusted* for splits/bonuses/rights (all common in India). Apply
   adjustments from a corporate-actions source before computing any price signal,
   or momentum/forecast will see phantom jumps.
2. **Shorting — go long-only on cash.** Retail cannot hold overnight shorts in
   cash equities. The dollar-neutral L/S backtest (`harness/backtest.py`) is not
   tradable on cash; use the long-only top-k **AER/IR** simulation
   (`harness/simulation.py`), or take shorts via single-stock futures (the ~180
   F&O names) or SLB.
3. **Trading calendar.** With real Kite data the panel index is already NSE
   sessions, so forward returns and signals use the right calendar automatically.
   The one exception is the Kronos `y_timestamp` horizon — use
   `kite_source.nse_future_sessions` (install `exchange_calendars` for the true
   NSE holiday calendar) instead of the business-day default.
4. **Costs are higher.** STT, stamp duty, exchange fees, GST, and brokerage add
   up (delivery STT alone ≈ 0.1%). Raise `cost_bps` in the backtest/simulation
   well above a US assumption.
5. **Universe & liquidity.** Restrict to a liquid set (Nifty 500); skip
   micro-caps. Upper/lower **circuit** days act like halts — the forecast
   builders' ragged-universe handling already skips windows with gaps.
6. **Fundamentals / the 10-K signal.** There is no EDGAR equivalent. The
   `text_signal` module needs an India source (BSE/NSE annual reports and
   corporate announcements, or a vendor like Screener.in / CMIE Prowess) before
   it can run; drop it initially and use momentum + Kronos.
7. **Kronos transfer.** Kronos was pretrained on 45 global exchanges (very likely
   including NSE/BSE), so zero-shot OHLCV forecasting should transfer — a good
   place to test whether its incremental IC over momentum is genuinely positive
   on Indian data.

## 4. Validate before believing anything

Run the overfitting checks (`run_validation.py` pattern: purged/embargoed CV +
deflated Sharpe) on the Indian signals, and paper-trade before committing
capital. Free feeds (`yfinance` `.NS`/`.BO`) are survivorship-biased — fine for a
first look, not for go/no-go.
