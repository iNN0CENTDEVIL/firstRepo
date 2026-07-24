"""Demonstrate the Kronos-style forecast-signal pipeline end-to-end.

The real Kronos model can't run in this sandbox (needs a HuggingFace download and
a torch model), so this uses `trailing_return_forecaster` as an offline stand-in.
It exercises the exact wiring a real `KronosForecaster` would use: build a
cross-sectional forecast signal, score it with RankIC, and run the paper's
long-only top-k investment simulation (AER / IR).

Run: `python run_kronos.py`
"""

from __future__ import annotations

from harness.data import close_panel, generate_synthetic_ohlcv
from harness.metrics import forward_returns, ic_summary, rank_ic
from harness.kronos_signal import build_forecast_signal, trailing_return_forecaster
from harness.simulation import top_k_long_only


def main() -> None:
    panel = generate_synthetic_ohlcv(n_stocks=60, n_days=900, seed=0)
    closes = close_panel(panel)

    # step=5: forecast weekly and forward-fill (keeps the demo fast).
    signal = build_forecast_signal(panel, trailing_return_forecaster, lookback=252, step=5)
    fwd = forward_returns(closes, horizon=1, gap=1)

    ic = ic_summary(rank_ic(signal, fwd))
    sim = top_k_long_only(signal, fwd, k=10)

    print("=== Kronos-style forecast signal (offline baseline stand-in) ===\n")
    print(f"Signal frame : {signal.shape[0]} days x {signal.shape[1]} tickers\n")
    print("Forecast quality")
    print(f"  mean RankIC : {ic['mean_ic']:+.4f}")
    print(f"  IR          : {ic['ir_annualized']:+.2f}")
    print(f"  t-stat      : {ic['t_stat']:+.2f}  (n={ic['n_days']})\n")
    print("Investment simulation — long-only top-10 vs equal-weight universe")
    print(f"  AER (annualized excess return): {sim['aer']:+.2%}")
    print(f"  IR  (information ratio)       : {sim['ir']:+.2f}")
    print(f"  hit rate                      : {sim['hit_rate']:.1%}  (n={sim['n_days']})\n")

    passed = ic["mean_ic"] > 0 and sim["aer"] > 0
    print(f"WIRING CHECK: {'PASS' if passed else 'FAIL'} "
          "(positive RankIC and positive AER)")
    print("\nSwap trailing_return_forecaster for KronosForecaster in a networked, "
          "torch-enabled environment to run the real model.")


if __name__ == "__main__":
    main()
