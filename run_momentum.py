"""The critical gate: reproduce 12-1 momentum through the harness.

If momentum does not show a positive information coefficient and a monotone
quantile profile here, the harness is broken -- fix it before trusting any
AI-derived signal. Run: `python run_momentum.py`
"""

from __future__ import annotations

from harness.backtest import backtest
from harness.data import generate_synthetic_prices
from harness.metrics import forward_returns, ic_summary, quantile_returns, rank_ic
from harness.signals import momentum


def main() -> None:
    prices = generate_synthetic_prices(n_stocks=200, n_days=900, seed=0)
    signal = momentum(prices)
    fwd = forward_returns(prices, horizon=1, gap=1)

    ic = rank_ic(signal, fwd)
    summary = ic_summary(ic)
    quantiles = quantile_returns(signal, fwd, q=10)
    bt = backtest(signal, prices, quantile=10, cost_bps=5.0)

    print("=== 12-1 Momentum — harness gate ===\n")
    print("Information coefficient")
    print(f"  mean IC        : {summary['mean_ic']:+.4f}")
    print(f"  IR (annualized): {summary['ir_annualized']:+.2f}")
    print(f"  hit rate       : {summary['hit_rate']:.1%}")
    print(f"  t-stat         : {summary['t_stat']:+.2f}  (n={summary['n_days']})\n")

    print("Decile mean forward return (0=low signal .. 9=high)")
    for label, value in quantiles.items():
        print(f"  D{label}: {value:+.5f}")
    print()

    print("Long/short backtest (net of 5bps one-way cost)")
    print(f"  Sharpe         : {bt['sharpe']:+.2f}")
    print(f"  ann. return    : {bt['ann_return']:+.2%}")
    print(f"  ann. vol       : {bt['ann_vol']:.2%}")
    print(f"  max drawdown   : {bt['max_drawdown']:.2%}")
    print(f"  avg turnover   : {bt['avg_daily_turnover']:.2f}\n")

    spread = quantiles.iloc[-1] - quantiles.iloc[0]
    passed = summary["mean_ic"] > 0 and spread > 0
    print(f"GATE: {'PASS' if passed else 'FAIL'} "
          f"(mean IC > 0 and top-minus-bottom decile spread > 0)")


if __name__ == "__main__":
    main()
