"""Run the anti-overfitting checks on the momentum signal.

Out-of-sample IC via purged/embargoed CV, plus the probabilistic and deflated
Sharpe ratios. The DSR inputs (`n_trials`, `sr_trials_std`) are illustrative
here -- in real use they come from your research log: how many configurations
you tested and how dispersed their Sharpes were.

Uses the synthetic source (live market data is blocked in this sandbox). To run
on a real vendor extract instead:

    from harness.sources import load_prices_from_csv
    prices = load_prices_from_csv("your_extract.csv")

Run: `python run_validation.py`
"""

from __future__ import annotations

from harness.backtest import backtest
from harness.data import generate_synthetic_prices
from harness.metrics import forward_returns
from harness.signals import momentum
from harness.validation import (
    PurgedKFold,
    cross_validated_ic,
    deflated_sharpe_ratio,
    probabilistic_sharpe_ratio,
)

# Illustrative research-log inputs -- replace with your real trial count and the
# dispersion (std) of annualized Sharpes across the configurations you tested.
N_TRIALS = 50
SR_TRIALS_STD = 0.5


def main() -> None:
    prices = generate_synthetic_prices(n_stocks=200, n_days=900, seed=0)
    signal = momentum(prices)
    fwd = forward_returns(prices, horizon=1, gap=1)

    cv = PurgedKFold(n_splits=5, horizon=1, embargo=0.01)
    oos_ic = cross_validated_ic(signal, fwd, cv)

    bt = backtest(signal, prices, quantile=10, cost_bps=5.0)
    returns = bt["returns"]
    psr = probabilistic_sharpe_ratio(returns, sr_benchmark=0.0)
    dsr = deflated_sharpe_ratio(returns, n_trials=N_TRIALS, sr_trials_std=SR_TRIALS_STD)

    print("=== Anti-overfitting checks — 12-1 momentum ===\n")
    print(f"Purged/embargoed CV out-of-sample IC ({cv.n_splits} folds)")
    for i, value in enumerate(oos_ic):
        print(f"  fold {i}: {value:+.4f}")
    print(f"  mean  : {oos_ic.mean():+.4f}   std: {oos_ic.std():.4f}\n")

    print(f"Backtest Sharpe (net)         : {bt['sharpe']:+.2f}")
    print(f"Probabilistic Sharpe (vs 0)   : {psr:.3f}   P(true SR > 0)")
    print(f"Deflated Sharpe               : {dsr:.3f}   "
          f"P(true SR > max of {N_TRIALS} trials, sr_std={SR_TRIALS_STD})")
    print()
    print("Read: PSR near 1 = track record long/clean enough to trust the sign.")
    print("      DSR near 1 = survives multiple-testing; near 0.5 = likely luck.")


if __name__ == "__main__":
    main()
