"""Demonstrate the 10-K risk-factor change signal end-to-end on a planted world.

The dataset has a known relationship (more risk-text rewritten -> lower forward
returns), so the change signal should show a clearly negative rank IC -- i.e.
you would short the big changers. This validates the plumbing exactly as
run_momentum.py validates the price side.

For real use, replace make_planted_dataset with EDGAR filings:
    from harness.edgar import get_cik, list_10k_filings, fetch_document, extract_risk_factors
and prices/forward returns from harness.sources.

Run: `python run_text_signal.py`
"""

from __future__ import annotations

from harness.metrics import ic_summary, rank_ic
from harness.text_signal import build_risk_change_signal, make_planted_dataset


def main() -> None:
    filings, fwd, calendar = make_planted_dataset(n_stocks=80, n_years=4, seed=0)
    signal = build_risk_change_signal(filings, calendar, feature="new_word_fraction")

    ic = rank_ic(signal, fwd)
    summary = ic_summary(ic)

    print("=== 10-K risk-factor change signal (planted world) ===\n")
    print(f"Filings           : {len(filings)} rows, {filings['ticker'].nunique()} tickers")
    print(f"Signal frame      : {signal.shape[0]} days x {signal.shape[1]} tickers "
          f"(availability-dated, forward-filled)\n")
    print("Rank IC of raw change vs forward return")
    print(f"  mean IC : {summary['mean_ic']:+.4f}   (expected negative: big changers underperform)")
    print(f"  IR      : {summary['ir_annualized']:+.2f}")
    print(f"  t-stat  : {summary['t_stat']:+.2f}  (n={summary['n_days']})\n")

    passed = summary["mean_ic"] < 0
    print(f"WIRING CHECK: {'PASS' if passed else 'FAIL'} "
          "(negative IC -> short the firms that rewrote risk factors the most)")
    print("\nOn real data the sign is confirmed empirically here, not assumed; "
          "flip the score (use -signal) to trade it long-top.")


if __name__ == "__main__":
    main()
