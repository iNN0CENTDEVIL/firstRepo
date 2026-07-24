"""Cross-sectional daily-equity evaluation harness.

Input to everything downstream is a daily score matrix (dates x tickers).
Swap the synthetic data source in `data.py` for a real point-in-time source
(Sharadar/Norgate) behind the same `prices` DataFrame interface.
"""
