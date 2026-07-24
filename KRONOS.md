# Kronos — analysis & integration notes

Paper: *Kronos: A Foundation Model for the Language of Financial Markets*
(Shi, Fu, Chen, Zhao, Xu, Zhang, Li — Tsinghua, arXiv:2508.02739, Aug 2025).
Code/weights: github.com/shiyu-coder/Kronos, HuggingFace `NeoQuasar/Kronos-*`.

## Summary

Kronos is a foundation model for financial K-line (candlestick) data — OHLCVA:
Open, High, Low, Close, Volume, Amount. Two stages:

1. **Tokenizer.** A Transformer autoencoder with **Binary Spherical Quantization**
   discretizes each continuous OHLCVA bar into a *hierarchical* token = a coarse
   subtoken + a fine subtoken (k=20 bits split 10/10). Coarse captures principal
   structure; fine encodes the residual. This shrinks a 2^20 vocabulary into two
   2^10 predictions.
2. **Autoregressive Transformer.** A decoder-only model predicts the next bar's
   coarse subtoken, then the fine subtoken conditioned on the coarse (chain rule),
   causally over history. Pretrained on **12B K-lines from 45 exchanges**; three
   sizes (24.7M / 102.3M / 499.2M params), context length 512.

**Inference & test-time scaling.** Forecasting is autoregressive generation with
temperature + top-p sampling. Because it's generative, you can draw **N stochastic
future trajectories and average the decoded values** ("Monte Carlo rollouts") to
cut variance — IC/RankIC rise monotonically with N. No retraining.

**Evaluation.** IC and **RankIC** for price/return forecasting; MAE/R² for realized
volatility; discriminative + TSTR scores for synthetic generation; and an
**investment simulation** — a long-only **top-k** portfolio ranked by the model's
signal, scored by **Annualized Excess Return (AER)** and **Information Ratio (IR)**.
Kronos reports +93% RankIC over the best time-series foundation-model baseline.

**Ablations that matter for us.** Discrete (tokenized) beats continuous regression;
sequential coarse→fine beats parallel subtokens; larger vocabulary helps; more
inference samples help.

## Why it's relevant here

- Its headline metric is **RankIC** — exactly what `harness/metrics.py` already
  computes. The paper is independent validation of this project's evaluation core.
- Kronos is, for our purposes, a **return-forecast signal generator**: OHLCV history
  in → forward-return forecast out → rank cross-sectionally → feed the existing
  metrics / backtest / validation loop. It slots into the `signal` seam alongside
  momentum and the 10-K text signal.

## What was integrated

1. **Investment simulation (`harness/simulation.py`).** `top_k_long_only` builds the
   paper's long-only top-k portfolio vs. an equal-weight-universe benchmark and
   reports **AER** and **IR**. Complements the existing dollar-neutral L/S backtest
   with a long-only, benchmark-relative view. Fully offline, tested.
2. **Test-time scaling (`ensemble_forecast` in `harness/kronos_signal.py`).** Averages
   N stochastic forecasts — the model-agnostic variance-reduction trick from the
   paper. Tested.
3. **Kronos forecast signal (`harness/kronos_signal.py`).**
   - `KronosForecaster` — a documented adapter around the pretrained Kronos model
     (lazy `torch`/`kronos` import) that forecasts a per-stock forward return via
     ensembled rollouts.
   - `build_forecast_signal` — rolls *any* `ReturnForecaster` across dates into an
     availability-dated cross-sectional signal frame, with **no look-ahead** (each
     date uses only its trailing OHLCV window).
   - `build_forecast_signal_batched` + `KronosForecaster.batch` — score the whole
     universe for a date in one `Kronos.predict_batch` call. Per-stock calls are
     impractical on a real GPU model at universe scale; this is the shape to use
     live. Verified to match the per-stock builder exactly.
   - `trailing_return_forecaster` — a transparent offline baseline (12-1 momentum
     expressed through the forecaster interface) that stands in for Kronos so the
     whole pipeline runs and self-validates on synthetic OHLCV.

## What was NOT integrated, and why

- **The actual pretrained model is not run here.** It needs a HuggingFace download
  and a ~0.5B-param torch model; this sandbox's egress is limited to package
  registries. `KronosForecaster` is written against the **confirmed** Kronos API
  (from github.com/shiyu-coder/Kronos): `KronosPredictor(model, tokenizer,
  device=None, max_context=512)` and `predict(df, x_timestamp, y_timestamp,
  pred_len, T=1.0, top_k=0, top_p=0.9, sample_count=1, verbose=True)`, where
  `sample_count` averages that many rollouts internally (test-time scaling) and
  the input df needs at least `[open, high, low, close]`.
- **The tokenizer / pretraining is not reimplemented.** BSQ tokenization + a 12B-row
  pretraining run is the paper's core research contribution and out of scope; the
  point of integration is to *consume* the released model as a signal, not rebuild it.
- **No semantic/LLM feature leakage guard is needed here** (Kronos sees only past
  OHLCV, not future-aware text), but the usual point-in-time discipline still applies:
  `build_forecast_signal` forecasts strictly from trailing windows.

## How to run it for real (networked env with a GPU)

```python
# pip install torch and the kronos package from github.com/shiyu-coder/Kronos
from harness.kronos_signal import KronosForecaster, build_forecast_signal
from harness.simulation import top_k_long_only
from harness.metrics import forward_returns, rank_ic

forecaster = KronosForecaster(model_id="NeoQuasar/Kronos-base", sample_count=20)
signal = build_forecast_signal(ohlcv_panel, forecaster, lookback=400)
fwd = forward_returns(close_panel, horizon=1, gap=1)
print(rank_ic(signal, fwd).mean())
print(top_k_long_only(signal, fwd, k=50))
```
