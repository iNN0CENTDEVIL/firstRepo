# AI-Native Equities Fund — Strategy & Build Plan

Working document. Daily-frequency, cross-sectional US equity long/short, with AI
concentrated in signal generation from unstructured data. Living plan — edit freely.

---

## 1. Thesis & positioning

- **Frequency: daily.** Fast enough that unstructured-data signals have room to work,
  slow enough to avoid competing on latency/colocation against HFT incumbents.
- **Edge source: data (#1), with a behavioral/structural tilt (#4).** AI is a
  *multiplier* on turning messy unstructured information into structured signal
  cheaply — not a substitute for having a real edge.
- **AI-native = small research org, same rigor.** Agents widen the funnel of testable
  ideas and cut research cost. The quant discipline (point-in-time data, out-of-sample
  validation, risk limits, attribution) stays exactly as strict.

**Non-goal:** LLM agents making live discretionary trades. They hallucinate, are
non-deterministic, and can't be backtested. Agents live in research; execution is
deterministic code.

---

## 2. Strategy archetype

Low-frequency quant equity long/short:

- **Universe:** liquid US names, ~top 1,500–3,000 by dollar volume. No microcaps early
  (borrow, capacity, data quality).
- **Daily:** cross-sectionally rank every name by expected forward return.
- **Portfolio:** long top decile, short bottom decile; dollar-neutral and
  sector-neutral — bet on *relative* ranking, not market/sector direction.
- **Signal:** baseline factors (value, momentum, quality, low-vol) + AI alpha overlay.

The whole game reduces to producing a good `N stocks × T days` matrix of scores.
Everything else is plumbing and risk control.

---

## 3. Where AI plugs in (for daily frequency)

Signal generation from unstructured data — not the trading loop:

- **Filings diffing** — YoY change in 10-K/10-Q risk factors, MD&A tone, guidance
  language. *Change* is more alpha-rich than *level*.
- **Earnings-call NLP** — tone, hedging, evasiveness on analyst questions, guidance shifts.
- **8-K / news event classification** at scale.
- **Research automation** — agents that propose a hypothesis, run it through the standard
  eval harness, and log the result.

Prefer **mechanical** features (did the risk-factor section grow? was a going-concern
clause removed?) over **interpretive** ones ("how bullish does this sound") — the latter
are far more exposed to LLM hindsight leakage (see §4).

---

## 4. Deep dive — Point-in-time (PIT) data architecture

The #1 correctness problem. If data is not PIT, every backtest result is fiction.

### 4.1 The three leakage sources

1. **Restatement / as-reported leakage.** Fundamentals get revised. A backtest must see
   the value *as it was reported on that date*, not today's restated figure. Requires a
   PIT fundamentals source (e.g., Sharadar SF1's `ARQ`/`ART` dimensions, or Compustat PIT).
2. **Reporting-lag leakage.** A fiscal quarter ends March 31 but is filed ~45 days later.
   You may only "know" Q1 numbers on the *filing date*, not the period-end date. Index
   every fundamental by its **availability date**, never its period date.
3. **LLM hindsight leakage (AI-specific).** The model was trained on data *after* your
   backtest date. An LLM reading a 2019 filing already knows 2020–2024 happened. Any
   sentiment/judgment it emits is contaminated. Mitigations:
   - Favor mechanical/structural features (diffs, counts, presence/absence of clauses).
   - Feed the model *only* the document text, never let it use world knowledge about
     outcomes ("based solely on this text…").
   - Where possible, validate a feature's IC is stable across pre- and post-training-cutoff
     periods — a big gap is a leakage red flag.

### 4.2 Data layers

| Layer | What | Indie-budget source |
|-------|------|--------------------|
| Prices (adj) | Daily OHLCV, splits/divs, **incl. delisted** | Norgate, Polygon, EODHD |
| Fundamentals | PIT, availability-dated | Sharadar SF1 |
| Filings text | 10-K/Q, 8-K | SEC EDGAR (free, full-text) |
| Transcripts | Earnings calls | Paid API (later) |
| Universe/meta | Sector, listing status, index membership | Sharadar tickers + your own PIT snapshot |

### 4.3 The one rule that prevents most bugs

Store every datum with an **`available_at` timestamp** and, at backtest time, filter with
`available_at <= as_of_date`. Build a single accessor `get_features(as_of_date)` that
enforces this — never let strategy code touch raw tables directly. Survivorship bias dies
here too: the universe on a past date must include names that later delisted.

---

## 5. Deep dive — Evaluation harness

Build this **before** any alpha. A bad harness silently lies for months. Input: a daily
score matrix. Output: an honest verdict.

### 5.1 Core metrics

- **Rank IC** — Spearman correlation between today's scores and forward returns, per day.
  Report mean IC, IC stdev, and **IR = mean/stdev × √(periods/yr)**. This is the primary
  signal-quality number (Grinold–Kahn fundamental law).
- **Quantile spread** — decile-sorted forward returns; monotonicity matters as much as the
  top-minus-bottom spread. Non-monotone deciles = fragile signal.
- **Turnover** — daily name turnover; drives cost and capacity.
- **Net Sharpe** — after realistic costs (see §5.3), dollar- and sector-neutralized.
- **Max drawdown & time-under-water.**

`Alphalens` computes most of this out of the box — use it, don't hand-roll yet. For the
backtest itself, a **vectorized pandas/numpy cross-sectional** approach beats event-driven
frameworks (backtrader/zipline) at daily frequency: simpler, faster, fewer hidden bugs.

### 5.2 The critical gate — reproduce a known result

Implement plain **12-1 momentum** and confirm you recover roughly the published long/short
Sharpe. **If you can't reproduce momentum, the harness is broken and every later result is
garbage.** Do not build a single AI feature until this passes. Highest-value task in month 1.

### 5.3 Realistic frictions

- Commissions + spread (half-spread per side, wider for less-liquid names).
- Slippage/impact scaled by participation vs. ADV.
- Borrow cost on shorts; hard-to-borrow exclusions.
- Trade at next-day open/VWAP after a signal computed on close — never same-bar close.

### 5.4 Anti-overfitting discipline (AI makes this worse)

Agents let you test thousands of hypotheses = thousands of chances to fool yourself.

- **Track your trial count.** Every tested variant counts against your significance.
- **Deflated Sharpe Ratio** (Bailey/López de Prado) to discount for number of trials.
- **Purged, embargoed cross-validation** — no train/test leakage across overlapping
  forward-return windows.
- **Hold out a final period you never look at** until the very end.
- Budget "researcher degrees of freedom" — pre-register the hypothesis before you test it.

---

## 6. 90-day plan

1. **Weeks 1–2 — Data foundation.** Survivorship-bias-free prices (incl. delisted) + PIT
   fundamentals. Build the `available_at`-enforced `get_features(as_of_date)` accessor and
   the PIT universe.
2. **Weeks 2–4 — Eval harness.** Rank IC, quantile spread, turnover, net Sharpe, drawdown
   via Alphalens + a vectorized cross-sectional backtest.
3. **Week 4 — Critical gate.** Reproduce 12-1 momentum. Must pass before proceeding.
4. **Weeks 5–10 — First AI signal.** One clean idea (e.g., YoY 10-K risk-factor change)
   through the *same* harness, honestly compared to the momentum baseline. Beat baseline or kill it.
5. **Weeks 10+ — Paper → tiny live.** Paper trading lies (perfect fills). Put small real
   money on to see actual fills and borrow before trusting the numbers.

---

## 7. Reading

- **López de Prado — _Advances in Financial Machine Learning_** — overfitting, purged CV,
  deflated Sharpe. Core defense for §5.4.
- **Grinold & Kahn — _Active Portfolio Management_** — IC / IR / fundamental law; the basis
  for the cross-sectional approach.
- **Ilmanen — _Expected Returns_** — factor breadth.
- **Qian, Hua & Sorensen — _Quantitative Equity Portfolio Management_** — mechanics.

---

## 8. Open decisions

- Fund vs. tools-for-funds business model (be honest which one).
- Broker/prime relationship and entity/regulatory stack (6+ months lead time).
- Own-capital track-record phase before raising outside money.
