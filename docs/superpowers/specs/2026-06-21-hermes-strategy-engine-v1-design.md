# Hermès Strategy Engine v1 — Design

**Date:** 2026-06-21
**Status:** approved (design), pending spec review → implementation plan
**Objective (user decision):** risk-adjusted return — maximize Deflated Sharpe / Calmar with controlled drawdowns. NOT raw growth.
**Horizon (user decision):** multi-horizon 1h + 4h; keep per strategy/regime whatever dominates in Deflated Sharpe.
**Constraints:** ~$100 micro-capital, Hyperliquid perps 1x (shorts allowed), BTC/USDC ETH/USDC SOL/USDC, paper-first, 24/7.

## Guiding principles

1. **Fees + over-trading are enemy #1; overfitting is enemy #2.** Every choice favours parameter parsimony and low turnover.
2. **Simple beats clever at low data** (DeMiguel-Garlappi-Uppal: 1/N beats 14 optimized models OOS; 1/N beats HRP in every tested setup). The combination layer starts near-trivial (inverse-vol + regime gating); sophisticated meta-layers (HRP/HMM/stacking ML) are deferred until ≥6-12 months OOS bars exist.
3. **Edge comes from a panel of low-correlation strategies, regime-gated, vol-targeted, judged by DSR/CPCV** — not from a predictive model.
4. **Nothing is promoted to live without surviving CPCV net of fees** with the true trial count in the DSR. Live models are never auto-bootstrapped from an overfit backtest.

## Scope

**In scope (one combined deliverable, per user):** the data foundation, the validation harness, the skill slate, the combination layer, wired so `hermes backtest` evaluates the whole engine on multi-year history and the live agent runs the same skills/combination.

**Out of scope / deferred (shadow or later):** HRP, HMM regimes, ML stacking, cross-sectional 3-asset momentum, overnight seasonality as alpha, SOL-ETH pair, systematic beta-hedge, online-learning (Hedge) in production (shadow only), meta-labeling in production (shadow only).

## Architecture (4 layers + data + validation)

```
DATA FOUNDATION ──► VALIDATION HARNESS (CPCV/purge/embargo/DSR/PBO/trial-registry)
       │                         ▲
       ▼                         │ judges
 ┌───────────────────────────────┴───────────────────────────────┐
 │ L1 SKILLS (deterministic)  L2 REGIME ROUTER  L3 INVERSE-VOL    │
 │  per-symbol + cross-asset   (hysteresis)      BLEND (daily)     │
 │           └──────────── OVERLAY: vol-targeting (HAR-RV) ────────┘
 └─ L4 DSR/CPCV GOVERNANCE: a meta brick activates only if it raises
    the COMBINATION's DSR without raising PBO; else revert to baseline.
```

### Data foundation (`src/hermes/data/history.py`)
- **Paginated CCXT fetch**: loop `fetch_ohlcv(symbol, tf, since, limit)` in batches (≤1000) to assemble **2-4 years** of 1h and 4h bars for BTC/ETH/SOL. Dedup, sort, gap-detect (flag/forward-fill small gaps, error on large).
- **Funding history**: Hyperliquid hourly funding for net-of-funding realism and the crowding filter.
- **Storage**: parquet/csv under `${HERMES_DATA_DIR}/history/<symbol>_<tf>.parquet`; a loader returns aligned DataFrames. Cross-check vs HF `linxy/CryptoCoin` to catch exchange anomalies.
- Target: ≥2 years (confortable 3-4y to cover ≥1 bull/bear/range cycle so the regime detector has "seen" each regime).

### Validation harness (`src/hermes/validation/cpcv.py`, extends existing pbo.py/report.py)
- **CPCV** (Combinatorial Purged CV): N groups, k in test, multi-path Sharpe distribution — primary judge, applied to the **whole combination**, not just isolated skills.
- **Purge + embargo**: purge overlapping triple-barrier labels; embargo ≥ max holding horizon to kill look-ahead.
- **Deflated Sharpe with a TRIAL REGISTRY**: every backtested config (lookbacks × η × regime thresholds × skill thresholds) is logged and counted as a trial; DSR uses the true count. Non-negotiable.
- **PBO** reported each iteration (target ≪ 0.5; current 0.72 must drop).
- Anchored walk-forward kept as a realistic complement; CPCV is the arbiter.

### L1 — Skills (frozen parameters, no fine-tuning)
The current `SkillSignal` interface (per-symbol `fn(df)->SkillSignal`) is kept for per-symbol skills. **New: a cross-asset skill interface** `fn(ctx)->SkillSignal` where `ctx` holds all symbols' aligned OHLCV — needed for pairs. The library evaluates both kinds per tick.

1. **vol_targeting (overlay, refine existing volforecast.py):** size = base × clamp(σ_target/σ̂_HAR-RV, 0.2, 1.0), flat if σ̂ > crisis threshold. Pure sizing, 0 marginal fee. Highest-confidence risk-adjusted lever.
2. **ts_mom_vol_scaled (per-symbol, 4h):** sign of vol-scaled return averaged over **fixed** lookbacks {12,24,48,96} bars; |avg| → confidence; long/short; ATR stop/tp. Replaces/generalizes `trend_follow_ema`. Multi-lookback averaging (not selection) kills cherry-pick.
3. **mean_reversion_regime (per-symbol, 4h):** RSI(2) + Bollinger %B, **active only when regime=range** (Kaufman ER gate) AND price on the mean-reverting side of MA200; else HOLD. Refactor of `mean_reversion_atr` (which loses because it traded in trends).
4. **pairs_eth_btc (cross-asset, 4h, market-neutral):** rolling cointegration test (Engle-Granger) on ETH~BTC; trade the z-score of the spread (enter |z|>2, exit |z|<0.5); both legs; disabled if cointegration p-value fails or half-life too long.
5. **intraday_tod_btc (per-symbol, 1h, BTC only):** time-of-day momentum — the first post-US-open window's return predicts the last window's; 1 trade/day (fee-friendly). BTC only; ETH/SOL speculative (not enabled v1).

### L2 — Regime router (`src/hermes/risk` or `graph`)
Kaufman efficiency-ratio (already built), detected on 4h, with **hysteresis** (enter trend ER>0.4, exit ER<0.3). Routes: trend → {ts_mom_vol_scaled, plus the existing breakout_vol_expansion}; range → {mean_reversion_regime}. Pairs/ToD run regardless (orthogonal). Prevents flip-flop churn. (No new "hardened breakout" skill in v1 — the existing breakout is reused as-is, deduped vs ts_mom via the correlation haircut.)

### L3 — Inverse-vol blending (combination v0)
`weight_i = (1/var_i) / Σ(1/var_j)` over active skills' recent net returns; **daily** reweight (not per-bar) with a dead-band to avoid fee churn. Default combination — robust to covariance mis-specification, estimates only vols.

### L4 — DSR/CPCV governance
A meta brick (Hedge online-learning, meta-labeling) is **shadow** until it beats the inverse-vol baseline across CPCV paths net of fees; only then promoted. The weekly Deflated-Sharpe re-rank (existing) selects the dominant skill subset (capital can't run all legs at once).

## Data flow
- **Backtest:** history loader → for each (symbol, tf) walk bars → skills (per-symbol + cross-asset) → triple-barrier net-of-fees labels → per-skill returns → regime router + inverse-vol blend → combination returns → CPCV/DSR/PBO report. Trial registry updated.
- **Live tick:** market data → regime → skills → router selects active set → inverse-vol weights (from rolling journal) → vol-targeting overlay → existing risk engine (correlation haircut, caps, kill-switch) → order.

## Error handling
- Insufficient history / NaN indicators → skill returns HOLD (never a sized trade on bad data).
- Cointegration unstable → pair auto-disabled (logged), book continues on directional skills.
- Data gaps in history fetch → small gaps forward-filled + flagged; large gaps abort the backtest with a clear error (no silent stitching).
- Funding API failure → funding filter degrades to neutral (no tilt), never blocks.

## Testing (TDD)
- Each skill: pure unit tests on synthetic series that deterministically trigger it (as done for existing skills).
- CPCV/purge/embargo, DSR-with-trials, inverse-vol weights, regime hysteresis: pure unit tests with hand-derived expectations.
- Cross-asset skill interface + pairs z-score logic: unit-tested on synthetic cointegrated/non-cointegrated pairs.
- Data loader: tested on a small fixture (gap detection, dedup).
- Full suite stays green; backtest run is the integration check.

## Anti-overfit guardrails (the 5 traps)
1. **Under-counting trials → inflated DSR.** Mandatory trial registry feeding the DSR.
2. **Parameter optimization.** Freeze all params a priori (lookbacks, RSI/BB thresholds, Donchian N/M, z-score bands); average, don't select.
3. **Correlated skills masquerading as diversification.** One regime gate only; correlation haircut; dedup ts_mom vs breakout.
4. **Look-ahead via meta-layer.** Purge+embargo; causal detectors only; all meta in shadow until CPCV-proven.
5. **Short sample treated as sufficient.** Target 2-4y; require a brick to beat baseline across CPCV paths, not an in-sample peak. DeMiguel reflex: if sophisticated doesn't beat naïve OOS, keep naïve.

## Rollout / success criteria
- Backtest the full engine on 2-4y history. **Success = combination Deflated Sharpe credibly > 0 (with true trial count) AND PBO ≪ 0.5 AND drawdown within target**, net of fees+funding.
- If credible: bootstrap live calibrator/meta-labeler from the (now longer, validated) backtest; keep shadow→promote discipline.
- If not: iterate on the slate; never promote on a weak/overfit backtest. The engine still runs in paper and accumulates real data in parallel.
