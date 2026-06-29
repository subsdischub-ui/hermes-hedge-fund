# Stream 2 v-next — cross-sectional market-neutral sleeves · FINDINGS

**Date:** 2026-06-29 (UTC) · **Branch:** `round2/s2-xsec` · **Mode:** research only, shadow size 0, nothing live.
**Lake:** frozen DuckDB (read-only), 23 Binance USDT-perps 4h + deep native funding, 2019-09 → 2026-06.
**Code:** `src/hermes/research/s2_xsec.py` (22 TDD tests, `tests/research/test_s2_xsec*.py`) · driver
`research_streams/S2/run_s2_xsec.py` → `results_s2_xsec.json` · trials in `TRIAL_REGISTRY.md`.

## TL;DR — HONEST NEGATIVE on the pre-registered family + a governance flag

- **No new cross-sectional sleeve survives.** The pre-registered family (2A funding, 2D vol-dispersion,
  2B OI-growth, 2.3 cross-venue dispersion) is a **clean, dated negative**. None clears the MC random-sign
  floor; the dominant killer is **two-leg turnover cost** (a daily-rebalanced XS book turns over ≈100%/day).
- **Governance flag (first-class):** the lab's only live edge, `funding_squeeze_continuation`, re-tested at
  4h with an explicit bull/bear split and **cluster-aware** nulls, is **SUGGESTIVE-but-UNCONFIRMED in bull
  and NOT-ESTABLISHED in bear.** It does NOT cleanly fail in bear (the sign does not flip negative → not the
  pure bull-beta we feared), but it also does **not** clear a deflated/clustered significance bar. Treat the
  live edge as **weakly supported, not robustly confirmed** — consistent with [[hermes-s3-universe]] (the 4h
  squeeze did not generalize to 1d).
- **Honest exploratory lead (NOT a result):** the INVERTED, SLOW XS funding signal (fade high funding-z,
  bi-weekly) shows a positive but post-hoc-selected tilt (true annual Sharpe ≈ +0.58). Logged for a future
  *pre-registered + deflated* test; not claimable now.

A clean negative is a valid deliverable. What would reopen each frontier is stated per-sleeve below.

---

## L0 — leak-injection on the engine (TEST #1, non-negotiable) — **PASS** [VÉRIFIÉ]

Before any sleeve ran, the new cross-sectional backtester was pinned by a look-ahead battery
(`tests/research/test_s2_xsec.py`):
- **Oracle vs lagged:** a signal that IS the forward return → huge edge (gross mean > +1%/rebal); the SAME
  signal lagged one stride (past-only) → collapses to ~0. Oracle dwarfs honest by >5×.
- **Truncation invariance:** removing future bars never changes any past sleeve return (bit-identical).
- **Cost/funding:** turnover charged on both legs; funding carry summed **once per settlement** from
  `binance_funding_native` (the forward-filled `binance_funding_on_4h` rate sums to 2× native → not used for cost).

An independent 4-agent adversarial review (419k tokens) **found no engine leak**; it did find — and we FIXED —
a same-day look-ahead in the *driver's* regime tag (the daily label was built from each day's 20:00 close and
applied to that day's 00:00 bar). It is now a tested, one-day-lagged library function `daily_regime_on_grid`
(2 regression tests). Correcting it **deflated** the squeeze bull excess (+203→+217bps after re-baselining the
drift) — i.e. it never inflated the headline.

---

## 2A — funding cross-sectional [PRIMARY] — **NEGATIVE (KILL)** [VÉRIFIÉ]

*Long the high-funding-z third, short the low, dollar-neutral; 23 perps; daily rebalance on the 4h grid;
funding-z over a 180-settlement trailing window; carry applied per settlement.*

| metric | value |
|---|---|
| net mean / rebalance | **−27.0 bps** |
| decomposition | gross(price) **−8.7** · carry **−1.45** · **cost −16.85** bps |
| gross-only ann Sharpe | **−1.38** (the sign is wrong-way) |
| MC random-sign p | **1.000** (the realised sign is the *worst* under flips) |
| ρ vs BTC daily | −0.03 (genuinely market-neutral) |
| LOO Δ-Sharpe (book) | **−0.109** (drags the book down) · DSR (book count 1869) ≈ 0 |

**KILL on two independent counts.** (1) The pre-registered sign is wrong-way: high-funding-z names
*underperform* peers next day — cross-sectional funding **mean-reverts**, it does not continue (the live
*time-series* squeeze and the *cross-sectional* spread are different objects). (2) Even at zero cost it loses
(gross ann Sharpe −1.38), and the honest two-leg cost (−16.85 bps/day) buries it regardless of sign: a daily
funding-z reshuffle turns over ≈100%/day. Cross-regime: loses in both bull (Sharpe −0.23) and bear (−0.22).
**Reopen:** a *pre-registered* test of the INVERTED sign at a SLOW (≥weekly) cadence to escape the cost wall —
see "Exploratory lead" below — under fresh out-of-sample + deflation, not a post-hoc claim.

## 2D — realised-vol dispersion cross-sectional [AVAILABLE] — **NEGATIVE (no edge)** [VÉRIFIÉ]

*Long high realised vol (42×4h), short low; 23 perps; daily.*

net −1.84 bps/rebal · gross +1.78, carry +0.15, **cost −3.77** · gross ann Sharpe **+0.20** ·
**MC random-sign p 0.311** · ρ_BTC +0.01. The price spread has a faint positive whiff but it is **not
significant** and is eaten by cost. Clean no-edge. **Reopen:** a lower-turnover construction (vol ranks are
stable, so cost is smaller than 2A) with a genuinely orthogonal *conditioning* — but the raw dispersion
carries no significant cross-sectional return here.

## 2B — OI-growth cross-sectional [SECONDARY, thin] — **NEGATIVE (data-gated / under-powered)** [VÉRIFIÉ]

*Long high OI-acceleration (14d z), short low; **7 coins** (the only CoinGlass-OI symbols: ARB, AVAX, BTC,
ETH, SOL, SUI, WIF), 1d; terciles → 2 long / 2 short.*

net −24.33 bps/rebal · gross +1.59, carry +0.12, **cost −26.04** · gross ann Sharpe **+0.13** ·
**MC random-sign p 0.377**. Cost-killed on a skeletal universe; gross is noise. **Under-powered, not a
clean refutation** (often only 4–5 names available; survivorship — the 7 are current survivors). **Reopen:**
an S0 ingest of deep OI for the full 23-name 4h universe (CoinGlass candlesticks exist for more pairs) would
give the width for a real OI-acceleration XS; until then it is data-gated.

## 2.3 — cross-venue funding dispersion [OPTIONAL, unblocked by S0] — **NEGATIVE** [VÉRIFIÉ]

*Long high cross-venue funding-std, short low; 5 coins × 4 venues (Binance/OKX/Bybit/Bitget), 1d.*
ran, n=2063, ann Sharpe **−0.44**, MC random-sign p 0.05 (negative-Sharpe → wrong-sign). Thin 5-coin
universe. The S0 cross-venue substrate is now in the lake (supersedes the old "Binance-only" data-gate of
[[hermes-s2-positioning]]), but the dispersion signal carries no XS edge on these 5 names. **Reopen:** the
deeper cross-venue panel (more coins) S0 scouted, plus a dispersion *level* signal (not just rank).

## Orthogonality (daily-PnL ρ, triple-view) [VÉRIFIÉ]

All pairwise |ρ| **< 0.30** (2A·2D −0.19, 2A·squeeze +0.17, 2D·squeeze +0.11, all others ≈0). The family IS
mutually orthogonal and orthogonal to the squeeze — the construction works — but orthogonality is **moot**
because no sleeve has positive edge to contribute (LOO Δ-Sharpe: 2A −0.109; 2D +0.014, 2B +0.005 are
variance-reduction artifacts on negative-Sharpe streams, not edge).

---

## BONUS (first-class) — funding_squeeze 4h, bull vs bear [VÉRIFIÉ, with caveats]

Re-test of the live `funding_squeeze_continuation` (|funding-z|≥2 → long, hold 24×4h=96h, no-overlap) on
the deep 4h history, split by BTC >MA200 / <MA200 (causal, one-day-lagged regime), net of carry + cost,
judged by a **regime-matched MC random-entry null** (the mandate's bull-beta KILL) and a **cluster-aware**
macro-event test (squeezes co-fire across alts; per-trade counts overstate independence).

| | n trades | macro-events | real mean | drift (random-entry) | excess | i.i.d. p | macro-event mean (t) |
|---|---|---|---|---|---|---|---|
| **bull** (all 23) | 927 | 113 | +265 bps | +48 bps | **+217 bps** | 0.0001 | **+177 bps (t=1.50)** |
| **bear** (all 23) | 35 | 21 | +347 bps | −18 bps | +365 bps | 0.017 | **+130 bps (t=1.23)** |
| bull (BTC/ETH/SOL) | 141 | — | +265 | +86 | +179 | 0.013 | — |
| bear (BTC/ETH/SOL) | 6 | — | +257 | −19 | +275 | **0.20 (NS)** | — |

**Verdict — SUGGESTIVE in bull, NOT ESTABLISHED in bear; NOT "confirmed":**
- **Bull:** beats the regime-matched random-entry drift on the i.i.d. null (excess +217 bps, p=0.0001) and on
  a daily-block null (review: t≈2.8) — so it is **more than pure bull-beta**. BUT it is **weak and
  right-tail/survivorship-flattered**: per-trade **median only +25 bps** (mean/median ≈10×), and under
  4-day **macro-event clustering the excess is +177 bps at only t=1.50 (not significant)**. The i.i.d. p is
  anti-conservative (alts co-fire). → suggestive, not confirmed.
- **Bear:** the sign does **not flip negative** (so NOT the feared pure bull-beta), but it is **not
  established**: 35 trades collapse to 21 macro-events (macro mean +130 bps, **t=1.23, NS**); the i.i.d.
  +365 bps is episode-driven (leave-one-event-out range [+58, +403] bps; deep majors n=6, p=0.20). 35 events
  cannot carry "survives bear".
- **DSR honesty:** no per-trade DSR was computed on the squeeze (the ≈0 DSR in the JSON is the 2A/2D/2B XS
  sleeves). Given effective n ≈ 21–113 macro-events, a deflation at the crypto book count (1869) would very
  likely NOT survive. Per-trade Sharpe is 0.16 (all) / 0.24 (deep) — the *real-but-weak* band; a leak would
  read ≫1, so the weak positive is credible as signal, not artifact.

**Governance implication [PROBABLE]:** the book's single live edge is **weakly supported, not robustly
confirmed** under bull/bear + cluster-aware deflation. It is not refuted as bull-beta (bull excess is real
but weak; bear does not flip), but it does not clear a clustered/deflated bar. Do **not** treat
`funding_squeeze` as a confirmed edge; size it as a fragile, right-tail, regime-broad continuation tilt.
Caveats: single frozen lake, no OOS/forward validation, **survivorship** (23/23 alive — the losing tail of a
long-only continuation bet is absent by construction, inflating the right tail).

---

## Exploratory lead (POST-HOC — NOT a result, NOT claimable) [PRÉSUMÉ]

A 2-sign × 6-cadence scan of 2A (12 configs, logged as 18 exploratory trials in `TRIAL_REGISTRY.md`) shows
the cross-sectional funding signal has real structure in the **INVERTED** direction (fade high funding-z =
cross-sectional funding reversion) at **slow** cadence, where turnover cost amortises:

| rebalance | inverted-2A net (true annual Sharpe) |
|---|---|
| daily | −1.06 |
| weekly (7d) | +0.38 |
| bi-weekly (14d) | **+0.58** |

This is p-hacked (best cell of a post-hoc search) and the **+0.58 is the annualisation-corrected** figure
(an earlier fixed-√365 bug reported +2.16 — overstated ≈3.7×). It is **not a survivor**. It is the precise
*"what would reopen the XS funding axis"*: a **pre-registered** XS funding-reversion sleeve at ≥weekly
cadence, deflated at the full crypto book count (1869 + this stream's 4 + 18 trials), OOS/forward-validated.

---

## Provenance & method
- Symbol-format trap handled [VÉRIFIÉ]: `binance_*` = `BTC/USDT`; `coinglass_multi_*` = bare `BTC`;
  `coinglass_open_interest`/`coinglass_funding_rate` = `BTCUSDT`. Schemas verified before every join.
- Costs: 17.5 bps/leg turnover; funding carry once per settlement (8h for 22 pairs, 4h for WIF) from native.
- Variable universe (NaN before listing) and survivorship (23 current survivors) handled/flagged → S10.
- Engine + builders are TDD-tested (22 tests, incl. L0) and adversarially reviewed (no leak; regime
  look-ahead found, fixed, and re-tested). Full numbers: `results_s2_xsec.json`.
