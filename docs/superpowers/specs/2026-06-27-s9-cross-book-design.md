# Stream 9 — Orthogonalité crypto↔FX & allocation cross-livre — DESIGN SPEC

- **Stream:** S9 (Senior portfolio) · **Branch:** `hermes/s9-cross-book` · **Date:** 2026-06-27
- **Status:** design approved (operator green light + 5 adjustments) → ready for implementation plan
- **Module (write-locked namespace):** `src/hermes/research/s9_cross_book.py` (+ `tests/test_s9_cross_book.py`)
- **Outputs:** `research_streams/S9/FINDINGS.md`, `research_streams/S9/CHECKPOINT.md`
- **Doctrine:** *measure ≠ allocate · conditioner ≠ alpha · shadow size-0 · look-ahead test first · TDD · production-lag in every correlation*

---

## 1. Mandate (recap)

Build the machinery to (9.1) measure crypto↔FX orthogonality at the **regime level** and allocate
risk **cross-book** (HRP); (9.2) test whether a sleeve diversifies the **global** book; (9.3) design a
shared **anti-black-swan overlay** and **measure its cost**. This is **not** lead-lag prediction
(X1/X2 are dead, ρ≈0 both ways) — it is **diversification by correlation structure**. Everything runs in
**shadow (size 0)**; nothing trades live. Live wiring (pulse registration, overlay activation) is the
operator's **merge gate**.

## 2. Locked decisions (from operator)

| # | Decision | Choice |
|---|----------|--------|
| A | 9.1 "dated" ρ definition / KILL | **Triple view**: report `daily`, `weekly`, `risk_off`; **KILL a pair when `max(|daily|,|weekly|,|risk_off|) ≥ 0.30`** |
| B | 9.3 overlay action | **Halt-new + de-gross existing** by a tunable fraction; **auto-rearm** after stress clears |
| C | 9.3 triggers | **All four** (vol spike, stablecoin depeg, price gap, synchronized cross-book DD) |

### 2b. Binding adjustments (operator, 5)

1. **Measure all, allocate only confirmed.** The ρ matrix (9.1) **includes** dead/fragile sleeves
   (`ts_mom_vol_scaled`, `mean_reversion_regime`, the two shadow sleeves) — it must, to reproduce the
   S1×S2=0.35 demasking. The **HRP allocator weights ONLY the confirmed/orthogonal subset** (today =
   `funding_squeeze_continuation`). `ts_mom_vol_scaled` is dead weight S1 purges (LOO ≈ −0.146); the two
   shadow sleeves are demoted "real-but-fragile". They get **zero HRP weight** even though they remain in
   the committed code on this branch (S1's purge is not yet merged).
2. **HRP: cluster on max-ρ, size on daily covariance.** The clustering tree / quasi-diagonal order uses
   the tail-aware distance `d = sqrt((1 − ρ_link)/2)` where `ρ_link` is the signed view (`daily`,
   `weekly`, `risk_off`) of **largest absolute value** — so sleeves that crowd in risk-off group together.
   The inverse-variance **sizing** step uses the **real causal daily covariance**, NOT max-ρ: the
   element-wise max of three correlation matrices is **not PSD**, so a cluster variance `wᵀVw` computed on
   it can be wrong/negative. **Clustering/order on max-ρ distance; variances/sizing on daily covariance.**
3. **9.2: double condition.** A sleeve is a cross-book diversifier **iff `global_LOO > 0` AND
   `global_LOO > native_LOO`** — net-accretive to the global book *and* worth more there than in its
   native book. Both conditions, not just the second.
4. **Overlay: report the JOINT effect, not only marginal.** Besides per-trigger ablation (marginal),
   report the **full retained overlay (all kept triggers together)** vs baseline: calmar-gain /
   sharpe-cost / dd-reduction. Triggers interact; the **final KILL is judged on the joint**.
5. **Verify L0 before building on it.** Everything rests on L0's realized PnL being causal/zero
   look-ahead. **Confirmed:** `research.harness.evaluate_signal` is committed (`766f03f`, clean), decides
   with `df.iloc[lo:t+1]` (≤ t) and realizes forward via `triple_barrier_label(... t, closes[t] ...)`.
   An explicit **L0 look-ahead test** is the first test written.

### 2c. Confirmed implementation choices

- **Overlay model = exposure-multiplier v1** (clean, fully testable). Ledger-level entry-dropping is a
  documented future option, **not now**.
- **HRP clustering = `scipy.cluster.hierarchy`** (proven linkage; do not reinvent). **scipy is already a
  hard prod dependency** (`scipy>=1.14`, used by `learning/stats.py` for the Deflated Sharpe) and imports
  cleanly under `uv` — so HRP adds **zero image weight and no new dependency**. In-module fallback only if
  that ever changes (it won't here). `duckdb` (the lake) stays a dev-group dep.

## 3. Verified facts (provenance)

- `[VÉRIFIÉ]` `research/harness.py` committed `766f03f`, clean; `evaluate_signal` causal (decision ≤ t,
  realized forward), `_daily_pnl` collapses the ledger to a daily series.
- `[VÉRIFIÉ]` `validation/book.py` and `datalake/` are **untracked** on this branch (working-tree
  stand-ins). `book.py` exposes `align_sleeves`, `causal_inverse_vol_weights`, `combine_book`,
  `leave_one_out_delta_sharpe`, `orthogonality_matrix` (raw daily Pearson), `book_report`. **No** Calmar
  / drawdown / regime-correlation / production-lag-correlation helper exists → S9 builds them.
- `[VÉRIFIÉ]` `scipy.cluster.hierarchy` importable under `uv`; scipy is a declared prod dep.
- `[VÉRIFIÉ]` committed sleeves emit scalar `SkillSignal`; return series come **only** via `evaluate_signal`.

## 4. Architecture — one bridge + five layers (all in `s9_cross_book.py`)

```
   committed SkillSignal fns        frozen DuckDB lake (read-only)
   funding_squeeze, ts_mom,    +    binance_klines, deribit_dvol,
   mean_rev, 2 shadow sleeves       defillama_stablecoin_supply, gold, fred
                  │
  L0  sleeve_returns()  ── reuses research.harness.evaluate_signal + _daily_pnl
                  │   → dict[sleeve_id → pd.Series]  (+ book tag, + confirmed flag)
   ┌──────────────┼───────────────────────────────────┐
 L1 regime ρ    L2 HRP alloc        L3 overlay          L4 global LOO
 (9.1 matrix)  (9.1 cross-book)    (9.3 shadow)         (9.2 merge-ready)
   └──────────────┴──── L5 read-only dashboard emitter (no pulse.py edit) ────┘
```

## 5. Component specs

### L0 — Sleeve sourcing (the bridge)

```python
Book = Literal["crypto", "fx"]

@dataclass(frozen=True)
class SleeveSpec:
    sleeve_id: str
    signal_fn: SignalFn               # committed fn(df) -> SkillSignal
    book: Book
    confirmed: bool                   # eligible for HRP allocation (measure-all / allocate-confirmed)
    symbols: tuple[str, ...]          # lake symbols to backtest over
    feature_cols: tuple[str, ...] = ()# extra lake columns the signal reads (e.g. funding_rate, dvol_z)
    feature_lag_bars: int = 0         # production lag applied to feature cols BEFORE the signal sees them

def load_ohlcv(symbols, interval="1d", features=(), lag_bars=0) -> dict[str, pd.DataFrame]:
    """Read OHLCV (+ production-lagged feature cols) from the frozen lake, read-only DuckDB."""

def sleeve_returns(specs: list[SleeveSpec]) -> dict[str, pd.Series]:
    """Per sleeve: evaluate_signal(...) → _daily_pnl → daily realized PnL series. Causal by construction.
    Returns dict[sleeve_id → pd.Series] plus a parallel registry of (book, confirmed) per sleeve."""
```

- **Causal by construction**; pinned by `test_L0_no_lookahead` (inject a future OHLCV spike; assert the
  ledger at/before `t` is byte-identical).

### L1 — Regime-level orthogonality (9.1)

New primitives (none exist today; also used by L3):

```python
def max_drawdown(returns) -> float          # most-negative peak-to-trough of the cumulative equity
def calmar_ratio(returns, periods_per_year) -> float   # annualized return / |max_drawdown|
```

Correlation views + regime engine:

```python
def _resample_sum(s: pd.Series, freq: str) -> pd.Series        # daily PnL → period PnL
def dated_correlation(a, b, freq="W") -> float                 # Pearson on resampled series (weekly view)
def regime_conditional_correlation(a, b, risk_off_mask) -> float  # Pearson inside risk-off only
def risk_off_mask(market_proxy_returns, vol_z_max, dd_threshold) -> pd.Series:
    """Self-contained risk-off labeller (vol-z / drawdown state). [PRÉSUMÉ] stand-in, swappable for the
    S8 regime engine at merge. Reuses the same vol-z/drawdown code L3 needs."""

@dataclass(frozen=True)
class PairOrtho:
    daily: float; weekly: float; risk_off: float
    @property
    def rho_kill(self) -> float:   # max(|daily|, |weekly|, |risk_off|) → KILL if >= 0.30
    @property
    def rho_link(self) -> float:   # signed view of largest |·| → tail-aware HRP distance

def regime_orthogonality(returns, risk_off_mask, weekly_freq="W") -> dict[frozenset[str], PairOrtho]
def orthogonality_kill_matrix(returns, risk_off_mask) -> pd.DataFrame   # the binding |ρ|_max matrix
def is_orthogonal(pair, bar=0.30) -> bool                              # pair.rho_kill < bar
def demask_report(ortho, bar=0.30) -> list[Demask]:
    """Pairs where daily < bar but weekly or risk_off >= bar — 'looked diversifying, unmasked'."""
```

- **TDD anchors:** weekly view unmasks an orthogonal-daily/crowded-weekly pair (the S1×S2 reproduction);
  risk_off view spikes on a calm-independent/stress-cocrash pair.

### L2 — HRP cross-book allocator (9.1) — adjustment #1 + #2

```python
def correlation_distance(rho_link_matrix) -> pd.DataFrame       # d_ij = sqrt((1 - rho_link_ij)/2)
def hrp_quasidiag_order(dist, method="single") -> list[str]     # scipy linkage leaf order
def _cluster_inv_var_weights(cov_daily, items) -> (w, var)      # inverse-variance within a cluster, wᵀVw
def hrp_recursive_bisection(cov_daily, order) -> dict[str, float]

def hrp_allocate(returns, *, confirmed_ids, risk_off_mask) -> dict[str, float]:
    """1. restrict to confirmed_ids (allocate-confirmed; non-confirmed -> weight 0.0 explicitly).
       2. rho_link matrix from regime_orthogonality over the confirmed subset.
       3. cluster/order on correlation_distance(rho_link)  ← tail-aware (max-ρ).
       4. size via recursive bisection on the CAUSAL DAILY COVARIANCE of the confirmed subset ← PSD.
       Returns weights summing to 1 over confirmed sleeves; a cluster map for the dashboard."""
```

- **Today** `confirmed_ids = {"funding_squeeze_continuation"}` → trivial `{...: 1.0}`. The multi-sleeve
  machinery is exercised by **synthetic fixtures** and **populated at merge**.
- The daily covariance is **trailing/realized** (allocation snapshot from data up to the refresh
  timestamp) — no look-ahead.
- **TDD anchors:** a colinear-in-risk-off block clusters and is down-weighted vs naive inverse-vol;
  cluster variance is never negative (PSD sizing); non-confirmed sleeves receive exactly 0.

### L3 — Anti-black-swan overlay (9.3) — decisions B/C + adjustment #4

```python
@dataclass(frozen=True)
class OverlayConfig:
    vol_z_max: float; depeg_bps: float; gap_pct: float
    cross_dd_window: int; cross_dd_threshold: float
    degross_fraction: float            # in [0,1]; existing book scaled to (1 - degross_fraction)
    rearm_clear_periods: int
    sharpe_cost_tol: float             # max net-Sharpe bleed a trigger may cost before it is dropped
    dvol_lag_bars: int = 1             # DVOL +1 bar
    stablecoin_lag_days: int = 3       # >= production lag; provisional rows dropped
    # NOTHING hardcoded at call sites — all thresholds (incl. sharpe_cost_tol) are config, to be
    # calibrated, not frozen.

# Causal triggers -> bool Series on the book grid (each uses ONLY data <= t):
def vol_spike_trigger(vol_series, cfg) -> pd.Series      # z-baseline from past only, +dvol_lag_bars
def gap_trigger(ohlc, cfg) -> pd.Series                  # gap measured at NEXT open vs prior close
def depeg_trigger(stablecoin_supply, cfg) -> pd.Series   # lagged supply contraction / peg deviation
def cross_dd_trigger(drawdown_by_book, cfg) -> pd.Series # both books DD within window (pending 2nd book)

def exposure_multiplier(triggers, cfg) -> pd.Series:
    """Union of active triggers → multiplier (1 - degross_fraction) during stress; auto-rearm to 1.0
       after rearm_clear_periods consecutive clear periods."""

def apply_overlay(book_returns, multiplier) -> pd.Series:
    """book_returns * multiplier.shift(1) — the multiplier decided at t acts on the t+1 return.
       The shift is the look-ahead guard: the overlay must NOT scale the very bar it reacts to.
       v1 NOTE: at the book-return granularity the single exposure scalar subsumes BOTH 'halt new
       entries' and 'de-gross existing' (decision B); the ledger-level split (drop entries in-window
       vs scale in-flight trades) is the documented future refinement, not v1."""

def overlay_cost_report(book_returns, triggers_by_name, cfg, periods_per_year) -> dict:
    """baseline {sharpe, calmar, max_dd};
       MARGINAL per trigger (ablate that one trigger from the full set);
       JOINT (all retained triggers together) vs baseline;
       per-trigger verdict KEEP iff calmar_gain > 0 AND sharpe_cost <= tol;
       FINAL verdict judged on the JOINT (adjustment #4)."""
```

- **KILL (9.3):** a trigger that improves drawdown but bleeds net Sharpe is a **bad trigger → it is
  dropped**. The overlay as a whole is accepted only if the **joint** improves Calmar/drawdown **without
  destroying net Sharpe**.
- **Status labels for FINDINGS:** vol spike + price gap = **measured now**; stablecoin depeg = **measured
  now (DeFiLlama history)**; synchronized cross-book DD = **machinery now, populated at merge**.
- **Shadow:** the overlay logs its cuts; it cuts nothing live.
- **TDD anchors:** every trigger at `t` uses only `≤ t` data (gap uses next open); `apply_overlay` acts
  next period (the `shift(1)` guard); rearm logic; a trigger that cuts only good days → `verdict = OUT`;
  joint report present.

### L4 — Cross-book diversifier test (9.2) — adjustment #3

```python
def native_loo(returns, book_of) -> dict[str, float]            # LOO within each sleeve's own book
def global_loo(crypto_sleeves, fx_sleeves) -> dict[str, float]  # LOO on the UNION book (committed loo reused)
def cross_book_diversifiers(crypto_sleeves, fx_sleeves) -> dict[str, dict]:
    """per sleeve: {global_loo, native_loo, is_diversifier}
       is_diversifier = (global_loo > 0) AND (global_loo > native_loo)   ← double condition."""
```

- Built + tested **now** on synthetic two-book fixtures; **real FX sleeves drop in at merge** (the FX
  book is empty today — honest-negative). Reuses committed `leave_one_out_delta_sharpe`.

### L5 — Dashboard emitter (read-only)

```python
def s9_pulse_emitter(kill_matrix, hrp_weights, overlay_state) -> dict:
    """{"kind": "s9_cross_book", "shadow": True, "rho_max": ..., "hrp": ..., "overlay": ...}"""
```

- **No edit to `pulse.py`.** Registration of this emitter is a **documented one-line human merge step**
  (respects the stream namespace + shadow doctrine).

## 6. Data sources & point-in-time lags

| Need | Lake table | Lag rule |
|------|-----------|----------|
| Crypto sleeve PnL | `binance_klines` (+ `binance_funding`, feature cols) | feature_lag_bars in L0 |
| Vol-spike trigger | `deribit_dvol` (or realized vol from klines) | **DVOL +1 bar** (`ts`=open) |
| Depeg trigger | `defillama_stablecoin_supply` | **≥3 d**, drop provisional last 1–2 rows |
| Gap trigger | `binance_klines` / `gold_prices_*` OHLC | measured at **next open** |
| Risk-off mask / market proxy | `binance_klines` BTC (crypto), `gold`/`dxy` (context) | trailing only |

## 7. TDD plan (look-ahead first)

1. `test_L0_no_lookahead` — future spike does not change past ledger. **(write first)**
2. `test_max_drawdown` / `test_calmar_ratio` — known fixtures.
3. `test_weekly_view_unmasks` — orthogonal daily, crowded weekly → `rho_kill ≥ 0.30`, demask flags it.
4. `test_risk_off_view_cocrash` — calm-independent/stress-cocrash → high `risk_off`.
5. `test_triple_view_max_and_kill`.
6. `test_hrp_cluster_maxrho_size_dailycov` — colinear block clusters; sizing on daily cov; no negative
   cluster variance; colinear sleeves down-weighted vs naive inverse-vol.
7. `test_hrp_allocates_only_confirmed` — non-confirmed → 0.
8. `test_overlay_triggers_causal` — each trigger uses ≤ t; gap at next open.
9. `test_apply_overlay_acts_next_period` — the `shift(1)` guard.
10. `test_overlay_rearm`.
11. `test_overlay_cost_marginal_and_joint` — good-day-only trigger → OUT; joint reported.
12. `test_global_loo_double_condition` — accretive AND > native → diversifier; accretive but < native → not.

## 8. Deliverables & provenance

- `src/hermes/research/s9_cross_book.py` + `tests/test_s9_cross_book.py` (TDD, all pure functions).
- `research_streams/S9/FINDINGS.md` — verdict per vector; **per-trigger Calmar-gain/Sharpe-cost table +
  the joint**; explicit **measured-now vs pending-merge** labels; `[VÉRIFIÉ]`/`[PROBABLE]`/`[PRÉSUMÉ]`.
- `research_streams/S9/CHECKPOINT.md` — ≤ 8 lines.
- This design spec, committed now.

## 9. Scope bounds / merge-gate deferrals

- **No** lead-lag (X1/X2 dead). **No** new mono-book signal. **No** edits to `validation/book.py`,
  `pulse.py`, or any other stream's files.
- **Nothing live**; shadow size-0 throughout; overlay logs only.
- HRP confirmed-set today = `{funding_squeeze_continuation}`; multi-sleeve via tests + merge.
- Risk-off proxy self-contained `[PRÉSUMÉ]`, swappable for S8 at merge.
- Cross-book DD trigger + FX sleeves: machinery now, data at merge.
- Pulse registration + overlay activation: documented, **operator merge gate**.

## 10. Open items (resolve at implementation)

- Exact lake column names for DVOL z and stablecoin supply (confirm via `DESCRIBE`).
- `periods_per_year`: crypto daily = 365, FX daily = 252 — **parameterized**, not assumed.
- Single confirmed sleeve today → HRP trivial now (acknowledged; machinery proven by tests).
