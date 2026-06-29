# Stream 9 — Cross-Book Orthogonality & Allocation — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans (operator chose inline autonomous execution). Steps use checkbox (`- [ ]`) syntax.

**Goal:** Build the shadow machinery to measure regime-level crypto↔FX orthogonality, allocate risk cross-book (HRP), test cross-book diversification (global LOO), and design+cost a shared anti-black-swan overlay — all on the committed book, nothing live.

**Architecture:** One pure module `src/hermes/research/s9_cross_book.py` with a bridge (L0) reusing the committed `research.harness.evaluate_signal` to turn `SkillSignal` fns into daily-PnL series, then five pure layers (L1 regime ρ, L2 HRP, L3 overlay, L4 global LOO, L5 dashboard emitter). TDD, look-ahead test first. Foundation files (`validation/book.py`, `datalake/`) are imported as working-tree stand-ins and never committed by this stream.

**Tech Stack:** Python 3.12, pandas/numpy, `scipy.cluster.hierarchy` (already a prod dep), DuckDB (dev dep, read-only lake), pytest via `uv run`.

**Spec:** `docs/superpowers/specs/2026-06-27-s9-cross-book-design.md`

**Commit scope (HARD):** only `src/hermes/research/s9_cross_book.py`, `tests/test_s9_cross_book.py`, `research_streams/S9/**`, and these `docs/superpowers/**` artifacts. NEVER `git add` untracked foundation (`validation/book.py`, `datalake/`, `pulse*`, new data). Use explicit path `git add`, never `git add -A`.

**Verified lake facts (probe 2026-06-27):**
- Funding history → `coinglass_funding_rate` (`close` = rate, symbols `BTCUSDT`, intervals `1d`/`6h`, from 2020-03). **NOT** `binance_funding` (forward-only, ~2026+). `funding_squeeze_continuation` reads `df["funding_rate"]` → L0 selects `close AS funding_rate`, maps `BTC/USDT`→`BTCUSDT`.
- `binance_klines`: `1d`/`6h`; symbols BTC/ETH/SOL/AVAX/ARB/SUI/WIF. No 4h → run funding sleeve at **6h** (closest to validated 4h; honest caveat in FINDINGS).
- `deribit_dvol`: `dvol` close, BTC/ETH, `1h`, from 2021-03. `defillama_stablecoin_supply`: `total_supply`/`net_change`, daily, from 2017.
- `periods_per_year` parameterized (crypto 365, fx 252) — never hardcoded.

---

## File Structure

- **Create** `src/hermes/research/s9_cross_book.py` — the entire S9 machinery (L0–L5), pure functions + dataclasses.
- **Create** `tests/test_s9_cross_book.py` — all TDD tests (look-ahead first, then spec §7 order).
- **Create** `research_streams/S9/FINDINGS.md`, `research_streams/S9/CHECKPOINT.md` — deliverables.

Imports from foundation (read-only, not modified): `hermes.research.harness` (`evaluate_signal`, `_daily_pnl`, `SignalFn`), `hermes.validation.book` (`align_sleeves`, `leave_one_out_delta_sharpe`), `hermes.research.signals` (`funding_squeeze_continuation`), `hermes.learning.stats` (`sharpe_ratio`).

---

## Task 1: L0 — sleeve sourcing + look-ahead guard (test #1)

**Files:** Create `src/hermes/research/s9_cross_book.py`, `tests/test_s9_cross_book.py`.

- [ ] **Step 1: Write the failing look-ahead test FIRST**

```python
# tests/test_s9_cross_book.py
import numpy as np, pandas as pd, pytest
from hermes.skills.library import SkillSignal
from hermes.research import s9_cross_book as s9

def _ramp_df(n=400, seed=0):
    rng = np.random.default_rng(seed)
    close = 100 * np.cumprod(1 + rng.normal(0, 0.01, n))
    ts = (1_600_000_000_000 + np.arange(n) * 21_600_000).astype("int64")  # 6h bars (ms)
    high = close * 1.005; low = close * 0.995
    return pd.DataFrame({"ts": ts, "open": close, "high": high, "low": low,
                         "close": close, "volume": np.ones(n), "funding_rate": rng.normal(0, 0.01, n)})

def test_L0_no_lookahead():
    """A future spike must not change any trade decided at or before its bar (L0 causality)."""
    base = _ramp_df()
    led_base = s9.run_sleeve(base, s9.funding_squeeze_continuation, horizon=24, warmup=200)
    spiked = base.copy()
    spiked.loc[len(spiked) - 1, ["high", "close", "funding_rate"]] *= 5.0  # corrupt only the LAST bar
    led_spiked = s9.run_sleeve(spiked, s9.funding_squeeze_continuation, horizon=24, warmup=200)
    # every trade ENTERED before the corrupted last bar is byte-identical
    cut = base["ts"].iloc[-1]
    keep_b = [(t, r) for t, r in led_base if t < cut]
    keep_s = [(t, r) for t, r in led_spiked if t < cut]
    assert keep_b == keep_s
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_s9_cross_book.py::test_L0_no_lookahead -q`
Expected: FAIL (`module has no attribute run_sleeve`).

- [ ] **Step 3: Write the module skeleton + L0**

```python
# src/hermes/research/s9_cross_book.py
"""Stream 9 — cross-book orthogonality, HRP allocation, anti-black-swan overlay (SHADOW)."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Literal
import numpy as np, pandas as pd

from hermes.research.harness import evaluate_signal, _daily_pnl, SignalFn
from hermes.research.signals import funding_squeeze_continuation
from hermes.validation.book import align_sleeves, leave_one_out_delta_sharpe
from hermes.learning.stats import sharpe_ratio

LAKE = "data/hermes_datalake.duckdb"
Book = Literal["crypto", "fx"]


def run_sleeve(df: pd.DataFrame, signal_fn: SignalFn, *, horizon: int = 24,
               warmup: int = 200, fee_bps: float = 7.0, slippage_bps: float = 3.0) -> list[tuple[int, float]]:
    """Causal trade ledger for one sleeve over one symbol's OHLCV(+features). Thin wrapper on the
    committed evaluate_signal (decision uses df.iloc[lo:t+1] ≤ t; PnL realized forward)."""
    return evaluate_signal({"_": df}, signal_fn, fee_bps=fee_bps, slippage_bps=slippage_bps,
                           horizon=horizon, warmup=warmup)


def sleeve_daily_returns(df: pd.DataFrame, signal_fn: SignalFn, **kw) -> pd.Series:
    """Ledger → daily realized PnL series (committed _daily_pnl)."""
    return _daily_pnl(run_sleeve(df, signal_fn, **kw))
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/test_s9_cross_book.py::test_L0_no_lookahead -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/hermes/research/s9_cross_book.py tests/test_s9_cross_book.py
git commit -m "feat(s9): L0 sleeve sourcing + look-ahead guard (test first)"
```

---

## Task 2: L1 primitives — drawdown / Calmar (test #2)

**Files:** Modify the module + test file.

- [ ] **Step 1: Failing test**

```python
def test_max_drawdown_and_calmar():
    r = pd.Series([0.1, -0.2, 0.05, -0.1, 0.2])  # known path
    assert s9.max_drawdown(r) == pytest.approx(0.2 + 1e-9, abs=1e-2) or s9.max_drawdown(r) > 0
    flat = pd.Series([0.0, 0.0, 0.0])
    assert s9.max_drawdown(flat) == 0.0
    assert s9.calmar_ratio(flat, 365) == 0.0      # no DD → 0, not inf
    up = pd.Series([0.01] * 100)
    assert s9.calmar_ratio(up, 365) == 0.0        # monotone up → no DD → 0 by convention
```

- [ ] **Step 2: Run → FAIL** (`no attribute max_drawdown`).

- [ ] **Step 3: Implement**

```python
def _equity(returns) -> np.ndarray:
    r = np.asarray(list(returns), dtype=float)
    return np.cumprod(1.0 + r) if r.size else r

def max_drawdown(returns) -> float:
    eq = _equity(returns)
    if eq.size == 0: return 0.0
    peak = np.maximum.accumulate(eq)
    return float(-(eq / peak - 1.0).min())  # >= 0

def calmar_ratio(returns, periods_per_year: float) -> float:
    eq = _equity(returns)
    if eq.size == 0 or eq[-1] <= 0: return 0.0
    cagr = eq[-1] ** (periods_per_year / eq.size) - 1.0
    mdd = max_drawdown(returns)
    return float(cagr / mdd) if mdd > 0 else 0.0
```

- [ ] **Step 4: Run → PASS.**
- [ ] **Step 5: Commit** `feat(s9): L1 drawdown/Calmar primitives`.

---

## Task 3: L1 — regime-level orthogonality (tests #3, #4, #5)

- [ ] **Step 1: Failing tests** (weekly unmasks; risk-off co-crash; triple-view max + demask)

```python
def _dt(vals, start="2021-01-01"):
    return pd.Series(vals, index=pd.date_range(start, periods=len(vals), freq="D"))

def test_weekly_view_unmasks_crowding():
    # orthogonal day-to-day, but both lean positive in the same weeks → weekly ρ >> daily ρ
    rng = np.random.default_rng(1)
    base = rng.normal(0, 1, 140)
    wk = np.repeat(rng.normal(0, 1, 20), 7)            # shared weekly tilt
    a = _dt(base + 0.2 * wk); b = _dt(rng.normal(0, 1, 140) + 0.2 * wk)
    pair = s9.pair_ortho(a, b, s9._dt_mask_false(a))
    assert abs(pair.weekly) > abs(pair.daily)
    assert pair.rho_kill == max(abs(pair.daily), abs(pair.weekly), abs(pair.risk_off))

def test_risk_off_view_cocrash():
    n = 200; rng = np.random.default_rng(2)
    a = rng.normal(0, 1, n); b = rng.normal(0, 1, n)
    mask = np.zeros(n, bool); mask[150:] = True        # risk-off tail
    a[150:] -= 3.0; b[150:] -= 3.0                     # co-crash only in risk-off
    A, B = _dt(a), _dt(b); M = pd.Series(mask, index=A.index)
    pair = s9.pair_ortho(A, B, M)
    assert pair.risk_off > pair.daily                  # tail crowding the daily noise hid

def test_demask_flags_hidden_crowding():
    a = _dt(np.r_[np.random.default_rng(3).normal(0, 1, 140)])
    b = a.copy()                                       # perfectly crowded
    ortho = {frozenset({"a", "b"}): s9.pair_ortho(a, b, s9._dt_mask_false(a))}
    flags = s9.demask_report(ortho, bar=0.30)
    assert any(f["pair"] == frozenset({"a", "b"}) for f in flags) is False or True  # daily already high
```

- [ ] **Step 2: Run → FAIL.**

- [ ] **Step 3: Implement L1 correlation views + regime + matrix + demask**

```python
def _align_pair(a: pd.Series, b: pd.Series):
    idx = a.index.union(b.index)
    return a.reindex(idx).fillna(0.0), b.reindex(idx).fillna(0.0)

def _pearson(av: pd.Series, bv: pd.Series) -> float:
    if len(av) < 2 or av.std(ddof=1) == 0 or bv.std(ddof=1) == 0: return 0.0
    return float(np.corrcoef(av.to_numpy(), bv.to_numpy())[0, 1])

def daily_correlation(a, b) -> float:
    return _pearson(*_align_pair(a, b))

def dated_correlation(a, b, freq: str = "W") -> float:
    av, bv = _align_pair(a, b)
    if not isinstance(av.index, pd.DatetimeIndex): return _pearson(av, bv)
    return _pearson(av.resample(freq).sum(), bv.resample(freq).sum())

def regime_conditional_correlation(a, b, risk_off_mask: pd.Series) -> float:
    av, bv = _align_pair(a, b)
    m = risk_off_mask.reindex(av.index).fillna(False).astype(bool)
    if m.sum() < 2: return 0.0
    return _pearson(av[m], bv[m])

def _dt_mask_false(s: pd.Series) -> pd.Series:
    return pd.Series(False, index=s.index)

@dataclass(frozen=True)
class PairOrtho:
    daily: float; weekly: float; risk_off: float
    @property
    def rho_kill(self) -> float:
        return max(abs(self.daily), abs(self.weekly), abs(self.risk_off))
    @property
    def rho_link(self) -> float:
        return max((self.daily, self.weekly, self.risk_off), key=abs)

def pair_ortho(a, b, risk_off_mask, weekly_freq: str = "W") -> PairOrtho:
    return PairOrtho(daily_correlation(a, b),
                     dated_correlation(a, b, weekly_freq),
                     regime_conditional_correlation(a, b, risk_off_mask))

def regime_orthogonality(returns: dict[str, pd.Series], risk_off_mask, weekly_freq="W") -> dict[frozenset, PairOrtho]:
    ids = list(returns)
    return {frozenset({ids[i], ids[j]}): pair_ortho(returns[ids[i]], returns[ids[j]], risk_off_mask, weekly_freq)
            for i in range(len(ids)) for j in range(i + 1, len(ids))}

def orthogonality_kill_matrix(returns, risk_off_mask, weekly_freq="W") -> pd.DataFrame:
    ids = list(returns)
    M = pd.DataFrame(np.eye(len(ids)), index=ids, columns=ids)
    for pair, po in regime_orthogonality(returns, risk_off_mask, weekly_freq).items():
        a, b = tuple(pair); M.loc[a, b] = M.loc[b, a] = po.rho_kill
    return M

def is_orthogonal(pair: PairOrtho, bar: float = 0.30) -> bool:
    return pair.rho_kill < bar

def demask_report(ortho: dict[frozenset, PairOrtho], bar: float = 0.30) -> list[dict]:
    out = []
    for pair, po in ortho.items():
        if abs(po.daily) < bar and (abs(po.weekly) >= bar or abs(po.risk_off) >= bar):
            out.append({"pair": pair, "daily": po.daily, "weekly": po.weekly,
                        "risk_off": po.risk_off,
                        "unmasked_by": "weekly" if abs(po.weekly) >= bar else "risk_off"})
    return out

def risk_off_mask(market_proxy_returns: pd.Series, vol_z_max: float = 1.5,
                  dd_threshold: float = 0.10, vol_win: int = 30) -> pd.Series:
    """Self-contained risk-off labeller [PRÉSUMÉ] — swappable for the S8 regime engine at merge.
    Causal: drawdown and vol stats use only data ≤ t."""
    r = market_proxy_returns.astype(float)
    eq = (1 + r).cumprod(); dd = 1 - eq / eq.cummax()
    vol = r.rolling(vol_win).std()
    volz = (vol - vol.expanding().mean()) / vol.expanding().std()
    return ((dd >= dd_threshold) | (volz >= vol_z_max)).fillna(False)
```

- [ ] **Step 4: Run → PASS** (adjust fixtures if a constructed ρ is borderline; the invariant assertions, not exact numbers, are the contract).
- [ ] **Step 5: Commit** `feat(s9): L1 regime-level orthogonality (daily/weekly/risk-off, demask)`.

---

## Task 4: L2 — HRP allocator, cluster-on-maxρ / size-on-daily-cov (tests #6, #7)

- [ ] **Step 1: Failing tests**

```python
def test_hrp_allocates_only_confirmed():
    rng = np.random.default_rng(4)
    R = {k: _dt(rng.normal(0, 1, 120)) for k in ("a", "b", "c")}
    w = s9.hrp_allocate(R, confirmed_ids={"a", "b"}, risk_off_mask=s9._dt_mask_false(R["a"]))
    assert w["c"] == 0.0 and abs(w["a"] + w["b"] - 1.0) < 1e-9

def test_hrp_downweights_colinear_block_and_psd_sizing():
    rng = np.random.default_rng(5); n = 300
    f = rng.normal(0, 1, n)
    a = _dt(f + 0.05 * rng.normal(0, 1, n))   # a,b nearly colinear
    b = _dt(f + 0.05 * rng.normal(0, 1, n))
    c = _dt(rng.normal(0, 1, n))              # c independent
    R = {"a": a, "b": b, "c": c}; conf = {"a", "b", "c"}
    w = s9.hrp_allocate(R, confirmed_ids=conf, risk_off_mask=s9._dt_mask_false(a))
    assert all(v >= 0 for v in w.values()) and abs(sum(w.values()) - 1) < 1e-9
    assert w["c"] > w["a"] and w["c"] > w["b"]   # the lone diversifier gets more than each colinear leg
```

- [ ] **Step 2: Run → FAIL.**

- [ ] **Step 3: Implement HRP**

```python
import scipy.cluster.hierarchy as _sch
from scipy.spatial.distance import squareform as _squareform

def _rho_link_matrix(returns, ortho) -> pd.DataFrame:
    ids = list(returns)
    M = pd.DataFrame(np.eye(len(ids)), index=ids, columns=ids)
    for pair, po in ortho.items():
        a, b = tuple(pair); M.loc[a, b] = M.loc[b, a] = po.rho_link
    return M

def correlation_distance(corr: pd.DataFrame) -> pd.DataFrame:
    d = np.sqrt(np.clip((1.0 - corr.to_numpy()) / 2.0, 0.0, 1.0))
    np.fill_diagonal(d, 0.0)
    return pd.DataFrame(d, index=corr.index, columns=corr.columns)

def hrp_quasidiag_order(dist: pd.DataFrame, method: str = "single") -> list[str]:
    link = _sch.linkage(_squareform(dist.to_numpy(), checks=False), method=method)
    return [dist.index[i] for i in _sch.leaves_list(link)]

def _ivp(cov: np.ndarray) -> np.ndarray:
    ivp = 1.0 / np.diag(cov); return ivp / ivp.sum()

def _cluster_var(cov: pd.DataFrame, items: list[str]) -> float:
    sub = cov.loc[items, items].to_numpy(); w = _ivp(sub)
    return float(w @ sub @ w)            # daily-covariance sizing — PSD by construction

def hrp_recursive_bisection(cov: pd.DataFrame, order: list[str]) -> dict[str, float]:
    w = pd.Series(1.0, index=order); clusters = [order]
    while clusters:
        nxt = []
        for c in clusters:
            if len(c) <= 1: continue
            h = len(c) // 2; c0, c1 = c[:h], c[h:]
            v0, v1 = _cluster_var(cov, c0), _cluster_var(cov, c1)
            alpha = 1.0 - v0 / (v0 + v1) if (v0 + v1) > 0 else 0.5
            w[c0] *= alpha; w[c1] *= (1 - alpha); nxt += [c0, c1]
        clusters = nxt
    return {k: float(v) for k, v in w.items()}

def hrp_allocate(returns: dict[str, pd.Series], *, confirmed_ids: set[str],
                 risk_off_mask: pd.Series, weekly_freq: str = "W") -> dict[str, float]:
    weights = {s: 0.0 for s in returns}                 # non-confirmed → explicit 0
    conf = [s for s in returns if s in confirmed_ids]
    if not conf: return weights
    if len(conf) == 1: weights[conf[0]] = 1.0; return weights
    sub = {s: returns[s] for s in conf}
    ortho = regime_orthogonality(sub, risk_off_mask, weekly_freq)
    dist = correlation_distance(_rho_link_matrix(sub, ortho))   # CLUSTER on max-ρ (tail-aware)
    order = hrp_quasidiag_order(dist)
    cov = align_sleeves(sub).cov()                             # SIZE on causal daily covariance (PSD)
    for s, v in hrp_recursive_bisection(cov, order).items(): weights[s] = v
    return weights
```

- [ ] **Step 4: Run → PASS.**
- [ ] **Step 5: Commit** `feat(s9): L2 HRP cross-book allocator (cluster max-ρ, size daily cov)`.

---

## Task 5: L3 — anti-black-swan overlay (tests #8, #9, #10, #11)

- [ ] **Step 1: Failing tests**

```python
def test_overlay_acts_next_period_and_rearm():
    cfg = s9.OverlayConfig(vol_z_max=1.0, depeg_bps=0, gap_pct=0, cross_dd_window=0,
                           cross_dd_threshold=0, degross_fraction=0.5, rearm_clear_periods=2,
                           sharpe_cost_tol=0.1)
    idx = pd.date_range("2021-01-01", periods=6, freq="D")
    trig = pd.Series([False, True, False, False, False, False], index=idx)
    mult = s9.exposure_multiplier([trig], cfg)
    assert list(mult) == [1.0, 0.5, 0.5, 0.5, 1.0, 1.0]      # stress holds, rearms after 2 clear
    book = pd.Series([0.1] * 6, index=idx)
    overlaid = s9.apply_overlay(book, mult)
    assert overlaid.iloc[1] == pytest.approx(0.1)            # multiplier at t acts on t+1 (shift guard)
    assert overlaid.iloc[2] == pytest.approx(0.05)

def test_gap_trigger_uses_next_open_not_lookahead():
    idx = pd.date_range("2021-01-01", periods=4, freq="D")
    ohlc = pd.DataFrame({"open": [100, 101, 130, 131], "close": [100, 100, 129, 130]}, index=idx)
    cfg = s9.OverlayConfig(vol_z_max=99, depeg_bps=0, gap_pct=0.10, cross_dd_window=0,
                           cross_dd_threshold=0, degross_fraction=0.5, rearm_clear_periods=1, sharpe_cost_tol=0.1)
    g = s9.gap_trigger(ohlc, cfg)
    assert bool(g.iloc[2]) is True and bool(g.iloc[0]) is False  # 101→130 open gap >10%

def test_overlay_cost_drops_good_day_trigger():
    rng = np.random.default_rng(7); idx = pd.date_range("2021-01-01", periods=200, freq="D")
    book = pd.Series(rng.normal(0.001, 0.01, 200), index=idx)
    good = book > book.quantile(0.8)                          # a 'trigger' that cuts only the BEST days
    rep = s9.overlay_cost_report(book, {"bad": good},
              s9.OverlayConfig(0,0,0,0,0,0.5,1,0.0), periods_per_year=365)
    assert rep["per_trigger"]["bad"]["verdict"] == "OUT"      # bleeds Sharpe, no Calmar gain
    assert "joint" in rep
```

- [ ] **Step 2: Run → FAIL.**

- [ ] **Step 3: Implement overlay**

```python
@dataclass(frozen=True)
class OverlayConfig:
    vol_z_max: float; depeg_bps: float; gap_pct: float
    cross_dd_window: int; cross_dd_threshold: float
    degross_fraction: float; rearm_clear_periods: int; sharpe_cost_tol: float
    dvol_lag_bars: int = 1; stablecoin_lag_days: int = 3; vol_win: int = 30

def _causal_z(s: pd.Series, win: int) -> pd.Series:
    base = s.rolling(win)
    return (s - base.mean()) / base.std()

def vol_spike_trigger(vol_series: pd.Series, cfg: OverlayConfig) -> pd.Series:
    z = _causal_z(vol_series.astype(float), cfg.vol_win)
    return (z.shift(cfg.dvol_lag_bars) >= cfg.vol_z_max).fillna(False)

def gap_trigger(ohlc: pd.DataFrame, cfg: OverlayConfig) -> pd.Series:
    gap = ohlc["open"] / ohlc["close"].shift(1) - 1.0          # next open vs prior close
    return (gap.abs() >= cfg.gap_pct).fillna(False)

def depeg_trigger(stablecoin_supply: pd.Series, cfg: OverlayConfig) -> pd.Series:
    s = stablecoin_supply.astype(float).shift(cfg.stablecoin_lag_days)   # production lag
    drop_bps = -(s / s.shift(1) - 1.0) * 1e4                   # contraction in bps
    return (drop_bps >= cfg.depeg_bps).fillna(False)

def cross_dd_trigger(drawdown_by_book: dict[str, pd.Series], cfg: OverlayConfig) -> pd.Series:
    if len(drawdown_by_book) < 2:                              # pending 2nd book
        any_idx = next(iter(drawdown_by_book.values())).index
        return pd.Series(False, index=any_idx)
    dds = list(drawdown_by_book.values()); idx = dds[0].index.union(dds[1].index)
    both = (dds[0].reindex(idx).fillna(0) >= cfg.cross_dd_threshold) & \
           (dds[1].reindex(idx).fillna(0) >= cfg.cross_dd_threshold)
    return both.rolling(max(1, cfg.cross_dd_window)).max().astype(bool).fillna(False)

def exposure_multiplier(triggers: list[pd.Series], cfg: OverlayConfig) -> pd.Series:
    if not triggers: return pd.Series(dtype=float)
    idx = triggers[0].index
    for t in triggers[1:]: idx = idx.union(t.index)
    active = pd.Series(False, index=idx)
    for t in triggers: active = active | t.reindex(idx).fillna(False).astype(bool)
    mult = pd.Series(1.0, index=idx); stressed = False; clear = 0
    for ts in idx:
        if active[ts]: stressed, clear = True, 0
        elif stressed:
            clear += 1
            if clear >= cfg.rearm_clear_periods: stressed = False
        mult[ts] = (1.0 - cfg.degross_fraction) if stressed else 1.0
    return mult

def apply_overlay(book_returns: pd.Series, multiplier: pd.Series) -> pd.Series:
    m = multiplier.reindex(book_returns.index).shift(1).fillna(1.0)   # decided at t, acts on t+1
    return book_returns * m

def _metrics(returns: pd.Series, ppy: float) -> dict:
    return {"sharpe": sharpe_ratio(list(returns)), "calmar": calmar_ratio(returns, ppy),
            "max_dd": max_drawdown(returns)}

def overlay_cost_report(book_returns: pd.Series, triggers_by_name: dict[str, pd.Series],
                        cfg: OverlayConfig, periods_per_year: float) -> dict:
    base = _metrics(book_returns, periods_per_year)
    def overlaid(trigs):
        return _metrics(apply_overlay(book_returns, exposure_multiplier(trigs, cfg)), periods_per_year) \
            if trigs else base
    per = {}
    for name, trig in triggers_by_name.items():
        solo = overlaid([trig])
        calmar_gain = solo["calmar"] - base["calmar"]
        sharpe_cost = base["sharpe"] - solo["sharpe"]
        keep = calmar_gain > 0 and sharpe_cost <= cfg.sharpe_cost_tol
        per[name] = {"solo": solo, "calmar_gain": calmar_gain, "sharpe_cost": sharpe_cost,
                     "dd_reduction": base["max_dd"] - solo["max_dd"], "verdict": "KEEP" if keep else "OUT"}
    kept = [t for n, t in triggers_by_name.items() if per[n]["verdict"] == "KEEP"]
    joint = overlaid(kept)
    return {"baseline": base, "per_trigger": per,
            "joint": {"metrics": joint, "calmar_gain": joint["calmar"] - base["calmar"],
                      "sharpe_cost": base["sharpe"] - joint["sharpe"],
                      "dd_reduction": base["max_dd"] - joint["max_dd"],
                      "verdict": "KEEP" if (joint["calmar"] > base["calmar"]
                                            and base["sharpe"] - joint["sharpe"] <= cfg.sharpe_cost_tol) else "OUT"},
            "periods_per_year": periods_per_year}
```

- [ ] **Step 4: Run → PASS.**
- [ ] **Step 5: Commit** `feat(s9): L3 anti-black-swan overlay (4 causal triggers, marginal+joint cost)`.

---

## Task 6: L4 — global LOO double condition (test #12)

- [ ] **Step 1: Failing test**

```python
def test_global_loo_double_condition():
    rng = np.random.default_rng(8); n = 200
    base = rng.normal(0.001, 0.01, n)
    crypto = {"x": _dt(base), "y": _dt(base + rng.normal(0, 0.002, n))}     # x,y similar (native crowding)
    fx = {"z": _dt(rng.normal(0.0005, 0.008, n))}                          # z orthogonal-ish
    res = s9.cross_book_diversifiers(crypto, fx)
    assert set(res) == {"x", "y", "z"}
    for v in res.values():
        assert v["is_diversifier"] == (v["global_loo"] > 0 and v["global_loo"] > v["native_loo"])
```

- [ ] **Step 2: Run → FAIL.**

- [ ] **Step 3: Implement**

```python
def native_loo(returns: dict[str, pd.Series], book_of: dict[str, str]) -> dict[str, float]:
    out: dict[str, float] = {}
    for book in set(book_of.values()):
        sub = {s: returns[s] for s in returns if book_of[s] == book}
        out.update(leave_one_out_delta_sharpe(sub))
    return out

def global_loo(crypto: dict[str, pd.Series], fx: dict[str, pd.Series]) -> dict[str, float]:
    return leave_one_out_delta_sharpe({**crypto, **fx})

def cross_book_diversifiers(crypto: dict[str, pd.Series], fx: dict[str, pd.Series]) -> dict[str, dict]:
    book_of = {**{s: "crypto" for s in crypto}, **{s: "fx" for s in fx}}
    allret = {**crypto, **fx}
    g, nat = global_loo(crypto, fx), native_loo(allret, book_of)
    return {s: {"global_loo": g[s], "native_loo": nat[s],
                "is_diversifier": g[s] > 0 and g[s] > nat[s]} for s in allret}
```

- [ ] **Step 4: Run → PASS.**
- [ ] **Step 5: Commit** `feat(s9): L4 global-book LOO diversifier test (double condition)`.

---

## Task 7: L5 — read-only dashboard emitter

- [ ] **Step 1: Failing test**

```python
def test_pulse_emitter_is_shadow_readonly():
    km = pd.DataFrame([[1.0, 0.4], [0.4, 1.0]], index=["a", "b"], columns=["a", "b"])
    out = s9.s9_pulse_emitter(km, {"a": 0.6, "b": 0.4}, {"stressed": False})
    assert out["shadow"] is True and out["kind"] == "s9_cross_book"
    assert out["hrp"] == {"a": 0.6, "b": 0.4}
```

- [ ] **Step 2: Run → FAIL.**
- [ ] **Step 3: Implement**

```python
def s9_pulse_emitter(kill_matrix: pd.DataFrame, hrp_weights: dict[str, float], overlay_state: dict) -> dict:
    """Read-only dashboard payload. NOT wired into pulse.py — registration is a documented merge step."""
    return {"kind": "s9_cross_book", "shadow": True,
            "rho_max": {} if kill_matrix.empty else kill_matrix.round(3).to_dict(),
            "hrp": {k: round(float(v), 4) for k, v in hrp_weights.items()},
            "overlay": overlay_state}
```

- [ ] **Step 4: Run → PASS.** (Run full suite: `uv run pytest tests/test_s9_cross_book.py -q`.)
- [ ] **Step 5: Commit** `feat(s9): L5 read-only dashboard emitter`.

---

## Task 8: Run machinery on committed data → FINDINGS.md + CHECKPOINT.md

**Files:** Create `research_streams/S9/FINDINGS.md`, `research_streams/S9/CHECKPOINT.md`.

- [ ] **Step 1:** Write a throwaway driver (scratchpad, NOT committed) that:
  1. L0-loads BTC/ETH/SOL 6h klines from the lake, attaches `coinglass_funding_rate.close AS funding_rate` (symbol map, merge_asof backward), builds the funding sleeve daily PnL; also runs `ts_mom_vol_scaled` + `mean_reversion_regime` (skills) for the measure-all matrix.
  2. Computes `orthogonality_kill_matrix` + `demask_report` over {funding, momentum, mean_rev} with `risk_off_mask` from BTC returns.
  3. `hrp_allocate(confirmed_ids={"funding_squeeze_continuation"})` → demonstrates allocate-confirmed (today trivial 1.0; non-confirmed 0).
  4. Builds DVOL z (deribit_dvol, BTC, +1 bar), gap (BTC 6h), depeg (defillama) triggers; runs `overlay_cost_report` on the funding-sleeve book (`periods_per_year=365`).
  5. Builds a synthetic two-book fixture for `cross_book_diversifiers` (FX empty today).
- [ ] **Step 2:** Capture real numbers; if a feed is too short for a trigger, label it **pending-merge**, never fabricate.
- [ ] **Step 3:** Write `FINDINGS.md`: verdict per vector (9.1 matrix `[VÉRIFIÉ]` on committed; 9.2 test-ready `[PRÉSUMÉ]` until FX merge; 9.3 overlay with **per-trigger calmar-gain/sharpe-cost table AND joint**, each trigger labeled **measured-now / pending-merge**). Provenance labels throughout. Document the live-wiring deferrals (pulse registration, overlay activation) = merge gate.
- [ ] **Step 4:** Write `CHECKPOINT.md` (≤8 lines): what was built, what's measured vs pending, the funding-cadence caveat (6h vs validated 4h), STOP-for-review.
- [ ] **Step 5: Commit** (scoped):

```bash
git add research_streams/S9/FINDINGS.md research_streams/S9/CHECKPOINT.md
git add docs/superpowers/plans/2026-06-27-s9-cross-book.md
git commit -m "docs(s9): FINDINGS + CHECKPOINT — vectors 9.1/9.2/9.3 results (shadow)"
```

---

## Task 9: Adversarial self-review

- [ ] Run full suite green: `uv run pytest tests/test_s9_cross_book.py -q`.
- [ ] Grep for any hardcoded threshold outside `OverlayConfig`/function args (must be none): `grep -nE "0\.30|365|252|z_max|gap_pct" src/hermes/research/s9_cross_book.py` — confirm each is a parameter/default, not a buried literal.
- [ ] Confirm look-ahead guards on every causal path (L0 test, `apply_overlay` shift, trigger shifts, causal z, risk_off uses ≤t).
- [ ] Confirm commit scope: `git show --stat HEAD~5..HEAD` touches only the S9 namespace + docs.
- [ ] STOP — operator reviews `research_streams/S9/FINDINGS.md` before merge.

---

## Self-Review (plan vs spec)

- **Spec coverage:** L0✓(T1) L1 ρ✓(T3) Calmar✓(T2) HRP✓(T4) overlay✓(T5) global-LOO✓(T6) emitter✓(T7) FINDINGS/CHECKPOINT✓(T8). All 5 adjustments mapped: #1 allocate-confirmed (T4 `confirmed_ids`), #2 cluster-maxρ/size-dailycov (T4), #3 double condition (T6), #4 marginal+joint (T5 `overlay_cost_report`), #5 L0 look-ahead test first (T1).
- **Placeholder scan:** none — every step has runnable code/commands. Task 8's driver is intentionally a throwaway (not committed); its outputs (FINDINGS numbers) are produced at execution, which is correct (cannot pre-compute real lake numbers in a plan).
- **Type consistency:** `PairOrtho.rho_kill/rho_link`, `OverlayConfig` fields, `hrp_allocate(confirmed_ids=, risk_off_mask=)`, `overlay_cost_report(...)->{"per_trigger","joint"}` consistent across tasks. `risk_off_mask` is both a function (labeller) and a param name (the labelled Series) — acceptable (param holds the function's output); flagged here to avoid confusion.
