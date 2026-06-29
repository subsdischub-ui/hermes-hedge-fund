"""Stream 9 — cross-book orthogonality, HRP allocation, anti-black-swan overlay (SHADOW).

Realises "un seul Hermès, deux livres, un cerveau": measure crypto↔FX orthogonality at the
**regime level** (not raw ρ), allocate risk **cross-book** (HRP), test whether a sleeve diversifies
the **global** book (LOO double condition), and design + **cost** a shared circuit-breaker overlay.

Doctrine baked in:
  * MEASURE ≠ ALLOCATE — the ρ matrix ingests every sleeve (incl. dead/fragile ones, to reproduce the
    S1×S2=0.35 demasking); the HRP allocator weights ONLY the confirmed/orthogonal subset.
  * ρ DATÉ ≠ ρ BRUT — a pair is killed on max(|daily|, |weekly|, |risk_off|) ≥ 0.30.
  * CONDITIONER ≠ ALPHA — the overlay is tail-risk reduction; it must improve Calmar/drawdown WITHOUT
    bleeding net Sharpe, else the trigger is dropped.
  * SHADOW size-0 — nothing here trades or wires into the live loop. Pulse registration and overlay
    activation are documented merge-gate steps, not done by this module.

All functions are pure. ``duckdb`` (dev-only) is imported lazily so this prod module imports cleanly
without it; ``scipy`` is a declared prod dependency (used by learning/stats for the Deflated Sharpe).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as _sch
from scipy.spatial.distance import squareform as _squareform

from hermes.learning.stats import sharpe_ratio
from hermes.research.harness import SignalFn, _daily_pnl, evaluate_signal
from hermes.research.signals import funding_squeeze_continuation  # noqa: F401 (re-exported for L0)
from hermes.validation.book import align_sleeves, leave_one_out_delta_sharpe

LAKE = "data/hermes_datalake.duckdb"
Book = Literal["crypto", "fx"]

__all__ = [
    "run_sleeve", "sleeve_daily_returns",
    "max_drawdown", "calmar_ratio",
    "PairOrtho", "pair_ortho", "regime_orthogonality", "orthogonality_kill_matrix",
    "is_orthogonal", "demask_report", "risk_off_mask",
    "correlation_distance", "hrp_quasidiag_order", "hrp_recursive_bisection", "hrp_allocate",
    "OverlayConfig", "vol_spike_trigger", "gap_trigger", "depeg_trigger", "cross_dd_trigger",
    "exposure_multiplier", "apply_overlay", "overlay_cost_report",
    "native_loo", "global_loo", "cross_book_diversifiers",
    "s9_pulse_emitter",
    "load_klines", "attach_funding",
]


# =========================================================================== #
# L0 — sleeve sourcing (bridge over the committed, causal evaluate_signal)
# =========================================================================== #
def run_sleeve(df: pd.DataFrame, signal_fn: SignalFn, *, horizon: int = 24, warmup: int = 200,
               fee_bps: float = 7.0, slippage_bps: float = 3.0) -> list[tuple[int, float]]:
    """Causal trade ledger ``[(event_ns, ret_net), ...]`` for one sleeve over one symbol's
    OHLCV(+features). Thin wrapper on the committed ``evaluate_signal`` (decision uses df.iloc[lo:t+1]
    ≤ t; PnL realised forward) — pinned by ``test_L0_no_lookahead``."""
    return evaluate_signal({"_": df}, signal_fn, fee_bps=fee_bps, slippage_bps=slippage_bps,
                           horizon=horizon, warmup=warmup)


def sleeve_daily_returns(df: pd.DataFrame, signal_fn: SignalFn, **kw) -> pd.Series:
    """Ledger → daily realised PnL series (committed ``_daily_pnl``)."""
    return _daily_pnl(run_sleeve(df, signal_fn, **kw))


def _ensure_dt(df: pd.DataFrame, col: str = "ts") -> pd.DataFrame:
    """Convert an integer-millisecond epoch column to datetime64[ns, UTC].

    The lake stores ``ts`` as BIGINT ms, but the committed harness does ``pd.Timestamp(ts).value``,
    which reads a bare int as NANOSECONDS — collapsing every trade to ~1970 and destroying the daily
    grouping. Converting here (not in the committed harness) keeps the realised-PnL dates honest."""
    if pd.api.types.is_integer_dtype(df[col]):
        df = df.copy()
        df[col] = pd.to_datetime(df[col], unit="ms", utc=True)
    return df


def load_klines(symbol: str, interval: str = "6h") -> pd.DataFrame:
    """OHLCV from the frozen lake (read-only), ``ts`` as datetime64. ``duckdb`` imported lazily."""
    import duckdb
    con = duckdb.connect(LAKE, read_only=True)
    try:
        df = con.execute(
            "SELECT ts, open, high, low, close, volume FROM binance_klines "
            "WHERE symbol = ? AND interval = ? ORDER BY ts", [symbol, interval]
        ).df()
    finally:
        con.close()
    return _ensure_dt(df)


def attach_funding(klines: pd.DataFrame, symbol: str, interval: str = "6h", lag_bars: int = 0) -> pd.DataFrame:
    """Attach the lake's REAL funding history as a column literally named ``funding_rate``.

    Funding history lives in ``coinglass_funding_rate`` (``close`` = the rate, symbols like ``BTCUSDT``,
    back to 2020-03) — NOT ``binance_funding`` (forward-only, ~2026+, which would void the backtest).
    Backward merge_asof keeps it causal (funding known at/before the kline bar)."""
    import duckdb
    cg_symbol = symbol.replace("/", "")  # BTC/USDT -> BTCUSDT
    con = duckdb.connect(LAKE, read_only=True)
    try:
        f = con.execute(
            "SELECT ts, close AS funding_rate FROM coinglass_funding_rate "
            "WHERE symbol = ? AND interval = ? ORDER BY ts", [cg_symbol, interval]
        ).df()
    finally:
        con.close()
    klines, f = _ensure_dt(klines), _ensure_dt(f)  # both datetime64 so merge_asof dtypes match
    merged = pd.merge_asof(klines.sort_values("ts"), f.sort_values("ts"), on="ts", direction="backward")
    if lag_bars:
        merged["funding_rate"] = merged["funding_rate"].shift(lag_bars)
    return merged


# =========================================================================== #
# L1 — drawdown / Calmar primitives + regime-level orthogonality
# =========================================================================== #
def _equity(returns) -> np.ndarray:
    r = np.asarray(list(returns), dtype=float)
    return np.cumprod(1.0 + r) if r.size else r


def max_drawdown(returns) -> float:
    """Most-negative peak-to-trough of the compounding equity curve (≥ 0)."""
    eq = _equity(returns)
    if eq.size == 0:
        return 0.0
    peak = np.maximum.accumulate(eq)
    return float(-(eq / peak - 1.0).min())


def calmar_ratio(returns, periods_per_year: float) -> float:
    """CAGR / |max_drawdown|. No drawdown → 0.0 by convention (not inf). ``periods_per_year`` is a
    parameter (crypto 365, fx 252) — never hardcoded."""
    eq = _equity(returns)
    if eq.size == 0 or eq[-1] <= 0:
        return 0.0
    cagr = eq[-1] ** (periods_per_year / eq.size) - 1.0
    mdd = max_drawdown(returns)
    return float(cagr / mdd) if mdd > 0 else 0.0


def _align_pair(a: pd.Series, b: pd.Series):
    idx = a.index.union(b.index)
    return a.reindex(idx).fillna(0.0), b.reindex(idx).fillna(0.0)


def _pearson(av: pd.Series, bv: pd.Series) -> float:
    if len(av) < 2 or av.std(ddof=1) == 0 or bv.std(ddof=1) == 0:
        return 0.0
    return float(np.corrcoef(av.to_numpy(), bv.to_numpy())[0, 1])


def daily_correlation(a: pd.Series, b: pd.Series) -> float:
    return _pearson(*_align_pair(a, b))


def dated_correlation(a: pd.Series, b: pd.Series, freq: str = "W") -> float:
    """Pearson on the period-summed series (weekly view) — unmasks regime-level crowding that daily
    noise hides. Falls back to daily Pearson if the index is not datetime."""
    av, bv = _align_pair(a, b)
    if not isinstance(av.index, pd.DatetimeIndex):
        return _pearson(av, bv)
    return _pearson(av.resample(freq).sum(), bv.resample(freq).sum())


def regime_conditional_correlation(a: pd.Series, b: pd.Series, risk_off_mask: pd.Series) -> float:
    """Pearson computed ONLY inside risk-off periods — catches ρ that blows toward +1 in co-crashes."""
    av, bv = _align_pair(a, b)
    m = risk_off_mask.reindex(av.index).fillna(False).astype(bool)
    if m.sum() < 2:
        return 0.0
    return _pearson(av[m], bv[m])


def _dt_mask_false(s: pd.Series) -> pd.Series:
    return pd.Series(False, index=s.index)


@dataclass(frozen=True)
class PairOrtho:
    """Triple-view orthogonality of a sleeve pair (decision A)."""
    daily: float
    weekly: float
    risk_off: float

    @property
    def rho_kill(self) -> float:
        """The binding number: max(|daily|, |weekly|, |risk_off|). KILL the pair at ≥ 0.30."""
        return max(abs(self.daily), abs(self.weekly), abs(self.risk_off))

    @property
    def rho_link(self) -> float:
        """Signed view of largest magnitude → tail-aware HRP clustering distance (adjustment #2)."""
        return max((self.daily, self.weekly, self.risk_off), key=abs)


def pair_ortho(a: pd.Series, b: pd.Series, risk_off_mask: pd.Series, weekly_freq: str = "W") -> PairOrtho:
    return PairOrtho(daily_correlation(a, b),
                     dated_correlation(a, b, weekly_freq),
                     regime_conditional_correlation(a, b, risk_off_mask))


def regime_orthogonality(returns: dict[str, pd.Series], risk_off_mask: pd.Series,
                         weekly_freq: str = "W") -> dict[frozenset, PairOrtho]:
    ids = list(returns)
    return {
        frozenset({ids[i], ids[j]}): pair_ortho(returns[ids[i]], returns[ids[j]], risk_off_mask, weekly_freq)
        for i in range(len(ids)) for j in range(i + 1, len(ids))
    }


def orthogonality_kill_matrix(returns: dict[str, pd.Series], risk_off_mask: pd.Series,
                              weekly_freq: str = "W") -> pd.DataFrame:
    """Symmetric matrix of the binding rho_kill (diagonal 1.0) — the live ρ matrix for the dashboard."""
    ids = list(returns)
    M = pd.DataFrame(np.eye(len(ids)), index=ids, columns=ids)
    for pair, po in regime_orthogonality(returns, risk_off_mask, weekly_freq).items():
        a, b = tuple(pair)
        M.loc[a, b] = M.loc[b, a] = po.rho_kill
    return M


def is_orthogonal(pair: PairOrtho, bar: float = 0.30) -> bool:
    return pair.rho_kill < bar


def demask_report(ortho: dict[frozenset, PairOrtho], bar: float = 0.30) -> list[dict]:
    """Pairs that LOOK diversifying daily (|daily| < bar) but crowd at the regime level
    (|weekly| or |risk_off| ≥ bar) — 'looked diversifying, unmasked'."""
    out: list[dict] = []
    for pair, po in ortho.items():
        if abs(po.daily) < bar and (abs(po.weekly) >= bar or abs(po.risk_off) >= bar):
            out.append({"pair": pair, "daily": po.daily, "weekly": po.weekly, "risk_off": po.risk_off,
                        "unmasked_by": "weekly" if abs(po.weekly) >= bar else "risk_off"})
    return out


def risk_off_mask(market_proxy_returns: pd.Series, vol_z_max: float = 1.5,
                  dd_threshold: float = 0.10, vol_win: int = 30) -> pd.Series:
    """Self-contained risk-off labeller [PRÉSUMÉ] — swappable for the S8 regime engine at merge.
    Causal: drawdown and the trailing-vol z use only data ≤ t."""
    r = market_proxy_returns.astype(float)
    eq = (1 + r).cumprod()
    dd = 1 - eq / eq.cummax()
    vol = r.rolling(vol_win).std()
    volz = (vol - vol.expanding().mean()) / vol.expanding().std()
    return ((dd >= dd_threshold) | (volz >= vol_z_max)).fillna(False)


# =========================================================================== #
# L2 — HRP cross-book allocator (cluster on max-ρ, size on daily covariance)
# =========================================================================== #
def _rho_link_matrix(returns: dict[str, pd.Series], ortho: dict[frozenset, PairOrtho]) -> pd.DataFrame:
    ids = list(returns)
    M = pd.DataFrame(np.eye(len(ids)), index=ids, columns=ids)
    for pair, po in ortho.items():
        a, b = tuple(pair)
        M.loc[a, b] = M.loc[b, a] = po.rho_link
    return M


def correlation_distance(corr: pd.DataFrame) -> pd.DataFrame:
    """López de Prado distance d = sqrt((1 - ρ)/2). Clipped to [0,1]; zero diagonal."""
    d = np.sqrt(np.clip((1.0 - corr.to_numpy()) / 2.0, 0.0, 1.0))
    np.fill_diagonal(d, 0.0)
    return pd.DataFrame(d, index=corr.index, columns=corr.columns)


def hrp_quasidiag_order(dist: pd.DataFrame, method: str = "single") -> list[str]:
    """Quasi-diagonal leaf order from hierarchical clustering on the (max-ρ) distance matrix."""
    link = _sch.linkage(_squareform(dist.to_numpy(), checks=False), method=method)
    return [dist.index[i] for i in _sch.leaves_list(link)]


def _ivp(cov: np.ndarray) -> np.ndarray:
    ivp = 1.0 / np.diag(cov)
    return ivp / ivp.sum()


def _cluster_var(cov: pd.DataFrame, items: list[str]) -> float:
    """Cluster variance wᵀVw on the REAL daily covariance (PSD) — NOT the non-PSD max-ρ matrix."""
    sub = cov.loc[items, items].to_numpy()
    w = _ivp(sub)
    return float(w @ sub @ w)


def hrp_recursive_bisection(cov: pd.DataFrame, order: list[str]) -> dict[str, float]:
    w = pd.Series(1.0, index=order)
    clusters = [order]
    while clusters:
        nxt = []
        for c in clusters:
            if len(c) <= 1:
                continue
            h = len(c) // 2
            c0, c1 = c[:h], c[h:]
            v0, v1 = _cluster_var(cov, c0), _cluster_var(cov, c1)
            alpha = 1.0 - v0 / (v0 + v1) if (v0 + v1) > 0 else 0.5
            w[c0] *= alpha
            w[c1] *= (1 - alpha)
            nxt += [c0, c1]
        clusters = nxt
    return {k: float(v) for k, v in w.items()}


def hrp_allocate(returns: dict[str, pd.Series], *, confirmed_ids: set[str],
                 risk_off_mask: pd.Series, weekly_freq: str = "W") -> dict[str, float]:
    """HRP weights over the CONFIRMED subset only (measure-all / allocate-confirmed, adjustment #1).

    Clustering/order on the tail-aware max-ρ distance; sizing on the causal daily covariance
    (adjustment #2). Non-confirmed sleeves receive an explicit 0.0."""
    weights = {s: 0.0 for s in returns}
    conf = [s for s in returns if s in confirmed_ids]
    if not conf:
        return weights
    if len(conf) == 1:
        weights[conf[0]] = 1.0
        return weights
    sub = {s: returns[s] for s in conf}
    ortho = regime_orthogonality(sub, risk_off_mask, weekly_freq)
    dist = correlation_distance(_rho_link_matrix(sub, ortho))   # CLUSTER on max-ρ
    order = hrp_quasidiag_order(dist)
    cov = align_sleeves(sub).cov()                             # SIZE on causal daily covariance
    for s, v in hrp_recursive_bisection(cov, order).items():
        weights[s] = v
    return weights


# =========================================================================== #
# L3 — shared anti-black-swan overlay (shadow; halt-new + de-gross, auto-rearm)
# =========================================================================== #
@dataclass(frozen=True)
class OverlayConfig:
    """All thresholds parameterized — nothing frozen at call sites (operator guardrail)."""
    vol_z_max: float
    depeg_bps: float
    gap_pct: float
    cross_dd_window: int
    cross_dd_threshold: float
    degross_fraction: float          # in [0,1]; existing book scaled to (1 - degross_fraction)
    rearm_clear_periods: int
    sharpe_cost_tol: float           # max net-Sharpe bleed a trigger may cost before it is dropped
    dvol_lag_bars: int = 1           # DVOL +1 bar
    stablecoin_lag_days: int = 3     # >= production lag; provisional rows dropped
    vol_win: int = 30


def _causal_z(s: pd.Series, win: int) -> pd.Series:
    base = s.rolling(win)
    return (s - base.mean()) / base.std()


def vol_spike_trigger(vol_series: pd.Series, cfg: OverlayConfig) -> pd.Series:
    """Trailing-vol (or DVOL) z breach, lagged +dvol_lag_bars. Causal: z baseline uses only past."""
    z = _causal_z(vol_series.astype(float), cfg.vol_win)
    return (z.shift(cfg.dvol_lag_bars) >= cfg.vol_z_max).fillna(False)


def gap_trigger(ohlc: pd.DataFrame, cfg: OverlayConfig) -> pd.Series:
    """Outsized gap measured at the NEXT open vs the prior close (no a-posteriori peeking)."""
    gap = ohlc["open"] / ohlc["close"].shift(1) - 1.0
    return (gap.abs() >= cfg.gap_pct).fillna(False)


def depeg_trigger(stablecoin_supply: pd.Series, cfg: OverlayConfig) -> pd.Series:
    """Sharp stablecoin supply contraction (bps), lagged ≥ production-lag."""
    s = stablecoin_supply.astype(float).shift(cfg.stablecoin_lag_days)
    drop_bps = -(s / s.shift(1) - 1.0) * 1e4
    return (drop_bps >= cfg.depeg_bps).fillna(False)


def cross_dd_trigger(drawdown_by_book: dict[str, pd.Series], cfg: OverlayConfig) -> pd.Series:
    """Both books drawn down ≥ threshold within a window — the genuinely cross-book tail signal.
    With < 2 books (FX empty today) returns all-False: machinery now, populated at merge."""
    if len(drawdown_by_book) < 2:
        idx = next(iter(drawdown_by_book.values())).index
        return pd.Series(False, index=idx)
    dds = list(drawdown_by_book.values())
    idx = dds[0].index.union(dds[1].index)
    both = (dds[0].reindex(idx).fillna(0.0) >= cfg.cross_dd_threshold) & \
           (dds[1].reindex(idx).fillna(0.0) >= cfg.cross_dd_threshold)
    return both.rolling(max(1, cfg.cross_dd_window)).max().astype(bool).fillna(False)


def exposure_multiplier(triggers: list[pd.Series], cfg: OverlayConfig) -> pd.Series:
    """Union of active triggers → exposure scalar (1 - degross_fraction) during stress, auto-rearm to
    1.0 after ``rearm_clear_periods`` consecutive clear periods. De-gross applies ON the trigger bar;
    ``apply_overlay``'s shift(1) is what makes the action land on the next bar (separate mechanisms)."""
    if not triggers:
        return pd.Series(dtype=float)
    idx = triggers[0].index
    for t in triggers[1:]:
        idx = idx.union(t.index)
    active = pd.Series(False, index=idx)
    for t in triggers:
        active = active | t.reindex(idx).fillna(False).astype(bool)
    mult = pd.Series(1.0, index=idx)
    stressed, clear = False, 0
    for ts in idx:
        if active[ts]:
            stressed, clear = True, 0
        mult[ts] = (1.0 - cfg.degross_fraction) if stressed else 1.0
        if (not active[ts]) and stressed:
            clear += 1
            if clear >= cfg.rearm_clear_periods:
                stressed = False
    return mult


def apply_overlay(book_returns: pd.Series, multiplier: pd.Series) -> pd.Series:
    """book[t+1] *= multiplier[t]. The shift(1) is the look-ahead guard: the overlay must never scale
    the very bar it reacts to."""
    m = multiplier.reindex(book_returns.index).shift(1).fillna(1.0)
    return book_returns * m


def _metrics(returns: pd.Series, ppy: float) -> dict:
    return {"sharpe": sharpe_ratio(list(returns)), "calmar": calmar_ratio(returns, ppy),
            "max_dd": max_drawdown(returns)}


def overlay_cost_report(book_returns: pd.Series, triggers_by_name: dict[str, pd.Series],
                        cfg: OverlayConfig, periods_per_year: float) -> dict:
    """Per-trigger SOLO calmar-gain/sharpe-cost table AND the JOINT (all kept triggers) vs baseline.

    A trigger is KEPT iff calmar_gain > 0 AND sharpe_cost ≤ sharpe_cost_tol; the FINAL verdict is
    judged on the JOINT (adjustment #4) — triggers interact, so the joint is the real KILL test."""
    base = _metrics(book_returns, periods_per_year)

    def overlaid(trigs):
        if not trigs:
            return base
        return _metrics(apply_overlay(book_returns, exposure_multiplier(trigs, cfg)), periods_per_year)

    per: dict[str, dict] = {}
    for name, trig in triggers_by_name.items():
        solo = overlaid([trig])
        calmar_gain = solo["calmar"] - base["calmar"]
        sharpe_cost = base["sharpe"] - solo["sharpe"]
        per[name] = {"solo": solo, "calmar_gain": calmar_gain, "sharpe_cost": sharpe_cost,
                     "dd_reduction": base["max_dd"] - solo["max_dd"],
                     "verdict": "KEEP" if (calmar_gain > 0 and sharpe_cost <= cfg.sharpe_cost_tol) else "OUT"}

    kept = [t for n, t in triggers_by_name.items() if per[n]["verdict"] == "KEEP"]
    joint = overlaid(kept)
    return {
        "baseline": base,
        "per_trigger": per,
        "joint": {"metrics": joint, "calmar_gain": joint["calmar"] - base["calmar"],
                  "sharpe_cost": base["sharpe"] - joint["sharpe"],
                  "dd_reduction": base["max_dd"] - joint["max_dd"],
                  "verdict": "KEEP" if (joint["calmar"] > base["calmar"]
                                        and base["sharpe"] - joint["sharpe"] <= cfg.sharpe_cost_tol) else "OUT"},
        "periods_per_year": periods_per_year,
    }


# =========================================================================== #
# L4 — cross-book diversifier test (global LOO, double condition)
# =========================================================================== #
def native_loo(returns: dict[str, pd.Series], book_of: dict[str, str]) -> dict[str, float]:
    """LOO-Δ-Sharpe of each sleeve WITHIN its own book only."""
    out: dict[str, float] = {}
    for book in set(book_of.values()):
        sub = {s: returns[s] for s in returns if book_of[s] == book}
        out.update(leave_one_out_delta_sharpe(sub))
    return out


def global_loo(crypto: dict[str, pd.Series], fx: dict[str, pd.Series]) -> dict[str, float]:
    """LOO-Δ-Sharpe of each sleeve on the UNION (global) book."""
    return leave_one_out_delta_sharpe({**crypto, **fx})


def cross_book_diversifiers(crypto: dict[str, pd.Series], fx: dict[str, pd.Series]) -> dict[str, dict]:
    """A sleeve is a cross-book diversifier iff global_LOO > 0 AND global_LOO > native_LOO
    (adjustment #3): net-accretive to the global book AND worth more there than in its native book."""
    book_of = {**{s: "crypto" for s in crypto}, **{s: "fx" for s in fx}}
    allret = {**crypto, **fx}
    g = global_loo(crypto, fx)
    nat = native_loo(allret, book_of)
    return {s: {"global_loo": g[s], "native_loo": nat[s],
                "is_diversifier": g[s] > 0 and g[s] > nat[s]} for s in allret}


# =========================================================================== #
# L5 — read-only dashboard emitter (NOT wired into pulse.py)
# =========================================================================== #
def s9_pulse_emitter(kill_matrix: pd.DataFrame, hrp_weights: dict[str, float],
                     overlay_state: dict) -> dict:
    """Read-only shadow payload (live ρ matrix, HRP weights, overlay state). Registration into the
    live pulse is a documented one-line human merge step — this module never touches pulse.py."""
    return {"kind": "s9_cross_book", "shadow": True,
            "rho_max": {} if kill_matrix.empty else kill_matrix.round(3).to_dict(),
            "hrp": {k: round(float(v), 4) for k, v in hrp_weights.items()},
            "overlay": overlay_state}
