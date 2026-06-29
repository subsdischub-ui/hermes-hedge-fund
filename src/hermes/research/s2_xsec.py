"""Stream 2 v-next — cross-sectional, dollar-neutral (market-neutral) sleeves.

Doctrine (inherited): every DIRECTIONAL crypto edge in this lab has died of
bull-beta — a universal mechanism. The only proven antidote is MARKET-NEUTRAL
(Stream 5). A single sleeve is not a book; we need several weak, mutually
orthogonal long-short sleeves. This module is the engine + the pre-registered
family (2A funding, 2D vol-dispersion, 2B OI-growth) and the funding_squeeze 4h
bull/bear re-test.

The recurring failure mode is look-ahead, and a NEW cross-sectional backtester is
exactly where a composition bug manufactures a false POSITIVE. So every function
here is causal by construction (signal at t uses only data <= t; the forward target
is bars (t, t+stride]) and the engine is pinned by the L0 leak-injection battery in
the tests BEFORE any sleeve is run. Pure functions only; the frozen DuckDB lake is a
thin, separately-tested I/O seam.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from hermes.research.s2_positioning import causal_zscore, oi_acceleration_zscore

__all__ = [
    "cross_sectional_weights",
    "xsec_backtest",
    "mc_random_sign_pvalue",
    "realized_vol_panel",
    "oi_accel_panel",
    "funding_settled_per_bar",
    "funding_signal_on_grid",
    "bull_bear_regime",
    "daily_regime_on_grid",
    "funding_squeeze_directional",
    "build_panel",
    "load_native_funding",
    "build_venue_funding",
]


# --------------------------------------------------------------------------- #
# cross-sectional weight construction — dollar-neutral long-short
# --------------------------------------------------------------------------- #
def cross_sectional_weights(
    signal: np.ndarray,
    *,
    top_frac: float = 1 / 3,
    bottom_frac: float = 1 / 3,
    min_names: int = 4,
) -> np.ndarray:
    """Long the top-frac of the cross-section by ``signal``, short the bottom-frac.

    Equal weight within each leg; the long leg sums to +0.5 and the short leg to
    -0.5, so the book is dollar-neutral (Σw = 0) and gross-1 (Σ|w| = 1). Names with a
    NaN signal are UNAVAILABLE at this rebalance (variable universe) and stay flat —
    only listed names are ranked. Below ``min_names`` available names, or when the
    available signals are degenerate (all equal, no high/low to separate), NO bet is
    taken (all zeros) — never force a long-short on a skeleton or tie-only universe.
    """
    s = np.asarray(signal, dtype=float)
    n = s.size
    w = np.zeros(n, dtype=float)
    avail = np.where(np.isfinite(s))[0]
    if avail.size < min_names:
        return w
    vals = s[avail]
    if float(np.max(vals)) == float(np.min(vals)):  # degenerate cross-section
        return w

    order = avail[np.argsort(-vals, kind="stable")]  # high signal first
    n_avail = avail.size
    n_long = max(1, int(n_avail * top_frac))
    n_short = max(1, int(n_avail * bottom_frac))
    # never let the legs overlap on a small universe
    while n_long + n_short > n_avail:
        if n_long >= n_short and n_long > 1:
            n_long -= 1
        elif n_short > 1:
            n_short -= 1
        else:
            return w  # cannot form two disjoint legs

    longs = order[:n_long]
    shorts = order[-n_short:]
    w[longs] = 0.5 / n_long
    w[shorts] = -0.5 / n_short
    return w


# --------------------------------------------------------------------------- #
# the cross-sectional engine
# --------------------------------------------------------------------------- #
def xsec_backtest(
    close_panel: pd.DataFrame,
    signal_panel: pd.DataFrame,
    funding_per_bar: pd.DataFrame | None,
    *,
    stride: int,
    top_frac: float = 1 / 3,
    bottom_frac: float = 1 / 3,
    cost_bps: float = 17.5,
    min_names: int = 4,
) -> dict:
    """Backtest a cross-sectional long-short on a panel, rebalanced every ``stride`` bars.

    At each rebalance bar ``t`` (t = 0, stride, 2·stride, …): rank the cross-section by
    ``signal_panel`` (causal — values at t use only data <= t), form dollar-neutral
    weights, hold for ``stride`` bars and realise:

      * gross price return  g_t = Σ_i w_i · (close[t+stride]/close[t] − 1)
      * funding PnL         f_t = Σ_i (−w_i) · (funding settled over bars (t, t+stride])
                            (a LONG of a positive-funding name PAYS; the carry is summed
                            ONCE over the holding window from per-settlement funding —
                            never per-bar-forward-filled, which would double-count)
      * turnover cost       c_t = (cost_bps/1e4) · Σ_i |w_i − w_i(prev)|  (both legs)

    net_t = g_t + f_t − c_t. A name with a non-finite forward target (delisted in the
    window) contributes nothing (cannot be realised), never NaN. Returns a dict with
    ``returns`` (net), ``gross_returns`` (price-only, for the random-sign null),
    ``weights`` (rebalance × symbol), and ``n_rebals``. Inputs are reindexed onto
    ``close_panel``'s grid so callers need not pre-align column order.
    """
    close = close_panel
    cols = list(close.columns)
    idx = close.index
    sig = signal_panel.reindex(index=idx, columns=cols)
    fund = (
        funding_per_bar.reindex(index=idx, columns=cols).fillna(0.0)
        if funding_per_bar is not None
        else None
    )

    C = close.to_numpy(dtype=float)
    S = sig.to_numpy(dtype=float)
    F = fund.to_numpy(dtype=float) if fund is not None else None
    n = C.shape[0]

    rebal_ts: list = []
    net_list: list[float] = []
    gross_list: list[float] = []
    weights: list[np.ndarray] = []
    prev_w = np.zeros(len(cols), dtype=float)

    for t in range(0, n - stride, stride):
        w = cross_sectional_weights(
            S[t], top_frac=top_frac, bottom_frac=bottom_frac, min_names=min_names
        )
        fwd = C[t + stride] / C[t] - 1.0
        fwd_ok = np.where(np.isfinite(fwd), fwd, 0.0)
        gross = float(np.sum(w * fwd_ok))

        funding_pnl = 0.0
        if F is not None:
            fwin = np.nansum(F[t + 1 : t + stride + 1], axis=0)
            funding_pnl = float(np.sum(-w * fwin))

        turnover = float(np.sum(np.abs(w - prev_w)))
        cost = (cost_bps / 1e4) * turnover
        net = gross + funding_pnl - cost

        rebal_ts.append(idx[t])
        gross_list.append(gross)
        net_list.append(net)
        weights.append(w)
        prev_w = w

    return {
        "returns": pd.Series(net_list, index=pd.Index(rebal_ts, name="ts")),
        "gross_returns": pd.Series(gross_list, index=pd.Index(rebal_ts, name="ts")),
        "weights": pd.DataFrame(weights, index=pd.Index(rebal_ts, name="ts"), columns=cols),
        "n_rebals": len(net_list),
    }


# --------------------------------------------------------------------------- #
# MC random-SIGN null — the FLOOR test (the long-short SIGN is the hypothesis)
# --------------------------------------------------------------------------- #
def mc_random_sign_pvalue(
    gross_returns, n_sims: int = 2000, seed: int = 0
) -> float:
    """One-sided p-value: P(mean(random-signed stream) >= mean(real stream)).

    The hypothesis a cross-sectional sleeve makes is the SIGN of the long-short bet, so
    the right null flips the sign of each rebalance's return independently (s_t ∈ {±1})
    and asks how often pure sign-noise matches the realised mean. p < 0.05 is the FLOOR,
    not the bar. Operates on the GROSS (price-only) stream so it tests the directional
    edge, not the cost drag.
    """
    g = np.asarray(list(gross_returns), dtype=float)
    g = g[np.isfinite(g)]
    if g.size < 2:
        return 1.0
    real = float(g.mean())
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(n_sims, g.size))
    sim_means = (signs * g).mean(axis=1)
    return float(np.mean(sim_means >= real))


# --------------------------------------------------------------------------- #
# causal signal builders
# --------------------------------------------------------------------------- #
def realized_vol_panel(close_panel: pd.DataFrame, window: int) -> pd.DataFrame:
    """Per-column trailing realised volatility = std (ddof=1) of the trailing ``window``
    log-returns ending at and including t. Causal (a rolling window looks only backward);
    the first ``window`` bars are NaN (warmup). The 2D sleeve longs high realised-vol,
    shorts low — a dollar-neutral dispersion bet.
    """
    logret = np.log(close_panel.astype(float)).diff()
    return logret.rolling(window).std(ddof=1)


def oi_accel_panel(oi_panel: pd.DataFrame, window: int) -> pd.DataFrame:
    """Per-column OI ACCELERATION = causal z-score of open-interest growth (reuses the
    tested ``s2_positioning.oi_acceleration_zscore``). The 2B sleeve longs the strongest
    OI acceleration, shorts the weakest. Causal; NaN warmup."""
    return oi_panel.apply(lambda col: oi_acceleration_zscore(col, window))


def funding_settled_per_bar(settlements: pd.Series, grid: pd.Index) -> pd.Series:
    """Bucket each native funding SETTLEMENT into the one grid bar [grid_i, grid_{i+1})
    that contains it, summed there. This is the per-bar carry used for cost — summing it
    over a holding window charges each settlement EXACTLY ONCE (never the forward-filled
    rate, which would double-count). Settlements before the grid are dropped; settlements
    at/after the last bar fall in the last bar. Returns a Series indexed by ``grid``.
    """
    out = pd.Series(0.0, index=grid)
    g = np.asarray(grid, dtype="int64")
    if g.size == 0 or settlements.empty:
        return out
    s_ts = np.asarray(settlements.index, dtype="int64")
    s_val = np.asarray(settlements.to_numpy(), dtype=float)
    bar = np.searchsorted(g, s_ts, side="right") - 1  # bar whose open <= settlement
    valid = bar >= 0
    np.add.at(out.values, bar[valid], s_val[valid])
    return out


def funding_signal_on_grid(
    settlements: pd.Series, grid: pd.Index, window: int
) -> pd.Series:
    """funding-z as of each grid bar: a causal z-score over the SETTLEMENT series (the
    last ``window`` settled prints), as-of-backward onto the grid. Each grid bar carries
    the z of the most recent settlement at or before it — never a future settlement
    (the look-ahead guard). Warmup/zero-variance settlements yield NaN.
    """
    z = causal_zscore(settlements, window)
    zf = pd.DataFrame(
        {"ts": np.asarray(z.index, dtype="int64"), "z": z.to_numpy(dtype=float)}
    ).sort_values("ts")
    gf = pd.DataFrame({"ts": np.asarray(grid, dtype="int64")}).sort_values("ts")
    merged = pd.merge_asof(gf, zf, on="ts", direction="backward")
    return pd.Series(merged["z"].to_numpy(), index=pd.Index(np.asarray(grid), name="ts"))


def bull_bear_regime(close: pd.Series, ma_window: int = 200) -> pd.Series:
    """Causal bull/bear label from price vs its trailing simple MA: close > MA -> 'bull',
    close <= MA -> 'bear'. Before a full ``ma_window`` the MA is undefined -> 'unknown'
    (never silently folded into a regime — that would be a look-ahead). This is the
    explicit bull/bear axis the funding_squeeze 4h re-test splits on (>MA200 / <MA200).
    """
    s = pd.Series(close, dtype=float)
    ma = s.rolling(ma_window).mean()
    out = pd.Series("unknown", index=s.index, dtype=object)
    valid = ma.notna()
    out[valid & (s > ma)] = "bull"
    out[valid & (s <= ma)] = "bear"
    return out


def daily_regime_on_grid(close_4h: pd.Series, ma_window: int = 200) -> pd.Series:
    """Map a causal DAILY bull/bear regime onto an intraday (e.g. 4h) grid, lagged so a
    bar on day D carries the regime knowable at D's OPEN (from days <= D-1).

    The driver originally built the daily regime from each day's LAST (end-of-day) close
    and applied it to that same day's earlier bars — a ~20h same-day look-ahead at MA200
    crossings (adversarial review). Here the daily regime is computed on the resampled
    daily close, LAGGED one day (``shift(1)``), then broadcast to every intraday bar of
    its day. Index = ``close_4h``'s index; warmup/first-day bars are ``"unknown"``.
    """
    s = pd.Series(close_4h, dtype=float)
    idx_ts = pd.to_datetime(np.asarray(s.index), unit="ms", utc=True)
    daily = pd.Series(s.to_numpy(), index=idx_ts).groupby(pd.Grouper(freq="D")).last().dropna()
    reg = bull_bear_regime(daily, ma_window).shift(1)  # day D uses the label as of D-1
    bar_day = idx_ts.floor("D")
    mapped = reg.reindex(bar_day).to_numpy()
    out = pd.Series(mapped, index=s.index, dtype=object)
    return out.where(out.notna(), "unknown")


# --------------------------------------------------------------------------- #
# funding_squeeze directional 4h re-test, bull vs bear (the live-edge audit)
# --------------------------------------------------------------------------- #
def _ret_stats(rets: list[float]) -> dict:
    a = np.asarray(rets, dtype=float)
    n = int(a.size)
    if n == 0:
        return {"n": 0, "mean": 0.0, "sharpe": 0.0, "win_rate": 0.0, "median": 0.0}
    sd = float(a.std(ddof=1)) if n >= 2 else 0.0
    return {
        "n": n,
        "mean": float(a.mean()),
        "sharpe": float(a.mean() / sd) if sd > 0.0 else 0.0,
        "win_rate": float((a > 0).mean()),
        "median": float(np.median(a)),
    }


def funding_squeeze_directional(
    close: pd.Series,
    funding_z: pd.Series,
    regime: pd.Series,
    funding_per_bar: pd.Series,
    *,
    z_thr: float = 2.0,
    hold_bars: int = 24,
    cost_bps: float = 17.5,
) -> dict:
    """Re-test the lab's ONLY live directional edge (``funding_squeeze_continuation``) on
    the deep 4h history, split bull vs bear.

    Arm a LONG when ``funding_z`` >= ``z_thr``; hold ``hold_bars`` bars, no-overlap (one
    position per hold window). The net per-trade return is the forward price return over
    the hold, MINUS the funding carry the long pays (Σ funding settled in the hold window,
    from ``funding_per_bar``), MINUS a round-trip cost (2·cost_bps). Each trade is tagged
    by the regime AT ENTRY. Returns per-regime stats under keys ``"bull"``, ``"bear"`` and
    ``"all"``.

    The governance implication is the whole point: if the edge lives only in the bull leg
    and the sign flips (or collapses) in bear, it is **bull-beta**, not an edge — meaning
    the live book has no confirmed edge. All inputs are aligned, same-length Series.
    """
    c = np.asarray(close, dtype=float)
    fz = np.asarray(funding_z, dtype=float)
    reg = np.asarray(regime, dtype=object)
    fpb = np.asarray(funding_per_bar, dtype=float)
    n = c.size
    rt_cost = 2.0 * cost_bps / 1e4

    by_regime: dict[str, list[float]] = {}
    all_rets: list[float] = []
    busy = -1
    for t in range(0, n - hold_bars):
        if not np.isfinite(fz[t]) or fz[t] < z_thr or t < busy:
            continue
        fwd = c[t + hold_bars] / c[t] - 1.0
        if not np.isfinite(fwd):
            continue
        carry = float(np.nansum(fpb[t + 1 : t + hold_bars + 1]))  # long pays
        net = fwd - carry - rt_cost
        by_regime.setdefault(str(reg[t]), []).append(net)
        all_rets.append(net)
        busy = t + hold_bars

    out = {r: _ret_stats(v) for r, v in by_regime.items()}
    for r in ("bull", "bear"):
        out.setdefault(r, _ret_stats([]))
    out["all"] = _ret_stats(all_rets)
    return out


# --------------------------------------------------------------------------- #
# lake I/O seam (thin; the frozen DuckDB lake is the only data source, read-only)
# --------------------------------------------------------------------------- #
_LAKE_VIEWS = {
    "binance_klines_4h",
    "binance_funding_native",
    "binance_funding_on_4h",
    "coinglass_open_interest",
    "coinglass_funding_rate",
    "coinglass_multi_funding_rate",
    "coinglass_multi_open_interest",
}
_VALUE_COLS = {"open", "high", "low", "close", "volume"}


def build_panel(con, symbols, *, view: str, interval: str, value_col: str = "close") -> pd.DataFrame:
    """Wide panel (index = ts ms, columns = ``symbols`` in requested order, values =
    ``value_col``) for one interval. Symbols absent at a ts are NaN — the honest
    variable-universe treatment (a name not yet listed contributes nothing). ``view`` and
    ``value_col`` are allowlisted (no arbitrary SQL); symbols/interval are bound params.
    """
    if view not in _LAKE_VIEWS:
        raise ValueError(f"view not in allowlist: {view!r}")
    if value_col not in _VALUE_COLS:
        raise ValueError(f"value_col not allowed: {value_col!r}")
    syms = list(symbols)
    ph = ",".join(["?"] * len(syms))
    q = (
        f"SELECT ts, symbol, {value_col} AS v FROM {view} "
        f"WHERE symbol IN ({ph}) AND interval = ? ORDER BY ts"
    )
    df = con.execute(q, [*syms, interval]).df()
    if df.empty:
        return pd.DataFrame(columns=syms)
    wide = df.pivot(index="ts", columns="symbol", values="v").sort_index()
    return wide.reindex(columns=[s for s in syms if s in wide.columns])


def load_native_funding(con, symbols) -> dict[str, pd.Series]:
    """Per-symbol native funding SETTLEMENT series (index = settlement ts ms, value =
    funding_rate as a fraction), deep from inception. This is the source of truth for
    both the funding signal (z over the settlement series) and the funding cost (summed
    per settlement) — NEVER the forward-filled ``binance_funding_on_4h`` rate.
    """
    out: dict[str, pd.Series] = {}
    for s in symbols:
        df = con.execute(
            "SELECT ts, funding_rate FROM binance_funding_native WHERE symbol = ? ORDER BY ts",
            [s],
        ).df()
        out[s] = pd.Series(df["funding_rate"].to_numpy(dtype=float), index=df["ts"].to_numpy())
    return out


def build_venue_funding(con, coin: str, *, interval: str = "1d") -> pd.DataFrame:
    """Cross-venue funding for one coin: wide panel (index = ts, columns = exchange,
    value = funding ``close``) from ``coinglass_multi_funding_rate``. The 2.3 dispersion
    signal is the cross-venue spread; units are CoinGlass-native and CONSISTENT across
    venues, so the dispersion is meaningful even though it differs from the Binance
    fraction unit.
    """
    df = con.execute(
        "SELECT ts, exchange, close AS v FROM coinglass_multi_funding_rate "
        "WHERE symbol = ? AND interval = ? ORDER BY ts",
        [coin, interval],
    ).df()
    if df.empty:
        return pd.DataFrame()
    return df.pivot(index="ts", columns="exchange", values="v").sort_index()
