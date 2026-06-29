"""Stream 6 — gold *regime conditioner* (mid-senior macro).

A point-in-time **state**, never a long-gold trade. The mandate's discovery: the
gold↔real-rate link broke — R²(log-gold, DFII10) collapsed from ~0.48 (2005-21) to
~0.11 (2024-26) in this lake — as central-bank buying (>1000 t/yr since 2022),
de-dollarisation and geopolitics took the wheel. So *which* gold signal works is
itself state-dependent: in a real-rate regime the DFII10 fundamental has bite; in a
flow/CB regime it is noise and positioning/flow leads.

This module builds ``gold_regime_state`` ∈ {REAL_RATE, FLOW_CB} (plus its continuous
``flow_score``) strictly causally — rolling correlations look only backward, the
percentile rank is expanding (no future), DFII10 carries its +1-business-day
publication lag, and quarterly WGC central-bank demand is stamped at its *availability*
date (quarter-end + publication lag), never its reference date. A downstream sleeve
(S5/S8) that consumes the state therefore cannot inherit a look-ahead leak.

Validation doctrine (mandate): the state is accepted ONLY as an *interaction term*
under a shuffle-label / circular-rotation null — it is useless if it improves no
sleeve. It is **never** scored as a standalone directional bet.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Reuse the committed §3 gate engine (no-overlap realisation + random-entry null +
# the synchronized-basket gate) instead of rebuilding it — mandate §2 "ne reconstruis
# RIEN". fxgate is a standalone lab module living in research_fx/ (a sibling of src/).
_RFX = str(Path(__file__).resolve().parents[3] / "research_fx")
if _RFX not in sys.path:
    sys.path.insert(0, _RFX)
import fxgate as _fg  # noqa: E402

__all__ = [
    "expanding_percentile",
    "rolling_corr",
    "cb_demand_pit",
    "gold_regime_state",
    "realrate_shock_events",
    "flow_divergence",
    "combined_etf_ounces",
    "subset_net_mean",
    "shuffle_label_p",
    "circular_rotation_p",
    "conditioning_verdict",
    "cb_demand_signal",
    "forward_return",
    "neutral_pair_forward_return",
    "ic_test",
    "conditional_diversification",
    "dollar_decoupling_state",
]


# --------------------------------------------------------------------------- #
# causal helpers
# --------------------------------------------------------------------------- #
def expanding_percentile(s: pd.Series, min_periods: int = 60) -> pd.Series:
    """Causal expanding percentile rank in [0, 1].

    At index *i* the value is the fraction of *strictly prior* observations
    (indices < i) that are below ``s[i]``. NaN until ``min_periods`` prior
    observations exist. NaNs in ``s`` are excluded from the history (so a leading
    rolling-window warmup of NaNs does not pollute the rank) and the result is
    reindexed back onto ``s``. Looks only backward — truncating the future never
    changes a past value.
    """
    clean = s.dropna()
    if clean.empty:
        return pd.Series(np.nan, index=s.index, name=s.name)
    ranks = clean.expanding(min_periods=min_periods + 1).apply(
        lambda w: float((w[:-1] < w[-1]).mean()), raw=True
    )
    return ranks.reindex(s.index)


def rolling_corr(a: pd.Series, b: pd.Series, window: int) -> pd.Series:
    """Trailing-``window`` Pearson correlation of two aligned series (backward only)."""
    return a.rolling(window).corr(b)


# --------------------------------------------------------------------------- #
# point-in-time central-bank demand
# --------------------------------------------------------------------------- #
def cb_demand_pit(cb_quarterly: pd.DataFrame, lag_days: int = 45) -> pd.Series:
    """Quarterly WGC central-bank demand → annualised, publication-lagged step series.

    ``cb_quarterly`` has columns {year, quarter, tonnes} (the authoritative
    ``wgc_cb_demand_gdt`` GDT series — declared + estimated). Returns a series indexed
    by *availability* date = quarter-end + ``lag_days`` (the WGC GDT print lands ~45 d
    after quarter-end), value = trailing 4-quarter sum (annualised tonnes). The first
    three quarters have no 4-quarter window and are dropped, so the value is only ever
    indexed at a date on which it was actually knowable.
    """
    df = cb_quarterly.sort_values(["year", "quarter"]).reset_index(drop=True)
    q_end = pd.to_datetime(
        dict(year=df["year"], month=(df["quarter"] * 3).astype(int), day=1)
    ) + pd.offsets.MonthEnd(0)
    avail = q_end + pd.Timedelta(days=lag_days)
    cb_ann = df["tonnes"].rolling(4).sum()
    out = pd.Series(cb_ann.to_numpy(), index=pd.DatetimeIndex(avail), name="cb_ann")
    return out.dropna().sort_index()


# --------------------------------------------------------------------------- #
# the integrated regime state
# --------------------------------------------------------------------------- #
def gold_regime_state(
    gold_close: pd.Series,
    dfii10: pd.Series,
    cb_ann_pit: pd.Series,
    *,
    window: int = 252,
    dfii10_lag_days: int = 1,
    min_periods: int = 60,
    flow_threshold: float = 0.5,
) -> pd.DataFrame:
    """Point-in-time gold regime state {REAL_RATE, FLOW_CB}.

    Two slow, exogenous axes, each ranked causally and blended:

    * ``coupling`` = −corr(gold returns, ΔDFII10) over ``window`` days. High (strongly
      negative correlation) ⇒ the real-rate fundamental is live; near zero ⇒ decoupled.
      DFII10 carries a ``dfii10_lag_days`` publication lag (the vintage audit proved
      keyless DFII10 is PIT-honest within rounding, so only the lag is needed).
    * ``cb_ann`` = annualised WGC central-bank demand, already availability-dated.

    ``flow_score`` = ½·pct(cb_ann) + ½·(1 − pct(coupling)) ∈ [0, 1] — high when CB
    buying is historically elevated *and* the real-rate link is historically weak.
    ``state`` = FLOW_CB where ``flow_score`` > ``flow_threshold`` else REAL_RATE; NaN
    while either axis is still in warm-up. Columns: coupling, cb_ann, p_cb, p_coup,
    flow_score, state.
    """
    df = pd.DataFrame({"gold": gold_close.astype(float), "dfii10": dfii10.astype(float)}).sort_index()
    df["dfii10"] = df["dfii10"].ffill()
    df["dfii10_pit"] = df["dfii10"].shift(dfii10_lag_days)
    df["gret"] = np.log(df["gold"]).diff()
    df["ddfii"] = df["dfii10_pit"].diff()

    df["coupling"] = -rolling_corr(df["gret"], df["ddfii"], window)
    df["cb_ann"] = cb_ann_pit.reindex(df.index, method="ffill")

    df["p_cb"] = expanding_percentile(df["cb_ann"], min_periods=min_periods)
    df["p_coup"] = expanding_percentile(df["coupling"], min_periods=min_periods)
    df["flow_score"] = 0.5 * df["p_cb"] + 0.5 * (1.0 - df["p_coup"])

    df["state"] = np.where(df["flow_score"] > flow_threshold, "FLOW_CB", "REAL_RATE")
    df.loc[df["flow_score"].isna(), "state"] = np.nan

    return df[["coupling", "cb_ann", "p_cb", "p_coup", "flow_score", "state"]]


# --------------------------------------------------------------------------- #
# frozen base-signal builders (NOT re-tuned — mandate forbids reviving B2/B5)
# --------------------------------------------------------------------------- #
def realrate_shock_events(
    dfii10: pd.Series,
    bar_dt,
    *,
    z_thr: float = 1.5,
    chg: int = 5,
    zwin: int = 252,
    lag_bdays: int = 1,
    max_stale_days: int = 7,
) -> list[tuple[int, int]]:
    """Frozen B2: DFII10 ``chg``-day shock z-score → (bar_index, side) events.

    ``z = (Δ_chg DFII10 − roll μ) / roll σ`` over ``zwin`` (ddof=1). z ≤ −z_thr ⇒
    side +1 (LONG gold: real rates fell), z ≥ +z_thr ⇒ side −1 (SHORT). Entry is the
    signal date + ``lag_bdays`` business days, mapped to the first ``bar_dt`` at/after
    it (no fill if that bar is > ``max_stale_days`` late); one event per bar. DFII10's
    keyless series is PIT-honest within rounding (vintage audit), so only the entry lag
    is applied — the recipe is verbatim from the prior hunt, not re-fit.
    """
    d = pd.DataFrame({"dfii10": dfii10.astype(float)}).sort_index()
    g = d["dfii10"].diff(chg)
    d["z"] = (g - g.rolling(zwin).mean()) / g.rolling(zwin).std(ddof=1)

    bar_dt = pd.DatetimeIndex(bar_dt)
    bar_np = bar_dt.values
    events: list[tuple[int, int]] = []
    seen: set[int] = set()
    for date, z in d["z"].items():
        if not np.isfinite(z):
            continue
        if z <= -z_thr:
            side = 1
        elif z >= z_thr:
            side = -1
        else:
            continue
        entry = pd.Timestamp(date) + pd.tseries.offsets.BDay(lag_bdays)
        idx = int(np.searchsorted(bar_np, np.datetime64(entry), side="left"))
        if idx >= len(bar_dt) or idx in seen:
            continue
        if (pd.Timestamp(bar_dt[idx]) - entry).days > max_stale_days:
            continue
        seen.add(idx)
        events.append((idx, side))
    events.sort(key=lambda e: e[0])
    return events


def flow_divergence(
    oz: pd.Series, price: pd.Series, *, lookback: int = 60, zwin: int = 60
) -> pd.DataFrame:
    """Frozen B5 variant B: flow-vs-price divergence ``div = zf − zp`` (causal).

    ``zf`` = z-score of the ``lookback``-day growth of ounces-in-trust; ``zp`` = z-score
    of the ``lookback``-day return of the gold price proxy; both z-scored over ``zwin``
    (ddof=1). ``div`` > 0 ⇒ flow accumulating faster than price (a candidate LONG); < 0
    ⇒ price outrunning flow. Expressed as a *state*, never a directional call (the prior
    directional B5 was refuted — flow lags price). Columns: zf, zp, div.
    """
    df = pd.DataFrame({"oz": oz.astype(float), "px": price.astype(float)}).sort_index()
    g = df["oz"] / df["oz"].shift(lookback) - 1.0
    df["zf"] = (g - g.rolling(zwin).mean()) / g.rolling(zwin).std(ddof=1)
    rg = df["px"] / df["px"].shift(lookback) - 1.0
    df["zp"] = (rg - rg.rolling(zwin).mean()) / rg.rolling(zwin).std(ddof=1)
    df["div"] = df["zf"] - df["zp"]
    return df[["zf", "zp", "div"]]


def combined_etf_ounces(
    gld_oz: pd.Series, iau_shares: pd.Series, iau_nav: pd.Series, gold_px: pd.Series
) -> pd.Series:
    """Combined GLD+IAU ounces-in-trust. IAU is held as shares × NAV ($/share) = AUM
    ($), converted to ounces at the gold spot ($/oz); added to GLD's reported ounces.
    The prior B5 used GLD alone — folding in IAU (now in the lake) is the one genuine
    new data increment over the refuted test.
    """
    iau_oz = iau_shares.astype(float) * iau_nav.astype(float) / gold_px.astype(float)
    return (gld_oz.astype(float) + iau_oz).rename("combined_oz")


# --------------------------------------------------------------------------- #
# the conditioning gate — shuffle-label null + base/cond/complement triplet
# --------------------------------------------------------------------------- #
def subset_net_mean(
    close, ts, events, horizon: int, cost_rt: float, *, two_leg: bool = False, warmup: int = 0
) -> tuple[float, int]:
    """(per-trade net mean, n_trades) for a subset of events, realised through the
    committed no-overlap engine (``fxgate.realize_event_returns``) so a sparse subset
    is costed and de-overlapped identically to the full set."""
    trades = _fg.realize_event_returns(
        np.asarray(close, dtype=float), np.asarray(ts), list(events), horizon, cost_rt, two_leg, warmup
    )
    if not trades:
        return 0.0, 0
    r = np.array([t["ret_net"] for t in trades], dtype=float)
    return float(r.mean()), len(trades)


def shuffle_label_p(
    close, ts, base_events, n_take: int, obs_net: float, horizon: int, cost_rt: float,
    *, n_perm: int = 5000, seed: int = 0,
) -> float:
    """X3 shuffled-label null (verbatim): one-sided p that a RANDOM same-size subsample
    of the base events matches/beats the conditioned subset's net mean. Each draw is
    re-realised through the no-overlap engine (so the no-overlap interaction is
    apples-to-apples). p ≥ 0.05 ⇒ the regime label is indistinguishable from a random
    thinning of the same base edge (the X3 death)."""
    close = np.asarray(close, dtype=float)
    ts = np.asarray(ts)
    n_base = len(base_events)
    if n_take <= 0 or n_take > n_base:
        return 1.0
    rng = np.random.default_rng(seed)
    idx_all = np.arange(n_base)
    beats = 0
    for _ in range(n_perm):
        chosen = rng.choice(idx_all, size=n_take, replace=False)
        net, nt = subset_net_mean(close, ts, [base_events[i] for i in chosen], horizon, cost_rt)
        if nt > 0 and net >= obs_net:
            beats += 1
    return beats / n_perm


def circular_rotation_p(
    close, ts, events, regime_on, obs_net: float, horizon: int, cost_rt: float,
    *, max_rot: int | None = None, seed: int = 0,
) -> float:
    """Persistence-preserving robustness null: circularly rotate the regime mask over the
    event order and recompute the conditioned net. ``np.roll`` preserves the regime-on
    COUNT (so the conditioned-subset size is held fixed) and the cyclic adjacency — run
    lengths are preserved up to the wrap seam — so the regime's slowness is largely
    respected, unlike a random subsample. p = fraction of rotations matching/beating the
    real alignment. Reported alongside, not gating (the mandated bar is shuffle_p)."""
    regime_on = np.asarray(regime_on, dtype=bool)
    n = len(events)
    offsets = list(range(1, n))
    if max_rot and len(offsets) > max_rot:
        rng = np.random.default_rng(seed)
        offsets = sorted(int(o) for o in rng.choice(offsets, size=max_rot, replace=False))
    beats = total = 0
    for off in offsets:
        rolled = np.roll(regime_on, off)
        sub = [events[i] for i in range(n) if rolled[i]]
        net, nt = subset_net_mean(close, ts, sub, horizon, cost_rt)
        if nt > 0:
            total += 1
            if net >= obs_net:
                beats += 1
    return beats / total if total else 1.0


def conditioning_verdict(
    close, ts, events, regime_on, horizon: int, cost_rt: float,
    *, n_perm: int = 5000, n_sims: int = 2000, seed: int = 0,
) -> dict:
    """The §6.2 gate: does conditioning the frozen base events on the regime add edge?

    Reports the base / conditioned / complement net-mean triplet, the shuffled-label p,
    a circular-rotation robustness p, and the conditioned subset's own random-entry null
    (``fxgate.mc_random_entry_null``). ``survives`` requires ALL of: conditioned beats
    base, conditioned beats complement (catches the X3 'complement wins' failure),
    shuffle_p < 0.05, and cond_mc_p < 0.05 — the mandate's high bar.
    """
    close = np.asarray(close, dtype=float)
    ts = np.asarray(ts)
    regime_on = np.asarray(regime_on, dtype=bool)
    cond_events = [e for e, m in zip(events, regime_on) if m]
    comp_events = [e for e, m in zip(events, regime_on) if not m]

    base_net, n_base = subset_net_mean(close, ts, events, horizon, cost_rt)
    comp_net, n_comp = subset_net_mean(close, ts, comp_events, horizon, cost_rt)

    # Realise the conditioned subset ONCE so its net mean, TRADE count and side mix are
    # mutually consistent — the random-entry null must use the realised trades' long/short
    # split (not the pre-overlap event tally, which can make n_short negative -> an all-long
    # null unrelated to the subset being tested).
    cond_trades = _fg.realize_event_returns(close, ts, cond_events, horizon, cost_rt)
    n_cond = len(cond_trades)
    cond_net = float(np.mean([t["ret_net"] for t in cond_trades])) if cond_trades else 0.0
    cond_long = sum(1 for t in cond_trades if t["side"] > 0)

    # X3-faithful: the null draws a random subsample of the SAME number of base EVENTS the
    # regime selected (len(cond_events)), realised identically — NOT the post-no-overlap
    # trade count, which would mis-size the null when no-overlap collapses runs differently.
    p_shuffle = shuffle_label_p(
        close, ts, events, len(cond_events), cond_net, horizon, cost_rt, n_perm=n_perm, seed=seed
    )
    p_circ = circular_rotation_p(close, ts, events, regime_on, cond_net, horizon, cost_rt, seed=seed)
    cond_mc_p = (
        _fg.mc_random_entry_null(
            close, ts, cond_net, n_cond, horizon, cond_long, n_cond - cond_long, cost_rt, n_sims, seed
        )
        if n_cond
        else 1.0
    )
    survives = bool(cond_net > base_net and cond_net > comp_net and p_shuffle < 0.05 and cond_mc_p < 0.05)
    return {
        "base_net": base_net,
        "cond_net": cond_net,
        "comp_net": comp_net,
        "shuffle_p": p_shuffle,
        "circular_p": p_circ,
        "cond_mc_p": cond_mc_p,
        "n_base": n_base,
        "n_cond": n_cond,
        "n_comp": n_comp,
        "n_cond_events": len(cond_events),
        "survives": survives,
    }


# =========================================================================== #
# v2 — central-bank flow AS A SIGNAL (not a label) + cross-book + dollar state
# =========================================================================== #
def cb_demand_signal(cb_quarterly: pd.DataFrame, *, mode: str = "level", lag_days: int = 45) -> pd.Series:
    """PIT continuous central-bank demand signal, availability-dated (the *cause* of the
    decoupling, used as a signal rather than a binary label).

    ``mode='level'`` = trailing-4-quarter annualised tonnes (= ``cb_demand_pit``);
    ``mode='accel'`` = quarter-on-quarter change of that level (Δ). The official buyer is
    price-insensitive and persistent, so the a-priori mechanism is a *slow forward drift*
    — the "flow that precedes price" candidate (unlike ETF flow, which lags).
    """
    level = cb_demand_pit(cb_quarterly, lag_days=lag_days)
    if mode == "level":
        return level.rename("cb_level")
    if mode == "accel":
        return level.diff().rename("cb_accel")
    raise ValueError(f"unknown mode: {mode!r} (use 'level' or 'accel')")


def forward_return(px: pd.Series, horizon: int) -> pd.Series:
    """Forward simple return over ``horizon`` rows: px[t+h]/px[t] − 1 (NaN past the end)."""
    px = px.astype(float)
    return px.shift(-horizon) / px - 1.0


def neutral_pair_forward_return(long_px: pd.Series, short_px: pd.Series, horizon: int) -> pd.Series:
    """Forward return of a beta-cancelled long/short pair: fwd(long) − fwd(short). For the CB
    signal the mechanism-driven neutral leg is gold-vs-silver (central banks buy gold, not
    silver), so this removes the common precious-metals / real-rate / bull-beta and isolates
    whether the signal predicts gold *outperformance*."""
    return forward_return(long_px, horizon) - forward_return(short_px, horizon)


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = pd.Series(a).rank().to_numpy()
    rb = pd.Series(b).rank().to_numpy()
    if ra.std() == 0 or rb.std() == 0:
        return 0.0
    return float(np.corrcoef(ra, rb)[0, 1])


def ic_test(signal: pd.Series, fwd_ret: pd.Series, *, n_perm: int = 2000, seed: int = 0,
            oos_frac: float = 0.4) -> dict:
    """Information-coefficient gate for a slow continuous predictor.

    Spearman IC of ``signal`` vs ``fwd_ret`` on their common (dropna) index. The null is a
    random **circular shift** of the signal, which preserves the signal's *entire*
    autocorrelation structure (the honest null for a persistent series — a plain shuffle would
    be wildly anti-conservative) and breaks only the signal↔target alignment. One-sided
    p = P(IC_shift ≥ IC_obs) for the pre-registered positive hypothesis. ``oos_ic`` is the IC
    on the last ``oos_frac`` of the timeline. Returns ic, p, n, oos_ic.

    CALIBRATION CAVEAT (measured, N=60): this null is well-calibrated for **stationary** inputs
    (false-positive rate ~0.07 at α=0.05) but **anti-conservative for TRENDING/level inputs**
    (~0.16) — a circularly-shifted trend still trends, so two independent trends spuriously
    rank-align. Apply only to differenced/stationary signals; a level (e.g. a cumulative demand
    level) must be differenced first, or its p is untrustworthy.
    """
    df = pd.concat([signal.rename("s"), fwd_ret.rename("f")], axis=1).dropna().sort_index()
    n = len(df)
    if n < 8:
        return {"ic": 0.0, "p": 1.0, "n": n, "oos_ic": 0.0}
    s = df["s"].to_numpy(dtype=float)
    f = df["f"].to_numpy(dtype=float)
    obs = _spearman(s, f)
    rng = np.random.default_rng(seed)
    beats = 0
    for k in rng.integers(1, n, size=n_perm):
        if _spearman(np.roll(s, int(k)), f) >= obs:
            beats += 1
    cut = int(n * (1.0 - oos_frac))
    oos = _spearman(s[cut:], f[cut:]) if n - cut >= 4 else 0.0
    return {"ic": obs, "p": beats / n_perm, "n": n, "oos_ic": oos}


def conditional_diversification(asset_ret: pd.Series, book_ret: pd.Series, regime_on: pd.Series,
                                *, n_perm: int = 2000, seed: int = 0) -> dict:
    """Is the asset a *regime-conditional* hedge to the book? (gold as a cross-book crypto
    diversifier — vol/tail, not direction).

    Correlation and beta of ``asset_ret`` to ``book_ret`` inside vs outside ``regime_on``.
    Hypothesis: the regime marks when gold genuinely diversifies, i.e. corr_on < corr_off
    (delta < 0). Null: random **circular shift** of the regime mask (preserves its run-length /
    persistence), p = P(perm delta ≤ obs delta). A real conditional hedge has p < 0.05; a
    regime that does not actually time the hedge has p ≈ 0.5.
    """
    df = pd.concat([asset_ret.rename("a"), book_ret.rename("b"),
                    regime_on.rename("r")], axis=1).dropna().sort_index()
    a = df["a"].to_numpy(float)
    b = df["b"].to_numpy(float)
    r = df["r"].to_numpy().astype(bool)

    def corr(x, y):
        if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
            return 0.0
        return float(np.corrcoef(x, y)[0, 1])

    def beta(x, y):
        if len(y) < 3 or np.var(y, ddof=1) == 0:
            return 0.0
        return float(np.cov(x, y, ddof=1)[0, 1] / np.var(y, ddof=1))

    corr_on, corr_off = corr(a[r], b[r]), corr(a[~r], b[~r])
    delta = corr_on - corr_off
    rng = np.random.default_rng(seed)
    n = len(r)
    beats = 0
    for _ in range(n_perm):
        rr = np.roll(r, int(rng.integers(1, n)))
        if corr(a[rr], b[rr]) - corr(a[~rr], b[~rr]) <= delta:
            beats += 1
    return {
        "corr_on": corr_on, "corr_off": corr_off,
        "beta_on": beta(a[r], b[r]), "beta_off": beta(a[~r], b[~r]),
        "delta": delta, "p": beats / n_perm, "n_on": int(r.sum()), "n_off": int((~r).sum()),
    }


def dollar_decoupling_state(gold: pd.Series, dxy: pd.Series, *, window: int = 252,
                            min_periods: int = 60, decouple_threshold: float = 0.5) -> pd.DataFrame:
    """A *distinct* de-dollarisation axis: the rolling DXY-gold correlation regime.

    Gold normally rises when the dollar falls (corr strongly negative). The 2024-25 "gold AND
    the dollar rise together" anomaly is a de-dollarisation signature — corr drifts toward zero
    / positive. ``state = DECOUPLED`` when the causal expanding percentile of the rolling corr
    exceeds ``decouple_threshold`` else ``NORMAL``. A conditioner/feature for S8, never a trade.
    """
    df = pd.DataFrame({"gold": gold.astype(float), "dxy": dxy.astype(float)}).sort_index()
    df["gret"] = np.log(df["gold"]).diff()
    df["dret"] = np.log(df["dxy"]).diff()
    df["corr_dxy"] = rolling_corr(df["gret"], df["dret"], window)
    df["p_corr"] = expanding_percentile(df["corr_dxy"], min_periods=min_periods)
    df["state"] = np.where(df["p_corr"] > decouple_threshold, "DECOUPLED", "NORMAL")
    df.loc[df["p_corr"].isna(), "state"] = np.nan
    return df[["corr_dxy", "p_corr", "state"]]
