"""Shared HUNT gate for STAGE-B FX/metals sleeves — the §3 honesty rules in one place.

This is NOT a signal generator. Each hunt produces a list of ``(bar_index, side)`` events
(side +1 long / -1 short) on one instrument's bar series; this module realises them HONESTLY
and runs the adversarial gate, so no hunt can silently reintroduce the recurring bugs:
  * charging 1x round-trip on a 2-leg market-neutral spread (the prior flow-search bug),
  * an out-of-sample split by trade COUNT instead of by CALENDAR DATE,
  * declaring an edge that a Monte-Carlo random-entry null would also produce (what exposed
    gold trend-following as bull-beta).

Cadence-agnostic: it works on the instrument's own bar index, so a daily-bar sleeve naturally
steps over weekends (no intraday weekend-straddle trap). Standalone (no hermes import) so a raw
``.venv/bin/python research_fx/<hunt>.py`` run never fails on package path quirks.

Per-asset honest round-trip cost (pass as ``cost_rt``, NOT the crypto 20bps):
  gold / FX majors ~1.6 bps = 0.00016 ; crosses ~2.1 bps = 0.00021 ; silver ~7.1 bps = 0.00071.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _sharpe(x) -> float:
    """Per-period Sharpe = mean / sample-std (ddof=1). 0.0 if undefined."""
    a = np.asarray(list(x), dtype=float)
    if a.size < 2:
        return 0.0
    sd = float(a.std(ddof=1))
    return float(a.mean() / sd) if sd > 0.0 else 0.0


def realize_event_returns(
    close,
    ts,
    events,
    horizon: int,
    cost_rt: float,
    two_leg: bool = False,
    warmup: int = 0,
):
    """Realise ``(bar_index, side)`` events as non-overlapping net-return trades.

    No-overlap (§3.3): greedy min-gap = ``horizon`` — once a trade is entered at bar i it
    blocks any new entry until bar ``i + horizon`` (a run of identical signals becomes ONE
    held position that pays cost once). ``ret_gross = side * (close[i+h]/close[i] - 1)`` with
    ``h = min(horizon, n-1-i)``. Cost is ``cost_rt`` (one round-trip), doubled when
    ``two_leg`` (a market-neutral spread pays both legs). Events are processed in bar order.
    """
    close = np.asarray(close, dtype=float)
    ts = np.asarray(ts)
    n = len(close)
    cost = cost_rt * (2.0 if two_leg else 1.0)
    out: list[dict] = []
    busy_until = -1
    for i, side in sorted(events, key=lambda e: e[0]):
        i = int(i)
        if i < warmup or i < busy_until:
            continue
        h = min(horizon, n - 1 - i)
        if h <= 0:
            continue
        gross = float(side) * (close[i + h] / close[i] - 1.0)
        out.append(
            {
                "entry_i": i,
                "entry_ts": int(ts[i]),
                "side": int(side),
                "ret_gross": float(gross),
                "ret_net": float(gross - cost),
                "horizon": int(h),
            }
        )
        busy_until = i + horizon
    return out


def oos_sharpe_by_date(trades, frac: float = 0.4) -> float:
    """Sharpe of the last ``frac`` of the timeline BY CALENDAR DATE (not by trade count).

    The boundary is ``t_min + (1-frac)*(t_max - t_min)``; trades entered at/after it are the
    out-of-sample set. A date split (vs a count split) is what catches an edge that only lived
    in a friendly early window and decayed later.
    """
    if not trades:
        return 0.0
    ts = np.array([t["entry_ts"] for t in trades], dtype=float)
    r = np.array([t["ret_net"] for t in trades], dtype=float)
    lo, hi = ts.min(), ts.max()
    if hi <= lo:
        return 0.0
    boundary = lo + (1.0 - frac) * (hi - lo)
    return _sharpe(r[ts >= boundary])


def subperiod_positive_frac(trades, n_folds: int = 3) -> float:
    """Fraction of ``n_folds`` contiguous (by-time) sub-periods whose net mean is > 0.

    The gate wants >= 0.67 over >= 3 folds — a real edge is not concentrated in one regime.
    """
    if len(trades) < n_folds:
        return 0.0
    s = sorted(trades, key=lambda t: t["entry_ts"])
    r = np.array([t["ret_net"] for t in s], dtype=float)
    folds = np.array_split(r, n_folds)
    pos = sum(1 for f in folds if f.size and f.mean() > 0.0)
    return pos / n_folds


def mc_random_entry_null(
    close,
    ts,
    observed_mean_net: float,
    n_trades: int,
    horizon: int,
    n_long: int,
    n_short: int,
    cost_rt: float,
    n_sims: int = 1000,
    seed: int = 0,
    two_leg: bool = False,
    warmup: int = 0,
) -> float:
    """p-value that the strategy's mean net beats a RANDOM-ENTRY null on the same path.

    The null draws ``n_trades`` random entry bars with the same long/short mix and horizon, on
    the same price series, and pays the same cost. ``p`` = fraction of sims whose mean net >=
    ``observed_mean_net``. This is the test that exposed gold trend-following (random entry beat
    the signal). Want p < 0.05.
    """
    close = np.asarray(close, dtype=float)
    n = len(close)
    cost = cost_rt * (2.0 if two_leg else 1.0)
    lo, hi = warmup, n - 1 - horizon
    if hi <= lo or n_trades <= 0:
        return 1.0
    base_sides = np.array([1] * max(0, n_long) + [-1] * max(0, n_short), dtype=float)
    if base_sides.size == 0:
        base_sides = np.array([1.0])
    rng = np.random.default_rng(seed)
    beats = 0
    for _ in range(n_sims):
        idx = rng.integers(lo, hi, size=n_trades)
        sides = rng.permutation(np.resize(base_sides, n_trades))
        gross = sides * (close[idx + horizon] / close[idx] - 1.0)
        if float((gross - cost).mean()) >= observed_mean_net:
            beats += 1
    return beats / n_sims


def to_periodic_returns(trades, freq: str = "W") -> pd.Series:
    """Bucket trade net returns onto a common CALENDAR grid (default ISO-weekly) by summing.

    The book marks every sleeve on this shared grid so heterogeneous-cadence sleeves
    (weekly COT, daily GVZ) co-occur in periods — making the orthogonality rho and the
    inverse-vol weights meaningful. Empty periods are simply absent (book.align_sleeves
    treats a sleeve absent in a period as flat).
    """
    if not trades:
        return pd.Series(dtype=float)
    df = pd.DataFrame(trades)
    per = pd.to_datetime(df["entry_ts"], unit="ms").dt.to_period(freq)
    s = df.groupby(per)["ret_net"].sum()
    s.index = s.index.to_timestamp()
    return s.sort_index()


def _dedup_basket_events(events, ref_ts, horizon, lag_days):
    """Greedy no-overlap at the basket level: keep an event only if its entry bar (first ``ref_ts``
    bar >= date + lag_days) is >= ``horizon`` bars after the last kept event's entry bar. Mirrors
    gate_report's busy_until so a PERSISTENT signal (z>=thr for many consecutive bars) becomes ONE
    held position, not many overlapping pseudo-independent 'events' — which would otherwise shrink
    the date-level null's variance to near zero and make it badly anti-conservative."""
    kept = []
    last_bar = -(10**9)
    for d, dr in sorted(events, key=lambda e: e[0]):
        target = int(d) + int(lag_days) * 86_400_000
        i = int(np.searchsorted(ref_ts, target, side="left"))
        if i >= len(ref_ts):
            continue
        if i >= last_bar + horizon:
            kept.append((d, dr))
            last_bar = i
    return kept


def _basket_event_return(date_ms, direction, instruments, horizon, cost, lag_days):
    """Equal-weight basket return for ONE event: mean over instruments of the per-instrument
    net horizon return, entering at the first bar with ts >= date + lag_days. The instrument's
    side = direction * instruments[sym]['sign']. Returns None if no instrument has a usable bar."""
    target = int(date_ms) + int(lag_days) * 86_400_000
    rets = []
    for sym, d in instruments.items():
        close, ts, sign = d["close"], d["ts"], d["sign"]
        # LOOK-AHEAD FIX (2026-07-02, adversarial audit s2-breadth): an instrument whose
        # history STARTS AFTER the event date must be excluded — searchsorted would
        # return idx 0 and "enter" on the instrument's first historical bar, months or
        # years after the event (garbage return, wrong regime). Applies identically to
        # the real leg and the MC-null leg (both call this function).
        if target < int(ts[0]):
            continue
        i = int(np.searchsorted(ts, target, side="left"))
        if i >= len(close) - 1:
            continue
        h = min(horizon, len(close) - 1 - i)
        if h <= 0:
            continue
        c = cost[sym] if isinstance(cost, dict) else cost
        rets.append(float(direction) * float(sign) * (close[i + h] / close[i] - 1.0) - c)
    return float(np.mean(rets)) if rets else None


def event_basket_gate(
    events,
    instruments,
    horizon: int,
    cost,
    n_sims: int = 2000,
    seed: int = 0,
    lag_days: int = 1,
    frac_oos: float = 0.4,
    no_overlap: bool = True,
):
    """Honest gate for a SYNCHRONIZED multi-instrument EVENT sleeve (e.g. one FOMC signal
    expressed across the whole USD complex).

    The fix for correlated multi-instrument sleeves (operator-mandated): N instruments firing
    on the SAME event dates are ~1 correlated bet, NOT N independent trades. So:
      * the sleeve is the PER-EVENT BASKET AVERAGE across instruments -> one return per event
        (Sharpe/OOS are not inflated by counting colinear trades as independent);
      * the MC random-entry null relocates the event DATES JOINTLY -- one random anchor per
        event, shared across all instruments -- preserving cross-instrument correlation so the
        null variance is correct (a per-instrument independent shuffle is anti-conservative).

    ``events`` = list of (event_date_ms, direction in {+1,-1}); ``instruments`` = {sym: {close,
    ts, sign}} where sign is the instrument's response to direction +1. Returns n_events,
    sleeve_mean/sharpe, oos_sharpe (last frac by date), and mc_p (date-level joint null).
    By default ``no_overlap=True`` collapses a persistent signal into non-overlapping held
    positions (>= horizon apart) BEFORE the null — without this, overlapping daily crossings are
    counted as independent events and the null becomes anti-conservative (spuriously tiny p).
    """
    if no_overlap and instruments:
        ref_ts = next(iter(instruments.values()))["ts"]
        events = _dedup_basket_events(events, ref_ts, horizon, lag_days)
    real = [
        (d, _basket_event_return(d, dr, instruments, horizon, cost, lag_days))
        for d, dr in events
    ]
    real = [(d, r) for d, r in real if r is not None]
    if not real:
        return {"n_events": 0, "sleeve_mean": 0.0, "sleeve_sharpe": 0.0,
                "oos_sharpe": 0.0, "mc_p": 1.0, "per_event": []}
    dates = np.array([d for d, _ in real], dtype=float)
    rets = np.array([r for _, r in real], dtype=float)
    real_mean = float(rets.mean())

    # OOS = last frac by calendar date over the per-event series.
    lo, hi = dates.min(), dates.max()
    oos = _sharpe(rets[dates >= lo + (1.0 - frac_oos) * (hi - lo)]) if hi > lo else 0.0

    # Date-level joint null: draw len(events) random anchor ms, keep the direction sequence,
    # recompute the basket return at each random date (same anchor across all instruments).
    dirs = [dr for _, dr in events]
    all_ts = np.concatenate([d["ts"] for d in instruments.values()])
    a_lo = int(all_ts.min())
    a_hi = int(all_ts.max()) - horizon * 2 * 86_400_000  # leave room for the horizon
    rng = np.random.default_rng(seed)
    beats = 0
    if a_hi <= a_lo:
        return {"n_events": len(real), "sleeve_mean": real_mean,
                "sleeve_sharpe": _sharpe(rets), "oos_sharpe": oos, "mc_p": 1.0,
                "per_event": list(map(float, rets))}
    for _ in range(n_sims):
        anchors = rng.integers(a_lo, a_hi, size=len(dirs))
        nr = [
            _basket_event_return(int(a), dr, instruments, horizon, cost, lag_days)
            for a, dr in zip(anchors, dirs)
        ]
        nr = [x for x in nr if x is not None]
        if nr and float(np.mean(nr)) >= real_mean:
            beats += 1
    return {
        "n_events": len(real),
        "sleeve_mean": real_mean,
        "sleeve_sharpe": _sharpe(rets),
        "oos_sharpe": oos,
        "mc_p": beats / n_sims,
        "per_event": list(map(float, rets)),
    }


def gate_report(
    close,
    ts,
    events,
    horizon: int,
    cost_rt: float,
    two_leg: bool = False,
    warmup: int = 0,
    frac_oos: float = 0.4,
    n_folds: int = 3,
    n_sims: int = 1000,
    seed: int = 0,
    winsor_k: int = 1,
) -> dict:
    """Full §3 gate for one instrument's events. Returns the honest metric bundle plus the
    realised trades (so the caller can build the book sleeve via ``to_periodic_returns``).

    ``winsor_net_mean`` drops the ``winsor_k`` largest-|return| trades and re-means — a real
    edge is not one trade or one news-spike tail.
    """
    trades = realize_event_returns(close, ts, events, horizon, cost_rt, two_leg, warmup)
    r = np.array([t["ret_net"] for t in trades], dtype=float)
    n_long = sum(1 for t in trades if t["side"] > 0)
    n_short = sum(1 for t in trades if t["side"] < 0)
    obs_mean = float(r.mean()) if r.size else 0.0

    if r.size > winsor_k:
        order = np.argsort(-np.abs(r))
        keep = np.ones(r.size, dtype=bool)
        keep[order[:winsor_k]] = False
        winsor_mean = float(r[keep].mean())
    else:
        winsor_mean = obs_mean

    mc_p = (
        mc_random_entry_null(
            close, ts, obs_mean, len(trades), horizon, n_long, n_short,
            cost_rt, n_sims, seed, two_leg, warmup,
        )
        if trades
        else 1.0
    )
    return {
        "n_trades": len(trades),
        "n_long": n_long,
        "n_short": n_short,
        "net_mean": obs_mean,
        "net_total": float(r.sum()) if r.size else 0.0,
        "sharpe": _sharpe(r),
        "oos_sharpe": oos_sharpe_by_date(trades, frac_oos),
        "subperiod_positive_frac": subperiod_positive_frac(trades, n_folds),
        "mc_p": mc_p,
        "winsor_net_mean": winsor_mean,
        "trades": trades,
    }
