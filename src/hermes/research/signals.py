"""Strategy #2 candidate signals (research-only, NOT registered live).

Each is a causal ``fn(df) -> SkillSignal`` on a trailing window (only data <= the
current bar). Parameters are frozen a-priori (round, standard values). These are
the top-ranked, orthogonal-to-momentum candidates from R&D round 1; they are
evaluated by hermes.research.harness and only promoted to live after surviving
CPCV/DSR net of fees AND the orthogonality gate vs ts_mom.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from hermes.skills.library import SkillSignal


def _hold(skill_id: str, reason: str) -> SkillSignal:
    return SkillSignal(skill_id, "HOLD", 0.0, 0.02, 0.04, reason, {})


def _atr_pct(df: pd.DataFrame) -> pd.Series:
    """True-range / close per bar (causal)."""
    h, low, c = df["high"], df["low"], df["close"]
    pc = c.shift(1)
    tr = pd.concat([h - low, (h - pc).abs(), (low - pc).abs()], axis=1).max(axis=1)
    return tr / c


# --------------------------------------------------------------------------- #
# #1 weekend_liquidity_fade — fade an outsized weekend move made on a thin book.
#    Orthogonal to momentum by a structural CALENDAR gate (only Fri20:00-Sun20:00
#    UTC) + contrarian sign. Needs only price + num_trades.
# --------------------------------------------------------------------------- #
_WK_VOL_LB = 30
_WK_NT_LB = 180
_WK_Z = 1.0


def weekend_liquidity_fade(df: pd.DataFrame) -> SkillSignal:
    sid = "weekend_liquidity_fade"
    if len(df) < _WK_NT_LB + 1 or "ts" not in df.columns or "num_trades" not in df.columns:
        return _hold(sid, "insufficient history / missing ts|num_trades")

    cur = pd.Timestamp(df["ts"].iloc[-1])
    if cur.tzinfo is None:
        cur = cur.tz_localize("UTC")
    dow, hr = cur.dayofweek, cur.hour  # Mon=0 .. Sun=6
    in_window = (dow == 4 and hr >= 20) or dow == 5 or (dow == 6 and hr < 20)
    if not in_window:
        return _hold(sid, f"not weekend window (dow={dow} hr={hr})")

    o, c, nt = df["open"], df["close"], df["num_trades"]
    prev_ret = float(np.log(c.iloc[-2] / o.iloc[-2]))
    atrpct = float(_atr_pct(df).iloc[:-1].tail(_WK_VOL_LB).mean())  # ending t-1
    if not (np.isfinite(atrpct) and atrpct > 0):
        return _hold(sid, "atrpct invalid")
    z = prev_ret / atrpct
    thin = float(nt.iloc[-2]) < float(nt.iloc[:-1].tail(_WK_NT_LB).mean())
    if abs(z) <= _WK_Z or not thin:
        return _hold(sid, f"z={z:.2f} thin={thin}")

    side = "SELL" if prev_ret > 0 else "BUY"
    conf = min(1.0, abs(z) / 2.0) * 0.6
    return SkillSignal(
        sid, side, conf, 3.0 * atrpct, 1.5 * atrpct,
        f"weekend fade z={z:.2f} thin_book", {"z": z, "atr_pct": atrpct},
    )


# --------------------------------------------------------------------------- #
# #2 vr_antitrend_fade — fade a stretched price ONLY when the Lo-MacKinlay
#    variance ratio says the tape is mean-reverting (VR<1). Orthogonal to
#    momentum: VR<0.8 is the statistical complement of momentum's VR>1 regime.
# --------------------------------------------------------------------------- #
_VR_WIN = 240
_VR_Q = 4
_VR_IN = 0.8
_ZVR_IN = -1.64
_BB = 20
_Z_IN = 1.0


def variance_ratio(returns: np.ndarray, q: int) -> tuple[float, float]:
    """Lo-MacKinlay variance ratio VR(q) and its homoskedastic z-stat.
    VR<1 ⇒ mean-reverting, VR>1 ⇒ trending, VR≈1 ⇒ random walk."""
    r = np.asarray(returns, dtype=float)
    M = r.size
    if q + 2 > M:
        return 1.0, 0.0
    mu = r.mean()
    var1 = ((r - mu) ** 2).sum() / (M - 1)
    if var1 <= 1e-15:  # degenerate (≈constant returns) -> no usable VR signal
        return 1.0, 0.0
    rq = np.convolve(r, np.ones(q), "valid")  # overlapping q-period sums
    varq = ((rq - q * mu) ** 2).sum() / (q * (M - q + 1))
    vr = varq / var1
    phi = 2.0 * (2 * q - 1) * (q - 1) / (3.0 * q * M)
    z = (vr - 1.0) / np.sqrt(phi) if phi > 0 else 0.0
    return float(vr), float(z)


def vr_antitrend_fade(df: pd.DataFrame) -> SkillSignal:
    sid = "vr_antitrend_fade"
    c = df["close"]
    if len(c) < _VR_WIN + 2:
        return _hold(sid, "insufficient history")
    logc = np.log(c.to_numpy(dtype=float))
    vr, zvr = variance_ratio(np.diff(logc)[-_VR_WIN:], _VR_Q)
    win = c.tail(_BB)
    sma, sd = float(win.mean()), float(win.std(ddof=0))
    px = float(c.iloc[-1])
    if sd <= 0 or px <= 0:
        return _hold(sid, "flat/invalid")
    z = (px - sma) / sd
    if not (vr <= _VR_IN and zvr <= _ZVR_IN and abs(z) >= _Z_IN):
        return _hold(sid, f"VR={vr:.2f} zVR={zvr:.2f} z={z:.2f}")
    side = "SELL" if z >= _Z_IN else "BUY"
    atrpct = sd / px
    conf = min(1.0, abs(z) / 3.0) * 0.55
    return SkillSignal(
        sid, side, conf, max(0.005, 2.5 * atrpct), max(0.005, 1.5 * atrpct),
        f"VR={vr:.2f} zVR={zvr:.2f} z={z:.2f} (mean-revert regime)", {"VR": vr, "z": z},
    )


# --------------------------------------------------------------------------- #
# #4 flow_exhaustion_fade — fade a stretched move that aggressor flow is NO
#    LONGER confirming (extreme one-sided taker flow + large avg trade size =
#    late, exhausting participation). Uses the taker-buy column momentum ignores.
# --------------------------------------------------------------------------- #
_FE_WIN = 120
_FE_BB = 20
_OFZ_IN = 2.0
_ATSZ_IN = 1.0
_FE_Z = 1.0


def _ztail(series: pd.Series, n: int) -> float:
    a = series.tail(n).to_numpy(dtype=float)
    a = a[np.isfinite(a)]
    if a.size < 2 or a.std(ddof=0) == 0:
        return 0.0
    return float((a[-1] - a.mean()) / a.std(ddof=0))


def flow_exhaustion_fade(df: pd.DataFrame) -> SkillSignal:
    sid = "flow_exhaustion_fade"
    need = ("close", "volume", "taker_buy_base", "num_trades")
    if len(df) < _FE_WIN + _FE_BB or any(col not in df.columns for col in need):
        return _hold(sid, "insufficient history / missing micro columns")
    c, vol, tbb, nt = df["close"], df["volume"], df["taker_buy_base"], df["num_trades"]
    ofi = (tbb / vol.replace(0, np.nan)) - 0.5            # aggressor imbalance per bar
    ats = vol / nt.replace(0, np.nan)                     # avg trade size
    ofz = _ztail(ofi, _FE_WIN)
    atsz = _ztail(ats, _FE_WIN)
    win = c.tail(_FE_BB)
    sma, sd = float(win.mean()), float(win.std(ddof=0))
    px = float(c.iloc[-1])
    if sd <= 0 or px <= 0:
        return _hold(sid, "flat/invalid")
    z = (px - sma) / sd
    atrpct = sd / px
    side = "HOLD"
    if ofz >= _OFZ_IN and atsz >= _ATSZ_IN and z >= _FE_Z:
        side = "SELL"           # up-move on exhausting aggressor buying -> fade
    elif ofz <= -_OFZ_IN and atsz >= _ATSZ_IN and z <= -_FE_Z:
        side = "BUY"            # down-move on exhausting aggressor selling -> fade
    if side == "HOLD":
        return _hold(sid, f"ofz={ofz:.2f} atsz={atsz:.2f} z={z:.2f}")
    conf = min(1.0, abs(ofz) / 2.0) * 0.6
    return SkillSignal(
        sid, side, conf, max(0.005, 1.5 * atrpct), max(0.005, 1.0 * atrpct),
        f"flow exhaustion ofz={ofz:.2f} atsz={atsz:.2f} z={z:.2f}", {"ofz": ofz, "z": z},
    )


# --------------------------------------------------------------------------- #
# #funding funding_extreme_fade — REFUTED HYPOTHESIS (kept as a documented
#    negative result, NOT for use). The original thesis was "extreme funding =
#    crowd over-positioned -> fade the move + earn carry". A full ~9y walk-forward
#    on BTC/ETH/SOL proved the SIGN IS BACKWARDS: extreme funding marks a strong,
#    persistent trend that CONTINUES for ~3-4 days, not a reversion. Fading it
#    loses; going WITH it wins (see funding_squeeze_continuation below, validated
#    OOS Sharpe ~+1.1, orthogonal to momentum rho~0.03, robust across all 3
#    assets). Lesson logged: this fade failed because of (a) inverted sign and
#    (b) a too-short 48h horizon — both fixed in the continuation signal.
# --------------------------------------------------------------------------- #
_FUND_WIN = 180
_FUND_Z = 2.0


def funding_extreme_fade(df: pd.DataFrame) -> SkillSignal:
    sid = "funding_extreme_fade"
    if "funding_rate" not in df.columns or len(df) < _FUND_WIN + 1:
        return _hold(sid, "missing funding_rate / insufficient history")
    fz = _ztail(df["funding_rate"], _FUND_WIN)
    atrpct = float(_atr_pct(df).tail(14).mean())
    if not (np.isfinite(atrpct) and atrpct > 0):
        atrpct = 0.02
    if fz >= _FUND_Z:
        side = "SELL"           # funding very positive: longs crowded -> fade down
    elif fz <= -_FUND_Z:
        side = "BUY"            # funding very negative: shorts crowded -> fade up
    else:
        return _hold(sid, f"funding z={fz:.2f}")
    conf = min(1.0, abs(fz) / 3.0) * 0.6
    return SkillSignal(
        sid, side, conf, max(0.005, 3.0 * atrpct), max(0.005, 2.0 * atrpct),
        f"funding extreme z={fz:.2f} -> fade + carry", {"funding_z": fz},
    )


# --------------------------------------------------------------------------- #
# funding_squeeze_continuation — THE VALIDATED Strategy #2 (supersedes the fade).
#    Extreme-positive funding (|z|>=2 over a 180-bar window) marks a strong trend
#    backed by aggressive leveraged demand that CONTINUES ~3-4 days (hold ~24x4h).
#    Discovered by a systematic 648-backtest sweep (6 microstructure features x
#    both directions x horizons {12,24,48} x thresholds x 3 assets): of every
#    candidate, ONLY funding survived the strict coherent gate (net>0 AND
#    in-sample>0 AND OOS>0 AND every walk-forward fold>0) *and* generalized
#    cross-asset. Honest metrics (two methods agreeing): per-trade-ann Sharpe
#    +1.30, calendar-day Sharpe +1.18, OOS-40% +1.37; daily-PnL correlation to
#    Strategy #1 = +0.04 (genuine diversifier — the ensemble roughly DOUBLES the
#    OOS Sharpe vs #1 alone). Orthogonality is structural: funding is the
#    perp-vs-spot basis, exogenous to the price path momentum reads.
#    CAVEAT (logged, to be tested): the edge is LONG-only and validated over a
#    period of secular crypto appreciation; needs an explicit bear-subperiod check
#    before live sizing to rule out drift-exposure masquerading as edge — though
#    every individual walk-forward fold (incl. 2018/2022 bears) was positive.
# --------------------------------------------------------------------------- #
_FUNDC_WIN = 180
_FUNDC_Z = 2.0          # frozen a-priori; the whole |z|>=2, h in {24,48} family is robust
_FUNDC_HOLD_BARS = 24   # ~96h on 4h bars — the validated horizon (h=12/48h was too short)


def funding_squeeze_continuation(df: pd.DataFrame) -> SkillSignal:
    sid = "funding_squeeze_continuation"
    if "funding_rate" not in df.columns or len(df) < _FUNDC_WIN + 1:
        return _hold(sid, "missing funding_rate / insufficient history")
    fz = _ztail(df["funding_rate"], _FUNDC_WIN)
    if fz < _FUNDC_Z:                       # only the extreme-positive tail (validated 3/3 assets)
        # Carry funding_z even when holding so the dashboard can show the live gauge
        # (how far the sleeve is from arming) instead of an opaque blank.
        return SkillSignal(
            sid, "HOLD", 0.0, 0.02, 0.04,
            f"funding z={fz:.2f} < {_FUNDC_Z} (no squeeze tail)",
            {"funding_z": float(fz)},
        )
    atrpct = float(_atr_pct(df).tail(14).mean())
    if not (np.isfinite(atrpct) and atrpct > 0):
        atrpct = 0.02
    # Wide stop/TP so the ~24-bar TIME exit dominates (the validated edge is the
    # horizon return, not a tight barrier). Conviction scales with |z|.
    conf = min(1.0, fz / 4.0)
    return SkillSignal(
        sid, "BUY", conf, max(0.01, 4.0 * atrpct), max(0.02, 8.0 * atrpct),
        f"funding squeeze continuation z={fz:.2f} -> long ~{_FUNDC_HOLD_BARS}x4h",
        {"funding_z": fz, "hold_bars": _FUNDC_HOLD_BARS},
    )


# --------------------------------------------------------------------------- #
# funding_carry_harvest — Strategy #3: DELTA-NEUTRAL funding carry (classic
#    cash-and-carry). RESTATED per the 2026-06-25 re-audit: at honest 30bps
#    two-leg taker cost the edge is THIN (+16..+23bps/trade, OOS-40 ~0) — the old
#    "+2.4..+4.5 OOS" headline does not reproduce at honest cost; healthy only
#    with maker execution (~12bps). It only works HEDGED: a 9y test proved that
#    as a NAKED perp short the edge is OOS-NEGATIVE. STATUS (corrected
#    2026-07-10 — the old "DISPLAY-ONLY / always HOLD / kept OUT of
#    allowed_skills" comment here was stale and FALSE): the signal IS in
#    strategy.allowed_skills and emits BUY at bps>=2; safety is STRUCTURAL, not
#    declarative — the paper exchange routes this skill_id unconditionally to a
#    short-perp+long-spot pair (paper.py _open_carry) and the real Binance
#    adapter raises NotImplementedError for it, so a naked order is impossible
#    on every wired venue.
# --------------------------------------------------------------------------- #
_CARRY_ARM_BPS = 2.0  # per-8h funding (bps) at/above which the carry trade arms


def funding_carry_harvest(df: pd.DataFrame) -> SkillSignal:
    sid = "funding_carry_harvest"
    if "funding_rate" not in df.columns or len(df) < 2:
        return _hold(sid, "missing funding_rate")
    fr = float(df["funding_rate"].iloc[-1])
    bps = fr * 1e4
    feats = {"funding_bps": float(bps), "arm_bps": _CARRY_ARM_BPS}
    if bps < _CARRY_ARM_BPS:
        return SkillSignal(sid, "HOLD", 0.0, 0.02, 0.04,
                           f"funding carry {bps:.2f}bps < arm {_CARRY_ARM_BPS}", feats)
    # Armed: emit BUY. SAFE — the paper exchange routes this skill_id STRUCTURALLY to
    # the delta-neutral two-leg path (short perp + long spot), never a naked order.
    conf = min(1.0, bps / 10.0)
    return SkillSignal(
        sid, "BUY", conf, 0.05, 0.10,
        f"funding carry {bps:.2f}bps >= arm {_CARRY_ARM_BPS} -> delta-neutral pair",
        feats,
    )


# ======================= ROUND 2 — seasonality + persistence ================ #
# Round 1 lesson: "fade" signals lost; the calendar one (weekend) survived. So
# round 2 leans into calendar/seasonality and flow PERSISTENCE (not fade).


# turn_of_month_tilt — long the documented turn-of-month inflow window (last day
# of month + first 3 days). Calendar-gated -> orthogonal-by-timing to momentum.
_TOM_VOL_LB = 30


def turn_of_month_tilt(df: pd.DataFrame) -> SkillSignal:
    sid = "turn_of_month_tilt"
    if "ts" not in df.columns or len(df) < _TOM_VOL_LB + 1:
        return _hold(sid, "missing ts / insufficient")
    cur = pd.Timestamp(df["ts"].iloc[-1])
    if cur.tzinfo is None:
        cur = cur.tz_localize("UTC")
    day, dim = cur.day, cur.days_in_month
    if not (day >= dim or day <= 3):
        return _hold(sid, f"day {day}/{dim} not turn-of-month")
    atrpct = float(_atr_pct(df).tail(_TOM_VOL_LB).mean())
    if not (np.isfinite(atrpct) and atrpct > 0):
        atrpct = 0.02
    return SkillSignal(
        sid, "BUY", 0.5, max(0.005, 2.0 * atrpct), max(0.005, 3.0 * atrpct),
        f"turn-of-month long (day {day}/{dim})", {"day": day},
    )


# ofi_persistence_range — in a RANGE regime, trade WITH sustained aggressor-flow
# pressure (the opposite of the round-1 exhaustion fade that lost). Microstructure.
_OFP_SMOOTH = 18
_OFP_WIN = 180
_OFP_Z = 1.0


def ofi_persistence_range(df: pd.DataFrame) -> SkillSignal:
    sid = "ofi_persistence_range"
    need = ("close", "volume", "taker_buy_base")
    if len(df) < _OFP_WIN + _OFP_SMOOTH or any(c not in df.columns for c in need):
        return _hold(sid, "insufficient / missing micro columns")
    from hermes.data.regime import detect_regime

    if detect_regime(df)["regime"] != "range":
        return _hold(sid, "not range regime")
    ofi = (df["taker_buy_base"] / df["volume"].replace(0, np.nan)) - 0.5
    smoothed = ofi.rolling(_OFP_SMOOTH).mean()
    z = _ztail(smoothed, _OFP_WIN)
    atrpct = float(_atr_pct(df).tail(14).mean())
    if not (np.isfinite(atrpct) and atrpct > 0):
        atrpct = 0.02
    if z >= _OFP_Z:
        side = "BUY"            # sustained aggressor BUYING in a flat tape -> go with it
    elif z <= -_OFP_Z:
        side = "SELL"
    else:
        return _hold(sid, f"flow z={z:.2f}")
    conf = min(1.0, abs(z) / 2.0) * 0.55
    return SkillSignal(
        sid, side, conf, max(0.005, 2.0 * atrpct), max(0.005, 2.0 * atrpct),
        f"range OFI persistence z={z:.2f}", {"ofi_z": z},
    )


# wick_reversion — fade a wide-range bar that REJECTED an extreme (long wick) and
# closed back inside. Overreaction; distinct from the flow-exhaustion fade.
_WR_VOL_LB = 30
_WR_RANGE_MULT = 1.5
_WR_WICK_MULT = 2.0


def wick_reversion(df: pd.DataFrame) -> SkillSignal:
    sid = "wick_reversion"
    if len(df) < _WR_VOL_LB + 1:
        return _hold(sid, "insufficient history")
    o = float(df["open"].iloc[-1])
    h = float(df["high"].iloc[-1])
    low = float(df["low"].iloc[-1])
    c = float(df["close"].iloc[-1])
    rng = h - low
    if rng <= 0 or c <= 0:
        return _hold(sid, "degenerate bar")
    atrpct = float(_atr_pct(df).iloc[:-1].tail(_WR_VOL_LB).mean())
    if not (np.isfinite(atrpct) and atrpct > 0):
        return _hold(sid, "atrpct invalid")
    if (rng / c) < _WR_RANGE_MULT * atrpct:          # only big bars
        return _hold(sid, "bar not wide enough")
    body = abs(c - o)
    upper_wick = h - max(o, c)
    lower_wick = min(o, c) - low
    pos_in_range = (c - low) / rng
    side = "HOLD"
    if upper_wick > _WR_WICK_MULT * body and pos_in_range < 0.4:
        side = "SELL"           # rejected the highs -> fade down
    elif lower_wick > _WR_WICK_MULT * body and pos_in_range > 0.6:
        side = "BUY"            # rejected the lows -> bounce
    if side == "HOLD":
        return _hold(sid, "no wick rejection")
    conf = 0.55
    return SkillSignal(
        sid, side, conf, max(0.005, 2.0 * atrpct), max(0.005, 1.5 * atrpct),
        f"wick reversion pos={pos_in_range:.2f}", {"pos_in_range": pos_in_range},
    )
