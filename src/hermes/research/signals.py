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
# #funding funding_extreme_fade — fade an extreme funding rate (crowd over-
#    positioned) and earn the carry while waiting for the squeeze. Needs a
#    funding_rate column (attach via hermes.research.funding.attach_funding).
#    Most defensible orthogonality: funding is exogenous to the price path.
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
