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
#    Orthogonal to momentum by a structural CALENDAR gate (only Fri20:00–Sun20:00
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
