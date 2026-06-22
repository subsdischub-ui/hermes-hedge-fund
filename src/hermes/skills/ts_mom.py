"""Time-series momentum, vol-scaled, multi-lookback (per-symbol, 4h).

Thesis: cross-sectional & time-series momentum is one of the most robust,
most-replicated anomalies. The fragile part is *picking* a lookback, so we
do not pick: we average a fixed panel of lookbacks {12,24,48,96} bars. For
each lookback L we take the L-bar log return and divide it by the realized
volatility scaled to that same L-bar horizon (per-bar std times sqrt(L)). This
is the information-ratio of the window: a scale-free, O(1) Sharpe-like number so
fast and slow legs are comparable before averaging (dividing a multi-bar
return by a one-bar vol would inflate the score by ~√L and is avoided). Long
when the averaged signal is positive, short when negative (perps allow
shorts), HOLD inside the dead-band.

Parameters are FROZEN a priori (no optimization): lookbacks {12,24,48,96}
(geometric ladder, standard in TSMOM literature), a round 0.1 score
threshold, and ATR(14) stop/tp multiples (2.5x / 5x) matching the existing
skills. Averaging — not selection — is the deliberate anti-cherry-pick guard.

No look-ahead: every quantity is computed from log returns up to and
including the current bar. The L-bar return is close[-1]/close[-1-L] and the
vol is the std of the trailing L returns; nothing references a future bar
(no negative .shift, no centered window).
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from hermes.skills.library import SkillSignal

_LOOKBACKS: tuple[int, ...] = (12, 24, 48, 96)
_SCORE_THRESHOLD: float = 0.1


def _atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    h, low, c = df["high"], df["low"], df["close"]
    tr = pd.concat([h - low, (h - c.shift(1)).abs(), (low - c.shift(1)).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()


def ts_mom_vol_scaled(df: pd.DataFrame) -> SkillSignal:
    """Average of vol-scaled momentum over fixed lookbacks → BUY/SELL/HOLD."""
    close = df["close"].astype(float)
    px = float(close.iloc[-1])
    log_close = np.log(close)
    log_rets = log_close.diff()

    max_lb = max(_LOOKBACKS)
    # Need max_lb returns (i.e. max_lb + 1 prices) for the longest leg.
    insufficient = len(close) <= max_lb or px <= 0

    per_lb: list[float] = []
    if not insufficient:
        for lb in _LOOKBACKS:
            # L-bar log return using only past bars (<= current bar).
            ret_l = float(log_close.iloc[-1] - log_close.iloc[-1 - lb])
            # Realized vol of the trailing L one-bar returns (<= current bar),
            # scaled to the L-bar horizon so the ratio is a unit-free IR.
            window = log_rets.iloc[-lb:]
            vol_l = float(window.std(ddof=0)) * np.sqrt(lb)
            if np.isfinite(vol_l) and vol_l > 0:
                per_lb.append(ret_l / vol_l)

    score = float(np.mean(per_lb)) if per_lb else 0.0
    if not np.isfinite(score):
        score = 0.0

    atr = _atr(df).iloc[-1]
    stop_pct = float(2.5 * atr / px) if (atr and px > 0 and np.isfinite(atr)) else 0.025
    tp_pct = float(5.0 * atr / px) if (atr and px > 0 and np.isfinite(atr)) else 0.05

    side: Literal["BUY", "SELL", "HOLD"] = "HOLD"
    conf = 0.0
    if not insufficient:
        if score > _SCORE_THRESHOLD:
            side, conf = "BUY", min(1.0, abs(score))
        elif score < -_SCORE_THRESHOLD:
            side, conf = "SELL", min(1.0, abs(score))

    return SkillSignal(
        skill_id="ts_mom_vol_scaled",
        side=side,
        confidence=conf,
        stop_pct=stop_pct,
        take_profit_pct=tp_pct,
        rationale=(
            f"vol-scaled TSMOM avg over {_LOOKBACKS}: score={score:.3f} "
            f"thr=±{_SCORE_THRESHOLD} px={px:.2f}"
        ),
        features={"score": score, "atr_pct": stop_pct / 2.5},
    )
