"""Intraday time-of-day skill for BTC (1h bars).

Thesis: BTC exhibits a weak but persistent intraday seasonality anchored to the
US cash-equity session. The sign of the first post-US-open hour
(13:00->14:00 UTC, i.e. ~09:00->10:00 ET) tends to be re-expressed in the day's
final hour before the US close (20:00->21:00 UTC, ~16:00->17:00 ET): a green
US open biases the late session higher, a red open biases it lower.

The skill only emits a directional signal *at* the 20:00 UTC bar (the entry for
the closing window), so at most one trade fires per UTC day. At that moment the
morning return is already fully observed -> no look-ahead: the decision reads
only the 13:00 and 14:00 bars, both strictly earlier than the current bar.

Parameters are FIXED a priori (no optimisation): round US-session window hours
and modest fixed risk brackets (1% stop / 1.5% take-profit). BTC-only; symbol
filtering is the integration layer's responsibility, not this skill's.
"""

from __future__ import annotations

from typing import Literal

import pandas as pd

from hermes.skills.library import SkillSignal

# Frozen parameters (round, standard, never tuned).
_OPEN_START_HOUR = 13  # first post-US-open hour, UTC
_OPEN_END_HOUR = 14  # end of the first post-open window, UTC
_DECISION_HOUR = 20  # entry bar for the day's closing window, UTC
_STOP_PCT = 0.01
_TAKE_PROFIT_PCT = 0.015
_CONFIDENCE = 0.5

_SKILL_ID = "intraday_tod_btc"


def _hours_and_dates(df: pd.DataFrame) -> tuple[list[int], list[object]]:
    """Extract per-bar UTC hour and calendar date from a 'ts' column or the
    DatetimeIndex, as positionally-indexed lists aligned to df rows (so plain
    integer indexing like [-1] is unambiguous regardless of the row index)."""
    if "ts" in df.columns:
        ts = pd.to_datetime(df["ts"], utc=True)
        return ts.dt.hour.tolist(), list(ts.dt.date)
    idx = pd.to_datetime(df.index, utc=True)
    return list(idx.hour), list(idx.date)


def _hold(rationale: str) -> SkillSignal:
    return SkillSignal(
        skill_id=_SKILL_ID,
        side="HOLD",
        confidence=0.0,
        stop_pct=_STOP_PCT,
        take_profit_pct=_TAKE_PROFIT_PCT,
        rationale=rationale,
        features={},
    )


def intraday_tod_btc(df: pd.DataFrame) -> SkillSignal:
    """Map a window of 1h BTC bars (ending at the current bar) to a SkillSignal.

    At the 20:00 UTC bar: BUY if the 13:00->14:00 UTC return of the same day was
    positive, SELL if negative, HOLD if flat or unavailable. Any other current
    hour yields HOLD. No look-ahead: only the morning window (<= 14:00) feeds the
    decision, and the morning window is strictly earlier than the 20:00 entry.
    """
    if df is None or len(df) == 0:
        return _hold("empty frame")

    hours, dates = _hours_and_dates(df)
    cur_hour = int(hours[-1])
    if cur_hour != _DECISION_HOUR:
        return _hold(f"hour={cur_hour:02d}h not the {_DECISION_HOUR:02d}h decision window")

    cur_date = dates[-1]
    close = df["close"].to_numpy(dtype=float)

    open_close = None
    open_open = None
    for i in range(len(df)):
        if dates[i] != cur_date:
            continue
        hr = int(hours[i])
        if hr == _OPEN_START_HOUR:
            open_open = float(close[i])
        elif hr == _OPEN_END_HOUR:
            open_close = float(close[i])

    if open_open is None or open_close is None or open_open <= 0:
        return _hold("morning 13:00->14:00 window unavailable")

    morning_ret = open_close / open_open - 1.0
    side: Literal["BUY", "SELL", "HOLD"] = "HOLD"
    conf = 0.0
    if morning_ret > 0:
        side, conf = "BUY", _CONFIDENCE
    elif morning_ret < 0:
        side, conf = "SELL", _CONFIDENCE

    return SkillSignal(
        skill_id=_SKILL_ID,
        side=side,
        confidence=conf,
        stop_pct=_STOP_PCT,
        take_profit_pct=_TAKE_PROFIT_PCT,
        rationale=(
            f"US-open {_OPEN_START_HOUR:02d}->{_OPEN_END_HOUR:02d}h ret="
            f"{morning_ret:+.4%} -> {side} the {_DECISION_HOUR:02d}h close window"
        ),
        features={"morning_ret": morning_ret},
    )
