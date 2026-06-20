"""Market-regime detection + regime-conditioned skill weighting.

Uses the Kaufman Efficiency Ratio (|net move| / sum|bar moves| over a window):
~1.0 = clean directional trend, ~0.0 = choppy range. This is a fast, deterministic
signal that lets the agent adapt which skills it trusts in real time, instead of
waiting for the weekly Sharpe re-rank (which reacts with a 1-4 week lag).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Skill family -> the regime it is suited to. Used to down-weight off-regime skills.
_TREND_SKILLS = ("trend", "breakout")
_RANGE_SKILLS = ("mean_reversion", "reversion", "range")
_OFF_REGIME_WEIGHT = 0.5


def detect_regime(df: pd.DataFrame, window: int = 48, trend_threshold: float = 0.4) -> dict:
    """Classify the recent window as trend_up / trend_down / range via the
    Kaufman efficiency ratio. Returns {regime, efficiency_ratio}."""
    closes = np.asarray(df["close"].tail(window).to_numpy(), dtype=float)
    if closes.size < 2:
        return {"regime": "range", "efficiency_ratio": 0.0}
    net = float(closes[-1] - closes[0])
    path = float(np.abs(np.diff(closes)).sum())
    er = abs(net) / path if path > 0 else 0.0
    if er >= trend_threshold and net != 0.0:
        regime = "trend_up" if net > 0 else "trend_down"
    else:
        regime = "range"
    return {"regime": regime, "efficiency_ratio": er}


def regime_signal_weight(skill_id: str, regime: str) -> float:
    """Multiplier for a skill's confidence given the regime: 1.0 if the skill
    suits the regime, 0.5 if it is off-regime, 1.0 for unrecognised skills."""
    sid = skill_id.lower()
    is_trend_skill = any(k in sid for k in _TREND_SKILLS)
    is_range_skill = any(k in sid for k in _RANGE_SKILLS)
    trending = regime in ("trend_up", "trend_down")
    if trending:
        if is_trend_skill:
            return 1.0
        if is_range_skill:
            return _OFF_REGIME_WEIGHT
    else:  # range
        if is_range_skill:
            return 1.0
        if is_trend_skill:
            return _OFF_REGIME_WEIGHT
    return 1.0
