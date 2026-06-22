"""Max-hold time exit — close a position once it has been held for the strategy's
holding horizon, mirroring the backtest's triple-barrier VERTICAL barrier.

The OOS-validated edge assumes each position is force-closed after `horizon` bars
(~48h on 4h bars) if neither the stop nor the take-profit was hit. A live system
without an equivalent time exit would hold winners/losers indefinitely, so its
return distribution would diverge from the backtested (validated) one. Pure
helpers here; wired into the risk engine's stop watcher.
"""

from __future__ import annotations

from datetime import datetime

_TF_SECONDS = {
    "1m": 60,
    "5m": 300,
    "15m": 900,
    "30m": 1800,
    "1h": 3600,
    "2h": 7200,
    "4h": 14400,
    "6h": 21600,
    "8h": 28800,
    "12h": 43200,
    "1d": 86400,
}


def timeframe_seconds(timeframe: str) -> int:
    """Bar length in seconds (e.g. '4h' -> 14400)."""
    return _TF_SECONDS[timeframe]


def max_hold_seconds(max_hold_bars: int, timeframe: str) -> int:
    """Holding-window length in seconds = bars x bar-length."""
    return int(max_hold_bars) * timeframe_seconds(timeframe)


def is_past_max_hold(opened_at: datetime, now: datetime, max_hold_secs: int) -> bool:
    """Whether a position opened at ``opened_at`` has exceeded the holding window.
    ``max_hold_secs <= 0`` disables the time exit (always False)."""
    if max_hold_secs <= 0:
        return False
    return (now - opened_at).total_seconds() >= max_hold_secs


def max_hold_bars_for(skill_id: str, strategy_cfg: dict) -> int:
    """Per-skill holding horizon, falling back to the global default.

    Different validated strategies have different vertical-barrier horizons:
    momentum (Strategy #1) exits at ``max_hold_bars`` (~12 bars/48h), while
    funding_squeeze_continuation (Strategy #2) was validated on a ~24-bar/96h hold.
    ``strategy.max_hold_bars_by_skill`` maps skill_id -> bars; anything not listed
    uses ``strategy.max_hold_bars``.
    """
    by_skill = strategy_cfg.get("max_hold_bars_by_skill", {}) or {}
    if skill_id in by_skill:
        return int(by_skill[skill_id])
    return int(strategy_cfg.get("max_hold_bars", 0) or 0)
