"""Macro blackout gate — a window of caution around high-impact US macro events
(FOMC rate decisions, CPI, NFP/Employment Situation).

Audit context (docs/SYSTEME_AUTONOME.md, partie III, 2026-07-10): the agent had NO
blackout of any kind — it would trade straight through FOMC/CPI/NFP prints, the exact
failure mode that cost the operator $12k on a manual account. This module closes that
gap with a STATIC, committed calendar (config/macro_calendar.yaml) — real dates
fetched from federalreserve.gov and bls.gov, documented + sourced in that file's
header — and a pure function that decides whether "now" falls inside a caution
window around any calendar event.

Fidelity decision (non-negotiable, set by the operator 2026-07-10): the live
strategies were VALIDATED on backtests that INCLUDE news periods. Blocking trades at
stage_0_paper would silently diverge live behaviour from the validated backtest
before we have ANY evidence the blackout helps — the same backtest/live-fidelity bug
class as the maker-fee issue. So this module is deliberately just the DECISION
function; whether it blocks or only shadow-logs is a call the RiskEngine makes based
on config/default.yaml's ``macro_blackout.enforce_stages`` (see guardrails.py).

No network call happens here or anywhere in this module — the calendar is read once
from disk. An operator refreshes the YAML periodically; ``horizon_warning`` gives a
visible nudge once the calendar's honest coverage horizon is close.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import yaml

from hermes.logging_setup import get_logger

log = get_logger(__name__)

# config/macro_calendar.yaml, two levels above src/hermes/risk/.
CALENDAR_PATH = Path(__file__).resolve().parents[3] / "config" / "macro_calendar.yaml"

DEFAULT_PRE_HOURS = 2.0
DEFAULT_POST_HOURS = 2.0
DEFAULT_WARN_WITHIN_DAYS = 60


@dataclass(frozen=True)
class MacroEvent:
    dt_utc: datetime  # tz-aware, UTC
    event_type: str  # "fomc_decision" | "cpi" | "nfp"
    label: str


def load_calendar(path: Path = CALENDAR_PATH) -> dict[str, Any]:
    """Parse the static macro calendar YAML into UTC-aware ``MacroEvent``s.

    Pure function of the file on disk — no network. Local wall-clock time +
    an IANA tz name are converted to UTC via ``zoneinfo`` so DST (EDT/EST) is
    handled correctly instead of a hand-maintained UTC offset per event.
    """
    raw = yaml.safe_load(path.read_text()) or {}
    events: list[MacroEvent] = []
    for e in raw.get("events", []):
        d = date.fromisoformat(str(e["date"]))
        t = time.fromisoformat(str(e["time_local"]))
        tz = ZoneInfo(str(e.get("tz", "America/New_York")))
        local_dt = datetime.combine(d, t, tzinfo=tz)
        events.append(
            MacroEvent(
                dt_utc=local_dt.astimezone(UTC),
                event_type=str(e["type"]),
                label=str(e.get("label", e["type"])),
            )
        )
    events.sort(key=lambda ev: ev.dt_utc)
    return {"events": events, "retrieved": raw.get("retrieved")}


@lru_cache(maxsize=1)
def get_macro_calendar() -> dict[str, Any]:
    """Cached load of the committed calendar (a plain file read; safe to cache for
    the process lifetime — refreshing the YAML requires a restart, same contract as
    ``hermes.settings.get_settings``)."""
    return load_calendar()


def blackout_status(
    now_utc: datetime,
    calendar: dict[str, Any],
    pre_h: float = DEFAULT_PRE_HOURS,
    post_h: float = DEFAULT_POST_HOURS,
) -> dict[str, Any]:
    """Is ``now_utc`` inside a [T-pre_h, T+post_h] caution window around any event?

    Both boundaries are INCLUSIVE. Returns the first matching event (the calendar is
    sorted, and caution windows around distinct high-impact US macro prints are not
    expected to meaningfully overlap in this calendar's density).
    """
    for ev in calendar["events"]:
        window_start = ev.dt_utc - timedelta(hours=pre_h)
        window_end = ev.dt_utc + timedelta(hours=post_h)
        if window_start <= now_utc <= window_end:
            return {
                "blocked": True,
                "event": ev.label,
                "event_type": ev.event_type,
                "event_time_utc": ev.dt_utc.isoformat(),
                "window": {"start": window_start.isoformat(), "end": window_end.isoformat()},
            }
    return {"blocked": False, "event": None, "event_type": None, "event_time_utc": None, "window": None}


def calendar_horizon(calendar: dict[str, Any]) -> date:
    """Honest coverage horizon: the EARLIEST 'last date' across the distinct event
    types present. A blackout calendar protects only as far as its WORST-covered
    event type — if FOMC is published through Dec 2027 but CPI/NFP only through Dec
    2026, the calendar stops being trustworthy in full at the Dec-2026 boundary, not
    the Dec-2027 one. Raises on an empty calendar (nothing to compute a horizon from)."""
    events = calendar["events"]
    last_by_type: dict[str, date] = {}
    for ev in events:
        d = ev.dt_utc.date()
        if ev.event_type not in last_by_type or d > last_by_type[ev.event_type]:
            last_by_type[ev.event_type] = d
    if not last_by_type:
        raise ValueError("empty macro calendar — cannot compute a horizon")
    return min(last_by_type.values())


def horizon_warning(
    now_utc: datetime,
    calendar: dict[str, Any],
    warn_within_days: int = DEFAULT_WARN_WITHIN_DAYS,
) -> str | None:
    """Visible warning once ``now_utc`` is within ``warn_within_days`` of the
    calendar's honest horizon (see ``calendar_horizon``). Returns None otherwise."""
    horizon = calendar_horizon(calendar)
    days_left = (horizon - now_utc.date()).days
    if days_left < warn_within_days:
        msg = (
            f"macro calendar horizon {horizon.isoformat()} is only {days_left}d away "
            f"— refresh config/macro_calendar.yaml (see file header for sources)"
        )
        log.warning(
            "macro_calendar_horizon_low", horizon=horizon.isoformat(), days_left=days_left
        )
        return msg
    return None
