"""Macro blackout gate — pure-function tests, no network (the calendar is a static
committed file: config/macro_calendar.yaml).

Covers:
- Calendar loading (real committed file) — sane counts, DST correctness on 2 real
  dates (one EDT, one EST) taken straight from the committed calendar.
- blackout_status(): inside/outside the window, exact edges (inclusive both sides).
- calendar_horizon() + horizon_warning(): honest "worst-covered event type" horizon,
  and the <60d visible warning.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
import yaml

from hermes.risk.macro_blackout import (
    CALENDAR_PATH,
    MacroEvent,
    blackout_status,
    calendar_horizon,
    horizon_warning,
    load_calendar,
)


def _cal(events: list[MacroEvent]) -> dict:
    return {"events": sorted(events, key=lambda e: e.dt_utc)}


def _ev(iso_utc: str, event_type: str = "fomc_decision", label: str = "test event") -> MacroEvent:
    return MacroEvent(dt_utc=datetime.fromisoformat(iso_utc), event_type=event_type, label=label)


# --------------------------------------------------------------------------- #
# Real committed calendar: loading + DST correctness
# --------------------------------------------------------------------------- #

def test_load_calendar_reads_the_real_committed_file():
    cal = load_calendar()
    events = cal["events"]
    assert len(events) == 40  # 16 FOMC + 12 CPI + 12 NFP, see config/macro_calendar.yaml
    by_type: dict[str, int] = {}
    for e in events:
        by_type[e.event_type] = by_type.get(e.event_type, 0) + 1
        assert e.dt_utc.tzinfo is not None  # always tz-aware UTC
    assert by_type == {"fomc_decision": 16, "cpi": 12, "nfp": 12}


def test_calendar_path_points_at_the_committed_file():
    assert CALENDAR_PATH.exists()
    assert CALENDAR_PATH.name == "macro_calendar.yaml"


def test_dst_conversion_winter_est():
    """2026-01-28 FOMC decision: 14:00 America/New_York in JANUARY is EST (UTC-5) ->
    19:00 UTC. Real date from the committed calendar (federalreserve.gov)."""
    cal = load_calendar()
    ev = next(e for e in cal["events"] if e.event_type == "fomc_decision" and e.dt_utc.date().isoformat() == "2026-01-28")
    assert ev.dt_utc == datetime(2026, 1, 28, 19, 0, tzinfo=UTC)


def test_dst_conversion_summer_edt():
    """2026-06-17 FOMC decision: 14:00 America/New_York in JUNE is EDT (UTC-4) ->
    18:00 UTC. Real date from the committed calendar (federalreserve.gov)."""
    cal = load_calendar()
    ev = next(e for e in cal["events"] if e.event_type == "fomc_decision" and e.dt_utc.date().isoformat() == "2026-06-17")
    assert ev.dt_utc == datetime(2026, 6, 17, 18, 0, tzinfo=UTC)


# --------------------------------------------------------------------------- #
# blackout_status — pure function, synthetic calendar
# --------------------------------------------------------------------------- #

def test_blackout_status_inside_window():
    cal = _cal([_ev("2026-03-18T18:00:00+00:00", label="FOMC decision (Mar 2026)")])
    now = datetime(2026, 3, 18, 19, 30, tzinfo=UTC)  # 1.5h after the event
    st = blackout_status(now, cal, pre_h=2, post_h=2)
    assert st["blocked"] is True
    assert st["event"] == "FOMC decision (Mar 2026)"
    assert st["event_type"] == "fomc_decision"


def test_blackout_status_outside_window():
    cal = _cal([_ev("2026-03-18T18:00:00+00:00")])
    now = datetime(2026, 3, 18, 15, 0, tzinfo=UTC)  # 3h before, outside the 2h pre-window
    st = blackout_status(now, cal, pre_h=2, post_h=2)
    assert st["blocked"] is False
    assert st["event"] is None


def test_blackout_status_exact_left_edge_inclusive():
    """now == T - pre_h exactly -> blocked (inclusive boundary)."""
    ev_dt = datetime(2026, 3, 18, 18, 0, tzinfo=UTC)
    cal = _cal([MacroEvent(dt_utc=ev_dt, event_type="fomc_decision", label="x")])
    now = ev_dt - timedelta(hours=2)
    assert blackout_status(now, cal, pre_h=2, post_h=2)["blocked"] is True


def test_blackout_status_just_before_left_edge_not_blocked():
    ev_dt = datetime(2026, 3, 18, 18, 0, tzinfo=UTC)
    cal = _cal([MacroEvent(dt_utc=ev_dt, event_type="fomc_decision", label="x")])
    now = ev_dt - timedelta(hours=2, seconds=1)
    assert blackout_status(now, cal, pre_h=2, post_h=2)["blocked"] is False


def test_blackout_status_exact_right_edge_inclusive():
    """now == T + post_h exactly -> blocked (inclusive boundary)."""
    ev_dt = datetime(2026, 3, 18, 18, 0, tzinfo=UTC)
    cal = _cal([MacroEvent(dt_utc=ev_dt, event_type="fomc_decision", label="x")])
    now = ev_dt + timedelta(hours=2)
    assert blackout_status(now, cal, pre_h=2, post_h=2)["blocked"] is True


def test_blackout_status_just_after_right_edge_not_blocked():
    ev_dt = datetime(2026, 3, 18, 18, 0, tzinfo=UTC)
    cal = _cal([MacroEvent(dt_utc=ev_dt, event_type="fomc_decision", label="x")])
    now = ev_dt + timedelta(hours=2, seconds=1)
    assert blackout_status(now, cal, pre_h=2, post_h=2)["blocked"] is False


def test_blackout_status_empty_calendar_never_blocks():
    assert blackout_status(datetime.now(UTC), _cal([]))["blocked"] is False


def test_blackout_status_picks_the_matching_event_among_several():
    cal = _cal(
        [
            _ev("2026-01-13T13:30:00+00:00", event_type="cpi", label="CPI (Dec 2025 data)"),
            _ev("2026-01-28T19:00:00+00:00", event_type="fomc_decision", label="FOMC decision (Jan 2026)"),
            _ev("2026-02-11T13:30:00+00:00", event_type="nfp", label="NFP (Jan 2026 data)"),
        ]
    )
    now = datetime(2026, 1, 28, 20, 30, tzinfo=UTC)  # inside the FOMC window only
    st = blackout_status(now, cal, pre_h=2, post_h=2)
    assert st["blocked"] is True
    assert st["event_type"] == "fomc_decision"


# --------------------------------------------------------------------------- #
# calendar_horizon / horizon_warning
# --------------------------------------------------------------------------- #

def test_calendar_horizon_is_the_worst_covered_event_type():
    """FOMC covers further out than CPI/NFP here — horizon must be the EARLIEST
    'last date' across types (the honest, worst-case bound), not the latest."""
    cal = _cal(
        [
            _ev("2026-01-01T00:00:00+00:00", event_type="fomc_decision"),
            _ev("2027-12-08T00:00:00+00:00", event_type="fomc_decision"),
            _ev("2026-06-01T00:00:00+00:00", event_type="cpi"),
            _ev("2026-08-01T00:00:00+00:00", event_type="nfp"),
        ]
    )
    h = calendar_horizon(cal)
    assert h.isoformat() == "2026-06-01"  # min(2027-12-08, 2026-06-01, 2026-08-01)


def test_calendar_horizon_raises_on_empty_calendar():
    with pytest.raises(ValueError):
        calendar_horizon(_cal([]))


def test_horizon_warning_fires_within_60_days():
    cal = _cal([_ev("2026-09-01T00:00:00+00:00", event_type="nfp")])
    now = datetime(2026, 8, 15, tzinfo=UTC)  # 17 days before horizon
    msg = horizon_warning(now, cal, warn_within_days=60)
    assert msg is not None
    assert "2026-09-01" in msg


def test_horizon_warning_silent_when_far_from_horizon():
    cal = _cal([_ev("2027-01-01T00:00:00+00:00", event_type="nfp")])
    now = datetime(2026, 1, 1, tzinfo=UTC)  # ~1 year out
    assert horizon_warning(now, cal, warn_within_days=60) is None


def test_real_calendar_yaml_is_well_formed_yaml():
    """Belt-and-suspenders: the file must also parse as plain YAML (independent of
    our own loader) so a hand-edit typo is caught early."""
    raw = yaml.safe_load(CALENDAR_PATH.read_text())
    assert isinstance(raw["events"], list)
    assert len(raw["events"]) == 40
    for e in raw["events"]:
        assert e["type"] in ("fomc_decision", "cpi", "nfp")
        assert e["tz"] == "America/New_York"
