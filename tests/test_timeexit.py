"""Max-hold time exit so live holding matches the backtest's vertical barrier
(horizon bars). Without it a live position could ride indefinitely and the
backtested return distribution would not transfer."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from hermes.risk.timeexit import is_past_max_hold, max_hold_seconds, timeframe_seconds


def test_timeframe_seconds():
    assert timeframe_seconds("1h") == 3600
    assert timeframe_seconds("4h") == 14400
    assert timeframe_seconds("1d") == 86400


def test_max_hold_seconds():
    assert max_hold_seconds(12, "4h") == 12 * 14400  # 48h


def test_is_past_max_hold_true_beyond_window():
    now = datetime(2026, 1, 2, tzinfo=UTC)
    opened = now - timedelta(hours=49)  # > 48h
    assert is_past_max_hold(opened, now, max_hold_seconds(12, "4h")) is True


def test_is_past_max_hold_false_within_window():
    now = datetime(2026, 1, 2, tzinfo=UTC)
    opened = now - timedelta(hours=47)  # < 48h
    assert is_past_max_hold(opened, now, max_hold_seconds(12, "4h")) is False


def test_is_past_max_hold_disabled_when_zero():
    now = datetime(2026, 1, 2, tzinfo=UTC)
    opened = now - timedelta(days=30)
    assert is_past_max_hold(opened, now, 0) is False  # 0 = no time exit
