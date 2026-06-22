"""Tests for the intraday time-of-day BTC skill.

Thesis under test: the sign of the first post-US-open window return
(13:00->14:00 UTC) predicts the direction of the day's last window
(20:00->21:00 UTC). The signal only fires at the 20:00 UTC bar, by which time
the morning return is fully observed (no look-ahead).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from hermes.skills.intraday_tod import intraday_tod_btc
from hermes.skills.library import SkillSignal


def _hourly_day(
    day: str = "2026-06-01",
    morning_direction: float = 0.0,
    base: float = 50_000.0,
    use_ts_column: bool = False,
) -> pd.DataFrame:
    """Build a flat 1h OHLCV day, then nudge the 13:00->14:00 window up/down.

    morning_direction > 0 -> morning rises, < 0 -> morning falls, 0 -> flat.
    By default the timestamp lives in a DatetimeUTC index; set use_ts_column to
    expose it as a 'ts' column instead.
    """
    idx = pd.date_range(f"{day} 00:00", periods=24, freq="h", tz="UTC")
    close = np.full(24, base, dtype=float)
    # Move only the 14:00 bar relative to the 13:00 bar to set the morning sign.
    if morning_direction > 0:
        close[14] = base * 1.01
    elif morning_direction < 0:
        close[14] = base * 0.99
    open_ = close.copy()
    high = close * 1.0005
    low = close * 0.9995
    vol = np.full(24, 100.0)
    df = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": vol},
        index=idx,
    )
    if use_ts_column:
        df = df.reset_index().rename(columns={"index": "ts"})
    return df


def _slice_until(df: pd.DataFrame, hour: int, use_ts_column: bool = False) -> pd.DataFrame:
    """Return only the bars up to and including the given UTC hour.

    Mimics the engine handing the skill data that ends at the current bar, so a
    look-ahead bug (reading future bars) would be exposed by tests.
    """
    if use_ts_column:
        return df[df["ts"].dt.hour <= hour].reset_index(drop=True)
    return df[df.index.hour <= hour]


def test_morning_up_gives_buy_at_2000():
    df = _slice_until(_hourly_day(morning_direction=1.0), 20)
    sig = intraday_tod_btc(df)
    assert isinstance(sig, SkillSignal)
    assert sig.skill_id == "intraday_tod_btc"
    assert sig.side == "BUY"
    assert sig.confidence > 0
    assert sig.stop_pct > 0
    assert sig.take_profit_pct > 0


def test_morning_down_gives_sell_at_2000():
    df = _slice_until(_hourly_day(morning_direction=-1.0), 20)
    sig = intraday_tod_btc(df)
    assert sig.side == "SELL"
    assert sig.confidence > 0


def test_off_window_hour_gives_hold():
    # Current bar is 17:00 UTC -> outside the 20:00 decision window.
    df = _slice_until(_hourly_day(morning_direction=1.0), 17)
    sig = intraday_tod_btc(df)
    assert sig.side == "HOLD"
    assert sig.confidence == 0.0


def test_flat_morning_gives_hold_at_2000():
    df = _slice_until(_hourly_day(morning_direction=0.0), 20)
    sig = intraday_tod_btc(df)
    assert sig.side == "HOLD"


def test_works_with_ts_column():
    df = _slice_until(
        _hourly_day(morning_direction=-1.0, use_ts_column=True), 20, use_ts_column=True
    )
    sig = intraday_tod_btc(df)
    assert sig.side == "SELL"


def test_no_lookahead_uses_only_morning_window():
    """The 20:00 decision must not depend on bars after 14:00.

    Two days share the same morning move but differ wildly after 14:00; the
    signal at 20:00 must be identical, proving only the morning window is used.
    """
    df_a = _hourly_day(morning_direction=1.0)
    df_b = _hourly_day(morning_direction=1.0)
    # Corrupt the afternoon of df_b (15:00..20:00) — these bars are known at
    # 20:00 but must be irrelevant to the decision under the stated thesis.
    df_b.iloc[15:21, df_b.columns.get_loc("close")] = 1.0
    sig_a = intraday_tod_btc(_slice_until(df_a, 20))
    sig_b = intraday_tod_btc(_slice_until(df_b, 20))
    assert sig_a.side == sig_b.side == "BUY"
