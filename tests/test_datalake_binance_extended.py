"""Offline tests for the EXTENDED Binance lake client (Stream 0 re-freeze).

NO live network. Exercises the NEW pure transforms layered on top of the proven
``binance.py`` pagination engine:

  * ``normalize_funding_info`` — parse /fapi/v1/fundingInfo (the variable funding
    interval: some perps moved 8h -> 4h) into a tidy per-symbol frame.
  * ``funding_interval_hours_for`` — resolve a symbol's cadence (default 8h when
    Binance omits it, which it does for every default-settings perp).
  * ``attach_funding_to_bars`` — the CAUSAL lagged forward-fill that pins each
    settled funding print to the right 4h bar with NO look-ahead. This is the
    heart of TACHE 3; it gets a spot-check against a dated event.

The 4h-klines and deep-funding paginators are imported, already proven in
tests/test_datalake_binance.py — we only assert the extended module re-exposes
them and the universe/interval wiring is correct.
"""

from __future__ import annotations

import math

import pandas as pd

from hermes.datalake.sources.binance import fetch_funding_paginated
from hermes.datalake.sources.binance_extended import (
    ALL_PAIRS,
    EXTENDED_PAIRS,
    FOUR_H_MS,
    FUNDING_DEEP_START_MS,
    LIVE_PAIRS,
    attach_funding_to_bars,
    funding_interval_hours_for,
    normalize_funding_info,
)

EIGHT_H_MS = 8 * 3_600_000


# --------------------------------------------------------------------------- #
# universe + interval wiring
# --------------------------------------------------------------------------- #
def test_four_hour_interval_constant():
    assert FOUR_H_MS == 14_400_000  # 4 * 3600 * 1000


def test_live_pairs_are_the_seven():
    assert LIVE_PAIRS == ["BTC", "ETH", "SOL", "AVAX", "SUI", "ARB", "WIF"]


def test_extended_universe_adds_breadth_without_overlap():
    # the extended set must NOT re-list any of the 7 live pairs (no dup pulls)
    assert set(EXTENDED_PAIRS).isdisjoint(set(LIVE_PAIRS))
    # S3 needs >=3 deep instruments; the brief asks for ~10-20 extra
    assert 10 <= len(EXTENDED_PAIRS) <= 20
    # ALL_PAIRS is the concatenation, live first, no dups
    assert ALL_PAIRS == LIVE_PAIRS + EXTENDED_PAIRS
    assert len(set(ALL_PAIRS)) == len(ALL_PAIRS)


# --------------------------------------------------------------------------- #
# REGRESSION: the live /fapi/v1/fundingRate clamp.
# The real endpoint, given startTime<=0 (or absent), returns ONLY the most-recent
# ~200 prints — NOT inception-to-now. That silent clamp is what made the existing
# ``binance_funding`` view a forward-only stub. The deep walk MUST start at a
# strictly-positive startTime (Binance then clamps it up to the perp's listing).
# This fake reproduces the clamp so the fix (FUNDING_DEEP_START_MS >= 1) is locked.
# --------------------------------------------------------------------------- #
def _clamping_funding_fetcher(all_rows, *, recent_n=200):
    """Mimic fapi/v1/fundingRate: startTime<=0 -> only the last `recent_n` rows;
    startTime>=1 -> rows with fundingTime >= startTime (deep), capped at limit."""
    def fetch(symbol, start_ms, limit):
        if start_ms <= 0:
            return all_rows[-recent_n:][:limit]
        return [r for r in all_rows if r["fundingTime"] >= start_ms][:limit]
    return fetch


def test_funding_deep_start_is_strictly_positive():
    # startTime=0 is the trap; the constant must dodge it.
    assert FUNDING_DEEP_START_MS >= 1


def test_funding_start_zero_clamps_to_recent_but_deep_start_walks_full_history():
    step = 8 * 3_600_000
    all_rows = [
        {"symbol": "BTCUSDT", "fundingTime": 1_568_000_000_000 + i * step,
         "fundingRate": "0.00010000"}
        for i in range(2300)  # ~2.1y of 8h prints
    ]
    fetcher = _clamping_funding_fetcher(all_rows, recent_n=200)

    shallow = fetch_funding_paginated(fetcher, "BTC/USDT", "BTCUSDT", start_ms=0)
    deep = fetch_funding_paginated(
        fetcher, "BTC/USDT", "BTCUSDT", start_ms=FUNDING_DEEP_START_MS
    )

    assert len(shallow) == 200  # the clamp — the bug the old view shipped
    assert len(deep) == 2300    # the fix — inception-to-now
    assert deep["ts"].iloc[0] == all_rows[0]["fundingTime"]
    assert deep["ts"].iloc[-1] == all_rows[-1]["fundingTime"]


# --------------------------------------------------------------------------- #
# normalize_funding_info — /fapi/v1/fundingInfo
# --------------------------------------------------------------------------- #
def _info_rows():
    # fapi/v1/fundingInfo only lists perps with NON-default settings; the
    # numeric fields arrive as strings, the interval as an int.
    return [
        {
            "symbol": "BTCUSDT",
            "adjustedFundingRateCap": "0.03000000",
            "adjustedFundingRateFloor": "-0.03000000",
            "fundingIntervalHours": 4,
            "disclaimer": False,
        },
        {
            "symbol": "SOLUSDT",
            "adjustedFundingRateCap": "0.02000000",
            "adjustedFundingRateFloor": "-0.02000000",
            "fundingIntervalHours": 8,
            "disclaimer": False,
        },
    ]


def test_normalize_funding_info_tidy_frame_and_types():
    df = normalize_funding_info(_info_rows(), run_as_of="2026-06-29T00:00:00")
    assert {"symbol", "funding_interval_hours", "funding_rate_cap",
            "funding_rate_floor", "run_as_of"}.issubset(df.columns)
    assert df["funding_interval_hours"].dtype.kind in ("i", "f")
    # the string caps were coerced to float
    btc = df[df["symbol"] == "BTCUSDT"].iloc[0]
    assert btc["funding_interval_hours"] == 4
    assert abs(btc["funding_rate_cap"] - 0.03) < 1e-12
    assert btc["run_as_of"] == "2026-06-29T00:00:00"


def test_normalize_funding_info_empty_is_canonical_empty_frame():
    df = normalize_funding_info([], run_as_of="2026-06-29T00:00:00")
    assert len(df) == 0
    assert "symbol" in df.columns and "funding_interval_hours" in df.columns


# --------------------------------------------------------------------------- #
# funding_interval_hours_for — default 8h when Binance omits the symbol
# --------------------------------------------------------------------------- #
def test_funding_interval_hours_for_reads_info():
    info = normalize_funding_info(_info_rows(), run_as_of="x")
    assert funding_interval_hours_for(info, "BTCUSDT") == 4
    assert funding_interval_hours_for(info, "SOLUSDT") == 8


def test_funding_interval_hours_for_defaults_to_eight_when_absent():
    info = normalize_funding_info(_info_rows(), run_as_of="x")
    # ETHUSDT not in the info payload -> Binance default 8h
    assert funding_interval_hours_for(info, "ETHUSDT") == 8


def test_funding_interval_hours_for_defaults_on_empty_info():
    info = normalize_funding_info([], run_as_of="x")
    assert funding_interval_hours_for(info, "BTCUSDT") == 8


# --------------------------------------------------------------------------- #
# attach_funding_to_bars — CAUSAL lagged forward-fill (no look-ahead)
# --------------------------------------------------------------------------- #
def _bars_df(ts_list):
    return pd.DataFrame({"ts": [int(t) for t in ts_list], "close": [100.0] * len(ts_list)})


def _funding_df(pairs):
    # pairs: list of (ts, rate)
    return pd.DataFrame({"ts": [int(t) for t, _ in pairs],
                         "funding_rate": [float(r) for _, r in pairs]})


def test_attach_forward_fills_last_settled_funding():
    bars = _bars_df([100, 200, 300])
    funding = _funding_df([(150, 0.01)])
    out = attach_funding_to_bars(bars, funding, lag_ms=0)
    # bar 100 precedes the only funding -> NaN; 200 & 300 carry 0.01
    assert math.isnan(out["funding_rate"].iloc[0])
    assert out["funding_rate"].iloc[1] == 0.01
    assert out["funding_rate"].iloc[2] == 0.01


def test_attach_picks_most_recent_not_stale():
    bars = _bars_df([300])
    funding = _funding_df([(150, 0.01), (250, 0.02)])
    out = attach_funding_to_bars(bars, funding, lag_ms=0)
    assert out["funding_rate"].iloc[0] == 0.02


def test_attach_no_lookahead_simultaneous_excluded_with_lag():
    # A funding settling at the SAME ms as the bar opens must NOT leak into that
    # bar once we honor a publication lag; it appears only on the next bar.
    bars = _bars_df([200, 600])
    funding = _funding_df([(200, 0.05)])
    out = attach_funding_to_bars(bars, funding, lag_ms=1)
    assert math.isnan(out["funding_rate"].iloc[0])   # 200 + 1 > 200  -> excluded
    assert out["funding_rate"].iloc[1] == 0.05       # known well before 600


def test_attach_lag_shifts_attachment_forward_by_one_bar():
    # funding at 200; bars every 100. With a lag that pushes effective
    # availability to 250, the 200-bar must miss it and the 300-bar catch it.
    bars = _bars_df([200, 300, 400])
    funding = _funding_df([(200, 0.07)])
    out = attach_funding_to_bars(bars, funding, lag_ms=50)
    assert math.isnan(out["funding_rate"].iloc[0])  # eff 250 > 200
    assert out["funding_rate"].iloc[1] == 0.07       # eff 250 <= 300
    assert out["funding_rate"].iloc[2] == 0.07


def test_attach_preserves_bar_rows_and_order():
    bars = _bars_df([300, 100, 200])  # deliberately unsorted input
    funding = _funding_df([(150, 0.01)])
    out = attach_funding_to_bars(bars, funding, lag_ms=0)
    # every original bar survives; ts ascending in the output
    assert len(out) == 3
    assert out["ts"].tolist() == [100, 200, 300]
    assert "close" in out.columns


def test_attach_empty_funding_yields_all_nan():
    bars = _bars_df([100, 200])
    out = attach_funding_to_bars(bars, _funding_df([]), lag_ms=0)
    assert out["funding_rate"].isna().all()
    assert len(out) == 2


def test_attach_spot_check_dated_event_lands_on_right_4h_bar():
    # SPOT-CHECK vs reality (offline, realistic numbers): a known-shape Binance
    # 8h funding print settling at 2024-03-05 08:00:00 UTC must attach to the
    # 4h bar OPENING at 08:00 (and the 12:00 bar), never the 04:00 bar before it.
    settle = 1_709_625_600_000  # 2024-03-05 08:00:00 UTC in ms
    bar_04 = settle - FOUR_H_MS  # 04:00 UTC
    bar_08 = settle             # 08:00 UTC
    bar_12 = settle + FOUR_H_MS  # 12:00 UTC
    bars = _bars_df([bar_04, bar_08, bar_12])
    funding = _funding_df([(settle, 0.000125)])  # +1.25bps, a realistic print
    out = attach_funding_to_bars(bars, funding, lag_ms=0)
    assert math.isnan(out.loc[out["ts"] == bar_04, "funding_rate"].iloc[0])
    assert out.loc[out["ts"] == bar_08, "funding_rate"].iloc[0] == 0.000125
    assert out.loc[out["ts"] == bar_12, "funding_rate"].iloc[0] == 0.000125
