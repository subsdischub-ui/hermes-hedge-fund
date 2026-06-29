"""Offline tests for the Deribit options-surface lake client.

NO live network. We feed:
  * synthetic ``get_book_summary_by_currency`` chains (list[dict]) to the pure
    RR25 / ATM / term-structure functions and assert the risk-reversal math, and
  * canned ``get_volatility_index_data`` pages to the dvol paginator (a fake
    transport) and assert backward-pagination dedup, the ts->ms convention, and
    the column names.

Imports come straight from the source module so the orchestrator integrates
against exactly what is tested.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from hermes.datalake.sources.deribit import (
    INCEPTION_MS,
    book_summary_to_rows,
    compute_surface_snapshot,
    dvol_pages_to_frame,
    interp_25d_iv,
    merge_surface_snapshot,
    paginate_dvol,
    parse_inst,
)


# --------------------------------------------------------------------------- #
# parse_inst
# --------------------------------------------------------------------------- #
def test_parse_inst_happy_path():
    exp, strike, cp = parse_inst("BTC-27JUN25-60000-C")
    assert exp == dt.datetime(2025, 6, 27, tzinfo=dt.UTC)
    assert strike == 60000.0
    assert cp == "C"


def test_parse_inst_rejects_bad_names():
    assert parse_inst("BTC-PERPETUAL") is None        # wrong arity
    assert parse_inst("BTC-99XYZ99-1-C") is None       # unparseable date
    assert parse_inst("ETH-27JUN25-3000-P-EXTRA") is None


# --------------------------------------------------------------------------- #
# DVOL paginator (fake transport, backward pagination + dedup + ms + columns)
# --------------------------------------------------------------------------- #
class _FakeResp:
    def __init__(self, payload):
        self._p = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._p


def _make_dvol_transport(pages):
    """Return a fake client whose .get(url, params=...) returns the next canned page.

    ``pages`` is a list of (data_rows, continuation) tuples, served in call order.
    Mirrors the real Deribit envelope: {"result": {"data": [...], "continuation": ...}}.
    """
    calls = {"n": 0, "params": []}

    class _Client:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def get(self, url, params=None):
            calls["params"].append(params)
            data, cont = pages[calls["n"]] if calls["n"] < len(pages) else ([], None)
            calls["n"] += 1
            return _FakeResp({"result": {"data": data, "continuation": cont}})

    return _Client(), calls


def test_dvol_paginates_backward_and_dedups():
    # Two overlapping pages (newest first); the boundary row repeats and must dedup.
    h = 3_600_000
    base = 1_700_000_000_000
    # Deribit returns each page ASCENDING (oldest row first); the paginator reads
    # data[0][0] as the page's earliest bar and steps end_timestamp below it.
    # page 1 (newest window): bars at base..base+3h, ascending
    p1 = [[base + 0 * h, 50, 51, 49, 50.2],
          [base + 1 * h, 50, 51, 49, 50.3],
          [base + 2 * h, 50, 51, 49, 50.4],
          [base + 3 * h, 50, 51, 49, 50.5]]
    # page 2 (older window): bars at base-3h..base, ascending; base repeats the boundary
    p2 = [[base - 3 * h, 50, 51, 49, 49.9],
          [base - 2 * h, 50, 51, 49, 50.0],
          [base - 1 * h, 50, 51, 49, 50.1],
          [base + 0 * h, 50, 51, 49, 50.2]]   # duplicate boundary row
    client, calls = _make_dvol_transport([(p1, "cont1"), (p2, None)])

    df = paginate_dvol("BTC", client=client, now_ms=base + 3 * h + 1, max_calls=10)

    # column names exactly as specified
    assert list(df.columns) == ["ts", "dvol_o", "dvol_h", "dvol_l", "dvol"]
    # dedup: 8 raw rows, 1 duplicate -> 7 unique
    assert len(df) == 7
    assert df["ts"].is_monotonic_increasing
    assert df["ts"].duplicated().sum() == 0
    # ms ints
    assert df["ts"].dtype == np.int64
    assert df["ts"].iloc[0] == base - 3 * h
    assert df["ts"].iloc[-1] == base + 3 * h
    # close column carries the 5th field
    assert df["dvol"].iloc[-1] == pytest.approx(50.5)
    # backward pagination: 2nd call's end_timestamp moved below page-1's earliest
    assert calls["params"][1]["end_timestamp"] < base


def test_dvol_stops_on_empty_page():
    client, calls = _make_dvol_transport([([], None)])
    df = paginate_dvol("ETH", client=client, now_ms=1_700_000_000_000, max_calls=10)
    assert df.empty
    assert list(df.columns) == ["ts", "dvol_o", "dvol_h", "dvol_l", "dvol"]
    assert calls["n"] == 1  # one call, empty, stop


def test_dvol_stops_at_inception_without_continuation_loop():
    # A page reaching inception must stop even if continuation is non-null.
    h = 3_600_000
    data = [[INCEPTION_MS + i * h, 50, 51, 49, 50 + i] for i in range(3)]
    client, calls = _make_dvol_transport([(data, "still-going")])
    df = paginate_dvol("BTC", client=client, now_ms=INCEPTION_MS + 10 * h, max_calls=10)
    assert calls["n"] == 1  # stopped at inception, did not chase continuation
    assert df["ts"].iloc[0] == INCEPTION_MS


def test_dvol_pages_to_frame_empty():
    df = dvol_pages_to_frame([])
    assert df.empty
    assert list(df.columns) == ["ts", "dvol_o", "dvol_h", "dvol_l", "dvol"]


# --------------------------------------------------------------------------- #
# RR25 / ATM / term-structure pure math
# --------------------------------------------------------------------------- #
def test_interp_25d_iv_brackets_and_interpolates():
    # deltas 0.10, 0.40 bracket 0.25 -> linear interp of ivs 60, 80 -> 70.
    sub = [
        {"cp": "C", "delta": 0.10, "iv": 60.0},
        {"cp": "C", "delta": 0.40, "iv": 80.0},
        {"cp": "P", "delta": -0.50, "iv": 90.0},
    ]
    assert interp_25d_iv(sub, "C") == pytest.approx(70.0)


def test_interp_25d_iv_uses_abs_delta_for_puts():
    sub = [
        {"cp": "P", "delta": -0.10, "iv": 50.0},
        {"cp": "P", "delta": -0.40, "iv": 80.0},
    ]
    # |delta| points 0.10->50, 0.40->80; interp at 0.25 -> 50 + 0.5*(80-50) = 65.
    assert interp_25d_iv(sub, "P") == pytest.approx(65.0)  # |0.25| bracketed via abs()


def test_interp_25d_iv_none_when_unbracketed_or_thin():
    assert interp_25d_iv([{"cp": "C", "delta": 0.10, "iv": 60.0}], "C") is None  # < 2 pts
    only_far = [
        {"cp": "C", "delta": 0.40, "iv": 80.0},
        {"cp": "C", "delta": 0.45, "iv": 82.0},
    ]
    assert interp_25d_iv(only_far, "C") is None  # 0.25 not bracketed
    missing_delta = [
        {"cp": "C", "delta": None, "iv": 60.0},
        {"cp": "C", "delta": 0.30, "iv": 70.0},
    ]
    assert interp_25d_iv(missing_delta, "C") is None  # drop None-delta -> < 2 pts


def _synthetic_chain(now: dt.datetime, underlying: float = 100.0):
    """A book_summary list with two expiries (~30d, ~90d) and a strike grid,
    plus a couple of unparseable / mark_iv-less rows that must be dropped."""
    e30 = (now + dt.timedelta(days=30)).strftime("%-d%b%y").upper()
    e90 = (now + dt.timedelta(days=90)).strftime("%-d%b%y").upper()
    rows = []

    # 30d expiry: ATM at 100; put-skew so RR25 = IV(25dC) - IV(25dP) < 0.
    for strike, iv in [(80, 75.0), (90, 68.0), (100, 60.0), (110, 64.0), (120, 70.0)]:
        rows.append({"instrument_name": f"BTC-{e30}-{strike}-C",
                     "mark_iv": iv, "open_interest": 10.0, "underlying_price": underlying})
        rows.append({"instrument_name": f"BTC-{e30}-{strike}-P",
                     "mark_iv": iv + 5.0, "open_interest": 8.0, "underlying_price": underlying})
    # 90d expiry: higher ATM (upward term slope), simple grid.
    for strike, iv in [(80, 78.0), (100, 65.0), (120, 72.0)]:
        rows.append({"instrument_name": f"BTC-{e90}-{strike}-C",
                     "mark_iv": iv, "open_interest": 5.0, "underlying_price": underlying})
        rows.append({"instrument_name": f"BTC-{e90}-{strike}-P",
                     "mark_iv": iv + 4.0, "open_interest": 4.0, "underlying_price": underlying})
    # junk rows that must be filtered out
    rows.append({"instrument_name": "BTC-PERPETUAL", "mark_iv": 99.0,
                 "open_interest": 1.0, "underlying_price": underlying})
    rows.append({"instrument_name": f"BTC-{e30}-100-C", "mark_iv": None,
                 "open_interest": 0.0, "underlying_price": underlying})  # no iv
    return rows


def test_book_summary_to_rows_filters_and_parses():
    now = dt.datetime(2025, 1, 1, tzinfo=dt.UTC)
    rows = book_summary_to_rows(_synthetic_chain(now))
    # all kept rows are parseable with a mark_iv
    assert all(r["cp"] in ("C", "P") for r in rows)
    assert all(r["iv"] is not None for r in rows)
    # PERPETUAL + the None-iv row are gone (5*2 + 3*2 = 16 option rows)
    assert len(rows) == 16
    assert all(r["u"] == 100.0 for r in rows)


def test_compute_surface_snapshot_rr_and_skew_sign():
    now = dt.datetime(2025, 1, 1, tzinfo=dt.UTC)
    rows = book_summary_to_rows(_synthetic_chain(now))

    # delta provider: a deterministic stub keyed on (strike, cp) so we don't hit
    # the network. Calls drop monotonically with strike for calls, rise for puts.
    def fake_delta(inst):
        _, strike, cp = parse_inst(inst)
        # crude moneyness->delta map bracketing 0.25 on both wings
        if cp == "C":
            d = {80: 0.85, 90: 0.65, 100: 0.50, 110: 0.20, 120: 0.10}[strike]
        else:
            d = {80: -0.12, 90: -0.22, 100: -0.50, 110: -0.70, 120: -0.88}[strike]
        return d

    snap = compute_surface_snapshot(
        "BTC", rows, now=now, ts_ms=int(now.timestamp() * 1000),
        delta_fn=fake_delta,
    )
    assert isinstance(snap, pd.DataFrame)
    assert len(snap) == 1
    row = snap.iloc[0]
    # schema / stamping
    assert row["ts"] == int(now.timestamp() * 1000)
    assert row["ccy"] == "BTC"
    assert row["underlying"] == pytest.approx(100.0)
    # ATM (strike nearest underlying = 100) IV present and positive
    assert row["atm_iv"] == pytest.approx(60.0)
    # put-skew chain: RR25 = IV(25dC) - IV(25dP) must be negative
    assert np.isfinite(row["rr25_1m"])
    assert row["rr25_1m"] < 0
    # normalized skew = RR25 / ATM_IV, same sign
    assert row["skew_1m"] == pytest.approx(row["rr25_1m"] / row["atm_iv"])
    # term slope: 90d ATM (65) > 30d ATM (60) -> positive
    assert row["term_slope"] > 0


def test_compute_surface_snapshot_fails_soft_on_empty():
    now = dt.datetime(2025, 1, 1, tzinfo=dt.UTC)
    snap = compute_surface_snapshot("BTC", [], now=now,
                                    ts_ms=int(now.timestamp() * 1000),
                                    delta_fn=lambda inst: 0.25)
    assert snap is None


def test_merge_surface_snapshot_keeps_one_row_per_utc_day():
    """The forward-log must be idempotent per day: re-running the freeze twice in
    one day keeps the LATEST snapshot for that day, not two rows. Prevents the
    duplicate-snapshot drift from repeated orchestrator runs."""
    day = 86_400_000
    prev = pd.DataFrame({"ts": [10 * day, 11 * day], "ccy": ["BTC", "BTC"], "rr25_1m": [1.0, 2.0]})
    new = pd.DataFrame({"ts": [11 * day + 3_600_000], "ccy": ["BTC"], "rr25_1m": [2.5]})  # same UTC day as 11*day
    out = merge_surface_snapshot(prev, new)
    assert len(out) == 2                      # day 10 + day 11 (deduped)
    assert out["rr25_1m"].tolist() == [1.0, 2.5]   # day 11 collapsed to the latest
    # and a fresh log (no prev) just returns the new row(s)
    assert len(merge_surface_snapshot(None, new)) == 1
