"""Offline unit tests for the FRED/ALFRED macro-lake source.

NO live network. Feeds captured/synthetic CSV + JSON to the PURE transforms and
injects a fake transport into the keyless fetcher so the wiring is exercised
without hitting fred.stlouisfed.org / api.stlouisfed.org.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from hermes.datalake.sources.fred import (
    FREDGRAPH_CSV_URL,
    SERIES_LATEST,
    fetch_latest_frame,
    normalize_fredgraph_csv,
    parse_alfred_observations,
)

# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #
# A real-shaped fredgraph CSV: ISO observation_date, a holiday encoded as '.',
# a blank ('') row, an out-of-order row, and one genuine numeric value.
_CSV_DFII10 = (
    "observation_date,DFII10\n"
    "2024-01-02,1.70\n"
    "2024-01-01,.\n"      # New Year holiday -> '.' missing
    "2024-01-03,\n"       # blank -> missing
    "2024-01-05,1.75\n"
    "2024-01-04,1.72\n"   # arrives out of order -> must sort
)


def test_normalize_drops_holidays_and_sorts():
    df = normalize_fredgraph_csv(_CSV_DFII10, "DFII10")
    # holiday '.' and blank '' rows dropped -> 3 real observations
    assert list(df["value"]) == [1.70, 1.72, 1.75]
    # sorted ascending by date
    assert list(df["date"].dt.strftime("%Y-%m-%d")) == ["2024-01-02", "2024-01-04", "2024-01-05"]


def test_normalize_dtypes_and_schema():
    df = normalize_fredgraph_csv(_CSV_DFII10, "DFII10")
    assert set(df.columns) == {"date", "value", "series_id", "ts"}
    assert df["date"].dtype == np.dtype("datetime64[ns]")
    assert df["value"].dtype == np.dtype("float64")
    assert (df["series_id"] == "DFII10").all()


def test_ts_is_epoch_milliseconds_int64():
    df = normalize_fredgraph_csv(_CSV_DFII10, "DFII10")
    assert df["ts"].dtype == np.dtype("int64")
    # 2024-01-02T00:00:00Z == 1704153600 s == 1704153600000 ms
    first = df.iloc[0]
    expected_ms = int(pd.Timestamp("2024-01-02", tz="UTC").value // 1_000_000)
    assert int(first["ts"]) == expected_ms


def test_normalize_empty_body_returns_empty_frame():
    df = normalize_fredgraph_csv("observation_date,DFII10\n", "DFII10")
    assert len(df) == 0
    # schema still present so downstream bounds logic is safe
    assert set(df.columns) == {"date", "value", "series_id", "ts"}


def test_normalize_all_holidays_returns_empty():
    txt = "observation_date,DGS10\n2024-01-01,.\n2024-01-02,\n"
    df = normalize_fredgraph_csv(txt, "DGS10")
    assert len(df) == 0


# --------------------------------------------------------------------------- #
# injected fake transport -> exercises fetch_latest_frame end-to-end offline
# --------------------------------------------------------------------------- #
class _FakeTransport:
    """Records the URL it was asked for; returns canned CSV text. Mirrors the
    keyless _get(url) -> str contract."""

    def __init__(self, text: str):
        self._text = text
        self.calls: list[str] = []

    def __call__(self, url: str, timeout: int = 90) -> str:
        self.calls.append(url)
        return self._text


def test_fetch_latest_frame_uses_keyless_url_and_normalizes():
    tr = _FakeTransport(_CSV_DFII10)
    df = fetch_latest_frame("DFII10", get=tr)
    # hit the keyless fredgraph endpoint with the right id
    assert len(tr.calls) == 1
    assert tr.calls[0] == FREDGRAPH_CSV_URL.format(series_id="DFII10")
    assert FREDGRAPH_CSV_URL.startswith("https://fred.stlouisfed.org/graph/fredgraph.csv")
    # and produced a normalized frame
    assert list(df["value"]) == [1.70, 1.72, 1.75]
    assert (df["series_id"] == "DFII10").all()


def test_spec_series_present():
    assert SERIES_LATEST == ["DFII10", "DGS10", "T10Y2Y", "BAMLH0A0HYM2", "DTWEXBGS"]


# --------------------------------------------------------------------------- #
# ALFRED vintage JSON parser (MODE B)
# --------------------------------------------------------------------------- #
def _alfred_json():
    return {
        "observations": [
            {"realtime_start": "2020-06-01", "realtime_end": "2020-06-01",
             "date": "2020-05-28", "value": "5.10"},
            {"realtime_start": "2020-06-01", "realtime_end": "2020-06-01",
             "date": "2020-05-29", "value": "."},   # missing on this vintage
            {"realtime_start": "2020-06-01", "realtime_end": "2020-06-01",
             "date": "2020-06-01", "value": "5.25"},
        ]
    }


def test_parse_alfred_drops_dot_and_normalizes():
    df = parse_alfred_observations(_alfred_json(), "BAMLH0A0HYM2")
    assert list(df["value"]) == [5.10, 5.25]
    assert set(df.columns) == {"date", "value", "series_id", "ts"}
    assert df["value"].dtype == np.dtype("float64")
    assert df["date"].dtype == np.dtype("datetime64[ns]")
    assert df["ts"].dtype == np.dtype("int64")
    assert (df["series_id"] == "BAMLH0A0HYM2").all()


def test_parse_alfred_empty_observations():
    df = parse_alfred_observations({"observations": []}, "BAMLH0A0HYM2")
    assert len(df) == 0
    assert set(df.columns) == {"date", "value", "series_id", "ts"}


def test_parse_alfred_missing_key_is_safe():
    df = parse_alfred_observations({}, "BAMLH0A0HYM2")
    assert len(df) == 0


def test_fetch_vintage_frame_sends_date_only_realtime():
    """ALFRED requires realtime_start/realtime_end as YYYY-MM-DD; passing the full
    run_as_of ISO timestamp 400s. fetch_vintage_frame must truncate to a date.
    Regression for the BAMLH0A0HYM2 vintage 400."""
    from hermes.datalake.sources.fred import fetch_vintage_frame

    captured = {}

    class _Resp:
        status_code = 200

        def json(self):
            return {"observations": [{"date": "2026-06-25", "value": "3.21"}]}

        def raise_for_status(self):
            pass

    class _Session:
        def get(self, url, params=None, timeout=None):
            captured.update(params or {})
            return _Resp()

    df = fetch_vintage_frame(
        "BAMLH0A0HYM2", api_key="K",
        vintage="2026-06-26T16:42:15.925235+00:00",
        sleep=lambda s: None, session=_Session(),
    )
    assert captured["realtime_start"] == "2026-06-26"
    assert captured["realtime_end"] == "2026-06-26"
    assert len(df) == 1 and float(df["value"].iloc[0]) == 3.21
