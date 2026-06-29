"""Offline tests for the gold/FX/metals data-lake source.

NO live network. We feed captured/synthetic data to the PURE transforms and, for
the price path, inject a fake yfinance-like downloader. Covers:
  * yfinance OHLCV frame -> canonical {ts(ms), open, high, low, close, volume, symbol, interval}
  * the ns->ms ts convention + dedup + sort
  * reuse of an existing on-disk csv/parquet price file -> same canonical frame
  * SPDR GLD historical archive xlsx-bytes -> [date_ts(ms), close, ounces, tonnes, ...]
  * SPDR holiday / "US Holiday" placeholder rows dropped
  * empty / missing-field edges yield empty frames (so cached_pull fail-softs)
  * the live run() entrypoint exists with the contract signature
"""

from __future__ import annotations

import inspect
import io
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hermes.datalake.sources.gold import (
    PRICE_COLS,
    canonicalize_ohlcv,
    canonicalize_price_csv,
    parse_spdr_archive_records,
    parse_spdr_archive_xlsx,
    run,
)


# --------------------------------------------------------------------------- #
# yfinance OHLCV -> canonical
# --------------------------------------------------------------------------- #
def _yf_frame(multiindex: bool = False) -> pd.DataFrame:
    idx = pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06"], utc=True)
    df = pd.DataFrame(
        {
            "Open": [1.0, 2.0, 3.0],
            "High": [1.5, 2.5, 3.5],
            "Low": [0.5, 1.5, 2.5],
            "Close": [1.2, 2.2, 3.2],
            "Adj Close": [1.2, 2.2, 3.2],
            "Volume": [100.0, 200.0, 300.0],
        },
        index=idx,
    )
    if multiindex:
        df.columns = pd.MultiIndex.from_product([df.columns, ["GDX"]])
    return df


def test_canonicalize_ohlcv_schema_and_ms():
    out = canonicalize_ohlcv(_yf_frame(), symbol="GDX", interval="1d")
    assert list(out.columns) == PRICE_COLS
    # ts is epoch MILLISECONDS int64
    assert out["ts"].dtype == np.int64
    assert int(out["ts"].iloc[0]) == 1577923200000  # 2020-01-02 00:00 UTC in ms
    # value columns preserved
    assert out["open"].iloc[0] == pytest.approx(1.0)
    assert out["close"].iloc[-1] == pytest.approx(3.2)
    assert (out["symbol"] == "GDX").all()
    assert (out["interval"] == "1d").all()
    assert out["ts"].is_monotonic_increasing


def test_canonicalize_ohlcv_flattens_multiindex():
    out = canonicalize_ohlcv(_yf_frame(multiindex=True), symbol="GDX", interval="1d")
    assert list(out.columns) == PRICE_COLS
    assert len(out) == 3


def test_canonicalize_ohlcv_dedup_and_sort():
    df = _yf_frame()
    dup = pd.concat([df, df.iloc[[0]]])  # duplicate first row
    out = canonicalize_ohlcv(dup, symbol="GDX", interval="1d")
    assert out["ts"].is_unique
    assert out["ts"].is_monotonic_increasing
    assert len(out) == 3


def test_canonicalize_ohlcv_empty_returns_empty():
    out = canonicalize_ohlcv(pd.DataFrame(), symbol="GDX", interval="1d")
    assert out.empty
    assert list(out.columns) == PRICE_COLS


# --------------------------------------------------------------------------- #
# existing on-disk csv reuse (ts already in ms)
# --------------------------------------------------------------------------- #
def test_canonicalize_price_csv_reuse(tmp_path: Path):
    src = tmp_path / "XAUUSD_1d.csv"
    pd.DataFrame(
        {
            "ts": [967593600000, 967680000000, 967593600000],  # last is a dup
            "open": [273.9, 274.8, 273.9],
            "high": [273.9, 278.3, 273.9],
            "low": [273.9, 274.8, 273.9],
            "close": [273.9, 278.3, 273.9],
            "volume": [0.0, 0.0, 0.0],
        }
    ).to_csv(src, index=False)
    out = canonicalize_price_csv(src, symbol="GC=F", interval="1d")
    assert list(out.columns) == PRICE_COLS
    assert out["ts"].dtype == np.int64
    assert out["ts"].is_unique and out["ts"].is_monotonic_increasing
    assert len(out) == 2
    assert (out["symbol"] == "GC=F").all()


# --------------------------------------------------------------------------- #
# SPDR GLD historical archive
# --------------------------------------------------------------------------- #
SPDR_SHEET = "US GLD Historical Archive"


def _spdr_records() -> list[dict]:
    return [
        {
            "Date": "18-Nov-2004",
            "Closing Price": 44.38,
            "Total Ounces of Gold in the Trust": 260000.0,
            "Tonnes of Gold": 8.09,
            "Total Net Asset Value in the Trust": 114920000.0,
            "Daily Share Volume": 5992000.0,
        },
        {
            "Date": "19-Nov-2004",
            "Closing Price": 44.78,
            "Total Ounces of Gold in the Trust": 1859994.06,
            "Tonnes of Gold": 57.85,
            "Total Net Asset Value in the Trust": 828806907.2,
            "Daily Share Volume": 11655000.0,
        },
        {
            # holiday placeholder: numeric cells are the literal string "US Holiday"
            "Date": "25-Nov-2004",
            "Closing Price": "US Holiday",
            "Total Ounces of Gold in the Trust": "US Holiday",
            "Tonnes of Gold": "US Holiday",
            "Total Net Asset Value in the Trust": "US Holiday",
            "Daily Share Volume": "US Holiday",
        },
    ]


def _spdr_xlsx_bytes() -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        pd.DataFrame({"x": ["legal disclaimer text"]}).to_excel(
            xw, sheet_name="Disclaimer", index=False
        )
        pd.DataFrame(_spdr_records()).to_excel(xw, sheet_name=SPDR_SHEET, index=False)
    return buf.getvalue()


def test_parse_spdr_records_normalizes_and_drops_holiday():
    out = parse_spdr_archive_records(_spdr_records())
    # holiday row dropped, 2 real trading days remain
    assert len(out) == 2
    assert "ts" in out.columns and out["ts"].dtype == np.int64
    assert "ounces" in out.columns and "tonnes" in out.columns
    # 18-Nov-2004 00:00 UTC == 1100736000000 ms
    assert int(out["ts"].iloc[0]) == 1100736000000
    assert out["ounces"].iloc[1] == pytest.approx(1859994.06)
    assert out["ts"].is_monotonic_increasing and out["ts"].is_unique
    assert (out["symbol"] == "GLD").all()


def test_parse_spdr_xlsx_bytes_matches_records():
    out = parse_spdr_archive_xlsx(_spdr_xlsx_bytes(), sheet_name=SPDR_SHEET)
    assert len(out) == 2
    assert int(out["ts"].iloc[0]) == 1100736000000
    assert out["tonnes"].iloc[0] == pytest.approx(8.09)


def test_parse_spdr_records_empty():
    assert parse_spdr_archive_records([]).empty
    # missing the ounces field entirely -> rows coerce to NaN and drop -> empty
    out = parse_spdr_archive_records([{"Date": "18-Nov-2004", "Closing Price": 44.38}])
    assert out.empty


# --------------------------------------------------------------------------- #
# contract: run() signature
# --------------------------------------------------------------------------- #
def test_run_signature():
    sig = inspect.signature(run)
    params = list(sig.parameters)
    assert params[:2] == ["raw_root", "manifest"]
    assert "run_as_of" in sig.parameters
    assert sig.parameters["run_as_of"].kind == inspect.Parameter.KEYWORD_ONLY
