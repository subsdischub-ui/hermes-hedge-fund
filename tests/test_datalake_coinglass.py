"""Offline tests for the CoinGlass V4 lake client.

NO live network. We exercise:
  * the pure transforms: futures envelope -> long frame, ts->ms int64 convention,
    aggregated-OI, the OPTIONS wide-map TRANSPOSE (the centerpiece), and the
    options /info snapshot.
  * backward pagination dedup + stop-on-no-new-ts.
  * the CoinGlassClient envelope unwrap (code=='0'), header-driven self-throttle,
    and a fake injectable transport so _get never touches the network.
  * an edge: empty page / missing field / code!='0'.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hermes.datalake.sources.coinglass import (
    BASE_URL,
    CoinGlassClient,
    normalize_aggregated_oi,
    normalize_futures,
    normalize_options,
    normalize_options_info,
    paginate_backward,
)


# --------------------------------------------------------------------------- #
# pure: futures normalization
# --------------------------------------------------------------------------- #
def test_normalize_futures_ts_ms_and_columns():
    data = [
        {"time": 1_700_000_000_000, "top_account_long_percent": 60.0,
         "top_account_short_percent": 40.0, "top_account_long_short_ratio": 1.5},
        {"time": 1_700_021_600_000, "top_account_long_percent": 55.0,
         "top_account_short_percent": 45.0, "top_account_long_short_ratio": 1.22},
    ]
    df = normalize_futures(data, symbol="BTCUSDT", exchange="Binance", interval="6h")
    # ts is an int64 column in MILLISECONDS, named 'ts', not 'time'.
    assert "ts" in df.columns and "time" not in df.columns
    assert df["ts"].dtype == np.int64
    assert df["ts"].tolist() == [1_700_000_000_000, 1_700_021_600_000]
    # value columns survive
    assert df["top_account_long_short_ratio"].tolist() == [1.5, 1.22]
    # provenance columns stamped
    assert (df["source"] == "coinglass").all()
    assert (df["symbol"] == "BTCUSDT").all()
    assert (df["exchange"] == "Binance").all()
    assert (df["interval"] == "6h").all()


def test_normalize_futures_ohlc_open_interest():
    data = [{"time": 1_700_000_000_000, "open": 1.0, "high": 2.0, "low": 0.5, "close": 1.5}]
    df = normalize_futures(data, symbol="ETHUSDT", exchange="Binance", interval="1d")
    assert df.loc[0, "close"] == 1.5
    assert df["ts"].dtype == np.int64


def test_normalize_futures_empty_returns_empty_frame():
    df = normalize_futures([], symbol="BTCUSDT", exchange="Binance", interval="1d")
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0


def test_normalize_futures_coerces_mixed_str_float_values_to_float64():
    """CoinGlass returns OHLC/liquidation as a MIX of str and float in the same
    column (live-verified). The normalizer must coerce value columns to clean
    float64 so the frame is Parquet-writeable and dtype-uniform across symbols.
    Regression for the ArrowTypeError that aborted the CoinGlass pull."""
    data = [
        {"time": 1000, "open": "1.5", "high": 2.0, "low": "1.0", "close": "1.8"},
        {"time": 2000, "open": 1.6, "high": "2.1", "low": 1.1, "close": 1.9},
    ]
    df = normalize_futures(data, symbol="BTCUSDT", exchange="Binance", interval="1d")
    for c in ["open", "high", "low", "close"]:
        assert str(df[c].dtype) == "float64", (c, df[c].dtype)
    assert df["close"].tolist() == [1.8, 1.9]
    assert df["ts"].dtype == np.int64  # ts stays int64, not coerced to float


def test_normalize_aggregated_oi_no_exchange():
    data = [{"time": 1_700_000_000_000, "open": 10, "high": 12, "low": 9, "close": 11}]
    df = normalize_aggregated_oi(data, symbol="BTC", interval="1d")
    assert df["ts"].dtype == np.int64
    assert (df["symbol"] == "BTC").all()
    # aggregated has no single exchange -> 'aggregated'
    assert (df["exchange"] == "aggregated").all()
    assert df.loc[0, "close"] == 11


# --------------------------------------------------------------------------- #
# pure: OPTIONS transpose (the most important test)
# --------------------------------------------------------------------------- #
def _options_payload():
    return {
        "time_list": [1_700_000_000_000, 1_700_086_400_000],
        "price_list": [40000.0, 41000.0],
        "data_map": {
            "Deribit": [1000.0, 1100.0],
            "OKX": [200.0, 250.0],
        },
    }


def test_normalize_options_transpose_wide_to_long():
    df = normalize_options(_options_payload(), symbol="BTC")
    # 2 timestamps x 2 exchanges = 4 long rows
    assert len(df) == 4
    assert set(df.columns) >= {"ts", "exchange", "value", "price", "symbol", "source"}
    assert df["ts"].dtype == np.int64
    # check one cell of the transpose explicitly
    row = df[(df["exchange"] == "Deribit") & (df["ts"] == 1_700_086_400_000)].iloc[0]
    assert row["value"] == 1100.0
    assert row["price"] == 41000.0
    assert row["symbol"] == "BTC"
    assert row["source"] == "coinglass"
    # both exchanges present
    assert set(df["exchange"].unique()) == {"Deribit", "OKX"}


def test_normalize_options_empty_or_missing_map():
    assert len(normalize_options({"time_list": [], "price_list": [], "data_map": {}}, symbol="BTC")) == 0
    # missing keys entirely -> empty frame, not a crash
    assert len(normalize_options({}, symbol="ETH")) == 0


def test_normalize_options_ragged_exchange_series_is_safe():
    """An exchange whose series is shorter than time_list must not crash; only the
    overlapping prefix produces rows."""
    payload = {
        "time_list": [1, 2, 3],
        "price_list": [10.0, 20.0, 30.0],
        "data_map": {"Deribit": [1.0, 2.0]},  # one short
    }
    df = normalize_options(payload, symbol="BTC")
    assert len(df) == 2
    assert df["ts"].tolist() == [1, 2]


def test_normalize_options_info_snapshot():
    payload = [
        {"exchange": "Deribit", "open_interest": 123.0, "volume_usd": 9.0},
        {"exchange": "OKX", "open_interest": 45.0, "volume_usd": 3.0},
    ]
    df = normalize_options_info(payload, symbol="BTC", run_as_of="2026-06-26")
    assert len(df) == 2
    assert (df["symbol"] == "BTC").all()
    assert (df["run_as_of"] == "2026-06-26").all()
    assert (df["source"] == "coinglass").all()
    assert set(df["exchange"]) == {"Deribit", "OKX"}


# --------------------------------------------------------------------------- #
# pure: backward pagination dedup + stop
# --------------------------------------------------------------------------- #
def test_paginate_backward_dedups_and_stops():
    """Three pages going backward via end_time; the 2nd page overlaps the 1st by
    one bar (must dedup on ts); the 3rd page returns only already-seen ts (must
    stop). Final frame is unique + sorted ascending."""
    day = 86_400_000
    page_newest = [
        {"time": 100 * day, "close": 1.0},
        {"time": 99 * day, "close": 2.0},
        {"time": 98 * day, "close": 3.0},
    ]
    page_mid = [
        {"time": 98 * day, "close": 3.0},  # overlap (already seen)
        {"time": 97 * day, "close": 4.0},
        {"time": 96 * day, "close": 5.0},
    ]
    page_old = [
        {"time": 97 * day, "close": 4.0},  # all already seen -> stop after this
        {"time": 96 * day, "close": 5.0},
    ]
    pages = [page_newest, page_mid, page_old]
    calls = {"n": 0, "end_times": []}

    def fetch_page(end_time):
        calls["end_times"].append(end_time)
        i = calls["n"]
        calls["n"] += 1
        if i < len(pages):
            return normalize_futures(pages[i], symbol="BTCUSDT", exchange="Binance", interval="1d")
        return normalize_futures([], symbol="BTCUSDT", exchange="Binance", interval="1d")

    df = paginate_backward(fetch_page, ts_col="ts")
    # unique ts, ascending
    assert df["ts"].is_monotonic_increasing
    assert df["ts"].tolist() == [96 * day, 97 * day, 98 * day, 99 * day, 100 * day]
    assert df["ts"].is_unique
    # first call must pass end_time=None (newest), subsequent ones earliest_ts-1
    assert calls["end_times"][0] is None
    assert calls["end_times"][1] == 98 * day - 1  # earliest of page 1 minus 1
    # it must have STOPPED (not looped forever) once a page added no new ts
    assert calls["n"] <= 4


def test_paginate_backward_empty_first_page():
    def fetch_page(end_time):
        return normalize_futures([], symbol="BTCUSDT", exchange="Binance", interval="1d")

    df = paginate_backward(fetch_page, ts_col="ts")
    assert len(df) == 0


def test_paginate_backward_floor_ts_keeps_recent_and_trims_below_floor():
    """6h-style bounded backfill: walk backward FROM NEWEST, stop once a page
    reaches floor_ts, and trim anything older than the floor (the now-360d cap).
    Regression: the 6h pull must include the most-recent bars, not the oldest."""
    day = 86_400_000
    floor = 96 * day
    page_newest = [
        {"time": 100 * day, "close": 1.0},
        {"time": 99 * day, "close": 2.0},
        {"time": 98 * day, "close": 3.0},
    ]
    page_old = [  # crosses below the floor
        {"time": 97 * day, "close": 4.0},
        {"time": 96 * day, "close": 5.0},
        {"time": 95 * day, "close": 6.0},
    ]
    pages = [page_newest, page_old]
    calls = {"n": 0}

    def fetch_page(end_time):
        i = calls["n"]
        calls["n"] += 1
        rows = pages[i] if i < len(pages) else []
        return normalize_futures(rows, symbol="BTCUSDT", exchange="Binance", interval="6h")

    df = paginate_backward(fetch_page, ts_col="ts", floor_ts=floor)
    assert df["ts"].max() == 100 * day      # most-recent bar present (the whole point)
    assert df["ts"].min() == floor          # 95*day trimmed out (below the 360d floor)
    assert (df["ts"] >= floor).all()
    assert calls["n"] == 2                   # stopped after the page that crossed the floor


# --------------------------------------------------------------------------- #
# the client: envelope unwrap + injectable transport + self-throttle
# --------------------------------------------------------------------------- #
class _FakeResponse:
    def __init__(self, payload, *, headers=None, status_code=200):
        self._payload = payload
        self.headers = headers or {}
        self.status_code = status_code

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class _FakeTransport:
    """Records requests; replies from a queued list of (payload, headers) tuples."""

    def __init__(self, replies):
        self._replies = list(replies)
        self.calls = []

    def get(self, url, *, headers=None, params=None):
        self.calls.append({"url": url, "headers": headers, "params": params})
        payload, hdrs = self._replies.pop(0)
        return _FakeResponse(payload, headers=hdrs)


def test_client_get_unwraps_envelope_and_sends_auth_header():
    transport = _FakeTransport([
        ({"code": "0", "msg": "success", "data": [{"time": 1, "close": 9.0}]},
         {"API-KEY-USE-LIMIT": "1", "API-KEY-MAX-LIMIT": "30"}),
    ])
    client = CoinGlassClient(api_key="SECRET", transport=transport)
    data = client._get("/api/futures/open-interest/history", symbol="BTCUSDT")
    assert data == [{"time": 1, "close": 9.0}]
    # auth header on every request
    assert transport.calls[0]["headers"]["CG-API-KEY"] == "SECRET"
    # base url joined
    assert transport.calls[0]["url"].startswith(BASE_URL)
    assert transport.calls[0]["url"].endswith("/api/futures/open-interest/history")
    # params forwarded
    assert transport.calls[0]["params"]["symbol"] == "BTCUSDT"


def test_client_get_non_zero_code_returns_empty():
    transport = _FakeTransport([
        ({"code": "50001", "msg": "rate limit", "data": None}, {}),
    ])
    client = CoinGlassClient(api_key="SECRET", transport=transport)
    assert client._get("/api/option/exchange-oi-history", symbol="WIF") == []


def test_client_get_missing_data_key_returns_empty():
    transport = _FakeTransport([({"code": "0", "msg": "success"}, {})])
    client = CoinGlassClient(api_key="SECRET", transport=transport)
    assert client._get("/api/futures/funding-rate/history", symbol="BTCUSDT") == []


def test_client_throttles_via_ratelimiter(monkeypatch):
    """Two _get calls must invoke the RateLimiter once each (so we never exceed
    30/min). We drive a fake clock + sleep so no real waiting happens."""
    clock = {"t": 0.0}
    slept = []

    transport = _FakeTransport([
        ({"code": "0", "msg": "success", "data": [{"time": 1}]}, {}),
        ({"code": "0", "msg": "success", "data": [{"time": 2}]}, {}),
    ])
    client = CoinGlassClient(
        api_key="SECRET",
        transport=transport,
        monotonic=lambda: clock["t"],
        sleep=lambda s: (slept.append(s), clock.__setitem__("t", clock["t"] + s)),
    )
    client._get("/api/futures/funding-rate/history", symbol="BTCUSDT")
    client._get("/api/futures/funding-rate/history", symbol="ETHUSDT")
    # second back-to-back call must have been spaced by the 30/min min-interval (2s)
    assert slept and slept[-1] == pytest.approx(2.0)


def test_client_supported_pairs_parsing():
    """supported-exchange-pairs returns a venue->list-of-pair-dicts map; the
    helper must report which (exchange, base) pairs are listed so run() can skip
    non-listed pairs without burning budget."""
    payload = {
        "code": "0", "msg": "success",
        "data": {
            "Binance": [
                {"instrument_id": "BTCUSDT", "base_asset": "BTC", "quote_asset": "USDT"},
                {"instrument_id": "ETHUSDT", "base_asset": "ETH", "quote_asset": "USDT"},
            ],
        },
    }
    transport = _FakeTransport([(payload, {})])
    client = CoinGlassClient(api_key="SECRET", transport=transport)
    listed = client.supported_pairs(exchange="Binance")
    assert "BTCUSDT" in listed
    assert "ETHUSDT" in listed
    assert "WIFUSDT" not in listed
