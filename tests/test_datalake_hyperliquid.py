"""TDD — Hyperliquid public-API lake source (Stream 0-alpha: cross-venue + hourly funding).

DATALAKE source (read-only history) — DISTINCT from ``hermes.execution.hyperliquid``
(the live-order adapter, off-limits). Public ``POST /info`` endpoint, no key/KYC.

Two views:
  * ``hyperliquid_funding_1h`` — HOURLY settled funding (HL pays hourly = 1/8 of the
    8h-equivalent rate; the stored rate is the per-hour rate -> annualize x8760).
  * ``hyperliquid_klines``     — perp OHLCV (4h, carry-grid-aligned).

Symbol is HL-native BARE (``BTC``); the cross-venue funding join normalizes both
venues to bare. ts is epoch ms int64 (lake convention). candleSnapshot is hard-capped
at 5000 rows/request -> the fetcher MUST paginate by time window.

All real logic is PURE (normalize_* + fetch_*_paginated over an INJECTED ``poster``)
and tested offline; ``run()`` / ``_make_hl_poster`` do live network, never run here.
"""

from __future__ import annotations

import pytest

from hermes.datalake.sources import hyperliquid as hl
from hermes.research.s10_provenance import detect_symbol_format, normalize_symbol


# --------------------------------------------------------------------------- #
# fakes — an injected `poster(payload) -> parsed json` mimicking POST /info
# --------------------------------------------------------------------------- #
class _FakeCandlePoster:
    """Mimic the REAL candleSnapshot: it is END-ANCHORED — it returns the MOST RECENT
    ``cap`` candles in [startTime, endTime] (verified live: requesting startTime far in
    the past still returns only the last ~5000 candles ending at endTime). A paginator
    that walks FORWARD from an early startTime silently truncates older history."""

    def __init__(self, candles: list[dict], cap: int = 5000):
        self._candles = sorted(candles, key=lambda r: r["t"])
        self._cap = cap
        self.payloads: list[dict] = []

    def __call__(self, payload: dict):
        self.payloads.append(payload)
        req = payload["req"]
        s, e = int(req["startTime"]), int(req["endTime"])
        window = [c for c in self._candles if s <= c["t"] <= e]
        return window[-self._cap:]  # END-anchored: most-recent `cap` rows


class _FakeFundingPoster:
    """Mimic fundingHistory: rows with time >= startTime, capped at ``cap``."""

    def __init__(self, rows: list[dict], cap: int = 500):
        self._rows = sorted(rows, key=lambda r: r["time"])
        self._cap = cap
        self.payloads: list[dict] = []

    def __call__(self, payload: dict):
        self.payloads.append(payload)
        s = int(payload["startTime"])
        window = [r for r in self._rows if r["time"] >= s]
        return window[: self._cap]


def _candle(t, interval="4h", coin="BTC"):
    return {"t": t, "T": t + 1, "o": "100.5", "h": "101.0", "l": "99.0",
            "c": "100.0", "v": "12.5", "n": 7, "i": interval, "s": coin}


def _funding(t, rate="0.0000125", coin="BTC"):
    return {"coin": coin, "fundingRate": rate, "premium": "0.0001", "time": t}


HOUR_MS = 3_600_000
FOUR_H_MS = 14_400_000


# --------------------------------------------------------------------------- #
# normalize_hl_klines
# --------------------------------------------------------------------------- #
def test_normalize_klines_schema_dtypes_and_string_prices():
    df = hl.normalize_hl_klines([_candle(1_700_000_000_000)], symbol="BTC", interval="4h")
    assert list(df.columns) == [
        "ts", "open", "high", "low", "close", "volume", "symbol", "exchange", "interval",
    ]
    assert df["ts"].dtype == "int64"
    for c in ("open", "high", "low", "close", "volume"):
        assert df[c].dtype == "float64"
    row = df.iloc[0]
    assert row["ts"] == 1_700_000_000_000           # ts taken from candle OPEN time `t`
    assert row["open"] == 100.5 and row["close"] == 100.0   # string -> float
    assert row["symbol"] == "BTC"                   # HL bare symbol
    assert row["exchange"] == "hyperliquid"
    assert row["interval"] == "4h"


def test_normalize_klines_empty_is_canonical_empty_frame():
    df = hl.normalize_hl_klines([], symbol="ETH", interval="4h")
    assert len(df) == 0
    assert df["ts"].dtype == "int64"
    assert list(df.columns)[:6] == ["ts", "open", "high", "low", "close", "volume"]


def test_normalize_klines_dedups_and_sorts_on_ts():
    rows = [_candle(300), _candle(100), _candle(100), _candle(200)]
    df = hl.normalize_hl_klines(rows, symbol="BTC", interval="4h")
    assert df["ts"].tolist() == [100, 200, 300]


# --------------------------------------------------------------------------- #
# normalize_hl_funding
# --------------------------------------------------------------------------- #
def test_normalize_funding_schema_and_hourly_rate():
    df = hl.normalize_hl_funding([_funding(1_700_000_000_000, "0.0000125")], coin="BTC")
    assert list(df.columns) == ["ts", "funding_rate", "premium", "symbol"]
    assert df["ts"].dtype == "int64"
    assert df["funding_rate"].dtype == "float64"
    row = df.iloc[0]
    assert row["ts"] == 1_700_000_000_000           # ts from `time`
    assert abs(row["funding_rate"] - 0.0000125) < 1e-15   # string -> float, per-HOUR rate
    assert row["symbol"] == "BTC"


def test_normalize_funding_empty_is_canonical_empty_frame():
    df = hl.normalize_hl_funding([], coin="SOL")
    assert len(df) == 0
    assert df["ts"].dtype == "int64"
    assert "funding_rate" in df.columns and "symbol" in df.columns


# --------------------------------------------------------------------------- #
# parse_hl_universe (metaAndAssetCtxs)
# --------------------------------------------------------------------------- #
def test_parse_universe_extracts_coin_names():
    meta_and_ctxs = [
        {"universe": [{"name": "BTC", "szDecimals": 5, "maxLeverage": 50},
                      {"name": "ETH", "szDecimals": 4, "maxLeverage": 50},
                      {"name": "HYPE", "szDecimals": 2, "maxLeverage": 5}]},
        [{"markPx": "1"}, {"markPx": "2"}, {"markPx": "3"}],
    ]
    names = hl.parse_hl_universe(meta_and_ctxs)
    assert names == ["BTC", "ETH", "HYPE"]


def test_parse_universe_handles_bare_list_shape():
    # defensive: some responses give the universe list directly
    names = hl.parse_hl_universe([[{"name": "BTC"}, {"name": "SOL"}]])
    assert names == ["BTC", "SOL"]


# --------------------------------------------------------------------------- #
# paginated candle fetch — forward walk, 5000-cap, dedup boundary, stop empty
# --------------------------------------------------------------------------- #
def test_candle_paginator_recovers_full_history_past_the_cap():
    # 5 candles, cap=2 over an END-ANCHORED API -> the paginator MUST walk BACKWARD to
    # recover the OLDEST candles (a forward walk would return only the most-recent 2).
    candles = [_candle(1_700_000_000_000 + i * FOUR_H_MS) for i in range(5)]
    poster = _FakeCandlePoster(candles, cap=2)
    df = hl.fetch_hl_candles_paginated(
        poster, "BTC", "4h",
        start_ms=0, end_ms=1_700_000_000_000 + 10 * FOUR_H_MS,
        interval_ms=FOUR_H_MS, max_batches=10,
    )
    assert len(df) == 5
    assert df["ts"].is_monotonic_increasing and df["ts"].is_unique
    # the regression assertion: the EARLIEST candle must be present (forward walk loses it)
    assert df["ts"].iloc[0] == 1_700_000_000_000
    assert (df["exchange"] == "hyperliquid").all()
    assert poster.payloads[0]["type"] == "candleSnapshot"
    assert "req" in poster.payloads[0]


def test_candle_paginator_empty_yields_empty_frame():
    poster = _FakeCandlePoster([], cap=5000)
    df = hl.fetch_hl_candles_paginated(
        poster, "WIF", "4h", start_ms=0, end_ms=10**13, interval_ms=FOUR_H_MS,
    )
    assert len(df) == 0


# --------------------------------------------------------------------------- #
# paginated funding fetch — forward walk, stop empty / no-progress
# --------------------------------------------------------------------------- #
def test_funding_paginator_walks_and_recovers_all_rows():
    rows = [_funding(1_700_000_000_000 + i * HOUR_MS, rate="0.00001") for i in range(7)]
    poster = _FakeFundingPoster(rows, cap=3)
    df = hl.fetch_hl_funding_paginated(
        poster, "BTC", start_ms=0, end_ms=10**13, max_pages=10,
    )
    assert len(df) == 7
    assert df["ts"].is_monotonic_increasing and df["ts"].is_unique
    assert poster.payloads[0]["type"] == "fundingHistory"


# --------------------------------------------------------------------------- #
# hl_request — per-request throttle + 429 backoff (the rate-limit fix)
# --------------------------------------------------------------------------- #
class _FakeResp:
    def __init__(self, status, payload=None):
        self.status_code = status
        self._payload = payload

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class _FakeHTTP:
    """Returns the queued responses in order, recording each POST."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = 0

    def post(self, url, json=None):
        self.calls += 1
        return self._responses.pop(0)


def test_hl_request_retries_on_429_then_succeeds():
    slept: list[float] = []
    http = _FakeHTTP([_FakeResp(429), _FakeResp(429), _FakeResp(200, [{"ok": 1}])])
    out = hl.hl_request(http, {"type": "x"}, sleep=slept.append, max_retries=6)
    assert out == [{"ok": 1}]
    assert http.calls == 3           # two 429s then a success
    assert len(slept) == 2           # backed off before each retry


def test_hl_request_raises_after_exhausting_retries():
    http = _FakeHTTP([_FakeResp(429)] * 4)
    with pytest.raises(RuntimeError):
        hl.hl_request(http, {"type": "x"}, sleep=lambda _s: None, max_retries=4)


def test_hl_request_throttles_via_rate_limiter():
    # the limiter MUST be consulted before each request (per-request spacing, the bug fix)
    waits: list[int] = []

    class _Rate:
        def wait(self):
            waits.append(1)

    http = _FakeHTTP([_FakeResp(200, [])])
    hl.hl_request(http, {"type": "x"}, rate=_Rate(), sleep=lambda _s: None)
    assert len(waits) == 1


# --------------------------------------------------------------------------- #
# symbol-format: HL bare joins cross-venue after normalize
# --------------------------------------------------------------------------- #
def test_hl_symbol_is_bare_and_cross_venue_normalizable():
    # HL stores bare 'BTC'; binance funding is 'BTC/USDT' (slash). Cross-venue carry
    # joins after normalizing BOTH to bare.
    assert detect_symbol_format("BTC") == "bare"
    assert normalize_symbol("BTC/USDT", "bare") == "BTC"
    assert normalize_symbol("BTC", "bare") == "BTC"
