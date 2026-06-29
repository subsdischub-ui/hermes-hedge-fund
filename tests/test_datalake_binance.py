"""Offline tests for the Binance USDT-M PERP klines + funding lake client.

NO live network. We exercise the pure transforms (the pagination dedup/stop
engine, the ts->ms normalization, the funding startTime walk, and the gap
count) by feeding canned bars / JSON pages to an injected fake ccxt client and
a fake funding-page fetcher — mirroring the offline-fake pattern used in
tests/test_newdata.py and tests/test_datalake_core.py.
"""

from __future__ import annotations

from hermes.datalake.sources.binance import (
    DAY_MS,
    SIX_H_MS,
    fetch_funding_paginated,
    fetch_ohlcv_paginated,
    gap_count,
    normalize_funding,
    normalize_ohlcv,
)


# --------------------------------------------------------------------------- #
# Fake ccxt-style OHLCV client: serves canned bars in 'limit'-sized pages from
# a master list, honoring CCXT's `since` (advance to last_ts re-serves boundary)
# and returning [] once exhausted (so pagination must stop).
# --------------------------------------------------------------------------- #
class FakeOHLCVClient:
    def __init__(self, bars: list[list[float]]):
        # bars: [[ts, o, h, l, c, v], ...] sorted ascending
        self.bars = sorted(bars, key=lambda r: r[0])
        self.calls: list[tuple[str, str, int, int]] = []

    def fetch_ohlcv(self, symbol, timeframe, since=0, limit=1000):
        self.calls.append((symbol, timeframe, since, limit))
        rows = [r for r in self.bars if r[0] >= since]
        return [list(r) for r in rows[:limit]]


def _bars(start_ts, n, interval_ms, price=100.0):
    return [
        [start_ts + i * interval_ms, price, price + 1, price - 1, price + 0.5, 10.0 + i]
        for i in range(n)
    ]


# --------------------------------------------------------------------------- #
# normalize_ohlcv — canonical columns, ts is int64 ms, tagging columns added
# --------------------------------------------------------------------------- #
def test_normalize_ohlcv_canonical_columns_and_ms_int64():
    raw = _bars(1_600_000_000_000, 3, DAY_MS)
    df = normalize_ohlcv(raw, symbol="BTC/USDT", exchange="binanceusdm", interval="1d")
    assert list(df.columns)[:6] == ["ts", "open", "high", "low", "close", "volume"]
    assert "symbol" in df.columns and "exchange" in df.columns and "interval" in df.columns
    assert str(df["ts"].dtype) == "int64"
    assert df["symbol"].iloc[0] == "BTC/USDT"
    assert df["exchange"].iloc[0] == "binanceusdm"
    assert df["interval"].iloc[0] == "1d"
    # ts values preserved exactly, sorted ascending
    assert df["ts"].tolist() == [r[0] for r in raw]


def test_normalize_ohlcv_empty_returns_canonical_empty_frame():
    df = normalize_ohlcv([], symbol="ETH/USDT", exchange="binanceusdm", interval="6h")
    assert list(df.columns)[:6] == ["ts", "open", "high", "low", "close", "volume"]
    assert len(df) == 0


# --------------------------------------------------------------------------- #
# fetch_ohlcv_paginated — forward pagination, dedup boundary bar, stop on empty
# --------------------------------------------------------------------------- #
def test_ohlcv_pagination_walks_full_history_and_dedups():
    # 2500 daily bars, page size 1000 -> needs 3 pages; boundary bar re-served.
    raw = _bars(1_600_000_000_000, 2500, DAY_MS)
    client = FakeOHLCVClient(raw)
    df = fetch_ohlcv_paginated(
        client, "BTC/USDT", "1d", since_ms=0, interval_ms=DAY_MS, limit=1000, max_batches=50
    )
    assert len(df) == 2500  # no duplicate boundary bars
    assert df["ts"].is_monotonic_increasing
    assert df["ts"].is_unique
    assert df["ts"].iloc[0] == raw[0][0]
    assert df["ts"].iloc[-1] == raw[-1][0]


def test_ohlcv_pagination_advances_since_past_boundary_bar():
    raw = _bars(1_600_000_000_000, 1500, DAY_MS)
    client = FakeOHLCVClient(raw)
    fetch_ohlcv_paginated(
        client, "ETH/USDT", "1d", since_ms=0, interval_ms=DAY_MS, limit=1000, max_batches=50
    )
    # second call's `since` must be the first page's last ts + interval (forward progress)
    assert client.calls[0][2] == 0
    assert client.calls[1][2] == raw[999][0] + DAY_MS


def test_ohlcv_pagination_stops_on_empty_page():
    raw = _bars(1_600_000_000_000, 10, DAY_MS)
    client = FakeOHLCVClient(raw)
    df = fetch_ohlcv_paginated(
        client, "SOL/USDT", "1d", since_ms=0, interval_ms=DAY_MS, limit=1000, max_batches=50
    )
    assert len(df) == 10
    # exactly two calls: the page with data, then the empty one that stops the loop
    assert len(client.calls) == 2


def test_ohlcv_pagination_respects_max_batches():
    raw = _bars(1_600_000_000_000, 5000, DAY_MS)
    client = FakeOHLCVClient(raw)
    df = fetch_ohlcv_paginated(
        client, "BTC/USDT", "1d", since_ms=0, interval_ms=DAY_MS, limit=1000, max_batches=2
    )
    # capped at 2 batches * 1000 (minus 1 dedup boundary) -> < full history
    assert len(client.calls) == 2
    assert len(df) <= 2000


def test_ohlcv_pagination_stops_on_no_forward_progress():
    # A pathological client that always returns the SAME single bar regardless
    # of `since` must not loop forever.
    class Stuck:
        def __init__(self):
            self.n = 0

        def fetch_ohlcv(self, symbol, timeframe, since=0, limit=1000):
            self.n += 1
            return [[1_600_000_000_000, 1, 1, 1, 1, 1]]

    s = Stuck()
    df = fetch_ohlcv_paginated(
        s, "BTC/USDT", "1d", since_ms=0, interval_ms=DAY_MS, limit=1000, max_batches=50
    )
    assert len(df) == 1
    assert s.n < 50  # broke out on no-progress, not by exhausting max_batches


# --------------------------------------------------------------------------- #
# normalize_funding — fundingTime->ts(ms int64), fundingRate->funding_rate
# --------------------------------------------------------------------------- #
def _funding_rows(start_ms, n, step_ms=8 * 3_600_000):
    return [
        {"symbol": "BTCUSDT", "fundingTime": start_ms + i * step_ms,
         "fundingRate": f"{0.0001 * (i + 1):.8f}"}
        for i in range(n)
    ]


def test_normalize_funding_ms_int64_and_float_rate():
    rows = _funding_rows(1_600_000_000_000, 3)
    df = normalize_funding(rows, symbol="BTC/USDT")
    assert list(df.columns)[:2] == ["ts", "funding_rate"]
    assert "symbol" in df.columns
    assert str(df["ts"].dtype) == "int64"
    assert df["funding_rate"].dtype == float
    assert df["ts"].tolist() == [r["fundingTime"] for r in rows]
    assert df["funding_rate"].iloc[0] == 0.0001


def test_normalize_funding_empty_returns_canonical_empty_frame():
    df = normalize_funding([], symbol="WIF/USDT")
    assert list(df.columns)[:2] == ["ts", "funding_rate"]
    assert len(df) == 0


# --------------------------------------------------------------------------- #
# fetch_funding_paginated — startTime walk, dedup, stop on short/empty page
# --------------------------------------------------------------------------- #
def test_funding_pagination_walks_and_dedups():
    step = 8 * 3_600_000
    all_rows = _funding_rows(1_600_000_000_000, 2300, step)

    def page_fetcher(symbol, start_ms, limit):
        # mirror fapi: rows with fundingTime >= start_ms, capped at limit
        rows = [r for r in all_rows if r["fundingTime"] >= start_ms]
        return rows[:limit]

    df = fetch_funding_paginated(
        page_fetcher, "BTC/USDT", "BTCUSDT", start_ms=0, limit=1000, max_pages=50
    )
    assert len(df) == 2300
    assert df["ts"].is_unique
    assert df["ts"].is_monotonic_increasing
    assert df["ts"].iloc[0] == all_rows[0]["fundingTime"]
    assert df["ts"].iloc[-1] == all_rows[-1]["fundingTime"]


def test_funding_pagination_stops_on_short_page():
    rows = _funding_rows(1_600_000_000_000, 10)

    seen_starts: list[int] = []

    def page_fetcher(symbol, start_ms, limit):
        seen_starts.append(start_ms)
        return [r for r in rows if r["fundingTime"] >= start_ms][:limit]

    df = fetch_funding_paginated(
        page_fetcher, "ETH/USDT", "ETHUSDT", start_ms=0, limit=1000, max_pages=50
    )
    assert len(df) == 10
    # a short (<limit) page means we have everything -> only one call
    assert len(seen_starts) == 1


def test_funding_pagination_stops_on_empty_first_page():
    def page_fetcher(symbol, start_ms, limit):
        return []

    df = fetch_funding_paginated(
        page_fetcher, "SUI/USDT", "SUIUSDT", start_ms=0, limit=1000, max_pages=50
    )
    assert len(df) == 0
    assert list(df.columns)[:2] == ["ts", "funding_rate"]


def test_funding_pagination_advances_starttime_past_last():
    step = 8 * 3_600_000
    all_rows = _funding_rows(1_600_000_000_000, 1500, step)
    seen_starts: list[int] = []

    def page_fetcher(symbol, start_ms, limit):
        seen_starts.append(start_ms)
        return [r for r in all_rows if r["fundingTime"] >= start_ms][:limit]

    fetch_funding_paginated(
        page_fetcher, "BTC/USDT", "BTCUSDT", start_ms=0, limit=1000, max_pages=50
    )
    # second page must start strictly after the first page's last fundingTime
    assert seen_starts[0] == 0
    assert seen_starts[1] == all_rows[999]["fundingTime"] + 1


# --------------------------------------------------------------------------- #
# gap_count — wraps core.detect_gaps over the ms grid
# --------------------------------------------------------------------------- #
def test_gap_count_zero_on_regular_grid():
    df = normalize_ohlcv(
        _bars(1_600_000_000_000, 5, DAY_MS),
        symbol="BTC/USDT", exchange="binanceusdm", interval="1d",
    )
    assert gap_count(df, DAY_MS) == 0


def test_gap_count_detects_missing_bar():
    raw = _bars(1_600_000_000_000, 5, DAY_MS)
    del raw[2]  # punch a hole -> a 2*interval jump
    df = normalize_ohlcv(raw, symbol="BTC/USDT", exchange="binanceusdm", interval="1d")
    assert gap_count(df, DAY_MS) == 1


def test_gap_count_empty_frame_is_zero():
    df = normalize_ohlcv([], symbol="BTC/USDT", exchange="binanceusdm", interval="6h")
    assert gap_count(df, SIX_H_MS) == 0
