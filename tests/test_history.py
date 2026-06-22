"""Tests for the long-history data foundation (paginated fetch, gap detect, CSV IO).

All tests use a FAKE injected client returning canned OHLCV batches — no network.
Hand-built fixtures with overlapping batches verify dedup+sort; an injected hole
verifies gap detection; round-trip verifies CSV save/load.
"""

from __future__ import annotations

import pandas as pd

from hermes.data.history import (
    detect_gaps,
    fetch_history,
    hf_symbol,
    load_history,
    normalize_hf_klines,
    save_history,
)

_TF_MS = 3_600_000  # 1h in ms
_COLS = ["ts", "open", "high", "low", "close", "volume"]
# Canonical snake_case names for the per-bar microstructure columns (Strategy #2).
_MICRO_COLS = ["quote_asset_volume", "num_trades", "taker_buy_base", "taker_buy_quote"]


def _bar(ts: int) -> list[float]:
    """A deterministic OHLCV row keyed off the timestamp (so dedup is observable)."""
    base = float(ts // _TF_MS)
    return [ts, base, base + 1.0, base - 1.0, base + 0.5, 10.0]


class _FakeClient:
    """Sync fake exchange: serves bars from a fixed grid, batches overlap by design.

    fetch_ohlcv returns up to `limit` bars with ts >= since. Successive paginated
    calls (since advanced to last_ts) re-serve the boundary bar, exercising dedup.
    """

    def __init__(self, start_ms: int, n_bars: int, skip: set[int] | None = None):
        skip = skip or set()
        self.grid = [start_ms + i * _TF_MS for i in range(n_bars) if i not in skip]
        self.calls = 0

    def fetch_ohlcv(self, symbol, timeframe, since, limit):
        self.calls += 1
        rows = [t for t in self.grid if t >= since][:limit]
        return [_bar(t) for t in rows]


class _AsyncFakeClient(_FakeClient):
    """Async variant: fetch_ohlcv is a coroutine (CCXT async_support style)."""

    async def fetch_ohlcv(self, symbol, timeframe, since, limit):
        return super().fetch_ohlcv(symbol, timeframe, since, limit)


async def test_fetch_history_paginates_dedups_and_sorts():
    # 5 batches of 1000 with a 1-bar overlap each time -> ~4001 unique bars.
    client = _FakeClient(start_ms=0, n_bars=4001)
    df = await fetch_history("BTC/USDC", "1h", since_ms=0, client=client, limit=1000)
    assert list(df.columns) == _COLS
    # Deduped on timestamp: no duplicate ts despite overlapping batches.
    assert df["ts"].is_unique
    assert len(df) == 4001
    # Sorted ascending.
    assert df["ts"].is_monotonic_increasing
    assert int(df["ts"].iloc[0]) == 0
    assert int(df["ts"].iloc[-1]) == 4000 * _TF_MS
    # More than one batch was needed.
    assert client.calls > 1


async def test_fetch_history_supports_async_client():
    client = _AsyncFakeClient(start_ms=0, n_bars=2500)
    df = await fetch_history("ETH/USDC", "1h", since_ms=0, client=client, limit=1000)
    assert df["ts"].is_unique
    assert df["ts"].is_monotonic_increasing
    assert len(df) == 2500


async def test_fetch_history_respects_max_batches():
    client = _FakeClient(start_ms=0, n_bars=10_000)
    df = await fetch_history(
        "BTC/USDC", "1h", since_ms=0, client=client, max_batches=2, limit=1000
    )
    # Capped at 2 batches -> at most ~2000 bars (one overlap removed).
    assert client.calls == 2
    assert len(df) <= 2000


async def test_fetch_history_stops_on_empty_batch():
    client = _FakeClient(start_ms=0, n_bars=500)
    df = await fetch_history(
        "BTC/USDC", "1h", since_ms=0, client=client, max_batches=50, limit=1000
    )
    assert len(df) == 500
    # One full batch returns everything; the next returns empty and stops the loop.
    assert client.calls == 2


def test_detect_gaps_finds_injected_hole():
    # Contiguous grid except bars 100..104 removed -> a single >1.5x gap.
    client = _FakeClient(start_ms=0, n_bars=200, skip={100, 101, 102, 103, 104})
    df = pd.DataFrame([_bar(t) for t in client.grid], columns=_COLS)
    gaps = detect_gaps(df, _TF_MS)
    assert len(gaps) == 1
    start_ts, end_ts = gaps[0]
    # Gap spans the bar before the hole to the bar after it.
    assert start_ts == 99 * _TF_MS
    assert end_ts == 105 * _TF_MS


def test_detect_gaps_none_when_contiguous():
    df = pd.DataFrame([_bar(t) for t in range(0, 50 * _TF_MS, _TF_MS)], columns=_COLS)
    assert detect_gaps(df, _TF_MS) == []


def test_hf_symbol_maps_hermes_pair_to_binance_usdt():
    assert hf_symbol("BTC/USDC") == "BTCUSDT"
    assert hf_symbol("ETH/USDC") == "ETHUSDT"
    assert hf_symbol("SOL/USDC") == "SOLUSDT"


def test_normalize_hf_klines_maps_open_time_to_ms_and_selects_cols():
    raw = pd.DataFrame(
        {
            "Open time": ["2017-08-17 04:00:00", "2017-08-17 08:00:00"],
            "open": [4261.48, 4333.32],
            "high": [4349.99, 4485.39],
            "low": [4261.32, 4333.32],
            "close": [4349.99, 4427.30],
            "volume": [82.088865, 63.619882],
            "Close time": ["2017-08-17 07:59:59.999", "2017-08-17 11:59:59.999"],
            "Number of trades": [334, 200],
            "Ignore": [0, 0],
        }
    )
    out = normalize_hf_klines(raw)
    # Only "Number of trades" of the micro columns is present here, so the loader
    # retains exactly num_trades alongside OHLCV (partial-micro guard).
    assert list(out.columns) == [*_COLS, "num_trades"]
    assert int(out["num_trades"].iloc[0]) == 334
    assert out["num_trades"].dtype == "int64"
    # 2017-08-17 04:00:00 UTC == 1502942400000 ms.
    assert int(out["ts"].iloc[0]) == 1_502_942_400_000
    assert int(out["ts"].iloc[1]) == 1_502_942_400_000 + 4 * _TF_MS
    assert out["ts"].dtype.kind in ("i", "u")  # integer epoch ms
    assert out["ts"].is_monotonic_increasing
    assert float(out["close"].iloc[1]) == 4427.30


def test_normalize_hf_klines_dedups_and_sorts():
    raw = pd.DataFrame(
        {
            "Open time": [
                "2020-01-01 08:00:00",
                "2020-01-01 00:00:00",
                "2020-01-01 08:00:00",  # duplicate of row 0
            ],
            "open": [2.0, 1.0, 2.5],
            "high": [2.0, 1.0, 2.5],
            "low": [2.0, 1.0, 2.5],
            "close": [2.0, 1.0, 2.5],
            "volume": [9.0, 9.0, 9.0],
        }
    )
    out = normalize_hf_klines(raw)
    assert out["ts"].is_unique
    assert out["ts"].is_monotonic_increasing
    assert len(out) == 2
    # Dedup keeps the first occurrence of the duplicated timestamp (close 2.0).
    assert float(out["close"].iloc[1]) == 2.0


def test_normalize_hf_klines_retains_microstructure_columns():
    """A full 12-column Binance-kline frame keeps the 4 microstructure columns,
    renamed to canonical snake_case with the right dtypes (Strategy #2 inputs)."""
    raw = pd.DataFrame(
        {
            "Open time": ["2017-08-17 04:00:00", "2017-08-17 08:00:00"],
            "open": [4261.48, 4333.32],
            "high": [4349.99, 4485.39],
            "low": [4261.32, 4333.32],
            "close": [4349.99, 4427.30],
            "volume": [82.088865, 63.619882],
            "Close time": ["2017-08-17 07:59:59.999", "2017-08-17 11:59:59.999"],
            "Quote asset volume": [350000.5, 280000.25],
            "Number of trades": [334, 200],
            "Taker buy base asset volume": [40.5, 30.25],
            "Taker buy quote asset volume": [175000.5, 140000.75],
            "Ignore": [0, 0],
        }
    )
    out = normalize_hf_klines(raw)
    # OHLCV first, then the micro columns in canonical order.
    assert list(out.columns) == _COLS + _MICRO_COLS
    # Values carried through under their new names.
    assert float(out["quote_asset_volume"].iloc[0]) == 350000.5
    assert int(out["num_trades"].iloc[1]) == 200
    assert float(out["taker_buy_base"].iloc[0]) == 40.5
    assert float(out["taker_buy_quote"].iloc[1]) == 140000.75
    # Dtypes: trade count is an integer; the three volumes are float64.
    assert out["num_trades"].dtype == "int64"
    assert out["quote_asset_volume"].dtype == "float64"
    assert out["taker_buy_base"].dtype == "float64"
    assert out["taker_buy_quote"].dtype == "float64"


def test_normalize_hf_klines_without_microstructure_returns_ohlcv():
    """Older/partial files lacking the micro columns still yield a clean OHLCV frame
    (the loader must not hard-fail when a microstructure column is absent)."""
    raw = pd.DataFrame(
        {
            "Open time": ["2017-08-17 04:00:00", "2017-08-17 08:00:00"],
            "open": [4261.48, 4333.32],
            "high": [4349.99, 4485.39],
            "low": [4261.32, 4333.32],
            "close": [4349.99, 4427.30],
            "volume": [82.088865, 63.619882],
        }
    )
    out = normalize_hf_klines(raw)
    assert list(out.columns) == _COLS
    assert int(out["ts"].iloc[0]) == 1_502_942_400_000
    assert out["ts"].is_monotonic_increasing


def test_save_load_round_trip(tmp_path):
    df = pd.DataFrame([_bar(t) for t in range(0, 20 * _TF_MS, _TF_MS)], columns=_COLS)
    path = tmp_path / "BTC_USDC_1h.csv"
    save_history(df, path)
    loaded = load_history(path)
    assert list(loaded.columns) == _COLS
    pd.testing.assert_frame_equal(loaded.reset_index(drop=True), df.reset_index(drop=True))


def test_save_load_round_trip_preserves_microstructure(tmp_path):
    """Cached HF history keeps its micro columns + dtypes across CSV save/load,
    so Strategy #2 can read them straight from disk."""
    n = 5
    df = pd.DataFrame(
        {
            "ts": [i * _TF_MS for i in range(n)],
            "open": [1.0] * n,
            "high": [1.0] * n,
            "low": [1.0] * n,
            "close": [1.0] * n,
            "volume": [9.0] * n,
            "quote_asset_volume": [100.5 + i for i in range(n)],
            "num_trades": [10 + i for i in range(n)],
            "taker_buy_base": [4.0 + i for i in range(n)],
            "taker_buy_quote": [50.5 + i for i in range(n)],
        }
    )
    path = tmp_path / "BTC_USDC_1h.csv"
    save_history(df, path)
    loaded = load_history(path)
    assert list(loaded.columns) == _COLS + _MICRO_COLS
    assert loaded["num_trades"].dtype == "int64"
    assert loaded["quote_asset_volume"].dtype == "float64"
    pd.testing.assert_frame_equal(loaded, df)
