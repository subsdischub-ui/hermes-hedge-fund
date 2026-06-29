"""TDD — Binance SPOT 4h klines lake source (Stream 0-alpha, cash-and-carry spot leg).

The spot leg is the ONLY piece missing for a delta-neutral cash-and-carry P&L: the
perp leg + causal settled funding already live in ``binance_funding_on_4h``. This
source must mirror the 23-perp universe and the EXACT lake schema/symbol-format of
``binance_klines_4h`` so the carry join is ``spot.symbol == perp.symbol`` (both the
slash ``BTC/USDT`` spelling) — a cross-format join silently yields zero rows.

All real logic is PURE (``fetch_spot_ohlcv_paginated`` over an INJECTED client) and
tested offline; ``run()`` / ``_make_spot_ccxt_client`` do live network and are never
executed here.
"""

from __future__ import annotations

from hermes.datalake.sources import binance_spot
from hermes.datalake.sources.binance_extended import ALL_PAIRS
from hermes.research.s10_provenance import validate_symbol_join


class _FakeSpotClient:
    """Mimic a ccxt SPOT client: ``fetch_ohlcv`` returns bars at ts >= ``since``,
    capped at ``limit`` — exactly the forward-window contract the paginator drives."""

    def __init__(self, bars: list[list[float]]):
        self._bars = sorted(bars, key=lambda r: r[0])
        self.calls: list[tuple] = []

    def fetch_ohlcv(self, symbol, timeframe, since=0, limit=1000):
        self.calls.append((symbol, timeframe, since, limit))
        window = [r for r in self._bars if r[0] >= int(since)]
        return [list(r) for r in window[: int(limit)]]


def _bars(start_ms: int, n: int, step_ms: int) -> list[list[float]]:
    # [ts, open, high, low, close, volume]
    return [[start_ms + i * step_ms, 1.0, 2.0, 0.5, 1.5, 10.0] for i in range(n)]


# --------------------------------------------------------------------------- #
# universe + interval constants
# --------------------------------------------------------------------------- #
def test_spot_universe_mirrors_the_23_perps_exactly():
    """The spot universe MUST equal the perp universe (every perp needs its spot
    leg), in the same order, with no extras/dupes — else the carry book is lopsided."""
    assert binance_spot.PAIRS == ALL_PAIRS
    assert len(binance_spot.PAIRS) == 23
    assert len(set(binance_spot.PAIRS)) == 23


def test_four_hour_interval_constant_matches_perp_grid():
    assert binance_spot.FOUR_H_MS == 14_400_000  # 4 * 3600 * 1000


def test_source_and_exchange_tags_mark_the_spot_venue():
    # view name derives from SOURCE dir -> 'binance_spot' + 'klines_4h'
    assert binance_spot.SOURCE == "binance_spot"
    # the spot venue is plain 'binance' (NOT 'binanceusdm', which is the perp venue)
    assert binance_spot.SPOT_EXCHANGE == "binance"


# --------------------------------------------------------------------------- #
# pure paginator — forward walk, spot venue tag, lake schema
# --------------------------------------------------------------------------- #
def test_paginator_walks_forward_and_tags_spot_schema():
    client = _FakeSpotClient(_bars(1_600_000_000_000, n=5, step_ms=binance_spot.FOUR_H_MS))
    out = binance_spot.fetch_spot_ohlcv_paginated(
        client, "BTC/USDT", "4h",
        since_ms=0, interval_ms=binance_spot.FOUR_H_MS, limit=2, max_batches=10,
    )
    # exact lake OHLCV schema + tag columns
    assert list(out.columns) == [
        "ts", "open", "high", "low", "close", "volume", "symbol", "exchange", "interval",
    ]
    assert out["ts"].dtype == "int64"
    for c in ("open", "high", "low", "close", "volume"):
        assert out[c].dtype == "float64"
    # all 5 bars recovered despite limit=2 (pagination worked), sorted, unique ts
    assert len(out) == 5
    assert out["ts"].is_monotonic_increasing
    assert out["ts"].is_unique
    # spot venue tag + slash symbol + 4h interval
    assert (out["exchange"] == "binance").all()
    assert (out["symbol"] == "BTC/USDT").all()
    assert (out["interval"] == "4h").all()


def test_paginator_dedups_boundary_bar_and_stops_on_empty():
    # limit=3 over 4 bars -> page1 returns 3 (last is the boundary), page2 returns the
    # 4th (+re-returns boundary at its since); dedup-on-ts must collapse the overlap.
    client = _FakeSpotClient(_bars(1_700_000_000_000, n=4, step_ms=binance_spot.FOUR_H_MS))
    out = binance_spot.fetch_spot_ohlcv_paginated(
        client, "ETH/USDT", "4h",
        since_ms=0, interval_ms=binance_spot.FOUR_H_MS, limit=3, max_batches=10,
    )
    assert len(out) == 4
    assert out["ts"].is_unique


def test_paginator_empty_client_yields_canonical_empty_frame():
    client = _FakeSpotClient([])
    out = binance_spot.fetch_spot_ohlcv_paginated(
        client, "WIF/USDT", "4h",
        since_ms=0, interval_ms=binance_spot.FOUR_H_MS, limit=1000, max_batches=10,
    )
    assert len(out) == 0
    assert list(out.columns)[:6] == ["ts", "open", "high", "low", "close", "volume"]
    assert out["ts"].dtype == "int64"


# --------------------------------------------------------------------------- #
# the join-critical invariant: spot joins perp with NO format mismatch
# --------------------------------------------------------------------------- #
def test_spot_symbols_are_join_compatible_with_perp_klines():
    """Spot symbols must use the SAME slash spelling as binance_klines_4h so a join
    on ``symbol`` is lossless (P2 of the S10 provenance guardrails)."""
    spot_syms = [f"{base}/USDT" for base in binance_spot.PAIRS]
    perp_syms = [f"{base}/USDT" for base in ALL_PAIRS]
    res = validate_symbol_join(spot_syms, perp_syms)
    assert res["compatible"] is True
    assert res["left_format"] == "slash"
    assert res["n_overlap"] == 23
