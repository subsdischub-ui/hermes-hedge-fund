"""TDD for the Binance USDT-M illiquid-tail source (Stream 0-β, Deliverable A).

Pulls spot+perp klines + deep native funding for perps BEYOND the ingested 23, selected
by Malachie's policy: hard >=1y depth filter, then liquidity-decile stratification that
over-samples the illiquid bottom. All real logic is PURE (discovery/selection over
INJECTED API payloads) and tested offline; run() / live ccxt+httpx transports are never
executed here. Symbols join via slash spelling (BTC/USDT) — the carry-join contract.
"""
from __future__ import annotations

from hermes.datalake.sources import binance_tail as bt

YR_MS = 365 * 86_400_000
NOW = 1_751_241_600_000  # 2026-06-30


def _xinfo():
    return {
        "symbols": [
            # already-ingested major -> excluded
            {"symbol": "BTCUSDT", "baseAsset": "BTC", "quoteAsset": "USDT",
             "contractType": "PERPETUAL", "status": "TRADING", "onboardDate": NOW - 5 * YR_MS},
            # deep tail perps -> kept
            {"symbol": "ZECUSDT", "baseAsset": "ZEC", "quoteAsset": "USDT",
             "contractType": "PERPETUAL", "status": "TRADING", "onboardDate": NOW - 4 * YR_MS},
            {"symbol": "XLMUSDT", "baseAsset": "XLM", "quoteAsset": "USDT",
             "contractType": "PERPETUAL", "status": "TRADING", "onboardDate": NOW - 6 * YR_MS},
            # too-young (< 1y) -> excluded by depth floor
            {"symbol": "NEWUSDT", "baseAsset": "NEW", "quoteAsset": "USDT",
             "contractType": "PERPETUAL", "status": "TRADING", "onboardDate": NOW - 100 * 86_400_000},
            # non-USDT quote -> excluded
            {"symbol": "ETHBUSD", "baseAsset": "ETH", "quoteAsset": "BUSD",
             "contractType": "PERPETUAL", "status": "TRADING", "onboardDate": NOW - 5 * YR_MS},
            # not trading -> excluded
            {"symbol": "OLDUSDT", "baseAsset": "OLD", "quoteAsset": "USDT",
             "contractType": "PERPETUAL", "status": "BREAK", "onboardDate": NOW - 5 * YR_MS},
            # delivery (quarterly), not perpetual -> excluded
            {"symbol": "DELUSDT", "baseAsset": "DEL", "quoteAsset": "USDT",
             "contractType": "CURRENT_QUARTER", "status": "TRADING", "onboardDate": NOW - 5 * YR_MS},
        ]
    }


def test_discover_keeps_only_usdt_perpetual_trading_tail_with_depth():
    out = bt.discover_tail_perps(
        _xinfo(), ingested={"BTC", "ETH"}, now_ms=NOW, min_history_ms=YR_MS
    )
    bases = {c["base"] for c in out}
    assert bases == {"ZEC", "XLM"}


def test_discover_excludes_the_ingested_universe():
    out = bt.discover_tail_perps(
        _xinfo(), ingested={"BTC", "ZEC"}, now_ms=NOW, min_history_ms=YR_MS
    )
    assert "ZEC" not in {c["base"] for c in out}
    assert "XLM" in {c["base"] for c in out}


def test_discover_returns_slash_lake_symbol_and_api_symbol():
    out = bt.discover_tail_perps(
        _xinfo(), ingested=set(), now_ms=NOW, min_history_ms=YR_MS
    )
    zec = next(c for c in out if c["base"] == "ZEC")
    assert zec["symbol"] == "ZEC/USDT"     # lake/join spelling (slash)
    assert zec["api_symbol"] == "ZECUSDT"  # Binance API spelling (concat)


def test_liquidity_map_from_ticker():
    ticker = [
        {"symbol": "ZECUSDT", "quoteVolume": "1234.5"},
        {"symbol": "XLMUSDT", "quoteVolume": "0"},
    ]
    m = bt.liquidity_map_from_ticker(ticker)
    assert m["ZECUSDT"] == 1234.5
    assert m["XLMUSDT"] == 0.0


def test_select_tail_universe_drops_zero_or_missing_liquidity():
    candidates = [
        {"base": "ZEC", "symbol": "ZEC/USDT", "api_symbol": "ZECUSDT", "onboard_ms": NOW - 4 * YR_MS},
        {"base": "XLM", "symbol": "XLM/USDT", "api_symbol": "XLMUSDT", "onboard_ms": NOW - 6 * YR_MS},
        {"base": "DEAD", "symbol": "DEAD/USDT", "api_symbol": "DEADUSDT", "onboard_ms": NOW - 3 * YR_MS},
    ]
    liq = {"ZECUSDT": 5_000_000.0, "XLMUSDT": 0.0}  # DEAD missing, XLM zero
    out = bt.select_tail_universe(candidates, liq, n_target=10)
    bases = {c["base"] for c in out}
    assert bases == {"ZEC"}  # only the positive-liquidity market survives


def test_select_tail_universe_oversamples_illiquid_and_annotates_decile():
    # 30 synthetic tail candidates spanning a liquidity gradient.
    candidates = [
        {"base": f"C{i:02d}", "symbol": f"C{i:02d}/USDT", "api_symbol": f"C{i:02d}USDT",
         "onboard_ms": NOW - 3 * YR_MS}
        for i in range(30)
    ]
    liq = {f"C{i:02d}USDT": float(i + 1) for i in range(30)}
    out = bt.select_tail_universe(candidates, liq, n_target=10)
    assert len(out) == 10
    assert all("decile" in c for c in out)
    # the illiquid floor must be represented
    assert min(c["decile"] for c in out) == 0


def test_selection_frame_records_provenance_and_spot_availability():
    selected = [
        {"base": "ZEC", "symbol": "ZEC/USDT", "api_symbol": "ZECUSDT",
         "onboard_ms": NOW - 4 * YR_MS, "quote_volume": 5e6, "decile": 7},
        {"base": "FOO", "symbol": "FOO/USDT", "api_symbol": "FOOUSDT",
         "onboard_ms": NOW - 2 * YR_MS, "quote_volume": 1e5, "decile": 1},
    ]
    frame = bt.selection_frame(selected, spot_bases={"ZEC"}, run_as_of="2026-06-30")
    assert list(frame["base"]) == ["ZEC", "FOO"]
    assert dict(zip(frame["base"], frame["has_spot"])) == {"ZEC": True, "FOO": False}
    assert (frame["source"] == bt.SOURCE).all()
    assert (frame["run_as_of"] == "2026-06-30").all()
