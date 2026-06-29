"""CoinGlass MULTI-VENUE funding + OI lake client (Stream 0 re-freeze élargi).

The scout (research_streams/S0/COVERAGE.md) PROVED the subscribed CoinGlass tier
serves DEEP per-exchange funding-rate + open-interest history beyond Binance —
OKX, Bybit and Bitget all page back to 2021 at 1d. That unblocks S2.3's
cross-venue funding/OI DISPERSION, which was previously data-gated ("the lake is
Binance-only").

This module ingests a focused cross-venue substrate: {funding-rate, open-interest}
x {Binance, OKX, Bybit, Bitget} x {BTC, ETH, SOL, XRP, DOGE} x 1d (all-time). All
venues (incl. Binance) land in ONE view per series so a research stream can read
dispersion with a single ``GROUP BY exchange`` — distinct from the original
single-venue ``coinglass_funding_rate`` / ``coinglass_open_interest`` (Binance
only, 7 pairs, 1d+6h), which stay as-is for back-compat.

  raw_root/coinglass_multi/funding_rate/<EX>_<COIN>_1d.parquet  -> coinglass_multi_funding_rate
  raw_root/coinglass_multi/open_interest/<EX>_<COIN>_1d.parquet -> coinglass_multi_open_interest

Reuses the proven ``CoinGlassClient`` (30/min throttle, envelope unwrap),
``normalize_futures`` and ``paginate_backward`` from ``coinglass.py``. The only
NEW pure logic is :func:`resolve_perp_instrument` — picking each venue's PERPETUAL
USDT instrument_id (the trap: Bybit lists dated futures alongside the perp).
"""

from __future__ import annotations

import os
import pathlib
from typing import Any

import pandas as pd

from hermes.datalake.core import Manifest, cached_pull
from hermes.datalake.sources.coinglass import (
    ENV_KEY,
    PAGE_LIMIT,
    SUPPORTED_PAIRS_PATH,
    CoinGlassClient,
    normalize_futures,
    paginate_backward,
)

SOURCE = "coinglass_multi"

# All venues (incl. Binance) land in one view per series for one-query dispersion.
EXCHANGES = ["Binance", "OKX", "Bybit", "Bitget"]
# Liquid majors listed with deep perp history on every venue above.
COINS = ["BTC", "ETH", "SOL", "XRP", "DOGE"]
INTERVAL = "1d"  # all-time depth; the cross-venue dispersion substrate

# endpoint path + the extra params each series needs (OI is requested in USD so
# venues are directly comparable).
SERIES: dict[str, str] = {
    "funding_rate": "/api/futures/funding-rate/history",
    "open_interest": "/api/futures/open-interest/history",
}
SERIES_EXTRA: dict[str, dict[str, Any]] = {
    "open_interest": {"unit": "usd"},
}


# --------------------------------------------------------------------------- #
# pure logic — resolve each venue's PERPETUAL USDT instrument_id
# --------------------------------------------------------------------------- #
def resolve_perp_instrument(
    supported_map: dict[str, list[dict[str, Any]]], exchange: str, base: str
) -> str | None:
    """Return ``exchange``'s PERPETUAL USDT-margined instrument_id for ``base``.

    Filters to base_asset==base, quote_asset=='USDT', and EXCLUDES dated futures
    (any entry carrying an ``expiry_time``) — the Bybit trap, where the perp
    ``BTCUSDT`` sits among many ``BTCUSDT-03JUL26`` dated contracts. Prefers a
    USDT settlement currency. Returns ``None`` when no perpetual exists (never a
    dated contract masquerading as the perp).
    """
    items = supported_map.get(exchange) or []
    candidates = [
        it for it in items
        if isinstance(it, dict)
        and it.get("base_asset") == base
        and it.get("quote_asset") == "USDT"
        and "expiry_time" not in it  # perpetuals only
    ]
    if not candidates:
        return None
    # prefer an explicit USDT settlement; else the first perp candidate
    for it in candidates:
        if it.get("settlement_currency") == "USDT" and it.get("instrument_id"):
            return str(it["instrument_id"])
    iid = candidates[0].get("instrument_id")
    return str(iid) if iid else None


# --------------------------------------------------------------------------- #
# live entrypoint — DEFINED, not executed in tests (live network)
# --------------------------------------------------------------------------- #
def run(raw_root: pathlib.Path, manifest: Manifest, *, run_as_of: str) -> None:
    """Pull cross-venue funding + OI (1d, all-time) for the core coins into
    ``raw_root/coinglass_multi/...`` Parquet, idempotent + fail-soft.

    Reads the CoinGlass key from os.environ at call time; a missing key is one
    blocker, not a crash. Each (exchange, coin, series) is a ``cached_pull`` unit.
    """
    raw_root = pathlib.Path(raw_root)
    root = raw_root / SOURCE

    api_key = os.environ.get(ENV_KEY)
    if not api_key:
        manifest.blocker(
            source=SOURCE, endpoint="(auth)", params={"env": ENV_KEY},
            error=f"{ENV_KEY} not set in environment",
            action=f"Malachie: export {ENV_KEY} in .env, then re-run --sources coinglass_multi.",
        )
        return

    client = CoinGlassClient(api_key=api_key)
    rate = client._rate  # share the one 30/min throttle with cached_pull
    try:
        supported_map = client._get_payload(SUPPORTED_PAIRS_PATH)
        if not isinstance(supported_map, dict):
            supported_map = {}

        for exchange in EXCHANGES:
            for coin in COINS:
                sym = resolve_perp_instrument(supported_map, exchange, coin)
                if sym is None:
                    manifest.log(
                        source=SOURCE, endpoint=SUPPORTED_PAIRS_PATH,
                        params={"exchange": exchange, "coin": coin},
                        status="empty",
                        note=f"{coin} has no USDT perp on {exchange} — skipped (research note)",
                    )
                    continue
                for series_name, path in SERIES.items():
                    _pull_series(
                        client, manifest, rate, root,
                        exchange=exchange, coin=coin, symbol=sym,
                        series_name=series_name, path=path,
                    )
    finally:
        client.close()


def _pull_series(
    client: CoinGlassClient, manifest: Manifest, rate: Any, root: pathlib.Path,
    *, exchange: str, coin: str, symbol: str, series_name: str, path: str,
) -> None:
    target = root / series_name / f"{exchange}_{coin}_{INTERVAL}.parquet"
    extra = SERIES_EXTRA.get(series_name, {})

    def _page(end_time: int | None) -> pd.DataFrame:
        data = client._get(
            path, exchange=exchange, symbol=symbol, interval=INTERVAL,
            limit=PAGE_LIMIT, end_time=end_time, **extra,
        )
        # stamp the BARE coin as symbol so cross-venue rows align on one key;
        # keep the venue's raw instrument_id for provenance.
        df = normalize_futures(data, symbol=coin, exchange=exchange, interval=INTERVAL)
        if len(df):
            df["instrument_id"] = symbol
        return df

    def fetch() -> pd.DataFrame:
        return paginate_backward(_page, ts_col="ts")

    cached_pull(
        target, fetch,
        source=SOURCE, endpoint=path,
        params={"exchange": exchange, "symbol": symbol, "coin": coin,
                "interval": INTERVAL, **extra},
        manifest=manifest, ts_col="ts", rate=rate,
    )


__all__ = ["COINS", "EXCHANGES", "resolve_perp_instrument", "run"]
