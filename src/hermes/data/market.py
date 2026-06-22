"""Market data via CCXT (async). One adapter per venue."""

from __future__ import annotations

from functools import lru_cache

import ccxt.async_support as ccxt
import pandas as pd

from hermes.logging_setup import get_logger
from hermes.settings import get_settings

log = get_logger(__name__)


class MarketData:
    def __init__(self, venue: str):
        self.venue = venue
        ex_cls = getattr(ccxt, venue)
        self.client = ex_cls({"enableRateLimit": True})

    async def close(self) -> None:
        await self.client.close()

    async def ohlcv(self, symbol: str, timeframe: str = "1h", limit: int = 200) -> pd.DataFrame:
        # NOTE: CCXT's fetch_ohlcv normalises every venue to the 6 OHLCV fields
        # only. The per-bar microstructure that Strategy #2 needs — quote_asset_volume,
        # num_trades, taker_buy_base, taker_buy_quote (canonical names defined in
        # hermes.data.history._MICRO_SOURCE / _MICRO_COLS) — is NOT available here.
        # LIVE source for those columns is Binance's raw klines REST endpoint, which
        # returns the full 12-field kline array matching the linxy/CryptoCoin HF
        # dataset 1:1: spot  GET /api/v3/klines  (ccxt: self.client.publicGetKlines),
        # USDT-M futures  GET /fapi/v1/klines  (ccxt: self.client.fapiPublicGetKlines).
        # When wiring microstructure into the live path, map those raw columns to the
        # SAME canonical snake_case names as the backtest (history._MICRO_SOURCE) so
        # the HF backtest and live feeds stay column-compatible.
        rows = await self.client.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
        df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
        return df

    async def ticker(self, symbol: str) -> dict:
        return await self.client.fetch_ticker(symbol)

    async def order_book(self, symbol: str, depth: int = 20) -> dict:
        return await self.client.fetch_order_book(symbol, limit=depth)


@lru_cache(maxsize=4)
def get_market(venue: str | None = None) -> MarketData:
    s = get_settings()
    return MarketData(venue or s.exchange_primary)
