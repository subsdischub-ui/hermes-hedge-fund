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

    async def ohlcv(
        self,
        symbol: str,
        timeframe: str = "1h",
        limit: int = 200,
        closed_only: bool = True,
    ) -> pd.DataFrame:
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
        # closed_only: CCXT returns the STILL-FORMING bar last; every strategy rule
        # here was validated on CLOSED bars, and skills read .iloc[-1]. Keeping the
        # forming bar makes the live agent trade a half-bar variant that was never
        # backtested (audited 2026-07-16: the tick fired at H+2 of the 4h cycle and
        # mean-reversion setups present at close had already un-armed intra-bar —
        # adverse selection for a reversion rule). Callers that genuinely need the
        # forming bar (none today) must ask for it explicitly.
        if closed_only and rows:
            tf_ms = int(self.client.parse_timeframe(timeframe)) * 1000
            if int(rows[-1][0]) + tf_ms > int(pd.Timestamp.now(tz="UTC").timestamp() * 1000):
                rows = rows[:-1]
        df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
        df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
        return df

    async def funding_history(self, symbol: str, limit: int = 200) -> pd.DataFrame:
        """Live perp funding-rate history from Binance USDT-M — the SAME source as
        the OOS-validated backtest (research/funding CSVs), so the live
        funding_squeeze_continuation skill sees the same series it was validated on.

        Maps "BTC/USDC" -> "BTCUSDT". Returns [ts (datetime UTC), funding_rate],
        sorted ascending. Public endpoint (no auth). On ANY failure returns an empty
        df so the funding skill simply HOLDs — never breaks the decision tick.
        """
        base = symbol.split("/")[0].upper()
        fsym = f"{base}USDT"
        try:
            import httpx

            async with httpx.AsyncClient(timeout=10.0) as cx:
                r = await cx.get(
                    "https://fapi.binance.com/fapi/v1/fundingRate",
                    params={"symbol": fsym, "limit": int(limit)},
                )
                r.raise_for_status()
                rows = r.json()
            if not rows:
                return pd.DataFrame(columns=["ts", "funding_rate"])
            df = pd.DataFrame(rows)
            df["ts"] = pd.to_datetime(df["fundingTime"].astype("int64"), unit="ms", utc=True)
            df["funding_rate"] = df["fundingRate"].astype(float)
            return df[["ts", "funding_rate"]].sort_values("ts").reset_index(drop=True)
        except Exception as e:  # network, symbol-not-found, schema drift -> graceful
            log.warning("funding_history_failed", symbol=symbol, error=str(e))
            return pd.DataFrame(columns=["ts", "funding_rate"])

    async def ticker(self, symbol: str) -> dict:
        return await self.client.fetch_ticker(symbol)

    async def order_book(self, symbol: str, depth: int = 20) -> dict:
        return await self.client.fetch_order_book(symbol, limit=depth)


@lru_cache(maxsize=4)
def get_market(venue: str | None = None) -> MarketData:
    s = get_settings()
    return MarketData(venue or s.exchange_primary)
