"""Hyperliquid spot/perp adapter (production path).

Wraps `hyperliquid-python-sdk`. The SDK is synchronous, so every call is
dispatched via `asyncio.to_thread` to avoid blocking the event loop. Falls
back to a NotImplementedError when no credentials are configured — the
agent will refuse to leave paper mode in that case.

Response handling
-----------------
HL responses look like:
    {'status': 'ok', 'response': {'data': {'statuses': [{'filled': {...}}]}}}
or
    {'status': 'err', 'response': '...message...'}
We treat anything other than a fully-realized `filled` block as an error and
raise; callers must not interpret zero-fill as "executed".
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any

from hermes.execution.base import Exchange, Fill, Order, Position
from hermes.logging_setup import get_logger
from hermes.settings import get_settings

log = get_logger(__name__)


class HyperliquidError(RuntimeError):
    pass


class HyperliquidExchange(Exchange):
    name = "hyperliquid"
    DEFAULT_SLIPPAGE = 0.005   # 0.5% — tight enough to avoid bad fills on majors

    def __init__(self) -> None:
        s = get_settings()
        secret = s.hl_secret_key.get_secret_value() if hasattr(s.hl_secret_key, "get_secret_value") else str(s.hl_secret_key)
        if not s.hl_account_address or not secret:
            raise RuntimeError("Hyperliquid credentials missing — refusing live mode.")
        from hyperliquid.info import Info  # type: ignore
        from hyperliquid.exchange import Exchange as HLExchange  # type: ignore
        from eth_account import Account  # type: ignore

        base_url = "https://api.hyperliquid-testnet.xyz" if s.hl_testnet else "https://api.hyperliquid.xyz"
        wallet = Account.from_key(secret)
        self.info = Info(base_url, skip_ws=True)
        self.exchange = HLExchange(wallet, base_url, account_address=s.hl_account_address)
        self.address = s.hl_account_address

    # --- sync->async wrapper ------------------------------------------------
    async def _to_thread(self, fn, *args, **kwargs):
        return await asyncio.to_thread(fn, *args, **kwargs)

    # --- read paths ---------------------------------------------------------
    async def equity_usd(self) -> float:
        state = await self._to_thread(self.info.user_state, self.address)
        return float(state.get("marginSummary", {}).get("accountValue", 0.0))

    async def open_positions(self) -> list[Position]:
        state = await self._to_thread(self.info.user_state, self.address)
        out: list[Position] = []
        for asset in state.get("assetPositions", []):
            p = asset["position"]
            qty = float(p["szi"])
            if abs(qty) < 1e-12:
                continue
            out.append(
                Position(
                    symbol=p["coin"] + "/USDC",
                    qty=qty,
                    entry_price=float(p["entryPx"]),
                    stop_price=None,
                    take_profit=None,
                    opened_at=datetime.now(timezone.utc),
                    unrealized_pnl_usd=float(p.get("unrealizedPnl", 0.0)),
                )
            )
        return out

    async def last_price(self, symbol: str) -> float:
        coin = symbol.split("/")[0]
        mids = await self._to_thread(self.info.all_mids)
        if coin not in mids:
            raise HyperliquidError(f"no mid for {coin}")
        return float(mids[coin])

    # --- write paths --------------------------------------------------------
    @staticmethod
    def _extract_filled(result: dict[str, Any]) -> dict[str, Any]:
        if result.get("status") != "ok":
            raise HyperliquidError(f"hl error: {result!r}")
        statuses = (
            result.get("response", {}).get("data", {}).get("statuses", [])
        )
        if not statuses:
            raise HyperliquidError(f"hl no statuses: {result!r}")
        st = statuses[0]
        if "error" in st:
            raise HyperliquidError(f"hl rejected: {st['error']}")
        filled = st.get("filled")
        if not filled or float(filled.get("totalSz", 0)) <= 0:
            raise HyperliquidError(f"hl zero fill: {st!r}")
        return filled

    async def place_order(self, order: Order) -> Fill:
        coin = order.symbol.split("/")[0]
        is_buy = order.side == "BUY"
        result = await self._to_thread(
            self.exchange.market_open,
            coin, is_buy, order.qty, None, self.DEFAULT_SLIPPAGE,
        )
        # Redact the SDK echo down to the bits we need before logging.
        log.info("hl_order_status", status=result.get("status"), symbol=order.symbol, side=order.side)
        filled = self._extract_filled(result)
        return Fill(
            order_id=str(filled.get("oid", order.client_id)),
            symbol=order.symbol,
            side=order.side,
            qty=float(filled["totalSz"]),
            price=float(filled.get("avgPx", 0.0)),
            fee_usd=float(filled.get("fee", 0.0)),
        )

    async def close_position(self, symbol: str, reason: str) -> Fill | None:
        coin = symbol.split("/")[0]
        result = await self._to_thread(self.exchange.market_close, coin)
        log.info("hl_close_status", status=result.get("status"), symbol=symbol, reason=reason)
        try:
            filled = self._extract_filled(result)
        except HyperliquidError as e:
            log.warning("hl_close_failed", symbol=symbol, error=str(e))
            return None
        return Fill(
            order_id=str(filled.get("oid", "")),
            symbol=symbol,
            side="SELL",
            qty=float(filled["totalSz"]),
            price=float(filled.get("avgPx", 0.0)),
            fee_usd=float(filled.get("fee", 0.0)),
        )

    async def close(self) -> None:
        return None
