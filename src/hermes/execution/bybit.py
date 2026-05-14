"""Bybit backup adapter via CCXT."""

from __future__ import annotations

from datetime import datetime, timezone

import ccxt.async_support as ccxt

from hermes.execution.base import Exchange, Fill, Order, Position
from hermes.settings import get_settings


class BybitExchange(Exchange):
    name = "bybit"

    def __init__(self) -> None:
        s = get_settings()
        if not s.bybit_api_key:
            raise RuntimeError("Bybit credentials missing")
        self.client = ccxt.bybit(
            {
                "apiKey": s.bybit_api_key,
                "secret": s.bybit_api_secret,
                "enableRateLimit": True,
                "options": {"defaultType": "spot"},
            }
        )
        if s.bybit_testnet:
            self.client.set_sandbox_mode(True)

    async def equity_usd(self) -> float:
        bal = await self.client.fetch_balance()
        return float(bal["total"].get("USDT", 0.0))

    async def open_positions(self) -> list[Position]:
        bal = await self.client.fetch_balance()
        out: list[Position] = []
        for sym, qty in bal["total"].items():
            if sym in ("USDT", "USDC") or not qty:
                continue
            try:
                t = await self.client.fetch_ticker(f"{sym}/USDT")
                px = float(t["last"])
            except Exception:
                continue
            out.append(
                Position(
                    symbol=f"{sym}/USDT",
                    qty=float(qty),
                    entry_price=px,  # unknown — placeholder
                    stop_price=None,
                    take_profit=None,
                    opened_at=datetime.now(timezone.utc),
                )
            )
        return out

    async def last_price(self, symbol: str) -> float:
        t = await self.client.fetch_ticker(symbol)
        return float(t["last"])

    async def place_order(self, order: Order) -> Fill:
        r = await self.client.create_order(
            symbol=order.symbol,
            type=order.order_type,
            side=order.side.lower(),
            amount=order.qty,
            price=order.price,
        )
        return Fill(
            order_id=str(r["id"]),
            symbol=order.symbol,
            side=order.side,
            qty=float(r.get("filled", order.qty)),
            price=float(r.get("average") or r.get("price") or 0.0),
            fee_usd=float((r.get("fee") or {}).get("cost", 0.0)),
        )

    async def close_position(self, symbol: str, reason: str) -> Fill | None:
        bal = await self.client.fetch_balance()
        base = symbol.split("/")[0]
        qty = float(bal["total"].get(base, 0.0))
        if qty <= 0:
            return None
        order = Order(symbol=symbol, side="SELL", qty=qty)
        return await self.place_order(order)

    async def close(self) -> None:
        await self.client.close()
