"""Bybit backup adapter via CCXT.

Position tracking
-----------------
Bybit spot has no native "position" concept — only balances. We REFUSE to
synthesize positions from raw balances because that would treat the user's
pre-existing or unrelated coins as Hermès-owned trades. Instead, this adapter
relies on the journal (`hermes.memory.journal`) for ownership, and only
returns positions whose symbol matches an open journal row.
"""

from __future__ import annotations

import ccxt.async_support as ccxt
from sqlalchemy import select

from hermes.execution.base import Exchange, Fill, Order, Position
from hermes.settings import get_settings


class BybitExchange(Exchange):
    name = "bybit"

    def __init__(self) -> None:
        s = get_settings()
        key = s.bybit_api_key.get_secret_value() if hasattr(s.bybit_api_key, "get_secret_value") else str(s.bybit_api_key)
        secret = s.bybit_api_secret.get_secret_value() if hasattr(s.bybit_api_secret, "get_secret_value") else str(s.bybit_api_secret)
        if not key:
            raise RuntimeError("Bybit credentials missing")
        self.client = ccxt.bybit(
            {
                "apiKey": key,
                "secret": secret,
                "enableRateLimit": True,
                "options": {"defaultType": "spot"},
            }
        )
        if s.bybit_testnet:
            self.client.set_sandbox_mode(True)

    async def equity_usd(self) -> float:
        bal = await self.client.fetch_balance()
        return float(bal["total"].get("USDT", 0.0) + bal["total"].get("USDC", 0.0))

    async def open_positions(self) -> list[Position]:
        """Read OPEN journal rows for ownership, validate quantity matches balance.

        Returns the journal rows that still have non-zero base-asset balance
        on the exchange. Pre-existing balances NOT in the journal are ignored.
        """
        from datetime import datetime, timezone

        from hermes.memory.store import JournalRow, get_db

        bal = await self.client.fetch_balance()
        totals = bal.get("total") or {}
        out: list[Position] = []
        db = get_db()
        async with db.session() as s:
            rows = (
                await s.execute(
                    select(JournalRow).where(JournalRow.closed_at.is_(None))
                )
            ).scalars().all()
        for row in rows:
            base = row.symbol.split("/")[0]
            on_book = float(totals.get(base, 0.0))
            if on_book <= 0:
                continue
            try:
                t = await self.client.fetch_ticker(row.symbol)
                px = float(t["last"])
            except Exception:
                px = float(row.entry_price)
            out.append(
                Position(
                    symbol=row.symbol,
                    qty=min(on_book, float(row.qty)),
                    entry_price=float(row.entry_price),
                    stop_price=None,
                    take_profit=None,
                    opened_at=row.opened_at or datetime.now(timezone.utc),
                    unrealized_pnl_usd=(px - float(row.entry_price)) * float(row.qty),
                    skill_id=row.skill_id or "",
                    thesis=row.thesis or "",
                )
            )
        return out

    async def last_price(self, symbol: str) -> float:
        t = await self.client.fetch_ticker(symbol)
        return float(t["last"])

    async def place_order(self, order: Order) -> Fill:
        if order.side == "SELL":
            # Bybit spot cannot short. Only allow SELL when we already own the base.
            bal = await self.client.fetch_balance()
            base = order.symbol.split("/")[0]
            if float(bal["total"].get(base, 0.0)) < order.qty:
                raise RuntimeError(
                    f"bybit spot refuses SELL {order.qty} {base}: insufficient holdings"
                )
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
        # Look up the open journal row to find the qty we actually own.
        from hermes.memory.store import JournalRow, get_db

        db = get_db()
        async with db.session() as s:
            row = (
                await s.execute(
                    select(JournalRow)
                    .where(JournalRow.symbol == symbol, JournalRow.closed_at.is_(None))
                    .order_by(JournalRow.id.asc())
                    .limit(1)
                )
            ).scalar_one_or_none()
        if row is None:
            return None
        bal = await self.client.fetch_balance()
        base = symbol.split("/")[0]
        qty = min(float(row.qty), float(bal["total"].get(base, 0.0)))
        if qty <= 0:
            return None
        order = Order(symbol=symbol, side="SELL", qty=qty)
        return await self.place_order(order)

    async def close(self) -> None:
        await self.client.close()
