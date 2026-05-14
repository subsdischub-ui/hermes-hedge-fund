"""Paper-trading exchange. Persists state in /data so restarts don't reset equity."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from hermes.data.market import get_market
from hermes.execution.base import Exchange, Fill, Order, Position
from hermes.logging_setup import get_logger

log = get_logger(__name__)


class PaperExchange(Exchange):
    name = "paper"
    FEE_BPS = 5  # 0.05% taker — matches Hyperliquid

    def __init__(self, state_path: Path, starting_equity: float, venue_for_prices: str = "binance"):
        self.state_path = state_path
        self.market = get_market(venue_for_prices)
        if state_path.exists():
            self._state = json.loads(state_path.read_text())
        else:
            self._state = {
                "cash_usd": starting_equity,
                "positions": {},   # symbol -> dict
                "history": [],
            }
        self._save()

    def _save(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps(self._state, default=str))

    async def equity_usd(self) -> float:
        total = self._state["cash_usd"]
        for sym, p in self._state["positions"].items():
            try:
                px = await self.last_price(sym)
                total += p["qty"] * px
            except Exception:
                total += p["qty"] * p["entry_price"]
        return total

    async def open_positions(self) -> list[Position]:
        out: list[Position] = []
        for sym, p in self._state["positions"].items():
            try:
                last = await self.last_price(sym)
            except Exception:
                last = p["entry_price"]
            pos = Position(
                symbol=sym,
                qty=p["qty"],
                entry_price=p["entry_price"],
                stop_price=p.get("stop_price"),
                take_profit=p.get("take_profit"),
                opened_at=datetime.fromisoformat(p["opened_at"]) if isinstance(p["opened_at"], str) else p["opened_at"],
                unrealized_pnl_usd=(last - p["entry_price"]) * p["qty"],
                skill_id=p.get("skill_id", ""),
                thesis=p.get("thesis", ""),
            )
            out.append(pos)
        return out

    async def last_price(self, symbol: str) -> float:
        t = await self.market.ticker(symbol)
        return float(t["last"] or t.get("close") or 0.0)

    async def place_order(self, order: Order) -> Fill:
        px = order.price or await self.last_price(order.symbol)
        notional = px * order.qty
        fee = notional * (self.FEE_BPS / 10_000)
        signed_qty = order.qty if order.side == "BUY" else -order.qty

        positions = self._state["positions"]
        pos = positions.get(order.symbol)
        if pos is None:
            positions[order.symbol] = {
                "qty": signed_qty,
                "entry_price": px,
                "stop_price": order.stop_price,
                "take_profit": order.take_profit,
                "opened_at": datetime.now(timezone.utc).isoformat(),
                "skill_id": order.extra.get("skill_id", ""),
                "thesis": order.extra.get("thesis", ""),
            }
            self._state["cash_usd"] -= notional + fee
        else:
            # close or flip
            new_qty = pos["qty"] + signed_qty
            realized = (px - pos["entry_price"]) * min(abs(signed_qty), abs(pos["qty"])) * (
                1 if pos["qty"] > 0 else -1
            )
            self._state["cash_usd"] += realized - fee
            self._state["history"].append(
                {
                    "symbol": order.symbol,
                    "closed_at": datetime.now(timezone.utc).isoformat(),
                    "qty_closed": min(abs(signed_qty), abs(pos["qty"])),
                    "pnl_usd": realized - fee,
                    "skill_id": pos.get("skill_id", ""),
                    "thesis": pos.get("thesis", ""),
                }
            )
            if abs(new_qty) < 1e-12:
                positions.pop(order.symbol, None)
            else:
                pos["qty"] = new_qty

        fill = Fill(
            order_id=order.client_id or f"paper-{uuid4().hex[:8]}",
            symbol=order.symbol,
            side=order.side,
            qty=order.qty,
            price=px,
            fee_usd=fee,
        )
        self._save()
        log.info("paper_fill", symbol=order.symbol, side=order.side, qty=order.qty, price=px, fee=fee)
        return fill

    async def close_position(self, symbol: str, reason: str) -> Fill | None:
        pos = self._state["positions"].get(symbol)
        if not pos:
            return None
        side = "SELL" if pos["qty"] > 0 else "BUY"
        order = Order(symbol=symbol, side=side, qty=abs(pos["qty"]), extra={"reason": reason})
        return await self.place_order(order)

    async def history(self) -> list[dict]:
        return list(self._state["history"])

    async def close(self) -> None:
        await self.market.close()
