"""Common types + Exchange protocol."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal, Protocol
from uuid import uuid4

Side = Literal["BUY", "SELL"]
OrderType = Literal["market", "limit"]
OrderStatus = Literal["pending", "filled", "cancelled", "rejected"]


@dataclass
class Order:
    symbol: str
    side: Side
    qty: float                          # base-asset quantity
    order_type: OrderType = "market"
    price: float | None = None
    stop_price: float | None = None
    take_profit: float | None = None
    client_id: str = field(default_factory=lambda: f"hermes-{uuid4().hex[:10]}")
    venue: str = ""
    extra: dict = field(default_factory=dict)


@dataclass
class Fill:
    order_id: str
    symbol: str
    side: Side
    qty: float
    price: float
    fee_usd: float
    ts: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class Position:
    symbol: str
    qty: float                          # signed (long > 0, short < 0)
    entry_price: float
    stop_price: float | None
    take_profit: float | None
    opened_at: datetime
    unrealized_pnl_usd: float = 0.0
    skill_id: str = ""
    thesis: str = ""


class Exchange(Protocol):
    name: str

    async def equity_usd(self) -> float: ...
    async def open_positions(self) -> list[Position]: ...
    async def place_order(self, order: Order) -> Fill: ...
    async def close_position(
        self, symbol: str, reason: str, skill_id: str | None = None
    ) -> Fill | None: ...
    async def last_price(self, symbol: str) -> float: ...
    async def close(self) -> None: ...
