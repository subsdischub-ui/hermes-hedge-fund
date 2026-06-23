"""Common types + Exchange protocol."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
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
    ts: datetime = field(default_factory=lambda: datetime.now(UTC))


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


@dataclass
class CarryPosition:
    """A delta-neutral funding-carry PAIR: SHORT perp + LONG spot of the same asset,
    equal notional. The price deltas cancel (delta ~ 0); the position earns the perp
    funding cashflow over the hold. Strategy #3's ONLY executable form — a naked perp
    short is OOS-negative, so the carry skill routes here, never to a single-leg order.
    """
    symbol: str
    qty: float                          # base-asset size of EACH leg (short perp / long spot)
    entry_price: float                  # ~ same for both legs (basis ~ 0 in paper)
    opened_at: datetime
    accrued_funding_usd: float = 0.0    # carry collected so far (realized at close)
    entry_funding_bps: float = 0.0      # per-8h funding at entry (audit / display estimate)
    skill_id: str = "funding_carry_harvest"
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
