"""Portfolio-manager node — final reviewer; submits the order if approved."""

from __future__ import annotations

from datetime import datetime, timezone

from hermes.execution import get_exchange
from hermes.execution.base import Order
from hermes.graph.state import HermesState
from hermes.logging_setup import get_logger
from hermes.memory.journal import JournalEntry, TradeJournal
from hermes.settings import get_settings

log = get_logger(__name__)


async def run(state: HermesState) -> HermesState:
    risk = state.risk_decision or {}
    if risk.get("decision") != "APPROVED":
        state.executed = False
        log.info("pm_rejected", reason=risk.get("reason"))
        return state

    order_d = risk["order"]
    exchange = get_exchange()
    order = Order(
        symbol=order_d["symbol"],
        side=order_d["side"],
        qty=order_d["qty"],
        order_type=order_d.get("order_type", "market"),
        price=order_d.get("price"),
        stop_price=order_d.get("stop_price"),
        take_profit=order_d.get("take_profit"),
        extra=order_d.get("extra", {}),
    )
    fill = await exchange.place_order(order)

    journal = TradeJournal()
    s = get_settings()
    trade_id = await journal.open(
        JournalEntry(
            symbol=fill.symbol,
            side=fill.side,
            qty=fill.qty,
            entry_price=fill.price,
            opened_at=fill.ts or datetime.now(timezone.utc),
            confidence=float(state.trader_proposal.get("confidence", 0.0)),
            skill_id=order.extra.get("skill_id", ""),
            thesis=order.extra.get("thesis", ""),
            stage=s.stage,
            market_state=state.snapshot,
        )
    )
    state.executed = True
    state.trade_id = trade_id
    log.info("pm_executed", trade_id=trade_id, symbol=fill.symbol, side=fill.side, qty=fill.qty)
    return state
