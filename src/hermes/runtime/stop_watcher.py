"""Server-side stop / take-profit watcher.

For paper mode (and as a safety net for live mode) we periodically check every
open position; if price has crossed its stop or TP, we close it, mark the
journal row, and fire Reflexion.
"""

from __future__ import annotations

from hermes.execution import get_exchange
from hermes.learning.reflexion import reflect_on_trade
from hermes.logging_setup import get_logger
from hermes.memory.journal import TradeJournal
from hermes.risk import get_risk_engine

log = get_logger(__name__)


async def tick() -> int:
    engine = get_risk_engine()
    exchange = get_exchange()
    journal = TradeJournal()
    closed = 0
    for pos, reason in await engine.check_stops(exchange):
        fill = await exchange.close_position(pos.symbol, reason)
        if fill is None:
            continue
        pnl = (fill.price - pos.entry_price) * pos.qty
        # Match this close to the most recent open journal entry for the symbol.
        from sqlalchemy import select  # local import to keep top tidy

        from hermes.memory.store import JournalRow, get_db

        db = get_db()
        async with db.session() as s:
            row = (
                await s.execute(
                    select(JournalRow)
                    .where(JournalRow.symbol == pos.symbol, JournalRow.closed_at.is_(None))
                    .order_by(JournalRow.id.desc())
                    .limit(1)
                )
            ).scalar_one_or_none()
        if row is None:
            continue
        await journal.close(
            trade_id=row.id,
            exit_price=fill.price,
            pnl_usd=float(pnl),
            fee_usd=float(fill.fee_usd),
            exit_reason=reason,
        )
        await reflect_on_trade(row.id)
        closed += 1
    if closed:
        log.info("stop_watcher_closed", n=closed)
    return closed
