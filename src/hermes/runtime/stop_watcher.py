"""Server-side stop / take-profit watcher.

For paper mode (and as a safety net for live mode) we periodically check every
open position; if price has crossed its stop or TP, we close it, mark the
journal row, and fire Reflexion.

Matching close events to journal rows
-------------------------------------
The PaperExchange may collapse multiple opens on the same symbol into one
position, but the journal records each open as its own row. When we close
a position we attribute the realized PnL to the OLDEST open journal row for
that symbol (FIFO). If multiple are open, we close them in order on
subsequent ticks; orphans are explicitly logged.
"""

from __future__ import annotations

from sqlalchemy import select

from hermes.execution import get_exchange
from hermes.learning.reflexion import reflect_on_trade
from hermes.logging_setup import get_logger
from hermes.memory.journal import TradeJournal
from hermes.memory.store import JournalRow, get_db
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
        # PnL nets the close-side fee. Open-side fee was already paid into cash
        # at place_order time, so we don't double-count it here.
        pnl = (fill.price - pos.entry_price) * pos.qty - float(fill.fee_usd)

        db = get_db()
        async with db.session() as s:
            row = (
                await s.execute(
                    select(JournalRow)
                    .where(JournalRow.symbol == pos.symbol, JournalRow.closed_at.is_(None))
                    .order_by(JournalRow.id.asc())   # FIFO
                    .limit(1)
                )
            ).scalar_one_or_none()
        if row is None:
            log.warning("stop_watcher_orphan_close", symbol=pos.symbol, reason=reason)
            continue
        await journal.close(
            trade_id=row.id,
            exit_price=fill.price,
            pnl_usd=float(pnl),
            fee_usd=float(fill.fee_usd),
            exit_reason=reason,
        )
        try:
            await reflect_on_trade(row.id)
        except Exception as e:
            log.warning("reflect_failed", trade_id=row.id, error=str(e))
        closed += 1
    if closed:
        log.info("stop_watcher_closed", n=closed)
    return closed
