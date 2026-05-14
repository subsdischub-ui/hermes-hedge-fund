"""Trade journal — append entries, derive aggregate stats."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select

from hermes.memory.store import JournalRow, get_db


@dataclass
class JournalEntry:
    symbol: str
    side: str
    qty: float
    entry_price: float
    opened_at: datetime
    confidence: float = 0.0
    skill_id: str = ""
    thesis: str = ""
    stage: str = ""
    market_state: dict[str, Any] = field(default_factory=dict)
    closed_at: datetime | None = None
    exit_price: float | None = None
    pnl_usd: float | None = None
    fee_usd: float = 0.0
    exit_reason: str = ""
    reflection: str = ""


class TradeJournal:
    async def open(self, entry: JournalEntry) -> int:
        db = get_db()
        async with db.session() as s:
            row = JournalRow(
                symbol=entry.symbol,
                side=entry.side,
                qty=entry.qty,
                entry_price=entry.entry_price,
                opened_at=entry.opened_at,
                confidence=entry.confidence,
                skill_id=entry.skill_id,
                thesis=entry.thesis,
                stage=entry.stage,
                market_state=entry.market_state,
            )
            s.add(row)
            await s.commit()
            await s.refresh(row)
            return row.id

    async def close(
        self,
        trade_id: int,
        exit_price: float,
        pnl_usd: float,
        fee_usd: float,
        exit_reason: str,
        reflection: str = "",
    ) -> None:
        db = get_db()
        async with db.session() as s:
            row = (await s.execute(select(JournalRow).where(JournalRow.id == trade_id))).scalar_one()
            row.exit_price = exit_price
            row.pnl_usd = pnl_usd
            row.fee_usd = fee_usd
            row.exit_reason = exit_reason
            row.reflection = reflection
            row.closed_at = datetime.now(timezone.utc)
            await s.commit()

    async def stats(self) -> dict[str, Any]:
        db = get_db()
        async with db.session() as s:
            rows = (
                await s.execute(select(JournalRow).where(JournalRow.closed_at.is_not(None)))
            ).scalars().all()
        if not rows:
            return {"count": 0, "win_rate": None, "avg_win_loss_ratio": None, "pnl_usd": 0.0}
        wins = [r for r in rows if (r.pnl_usd or 0) > 0]
        losses = [r for r in rows if (r.pnl_usd or 0) < 0]
        win_rate = len(wins) / len(rows)
        avg_win = sum(r.pnl_usd for r in wins) / len(wins) if wins else 0.0
        avg_loss = abs(sum(r.pnl_usd for r in losses) / len(losses)) if losses else 0.0
        ratio = (avg_win / avg_loss) if avg_loss > 0 else None
        return {
            "count": len(rows),
            "win_rate": win_rate,
            "avg_win_loss_ratio": ratio,
            "pnl_usd": sum(r.pnl_usd or 0 for r in rows),
        }

    async def recent_closed(self, since: datetime) -> list[JournalRow]:
        db = get_db()
        async with db.session() as s:
            return (
                await s.execute(
                    select(JournalRow)
                    .where(JournalRow.closed_at.is_not(None), JournalRow.closed_at >= since)
                    .order_by(JournalRow.closed_at.desc())
                )
            ).scalars().all()
