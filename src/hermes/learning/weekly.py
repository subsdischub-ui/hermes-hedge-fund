"""Weekly: re-rank skills by realized 30-day Sharpe; archive underperformers."""

from __future__ import annotations

import math
from collections import defaultdict
from datetime import datetime, timedelta, timezone

from sqlalchemy import select

from hermes.logging_setup import get_logger
from hermes.memory.store import JournalRow, SkillStatRow, get_db
from hermes.skills import get_skill_library

log = get_logger(__name__)


async def run_weekly_skill_rerank(lookback_days: int = 30, min_trades: int = 5) -> dict[str, dict]:
    since = datetime.now(timezone.utc) - timedelta(days=lookback_days)
    db = get_db()
    async with db.session() as s:
        rows = (
            await s.execute(
                select(JournalRow).where(
                    JournalRow.closed_at.is_not(None),
                    JournalRow.closed_at >= since,
                )
            )
        ).scalars().all()

    groups: dict[str, list[JournalRow]] = defaultdict(list)
    for r in rows:
        groups[r.skill_id or "unknown"].append(r)

    stats: dict[str, dict] = {}
    for skill_id, trades in groups.items():
        n = len(trades)
        wins = sum(1 for t in trades if (t.pnl_usd or 0) > 0)
        pnl = sum((t.pnl_usd or 0) for t in trades)
        rets = [(t.pnl_usd or 0) / max(abs(t.entry_price * t.qty), 1e-9) for t in trades]
        mean = sum(rets) / n if n else 0.0
        var = sum((r - mean) ** 2 for r in rets) / n if n > 1 else 0.0
        sharpe = mean / math.sqrt(var) * math.sqrt(252) if var > 0 else 0.0
        stats[skill_id] = {"trades": n, "wins": wins, "pnl_usd": pnl, "sharpe": sharpe}

    ranked = sorted(stats.items(), key=lambda kv: kv[1]["sharpe"], reverse=True)
    week_start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)

    lib = get_skill_library()
    async with db.session() as s:
        for rank, (skill_id, st) in enumerate(ranked):
            archived = st["trades"] >= min_trades and st["sharpe"] < 0
            s.add(
                SkillStatRow(
                    week_starting=week_start,
                    skill_id=skill_id,
                    trades=st["trades"],
                    wins=st["wins"],
                    pnl_usd=st["pnl_usd"],
                    sharpe=st["sharpe"],
                    rank=rank,
                    archived=archived,
                )
            )
            if archived:
                lib.archive(skill_id)
                log.info("skill_archived", skill_id=skill_id, sharpe=st["sharpe"])
        await s.commit()

    log.info("weekly_rerank_done", n_skills=len(stats))
    return stats
