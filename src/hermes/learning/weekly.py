"""Weekly: re-rank skills by Deflated Sharpe Ratio (anti-cherry-pick) and archive
only skills that have a real track record AND show no statistical evidence of edge.

Replaces the previous naive 30-day annualized Sharpe (which treated trades as
i.i.d., bolted on a spurious sqrt(252), and pruned on Sharpe < 0 across several
skills tested in parallel — textbook backtest overfitting). See learning/stats.py.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import UTC, datetime, timedelta

from sqlalchemy import select

from hermes.learning.stats import score_skills
from hermes.logging_setup import get_logger
from hermes.memory.store import JournalRow, SkillStatRow, get_db
from hermes.skills import get_skill_library

log = get_logger(__name__)


def returns_by_skill(rows) -> dict[str, list[float]]:
    """Group closed-trade rows into per-skill, per-trade returns (pnl / notional)."""
    groups: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        notional = max(abs(float(r.entry_price) * float(r.qty)), 1e-9)
        groups[r.skill_id or "unknown"].append(float(r.pnl_usd or 0.0) / notional)
    return dict(groups)


async def run_weekly_skill_rerank(lookback_days: int = 30, min_trades: int = 10) -> dict[str, dict]:
    since = datetime.now(UTC) - timedelta(days=lookback_days)
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

    # Win count + cumulative PnL per skill, for the persisted report.
    wins_pnl: dict[str, tuple[int, float]] = {}
    grouped: dict[str, list] = defaultdict(list)
    for r in rows:
        grouped[r.skill_id or "unknown"].append(r)
    for sid, rs in grouped.items():
        wins_pnl[sid] = (
            sum(1 for t in rs if (t.pnl_usd or 0) > 0),
            sum((t.pnl_usd or 0) for t in rs),
        )

    # DSR-based scoring + archival decision (deflated by the number of skills tested).
    scored = score_skills(returns_by_skill(rows), min_trades=min_trades)
    ranked = sorted(scored.items(), key=lambda kv: kv[1]["dsr"], reverse=True)
    week_start = datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0)

    lib = get_skill_library()
    async with db.session() as s:
        for rank, (skill_id, st) in enumerate(ranked):
            wins, pnl = wins_pnl.get(skill_id, (0, 0.0))
            st["wins"], st["pnl_usd"] = wins, pnl  # enrich the returned report
            s.add(
                SkillStatRow(
                    week_starting=week_start,
                    skill_id=skill_id,
                    trades=st["trades"],
                    wins=wins,
                    pnl_usd=pnl,
                    sharpe=st["sharpe"],
                    dsr=st["dsr"],
                    rank=rank,
                    archived=st["archived"],
                )
            )
            if st["archived"]:
                lib.archive(skill_id)
                log.info("skill_archived", skill_id=skill_id, dsr=st["dsr"], sharpe=st["sharpe"])
        await s.commit()

    log.info("weekly_rerank_done", n_skills=len(scored))
    return scored
