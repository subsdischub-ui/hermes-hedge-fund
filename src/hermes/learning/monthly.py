"""Monthly: compile-time prompt optimization stub.

We sketch the DSPy/GEPA optimization loop here as a stub. Wiring real DSPy
program graphs against our LangGraph nodes is out-of-scope for the first
deployment; this entry point is a no-op that just exports a training set of
(prompt_inputs, prompt_outputs, realized_pnl) triples for offline iteration.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

from sqlalchemy import select

from hermes.logging_setup import get_logger
from hermes.memory.store import JournalRow, get_db
from hermes.settings import get_settings

log = get_logger(__name__)


async def run_monthly_compile(lookback_days: int = 30) -> str:
    """Export a training set; user runs DSPy/GEPA offline against it."""
    s = get_settings()
    out_dir = s.data_dir / "compile"
    out_dir.mkdir(parents=True, exist_ok=True)
    since = datetime.now(timezone.utc) - timedelta(days=lookback_days)
    db = get_db()
    async with db.session() as sess:
        rows = (
            await sess.execute(
                select(JournalRow).where(
                    JournalRow.closed_at.is_not(None), JournalRow.closed_at >= since
                )
            )
        ).scalars().all()

    examples = [
        {
            "symbol": r.symbol,
            "side": r.side,
            "skill_id": r.skill_id,
            "thesis": r.thesis,
            "confidence": r.confidence,
            "market_state": r.market_state,
            "pnl_usd": r.pnl_usd,
            "reflection": r.reflection,
        }
        for r in rows
    ]
    path = out_dir / f"trainset_{datetime.now(timezone.utc):%Y%m%d}.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, default=str) + "\n")
    log.info("compile_trainset_written", path=str(path), n=len(examples))
    return str(path)
