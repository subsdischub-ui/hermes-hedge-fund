"""Nightly reflection: distill the day's reflections into 1-3 generalizable lessons."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from hermes.llm import get_router
from hermes.logging_setup import get_logger
from hermes.memory.journal import TradeJournal
from hermes.memory.lessons import LessonsStore
from hermes.risk.calibration import reset_confidence_calibrator

log = get_logger(__name__)

LESSON_SYS = """You are Hermès' nightly lesson-distiller.
You receive a batch of closed-trade reflections from the last 24h.
Produce 1-3 GENERALIZABLE LESSONS the agent should apply tomorrow.
Each lesson must be:
  - one sentence, declarative, falsifiable (no platitudes),
  - tagged with up to 3 short tags,
  - scored 0..1 by your subjective importance.
Output JSON: {"lessons": [{"text": "...", "tags": ["..."], "score": 0.x}, ...]}.
Return an empty list if the day's signal is too noisy to extract a lesson.
"""


async def run_nightly_reflection() -> int:
    # Refit the confidence calibrator on the latest realized trade outcomes so
    # tomorrow's sizing/gating uses an up-to-date P(win) mapping.
    reset_confidence_calibrator()
    journal = TradeJournal()
    since = datetime.now(UTC) - timedelta(hours=24)
    rows = await journal.recent_closed(since=since)
    if not rows:
        log.info("nightly_no_trades")
        return 0

    payload = "\n\n".join(
        f"[{r.symbol} {r.side} skill={r.skill_id} pnl=${(r.pnl_usd or 0):.2f} reason={r.exit_reason}] "
        f"REFLECTION: {r.reflection}"
        for r in rows
    )
    router = get_router()
    out = await router.structured(
        agent_role="reflector",
        messages=[{"role": "system", "content": LESSON_SYS}, {"role": "user", "content": payload}],
        schema={
            "type": "object",
            "properties": {
                "lessons": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "text": {"type": "string"},
                            "tags": {"type": "array", "items": {"type": "string"}},
                            "score": {"type": "number"},
                        },
                        "required": ["text", "tags", "score"],
                        "additionalProperties": False,
                    },
                    "maxItems": 3,
                }
            },
            "required": ["lessons"],
            "additionalProperties": False,
        },
        temperature=0.4,
    )
    lessons = (out or {}).get("lessons", []) if isinstance(out, dict) else []
    store = LessonsStore()
    ids = [r.id for r in rows]
    added = 0
    for lesson in lessons:
        await store.add(
            text=lesson["text"],
            tags=list(lesson.get("tags", [])),
            source_trade_ids=ids,
            score=float(lesson.get("score", 0.0)),
        )
        added += 1
    log.info("nightly_lessons_added", count=added, trades=len(rows))
    return added
