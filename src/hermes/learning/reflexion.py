"""Per-trade Reflexion: when a trade closes, distill 'what surprised me / what to keep'."""

from __future__ import annotations

from hermes.llm import get_router
from hermes.memory.journal import TradeJournal
from hermes.memory.store import JournalRow, get_db
from hermes.logging_setup import get_logger

log = get_logger(__name__)

REFLECTION_SYS = """You are Hermès' reflective journaling agent.
Given a closed trade, produce a SHORT reflection (3-5 sentences) and 1-3 tags.
Be honest. Do not invent details. Be specific about what surprised the trader.
Output JSON: {"reflection": "...", "tags": ["..."]}."""


async def reflect_on_trade(trade_id: int) -> str:
    db = get_db()
    async with db.session() as s:
        row = await s.get(JournalRow, trade_id)
        if row is None or row.closed_at is None:
            return ""

    router = get_router()
    user = (
        f"Symbol: {row.symbol} | Side: {row.side} | Skill: {row.skill_id}\n"
        f"Thesis: {row.thesis}\n"
        f"Entry: ${row.entry_price:.4f} | Exit: ${row.exit_price:.4f} | "
        f"PnL: ${row.pnl_usd:.2f} | Reason: {row.exit_reason}\n"
        f"Confidence at entry: {row.confidence:.2f}\n"
        f"Market state snapshot: {row.market_state}\n"
    )
    out = await router.structured(
        agent_role="reflector",
        messages=[{"role": "system", "content": REFLECTION_SYS}, {"role": "user", "content": user}],
        schema={
            "type": "object",
            "properties": {
                "reflection": {"type": "string"},
                "tags": {"type": "array", "items": {"type": "string"}, "maxItems": 3},
            },
            "required": ["reflection", "tags"],
            "additionalProperties": False,
        },
        temperature=0.3,
    )
    text = out.get("reflection", "") if isinstance(out, dict) else ""
    tags = out.get("tags", []) if isinstance(out, dict) else []

    # Persist reflection on the journal row.
    journal = TradeJournal()
    await journal.close(
        trade_id=trade_id,
        exit_price=row.exit_price or 0.0,
        pnl_usd=row.pnl_usd or 0.0,
        fee_usd=row.fee_usd,
        exit_reason=row.exit_reason,
        reflection=text,
    )
    log.info("trade_reflected", trade_id=trade_id, tags=tags)
    return text
