"""Bull + Bear researchers."""

from __future__ import annotations

import json

from hermes.agents.base import load_prompt
from hermes.graph.state import HermesState
from hermes.llm import get_router
from hermes.memory.lessons import LessonsStore


def _context(state: HermesState) -> str:
    return json.dumps(
        {
            "symbol": state.symbol,
            "timeframe": state.timeframe,
            "snapshot": state.snapshot,
            "news_headlines": [n.get("title") for n in state.news[:10]],
            "sentiment": state.sentiment,
            "skill_signals": state.skill_signals,
            "lessons": [l["text"] for l in state.relevant_lessons[:6]],
        },
        default=str,
    )


async def fetch_lessons(state: HermesState) -> HermesState:
    store = LessonsStore()
    q = f"{state.symbol} {state.timeframe} regime atr {state.snapshot.get('atr_pct', 0):.4f}"
    state.relevant_lessons = await store.search(q, k=6)
    return state


async def bull(state: HermesState) -> HermesState:
    router = get_router()
    out = await router.structured(
        agent_role="bull_researcher",
        messages=[
            {"role": "system", "content": load_prompt("bull")},
            {"role": "user", "content": _context(state)},
        ],
        schema={
            "type": "object",
            "properties": {
                "thesis": {"type": "string"},
                "skill_id": {"type": "string"},
                "confidence": {"type": "number"},
                "counter_risk": {"type": "string"},
            },
            "required": ["thesis", "skill_id", "confidence", "counter_risk"],
            "additionalProperties": False,
        },
    )
    state.bull_view = json.dumps(out) if isinstance(out, dict) else str(out)
    return state


async def bear(state: HermesState) -> HermesState:
    router = get_router()
    out = await router.structured(
        agent_role="bear_researcher",
        messages=[
            {"role": "system", "content": load_prompt("bear")},
            {"role": "user", "content": _context(state)},
        ],
        schema={
            "type": "object",
            "properties": {
                "thesis": {"type": "string"},
                "skill_id": {"type": "string"},
                "side": {"type": "string", "enum": ["SELL", "HOLD"]},
                "confidence": {"type": "number"},
                "counter_risk": {"type": "string"},
            },
            "required": ["thesis", "skill_id", "side", "confidence", "counter_risk"],
            "additionalProperties": False,
        },
    )
    state.bear_view = json.dumps(out) if isinstance(out, dict) else str(out)
    return state
