"""Trader node — picks BUY/SELL/HOLD and proposes a trade."""

from __future__ import annotations

import json

from hermes.agents.base import load_prompt
from hermes.graph.state import HermesState
from hermes.llm import get_router
from hermes.llm.tool_schema import TRADER_TOOLS


async def run(state: HermesState) -> HermesState:
    router = get_router()
    user = json.dumps(
        {
            "symbol": state.symbol,
            "snapshot": state.snapshot,
            "bull": state.bull_view,
            "bear": state.bear_view,
            "skill_signals": state.skill_signals,
            "lessons": [l["text"] for l in state.relevant_lessons[:6]],
        },
        default=str,
    )
    out = await router.chat(
        agent_role="trader",
        messages=[
            {"role": "system", "content": load_prompt("trader")},
            {"role": "user", "content": user},
        ],
        tools=TRADER_TOOLS,
        temperature=0.2,
    )
    tool_calls = out.get("tool_calls", [])
    proposal: dict = {}
    for tc in tool_calls:
        if tc["name"] == "propose_trade":
            proposal = dict(tc["arguments"]) if isinstance(tc["arguments"], dict) else {}
            break
    if not proposal:
        proposal = {
            "symbol": state.symbol,
            "side": "HOLD",
            "thesis": "no proposal emitted; default HOLD",
            "confidence": 0.0,
            "stop_pct": 0.02,
            "take_profit_pct": 0.04,
            "skill_id": "",
        }
    state.trader_proposal = proposal
    return state
