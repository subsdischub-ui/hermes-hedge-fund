"""Trader node — picks BUY/SELL/HOLD and proposes a trade."""

from __future__ import annotations

import json

from hermes.agents.base import load_prompt
from hermes.graph.state import HermesState
from hermes.llm import get_router
from hermes.llm.tool_schema import TRADER_TOOLS
from hermes.logging_setup import get_logger

log = get_logger(__name__)


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
    try:
        out = await router.chat(
            agent_role="trader",
            messages=[
                {"role": "system", "content": load_prompt("trader")},
                {"role": "user", "content": user},
            ],
            tools=TRADER_TOOLS,
            temperature=0.2,
        )
    except Exception as e:
        # LLM totally unreachable (primary AND fallback). The LLM is ADVISORY: its
        # outage must not abort the tick — in selective_mode risk_manager's
        # deterministic drive_signal overrides this proposal anyway, so defaulting
        # to HOLD keeps the validated rule (S2 & co) trading while the LLM is down
        # instead of silently suspending the whole plan (audit sûreté 2026-07-10).
        log.warning("trader_llm_down_default_hold", symbol=state.symbol, error=str(e))
        state.trader_proposal = {
            "symbol": state.symbol,
            "side": "HOLD",
            "thesis": "LLM indisponible — HOLD par défaut; le driver déterministe décide",
            "confidence": 0.0,
            "stop_pct": 0.02,
            "take_profit_pct": 0.04,
            "skill_id": "",
        }
        return state
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
