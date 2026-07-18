"""Portfolio-manager node — final reviewer; submits the order if approved.

Flow:
  1. If the deterministic risk engine REJECTED, we stop (no LLM call).
  2. Otherwise, we ask the LLM PM (per config/prompts/portfolio_manager.md)
     to confirm or veto. The LLM CAN downgrade to REJECTED but CANNOT
     upgrade a REJECTED — that contract is enforced here, not in the prompt.
  3. On APPROVED, we execute and journal.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

from hermes.agents.base import load_prompt
from hermes.execution import get_exchange
from hermes.execution.base import Order
from hermes.graph.state import HermesState
from hermes.llm import get_router
from hermes.logging_setup import get_logger
from hermes.memory.journal import JournalEntry, TradeJournal
from hermes.settings import get_settings

log = get_logger(__name__)


_PM_SCHEMA = {
    "type": "object",
    "properties": {
        "final_decision": {"type": "string", "enum": ["APPROVED", "REJECTED"]},
        "reason": {"type": "string"},
    },
    "required": ["final_decision", "reason"],
    "additionalProperties": False,
}


async def _llm_confirm(state: HermesState) -> tuple[str, str]:
    """Return (final_decision, reason). Defaults to APPROVED on LLM failure
    (the risk engine already approved; the LLM only adds an extra veto)."""
    try:
        router = get_router()
        ctx = {
            "trader_proposal": state.trader_proposal,
            "risk_decision": state.risk_decision,
            "trm_refinement": state.trm_refinement,
            "snapshot": state.snapshot,
        }
        out = await router.structured(
            agent_role="portfolio_manager",
            messages=[
                {"role": "system", "content": load_prompt("portfolio_manager")},
                {"role": "user", "content": json.dumps(ctx, default=str)},
            ],
            schema=_PM_SCHEMA,
            temperature=0.0,
        )
        if isinstance(out, dict) and out.get("final_decision") in {"APPROVED", "REJECTED"}:
            return out["final_decision"], out.get("reason", "")
    except Exception as e:
        log.warning("pm_llm_failed", error=str(e))
    return "APPROVED", "LLM PM unavailable — defaulting to risk-engine APPROVED"


async def run(state: HermesState) -> HermesState:
    risk = state.risk_decision or {}
    if risk.get("decision") != "APPROVED":
        state.executed = False
        log.info("pm_rejected_by_risk", reason=risk.get("reason"))
        return state

    final, reason = await _llm_confirm(state)
    if final == "REJECTED":
        state.executed = False
        state.risk_decision = {**risk, "decision": "REJECTED", "reason": f"PM veto: {reason}"}
        log.info("pm_rejected_by_llm", reason=reason)
        return state

    order_d = risk["order"]
    exchange = get_exchange()
    order = Order(
        symbol=order_d["symbol"],
        side=order_d["side"],
        qty=order_d["qty"],
        order_type=order_d.get("order_type", "market"),
        price=order_d.get("price"),
        stop_price=order_d.get("stop_price"),
        take_profit=order_d.get("take_profit"),
        extra=order_d.get("extra", {}),
    )
    try:
        fill = await exchange.place_order(order)
    except Exception as e:
        state.executed = False
        state.error = f"place_order failed: {e}"
        log.exception("pm_place_order_failed", symbol=order.symbol)
        return state

    # P1 shadow measurement (research_maker/PREREG_SHADOW_FILL.md): log a
    # hypothetical maker order at this same decision moment — pure shadow
    # accounting, never affects the real/paper fill above. Best-effort: a
    # shadow-measurement bug must never break a real entry.
    try:
        from hermes.execution.shadow_maker import record_entry

        _dec_ts = state.ohlcv["ts"].iloc[-1] if state.ohlcv is not None else None
        record_entry(
            ts=fill.ts,
            symbol=fill.symbol,
            side=order.side,
            skill_id=order.extra.get("skill_id", ""),
            decision_close=float(state.snapshot.get("price", fill.price)),
            # OPEN epoch-ms de la barre de décision (la dernière close) — ancre de la
            # garde anti-écart : seule LA barre suivante pourra résoudre cette entrée.
            decision_bar_ts=int(_dec_ts.timestamp() * 1000)
            if _dec_ts is not None and hasattr(_dec_ts, "timestamp")
            else (int(_dec_ts) if _dec_ts is not None else None),
        )
    except Exception as e:
        log.warning("shadow_maker_record_failed", symbol=fill.symbol, error=str(e))

    journal = TradeJournal()
    s = get_settings()
    trade_id = await journal.open(
        JournalEntry(
            symbol=fill.symbol,
            side=fill.side,
            qty=fill.qty,
            entry_price=fill.price,
            opened_at=fill.ts or datetime.now(UTC),
            confidence=float(state.trader_proposal.get("confidence", 0.0)),
            skill_id=order.extra.get("skill_id", ""),
            thesis=order.extra.get("thesis", ""),
            stage=s.stage,
            market_state=state.snapshot,
        )
    )
    state.executed = True
    state.trade_id = trade_id
    log.info("pm_executed", trade_id=trade_id, symbol=fill.symbol, side=fill.side, qty=fill.qty)
    return state
