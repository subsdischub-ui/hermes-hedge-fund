"""Risk-manager node — runs the deterministic risk engine + TRM verifier."""

from __future__ import annotations

import numpy as np

from hermes.execution import get_exchange
from hermes.graph.state import HermesState
from hermes.memory.journal import TradeJournal
from hermes.risk import get_risk_engine
from hermes.risk.guardrails import TradeProposal
from hermes.trm import get_trm
from hermes.trm.encoder import ActionCodec, MarketStateEncoder, Action


async def run(state: HermesState) -> HermesState:
    proposal_d = state.trader_proposal or {}
    side = proposal_d.get("side", "HOLD")
    proposal = TradeProposal(
        symbol=proposal_d.get("symbol", state.symbol),
        side=side,
        confidence=float(proposal_d.get("confidence", 0.0)),
        stop_pct=float(proposal_d.get("stop_pct", 0.02)),
        take_profit_pct=float(proposal_d.get("take_profit_pct", 0.04)),
        thesis=proposal_d.get("thesis", ""),
        skill_id=proposal_d.get("skill_id", ""),
    )

    # --- TRM refinement (only for non-HOLD) ---
    if state.ohlcv is not None and side != "HOLD":
        try:
            enc = MarketStateEncoder()
            grid = enc.encode(state.ohlcv)
            codec = ActionCodec()
            draft = codec.encode(
                Action(
                    side=proposal.side,
                    size_pct=proposal.confidence,
                    stop_pct=proposal.stop_pct,
                    tp_pct=proposal.take_profit_pct,
                )
            )
            verifier = get_trm()
            verdict = verifier.refine(grid, draft, instrument_id=0)
            refined = codec.decode(np.array(verdict.refined_action))
            state.trm_refinement = {
                "side": refined.side,
                "size_pct": refined.size_pct,
                "stop_pct": refined.stop_pct,
                "tp_pct": refined.tp_pct,
                "confidence": verdict.confidence,
                "used_trm": verdict.used_trm,
            }
            # Soft-override: TRM may downgrade to HOLD; keep its confidence as a floor.
            if refined.side != proposal.side and verdict.used_trm and verdict.confidence > 0.7:
                proposal.side = refined.side
            proposal.confidence = min(proposal.confidence, verdict.confidence)
        except Exception as e:
            state.trm_refinement = {"error": str(e), "used_trm": False, "confidence": 1.0}

    # --- deterministic risk engine ---
    engine = get_risk_engine()
    exchange = get_exchange()
    journal = TradeJournal()
    stats = await journal.stats()
    decision = await engine.evaluate(
        proposal=proposal,
        exchange=exchange,
        atr=float(state.snapshot.get("atr", 0.0)),
        price=float(state.snapshot.get("price", 0.0)),
        journal_stats=stats,
    )
    state.risk_decision = {
        "decision": decision.decision,
        "reason": decision.reason,
        "order": decision.order.__dict__ if decision.order else None,
        "sizing": decision.sizing.__dict__ if decision.sizing else None,
    }
    return state
