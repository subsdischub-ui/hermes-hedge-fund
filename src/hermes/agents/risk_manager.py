"""Risk-manager node — runs the deterministic risk engine + TRM verifier."""

from __future__ import annotations

import numpy as np

from hermes.execution import get_exchange
from hermes.graph.state import HermesState
from hermes.logging_setup import get_logger
from hermes.memory.journal import TradeJournal
from hermes.risk import get_risk_engine
from hermes.risk.calibration import get_confidence_calibrator
from hermes.risk.guardrails import TradeProposal
from hermes.trm import get_trm
from hermes.trm.encoder import Action, ActionCodec, MarketStateEncoder

log = get_logger(__name__)


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
            # Only let the verdict override the trader's side when (a) TRM is real
            # (not the Null fallback) AND (b) it is downgrading to HOLD AND
            # (c) its halt-confidence is high. We refuse to flip BUY<->SELL or
            # raise HOLD to a trade — those are LLM-level decisions.
            if (
                verdict.used_trm
                and refined.side == "HOLD"
                and verdict.confidence > 0.7
            ):
                log.info(
                    "trm_downgrade_to_hold",
                    symbol=proposal.symbol,
                    original_side=proposal.side,
                    trm_confidence=verdict.confidence,
                )
                proposal.side = "HOLD"
            # TRM acts as a confidence CEILING when it's real; the NullVerifier
            # returns confidence=1.0 so this is a no-op when TRM is disabled.
            if verdict.used_trm:
                proposal.confidence = min(proposal.confidence, verdict.confidence)
        except Exception as e:
            # On any verifier error treat the proposal as UNVERIFIED — set
            # confidence to 0 so the risk engine's confidence_floor will reject.
            log.warning("trm_refine_failed", error=str(e), symbol=proposal.symbol)
            state.trm_refinement = {
                "error": str(e),
                "used_trm": False,
                "confidence": 0.0,
            }
            proposal.confidence = 0.0

    # --- confidence calibration: honest P(win) before the gate + Kelly sizing ---
    # The LLM is systematically over-confident; map its confidence through an
    # isotonic calibrator fitted on realized outcomes so the SAME value that gates
    # and sizes the trade reflects the true historical hit-rate. Identity (no-op)
    # until enough closed trades exist, so it can only ever lower over-confidence.
    if proposal.side != "HOLD":
        cal = await get_confidence_calibrator()
        raw_conf = proposal.confidence
        proposal.confidence = cal.calibrate(raw_conf)
        state.confidence_calibration = {
            "raw": raw_conf,
            "calibrated": proposal.confidence,
            "fitted": cal.is_fitted,
        }
        if cal.is_fitted:
            log.info(
                "confidence_calibrated",
                symbol=proposal.symbol,
                raw=round(raw_conf, 4),
                calibrated=round(proposal.confidence, 4),
            )

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
