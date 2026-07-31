"""HermesOrchestrator — runs the multi-agent decision graph for one symbol.

Linear flow (parallelizable steps run concurrently):
    market_analyst  ─┐
    news_analyst    ─┼─►  fetch_lessons  ►  bull ║ bear  ►  trader  ►  risk_manager  ►  portfolio_manager
    sentiment       ─┘
"""

from __future__ import annotations

import asyncio

from hermes.agents import (
    market_analyst,
    news_analyst,
    portfolio_manager,
    researcher,
    risk_manager,
    sentiment_analyst,
    trader,
)
from hermes.graph.state import HermesState
from hermes.logging_setup import get_logger

log = get_logger(__name__)


class HermesOrchestrator:
    async def tick(self, symbol: str, timeframe: str = "1h") -> HermesState:
        state = HermesState(symbol=symbol, timeframe=timeframe)
        # Only market_analyst is CRITICAL: without OHLCV/skill_signals no decision
        # (deterministic or LLM) is possible, so its failure must abort the tick.
        # news/sentiment are ADVISORY context for the LLM path — the deterministic
        # selective_mode override needs neither. Before this fix both used the same
        # `_safe` (which sets the shared, tick-aborting `state.error`), so ANY of
        # the three failing killed the WHOLE tick, including the deterministic path
        # that doesn't even read news/sentiment (audit 2026-07-31: a transient news-
        # feed failure was silently blocking ~23 real armed mean_reversion_regime
        # signals over 14 days, reproduced in tests/test_orchestrator.py). news_analyst
        # itself now degrades internally too (belt AND suspenders — a future agent
        # added to this gather must not silently regain the power to kill the tick).
        await asyncio.gather(
            self._safe(market_analyst.run, state),
            self._safe_advisory(news_analyst.run, state),
            self._safe_advisory(sentiment_analyst.run, state),
        )
        if state.error:
            log.warning("tick_aborted", symbol=symbol, error=state.error)
            return state

        await researcher.fetch_lessons(state)
        await asyncio.gather(
            self._safe(researcher.bull, state),
            self._safe(researcher.bear, state),
        )
        await trader.run(state)
        await risk_manager.run(state)
        await portfolio_manager.run(state)
        log.info(
            "tick_done",
            symbol=symbol,
            executed=state.executed,
            decision=state.risk_decision.get("decision"),
            confidence=state.trader_proposal.get("confidence"),
        )
        return state

    @staticmethod
    async def _safe(fn, state: HermesState) -> None:
        try:
            await fn(state)
        except Exception as e:
            log.exception("agent_node_failed", node=fn.__module__, error=str(e))
            state.error = state.error or f"{fn.__module__}: {e}"

    @staticmethod
    async def _safe_advisory(fn, state: HermesState) -> None:
        """Same as ``_safe`` but for a NON-CRITICAL parallel node (news/sentiment):
        logs and swallows — never sets ``state.error``, so it can never abort the
        tick. The deterministic path does not read either agent's output."""
        try:
            await fn(state)
        except Exception as e:
            log.warning("advisory_node_failed", node=fn.__module__, error=str(e))
