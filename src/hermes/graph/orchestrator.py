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
        await asyncio.gather(
            self._safe(market_analyst.run, state),
            self._safe(news_analyst.run, state),
            self._safe(sentiment_analyst.run, state),
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
