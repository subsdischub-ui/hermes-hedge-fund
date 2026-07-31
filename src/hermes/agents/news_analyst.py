"""News-analyst node: fetch + lightweight relevance filter (no LLM needed)."""

from __future__ import annotations

from hermes.data import get_news
from hermes.graph.state import HermesState
from hermes.logging_setup import get_logger

log = get_logger(__name__)


async def run(state: HermesState) -> HermesState:
    # Fail-soft, mirroring sentiment_analyst.py. News/sentiment/market run
    # CONCURRENTLY in the orchestrator (asyncio.gather) and ANY exception here sets
    # state.error, which aborts the WHOLE tick (orchestrator.py: `if state.error:
    # return state`) BEFORE risk_manager ever runs — including the deterministic
    # selective_mode path, which needs neither news nor any LLM. Before this fix,
    # this function had no try/except: a transient httpx failure on the news feed
    # (network blip, rate limit, timeout) silently killed every decision for that
    # symbol's tick, deterministic or not (audit 2026-07-31, reproduced in
    # tests/test_orchestrator.py). News is advisory context — its absence must
    # degrade to no headlines, never abort the tick.
    try:
        feed = get_news()
        state.news = await feed.headlines(state.symbol, limit=15)
    except Exception as e:
        log.warning("news_fetch_failed", symbol=state.symbol, error=str(e))
        state.news = []
    return state
