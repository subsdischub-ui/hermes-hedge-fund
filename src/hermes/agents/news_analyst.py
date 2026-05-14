"""News-analyst node: fetch + lightweight relevance filter (no LLM needed)."""

from __future__ import annotations

from hermes.data import get_news
from hermes.graph.state import HermesState


async def run(state: HermesState) -> HermesState:
    feed = get_news()
    state.news = await feed.headlines(state.symbol, limit=15)
    return state
