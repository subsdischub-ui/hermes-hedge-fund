"""Sentiment-analyst node."""

from __future__ import annotations

from hermes.data import get_sentiment
from hermes.graph.state import HermesState
from hermes.logging_setup import get_logger

log = get_logger(__name__)


async def run(state: HermesState) -> HermesState:
    feed = get_sentiment()
    try:
        state.sentiment = await feed.fear_greed()
    except Exception as e:
        log.warning("sentiment_fetch_failed", error=str(e))
        state.sentiment = {}
    return state
