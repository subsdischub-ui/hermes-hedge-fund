"""Lightweight sentiment signals: Fear & Greed, Reddit upvote ratio."""

from __future__ import annotations

from functools import lru_cache

import httpx


class SentimentFeed:
    def __init__(self) -> None:
        self.http = httpx.AsyncClient(timeout=10.0)

    async def close(self) -> None:
        await self.http.aclose()

    async def fear_greed(self) -> dict:
        r = await self.http.get("https://api.alternative.me/fng/?limit=1")
        r.raise_for_status()
        d = r.json()["data"][0]
        return {"value": int(d["value"]), "classification": d["value_classification"]}


@lru_cache(maxsize=1)
def get_sentiment() -> SentimentFeed:
    return SentimentFeed()
