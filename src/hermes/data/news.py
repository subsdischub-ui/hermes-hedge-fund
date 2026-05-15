"""Crypto news feed — CryptoPanic free tier with Reddit fallback."""

from __future__ import annotations

from functools import lru_cache

import httpx

from hermes.logging_setup import get_logger
from hermes.settings import get_settings

log = get_logger(__name__)


class NewsFeed:
    def __init__(self, cryptopanic_token: str):
        self.token = cryptopanic_token
        self.http = httpx.AsyncClient(timeout=10.0)

    async def close(self) -> None:
        await self.http.aclose()

    async def headlines(self, symbol: str, limit: int = 20) -> list[dict]:
        currencies = symbol.split("/")[0]
        if self.token:
            r = await self.http.get(
                "https://cryptopanic.com/api/developer/v2/posts/",
                params={"auth_token": self.token, "currencies": currencies, "public": "true"},
            )
            if r.status_code == 200:
                items = r.json().get("results", [])[:limit]
                return [
                    {
                        "title": it.get("title"),
                        "url": it.get("url"),
                        "published_at": it.get("published_at"),
                        "kind": it.get("kind"),
                        "votes": it.get("votes", {}),
                    }
                    for it in items
                ]
        # Reddit fallback (no auth)
        try:
            r = await self.http.get(
                "https://www.reddit.com/r/CryptoCurrency/.json",
                headers={"User-Agent": "hermes-agent/0.1"},
                params={"limit": limit},
            )
            data = r.json().get("data", {}).get("children", [])
            return [
                {
                    "title": c["data"].get("title"),
                    "url": "https://reddit.com" + c["data"].get("permalink", ""),
                    "score": c["data"].get("score"),
                }
                for c in data
            ]
        except Exception as e:  # pragma: no cover
            log.warning("news_fallback_failed", error=str(e))
            return []


@lru_cache(maxsize=1)
def get_news() -> NewsFeed:
    s = get_settings()
    token = s.cryptopanic_token.get_secret_value() if hasattr(s.cryptopanic_token, "get_secret_value") else str(s.cryptopanic_token)
    return NewsFeed(cryptopanic_token=token)
