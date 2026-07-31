"""news_analyst doit dégrader proprement, jamais propager (2026-07-31).

Défaut réel trouvé par audit d'exécution : `sentiment_analyst.py` capture déjà ses
propres exceptions (fetch échoué -> `state.sentiment = {}`), mais `news_analyst.py`
n'avait AUCUN try/except. Comme les trois analystes tournent en parallèle sous
`asyncio.gather` dans `orchestrator.py` et qu'une exception de n'importe lequel des
trois fait avorter TOUT le tick (`if state.error: return state`, orchestrator.py:36-38),
une simple panne de l'API de news (réseau, rate-limit, timeout — `NewsFeed` utilise
httpx) tuait le tick ENTIER, y compris le chemin déterministe (`selective_mode`) qui
n'a besoin ni de news ni d'aucun LLM. C'est la cause vérifiée de zéro trade exécuté
alors que `mean_reversion_regime` armait ~23 fois sur les 14 derniers jours dans les
données réelles (docs de l'audit du 2026-07-31).
"""

from __future__ import annotations

from hermes.agents import news_analyst
from hermes.graph.state import HermesState


class _RaisingFeed:
    async def headlines(self, symbol, limit=15):
        raise RuntimeError("news API down")


class _OkFeed:
    async def headlines(self, symbol, limit=15):
        return [{"title": "x", "symbol": symbol}]


async def test_news_fetch_failure_degrades_to_empty_headlines(monkeypatch):
    monkeypatch.setattr(news_analyst, "get_news", lambda: _RaisingFeed())
    state = HermesState(symbol="BTC/USDT:USDT")

    out = await news_analyst.run(state)

    assert out.news == [], "une panne de fetch doit dégrader à [] — jamais lever"
    assert out.error is None, "news_analyst ne doit JAMAIS mettre state.error (non critique)"


async def test_news_fetch_success_still_populates_headlines(monkeypatch):
    monkeypatch.setattr(news_analyst, "get_news", lambda: _OkFeed())
    state = HermesState(symbol="BTC/USDT:USDT")

    out = await news_analyst.run(state)

    assert out.news == [{"title": "x", "symbol": "BTC/USDT:USDT"}]
