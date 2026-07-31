"""Garde de non-régression du tick : une panne d'un analyste NON CRITIQUE ne doit
jamais bloquer le chemin déterministe (2026-07-31).

Défaut réel trouvé par audit d'exécution le 2026-07-31, après que l'opérateur ait
signalé zéro trade en 29 jours malgré un marché actif. Vérifié pas à pas contre le
code réel avec des données de marché réelles : mean_reversion_regime, le régime
Kaufman, la pondération de régime, la couche combination, drive_signal/selective_gate,
et RiskEngine.evaluate() APPROUVENT tous, en isolation, plusieurs signaux BUY réels
des 14 derniers jours (ex. BTC 2026-07-23 16:00, notional $19.12). Le seul maillon
non testé isolément était `HermesOrchestrator.tick()` lui-même :

    await asyncio.gather(
        self._safe(market_analyst.run, state),
        self._safe(news_analyst.run, state),
        self._safe(sentiment_analyst.run, state),
    )
    if state.error:                      # <-- N'IMPORTE LEQUEL des 3 peut mettre ceci
        return state                     # <-- et TOUT le tick avorte, y compris
                                          #     risk_manager/portfolio_manager

`sentiment_analyst.run` capture déjà ses erreurs en interne (dégrade à `{}`).
`news_analyst.run` ne le faisait PAS (fix : tests/test_news_analyst.py, même
commit) — mais le vrai défaut structurel est que RIEN ne garantit que ce restera
vrai : le prochain agent ajouté au `asyncio.gather` (ou une régression future dans
sentiment_analyst) referait avorter le chemin déterministe sur une panne non
critique. Ce test garde le COMPORTEMENT du tick, pas seulement le module actuel.
"""

from __future__ import annotations

import hermes.graph.orchestrator as orch_mod
from hermes.graph.orchestrator import HermesOrchestrator


async def test_a_non_critical_analyst_failure_never_blocks_the_deterministic_path(
    monkeypatch,
):
    calls: list[str] = []

    async def fake_market_analyst(state):
        state.ohlcv = "fake_df"
        state.snapshot = {"price": 100.0, "atr": 1.0}
        state.regime = {"regime": "range", "efficiency_ratio": 0.1}
        state.skill_signals = [
            {"skill_id": "mean_reversion_regime", "side": "BUY", "confidence": 0.6,
             "confidence_raw": 0.6, "stop_pct": 0.02, "take_profit_pct": 0.04}
        ]
        calls.append("market_analyst")

    async def raising_news_analyst(state):
        # Simule EXACTEMENT le défaut réel : une exception non capturée d'un
        # analyste non critique, avant le fix de news_analyst.run.
        raise RuntimeError("news API down")

    async def fake_sentiment_analyst(state):
        calls.append("sentiment_analyst")

    async def fake_fetch_lessons(state):
        calls.append("fetch_lessons")

    async def fake_bull(state):
        calls.append("bull")

    async def fake_bear(state):
        calls.append("bear")

    async def fake_trader(state):
        state.trader_proposal = {"side": "HOLD", "confidence": 0.0}
        calls.append("trader")

    async def fake_risk_manager(state):
        state.risk_decision = {"decision": "APPROVED", "reason": "test"}
        calls.append("risk_manager")

    async def fake_portfolio_manager(state):
        calls.append("portfolio_manager")

    monkeypatch.setattr(orch_mod.market_analyst, "run", fake_market_analyst)
    monkeypatch.setattr(orch_mod.news_analyst, "run", raising_news_analyst)
    monkeypatch.setattr(orch_mod.sentiment_analyst, "run", fake_sentiment_analyst)
    monkeypatch.setattr(orch_mod.researcher, "fetch_lessons", fake_fetch_lessons)
    monkeypatch.setattr(orch_mod.researcher, "bull", fake_bull)
    monkeypatch.setattr(orch_mod.researcher, "bear", fake_bear)
    monkeypatch.setattr(orch_mod.trader, "run", fake_trader)
    monkeypatch.setattr(orch_mod.risk_manager, "run", fake_risk_manager)
    monkeypatch.setattr(orch_mod.portfolio_manager, "run", fake_portfolio_manager)

    state = await HermesOrchestrator().tick("BTC/USDT:USDT", timeframe="4h")

    assert "risk_manager" in calls, (
        "risk_manager n'a jamais tourné — une panne de news_analyst (non critique, "
        "n'affecte pas le chemin déterministe) a fait avorter tout le tick"
    )
    assert "portfolio_manager" in calls
    assert state.risk_decision.get("decision") == "APPROVED"


async def test_a_market_analyst_failure_correctly_aborts_the_tick(monkeypatch):
    """Le pendant : market_analyst EST critique (sans OHLCV, aucune décision n'est
    possible) — son échec doit continuer d'avorter le tick. Ce test empêche de
    'corriger' le défaut ci-dessus en supprimant l'abort en entier."""
    calls: list[str] = []

    async def raising_market_analyst(state):
        raise RuntimeError("exchange API down")

    async def fake_news_analyst(state):
        calls.append("news_analyst")

    async def fake_sentiment_analyst(state):
        calls.append("sentiment_analyst")

    async def fake_risk_manager(state):
        calls.append("risk_manager")

    monkeypatch.setattr(orch_mod.market_analyst, "run", raising_market_analyst)
    monkeypatch.setattr(orch_mod.news_analyst, "run", fake_news_analyst)
    monkeypatch.setattr(orch_mod.sentiment_analyst, "run", fake_sentiment_analyst)
    monkeypatch.setattr(orch_mod.risk_manager, "run", fake_risk_manager)

    state = await HermesOrchestrator().tick("BTC/USDT:USDT", timeframe="4h")

    assert state.error is not None
    assert "risk_manager" not in calls, (
        "risk_manager a tourné SANS OHLCV valide — market_analyst doit rester critique"
    )
