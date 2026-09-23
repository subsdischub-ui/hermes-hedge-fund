"""Tick status telemetry, with no live scheduler, exchange or model calls."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from hermes import scheduler
from hermes.agents import market_analyst, news_analyst, sentiment_analyst
from hermes.graph.state import HermesState
from hermes.runtime.activity import ActivityLog
from hermes.settings import get_settings


@pytest.mark.asyncio
async def test_market_failure_reaches_activity_without_exception_secrets(monkeypatch, tmp_path):
    monkeypatch.setattr(get_settings(), "data_dir", tmp_path)
    async def fail(state):
        raise RuntimeError("provider failed token=DO_NOT_EXPOSE")

    async def advisory(state):
        return state

    monkeypatch.setattr(market_analyst, "run", fail)
    monkeypatch.setattr(news_analyst, "run", advisory)
    monkeypatch.setattr(sentiment_analyst, "run", advisory)
    sched = scheduler.build_scheduler()
    await sched.get_job("market_tick").func()
    items = ActivityLog(tmp_path / "activity.json").recent(100)
    assert len(items) == len(get_settings().universe)
    assert all(x["kind"] == "error" for x in items)
    assert all(x["decision"] == "ABORTED" for x in items)
    assert all(x["reason"] and not x["executed"] for x in items)
    assert "DO_NOT_EXPOSE" not in str(items)


@pytest.mark.asyncio
@pytest.mark.parametrize("decision,executed", [
    ("REJECTED", False), ("APPROVED", True), ("APPROVED", False),
])
async def test_completed_decisions_keep_their_status(monkeypatch, tmp_path, decision, executed):
    monkeypatch.setattr(get_settings(), "data_dir", tmp_path)
    async def tick(self, symbol, timeframe):
        return HermesState(
            symbol=symbol, executed=executed,
            risk_decision={"decision": decision, "reason": "original reason"},
            trader_proposal={"side": "BUY"},
            error="order failure DO_NOT_EXPOSE" if decision == "APPROVED" and not executed else None,
        )

    monkeypatch.setattr(scheduler.HermesOrchestrator, "tick", tick)
    sched = scheduler.build_scheduler()
    await sched.get_job("market_tick").func()
    items = ActivityLog(tmp_path / "activity.json").recent(100)
    assert items
    assert all(x["kind"] == "decision" and x["decision"] == decision for x in items)
    assert all(x["reason"] == "original reason" and x["executed"] == executed for x in items)
    assert "DO_NOT_EXPOSE" not in str(items)


@pytest.mark.asyncio
@pytest.mark.parametrize("raises,expected", [(False, "INCOMPLETE"), (True, "ERROR")])
async def test_incomplete_and_propagated_error_are_public_safe(monkeypatch, tmp_path, raises, expected):
    monkeypatch.setattr(get_settings(), "data_dir", tmp_path)
    async def tick(self, symbol, timeframe):
        if raises:
            raise RuntimeError("provider token=DO_NOT_EXPOSE")
        return HermesState(symbol=symbol)

    monkeypatch.setattr(scheduler.HermesOrchestrator, "tick", tick)
    sched = scheduler.build_scheduler()
    await sched.get_job("market_tick").func()
    items = ActivityLog(tmp_path / "activity.json").recent(100)
    assert len(items) == len(get_settings().universe)
    assert all(x["kind"] == "error" for x in items)
    assert all(x["decision"] == expected for x in items)
    reason = ("Échec du tick — consulter les journaux serveur" if raises else
              "Évaluation interrompue avant décision — consulter les journaux serveur")
    assert all(x["reason"] == reason for x in items)
    assert all(x["executed"] is False for x in items)
    assert "DO_NOT_EXPOSE" not in str(items)


@pytest.mark.asyncio
@pytest.mark.parametrize("status,executed,label", [
    ("ABORTED", False, "non"), ("INCOMPLETE", False, "non"),
    ("ERROR", False, "non"), ("APPROVED", False, "non"),
    ("APPROVED", True, "oui"), ("APPROVED", None, "inconnu"),
])
async def test_chat_exposes_tick_failure(monkeypatch, status, executed, label):
    from hermes import api

    router = SimpleNamespace(chat=AsyncMock(return_value={"content": "fixture", "model": "fixture"}))
    exchange = SimpleNamespace(equity_usd=AsyncMock(return_value=100),
                               open_positions=AsyncMock(return_value=[]))
    journal = SimpleNamespace(stats=AsyncMock(return_value={}),
                              recent_closed=AsyncMock(return_value=[]),
                              count_closed_by_skills=AsyncMock(return_value=0))
    monkeypatch.setattr(api, "get_router", lambda: router)
    monkeypatch.setattr(api, "get_exchange", lambda: exchange)
    monkeypatch.setattr(api, "TradeJournal", lambda: journal)
    monkeypatch.setattr(api, "_strategies_ab", lambda *args: {})
    monkeypatch.setattr(api, "_restructuration_chat_block", lambda *args: "")
    monkeypatch.setattr(api, "_recent_activity", lambda s: [{
        "kind": "decision" if status == "APPROVED" else "error",
        "symbol": "BTC/USDT:USDT", "decision": status, "executed": executed,
        "ts": "2026-09-22T16:01:00Z", "reason": "Évaluation interrompue",
    }])
    await api.chat(api.ChatRequest(message="Pourquoi aucun trade ?"))
    assert status in router.chat.call_args.kwargs["messages"][0]["content"]
    assert f"exécuté: {label}" in router.chat.call_args.kwargs["messages"][0]["content"]


def test_health_has_static_diagnostics_version():
    from fastapi.testclient import TestClient

    from hermes import api

    # No context manager: do not run lifespan or start background jobs.
    response = TestClient(api.app).get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "diagnostics_version": "tick-status-v1"}
