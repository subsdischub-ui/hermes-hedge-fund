"""Operator chat: a read-only conversational endpoint that answers from Hermès'
live state. The LLM is mocked so tests never hit the network."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from fastapi.testclient import TestClient

from hermes import api
from hermes.api import app
from hermes.memory.journal import JournalEntry, TradeJournal
from hermes.memory.store import get_db


class _FakeRouter:
    last_messages: list = []

    async def chat(self, **kwargs):
        _FakeRouter.last_messages = kwargs["messages"]
        return {"content": "Voici où en est notre entreprise.", "model": "deepseek-v4-pro"}


def test_chat_replies_with_live_context(monkeypatch):
    monkeypatch.setattr(api, "get_router", lambda: _FakeRouter())
    with TestClient(app) as c:
        r = c.post("/chat", json={"message": "où en est l'entreprise ?"})
        assert r.status_code == 200
        j = r.json()
        assert j["reply"] == "Voici où en est notre entreprise."
        assert j["model"] == "deepseek-v4-pro"
    # The system message must carry Hermès' identity + the injected live state.
    sysmsg = _FakeRouter.last_messages[0]["content"]
    assert "Hermès" in sysmsg
    assert "Stratégies" in sysmsg or "ENTREPRISE" in sysmsg
    # The user's message is the last turn.
    assert _FakeRouter.last_messages[-1]["content"] == "où en est l'entreprise ?"


@pytest.mark.asyncio
async def test_chat_context_carries_real_per_trade_detail_not_just_aggregates(monkeypatch):
    """Regression guard: the chat once invented 4/6 trades (fake symbols, fake PnL,
    fake reasons) because the context only carried aggregate stats (count/win_rate),
    never the itemised journal. This pins that the real per-trade rows reach the LLM
    context verbatim, so it has no excuse to confabulate. Don't let this regress."""
    await get_db().init()
    journal = TradeJournal()
    tid = await journal.open(
        JournalEntry(
            symbol="ETH/USDT:USDT",
            side="SELL",
            qty=0.005,
            entry_price=1700.0,
            opened_at=datetime.now(UTC),
            skill_id="ts_mom_vol_scaled",
            stage="stage_0_paper",
        )
    )
    await journal.close(
        trade_id=tid, exit_price=1650.0, pnl_usd=1.234, fee_usd=0.01, exit_reason="take_profit"
    )

    monkeypatch.setattr(api, "get_router", lambda: _FakeRouter())
    with TestClient(app) as c:
        r = c.post("/chat", json={"message": "détaille mes trades"})
        assert r.status_code == 200

    sysmsg = _FakeRouter.last_messages[0]["content"]
    # The REAL trade's specifics must be present verbatim — symbol, skill, PnL.
    assert "ETH/USDT:USDT" in sysmsg
    assert "ts_mom_vol_scaled" in sysmsg
    assert "1.234" in sysmsg
    assert "take_profit" in sysmsg
    # The explicit anti-fabrication guardrail must be present in every call.
    assert "n'est pas dans mon contexte actuel" in sysmsg or "JAMAIS l'inventer" in sysmsg


def test_chat_rejects_empty_message(monkeypatch):
    monkeypatch.setattr(api, "get_router", lambda: _FakeRouter())
    with TestClient(app) as c:
        r = c.post("/chat", json={"message": "   "})
        assert r.status_code == 400


def test_chat_page_is_served():
    with TestClient(app) as c:
        r = c.get("/chat")
        assert r.status_code == 200
        assert "Hermès" in r.text
