"""Operator chat: a read-only conversational endpoint that answers from Hermès'
live state. The LLM is mocked so tests never hit the network."""

from __future__ import annotations

from fastapi.testclient import TestClient

from hermes import api
from hermes.api import app


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
