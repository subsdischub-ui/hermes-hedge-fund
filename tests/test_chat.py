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


@pytest.mark.asyncio
async def test_chat_context_carries_restructuration_facts():
    """The chat must be able to answer 'quel est ton slate et pourquoi S1 a été
    retiré ?', 'où en est la collecte or/argent/top_lsr ?', 'c'est quoi le gate
    maintenant ?' — so the compact restructuration block (slate, retrait S1 +
    chiffres, gate, échéances datées) must reach the LLM context verbatim."""
    from hermes.api import _chat_context

    ctx = await _chat_context()
    # Current slate + the retirement, with its numbers (not just a bare claim).
    assert "mean_reversion_regime" in ctx
    assert "funding_squeeze_continuation" in ctx
    assert "ts_mom_vol_scaled" in ctx
    assert "0.146" in ctx  # LOO-Δ Sharpe
    assert "0.507" in ctx  # joint-null p (post-fix look-ahead 2026-07-02)
    # The gate, so it can answer "c'est quoi le gate maintenant ?"
    assert "edge_paper_trades" in ctx or "40" in ctx
    # Dated deadlines for metals/FX/global cutoff.
    assert "2027-07-02" in ctx
    assert "2028-01-02" in ctx


@pytest.mark.asyncio
async def test_chat_context_survives_shadow_gauges(monkeypatch):
    """Régression du 500 prod du 2026-07-02 : depuis le retrait de S1, le pulse émet des
    jauges SHADOW ({shadow, skill, side, conf, armed} — sans sleeve/metric/val/thr) pour
    les skills hors slate. Le contexte chat doit les rendre proprement, jamais crasher."""
    prod_shaped_pulse = {
        "kind": "pulse",
        "ts": "2026-07-02T21:00:00+00:00",
        "symbol": "BTC/USDT:USDT",
        "regime": "range",
        "gauges": [
            {"sleeve": "Funding #2", "metric": "funding z", "val": 0.4, "thr": 2.0, "armed": False},
            # la forme exacte qui a fait crasher la prod (KeyError 'sleeve'):
            {"shadow": True, "skill": "ts_mom_vol_scaled", "side": "BUY", "conf": 0.37, "armed": True},
            {},  # jauge dégénérée: doit être ignorée sans exception
        ],
    }
    monkeypatch.setattr(api, "_recent_activity", lambda s: [prod_shaped_pulse])
    ctx = await api._chat_context()
    assert "shadow hors-slate" in ctx and "ts_mom_vol_scaled" in ctx
    assert "Funding #2" in ctx


def test_chat_returns_200_even_if_context_builder_raises(monkeypatch):
    """Défense en profondeur : un bug du contexte doit dégrader la réponse, jamais 500."""
    async def _boom():
        raise RuntimeError("bug d'affichage simulé")

    monkeypatch.setattr(api, "_chat_context", _boom)
    monkeypatch.setattr(api, "get_router", lambda: _FakeRouter())
    with TestClient(app) as c:
        r = c.post("/chat", json={"message": "situation ?"})
        assert r.status_code == 200
        sys_msg = _FakeRouter.last_messages[0]["content"]
        assert "état live indisponible" in sys_msg
