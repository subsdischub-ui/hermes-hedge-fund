"""Risk-engine behavioral tests using a fake exchange (no network).

Covers:
- Daily-loss kill switch fires when realized PnL crosses the threshold.
- Spot-stage SELL is hard-rejected (no shorting on spot).
- HOLD is hard-rejected.
- Below-floor confidence is rejected.
"""

from __future__ import annotations

import pytest

from hermes.execution.base import Exchange, Fill, Order, Position
from hermes.risk.guardrails import RiskEngine, TradeProposal


class _FakeExchange(Exchange):
    name = "paper"

    def __init__(self, equity: float = 100.0):
        self._equity = equity
        self._positions: list[Position] = []

    async def equity_usd(self) -> float:
        return self._equity

    async def open_positions(self) -> list[Position]:
        return list(self._positions)

    async def place_order(self, order: Order) -> Fill:   # pragma: no cover
        raise NotImplementedError

    async def close_position(self, symbol: str, reason: str):   # pragma: no cover
        return None

    async def last_price(self, symbol: str) -> float:
        return 100.0

    async def close(self) -> None:
        return None


@pytest.mark.asyncio
async def test_daily_kill_switch_fires():
    engine = RiskEngine()
    exch = _FakeExchange(equity=100.0)
    # Spend 6% of equity in realized losses today; stage cap is 5% (stage_0).
    decision = await engine.evaluate(
        proposal=TradeProposal(
            symbol="BTC/USDC", side="BUY", confidence=0.9, stop_pct=0.02,
            take_profit_pct=0.04, thesis="x",
        ),
        exchange=exch,
        atr=1.0,
        price=100.0,
        journal_stats={"count": 0},
        realized_pnl_today_usd=-6.0,
    )
    assert decision.decision == "REJECTED"
    assert "kill" in decision.reason.lower()


@pytest.mark.asyncio
async def test_spot_sell_rejected():
    engine = RiskEngine()
    # The paper stage now runs as perp (shorts allowed, matching the validated
    # strategy); force a SPOT instrument config to exercise the no-short rule itself.
    engine.stage_cfg = {**engine.stage_cfg, "instruments": "spot"}
    exch = _FakeExchange()
    decision = await engine.evaluate(
        proposal=TradeProposal(
            symbol="BTC/USDC", side="SELL", confidence=0.9, stop_pct=0.02,
            take_profit_pct=0.04, thesis="x",
        ),
        exchange=exch,
        atr=1.0,
        price=100.0,
        journal_stats={"count": 0},
        realized_pnl_today_usd=0.0,
    )
    assert decision.decision == "REJECTED"
    assert "spot" in decision.reason.lower() or "sell" in decision.reason.lower()


@pytest.mark.asyncio
async def test_perp_sell_approved():
    """On a perp stage (the paper default now), a SHORT must NOT be blocked by the
    spot no-short rule — Strategy #1 is long/short, so SELL must reach approval."""
    engine = RiskEngine()
    engine.stage_cfg = {**engine.stage_cfg, "instruments": "perp"}
    exch = _FakeExchange()
    decision = await engine.evaluate(
        proposal=TradeProposal(
            symbol="BTC/USDC", side="SELL", confidence=0.9, stop_pct=0.02,
            take_profit_pct=0.04, thesis="mean-reversion short",
        ),
        exchange=exch,
        atr=1.0,
        price=100.0,
        journal_stats={"count": 0},
        realized_pnl_today_usd=0.0,
    )
    assert decision.decision == "APPROVED"
    assert decision.order is not None and decision.order.side == "SELL"


@pytest.mark.asyncio
async def test_hold_rejected():
    engine = RiskEngine()
    exch = _FakeExchange()
    decision = await engine.evaluate(
        proposal=TradeProposal(
            symbol="BTC/USDC", side="HOLD", confidence=0.9, stop_pct=0.02,
            take_profit_pct=0.04, thesis="x",
        ),
        exchange=exch,
        atr=1.0,
        price=100.0,
        journal_stats={"count": 0},
    )
    assert decision.decision == "REJECTED"


@pytest.mark.asyncio
async def test_low_confidence_rejected():
    engine = RiskEngine()
    exch = _FakeExchange()
    decision = await engine.evaluate(
        proposal=TradeProposal(
            symbol="BTC/USDC", side="BUY", confidence=0.10, stop_pct=0.02,
            take_profit_pct=0.04, thesis="x",
        ),
        exchange=exch,
        atr=1.0,
        price=100.0,
        journal_stats={"count": 0},
        realized_pnl_today_usd=0.0,
    )
    assert decision.decision == "REJECTED"
    assert "confidence" in decision.reason.lower()
