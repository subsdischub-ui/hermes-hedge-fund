"""Risk-engine behavioral tests using a fake exchange (no network).

Covers:
- Daily-loss kill switch fires when realized PnL crosses the threshold.
- Spot-stage SELL is hard-rejected (no shorting on spot).
- HOLD is hard-rejected.
- Below-floor confidence is rejected.
- Macro blackout gate: shadow at stage_0_paper (never blocks, verdict exposed),
  hard-reject at real stages, no false block outside the window.
- Rolling drawdown circuit-breaker: shadow at stage_0_paper, RED/GREEN mutation at
  real stages (mirrors test_daily_kill_switch_fires' pattern).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from hermes.execution.base import Exchange, Fill, Order, Position
from hermes.risk.guardrails import RiskEngine, TradeProposal
from hermes.risk.macro_blackout import MacroEvent


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
            symbol="BTC/USDT:USDT", side="BUY", confidence=0.9, stop_pct=0.02,
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
            symbol="BTC/USDT:USDT", side="SELL", confidence=0.9, stop_pct=0.02,
            take_profit_pct=0.04, thesis="x",
        ),
        exchange=exch,
        atr=1.0,
        price=100.0,
        journal_stats={"count": 0},
        realized_pnl_today_usd=0.0, realized_pnl_window_usd=0.0,
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
            symbol="BTC/USDT:USDT", side="SELL", confidence=0.9, stop_pct=0.02,
            take_profit_pct=0.04, thesis="mean-reversion short",
        ),
        exchange=exch,
        atr=1.0,
        price=100.0,
        journal_stats={"count": 0},
        realized_pnl_today_usd=0.0, realized_pnl_window_usd=0.0,
    )
    assert decision.decision == "APPROVED"
    assert decision.order is not None and decision.order.side == "SELL"


@pytest.mark.asyncio
async def test_hold_rejected():
    engine = RiskEngine()
    exch = _FakeExchange()
    decision = await engine.evaluate(
        proposal=TradeProposal(
            symbol="BTC/USDT:USDT", side="HOLD", confidence=0.9, stop_pct=0.02,
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
            symbol="BTC/USDT:USDT", side="BUY", confidence=0.10, stop_pct=0.02,
            take_profit_pct=0.04, thesis="x",
        ),
        exchange=exch,
        atr=1.0,
        price=100.0,
        journal_stats={"count": 0},
        realized_pnl_today_usd=0.0, realized_pnl_window_usd=0.0,
    )
    assert decision.decision == "REJECTED"
    assert "confidence" in decision.reason.lower()


def _pos(symbol: str, skill_id: str) -> Position:
    from datetime import UTC, datetime

    return Position(
        symbol=symbol, qty=0.001, entry_price=100.0, stop_price=98.0,
        take_profit=104.0, opened_at=datetime.now(UTC), skill_id=skill_id,
    )


@pytest.mark.asyncio
async def test_no_pyramid_is_per_symbol_and_skill_not_per_symbol_alone():
    """RÉGRESSION (audit sûreté 2026-07-10, défaut prouvé par exécution) : le no-pyramid
    de guardrails rejetait TOUTE seconde position sur un symbole déjà tenu, tous skills
    confondus — neutralisant le design multi-sleeves que selective.py implémente et teste
    (« momentum holding BTC must NOT prevent funding from being driven on BTC »). La clé
    du book est (symbol, skill_id) : un skill DIFFÉRENT doit pouvoir trader le symbole."""
    engine = RiskEngine()
    exch = _FakeExchange(equity=100.0)
    exch._positions = [_pos("BTC/USDT:USDT", "mean_reversion_regime")]
    decision = await engine.evaluate(
        proposal=TradeProposal(
            symbol="BTC/USDT:USDT", side="BUY", confidence=0.9, stop_pct=0.02,
            take_profit_pct=0.04, thesis="funding squeeze armé",
            skill_id="funding_squeeze_continuation",
        ),
        exchange=exch,
        atr=1.0,
        price=100.0,
        journal_stats={"count": 0},
        realized_pnl_today_usd=0.0, realized_pnl_window_usd=0.0,
    )
    assert "already have open position" not in decision.reason


@pytest.mark.asyncio
async def test_no_pyramid_still_blocks_same_skill_same_symbol():
    """Le pyramiding au sens strict (même skill, même symbole) reste interdit."""
    engine = RiskEngine()
    exch = _FakeExchange(equity=100.0)
    exch._positions = [_pos("BTC/USDT:USDT", "funding_squeeze_continuation")]
    decision = await engine.evaluate(
        proposal=TradeProposal(
            symbol="BTC/USDT:USDT", side="BUY", confidence=0.9, stop_pct=0.02,
            take_profit_pct=0.04, thesis="re-entry",
            skill_id="funding_squeeze_continuation",
        ),
        exchange=exch,
        atr=1.0,
        price=100.0,
        journal_stats={"count": 0},
        realized_pnl_today_usd=0.0, realized_pnl_window_usd=0.0,
    )
    assert decision.decision == "REJECTED"
    assert "already have open position" in decision.reason


@pytest.mark.asyncio
async def test_confidence_floor_does_not_strangle_self_selecting_slate_skills():
    """RÉGRESSION (audit 2026-07-10) : le plancher de confiance (trm_confidence_floor
    0.55) rejetait les signaux fraîchement armés des sleeves auto-sélectives — S2 à
    z∈[2.0,2.2) (conf=z/4 ∈ [0.50,0.55)) et le carry à bps∈[2,5.5) (conf=bps/10) —
    rétrécissant silencieusement les seuils VALIDÉS du backtest. Les skills
    auto-sélectifs (hors _CONVICTION_GATED) portent leur validation dans leur seuil
    d'armement : le plancher générique ne doit pas s'y superposer."""
    engine = RiskEngine()
    exch = _FakeExchange(equity=100.0)
    # S2 à z=2.0 exactement -> conf 0.50
    d1 = await engine.evaluate(
        proposal=TradeProposal(
            symbol="BTC/USDT:USDT", side="BUY", confidence=0.50, stop_pct=0.02,
            take_profit_pct=0.04, thesis="funding z=2.0",
            skill_id="funding_squeeze_continuation",
        ),
        exchange=exch, atr=1.0, price=100.0,
        journal_stats={"count": 0}, realized_pnl_today_usd=0.0, realized_pnl_window_usd=0.0,
    )
    assert d1.decision == "APPROVED", d1.reason  # passait: rejeté "confidence 0.50 < floor 0.55"
    # carry à bps=2.0 -> conf 0.20
    d2 = await engine.evaluate(
        proposal=TradeProposal(
            symbol="ETH/USDT:USDT", side="BUY", confidence=0.20, stop_pct=0.05,
            take_profit_pct=0.10, thesis="carry bps=2.0",
            skill_id="funding_carry_harvest",
        ),
        exchange=exch, atr=1.0, price=100.0,
        journal_stats={"count": 0}, realized_pnl_today_usd=0.0, realized_pnl_window_usd=0.0,
    )
    assert d2.decision == "APPROVED", d2.reason  # passait: rejeté "confidence 0.20 < floor 0.55"


@pytest.mark.asyncio
async def test_confidence_floor_still_rejects_trm_halt_and_llm_proposals():
    """Le plancher garde ses deux rôles légitimes : (a) la halte TRM (confiance forcée
    à 0) rejette TOUJOURS, même un skill auto-sélectif ; (b) une proposition d'origine
    LLM (skill_id vide) sous le plancher reste rejetée."""
    engine = RiskEngine()
    exch = _FakeExchange(equity=100.0)
    d_halt = await engine.evaluate(
        proposal=TradeProposal(
            symbol="BTC/USDT:USDT", side="BUY", confidence=0.0, stop_pct=0.02,
            take_profit_pct=0.04, thesis="TRM halt",
            skill_id="funding_squeeze_continuation",
        ),
        exchange=exch, atr=1.0, price=100.0,
        journal_stats={"count": 0}, realized_pnl_today_usd=0.0, realized_pnl_window_usd=0.0,
    )
    assert d_halt.decision == "REJECTED"
    d_llm = await engine.evaluate(
        proposal=TradeProposal(
            symbol="BTC/USDT:USDT", side="BUY", confidence=0.30, stop_pct=0.02,
            take_profit_pct=0.04, thesis="intuition LLM", skill_id="",
        ),
        exchange=exch, atr=1.0, price=100.0,
        journal_stats={"count": 0}, realized_pnl_today_usd=0.0, realized_pnl_window_usd=0.0,
    )
    assert d_llm.decision == "REJECTED"
    assert "floor" in d_llm.reason.lower()


def _fixed_calendar(event_dt: datetime, event_type: str = "fomc_decision") -> dict:
    return {"events": [MacroEvent(dt_utc=event_dt, event_type=event_type, label="FOMC decision (test)")]}


def _buy_proposal(**overrides) -> TradeProposal:
    kwargs = dict(
        symbol="BTC/USDT:USDT", side="BUY", confidence=0.9, stop_pct=0.02,
        take_profit_pct=0.04, thesis="x",
    )
    kwargs.update(overrides)
    return TradeProposal(**kwargs)


# --------------------------------------------------------------------------- #
# Macro blackout gate
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_macro_blackout_shadow_at_stage_0_never_blocks_but_exposes_verdict(monkeypatch):
    """stage_0_paper is deliberately ABSENT from macro_blackout.enforce_stages
    (fidelity decision: the backtests that validated the live strategies include
    news periods). The gate must still COMPUTE + EXPOSE the verdict on the
    RiskDecision (visibility rule) — it must simply never change the decision."""
    import hermes.risk.guardrails as guardrails_mod

    event_dt = datetime(2026, 3, 18, 18, 0, tzinfo=UTC)
    monkeypatch.setattr(guardrails_mod, "get_macro_calendar", lambda: _fixed_calendar(event_dt))
    engine = RiskEngine()
    assert engine.stage == "stage_0_paper"  # sanity: default test stage is shadow-only
    engine.now = lambda: event_dt  # exactly at the event -> inside the window
    exch = _FakeExchange(equity=100.0)
    decision = await engine.evaluate(
        proposal=_buy_proposal(),
        exchange=exch, atr=1.0, price=100.0, journal_stats={"count": 0},
        realized_pnl_today_usd=0.0, realized_pnl_window_usd=0.0,
    )
    assert decision.decision == "APPROVED"  # shadow: never blocks at stage_0
    assert decision.shadow["macro_blackout"]["blocked"] is True
    assert decision.shadow["macro_blackout"]["event"] == "FOMC decision (test)"


@pytest.mark.asyncio
async def test_macro_blackout_hard_rejects_at_real_stage(monkeypatch):
    """monkeypatch the stage (per the mission's own guidance) to exercise the
    real-stage ENFORCE path without needing an actual live-money stage in tests."""
    import hermes.risk.guardrails as guardrails_mod

    event_dt = datetime(2026, 3, 18, 18, 0, tzinfo=UTC)
    monkeypatch.setattr(guardrails_mod, "get_macro_calendar", lambda: _fixed_calendar(event_dt))
    engine = RiskEngine()
    engine.stage = "stage_1_micro"  # real stage -> macro_blackout.enforce_stages hits
    engine.now = lambda: event_dt
    exch = _FakeExchange(equity=100.0)
    decision = await engine.evaluate(
        proposal=_buy_proposal(),
        exchange=exch, atr=1.0, price=100.0, journal_stats={"count": 0},
        realized_pnl_today_usd=0.0, realized_pnl_window_usd=0.0,
    )
    assert decision.decision == "REJECTED"
    assert "macro blackout" in decision.reason.lower()
    assert decision.shadow["macro_blackout"]["blocked"] is True


@pytest.mark.asyncio
async def test_macro_blackout_outside_window_never_blocks_even_at_real_stage(monkeypatch):
    import hermes.risk.guardrails as guardrails_mod

    event_dt = datetime(2026, 3, 18, 18, 0, tzinfo=UTC)
    monkeypatch.setattr(guardrails_mod, "get_macro_calendar", lambda: _fixed_calendar(event_dt))
    engine = RiskEngine()
    engine.stage = "stage_1_micro"
    engine.now = lambda: event_dt - timedelta(hours=5)  # well outside the ±2h window
    exch = _FakeExchange(equity=100.0)
    decision = await engine.evaluate(
        proposal=_buy_proposal(),
        exchange=exch, atr=1.0, price=100.0, journal_stats={"count": 0},
        realized_pnl_today_usd=0.0, realized_pnl_window_usd=0.0,
    )
    assert decision.decision == "APPROVED"
    assert decision.shadow["macro_blackout"]["blocked"] is False


# --------------------------------------------------------------------------- #
# Rolling drawdown circuit-breaker
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_rolling_drawdown_kill_shadow_at_stage_0_never_blocks_but_exposes_verdict():
    engine = RiskEngine()
    assert engine.stage == "stage_0_paper"
    engine.macro_blackout_cfg = {**engine.macro_blackout_cfg, "enabled": False}
    exch = _FakeExchange(equity=100.0)
    # -11% over the 30d window (config kill_pct default 10.0) -> RED condition
    # present, but stage_0 is shadow-only: must NOT block, must expose blocked=True.
    decision = await engine.evaluate(
        proposal=_buy_proposal(),
        exchange=exch, atr=1.0, price=100.0, journal_stats={"count": 0},
        realized_pnl_today_usd=0.0, realized_pnl_window_usd=-11.0,
    )
    assert decision.decision == "APPROVED"
    assert decision.shadow["rolling_drawdown_kill"]["blocked"] is True
    assert decision.shadow["rolling_drawdown_kill"]["window_days"] == 30


@pytest.mark.asyncio
async def test_rolling_drawdown_kill_fires_red_at_real_stage():
    """RED: a rolling loss beyond kill_pct at an enforce stage must hard-reject —
    mutation-proof mirror of test_daily_kill_switch_fires."""
    engine = RiskEngine()
    engine.stage = "stage_1_micro"
    engine.macro_blackout_cfg = {**engine.macro_blackout_cfg, "enabled": False}
    exch = _FakeExchange(equity=100.0)
    decision = await engine.evaluate(
        proposal=_buy_proposal(),
        exchange=exch, atr=1.0, price=100.0, journal_stats={"count": 0},
        realized_pnl_today_usd=0.0, realized_pnl_window_usd=-11.0,
    )
    assert decision.decision == "REJECTED"
    assert "rolling drawdown kill" in decision.reason.lower()


@pytest.mark.asyncio
async def test_rolling_drawdown_kill_green_below_threshold_at_real_stage():
    """GREEN: restore the loss to just below kill_pct -> must NOT reject, proving
    the RED case above is really the threshold biting, not an always-on gate."""
    engine = RiskEngine()
    engine.stage = "stage_1_micro"
    engine.macro_blackout_cfg = {**engine.macro_blackout_cfg, "enabled": False}
    exch = _FakeExchange(equity=100.0)
    decision = await engine.evaluate(
        proposal=_buy_proposal(),
        exchange=exch, atr=1.0, price=100.0, journal_stats={"count": 0},
        realized_pnl_today_usd=0.0, realized_pnl_window_usd=-5.0,  # -5% < 10% kill_pct
    )
    assert decision.decision == "APPROVED"
    assert decision.shadow["rolling_drawdown_kill"]["blocked"] is False


@pytest.mark.asyncio
async def test_rolling_drawdown_kill_exact_threshold_is_inclusive():
    """Exact boundary: pnl_window/equity == -kill_pct must block (>=, not >)."""
    engine = RiskEngine()
    engine.stage = "stage_1_micro"
    engine.macro_blackout_cfg = {**engine.macro_blackout_cfg, "enabled": False}
    exch = _FakeExchange(equity=100.0)
    decision = await engine.evaluate(
        proposal=_buy_proposal(),
        exchange=exch, atr=1.0, price=100.0, journal_stats={"count": 0},
        realized_pnl_today_usd=0.0, realized_pnl_window_usd=-10.0,  # exactly -10%
    )
    assert decision.decision == "REJECTED"
    assert "rolling drawdown kill" in decision.reason.lower()
