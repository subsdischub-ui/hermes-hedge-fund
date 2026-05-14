"""Risk engine — final gate before any order leaves the agent.

Inputs: a TradeProposal from the Portfolio Manager + current portfolio state.
Outputs: APPROVED with sized order, or REJECTED with reason.

Hard rules (cannot be overridden by LLM):
  - Refuse trades outside the configured universe.
  - Refuse if equity is below cold-floor (≤ $5).
  - Refuse if open positions ≥ stage max.
  - Refuse if confidence < TRM confidence floor.
  - Refuse if today's realized PnL ≤ -daily_loss_kill_pct of equity.
  - Refuse if computed notional < $5 (Hyperliquid min) or > 50% of equity.
  - Refuse if symbol already has an open position (no pyramiding in v1).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from typing import Literal

from hermes.execution.base import Exchange, Order, Position
from hermes.logging_setup import get_logger
from hermes.risk.sizing import PositionSizer, SizingInputs, SizingOutput
from hermes.settings import get_settings

log = get_logger(__name__)

Decision = Literal["APPROVED", "REJECTED"]


@dataclass
class TradeProposal:
    symbol: str
    side: Literal["BUY", "SELL", "HOLD"]
    confidence: float
    stop_pct: float
    take_profit_pct: float
    thesis: str
    skill_id: str = ""


@dataclass
class RiskDecision:
    decision: Decision
    reason: str
    order: Order | None = None
    sizing: SizingOutput | None = None


class RiskEngine:
    def __init__(self) -> None:
        s = get_settings()
        self.settings = s
        self.universe = set(s.universe)
        self.stage_cfg = s.stage_config
        self.sizer = PositionSizer()
        self.confidence_floor = s.trm_confidence_floor
        sizing_cfg = s.yaml("sizing", default={}) or {}
        self.atr_k = float(sizing_cfg.get("atr_k", 2.0))
        self.floor_pct = float(sizing_cfg.get("per_trade_floor_pct", 0.25))
        self.fractional_kelly = float(sizing_cfg.get("fractional_kelly", 0.20))

    def _hard_reject(self, reason: str) -> RiskDecision:
        log.info("risk_reject", reason=reason)
        return RiskDecision(decision="REJECTED", reason=reason)

    async def evaluate(
        self,
        proposal: TradeProposal,
        exchange: Exchange,
        atr: float,
        price: float,
        journal_stats: dict,
        realized_pnl_today_usd: float = 0.0,
    ) -> RiskDecision:
        if proposal.side == "HOLD":
            return self._hard_reject("HOLD action — nothing to execute")
        if proposal.symbol not in self.universe:
            return self._hard_reject(f"symbol {proposal.symbol} not in universe")
        if proposal.confidence < self.confidence_floor:
            return self._hard_reject(
                f"confidence {proposal.confidence:.2f} < floor {self.confidence_floor:.2f}"
            )

        equity = await exchange.equity_usd()
        if equity < 5.0:
            return self._hard_reject(f"equity ${equity:.2f} below cold-floor")

        # daily kill switch
        kill_pct = float(self.stage_cfg.get("daily_loss_kill_pct", 2.0)) / 100
        if realized_pnl_today_usd / max(equity, 1e-9) <= -kill_pct:
            return self._hard_reject(f"daily loss kill triggered ({realized_pnl_today_usd:.2f})")

        positions = await exchange.open_positions()
        max_open = int(self.stage_cfg.get("max_open_positions", 3))
        if len(positions) >= max_open:
            return self._hard_reject(f"max open positions {max_open} reached")
        if any(p.symbol == proposal.symbol for p in positions):
            return self._hard_reject(f"already have open position on {proposal.symbol}")

        # paper mode: enforce paper venue at stage 0
        if self.stage_cfg.get("venue", "paper") == "paper" and exchange.name != "paper":
            return self._hard_reject("stage_0_paper but exchange is not paper — refusing live trade")

        stage_cap_pct = float(self.stage_cfg.get("per_trade_risk_cap_pct", 0.5))
        if stage_cap_pct == 0.0:
            return self._hard_reject("paper-only stage: no live sizing")

        sizing = self.sizer.size(
            SizingInputs(
                equity_usd=equity,
                confidence=proposal.confidence,
                win_rate=journal_stats.get("win_rate"),
                avg_win_loss_ratio=journal_stats.get("avg_win_loss_ratio"),
                atr=atr,
                price=price,
                trades_in_journal=int(journal_stats.get("count", 0)),
                stage_cap_pct=stage_cap_pct,
                floor_pct=self.floor_pct,
                fractional_kelly=self.fractional_kelly,
            )
        )

        if sizing.notional_usd < 5.0:
            return self._hard_reject(f"notional ${sizing.notional_usd:.2f} < $5 minimum")
        if sizing.notional_usd > 0.5 * equity:
            return self._hard_reject(
                f"notional ${sizing.notional_usd:.2f} > 50% of equity ${equity:.2f}"
            )

        stop = price * (1 - proposal.stop_pct) if proposal.side == "BUY" else price * (1 + proposal.stop_pct)
        tp = price * (1 + proposal.take_profit_pct) if proposal.side == "BUY" else price * (1 - proposal.take_profit_pct)
        order = Order(
            symbol=proposal.symbol,
            side=proposal.side,  # type: ignore[arg-type]
            qty=sizing.qty,
            price=None,
            stop_price=stop,
            take_profit=tp,
            extra={"skill_id": proposal.skill_id, "thesis": proposal.thesis},
        )
        log.info(
            "risk_approve",
            symbol=proposal.symbol,
            side=proposal.side,
            notional=sizing.notional_usd,
            risk_pct=sizing.risk_pct,
        )
        return RiskDecision(decision="APPROVED", reason=sizing.rationale, order=order, sizing=sizing)

    async def check_stops(self, exchange: Exchange) -> list[tuple[Position, str]]:
        """Server-side stop/TP fallback for paper mode (real exchanges enforce these)."""
        if exchange.name != "paper":
            return []
        out: list[tuple[Position, str]] = []
        for p in await exchange.open_positions():
            px = await exchange.last_price(p.symbol)
            if p.qty > 0:
                if p.stop_price and px <= p.stop_price:
                    out.append((p, "stop"))
                elif p.take_profit and px >= p.take_profit:
                    out.append((p, "take_profit"))
            else:
                if p.stop_price and px >= p.stop_price:
                    out.append((p, "stop"))
                elif p.take_profit and px <= p.take_profit:
                    out.append((p, "take_profit"))
        return out

    @staticmethod
    def now() -> datetime:
        return datetime.now(timezone.utc)


@lru_cache(maxsize=1)
def get_risk_engine() -> RiskEngine:
    return RiskEngine()
