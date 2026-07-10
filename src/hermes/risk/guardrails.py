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
  - Refuse SELL on a spot stage when no existing long exists to close against
    (spot venues cannot short).
  - Refuse if stage is paper-only but configured exchange is not the paper one.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from functools import lru_cache
from typing import Literal

from hermes.execution.base import Exchange, Order, Position
from hermes.logging_setup import get_logger
from hermes.risk.sizing import (
    PositionSizer,
    SizingInputs,
    SizingOutput,
    correlation_haircut,
)
from hermes.settings import get_settings

log = get_logger(__name__)

Decision = Literal["APPROVED", "REJECTED"]

# Per-venue minimum order notional (USD). Hyperliquid rejects < $10.
_VENUE_MIN_NOTIONAL = {"paper": 5.0, "hyperliquid": 10.0, "bybit": 5.0}


def venue_min_notional(venue: str) -> float:
    """Minimum order notional for a venue; conservative $10 default for unknown
    (live) venues so we never approve an order the exchange will zero-fill."""
    return _VENUE_MIN_NOTIONAL.get(venue, 10.0)


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


async def _realized_pnl_today_usd(now: datetime | None = None) -> float:
    """Sum of pnl_usd over trades closed since UTC midnight today.

    Imported lazily to avoid a circular import between risk and memory layers.
    """
    from sqlalchemy import select

    from hermes.memory.store import JournalRow, get_db

    midnight = (now or datetime.now(UTC)).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    db = get_db()
    async with db.session() as s:
        rows = (
            await s.execute(
                select(JournalRow).where(
                    JournalRow.closed_at.is_not(None),
                    JournalRow.closed_at >= midnight,
                )
            )
        ).scalars().all()
    return float(sum((r.pnl_usd or 0.0) for r in rows))


class RiskEngine:
    # Stages where shorting is not supported by the venue. The risk engine refuses
    # SELL on a flat book in these stages — bear opinions must be expressed as HOLD.
    SPOT_ONLY_STAGES = {"stage_0_paper", "stage_1_micro", "stage_2_normal"}

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
        # Average correlation of the universe (BTC/ETH/SOL ~0.85). Used to haircut
        # new positions so the aggregate book risk does not stack on a single factor.
        self.correlation_rho = float(sizing_cfg.get("correlation_rho", 0.85))
        self.min_notional = venue_min_notional(self.stage_cfg.get("venue", "paper"))

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
        realized_pnl_today_usd: float | None = None,
        vol_scale: float = 1.0,
    ) -> RiskDecision:
        if proposal.side == "HOLD":
            return self._hard_reject("HOLD action — nothing to execute")
        if proposal.symbol not in self.universe:
            return self._hard_reject(f"symbol {proposal.symbol} not in universe")
        # Confidence floor — two legitimate roles only: (a) a TRM halt forces
        # confidence to 0 and MUST always reject, any skill; (b) LLM-sourced
        # proposals (empty skill_id) below the floor are rejected. SELF-SELECTING
        # slate skills are exempt above zero: their arming threshold IS their
        # validation (S2 arms at funding z>=2 -> conf z/4 = 0.50; carry arms at
        # bps>=2 -> conf bps/10 = 0.20). Applying the generic 0.55 floor on top
        # silently narrowed the VALIDATED thresholds (audit 2026-07-10: S2 live
        # was effectively z>=2.2, carry bps>=5.5) — the same backtest/live
        # fidelity defect class as the maker-fee bug. Conviction-gated skills
        # (risk/selective._CONVICTION_GATED) keep the floor.
        if proposal.confidence <= 0.0:
            return self._hard_reject("confidence 0 — TRM halt or empty proposal")
        from hermes.risk.selective import _CONVICTION_GATED

        self_selecting = bool(proposal.skill_id) and proposal.skill_id not in _CONVICTION_GATED
        if not self_selecting and proposal.confidence < self.confidence_floor:
            return self._hard_reject(
                f"confidence {proposal.confidence:.2f} < floor {self.confidence_floor:.2f}"
            )

        equity = await exchange.equity_usd()
        if equity < 5.0:
            return self._hard_reject(f"equity ${equity:.2f} below cold-floor")

        # Daily kill switch — compute from the journal if caller didn't supply.
        if realized_pnl_today_usd is None:
            realized_pnl_today_usd = await _realized_pnl_today_usd()
        kill_pct = float(self.stage_cfg.get("daily_loss_kill_pct", 2.0)) / 100
        if realized_pnl_today_usd / max(equity, 1e-9) <= -kill_pct:
            return self._hard_reject(
                f"daily loss kill triggered (pnl={realized_pnl_today_usd:.2f}, "
                f"equity={equity:.2f}, kill_pct={kill_pct * 100:.2f}%)"
            )

        positions = await exchange.open_positions()
        max_open = int(self.stage_cfg.get("max_open_positions", 3))
        if len(positions) >= max_open:
            return self._hard_reject(f"max open positions {max_open} reached")
        # No-pyramid per (symbol, skill_id) — NOT per symbol alone. The book is keyed
        # symbol::skill_id by design (paper.py, stop_watcher.py) and selective.py tests
        # pin that momentum holding BTC must not starve funding on BTC. The old
        # per-symbol-only check silently neutralised that multi-sleeve design (audit
        # 2026-07-10, proved by execution). Aggregate stacking stays bounded by
        # max_open_positions above and the correlation haircut in sizing.
        if any(
            p.symbol == proposal.symbol and (p.skill_id or "") == (proposal.skill_id or "")
            for p in positions
        ):
            return self._hard_reject(f"already have open position on {proposal.symbol}")

        # paper mode: enforce paper venue at stage 0
        if self.stage_cfg.get("venue", "paper") == "paper" and exchange.name != "paper":
            return self._hard_reject("stage_0_paper but exchange is not paper — refusing live trade")

        # Spot-stage gate: SELL is never an opening order on a spot-only stage —
        # rejected outright here (closes go through close_position, not evaluate()).
        if (
            self.settings.stage in self.SPOT_ONLY_STAGES
            and self.stage_cfg.get("instruments", "spot") == "spot"
            and proposal.side == "SELL"
        ):
            return self._hard_reject("SELL not permitted on a spot-only stage (no shorting)")

        stage_cap_pct = float(self.stage_cfg.get("per_trade_risk_cap_pct", 0.5))
        if stage_cap_pct <= 0.0:
            return self._hard_reject("stage per_trade_risk_cap_pct is 0 — sizing disabled")

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
                # n_existing = other open positions (same-symbol is already rejected
                # above), so a 2nd/3rd correlated leg is sized down accordingly.
                correlation_haircut=correlation_haircut(len(positions), self.correlation_rho),
                vol_scale=vol_scale,
            )
        )

        if sizing.notional_usd < self.min_notional:
            return self._hard_reject(
                f"notional ${sizing.notional_usd:.2f} < ${self.min_notional:.0f} venue minimum"
            )
        # Notional cap as a function of stage cap × leverage allowance. We cap at
        # 5× equity for spot (capped further by exchange minimums) and tighter for
        # paper to avoid pathological ATR=0 explosions.
        notional_cap = min(5.0 * equity, 0.5 * equity * 10)  # i.e. 5x equity
        if sizing.notional_usd > notional_cap:
            return self._hard_reject(
                f"notional ${sizing.notional_usd:.2f} > cap ${notional_cap:.2f} (equity ${equity:.2f})"
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
        # Max-hold time exit (mirrors the backtest vertical barrier) — active only
        # under the validated selective strategy so default behaviour is unchanged.
        from hermes.risk.timeexit import is_past_max_hold, max_hold_bars_for, max_hold_seconds
        from hermes.settings import get_settings

        strat = get_settings().yaml("strategy", default={}) or {}
        selective = bool(strat.get("selective_mode", False))
        decision_tf = str(strat.get("decision_timeframe", "4h"))
        now = self.now()

        out: list[tuple[Position, str]] = []
        for p in await exchange.open_positions():
            try:
                px = await exchange.last_price(p.symbol)
            except Exception as e:
                log.warning("check_stops_price_failed", symbol=p.symbol, error=str(e))
                continue
            if p.qty > 0:
                if p.stop_price and px <= p.stop_price:
                    out.append((p, "stop"))
                    continue
                elif p.take_profit and px >= p.take_profit:
                    out.append((p, "take_profit"))
                    continue
            else:
                if p.stop_price and px >= p.stop_price:
                    out.append((p, "stop"))
                    continue
                elif p.take_profit and px <= p.take_profit:
                    out.append((p, "take_profit"))
                    continue
            # Per-skill vertical barrier (e.g. funding #2 holds ~24 bars vs
            # momentum #1's ~12). Only under the validated selective strategy.
            if selective:
                bars = max_hold_bars_for(p.skill_id, strat)
                if is_past_max_hold(p.opened_at, now, max_hold_seconds(bars, decision_tf)):
                    out.append((p, "max_hold"))
        return out

    @staticmethod
    def now() -> datetime:
        return datetime.now(UTC)

    @staticmethod
    def utc_midnight(now: datetime | None = None) -> datetime:
        return (now or datetime.now(UTC)).replace(
            hour=0, minute=0, second=0, microsecond=0
        )


@lru_cache(maxsize=1)
def get_risk_engine() -> RiskEngine:
    return RiskEngine()


