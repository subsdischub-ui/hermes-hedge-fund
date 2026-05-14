"""Position sizing: fractional Kelly + ATR-normalized risk."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SizingInputs:
    equity_usd: float
    confidence: float           # 0..1 from PM agent (or TRM)
    win_rate: float | None      # historical; None → assume 0.5
    avg_win_loss_ratio: float | None  # b in Kelly
    atr: float                  # ATR (same unit as price)
    price: float
    trades_in_journal: int      # used to gate Kelly usage
    stage_cap_pct: float        # cap from stage config (e.g. 0.5%)
    floor_pct: float = 0.25     # 0.25%
    fractional_kelly: float = 0.20


@dataclass
class SizingOutput:
    risk_pct: float             # of equity, e.g. 0.005 = 0.5%
    notional_usd: float
    qty: float
    stop_distance_pct: float    # ATR-normalized stop distance (k*ATR/price)
    rationale: str


class PositionSizer:
    MIN_TRADES_FOR_KELLY = 100
    ATR_K = 2.0

    def size(self, inp: SizingInputs) -> SizingOutput:
        stop_dist_pct = max(1e-4, self.ATR_K * inp.atr / inp.price)

        # 1) Kelly contribution (only after enough trades)
        if (
            inp.trades_in_journal >= self.MIN_TRADES_FOR_KELLY
            and inp.win_rate is not None
            and inp.avg_win_loss_ratio is not None
            and inp.avg_win_loss_ratio > 0
        ):
            p, q, b = inp.win_rate, 1 - inp.win_rate, inp.avg_win_loss_ratio
            kelly = max(0.0, (p * b - q) / b)
        else:
            kelly = 0.0

        # 2) Combine: fractional Kelly * confidence, with floor & stage cap.
        floor_frac = inp.floor_pct / 100
        stage_cap_frac = inp.stage_cap_pct / 100
        confidence = max(0.0, min(1.0, inp.confidence))

        if kelly > 0:
            raw = inp.fractional_kelly * kelly * confidence
        else:
            # Cold-start: confidence-scaled floor
            raw = floor_frac + (stage_cap_frac - floor_frac) * confidence * 0.5

        risk_pct = max(floor_frac, min(stage_cap_frac, raw))

        # 3) Convert risk to position size: risk_usd = risk_pct * equity,
        #    notional = risk_usd / stop_dist_pct (so a stop-out loses risk_usd).
        risk_usd = risk_pct * inp.equity_usd
        notional = risk_usd / stop_dist_pct
        qty = notional / inp.price

        rationale = (
            f"risk={risk_pct * 100:.3f}% (kelly={kelly:.3f}, conf={confidence:.2f}, "
            f"cap={inp.stage_cap_pct}%, floor={inp.floor_pct}%) | "
            f"stop_dist={stop_dist_pct * 100:.2f}% (2*ATR/price) | "
            f"notional=${notional:.2f}"
        )
        return SizingOutput(
            risk_pct=risk_pct,
            notional_usd=notional,
            qty=qty,
            stop_distance_pct=stop_dist_pct,
            rationale=rationale,
        )
