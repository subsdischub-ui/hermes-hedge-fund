"""Shared agent-graph state."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class HermesState:
    symbol: str
    timeframe: str = "1h"

    # populated by analyst nodes
    ohlcv: Any = None              # pd.DataFrame
    snapshot: dict[str, Any] = field(default_factory=dict)
    news: list[dict] = field(default_factory=list)
    sentiment: dict[str, Any] = field(default_factory=dict)
    skill_signals: list[dict] = field(default_factory=list)
    relevant_lessons: list[dict] = field(default_factory=list)

    # populated by researcher / trader / risk
    bull_view: str = ""
    bear_view: str = ""
    trader_proposal: dict[str, Any] = field(default_factory=dict)
    risk_decision: dict[str, Any] = field(default_factory=dict)
    trm_refinement: dict[str, Any] = field(default_factory=dict)
    confidence_calibration: dict[str, Any] = field(default_factory=dict)

    # outcome
    executed: bool = False
    trade_id: int | None = None
    error: str | None = None
