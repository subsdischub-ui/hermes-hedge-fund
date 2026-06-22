"""Maturity-ladder stage manager.

Reads stage gates from config/default.yaml:stages.*.advance_when and decides
whether to recommend advancement. Advancement itself is NOT automatic — the
operator must set HERMES_STAGE manually after reviewing the recommendation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from hermes.settings import get_settings

STAGES_ORDER = ["stage_0_paper", "stage_1_micro", "stage_2_normal", "stage_3_perps"]


@dataclass
class StageReport:
    current: str
    next_stage: str | None
    gates: dict[str, tuple[Any, Any, bool]]  # name -> (target, observed, passed)
    can_advance: bool
    summary: str


class StageManager:
    def __init__(self) -> None:
        s = get_settings()
        self.cfg = s.yaml("stages", default={}) or {}
        self.current = s.stage

    def next_stage(self) -> str | None:
        try:
            i = STAGES_ORDER.index(self.current)
        except ValueError:
            return None
        return STAGES_ORDER[i + 1] if i + 1 < len(STAGES_ORDER) else None

    def evaluate(self, metrics: dict[str, Any]) -> StageReport:
        nxt = self.next_stage()
        gates: dict[str, tuple[Any, Any, bool]] = {}
        if nxt is None:
            return StageReport(self.current, None, gates, False, "Already at max stage.")

        rules = self.cfg.get(self.current, {}).get("advance_when", {}) or {}
        for name, target in rules.items():
            observed = metrics.get(name)
            passed = _gate_pass(name, target, observed)
            gates[name] = (target, observed, passed)

        can = bool(gates) and all(p for _, _, p in gates.values())
        summary = (
            f"{'READY' if can else 'BLOCKED'}: {self.current} → {nxt}; "
            + ", ".join(f"{k}={v[1]}/{v[0]}{'✓' if v[2] else '✗'}" for k, v in gates.items())
        )
        return StageReport(self.current, nxt, gates, can, summary)


def _gate_pass(name: str, target: Any, observed: Any) -> bool:
    if observed is None:
        return False
    # Conventions: max_drawdown_pct + rule_violations are upper bounds; everything else is a lower bound.
    upper_bounds = {"max_drawdown_pct", "rule_violations"}
    try:
        if name in upper_bounds:
            return float(observed) <= float(target)
        return float(observed) >= float(target)
    except (TypeError, ValueError):
        return False
