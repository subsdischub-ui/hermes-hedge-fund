"""Selective-mode gate — enforces the OOS-validated trading rule in the live path.

The ~9-year walk-forward showed the engine's edge is real but thin, and only
survives net of fees when you (1) trade ONLY the high-conviction momentum tail
(ts_mom |score|>=1.0) plus range mean-reversion, (2) consolidate to ONE position
per move (no pyramiding), and (3) drop the skills that lose even gross
(breakout, pairs). This module is the deterministic guard that imposes (1)-(3)
on whatever the LLM trader proposes.

It is a pure decision function (easy to unit-test) and is wired into the
risk-manager BEFORE the sizing engine. Controlled by `strategy.selective_mode`
in config — DEFAULT OFF, so flipping it on is an explicit, reviewable change.
Maker-order execution (lower fees) is a separate execution-layer concern measured
on testnet, not here.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

# Skills whose conviction (signal strength) is meaningful and must clear the floor.
# Skills NOT listed here but in the slate (e.g. mean_reversion_regime) self-select
# at extremes with a fixed confidence and are exempt from the conviction floor.
_CONVICTION_GATED = {"ts_mom_vol_scaled"}


@dataclass(frozen=True)
class SelectiveConfig:
    enabled: bool
    allowed_skills: tuple[str, ...]
    conviction_min: float
    no_pyramid: bool


def load_selective_config(settings) -> SelectiveConfig:
    """Build the config from the `strategy:` YAML block (all keys optional)."""
    s = settings.yaml("strategy", default={}) or {}
    return SelectiveConfig(
        enabled=bool(s.get("selective_mode", False)),
        allowed_skills=tuple(s.get("allowed_skills", []) or []),
        conviction_min=float(s.get("conviction_min", 1.0)),
        no_pyramid=bool(s.get("no_pyramid", True)),
    )


def selective_gate(
    side: str,
    symbol: str,
    skill_id: str,
    skill_signals: Iterable[dict],
    open_symbols: set[str],
    cfg: SelectiveConfig,
) -> tuple[bool, str]:
    """Return ``(allowed, reason)`` for a proposed trade under the selective rule.

    No-op (always allowed) when disabled or the side is HOLD. Otherwise the
    proposal must (a) use a skill in the selective slate, (b) for conviction-gated
    skills, be backed by a raw signal strength >= conviction_min, and (c) not
    pyramid onto a symbol already holding a position.
    """
    if not cfg.enabled or side == "HOLD":
        return True, "selective_mode off" if not cfg.enabled else "hold"

    if cfg.allowed_skills and skill_id not in cfg.allowed_skills:
        return False, f"skill '{skill_id}' not in selective slate {cfg.allowed_skills}"

    if skill_id in _CONVICTION_GATED:
        conf = max(
            (
                float(s.get("confidence_raw", s.get("confidence", 0.0)))
                for s in skill_signals
                if s.get("skill_id") == skill_id
            ),
            default=0.0,
        )
        if conf < cfg.conviction_min:
            return False, f"conviction {conf:.2f} < {cfg.conviction_min} (low-conviction tail)"

    if cfg.no_pyramid and symbol in open_symbols:
        return False, "already holding a position (no-pyramid / one position per move)"

    return True, "passed selective gate"
