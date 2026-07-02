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
# EMPTY since the 2026-07-02 restructuration: ts_mom_vol_scaled (the only conviction-
# gated skill) was retired from the slate (bull-beta drag — see config allowed_skills
# comment). The mechanism stays wired for any future conviction-scored skill.
_CONVICTION_GATED: set[str] = set()


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
    open_keys: set[tuple[str, str]],
    cfg: SelectiveConfig,
) -> tuple[bool, str]:
    """Return ``(allowed, reason)`` for a proposed trade under the selective rule.

    No-op (always allowed) when disabled or the side is HOLD. Otherwise the
    proposal must (a) use a skill in the selective slate, (b) for conviction-gated
    skills, be backed by a raw signal strength >= conviction_min, and (c) not
    pyramid onto a (symbol, skill) it is ALREADY holding.

    ``open_keys`` is the set of ``(symbol, skill_id)`` currently open. No-pyramid is
    PER (symbol, skill): Strategy #1 (momentum) holding BTC does NOT block Strategy
    #2 (funding) from also opening BTC — the two run as concurrent, independent
    sleeves (each capped to one position per symbol).
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

    if cfg.no_pyramid and (symbol, skill_id) in open_keys:
        return False, "already holding this symbol via this skill (no-pyramid / one position per sleeve)"

    return True, "passed selective gate"


def _as_proposal(s: dict, conf: float, symbol: str) -> dict:
    return {
        "symbol": symbol,
        "side": s["side"],
        "confidence": conf,
        "stop_pct": float(s.get("stop_pct", 0.02)),
        "take_profit_pct": float(s.get("take_profit_pct", 0.04)),
        "skill_id": s.get("skill_id", ""),
        "thesis": "selective deterministic signal (validated rule drives; LLM advisory)",
    }


def rank_selective_signals(
    skill_signals: Iterable[dict], cfg: SelectiveConfig, symbol: str
) -> list[dict]:
    """All qualifying deterministic proposals for this symbol, highest conviction
    first. 'Qualifying' = non-HOLD, in the slate, and (for conviction-gated skills)
    clearing the conviction floor. The caller drops any whose (symbol, skill) is
    already held (no-pyramid) and takes the first survivor, so a held momentum
    position does NOT waste the tick — the funding sleeve can still be selected.
    """
    if not cfg.enabled:
        return []
    cands: list[tuple[float, dict]] = []
    for s in skill_signals:
        sid = s.get("skill_id", "")
        if s.get("side", "HOLD") == "HOLD":
            continue
        if cfg.allowed_skills and sid not in cfg.allowed_skills:
            continue
        conf = float(s.get("confidence_raw", s.get("confidence", 0.0)))
        if sid in _CONVICTION_GATED and conf < cfg.conviction_min:
            continue
        cands.append((conf, s))
    cands.sort(key=lambda x: -x[0])
    return [_as_proposal(s, conf, symbol) for conf, s in cands]


def pick_selective_signal(
    skill_signals: Iterable[dict], cfg: SelectiveConfig, symbol: str
) -> dict | None:
    """The single highest-conviction qualifying proposal (or None). Thin wrapper
    over :func:`rank_selective_signals` for callers that don't do per-sleeve
    no-pyramid filtering."""
    ranked = rank_selective_signals(skill_signals, cfg, symbol)
    return ranked[0] if ranked else None


def drive_signal(
    skill_signals: Iterable[dict],
    cfg: SelectiveConfig,
    symbol: str,
    open_keys: set[tuple[str, str]],
) -> dict | None:
    """The deterministic proposal that should DRIVE this tick, or None.

    Gate-aware: returns the highest-conviction qualifying signal whose
    ``(symbol, skill_id)`` is NOT already held under no-pyramid. This is what makes
    two sleeves coexist — if Strategy #1 (momentum) already holds the symbol, it is
    skipped here so Strategy #2 (funding) can still be driven on the same tick
    instead of the tick being wasted on a signal that the gate will reject.
    """
    for cand in rank_selective_signals(skill_signals, cfg, symbol):
        if cfg.no_pyramid and (symbol, cand["skill_id"]) in open_keys:
            continue
        return cand
    return None
