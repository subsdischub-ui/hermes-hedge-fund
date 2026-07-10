"""L2 regime router + L3 inverse-vol blend + skill selection (combination layer).

Thesis (DeMiguel-Garlappi-Uppal): at low data, 1/N and inverse-vol beat
covariance-optimised allocators out of sample. So the default combination is
deliberately trivial — inverse-vol over recent net returns, gated by a
hysteresis regime router that prevents flip-flop churn (and its fees).

All functions are pure and stateless. No look-ahead by construction: every
input (`returns_by_skill`, `efficiency_ratio`, `prev_state`) is a value already
realised at the decision bar — these functions never read future data and never
re-index a series, so there is nothing to shift. Parameters (enter=0.4,
exit=0.3) are frozen a priori at the round values from the design spec; nothing
here is fitted.
"""

from __future__ import annotations

from collections.abc import Sequence
from statistics import variance

__all__ = [
    "annotate_active_and_blend",
    "blend_live_shadow",
    "inverse_vol_weights",
    "route_regime",
    "select_active_skills",
]

# The LIVE crypto slate → regime buckets. Must stay in sync with
# config/default.yaml:strategy.allowed_skills (pinned by
# test_combination_slate_matches_live_allowed_skills, so it can't silently drift
# again). RESTRUCTURATION 2026-07-02: ts_mom_vol_scaled retired (bull-beta drag,
# book LOO-Δ −0.146, joint-null p=0.69 — docs/RESTRUCTURATION_2026-07.md §3), so the
# trend bucket is EMPTY: in a trend regime only ALWAYS_ON skills are active.
# mean_reversion_regime (range) stays; Strategy #2 funding_squeeze_continuation and
# #3 funding_carry_harvest are regime-INDEPENDENT (they arm on the perp funding
# rate / are delta-neutral), so they live in ALWAYS_ON. Previously retired families:
# breakout_vol_expansion, pairs_eth_btc, intraday_tod_btc.
TREND_SKILLS: list[str] = []
RANGE_SKILLS = ["mean_reversion_regime"]
# funding_carry_harvest SUSPENDU du slate le 2026-07-11 (LOO-Δ −0.081, edge taker
# mince, maker jamais mesuré — réveil : fill-rate maker réel au stage_1).
ALWAYS_ON = ["funding_squeeze_continuation"]


def _equal_weights(keys: Sequence[str]) -> dict[str, float]:
    """Uniform 1/N over `keys`; the robust fallback when no usable vol exists."""
    n = len(keys)
    if n == 0:
        return {}
    w = 1.0 / n
    return {k: w for k in keys}


def inverse_vol_weights(returns_by_skill: dict[str, list[float]]) -> dict[str, float]:
    """Weight each skill ∝ 1/variance(returns) (ddof=1), normalised to sum 1.

    The least volatile skill gets the most capital. Skills with fewer than two
    observations or zero variance have an undefined inverse, so the whole book
    falls back to equal weight (1/N) — a safe default rather than a blow-up or a
    silently dropped leg. Returns {} for an empty input.
    """
    keys = list(returns_by_skill.keys())
    if not keys:
        return {}

    inv_var: dict[str, float] = {}
    for skill, rets in returns_by_skill.items():
        if len(rets) < 2:
            return _equal_weights(keys)
        var = variance(rets)  # sample variance, ddof=1
        if var <= 0.0:
            return _equal_weights(keys)
        inv_var[skill] = 1.0 / var

    total = sum(inv_var.values())
    if total <= 0.0:  # defensive; unreachable given the guards above
        return _equal_weights(keys)
    return {skill: iv / total for skill, iv in inv_var.items()}


def route_regime(
    efficiency_ratio: float,
    prev_state: str | None,
    enter: float = 0.4,
    exit: float = 0.3,
) -> str:
    """Classify the regime as "trend" or "range" with hysteresis.

    Enter trend when the Kaufman efficiency ratio is >= `enter`; fall back to
    range when it drops <= `exit`; in the dead-band between, hold `prev_state`.
    The dead-band is what kills flip-flop churn (and its fees) around a single
    threshold. Defaults to "range" when there is no prior state in the band.
    """
    if efficiency_ratio >= enter:
        return "trend"
    if efficiency_ratio <= exit:
        return "range"
    return prev_state if prev_state is not None else "range"


def select_active_skills(
    regime: str,
    trend_skills: list[str],
    range_skills: list[str],
    always_on: list[str],
) -> list[str]:
    """Pick the active skill set for `regime`, plus the always-on (orthogonal) legs.

    trend -> trend_skills + always_on; anything else -> range_skills + always_on.
    The funding sleeves live in `always_on` because they are regime-independent
    (they arm on the perp funding rate / are delta-neutral) and so run regardless.
    """
    regime_skills = trend_skills if regime == "trend" else range_skills
    return [*regime_skills, *always_on]


def annotate_active_and_blend(
    signals: list[dict],
    efficiency_ratio: float,
    prev_state: str | None,
    returns_by_skill: dict[str, list[float]],
) -> tuple[str, list[dict]]:
    """Tag each skill signal with the combination layer's view (live advisory).

    Composes the three pure primitives above: route the regime with hysteresis,
    select the active skill set for that regime, and compute inverse-vol blend
    weights over the active skills that have usable history. Each returned signal
    (a shallow copy) gains ``regime_routed``, ``active`` and ``blend_weight`` so
    the downstream researcher/trader agents reason with a principled prior.

    This is SHADOW in live: it enriches context but does not itself size or place
    orders. The same primitives drive the backtest, where the blend is validated
    by CPCV/DSR before it is ever trusted to move capital. When no usable
    per-skill returns are supplied, blend weights are 0 (no live blend yet) while
    routing/active still apply. Pure and stateless — the caller owns `prev_state`.
    """
    regime = route_regime(efficiency_ratio, prev_state)
    active = set(select_active_skills(regime, TREND_SKILLS, RANGE_SKILLS, ALWAYS_ON))
    usable = {
        sid: rets
        for sid, rets in returns_by_skill.items()
        if sid in active and len(rets) >= 2
    }
    weights = inverse_vol_weights(usable)

    out: list[dict] = []
    for sig in signals:
        sid = sig.get("skill_id", "")
        enriched = {
            **sig,
            "regime_routed": regime,
            "active": sid in active,
            "blend_weight": float(weights.get(sid, 0.0)),
        }
        out.append(enriched)
    return regime, out


def blend_live_shadow(
    signals: list[dict],
    efficiency_ratio: float,
    prev_state: str | None,
    returns_by_skill: dict[str, list[float]],
) -> tuple[str, list[dict], dict]:
    """Promote the combination layer from *advisory* to *blend live shadow* (§1.2).

    The advisory layer (``annotate_active_and_blend``) only tags signals, and in
    live its blend weights are 0 because no per-skill returns are plumbed in. This
    wrapper takes the SAME pure primitives but, when real ``returns_by_skill`` (e.g.
    the per-skill realised returns from the trade journal) are supplied, packages
    the layer's actual blend as a structured SHADOW RECORD — regime, active sleeve
    set, the normalised blend weights, and the gross — ready to be LOGGED so the
    combination layer's real apport can be measured over time.

    It is still SHADOW / log-only: ``size == 0.0`` and ``shadow is True``. It logs
    what it WOULD allocate; it never sizes a position or places an order. Purity is
    preserved — this returns the record; the caller owns the log emission (and, at
    the human merge gate, the one-line wiring that feeds journal returns here).
    Returns ``(regime, enriched_signals, shadow_record)``.
    """
    regime, enriched = annotate_active_and_blend(
        signals, efficiency_ratio, prev_state, returns_by_skill
    )
    blend = {
        s["skill_id"]: float(s["blend_weight"])
        for s in enriched
        if s.get("active") and s.get("blend_weight", 0.0) > 0.0
    }
    record = {
        "regime": regime,
        "active_skills": sorted({s["skill_id"] for s in enriched if s.get("active")}),
        "blend_weights": blend,
        "gross": float(sum(blend.values())),
        "shadow": True,  # log-only: this layer never sizes or trades
        "size": 0.0,
    }
    return regime, enriched, record
