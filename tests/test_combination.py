"""Combination layer (L2 regime router + L3 inverse-vol blend + skill selection).

All functions are pure. Expectations are hand-derived so the math (and the
anti-look-ahead/anti-overfit guarantees) is pinned, not just smoke-tested.
"""

from __future__ import annotations

import math

from hermes.strategies.combination import (
    annotate_active_and_blend,
    inverse_vol_weights,
    route_regime,
    select_active_skills,
)

# --------------------------------------------------------------------------- #
# inverse_vol_weights
# --------------------------------------------------------------------------- #


def test_inverse_vol_more_weight_to_least_volatile_and_sums_to_one():
    # "calm" has tiny dispersion, "wild" has large dispersion.
    returns = {
        "calm": [0.01, 0.011, 0.009, 0.0105, 0.0095],
        "wild": [0.05, -0.04, 0.06, -0.05, 0.045],
    }
    w = inverse_vol_weights(returns)
    assert math.isclose(sum(w.values()), 1.0, rel_tol=1e-9)
    assert w["calm"] > w["wild"]


def test_inverse_vol_exact_values():
    # Hand-derived: variance (ddof=1) of [0,2] is 2; of [0,1] is 0.5.
    # inv-var = {a: 1/2, b: 1/0.5=2}; normalise -> a=0.2, b=0.8.
    returns = {"a": [0.0, 2.0], "b": [0.0, 1.0]}
    w = inverse_vol_weights(returns)
    assert math.isclose(w["a"], 0.2, rel_tol=1e-9)
    assert math.isclose(w["b"], 0.8, rel_tol=1e-9)


def test_inverse_vol_zero_variance_falls_back_to_equal_weight():
    # Constant series -> variance 0 (undefined inverse). Must not divide by zero;
    # fall back to equal weight across all skills.
    returns = {"flat": [0.01, 0.01, 0.01], "alsoflat": [0.02, 0.02, 0.02]}
    w = inverse_vol_weights(returns)
    assert math.isclose(sum(w.values()), 1.0, rel_tol=1e-9)
    assert math.isclose(w["flat"], 0.5, rel_tol=1e-9)
    assert math.isclose(w["alsoflat"], 0.5, rel_tol=1e-9)


def test_inverse_vol_too_few_returns_falls_back_to_equal_weight():
    # <2 observations -> variance undefined -> equal-weight fallback.
    returns = {"x": [0.01], "y": []}
    w = inverse_vol_weights(returns)
    assert math.isclose(sum(w.values()), 1.0, rel_tol=1e-9)
    assert math.isclose(w["x"], 0.5, rel_tol=1e-9)
    assert math.isclose(w["y"], 0.5, rel_tol=1e-9)


def test_inverse_vol_empty_dict():
    assert inverse_vol_weights({}) == {}


def test_inverse_vol_single_skill_gets_all_weight():
    w = inverse_vol_weights({"only": [0.01, -0.02, 0.03]})
    assert math.isclose(w["only"], 1.0, rel_tol=1e-9)


# --------------------------------------------------------------------------- #
# route_regime (hysteresis)
# --------------------------------------------------------------------------- #


def test_route_regime_enters_trend_above_enter_threshold():
    assert route_regime(0.45, prev_state=None) == "trend"
    assert route_regime(0.40, prev_state="range") == "trend"  # >= enter


def test_route_regime_exits_to_range_below_exit_threshold():
    assert route_regime(0.25, prev_state="trend") == "range"
    assert route_regime(0.30, prev_state="trend") == "range"  # <= exit


def test_route_regime_hysteresis_keeps_previous_state_in_deadband():
    # 0.35 is between exit (0.3) and enter (0.4): keep whatever we had.
    assert route_regime(0.35, prev_state="trend") == "trend"
    assert route_regime(0.35, prev_state="range") == "range"


def test_route_regime_defaults_to_range_when_no_prev_state_in_deadband():
    assert route_regime(0.35, prev_state=None) == "range"


# --------------------------------------------------------------------------- #
# select_active_skills
# --------------------------------------------------------------------------- #


def test_select_active_skills_trend_routes_trend_plus_always_on():
    out = select_active_skills(
        "trend",
        trend_skills=["ts_mom", "breakout"],
        range_skills=["mean_rev"],
        always_on=["pairs", "tod"],
    )
    assert out == ["ts_mom", "breakout", "pairs", "tod"]


def test_select_active_skills_range_routes_range_plus_always_on():
    out = select_active_skills(
        "range",
        trend_skills=["ts_mom", "breakout"],
        range_skills=["mean_rev"],
        always_on=["pairs", "tod"],
    )
    assert out == ["mean_rev", "pairs", "tod"]


# --------------------------------------------------------------------------- #
# annotate_active_and_blend (live advisory wiring of the combination layer)
# --------------------------------------------------------------------------- #


def test_annotate_marks_active_set_and_normalised_blend_weights():
    signals = [
        {"skill_id": "ts_mom_vol_scaled", "confidence": 0.5},
        {"skill_id": "mean_reversion_regime", "confidence": 0.5},
        {"skill_id": "pairs_eth_btc", "confidence": 0.5},
    ]
    returns = {
        "ts_mom_vol_scaled": [0.01, -0.01, 0.02],
        "pairs_eth_btc": [0.005, -0.002, 0.001],
    }
    regime, out = annotate_active_and_blend(
        signals, efficiency_ratio=0.6, prev_state=None, returns_by_skill=returns
    )
    assert regime == "trend"
    by = {s["skill_id"]: s for s in out}
    assert by["ts_mom_vol_scaled"]["active"] is True  # trend skill, trend regime
    assert by["mean_reversion_regime"]["active"] is False  # range skill, off-regime
    assert by["pairs_eth_btc"]["active"] is True  # always-on
    # Blend weights are over the active skills that have usable history, summing 1.
    assert by["ts_mom_vol_scaled"]["blend_weight"] > 0.0
    assert by["mean_reversion_regime"]["blend_weight"] == 0.0
    assert math.isclose(
        sum(s["blend_weight"] for s in out), 1.0, rel_tol=1e-9
    )
    # Each signal also carries the routed regime for downstream agents.
    assert all(s["regime_routed"] == "trend" for s in out)


def test_annotate_regime_hysteresis_holds_prev_in_deadband():
    regime, _ = annotate_active_and_blend(
        [], efficiency_ratio=0.35, prev_state="trend", returns_by_skill={}
    )
    assert regime == "trend"


def test_annotate_no_history_gives_zero_blend_but_still_routes():
    signals = [{"skill_id": "ts_mom_vol_scaled", "confidence": 0.5}]
    regime, out = annotate_active_and_blend(
        signals, efficiency_ratio=0.6, prev_state=None, returns_by_skill={}
    )
    assert regime == "trend"
    assert out[0]["active"] is True
    assert out[0]["blend_weight"] == 0.0  # no usable returns -> no live blend yet
