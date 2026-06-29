"""SHADOW new-data sleeves: directional correctness, graceful HOLD, and the
load-bearing SAFETY property — a shadow skill (registered but not in
strategy.allowed_skills) can NEVER be routed to execution by the selective gate."""

from __future__ import annotations

import pandas as pd

from hermes.risk.selective import (
    SelectiveConfig,
    drive_signal,
    load_selective_config,
    rank_selective_signals,
    selective_gate,
)
from hermes.runtime.pulse import pulse_entry
from hermes.settings import get_settings
from hermes.skills.library import SkillSignal, get_skill_library
from hermes.skills.newdata_shadow import (
    onchain_stablecoin_growth,
    options_iv_dvol_reversion,
)

_SHADOW = ("options_iv_dvol_reversion", "onchain_stablecoin_growth")


def _df(col: str, val: float, n: int = 5) -> pd.DataFrame:
    return pd.DataFrame({"close": [100.0] * n, col: [0.0] * (n - 1) + [val]})


# --- directional correctness (faithful to the validated configs) ----------- #


def test_dvol_reversion_long_on_fear_short_on_complacency():
    assert options_iv_dvol_reversion(_df("dvol_z_90d", 2.5)).side == "BUY"   # fear -> long
    assert options_iv_dvol_reversion(_df("dvol_z_90d", -2.5)).side == "SELL"  # complacency -> short
    assert options_iv_dvol_reversion(_df("dvol_z_90d", 0.5)).side == "HOLD"   # within band


def test_stablecoin_growth_long_on_expansion_short_on_contraction():
    assert onchain_stablecoin_growth(_df("stbl_supply_growth_z", 1.5)).side == "BUY"
    assert onchain_stablecoin_growth(_df("stbl_supply_growth_z", -1.5)).side == "SELL"
    assert onchain_stablecoin_growth(_df("stbl_supply_growth_z", 0.2)).side == "HOLD"


def test_shadow_skills_hold_when_their_data_column_is_absent():
    """No options/on-chain data attached (e.g. offline, or an alt) -> safe HOLD,
    and the feature is still surfaced for the dashboard gauge."""
    plain = pd.DataFrame({"close": [100.0] * 5})
    iv = options_iv_dvol_reversion(plain)
    oc = onchain_stablecoin_growth(plain)
    assert iv.side == "HOLD" and oc.side == "HOLD"
    # the always-present features carry the live reading for the gauge
    assert options_iv_dvol_reversion(_df("dvol_z_90d", 2.5)).features["dvol_z"] == 2.5


# --- the SAFETY property: shadow == log-only, never executed ---------------- #


def test_shadow_skills_registered_but_absent_from_live_slate():
    """They must be in the library (so they compute + log every tick) but NOT in
    the live config's allowed_skills (so they are never tradable)."""
    lib = get_skill_library()
    cfg = load_selective_config(get_settings())
    for sid in _SHADOW:
        assert lib.get(sid) is not None, f"{sid} must be registered to compute+log"
        assert sid not in cfg.allowed_skills, f"{sid} must stay OUT of the live slate"


def test_selective_gate_rejects_shadow_skill_even_with_strong_buy():
    """The decisive guard: a strong BUY from a shadow skill is never selected to
    drive a tick and is rejected by the gate — regardless of confidence."""
    cfg = SelectiveConfig(
        enabled=True,
        allowed_skills=("ts_mom_vol_scaled", "funding_squeeze_continuation"),
        conviction_min=1.0,
        no_pyramid=True,
    )
    strong = [
        {"skill_id": "options_iv_dvol_reversion", "side": "BUY", "confidence": 0.99,
         "confidence_raw": 5.0},
        {"skill_id": "onchain_stablecoin_growth", "side": "BUY", "confidence": 0.99,
         "confidence_raw": 5.0},
    ]
    assert rank_selective_signals(strong, cfg, "BTC/USDT:USDT") == []
    assert drive_signal(strong, cfg, "BTC/USDT:USDT", set()) is None
    for sid in _SHADOW:
        ok, why = selective_gate("BUY", "BTC/USDT:USDT", sid, strong, set(), cfg)
        assert ok is False and "not in selective slate" in why


# --- dashboard contract: distinct shadow gauges, excluded from would_act ---- #


def test_pulse_emits_distinct_shadow_gauges_not_driving_would_act():
    """The pulse must tag shadow gauges (shadow/skill/side) so the dashboard can
    render them in a SEPARATE panel — and they must never set would_act."""
    cfg = SelectiveConfig(
        enabled=True,
        allowed_skills=("ts_mom_vol_scaled", "funding_squeeze_continuation"),
        conviction_min=1.0,
        no_pyramid=True,
    )
    sigs = [
        SkillSignal("options_iv_dvol_reversion", "BUY", 0.6, 0.04, 0.08, "fear", {"dvol_z": 2.4}),
        SkillSignal("onchain_stablecoin_growth", "BUY", 0.8, 0.05, 0.10, "dry", {"stbl_z": 1.6}),
    ]
    e = pulse_entry("BTC/USDT:USDT", {"regime": "range", "efficiency_ratio": 0.1}, sigs, set(), cfg)
    shadow = [g for g in e["gauges"] if g.get("shadow")]
    assert {g["skill"] for g in shadow} == set(_SHADOW)
    assert all(g["side"] == "BUY" and g["armed"] for g in shadow)
    # armed shadow BUYs must NOT cause the symbol to "would_act" (they are excluded)
    assert e["would_act"] == "HOLD"
