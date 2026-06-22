"""The hourly pulse summarises, per symbol, what the agent SEES (regime + skill
signals) and what it WOULD do under the selective gate — so the dashboard shows
continuous activity between the 4h decisions. Pure summary is unit-tested."""

from __future__ import annotations

from hermes.risk.selective import SelectiveConfig
from hermes.runtime.pulse import pulse_entry
from hermes.skills.library import SkillSignal

CFG = SelectiveConfig(
    enabled=True,
    allowed_skills=("ts_mom_vol_scaled", "mean_reversion_regime"),
    conviction_min=1.0,
    no_pyramid=True,
)


def _sig(skill, side, conf):
    return SkillSignal(skill, side, conf, 0.02, 0.04, "r", {})


def test_pulse_entry_would_enter_on_saturated_signal():
    sigs = [_sig("ts_mom_vol_scaled", "BUY", 1.0)]
    e = pulse_entry("BTC/USDC", {"regime": "trend_up", "efficiency_ratio": 0.7}, sigs, set(), CFG)
    assert e["symbol"] == "BTC/USDC"
    assert e["regime"] == "trend_up"
    assert e["would_act"] == "BUY"
    assert e["would_skill"] == "ts_mom_vol_scaled"


def test_pulse_entry_holds_low_conviction_with_reason():
    sigs = [_sig("ts_mom_vol_scaled", "BUY", 0.4)]
    e = pulse_entry("ETH/USDC", {"regime": "range", "efficiency_ratio": 0.2}, sigs, set(), CFG)
    assert e["would_act"] == "HOLD"
    assert "conviction" in e["reason"].lower()


def test_pulse_entry_no_actionable_signal():
    sigs = [_sig("ts_mom_vol_scaled", "HOLD", 0.0)]
    e = pulse_entry("SOL/USDC", {"regime": "range", "efficiency_ratio": 0.1}, sigs, set(), CFG)
    assert e["would_act"] == "HOLD"
    assert e["signals"] == []  # no non-HOLD signals surfaced
