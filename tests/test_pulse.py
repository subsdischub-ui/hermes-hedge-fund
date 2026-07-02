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


def test_pulse_entry_holds_low_conviction_with_reason(monkeypatch):
    # The live _CONVICTION_GATED set is empty since the 2026-07-02 restructuration
    # (ts_mom retired); gate it back in for the scope of this test so the pulse's
    # low-conviction HOLD reason stays covered.
    from hermes.risk import selective as sel

    monkeypatch.setattr(sel, "_CONVICTION_GATED", {"ts_mom_vol_scaled"})
    sigs = [_sig("ts_mom_vol_scaled", "BUY", 0.4)]
    e = pulse_entry("ETH/USDC", {"regime": "range", "efficiency_ratio": 0.2}, sigs, set(), CFG)
    assert e["would_act"] == "HOLD"
    assert "conviction" in e["reason"].lower()


def test_pulse_entry_no_actionable_signal():
    sigs = [_sig("ts_mom_vol_scaled", "HOLD", 0.0)]
    e = pulse_entry("SOL/USDC", {"regime": "range", "efficiency_ratio": 0.1}, sigs, set(), CFG)
    assert e["would_act"] == "HOLD"
    assert e["signals"] == []  # no non-HOLD signals surfaced


def test_pulse_entry_emits_live_gauges_for_both_sleeves_even_when_holding():
    """Visibility: the dashboard must show each sleeve's live reading vs its arm
    threshold, even while both hold (momentum conviction, funding z)."""
    sigs = [
        _sig("ts_mom_vol_scaled", "BUY", 0.4),
        SkillSignal("funding_squeeze_continuation", "HOLD", 0.0, 0.02, 0.04, "z<2", {"funding_z": 0.8}),
    ]
    e = pulse_entry("BTC/USDC", {"regime": "range", "efficiency_ratio": 0.1}, sigs, set(), CFG)
    g = {x["sleeve"]: x for x in e["gauges"]}
    assert g["Momentum #1"]["val"] == 0.4 and g["Momentum #1"]["thr"] == 1.0
    assert g["Momentum #1"]["armed"] is False
    assert g["Funding #2"]["val"] == 0.8 and g["Funding #2"]["thr"] == 2.0
    assert g["Funding #2"]["armed"] is False


def test_pulse_entry_momentum_gauge_is_flagged_retired_and_never_armed():
    """RESTRUCTURATION 2026-07-02: ts_mom_vol_scaled was retired from
    strategy.allowed_skills (bull-beta drag, LOO-Δ Sharpe -0.146). The skill library
    still computes its signal (so the gauge keeps showing what it SEES), but
    selective_gate rejects it unconditionally now — so the gauge must never render
    as ARMED/actionable even at saturated conviction, and must carry a `retired`
    flag so the dashboard can render it as a dead gauge instead of a live one."""
    sigs = [_sig("ts_mom_vol_scaled", "BUY", 5.0)]  # far above the old 1.0 threshold
    e = pulse_entry("BTC/USDC", {"regime": "trend_up", "efficiency_ratio": 0.7}, sigs, set(), CFG)
    g = {x["sleeve"]: x for x in e["gauges"]}
    assert g["Momentum #1"]["armed"] is False
    assert g["Momentum #1"]["retired"] is True
    # And it can never win would_act even though CFG still lists it in allowed_skills
    # for this unit test's fixture — the real live config no longer does.
