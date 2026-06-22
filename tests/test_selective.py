"""The selective-mode gate enforces the OOS-validated rule in the live path:
trade only the high-conviction momentum tail + range mean-reversion, one position
per symbol (consolidation), dropping skills that lose even gross. Default OFF."""

from __future__ import annotations

from hermes.risk.selective import SelectiveConfig, selective_gate

CFG = SelectiveConfig(
    enabled=True,
    allowed_skills=("ts_mom_vol_scaled", "mean_reversion_regime"),
    conviction_min=1.0,
    no_pyramid=True,
)


def _sig(skill: str, conf_raw: float) -> dict:
    return {"skill_id": skill, "confidence_raw": conf_raw, "confidence": conf_raw}


def test_disabled_mode_is_noop():
    ok, _ = selective_gate(
        "BUY", "BTC/USDC", "anything", [], set(),
        SelectiveConfig(False, (), 1.0, True),
    )
    assert ok


def test_hold_always_passes():
    ok, _ = selective_gate("HOLD", "BTC/USDC", "", [], set(), CFG)
    assert ok


def test_blocks_skill_outside_selective_slate():
    ok, reason = selective_gate(
        "SELL", "BTC/USDC", "breakout_vol_expansion",
        [_sig("breakout_vol_expansion", 1.0)], set(), CFG,
    )
    assert not ok
    assert "slate" in reason.lower()


def test_blocks_low_conviction_ts_mom():
    ok, reason = selective_gate(
        "BUY", "BTC/USDC", "ts_mom_vol_scaled",
        [_sig("ts_mom_vol_scaled", 0.6)], set(), CFG,
    )
    assert not ok
    assert "conviction" in reason.lower()


def test_allows_saturated_ts_mom():
    ok, _ = selective_gate(
        "BUY", "BTC/USDC", "ts_mom_vol_scaled",
        [_sig("ts_mom_vol_scaled", 1.0)], set(), CFG,
    )
    assert ok


def test_mean_reversion_exempt_from_conviction_floor():
    # mean_reversion_regime self-selects at RSI/%B extremes (fixed conf 0.6); the
    # conviction floor is a ts_mom concept and must not block it.
    ok, _ = selective_gate(
        "BUY", "BTC/USDC", "mean_reversion_regime",
        [_sig("mean_reversion_regime", 0.6)], set(), CFG,
    )
    assert ok


def test_no_pyramid_blocks_when_already_holding_symbol():
    ok, reason = selective_gate(
        "BUY", "BTC/USDC", "ts_mom_vol_scaled",
        [_sig("ts_mom_vol_scaled", 1.0)], {"BTC/USDC"}, CFG,
    )
    assert not ok
    assert "hold" in reason.lower() or "pyramid" in reason.lower()


def test_no_pyramid_allows_different_symbol():
    ok, _ = selective_gate(
        "BUY", "ETH/USDC", "ts_mom_vol_scaled",
        [_sig("ts_mom_vol_scaled", 1.0)], {"BTC/USDC"}, CFG,
    )
    assert ok
