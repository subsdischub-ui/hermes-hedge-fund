"""The selective-mode gate enforces the OOS-validated rule in the live path:
trade only the high-conviction momentum tail + range mean-reversion, one position
per symbol (consolidation), dropping skills that lose even gross. Default OFF."""

from __future__ import annotations

from hermes.risk.selective import (
    SelectiveConfig,
    drive_signal,
    pick_selective_signal,
    selective_gate,
)

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


def test_blocks_low_conviction_when_skill_is_conviction_gated(monkeypatch):
    # The live _CONVICTION_GATED set is EMPTY since the 2026-07-02 restructuration
    # (ts_mom retired); the floor MECHANISM stays wired for any future scored skill —
    # covered here by gating ts_mom back in for the scope of this test.
    from hermes.risk import selective as sel

    monkeypatch.setattr(sel, "_CONVICTION_GATED", {"ts_mom_vol_scaled"})
    ok, reason = selective_gate(
        "BUY", "BTC/USDC", "ts_mom_vol_scaled",
        [_sig("ts_mom_vol_scaled", 0.6)], set(), CFG,
    )
    assert not ok
    assert "conviction" in reason.lower()


def test_ungated_skill_is_exempt_from_conviction_floor_by_default():
    # Live behaviour since the restructuration: no skill is conviction-gated, so a
    # slate skill below conviction_min passes the conviction check (it self-selects
    # at its own extremes upstream instead).
    ok, _ = selective_gate(
        "BUY", "BTC/USDC", "ts_mom_vol_scaled",
        [_sig("ts_mom_vol_scaled", 0.6)], set(), CFG,
    )
    assert ok


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


def test_no_pyramid_blocks_when_already_holding_same_symbol_and_skill():
    ok, reason = selective_gate(
        "BUY", "BTC/USDC", "ts_mom_vol_scaled",
        [_sig("ts_mom_vol_scaled", 1.0)], {("BTC/USDC", "ts_mom_vol_scaled")}, CFG,
    )
    assert not ok
    assert "pyramid" in reason.lower() or "holding" in reason.lower()


def test_no_pyramid_allows_different_symbol():
    ok, _ = selective_gate(
        "BUY", "ETH/USDC", "ts_mom_vol_scaled",
        [_sig("ts_mom_vol_scaled", 1.0)], {("BTC/USDC", "ts_mom_vol_scaled")}, CFG,
    )
    assert ok


def test_no_pyramid_allows_different_skill_same_symbol():
    # THE concurrent-sleeve invariant: momentum holding BTC must NOT block funding
    # from also opening BTC (two independent strategies, one position each).
    ok, _ = selective_gate(
        "BUY", "BTC/USDC", "funding_squeeze_continuation",
        [_sig("funding_squeeze_continuation", 0.6)],
        {("BTC/USDC", "ts_mom_vol_scaled")}, CFG_AB,
    )
    assert ok


# --- deterministic drive: the validated signal overrides the LLM's veto -------
def _ssig(skill, side, conf):
    return {"skill_id": skill, "side": side, "confidence_raw": conf, "confidence": conf,
            "stop_pct": 0.03, "take_profit_pct": 0.05}


def test_pick_selective_returns_best_qualifying_signal():
    sigs = [_ssig("ts_mom_vol_scaled", "BUY", 1.0), _ssig("breakout_vol_expansion", "SELL", 0.65)]
    p = pick_selective_signal(sigs, CFG, "BTC/USDC")
    assert p is not None
    assert p["side"] == "BUY" and p["skill_id"] == "ts_mom_vol_scaled" and p["symbol"] == "BTC/USDC"


def test_pick_selective_none_when_low_conviction(monkeypatch):
    # Mechanism coverage (live set empty since restructuration — see gate test above).
    from hermes.risk import selective as sel

    monkeypatch.setattr(sel, "_CONVICTION_GATED", {"ts_mom_vol_scaled"})
    assert pick_selective_signal([_ssig("ts_mom_vol_scaled", "BUY", 0.5)], CFG, "BTC/USDC") is None


def test_pick_selective_mean_reversion_exempt_from_conviction():
    p = pick_selective_signal([_ssig("mean_reversion_regime", "SELL", 0.6)], CFG, "ETH/USDC")
    assert p is not None and p["side"] == "SELL"


def test_pick_selective_disabled_returns_none():
    off = SelectiveConfig(False, (), 1.0, True)
    assert pick_selective_signal([_ssig("ts_mom_vol_scaled", "BUY", 1.0)], off, "BTC/USDC") is None


# --- Strategy #2 (funding) sharing the selective slate ------------------------
CFG_AB = SelectiveConfig(
    enabled=True,
    allowed_skills=("ts_mom_vol_scaled", "mean_reversion_regime", "funding_squeeze_continuation"),
    conviction_min=1.0,
    no_pyramid=True,
)


def test_funding_exempt_from_conviction_floor():
    # funding self-selects at z>=2 (conf can be 0.5); it is NOT conviction-gated,
    # so the ts_mom conviction floor must not block it.
    ok, _ = selective_gate(
        "BUY", "BTC/USDC", "funding_squeeze_continuation",
        [_sig("funding_squeeze_continuation", 0.5)], set(), CFG_AB,
    )
    assert ok


def test_pick_selective_drives_funding_when_momentum_quiet():
    # On a tick where momentum doesn't fire but funding does, funding drives.
    sigs = [
        _ssig("ts_mom_vol_scaled", "HOLD", 0.0),
        _ssig("funding_squeeze_continuation", "BUY", 0.6),
    ]
    p = pick_selective_signal(sigs, CFG_AB, "BTC/USDC")
    assert p is not None
    assert p["side"] == "BUY" and p["skill_id"] == "funding_squeeze_continuation"


def test_default_config_includes_funding_track():
    # Lock the live wiring: the shipped config must allow Strategy #2.
    from hermes.risk.selective import load_selective_config
    from hermes.settings import get_settings

    cfg = load_selective_config(get_settings())
    assert cfg.enabled
    assert "funding_squeeze_continuation" in cfg.allowed_skills


# --- drive_signal: the gate-aware anti-starvation selector --------------------
def test_drive_signal_funding_not_starved_when_momentum_holds_symbol():
    """THE fix: momentum holding BTC must NOT prevent funding from being driven on
    BTC the same tick. Both fire; momentum is held -> funding is selected."""
    sigs = [
        _ssig("ts_mom_vol_scaled", "BUY", 1.0),               # higher confidence
        _ssig("funding_squeeze_continuation", "BUY", 0.6),    # lower, but momentum is held
    ]
    open_keys = {("BTC/USDC", "ts_mom_vol_scaled")}
    det = drive_signal(sigs, CFG_AB, "BTC/USDC", open_keys)
    assert det is not None
    assert det["skill_id"] == "funding_squeeze_continuation"


def test_drive_signal_prefers_highest_conviction_when_nothing_held():
    sigs = [
        _ssig("ts_mom_vol_scaled", "BUY", 1.0),
        _ssig("funding_squeeze_continuation", "BUY", 0.6),
    ]
    det = drive_signal(sigs, CFG_AB, "BTC/USDC", set())
    assert det is not None and det["skill_id"] == "ts_mom_vol_scaled"


def test_drive_signal_none_when_all_held():
    sigs = [_ssig("ts_mom_vol_scaled", "BUY", 1.0)]
    det = drive_signal(sigs, CFG_AB, "BTC/USDC", {("BTC/USDC", "ts_mom_vol_scaled")})
    assert det is None
