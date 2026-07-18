"""Shadow measurement of the maker fill-rate on OUR OWN live entries (P1).

Frozen definitions/verdict live in research_maker/PREREG_SHADOW_FILL.md — this
module measures a hypothetical maker order laid at the decision-bar close, and
whether the NEXT 4h bar would have filled it. Two fill semantics are always
computed side by side:
  * STRICT (honest lower bound): the next bar must STRICTLY cross the limit
    (low < limit for a buy, high > limit for a sell) — an exact touch does NOT
    count, because resting exactly at the touched price with no further
    penetration is not a safe assumption of a fill.
  * AU-TOUCHER (upper bound): low <= limit / high >= limit.
The gap between the two is deliberate (prereg: "l'encadrement est gratuit").
"""

from __future__ import annotations

import os
import stat

from hermes.execution import shadow_maker as sm


def _bar(o, h, low_, c):
    return {"open": o, "high": h, "low": low_, "close": c}


# ---------------------------------------------------------------------------
# 1) STRICT vs AU-TOUCHER on an EXACT touch — the test that must bite.
# ---------------------------------------------------------------------------
def test_exact_touch_fills_touch_but_not_strict_on_a_buy(tmp_path):
    # BUY at decision_close=100, offset 0 -> limit_0 = 100. Next bar's low lands
    # EXACTLY on the limit (100) and never goes below it.
    sm.record_entry(ts="2026-01-01T00:00:00+00:00", symbol="BTC/USDT", side="BUY",
                     skill_id="mean_reversion_regime", decision_close=100.0, data_dir=tmp_path)
    sm.resolve_pending("BTC/USDT", _bar(o=100.5, h=101.0, low_=100.0, c=100.5), data_dir=tmp_path)

    entries = sm._load(tmp_path)
    assert len(entries) == 1
    e = entries[0]
    assert e["fill_touch_0"] is True   # low <= limit (100 <= 100) -> touch fills
    assert e["fill_strict_0"] is False  # low < limit (100 < 100 is False) -> strict does NOT


def test_exact_touch_fills_touch_but_not_strict_on_a_sell(tmp_path):
    # SELL at decision_close=100, offset 0 -> limit_0 = 100. Next bar's high lands
    # EXACTLY on the limit and never exceeds it.
    sm.record_entry(ts="2026-01-01T00:00:00+00:00", symbol="ETH/USDT", side="SELL",
                     skill_id="funding_squeeze_continuation", decision_close=100.0, data_dir=tmp_path)
    sm.resolve_pending("ETH/USDT", _bar(o=99.5, h=100.0, low_=99.0, c=99.5), data_dir=tmp_path)

    e = sm._load(tmp_path)[0]
    assert e["fill_touch_0"] is True    # high >= limit (100 >= 100) -> touch fills
    assert e["fill_strict_0"] is False  # high > limit (100 > 100 is False) -> strict does NOT


# ---------------------------------------------------------------------------
# 2) Buy AND sell, comfortably away from the boundary (sanity, not edge case).
# ---------------------------------------------------------------------------
def test_buy_strict_and_touch_fill_when_low_clears_the_limit(tmp_path):
    sm.record_entry(ts="t0", symbol="BTC/USDT", side="BUY", skill_id="s",
                     decision_close=100.0, data_dir=tmp_path)
    sm.resolve_pending("BTC/USDT", _bar(o=99.0, h=101.0, low_=98.0, c=99.5), data_dir=tmp_path)
    e = sm._load(tmp_path)[0]
    assert e["fill_strict_0"] is True
    assert e["fill_touch_0"] is True


def test_sell_strict_and_touch_fill_when_high_clears_the_limit(tmp_path):
    sm.record_entry(ts="t0", symbol="BTC/USDT", side="SELL", skill_id="s",
                     decision_close=100.0, data_dir=tmp_path)
    sm.resolve_pending("BTC/USDT", _bar(o=101.0, h=102.0, low_=100.5, c=101.5), data_dir=tmp_path)
    e = sm._load(tmp_path)[0]
    assert e["fill_strict_0"] is True
    assert e["fill_touch_0"] is True


def test_buy_no_fill_when_low_never_reaches_the_limit(tmp_path):
    sm.record_entry(ts="t0", symbol="BTC/USDT", side="BUY", skill_id="s",
                     decision_close=100.0, data_dir=tmp_path)
    sm.resolve_pending("BTC/USDT", _bar(o=101.0, h=103.0, low_=100.5, c=102.0), data_dir=tmp_path)
    e = sm._load(tmp_path)[0]
    assert e["fill_strict_0"] is False
    assert e["fill_touch_0"] is False


# ---------------------------------------------------------------------------
# 3) Offsets 0 AND 5 bps — both always computed, both must differ correctly.
# ---------------------------------------------------------------------------
def test_offsets_0_and_5_bps_computed_and_stored_for_a_buy():
    limits = sm._limits_for(side="BUY", decision_close=100.0)
    assert limits[0.0] == 100.0
    assert abs(limits[5.0] - 100.0 * (1 - 5 / 10_000.0)) < 1e-9  # below close


def test_offsets_0_and_5_bps_computed_and_stored_for_a_sell():
    limits = sm._limits_for(side="SELL", decision_close=100.0)
    assert limits[0.0] == 100.0
    assert abs(limits[5.0] - 100.0 * (1 + 5 / 10_000.0)) < 1e-9  # above close


def test_offset_0_fills_more_readily_than_offset_5_on_a_shallow_dip_buy(tmp_path):
    # Buy limit_0 = 100.0 (at the decision close); limit_5 = 99.95 (further below,
    # needs a bigger dip). A shallow dip to 99.97 clears limit_0 but not limit_5 —
    # a HIGHER offset is a MORE conservative (harder to fill) maker order for a buy.
    sm.record_entry(ts="t0", symbol="BTC/USDT", side="BUY", skill_id="s",
                     decision_close=100.0, data_dir=tmp_path)
    sm.resolve_pending("BTC/USDT", _bar(o=100.2, h=100.5, low_=99.97, c=100.1), data_dir=tmp_path)
    e = sm._load(tmp_path)[0]
    assert e["fill_strict_0"] is True
    assert e["fill_strict_5"] is False


# ---------------------------------------------------------------------------
# 4) Persistence across two "instances" (restart) — no in-memory state at all;
#    every call reads/writes the file, so a fresh call sequence must see what a
#    prior one wrote.
# ---------------------------------------------------------------------------
def test_state_survives_a_restart(tmp_path):
    sm.record_entry(ts="t0", symbol="SOL/USDT", side="BUY", skill_id="s",
                     decision_close=50.0, data_dir=tmp_path)
    # "restart": nothing kept in memory, re-enter purely via the file-backed API.
    sm.resolve_pending("SOL/USDT", _bar(o=49.0, h=50.5, low_=48.5, c=49.5), data_dir=tmp_path)
    s1 = sm.summary(data_dir=tmp_path)
    assert s1["n_logged"] == 1
    assert s1["n_resolved"] == 1

    # A brand new "process" only knows the data_dir path — same file, same truth.
    s2 = sm.summary(data_dir=tmp_path)
    assert s2 == s1
    entries = sm._load(tmp_path)
    assert len(entries) == 1
    assert entries[0]["resolved"] is True


# ---------------------------------------------------------------------------
# 5) Chase cost: STRICT(0) non-fill with an adverse next-bar open -> correct
#    empirical average.
# ---------------------------------------------------------------------------
def test_chase_cost_on_buy_non_fill_strict_is_the_adverse_open_move(tmp_path):
    # decision_close=100, buy limit_0=100, never touched (low=100.5) -> non-fill.
    # Next bar opens at 100.8 (worse for a late buyer) -> adverse chase cost > 0.
    sm.record_entry(ts="t0", symbol="BTC/USDT", side="BUY", skill_id="s",
                     decision_close=100.0, data_dir=tmp_path)
    sm.resolve_pending("BTC/USDT", _bar(o=100.8, h=101.0, low_=100.5, c=100.9), data_dir=tmp_path)
    e = sm._load(tmp_path)[0]
    assert e["fill_strict_0"] is False
    expected = (100.8 - 100.0) / 100.0 * 10_000.0
    assert abs(e["chase_cost_bps"] - expected) < 1e-9


def test_chase_cost_on_sell_non_fill_strict_is_the_adverse_open_move(tmp_path):
    # decision_close=100, sell limit_0=100, never touched (high=99.5) -> non-fill.
    # Next bar opens at 99.2 (worse for a late seller) -> adverse chase cost > 0.
    sm.record_entry(ts="t0", symbol="BTC/USDT", side="SELL", skill_id="s",
                     decision_close=100.0, data_dir=tmp_path)
    sm.resolve_pending("BTC/USDT", _bar(o=99.2, h=99.5, low_=99.0, c=99.3), data_dir=tmp_path)
    e = sm._load(tmp_path)[0]
    assert e["fill_strict_0"] is False
    expected = (100.0 - 99.2) / 100.0 * 10_000.0
    assert abs(e["chase_cost_bps"] - expected) < 1e-9


def test_chase_cost_is_none_when_strict_0_fills(tmp_path):
    sm.record_entry(ts="t0", symbol="BTC/USDT", side="BUY", skill_id="s",
                     decision_close=100.0, data_dir=tmp_path)
    sm.resolve_pending("BTC/USDT", _bar(o=99.0, h=101.0, low_=98.0, c=99.5), data_dir=tmp_path)
    e = sm._load(tmp_path)[0]
    assert e["fill_strict_0"] is True
    assert e["chase_cost_bps"] is None


def test_chase_cost_mean_is_the_empirical_average_of_non_fills(tmp_path):
    # Two non-fills with chase costs +10bps and +20bps -> mean 15bps. Fill_strict_0
    # for these two is 0/2 -> gain_net uses the mean over the non-fill population only.
    sm.record_entry(ts="t0", symbol="BTC/USDT", side="BUY", skill_id="s",
                     decision_close=100.0, data_dir=tmp_path)
    sm.resolve_pending("BTC/USDT", _bar(o=100.10, h=100.2, low_=100.05, c=100.1), data_dir=tmp_path)
    sm.record_entry(ts="t1", symbol="BTC/USDT", side="BUY", skill_id="s",
                     decision_close=100.0, data_dir=tmp_path)
    sm.resolve_pending("BTC/USDT", _bar(o=100.20, h=100.3, low_=100.05, c=100.2), data_dir=tmp_path)
    s = sm.summary(data_dir=tmp_path)
    assert s["n_resolved"] == 2
    assert abs(s["chase_cost_bps_mean"] - 15.0) < 1e-6
    assert s["fill_strict_0"] == 0.0
    expected_gain_net = 0.0 * sm.GAIN_PAR_FILL_BPS - 1.0 * 15.0
    assert abs(s["gain_net_bps_est"] - expected_gain_net) < 1e-9


# ---------------------------------------------------------------------------
# 6) Verdict mechanic, frozen at the first 40 resolved signals.
# ---------------------------------------------------------------------------
def _log_n_resolved(tmp_path, n, *, fill_strict_frac, gain_net_target_bps):
    """Synthesize `n` resolved BUY entries with a chosen strict(0) fill fraction
    and a chosen resulting gain_net (by choosing the non-fill chase cost so the
    frozen formula lands exactly on gain_net_target_bps)."""
    n_fill = round(n * fill_strict_frac)
    n_nonfill = n - n_fill
    fill_frac = n_fill / n
    if n_nonfill > 0:
        # gain_net = fill_frac*3.0 - (1-fill_frac)*chase  =>  solve chase
        chase = (fill_frac * sm.GAIN_PAR_FILL_BPS - gain_net_target_bps) / (1 - fill_frac)
    else:
        chase = 0.0
    for i in range(n):
        sm.record_entry(ts=f"t{i}", symbol="BTC/USDT", side="BUY", skill_id="s",
                         decision_close=100.0, data_dir=tmp_path)
        if i < n_fill:
            # strict(0) fills: low clears the limit.
            bar = _bar(o=99.0, h=101.0, low_=98.0, c=99.5)
        else:
            # strict(0) non-fill; the adverse open move reproduces `chase` bps.
            adverse_open = 100.0 * (1 + chase / 10_000.0)
            bar = _bar(o=adverse_open, h=adverse_open + 1.0, low_=100.05, c=adverse_open)
        sm.resolve_pending("BTC/USDT", bar, data_dir=tmp_path)


def test_verdict_none_below_40_resolved(tmp_path):
    _log_n_resolved(tmp_path, 39, fill_strict_frac=0.75, gain_net_target_bps=2.5)
    s = sm.summary(data_dir=tmp_path)
    assert s["n_resolved"] == 39
    assert s["verdict_at_40"] is None


def test_verdict_pass_at_40_with_fill_075_gain_25(tmp_path):
    _log_n_resolved(tmp_path, 40, fill_strict_frac=0.75, gain_net_target_bps=2.5)
    s = sm.summary(data_dir=tmp_path)
    assert s["n_resolved"] == 40
    assert abs(s["fill_strict_0"] - 0.75) < 1e-9
    assert s["verdict_at_40"] == "PASS"


def test_verdict_fail_at_40_with_fill_060(tmp_path):
    _log_n_resolved(tmp_path, 40, fill_strict_frac=0.60, gain_net_target_bps=2.5)
    s = sm.summary(data_dir=tmp_path)
    assert s["n_resolved"] == 40
    assert abs(s["fill_strict_0"] - 0.60) < 1e-9
    assert s["verdict_at_40"] == "FAIL"  # fill_strict(0) < 0.70 -> FAIL regardless of gain


def test_verdict_frozen_does_not_change_once_more_signals_accrue(tmp_path):
    # First 40 -> PASS territory. Then log 10 MORE bad signals (would drag the
    # LIVE running fill-rate down) — the frozen verdict_at_40 must not move.
    _log_n_resolved(tmp_path, 40, fill_strict_frac=0.75, gain_net_target_bps=2.5)
    s40 = sm.summary(data_dir=tmp_path)
    assert s40["verdict_at_40"] == "PASS"
    for i in range(10):
        sm.record_entry(ts=f"extra{i}", symbol="BTC/USDT", side="BUY", skill_id="s",
                         decision_close=100.0, data_dir=tmp_path)
        # all non-fills, huge chase cost -> would clearly fail if it were re-read.
        sm.resolve_pending(
            "BTC/USDT", _bar(o=110.0, h=111.0, low_=100.05, c=110.0), data_dir=tmp_path
        )
    s50 = sm.summary(data_dir=tmp_path)
    assert s50["n_resolved"] == 50
    assert s50["verdict_at_40"] == "PASS"  # frozen: first 40 never re-read


# ---------------------------------------------------------------------------
# 7) Dashboard block present and renders without exception.
# ---------------------------------------------------------------------------
def test_dashboard_maker_shadow_block_renders_without_exception():
    from hermes.api import _maker_shadow_safe

    out = _maker_shadow_safe()
    assert isinstance(out, dict)
    assert "n_logged" in out
    assert "n_resolved" in out
    assert "verdict_at_40" in out
    assert out.get("prereg") == "research_maker/PREREG_SHADOW_FILL.md"


# ---------------------------------------------------------------------------
# 8) Fail-soft: record_entry must never raise, even with an unwritable data_dir.
# ---------------------------------------------------------------------------
def test_record_entry_is_fail_soft_on_read_only_data_dir(tmp_path):
    ro_dir = tmp_path / "readonly"
    ro_dir.mkdir()
    os.chmod(ro_dir, stat.S_IREAD | stat.S_IEXEC)
    try:
        # Must not raise.
        sm.record_entry(ts="t0", symbol="BTC/USDT", side="BUY", skill_id="s",
                         decision_close=100.0, data_dir=ro_dir)
        sm.resolve_pending("BTC/USDT", _bar(o=99.0, h=101.0, low_=98.0, c=99.5), data_dir=ro_dir)
        out = sm.summary(data_dir=ro_dir)
        assert isinstance(out, dict)
    finally:
        os.chmod(ro_dir, stat.S_IRWXU)


# --------------------------------------------------------------------------- #
# Garde anti-écart (2026-07-18, audit orchestrateur) : la barre de résolution
# doit être LA barre SUIVANTE — jamais une barre plus tardive.
# --------------------------------------------------------------------------- #
# Le défaut qu'elle ferme : sans elle, après un redéploiement (4 ce soir-là) une
# entrée en attente serait résolue contre une barre 2+ périodes plus tard — plus
# de temps pour traverser la limite = fill-rate GONFLÉ, précisément le sens
# d'auto-flatterie que le prereg existe pour empêcher. Un signal dont la barre
# suivante n'a pas été observée n'a PAS de mesure définie par la règle gelée :
# il est VOID (transparent, compté, jamais dans les stats) — ce n'est pas une
# exclusion post-hoc, aucun outcome n'existe à exclure.

_4H = 4 * 60 * 60 * 1000


def test_resolution_against_the_immediate_next_bar_counts(tmp_path):
    sm.record_entry(ts="t0", symbol="BTC/USDT", side="BUY", skill_id="s",
                    decision_close=100.0, decision_bar_ts=1_000 * _4H,
                    data_dir=tmp_path)
    sm.resolve_pending("BTC/USDT", {"open": 100.0, "high": 101.0, "low": 99.0,
                                    "close": 100.5}, bar_ts=1_001 * _4H,
                       data_dir=tmp_path)
    s = sm.summary(data_dir=tmp_path)
    assert s["n_resolved"] == 1 and s.get("n_void", 0) == 0
    assert s["fill_strict_0"] == 1.0


def test_gap_bar_voids_the_entry_instead_of_resolving_it(tmp_path):
    """Barre reçue = décision + 2 périodes (redéploiement au milieu) → VOID,
    jamais un fill — même si la barre AURAIT traversé la limite."""
    sm.record_entry(ts="t0", symbol="BTC/USDT", side="BUY", skill_id="s",
                    decision_close=100.0, decision_bar_ts=1_000 * _4H,
                    data_dir=tmp_path)
    sm.resolve_pending("BTC/USDT", {"open": 100.0, "high": 101.0, "low": 90.0,
                                    "close": 95.0}, bar_ts=1_002 * _4H,
                       data_dir=tmp_path)
    s = sm.summary(data_dir=tmp_path)
    assert s.get("n_void", 0) == 1
    assert s["fill_strict_0"] is None, "une entrée VOID ne doit JAMAIS compter comme fill"


def test_same_or_older_bar_leaves_the_entry_pending(tmp_path):
    """La barre de décision elle-même (double appel dans le même tick) ne résout pas."""
    sm.record_entry(ts="t0", symbol="BTC/USDT", side="BUY", skill_id="s",
                    decision_close=100.0, decision_bar_ts=1_000 * _4H,
                    data_dir=tmp_path)
    sm.resolve_pending("BTC/USDT", {"open": 100.0, "high": 101.0, "low": 90.0,
                                    "close": 95.0}, bar_ts=1_000 * _4H,
                       data_dir=tmp_path)
    s = sm.summary(data_dir=tmp_path)
    assert s["n_resolved"] == 0 and s.get("n_void", 0) == 0
    # puis la vraie barre suivante arrive : résolution normale
    sm.resolve_pending("BTC/USDT", {"open": 100.0, "high": 101.0, "low": 99.0,
                                    "close": 100.5}, bar_ts=1_001 * _4H,
                       data_dir=tmp_path)
    assert sm.summary(data_dir=tmp_path)["n_resolved"] == 1


def test_void_entries_never_enter_the_verdict_window(tmp_path):
    """40 résolus VALIDES + des voids intercalés → le verdict se calcule sur les
    40 valides ; les voids sont comptés à part (transparence, pas d'exclusion cachée)."""
    for i in range(43):
        sm.record_entry(ts=f"t{i}", symbol="BTC/USDT", side="BUY", skill_id="s",
                        decision_close=100.0, decision_bar_ts=(1_000 + 2 * i) * _4H,
                        data_dir=tmp_path)
        gap = i in (1, 7, 20)  # 3 entrées frappées par un écart
        sm.resolve_pending(
            "BTC/USDT",
            {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5},
            bar_ts=(1_000 + 2 * i + (2 if gap else 1)) * _4H,
            data_dir=tmp_path)
    s = sm.summary(data_dir=tmp_path)
    assert s["n_void"] == 3
    assert s["n_resolved"] == 40
    assert s["verdict_at_40"] == "PASS"  # fill 100% > 0.70, gain 3.0 > 2.0
