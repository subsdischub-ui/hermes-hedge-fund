"""Tests for the Stream 10 governance machinery (src/hermes/research/s10_governance.py).

Stream 10 is the adversarial deflation / data-governance guard. These tests pin the
five non-negotiables the mandate names:
  10.1  a HARDENED, per-book promotion bar wired to the TRUE trial count (FX stricter);
        the Monte-Carlo random-entry null is a FLOOR, never the bar; the LOO-Δ-Sharpe
        (book contribution) decides, never the solo Sharpe.
  10.2  a queryable dead-ends ledger that returns BLOCKED-BY-LEDGER for a re-spend on a
        falsified hypothesis lacking a NEW data axis, and an acquisition map that governs
        data-$ (CoinGlass acquired, Amberdata arbitrate, Databento/alt-data DO-NOT-SPEND).
  10.3  an explicit GP/D1 lock that stays LOCKED until its conditions are met.
"""

from __future__ import annotations

import numpy as np
import pytest

from hermes.research import s10_governance as gov


# --------------------------------------------------------------------------- #
# 10.1 — hardened per-book promotion bar
# --------------------------------------------------------------------------- #
def test_fx_bar_is_strictly_harder_than_crypto_bar():
    """Cheap FX cost lets more candidates clear arithmetic → the overfitting wall must
    be tighter on the FX book than the crypto book (mandate 10.1)."""
    crypto = gov.bar_for_book(gov.Book.CRYPTO)
    fx = gov.bar_for_book(gov.Book.FX_METALS)
    assert fx.max_pbo < crypto.max_pbo          # PBO stricter on FX
    assert fx.min_dsr > crypto.min_dsr          # deflated-Sharpe confidence stricter on FX
    # The floor (random-entry null, generalization) is identical across books.
    assert fx.max_mc_null_p == crypto.max_mc_null_p
    assert fx.min_instruments == crypto.min_instruments


def test_cross_book_bar_is_at_least_as_strict_as_fx():
    cross = gov.bar_for_book(gov.Book.CROSS)
    fx = gov.bar_for_book(gov.Book.FX_METALS)
    assert cross.max_pbo <= fx.max_pbo
    assert cross.min_dsr >= fx.min_dsr


def _clean_passing_evidence(**overrides):
    base = dict(
        name="hypothetical_clean_sleeve",
        book=gov.Book.CRYPTO,
        mc_null_p=0.01,
        n_instruments=3,
        beta_controlled=True,
        dsr=0.98,
        pbo=0.20,
        max_abs_rho=0.10,
        loo_delta_sharpe=0.05,
        oos_sharpe=0.45,
        n_trials=50,
    )
    base.update(overrides)
    return gov.SleeveEvidence(**base)


def test_clean_book_is_promoted():
    verdict = gov.PromotionBar().evaluate(_clean_passing_evidence())
    assert verdict.verdict is gov.Verdict.PROMOTE
    assert verdict.failed == []


def test_real_fx_book_of_four_is_rejected():
    """The §11(b) honest negative: the 4-sleeve FX book (DSR 0.287, PBO 0.782, no sleeve
    clears the MC-null) must be REJECTED, and the failure must name deflation + the floor."""
    ev = gov.SleeveEvidence(
        name="fx_book_of_4_B1B2B3B5",
        book=gov.Book.FX_METALS,
        mc_null_p=0.059,         # best sleeve never cleared p<0.05
        n_instruments=1,         # single-asset artifacts; fails generalization
        beta_controlled=False,   # secular-bull-beta unmasked
        dsr=0.287,
        pbo=0.782,
        max_abs_rho=0.09,        # orthogonal — but orthogonal NOISE
        loo_delta_sharpe=-0.0015,
        oos_sharpe=0.02,
        n_trials=24,
    )
    verdict = gov.PromotionBar().evaluate(ev)
    assert verdict.verdict is gov.Verdict.REJECT
    assert "deflated_sharpe" in verdict.failed
    assert "pbo" in verdict.failed
    assert "mc_null_floor" in verdict.failed


def test_mc_null_is_a_floor_not_the_bar():
    """A sleeve can BEAT the random-entry null and still be rejected by deflation. Encodes
    'random-entry null surviving = plancher, pas barre'."""
    ev = _clean_passing_evidence(
        book=gov.Book.FX_METALS,
        mc_null_p=0.001,   # smashes the floor
        dsr=0.60,          # but below the FX deflation bar (0.975)
        pbo=0.30,
    )
    verdict = gov.PromotionBar().evaluate(ev)
    assert verdict.verdict is gov.Verdict.REJECT
    assert "deflated_sharpe" in verdict.failed
    assert "mc_null_floor" in verdict.passed  # it DID clear the floor


def test_negative_loo_delta_sharpe_blocks_promotion():
    """The unit is the BOOK: a sleeve that does not improve the book (LOO-Δ-Sharpe ≤ 0) is
    rejected even with everything else perfect. Encodes 'jamais par le Sharpe solo'."""
    ev = _clean_passing_evidence(loo_delta_sharpe=-0.01)
    verdict = gov.PromotionBar().evaluate(ev)
    assert verdict.verdict is gov.Verdict.REJECT
    assert "loo_delta_sharpe" in verdict.failed


def test_single_instrument_win_is_rejected():
    ev = _clean_passing_evidence(n_instruments=1)
    verdict = gov.PromotionBar().evaluate(ev)
    assert verdict.verdict is gov.Verdict.REJECT
    assert "generalization" in verdict.failed


def test_min_oos_sharpe_zero_still_enforces_nonnegative():
    """A bar with min_oos_sharpe=0.0 must ENFORCE 'oos_sharpe >= 0', not silently disable the gate
    (the 0.0 footgun): a sleeve with a negative OOS Sharpe cannot pass."""
    bar = gov.PromotionThresholds(max_pbo=0.5, min_dsr=0.95, min_oos_sharpe=0.0)
    pb = gov.PromotionBar({gov.Book.CRYPTO: bar})
    verdict = pb.evaluate(_clean_passing_evidence(oos_sharpe=-5.0))
    assert "oos_sharpe" in verdict.failed


def test_deflated_sharpe_collapses_as_true_trial_count_rises():
    """The core 10.1 mechanism: the SAME track record deflates harder as the true number of
    trials grows — so under-counting trials fraudulently inflates the DSR."""
    rng = np.random.default_rng(0)
    returns = (0.002 + 0.01 * rng.standard_normal(500)).tolist()
    sr_var = 0.04
    dsr_few = gov.deflated_sharpe_at_true_count(returns, sr_var, n_trials=1)
    dsr_many = gov.deflated_sharpe_at_true_count(returns, sr_var, n_trials=1000)
    assert dsr_few > dsr_many


# --------------------------------------------------------------------------- #
# per-book TrialRegistry
# --------------------------------------------------------------------------- #
def test_registry_counts_are_isolated_per_book(tmp_path):
    reg = gov.BookTrialRegistry(tmp_path)
    reg.bump(gov.Book.CRYPTO, 10, source="funding_sweep")
    assert reg.count(gov.Book.CRYPTO) == 10
    assert reg.count(gov.Book.FX_METALS) == 0


def test_registry_persists_across_instances(tmp_path):
    gov.BookTrialRegistry(tmp_path).bump(gov.Book.FX_METALS, 7, source="b_frontiers")
    reloaded = gov.BookTrialRegistry(tmp_path)
    assert reloaded.count(gov.Book.FX_METALS) == 7


def test_registry_records_provenance(tmp_path):
    reg = gov.BookTrialRegistry(tmp_path)
    reg.bump(gov.Book.CRYPTO, 648, source="funding_discovery_sweep")
    reg.bump(gov.Book.CRYPTO, 1024, source="options_surface_grid")
    prov = reg.provenance(gov.Book.CRYPTO)
    sources = {p["source"] for p in prov}
    assert "funding_discovery_sweep" in sources and "options_surface_grid" in sources
    assert reg.count(gov.Book.CRYPTO) == 648 + 1024


def test_registry_seed_defaults_is_idempotent(tmp_path):
    reg = gov.BookTrialRegistry(tmp_path)
    reg.seed_defaults()
    first_crypto = reg.count(gov.Book.CRYPTO)
    first_fx = reg.count(gov.Book.FX_METALS)
    reg.seed_defaults()  # second seed must NOT double-count
    assert reg.count(gov.Book.CRYPTO) == first_crypto
    assert reg.count(gov.Book.FX_METALS) == first_fx
    assert first_crypto > 0 and first_fx > 0


# --------------------------------------------------------------------------- #
# 10.2 — dead-ends ledger
# --------------------------------------------------------------------------- #
def test_ledger_blocks_known_dead_end_without_new_axis():
    ledger = gov.DeadEndsLedger.load()
    res = ledger.check("gold trend following")
    assert res["verdict"] is gov.LedgerVerdict.BLOCKED
    assert "bull-beta" in (res["why_dead"] + res["reason"]).lower()
    assert res["revive_condition"]


def test_ledger_blocks_directional_fx_respend():
    ledger = gov.DeadEndsLedger.load()
    res = ledger.check("DFII10 real-rate shock directional gold signal")
    assert res["verdict"] is gov.LedgerVerdict.BLOCKED


def test_ledger_allows_respend_with_genuinely_new_data_axis():
    """A falsified hypothesis may be retried ONLY with a NEW data axis (e.g. the OHLCV
    liquidation proxy died, but real CoinGlass liquidations ≥2019 is a new axis)."""
    ledger = gov.DeadEndsLedger.load()
    res = ledger.check(
        "liquidation cascade reversion",
        new_data_axis="coinglass real historical liquidations >=2019 with bull/bear separation",
    )
    assert res["verdict"] is gov.LedgerVerdict.ALLOWED_NEW_AXIS
    assert "re-gate" in res["reason"].lower() or "regate" in res["reason"].lower()


def test_ledger_passes_through_a_novel_proposal():
    ledger = gov.DeadEndsLedger.load()
    res = ledger.check("a never-before-tested orthogonal central-bank-demand divergence sleeve")
    assert res["verdict"] is gov.LedgerVerdict.NOT_A_DEAD_END


def test_ledger_does_not_overblock_on_short_substring_tags():
    """Bare substring tags like 'd1' / 'cot ' must NOT false-block unrelated proposals that merely
    contain the bigram (the governance layer's worst failure mode = suppressing legitimate work)."""
    ledger = gov.DeadEndsLedger.load()
    assert ledger.check("a btcusd1h scalper concept")["verdict"] is gov.LedgerVerdict.NOT_A_DEAD_END
    assert ledger.check("an apricot futures basis idea")["verdict"] is gov.LedgerVerdict.NOT_A_DEAD_END
    # but the genuine D1 feature-zoo proposal is STILL blocked via its specific multi-word tags
    res = ledger.check("D1 ML feature-zoo combination over 8 features")
    assert res["verdict"] is gov.LedgerVerdict.BLOCKED


def test_ledger_has_the_canonical_falsified_families():
    names = " ".join(e.name.lower() for e in gov.DeadEndsLedger.load().entries())
    for needle in ("gold trend", "cointegration", "maker", "d1"):
        assert needle in names


def test_ledger_blocks_other_stream_recorded_negatives():
    """Every dated negative from the merged streams (S2-S8 outcomes) enters the ledger so no
    team re-spends an agent on it (mandate 10.2)."""
    ledger = gov.DeadEndsLedger.load()
    for proposal in (
        "top-LSR contrarian positioning sleeve",          # S2.1 — the −0.27 ρ was a 30d mirage
        "Fed hawk/dove tone drift basket",                 # S7 — dovish-beta, lexicon non-replication
        "VRP-DVOL sizing gate on funding_squeeze",         # S4.3 — does not lift S2 (shuffle-p 0.14)
        "funding_squeeze on 1d bars universe expansion",   # S3 — does not generalize off 4h
    ):
        assert ledger.check(proposal)["verdict"] is gov.LedgerVerdict.BLOCKED, proposal


# --------------------------------------------------------------------------- #
# 10.2 — acquisition map (data-$ governance)
# --------------------------------------------------------------------------- #
def test_acquisition_coinglass_is_acquired():
    amap = gov.AcquisitionMap.load()
    assert amap.status("CoinGlass") is gov.DataStatus.ACQUIRED


def test_acquisition_databento_is_do_not_spend():
    amap = gov.AcquisitionMap.load()
    res = amap.can_spend("Databento")
    assert res["allowed"] is False
    assert "do_not_spend" in res["status"].lower() or res["status"] == gov.DataStatus.DO_NOT_SPEND.value


def test_acquisition_amberdata_is_arbitrate_recommend_no():
    amap = gov.AcquisitionMap.load()
    assert amap.status("Amberdata") is gov.DataStatus.NOT_ACQUIRED
    res = amap.can_spend("Amberdata")
    # Not a hard block, but the standing recommendation is NO until Stream 4 demonstrates a
    # backtestable forward edge from the 25Δ skew / GEX surface.
    assert res["allowed"] is False
    assert "stream 4" in res["reason"].lower() or "s4" in res["reason"].lower()


def test_acquisition_free_sources_need_no_spend():
    amap = gov.AcquisitionMap.load()
    for src in ("FRED", "WGC", "GLD"):
        assert amap.status(src) in (gov.DataStatus.FREE, gov.DataStatus.ACQUIRED)


# --------------------------------------------------------------------------- #
# 10.3 — GP / D1 lock
# --------------------------------------------------------------------------- #
def test_gp_is_locked_by_default():
    status = gov.gp_lock_status()
    assert status["locked"] is True
    # The lock cites the D1 NO_EDGE evidence.
    assert status["evidence"]["verdict"] == "NO_EDGE"
    assert status["evidence"]["mc_null_p"] == pytest.approx(1.0)


def test_gp_unlocks_only_when_all_conditions_met():
    status = gov.gp_lock_status(
        deflation_trusted=True,
        book_harness_trusted=True,
        validated_orthogonal_feature_count=3,
        forward_validated_candidate=True,  # Task 4 — OOS-forward precondition now gates GP too
    )
    assert status["locked"] is False
    assert all(c["met"] for c in status["conditions"])


def test_gp_stays_locked_if_any_condition_unmet():
    status = gov.gp_lock_status(
        deflation_trusted=True,
        book_harness_trusted=True,
        validated_orthogonal_feature_count=2,  # one short of the zoo requirement
        forward_validated_candidate=True,
    )
    assert status["locked"] is True
    assert any(not c["met"] for c in status["conditions"])


def test_gp_locked_when_no_forward_validated_candidate_today():
    """v-next: even with deflation+harness+zoo trusted, GP stays locked while NO sleeve has
    cleared the OOS-forward precondition (funding_squeeze unconfirmed, cross-sectional dead)."""
    status = gov.gp_lock_status(
        deflation_trusted=True,
        book_harness_trusted=True,
        validated_orthogonal_feature_count=3,
        forward_validated_candidate=False,  # the reality today
    )
    assert status["locked"] is True
    # the lock must surface that there is no candidate sleeve right now
    assert status["candidate_sleeves_today"] == []


# =========================================================================== #
# v-next Task 1 — SHARED, TRACKED registry location (not gitignored per-worktree)
# =========================================================================== #
def test_governance_registry_dir_is_tracked_not_gitignored():
    """The canonical registry state must live in a COMMITTED dir (root governance/), not the
    gitignored per-worktree research_streams/S10/ that made the count un-authoritative."""
    d = gov.governance_registry_dir()
    assert d.name == "governance"
    assert "research_streams" not in str(d)
    assert not d.name.startswith("research_")  # /research_*/ is the gitignore glob


def test_book_trial_registry_defaults_to_shared_tracked_dir():
    reg = gov.BookTrialRegistry()  # no arg → the canonical shared dir, NOT settings.data_dir
    assert reg.path.parent == gov.governance_registry_dir()
    assert reg.path.name == "trials_by_book.json"


def test_crypto_embedded_seed_is_1891_after_s2(tmp_path):
    """The EMBEDDED code seed = 1869 (S0-era) + 22 (S2 v-next: 4 pre-reg + 18 exploratory) = 1891.
    The +1 forward XS-reversion pre-reg lives only in the committed governance/ JSON (bump,
    not seed) → the canonical committed count is 1892, asserted by test_committed_canonical_count_is_1892."""
    reg = gov.BookTrialRegistry(tmp_path)
    reg.seed_defaults()
    assert reg.count(gov.Book.CRYPTO) == 1891
    # the FX and cross books are untouched by S2
    assert reg.count(gov.Book.FX_METALS) == 53
    assert reg.count(gov.Book.CROSS) == 5


def test_committed_canonical_count_is_1892():
    """The MIGRATED, committed governance/trials_by_book.json pins the canonical crypto count
    (1869 seed + 22 S2 v-next + 1 forward XS-reversion pre-reg = 1892 after the full round-2 merge)."""
    reg = gov.BookTrialRegistry()  # reads the committed governance/ JSON
    assert reg.count(gov.Book.CRYPTO) == 1892


def test_governance_snapshots_are_synced_with_embedded_defaults():
    """The committed governance/*.json snapshots are generated FROM the embedded canonical
    defaults; a drift between code-as-truth and the shared serialized state is a governance bug."""
    import json as _json

    d = gov.governance_registry_dir()
    dead = _json.loads((d / "dead_ends_ledger.json").read_text())
    amap = _json.loads((d / "acquisition_map.json").read_text())
    assert {e["name"] for e in dead} == {e.name for e in gov.DEFAULT_DEAD_ENDS}
    assert {s["source"] for s in amap} == {s.source for s in gov.DEFAULT_ACQUISITION_MAP}
    # the 4 S2 negatives must be present in the committed snapshot
    s2_names = {e["name"] for e in dead if "Stream 2 v-next" in e["date"]}
    assert len(s2_names) == 4


# =========================================================================== #
# v-next Task 3 — registry updates: the 4 S2 negatives + acquisition correction
# =========================================================================== #
def test_ledger_blocks_the_four_s2_negatives():
    """The 4 S2 v-next cross-sectional negatives enter the ledger so no team re-spends them."""
    ledger = gov.DeadEndsLedger.load()
    for proposal in (
        "cross-sectional funding continuation sleeve",  # 2A — wrong-way (XS funding mean-reverts)
        "volatility dispersion cross-sectional sleeve",  # 2D — no edge, cost-eaten
        "open-interest growth cross-sectional sleeve",  # 2B — data-gated / under-powered, 7 coins
        "cross-venue funding dispersion sleeve",  # 2.3 — no edge, 5 coins, Sharpe -0.44
    ):
        assert ledger.check(proposal)["verdict"] is gov.LedgerVerdict.BLOCKED, proposal


def test_2a_funding_xs_records_wrong_way_mechanism():
    """2A's kill reason must name the wrong-way sign (XS funding mean-reverts) AND the cost."""
    ledger = gov.DeadEndsLedger.load()
    res = ledger.check("cross-sectional funding continuation sleeve")
    blob = (res["why_dead"] + res.get("kill_metric", "")).lower()
    assert "mean-revert" in blob or "wrong-way" in blob or "wrong way" in blob
    assert "16.85" in blob or "cost" in blob


def test_acquisition_coinglass_notes_multi_venue_depth():
    """CoinGlass must now be described as DEEP multi-venue (OKX/Bybit→2020, Bitget→2021),
    correcting the stale 'Binance-only' framing that gated S2.3/S9."""
    amap = gov.AcquisitionMap.load()
    reason = amap.can_spend("CoinGlass")["reason"].lower()
    assert "multi-venue" in reason or "multi venue" in reason
    assert "okx" in reason or "bybit" in reason or "bitget" in reason


# =========================================================================== #
# v-next Task 3 — funding_squeeze status DOWNGRADE
# =========================================================================== #
def test_funding_squeeze_is_downgraded_not_confirmed():
    """Status moves from 'validated survivor' to 'WEAKLY SUPPORTED, not robustly confirmed':
    deflation re-gates REJECT at the true count AND S2 4h cluster-aware finds it NS."""
    st = gov.funding_squeeze_status()
    assert st["confirmed"] is False
    assert st["status"] == "WEAKLY_SUPPORTED"
    # the convergent evidence must be carried (bull macro-event t=1.50 NS; per-trade Sharpe weak)
    blob = str(st).lower()
    assert "1.50" in blob or "1.5" in blob
    assert "0.16" in blob or "0.24" in blob  # per-trade Sharpe real-but-weak band


# =========================================================================== #
# v-next Task 4 — OOS-FORWARD as a HARD precondition of promotion
# =========================================================================== #
def test_in_sample_pass_but_no_forward_is_rejected():
    """A sleeve may pass the full in-sample gate yet NOT be promotable without clean accumulated
    OOS-forward (the Forward-Shadow exit). S0 survivorship + S2 single-frozen-lake establish that
    no in-sample result, however clean, can CONFIRM an edge."""
    ev = _clean_passing_evidence()
    assert gov.PromotionBar().evaluate(ev).verdict is gov.Verdict.PROMOTE  # in-sample screen passes
    v = gov.PromotionBar().evaluate_with_forward(ev, None)  # but no forward evidence
    assert v.verdict is gov.Verdict.REJECT
    assert "oos_forward_present" in v.failed


def test_clean_in_sample_plus_clean_forward_promotes():
    ev = _clean_passing_evidence()
    fwd = gov.ForwardOOSEvidence(
        forward_sharpe=0.30, n_trades=40, window_days=180, regime_real=True, leak_replayed=True
    )
    v = gov.PromotionBar().evaluate_with_forward(ev, fwd)
    assert v.verdict is gov.Verdict.PROMOTE


def test_forward_must_be_real_regime_and_leak_replayed():
    ev = _clean_passing_evidence()
    not_real = gov.ForwardOOSEvidence(
        forward_sharpe=0.30, n_trades=40, window_days=180, regime_real=False, leak_replayed=True
    )
    v1 = gov.PromotionBar().evaluate_with_forward(ev, not_real)
    assert v1.verdict is gov.Verdict.REJECT
    assert "oos_forward_regime_real" in v1.failed

    no_replay = gov.ForwardOOSEvidence(
        forward_sharpe=0.30, n_trades=40, window_days=180, regime_real=True, leak_replayed=False
    )
    v2 = gov.PromotionBar().evaluate_with_forward(ev, no_replay)
    assert v2.verdict is gov.Verdict.REJECT
    assert "oos_forward_leak_replayed" in v2.failed


def test_forward_must_have_enough_trades_and_window():
    ev = _clean_passing_evidence()
    thin = gov.ForwardOOSEvidence(
        forward_sharpe=0.30, n_trades=3, window_days=10, regime_real=True, leak_replayed=True
    )
    v = gov.PromotionBar().evaluate_with_forward(ev, thin)
    assert v.verdict is gov.Verdict.REJECT
    assert "oos_forward_trades" in v.failed or "oos_forward_window" in v.failed


def test_forward_does_not_rescue_a_failing_in_sample_sleeve():
    """The forward precondition is ADDITIVE, never a bypass: a sleeve that fails the in-sample
    deflation bar is rejected even with a pristine forward record."""
    ev = _clean_passing_evidence(book=gov.Book.FX_METALS, dsr=0.50)  # below FX bar 0.975
    fwd = gov.ForwardOOSEvidence(
        forward_sharpe=0.9, n_trades=200, window_days=365, regime_real=True, leak_replayed=True
    )
    v = gov.PromotionBar().evaluate_with_forward(ev, fwd)
    assert v.verdict is gov.Verdict.REJECT
    assert "deflated_sharpe" in v.failed
