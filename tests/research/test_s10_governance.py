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
    )
    assert status["locked"] is False
    assert all(c["met"] for c in status["conditions"])


def test_gp_stays_locked_if_any_condition_unmet():
    status = gov.gp_lock_status(
        deflation_trusted=True,
        book_harness_trusted=True,
        validated_orthogonal_feature_count=2,  # one short of the zoo requirement
    )
    assert status["locked"] is True
    assert any(not c["met"] for c in status["conditions"])
