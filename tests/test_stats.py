"""Tests for the statistical-rigor helpers (Probabilistic / Deflated Sharpe).

Formulas follow Bailey & López de Prado, "The Deflated Sharpe Ratio" (2014).
Expected values below are hand-derived, not recomputed from the implementation.
"""

import math

import pytest

from hermes.learning.stats import (
    deflated_sharpe_ratio,
    expected_max_sharpe,
    min_track_record_length,
    probabilistic_sharpe_ratio,
    score_skills,
    sharpe_ratio,
)


# --- sharpe_ratio (per-period, sample std ddof=1) --------------------------
def test_sharpe_ratio_basic():
    # [0.01,0.02,0.01,0.02] -> mean 0.015, sample std 0.0057735 -> 2.5981
    assert sharpe_ratio([0.01, 0.02, 0.01, 0.02]) == pytest.approx(2.5981, rel=1e-3)


def test_sharpe_ratio_degenerate_returns_zero():
    assert sharpe_ratio([]) == 0.0
    assert sharpe_ratio([0.01]) == 0.0          # n < 2
    assert sharpe_ratio([0.01, 0.01, 0.01]) == 0.0  # zero variance


# --- probabilistic_sharpe_ratio --------------------------------------------
def test_psr_is_half_when_observed_sharpe_is_zero():
    # Perfectly alternating, mean exactly 0 -> SR_hat = 0 -> PSR = Phi(0) = 0.5
    rets = [0.02, -0.02] * 6
    assert probabilistic_sharpe_ratio(rets) == pytest.approx(0.5, abs=1e-6)


def test_psr_near_one_for_strong_consistent_edge():
    # mean 0.01, tiny variance, n=30 -> huge SR -> PSR ~ 1.0
    rets = [0.011, 0.009] * 15
    assert probabilistic_sharpe_ratio(rets) > 0.999


def test_psr_bounded_unit_interval():
    for rets in ([0.05, -0.2, 0.3, -0.1, 0.02, 0.4], [-0.1, -0.2, -0.05] * 5):
        p = probabilistic_sharpe_ratio(rets)
        assert 0.0 <= p <= 1.0


# --- expected_max_sharpe (deflation benchmark) -----------------------------
def test_expected_max_sharpe_grows_with_trials():
    e3 = expected_max_sharpe(0.25, 3)
    e10 = expected_max_sharpe(0.25, 10)
    assert 0.0 < e3 < e10

def test_expected_max_sharpe_no_deflation_for_single_trial():
    assert expected_max_sharpe(0.25, 1) == 0.0


# --- deflated_sharpe_ratio --------------------------------------------------
def test_dsr_never_exceeds_psr_under_multiple_trials():
    rets = [0.011, 0.009] * 15  # positive SR
    psr = probabilistic_sharpe_ratio(rets)
    dsr = deflated_sharpe_ratio(rets, sr_variance=0.25, n_trials=10)
    assert dsr <= psr + 1e-9
    assert 0.0 <= dsr <= 1.0


def test_dsr_equals_psr_when_single_trial():
    rets = [0.011, 0.009] * 15
    assert deflated_sharpe_ratio(rets, sr_variance=0.25, n_trials=1) == pytest.approx(
        probabilistic_sharpe_ratio(rets), abs=1e-9
    )


# --- min_track_record_length -----------------------------------------------
def test_mintrl_finite_for_positive_edge_and_inf_otherwise():
    strong = [0.011, 0.009] * 15
    trl = min_track_record_length(strong, sr_star=0.0, prob=0.95)
    assert math.isfinite(trl) and trl > 0
    # No edge (SR <= benchmark) -> infinite track record required.
    flat = [0.02, -0.02] * 6
    assert math.isinf(min_track_record_length(flat, sr_star=0.0, prob=0.95))


# --- score_skills (the weekly re-rank decision core) -----------------------
def test_score_skills_keeps_winner_archives_loser():
    returns_by_skill = {
        "winner": [0.021, 0.019] * 8,    # 16 trades, clear positive edge
        "loser": [-0.019, -0.021] * 8,   # 16 trades, clear negative edge
    }
    out = score_skills(returns_by_skill, min_trades=10)
    assert out["winner"]["archived"] is False
    assert out["loser"]["archived"] is True
    assert out["winner"]["dsr"] > out["loser"]["dsr"]


def test_score_skills_does_not_archive_on_insufficient_track_record():
    # A losing skill with too few trades must NOT be archived (anti-cherry-pick).
    out = score_skills({"young_loser": [-0.02, -0.03, -0.01]}, min_trades=10)
    assert out["young_loser"]["archived"] is False
    assert out["young_loser"]["trades"] == 3
