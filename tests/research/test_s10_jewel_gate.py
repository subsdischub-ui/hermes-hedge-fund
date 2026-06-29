"""Tests for the JEWEL combination gate SPEC (src/hermes/research/s10_jewel_gate.py).

The "jewel" is the eventual combination of surviving market-neutral pillars into one book. The
MERGE IS DEFERRED — S2 delivered no market-neutral pillar — so this gate is a SPEC: it is coded and
tested ON SYNTHETIC FIXTURES ONLY and is never run on a real book. The gate enforces that a
combination is treated as ONE pre-registered trial and must clear, on the COMBINED series:
  * a combined null (random-sign PER sleeve, autocorrelation preserved),
  * CPCV / PBO / DSR at the true (book + 1) trial count,
  * LOO-Δ-Sharpe > 0 PER sleeve,
  * orthogonality judged on CO-ACTIVE weeks,
  * and the Task-4 OOS-forward precondition.
"""

from __future__ import annotations

import numpy as np

from hermes.research import s10_governance as gov
from hermes.research import s10_jewel_gate as jewel


def _clean_forward():
    return gov.ForwardOOSEvidence(
        forward_sharpe=0.30, n_trades=60, window_days=200, regime_real=True, leak_replayed=True
    )


def _independent_positive_sleeves(k, n, sharpe_per_trade, rng):
    """k independent sleeves each with a small positive per-trade drift (a clean synthetic book)."""
    out = {}
    for i in range(k):
        noise = rng.standard_normal(n)
        out[f"pillar_{i}"] = (sharpe_per_trade + noise) * 0.01
    return out


# --------------------------------------------------------------------------- #
# the gate is a SPEC — it refuses to run without confirmed pillars
# --------------------------------------------------------------------------- #
def test_jewel_gate_rejects_with_no_confirmed_pillar():
    rng = np.random.default_rng(0)
    sleeves = _independent_positive_sleeves(5, 400, 0.25, rng)
    res = jewel.jewel_combination_gate(
        sleeves, book=gov.Book.CRYPTO, book_trial_count=1891, forward=_clean_forward(),
        confirmed_pillars=[],  # NONE today — the merge is deferred
    )
    assert res["verdict"] == "REJECT"
    assert "confirmed_pillars" in res["failed"]
    assert "deferred" in res["reasons"]["confirmed_pillars"].lower() or \
        "no confirmed" in res["reasons"]["confirmed_pillars"].lower()


def test_jewel_gate_counts_combination_as_one_preregistered_trial():
    rng = np.random.default_rng(1)
    sleeves = _independent_positive_sleeves(5, 400, 0.25, rng)
    res = jewel.jewel_combination_gate(
        sleeves, book=gov.Book.CRYPTO, book_trial_count=1891, forward=_clean_forward(),
        confirmed_pillars=list(sleeves),
    )
    assert res["metrics"]["n_trials"] == 1892  # 1891 + the combination itself = one new trial


# --------------------------------------------------------------------------- #
# the combined null + the OOS-forward precondition
# --------------------------------------------------------------------------- #
def test_jewel_gate_random_sign_null_rejects_pure_noise():
    rng = np.random.default_rng(2)
    sleeves = {f"noise_{i}": rng.standard_normal(400) * 0.01 for i in range(5)}  # zero-mean noise
    res = jewel.jewel_combination_gate(
        sleeves, book=gov.Book.CRYPTO, book_trial_count=10, forward=_clean_forward(),
        confirmed_pillars=list(sleeves),
    )
    assert res["verdict"] == "REJECT"
    assert "combined_sign_null" in res["failed"]


def test_jewel_gate_requires_oos_forward():
    rng = np.random.default_rng(3)
    sleeves = _independent_positive_sleeves(6, 600, 0.30, rng)
    res = jewel.jewel_combination_gate(
        sleeves, book=gov.Book.CRYPTO, book_trial_count=5, forward=None,  # Forward-Shadow not exited
        confirmed_pillars=list(sleeves),
    )
    assert res["verdict"] == "REJECT"
    assert "oos_forward_present" in res["failed"]


def test_jewel_gate_requires_positive_loo_per_sleeve():
    rng = np.random.default_rng(4)
    sleeves = _independent_positive_sleeves(4, 500, 0.30, rng)
    # add a sleeve that loses money with low vol → high inv-vol weight, drags the book → LOO < 0
    sleeves["drag"] = (-0.30 + 0.2 * rng.standard_normal(500)) * 0.01
    res = jewel.jewel_combination_gate(
        sleeves, book=gov.Book.CRYPTO, book_trial_count=5, forward=_clean_forward(),
        confirmed_pillars=list(sleeves),
    )
    assert res["verdict"] == "REJECT"
    assert "loo_per_sleeve" in res["failed"]


def test_jewel_gate_orthogonality_uses_coactive_weeks():
    """Two pillars colinear on co-active weeks but sparse must FAIL orthogonality even though the
    full zero-filled grid would call them orthogonal (R3 / S5)."""
    rng = np.random.default_rng(5)
    n = 500
    sleeves = _independent_positive_sleeves(4, n, 0.30, rng)
    a = np.zeros(n)
    b = np.zeros(n)
    shared = 0.30 + rng.standard_normal(40)
    a[:40] = shared * 0.01
    b[:40] = shared * 0.01  # identical on the 40 co-active weeks
    a[40:270] = (0.30 + rng.standard_normal(230)) * 0.01  # active alone afterwards
    b[270:] = (0.30 + rng.standard_normal(230)) * 0.01
    sleeves["twin_a"] = a
    sleeves["twin_b"] = b
    res = jewel.jewel_combination_gate(
        sleeves, book=gov.Book.CRYPTO, book_trial_count=5, forward=_clean_forward(),
        confirmed_pillars=list(sleeves),
    )
    assert "orthogonality_coactive" in res["failed"]


def test_jewel_gate_dsr_has_teeth_at_true_count():
    """S10's core lever: with the searched-candidate Sharpe dispersion passed explicitly, the DSR
    deflation at the TRUE book count (1891) must BITE — a moderate book fails deflated_sharpe even
    though it would saturate the DSR at a tiny trial count. Proves the deflation is not defanged."""
    rng = np.random.default_rng(30)
    sleeves = _independent_positive_sleeves(6, 600, 0.25, rng)  # moderate book
    res = jewel.jewel_combination_gate(
        sleeves, book=gov.Book.CRYPTO, book_trial_count=1891, forward=_clean_forward(),
        confirmed_pillars=list(sleeves),
        sr_variance=0.04,  # realistic dispersion of the 1891 searched candidates (NOT the 6 chosen)
    )
    assert "deflated_sharpe" in res["failed"]


def test_jewel_gate_sign_null_floor_requires_six_sleeves():
    """The combined sign-null floor is 2/(2^K+1): a strong but only-5-sleeve book cannot beat
    p<0.05 (floor 2/33 = 0.0606) — by design, sign-luck dominates below K=6."""
    rng = np.random.default_rng(31)
    sleeves = _independent_positive_sleeves(5, 800, 0.40, rng)  # strong, but only 5 sleeves
    res = jewel.jewel_combination_gate(
        sleeves, book=gov.Book.CRYPTO, book_trial_count=5, forward=_clean_forward(),
        confirmed_pillars=list(sleeves), sr_variance=0.001,
    )
    assert "combined_sign_null" in res["failed"]
    assert res["metrics"]["combined_sign_null_p"] >= 2 / (2**5 + 1) - 1e-9


def test_jewel_gate_promotes_a_clean_synthetic_book():
    """SYNTHETIC happy path: 6 independent strong DISTINGUISHABLE pillars + clean forward → PROMOTE.
    This is a fixture, NOT a real pillar; the gate is never executed on a live book (merge deferred).
    The pillars carry a modest strength gradient so they are not interchangeable clones (PBO low)."""
    rng = np.random.default_rng(20)
    n = 1200
    sleeves = {}
    for i in range(6):
        sharpe_i = 0.34 + 0.03 * i  # distinguishable strengths 0.34..0.49
        sleeves[f"pillar_{i}"] = (sharpe_i + rng.standard_normal(n)) * 0.01
    res = jewel.jewel_combination_gate(
        sleeves, book=gov.Book.CRYPTO, book_trial_count=5, forward=_clean_forward(),
        confirmed_pillars=list(sleeves),
    )
    assert res["verdict"] == "PROMOTE", res["failed"]
