"""Causal HRP allocator (vector 1.2) — Hierarchical Risk Parity weights computed
forward-only (zero look-ahead).

The recipe is López de Prado (AFML ch.16): correlation-distance d=√((1-rho)/2),
SINGLE-linkage agglomerative clustering, quasi-diagonalisation by the dendrogram
leaf order, then recursive bisection allocating inversely to *cluster* variance.
The value over naive inverse-variance is that HRP accounts for correlation: it
clusters redundant sleeves and diversifies BETWEEN clusters rather than piling
capital into a correlated bloc.

Causality mirrors ``causal_inverse_vol_weights``: the weight on a sleeve in period
t is computed from returns strictly BEFORE t, so the book a trader could have
earned never peeks forward.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from hermes.validation.book import (
    _hrp_weights,
    align_sleeves,
    causal_hrp_weights,
    combine_book,
    weighting_comparison,
)


# --------------------------------------------------------------------------- #
# Static HRP recipe (_hrp_weights on a covariance matrix)
# --------------------------------------------------------------------------- #
def test_hrp_two_uncorrelated_sleeves_reduces_to_inverse_variance():
    # With two assets the single bisection IS inverse-variance: weight ∝ 1/var.
    cov = np.array([[0.04, 0.0], [0.0, 0.01]])
    w = _hrp_weights(cov)
    np.testing.assert_allclose(w, [0.2, 0.8])  # 0.01/0.05 , 0.04/0.05


def test_hrp_weights_sum_to_one_and_are_nonneg():
    rng = np.random.default_rng(5)
    x = rng.normal(0, 0.01, (500, 4))
    cov = np.cov(x, rowvar=False, ddof=1)
    w = _hrp_weights(cov)
    assert w.shape == (4,)
    np.testing.assert_allclose(w.sum(), 1.0)
    assert (w >= 0).all()


def test_hrp_top_split_favours_the_lower_variance_cluster():
    """The core HRP property: two correlated clusters — one calm, one wild — get
    capital allocated INVERSELY to cluster variance, so the calm cluster takes the
    larger half. (Naive inverse-variance has no notion of a cluster.)"""
    rng = np.random.default_rng(7)
    n = 800
    calm_factor = rng.normal(0, 0.003, n)  # a, b share this -> tight, low-vol cluster
    wild_factor = rng.normal(0, 0.020, n)  # c, d share this -> tight, high-vol cluster
    a = calm_factor + rng.normal(0, 0.0003, n)
    b = calm_factor + rng.normal(0, 0.0003, n)
    c = wild_factor + rng.normal(0, 0.0020, n)
    d = wild_factor + rng.normal(0, 0.0020, n)
    aligned = align_sleeves(
        {"a": pd.Series(a), "b": pd.Series(b), "c": pd.Series(c), "d": pd.Series(d)}
    )
    cov = np.cov(aligned.to_numpy(), rowvar=False, ddof=1)
    w = _hrp_weights(cov)
    cols = list(aligned.columns)
    calm = w[cols.index("a")] + w[cols.index("b")]
    wild = w[cols.index("c")] + w[cols.index("d")]
    assert calm > 0.5 > wild


# --------------------------------------------------------------------------- #
# Causal expanding wrapper (causal_hrp_weights)
# --------------------------------------------------------------------------- #
def test_causal_hrp_warmup_falls_back_to_equal_weight():
    n = 20
    a = pd.Series([0.01, -0.01] * (n // 2))
    b = pd.Series([0.02, -0.02] * (n // 2))
    w = causal_hrp_weights(align_sleeves({"a": a, "b": b}), min_periods=8)
    np.testing.assert_allclose(w.iloc[0].to_numpy(), [0.5, 0.5])


def test_causal_hrp_rows_sum_to_one():
    rng = np.random.default_rng(1)
    n = 120
    sl = {k: pd.Series(rng.normal(0, 0.01, n)) for k in ("a", "b", "c", "d")}
    w = causal_hrp_weights(align_sleeves(sl), min_periods=8)
    np.testing.assert_allclose(w.sum(axis=1).to_numpy(), np.ones(n))


def test_causal_hrp_weights_do_not_peek_forward():
    n = 40
    idx = list(range(n))
    a = pd.Series([0.005 if i % 2 == 0 else -0.005 for i in range(n)], index=idx)
    b = pd.Series([0.005 if i % 2 == 1 else -0.005 for i in range(n)], index=idx)
    w_base = causal_hrp_weights(align_sleeves({"a": a, "b": b}), min_periods=8)
    a2 = a.copy()
    a2.iloc[20] = 5.0  # a future-relative spike at an interior period
    w_spk = causal_hrp_weights(align_sleeves({"a": a2, "b": b}), min_periods=8)
    # Weight at period t uses ONLY returns strictly before t: rows 0..20 unchanged
    # (the weight at t=20 is set from data 0..19, before the spike at 20).
    np.testing.assert_allclose(w_base.iloc[:21].to_numpy(), w_spk.iloc[:21].to_numpy())
    # ...and the spike MUST move a later weight (the window really uses its input).
    assert not np.allclose(w_base.iloc[21].to_numpy(), w_spk.iloc[21].to_numpy())


def test_combine_book_supports_hrp_weighting():
    rng = np.random.default_rng(2)
    n = 120
    sl = {k: pd.Series(rng.normal(0.0005, 0.01, n)) for k in ("a", "b", "c")}
    book = combine_book(sl, weighting="hrp", min_periods=8)
    assert len(book) == n
    assert np.isfinite(book.to_numpy()).all()


# --------------------------------------------------------------------------- #
# HRP vs inverse-vol comparative (the §1.2 KILL: adopt HRP only if it beats inv-vol)
# --------------------------------------------------------------------------- #
def test_weighting_comparison_reports_per_fold_and_verdict():
    rng = np.random.default_rng(3)
    n = 240
    sl = {k: pd.Series(rng.normal(0.0003, 0.01, n)) for k in ("a", "b", "c")}
    cmp = weighting_comparison(sl, n_groups=6, k_test=2, min_periods=8)
    assert cmp["n_folds"] > 0
    assert 0 <= cmp["hrp_better_folds"] <= cmp["n_folds"]
    assert isinstance(cmp["adopt_hrp"], bool)
    assert "inv_vol_mean_oos_sharpe" in cmp
    assert "hrp_mean_oos_sharpe" in cmp
    assert len(cmp["per_fold"]) == cmp["n_folds"]
