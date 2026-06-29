"""Tests for the combined-book validation harness (STAGE A / §7 combination thesis).

The unit of analysis is the BOOK, not the signal: heterogeneous sleeves (different
cadences, instruments) are aligned onto a common mark grid, combined with CAUSAL
(no-look-ahead) weights, and judged by the book's CPCV/PBO/DSR plus each sleeve's
leave-one-out Δ-Sharpe contribution and its orthogonality to the rest.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hermes.validation.book import (
    align_sleeves,
    book_report,
    causal_inverse_vol_weights,
    combine_book,
    leave_one_out_delta_sharpe,
    orthogonality_matrix,
)


def _cycle(pattern: list[float], n: int) -> pd.Series:
    vals = [pattern[i % len(pattern)] for i in range(n)]
    return pd.Series(vals, index=list(range(n)))


def test_align_sleeves_unions_index_and_fills_missing_with_zero():
    a = pd.Series([0.1, 0.2, 0.3], index=[0, 1, 2])
    b = pd.Series([1.0, 2.0], index=[1, 3])
    out = align_sleeves({"a": a, "b": b})
    # Union of all mark periods; a sleeve absent in a period is FLAT (0.0), not NaN.
    assert list(out.index) == [0, 1, 2, 3]
    assert list(out.columns) == ["a", "b"]
    assert out.loc[0, "a"] == 0.1
    assert out.loc[0, "b"] == 0.0  # b has no mark at period 0 -> flat
    assert out.loc[3, "a"] == 0.0
    assert out.loc[3, "b"] == 2.0


def test_combine_book_equal_weight_is_row_mean():
    a = pd.Series([0.02, -0.01, 0.03], index=[0, 1, 2])
    b = pd.Series([0.00, 0.01, -0.02], index=[0, 1, 2])
    book = combine_book({"a": a, "b": b}, weighting="equal")
    np.testing.assert_allclose(book.to_numpy(), [0.01, 0.0, 0.005])


def test_causal_inverse_vol_gives_more_weight_to_calmer_sleeve():
    n = 40
    idx = list(range(n))
    calm = pd.Series([0.001 if i % 2 == 0 else -0.001 for i in range(n)], index=idx)
    wild = pd.Series([0.02 if i % 2 == 0 else -0.02 for i in range(n)], index=idx)
    w = causal_inverse_vol_weights(
        align_sleeves({"calm": calm, "wild": wild}), min_periods=8
    )
    np.testing.assert_allclose(w.sum(axis=1).to_numpy(), np.ones(n))
    # 1/variance weighting -> the calmer sleeve carries far more capital after warmup.
    assert w["calm"].iloc[-1] > w["wild"].iloc[-1]


def test_causal_inverse_vol_weights_do_not_peek_forward():
    n = 30
    idx = list(range(n))
    a = pd.Series([0.005 if i % 2 == 0 else -0.005 for i in range(n)], index=idx)
    b = pd.Series([0.005 if i % 2 == 1 else -0.005 for i in range(n)], index=idx)
    w_base = causal_inverse_vol_weights(align_sleeves({"a": a, "b": b}), min_periods=8)
    a2 = a.copy()
    a2.iloc[15] = 5.0  # a future-relative spike at an interior period
    w_spk = causal_inverse_vol_weights(align_sleeves({"a": a2, "b": b}), min_periods=8)
    # Weight at period t may use ONLY returns strictly before t: rows 0..15 unchanged
    # (the weight at t=15 is set from data 0..14, before the spike).
    np.testing.assert_allclose(w_base.iloc[:16].to_numpy(), w_spk.iloc[:16].to_numpy())
    # ...and the spike MUST move a later weight (proves the window really uses the data,
    # i.e. the invariance above is causality, not the function ignoring its input).
    assert not np.allclose(w_base.iloc[16].to_numpy(), w_spk.iloc[16].to_numpy())


def test_warmup_periods_fall_back_to_equal_weight():
    n = 20
    idx = list(range(n))
    a = pd.Series([0.01, -0.01] * (n // 2), index=idx)
    b = pd.Series([0.02, -0.02] * (n // 2), index=idx)
    w = causal_inverse_vol_weights(align_sleeves({"a": a, "b": b}), min_periods=8)
    # Before enough history accrues, weights are equal (no spurious early tilt).
    np.testing.assert_allclose(w.iloc[0].to_numpy(), [0.5, 0.5])


def test_loo_delta_sharpe_rewards_a_volatility_dampening_diversifier():
    core = _cycle([0.03, 0.01, -0.02, 0.0], 60)
    div = _cycle([-0.01, 0.0, 0.02, 0.01], 60)  # partially anti-correlated, same mean
    loo = leave_one_out_delta_sharpe({"core": core, "div": div}, weighting="equal")
    # The combined book has lower vol than either leg alone (same mean), so each leg
    # is accretive: dropping it lowers the book Sharpe -> positive LOO contribution.
    assert loo["div"] > 0
    assert loo["core"] > 0


def test_loo_delta_sharpe_penalises_a_constant_drag_sleeve():
    core = _cycle([0.03, 0.01, -0.02, 0.0], 60)
    div = _cycle([-0.01, 0.0, 0.02, 0.01], 60)
    drag = pd.Series([-0.02] * 60, index=list(range(60)))  # pure negative drift
    loo = leave_one_out_delta_sharpe(
        {"core": core, "div": div, "drag": drag}, weighting="equal"
    )
    # The book is strictly better WITHOUT the pure-loss sleeve -> negative contribution.
    assert loo["drag"] < 0


def test_orthogonality_matrix_is_unit_diagonal_and_symmetric():
    rng = np.arange(50)
    a = pd.Series(np.sin(rng), index=list(rng))
    b = pd.Series(np.sin(rng), index=list(rng))  # identical to a -> rho 1
    c = pd.Series(np.cos(rng * 0.7 + 1.3), index=list(rng))
    m = orthogonality_matrix({"a": a, "b": b, "c": c})
    assert m.loc["a", "a"] == pytest.approx(1.0)
    assert m.loc["a", "b"] == pytest.approx(1.0, abs=1e-9)
    assert m.loc["a", "c"] == pytest.approx(m.loc["c", "a"])  # symmetric


def test_book_report_assembles_book_stats_loo_and_orthogonality():
    core = _cycle([0.03, 0.01, -0.02, 0.0], 60)
    div = _cycle([-0.01, 0.0, 0.02, 0.01], 60)
    rep = book_report({"core": core, "div": div}, weighting="equal", n_trials=5)
    assert set(rep["sleeves"]) == {"core", "div"}
    assert "loo_delta_sharpe" in rep["sleeves"]["core"]
    assert "solo_sharpe" in rep["sleeves"]["core"]
    assert 0.0 <= rep["book"]["dsr"] <= 1.0
    assert 0.0 <= rep["book"]["pbo"] <= 1.0
    assert rep["book"]["n"] == 60
    assert rep["n_trials"] == 5
    assert "div" in rep["orthogonality"]["core"]
