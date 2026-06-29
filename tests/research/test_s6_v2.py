"""Stream 6 v2 — CB flow AS A SIGNAL (not a label) + cross-book diversifier + dollar state.

v1's own lesson: conditioning a dead signal on the state is a random thinning. v2 turns the
*cause* of the decoupling — official central-bank flow — into the signal, tested with an
honest continuous-signal gate (information coefficient + an autocorrelation-preserving
circular-shift null + OOS), plus the two well-powered new axes the N-audit blesses: gold as a
regime-conditional cross-book (crypto) diversifier, and a distinct dollar-decoupling state.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hermes.research import s6_gold_regime as s6


# --------------------------------------------------------------------------- #
# cb_demand_signal — PIT continuous signal (level / acceleration)
# --------------------------------------------------------------------------- #
def test_cb_demand_signal_level_is_annualised_pit():
    rows = pd.DataFrame({"year": [2021] * 4, "quarter": [1, 2, 3, 4], "tonnes": [100.0, 100.0, 100.0, 100.0]})
    lvl = s6.cb_demand_signal(rows, mode="level", lag_days=45)
    assert lvl.iloc[-1] == 400.0                       # trailing-4q annualised, availability-dated


def test_cb_demand_signal_accel_is_change_of_level():
    rows = pd.DataFrame({"year": [2021, 2021, 2021, 2021, 2022], "quarter": [1, 2, 3, 4, 1],
                         "tonnes": [100.0, 100.0, 100.0, 100.0, 200.0]})
    acc = s6.cb_demand_signal(rows, mode="accel", lag_days=45)
    # level: 2021Q4=400, 2022Q1=100+100+100+200=500 -> accel at 2022Q1 = +100
    assert acc.dropna().iloc[-1] == 100.0


# --------------------------------------------------------------------------- #
# forward_return / neutral_pair_forward_return
# --------------------------------------------------------------------------- #
def test_forward_return_is_h_step_ahead():
    px = pd.Series([100.0, 110.0, 121.0], index=pd.bdate_range("2020-01-01", periods=3))
    fr = s6.forward_return(px, horizon=1)
    assert fr.iloc[0] == pytest.approx(0.10)          # 110/100 - 1
    assert np.isnan(fr.iloc[-1])                       # no future beyond the end


def test_neutral_pair_forward_return_cancels_common_move():
    idx = pd.bdate_range("2020-01-01", periods=3)
    gold = pd.Series([100.0, 110.0, 120.0], index=idx)
    silver = pd.Series([10.0, 11.0, 12.0], index=idx)   # +10% same as gold step 1
    nr = s6.neutral_pair_forward_return(gold, silver, horizon=1)
    assert nr.iloc[0] == pytest.approx(0.0)            # gold +10% - silver +10% = 0 (beta cancelled)


# --------------------------------------------------------------------------- #
# ic_test — information coefficient + circular-shift null + OOS
# --------------------------------------------------------------------------- #
def test_ic_test_perfect_predictor_has_high_ic_and_small_p():
    n = 80
    idx = pd.bdate_range("2015-01-01", periods=n)
    sig = pd.Series(np.arange(n, dtype=float), index=idx)
    fwd = pd.Series(np.arange(n, dtype=float) + 0.0, index=idx)  # forward return rises with signal
    out = s6.ic_test(sig, fwd, n_perm=200, seed=0)
    assert out["ic"] > 0.9
    assert out["p"] < 0.05
    assert out["n"] == n


def test_ic_test_random_signal_is_not_significant():
    n = 120
    idx = pd.bdate_range("2015-01-01", periods=n)
    rng = np.random.default_rng(3)
    sig = pd.Series(rng.normal(size=n), index=idx)
    fwd = pd.Series(rng.normal(size=n), index=idx)      # independent
    out = s6.ic_test(sig, fwd, n_perm=400, seed=0)
    assert out["p"] > 0.05                              # no spurious significance


# --------------------------------------------------------------------------- #
# conditional_diversification — gold as a regime-conditional cross-book hedge
# --------------------------------------------------------------------------- #
def test_conditional_diversification_detects_regime_dependent_hedge():
    n = 400
    idx = pd.bdate_range("2018-01-01", periods=n)
    rng = np.random.default_rng(0)
    book = pd.Series(rng.normal(0, 0.02, n), index=idx)
    regime_on = np.array([i >= n // 2 for i in range(n)])
    # in-regime: gold moves OPPOSITE the book (hedge); out-regime: gold moves WITH the book
    asset = book.copy()
    asset[regime_on] = -book[regime_on] + rng.normal(0, 0.001, regime_on.sum())
    asset[~regime_on] = book[~regime_on] + rng.normal(0, 0.001, (~regime_on).sum())
    out = s6.conditional_diversification(asset, book, pd.Series(regime_on, index=idx))
    assert out["corr_on"] < -0.5                        # genuine hedge inside the regime
    assert out["corr_off"] > 0.5                        # risk-on co-movement outside
    assert out["p"] < 0.05                              # the regime split is not random


# --------------------------------------------------------------------------- #
# dollar_decoupling_state — distinct de-dollarisation axis (DXY-gold corr)
# --------------------------------------------------------------------------- #
def test_dollar_decoupling_state_flags_gold_and_dollar_rising_together():
    n = 700
    idx = pd.bdate_range("2014-01-01", periods=n)
    rng = np.random.default_rng(1)
    half = n // 2
    dret = rng.normal(0, 0.005, n)
    gret = np.empty(n)
    gret[:half] = -1.2 * dret[:half] + rng.normal(0, 0.0005, half)   # normal: gold up when USD down
    gret[half:] = +1.2 * dret[half:] + rng.normal(0, 0.0005, n - half)  # decoupled: rise together
    gold = pd.Series(100 * np.exp(np.cumsum(gret)), index=idx)
    dxy = pd.Series(100 * np.exp(np.cumsum(dret)), index=idx)
    st = s6.dollar_decoupling_state(gold, dxy, window=120, min_periods=40)
    late = st["state"].dropna().iloc[-100:]
    assert (late == "DECOUPLED").mean() > 0.5
