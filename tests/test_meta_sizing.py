"""Meta-labeled fractional-Kelly sizing layer (vector 1.3).

DIRECTION is resolved by the confirmed sleeve (S2 funding-squeeze = long-only);
this layer optimises SIZE/TIMING only. A meta-label predicts P(win) from each
event's decision-time features (including a ``news_blackout`` flag, learned as a
down-sizing feature, not a hard veto); fractional-Kelly (0.20) turns P(win) into a
bet fraction. The honest gate is OUT-OF-SAMPLE: train on some instruments, size a
HELD-OUT one, and require the sized stream's DEFLATED Sharpe to beat the un-sized
baseline — else it is size overfit (§1.3 KILL).
"""

from __future__ import annotations

import numpy as np
import pytest

from hermes.learning.stats import sharpe_ratio
from hermes.validation.meta_sizing import (
    estimate_win_loss_ratio,
    kelly_fraction_from_pwin,
    loo_sizing_gate,
    meta_sized_returns,
)


# --------------------------------------------------------------------------- #
# fractional-Kelly from P(win)
# --------------------------------------------------------------------------- #
def test_kelly_fraction_zero_below_breakeven_and_grows_with_pwin():
    assert kelly_fraction_from_pwin(0.30, 1.0, fraction=0.20) == 0.0  # negative edge
    assert kelly_fraction_from_pwin(0.50, 1.0, fraction=0.20) == 0.0  # breakeven
    f07 = kelly_fraction_from_pwin(0.70, 1.0, fraction=0.20)
    f08 = kelly_fraction_from_pwin(0.80, 1.0, fraction=0.20)
    assert 0.0 < f07 < f08
    # full Kelly f* = (p*b − q)/b = (0.7 − 0.3)/1 = 0.4 ; 0.20·0.4 = 0.08
    assert f07 == pytest.approx(0.08)


def test_estimate_win_loss_ratio_is_avg_win_over_avg_loss():
    b = estimate_win_loss_ratio([0.02, 0.04, -0.01, -0.03], [1, 1, 0, 0])
    assert b == pytest.approx(0.03 / 0.02)  # 1.5


# --------------------------------------------------------------------------- #
# meta_sized_returns
# --------------------------------------------------------------------------- #
def _signal_set(rng, n):
    """Events where feature x is predictive of the win (non-separable: noise added
    so the logistic fit converges cleanly)."""
    x = rng.normal(0, 1, n)
    outcomes = ((x + rng.normal(0, 0.5, n)) > 0).astype(int)
    gross = np.where(outcomes == 1, 0.02, -0.02).tolist()
    feats = [{"x": float(xi)} for xi in x]
    return feats, outcomes.tolist(), gross


def test_meta_sizing_upweights_predictable_winners():
    rng = np.random.default_rng(0)
    tf, to, tg = _signal_set(rng, 400)
    ef, _eo, eg = _signal_set(rng, 200)
    res = meta_sized_returns(tf, to, tg, ef, eg, fraction=0.20, min_samples=50)
    assert res["fitted"] is True
    # Sizing concentrates on the high-P(win) events -> higher risk-adjusted return.
    assert sharpe_ratio(res["sized"]) > sharpe_ratio(res["baseline"])


def test_meta_sizing_unfit_falls_back_to_flat_baseline():
    feats = [{"x": 0.1} for _ in range(10)]
    outs = [1, 0] * 5
    gross = [0.01, -0.01] * 5
    res = meta_sized_returns(feats, outs, gross, feats, gross, min_samples=50)
    assert res["fitted"] is False  # 10 < min_samples -> no model
    assert res["sized"] == res["baseline"]  # un-sized; sizing adds nothing yet


def test_news_blackout_is_learned_as_a_downsizing_feature():
    rng = np.random.default_rng(1)
    n = 600
    blackout = rng.random(n) < 0.4
    win_prob = np.where(blackout, 0.2, 0.7)  # blackout events mostly lose
    outcomes = (rng.random(n) < win_prob).astype(int)
    gross = np.where(outcomes == 1, 0.02, -0.02).tolist()
    feats = [
        {"news_blackout": float(b), "x": float(xi)}
        for b, xi in zip(blackout, rng.normal(0, 1, n))
    ]
    res = meta_sized_returns(
        feats, outcomes.tolist(), gross, feats, gross, fraction=0.20, min_samples=50
    )
    bet = np.asarray(res["bet_fraction"])
    # The meta-label sizes blackout events SMALLER (learned veto-as-feature).
    assert bet[blackout].mean() < bet[~blackout].mean()


# --------------------------------------------------------------------------- #
# leave-one-out held-out gate (the §1.3 KILL)
# --------------------------------------------------------------------------- #
def test_loo_sizing_gate_runs_three_folds_and_reports_each():
    rng = np.random.default_rng(2)
    instruments = {}
    for name in ("BTC", "ETH", "SOL"):
        f, o, g = _signal_set(rng, 250)
        instruments[name] = {"features": f, "outcomes": o, "gross": g}
    res = loo_sizing_gate(instruments, fraction=0.20, min_samples=50)
    assert res["n_folds"] == 3
    assert set(res["per_fold"]) == {"BTC", "ETH", "SOL"}
    for fold in res["per_fold"].values():
        assert {"baseline_dsr", "sized_dsr", "fitted", "improved"} <= set(fold)
    assert isinstance(res["passed"], bool)
    # A clean cross-instrument signal generalises -> sizing helps on >=2/3.
    assert res["improved_folds"] >= 2
    assert res["passed"] is True


def test_loo_sizing_gate_does_not_pass_when_features_carry_no_signal():
    rng = np.random.default_rng(3)
    instruments = {}
    for name in ("BTC", "ETH", "SOL"):
        n = 250
        gross = rng.normal(0, 0.02, n).tolist()
        outcomes = [1 if g > 0 else 0 for g in gross]
        # CONSTANT (uninformative) feature: the meta-label cannot discriminate, so
        # every event gets the same bet -> sized Sharpe == baseline Sharpe exactly,
        # never strictly better. The gate must NOT rubber-stamp this.
        feats = [{"x": 1.0} for _ in range(n)]
        instruments[name] = {"features": feats, "outcomes": outcomes, "gross": gross}
    res = loo_sizing_gate(instruments, fraction=0.20, min_samples=50)
    assert res["improved_folds"] == 0
    assert res["passed"] is False
