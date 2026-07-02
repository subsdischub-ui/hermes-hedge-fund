"""Stream 6 — the conditioning gate (the X3 lesson done right).

X3 died because the conditioner was a hard gate thinning the *same* base events
(mechanically indistinguishable from a random thinning -> shuffle-p~0.42) using a
regime collinear with the base signal. S6 keeps the exact X3 machinery — frozen base
events, no-overlap realisation, the identical shuffled-LABEL null — and adds the
mandate's higher bar: the conditioned subset must beat BOTH the base AND its
complement, at shuffle-p < 0.05, with the conditioned subset's own random-entry null
also < 0.05.

These tests pin the null against an ORACLE on a deterministic stepped price path where
every event's return is known exactly, so "does the null discriminate a good split
from a bad one" has a checkable answer.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip(
    "fxgate",
    reason="s6_gold_regime imports the standalone lab module fxgate (research_fx/, gitignored) — absent in CI",
)
from hermes.research import s6_gold_regime as s6


def _stepped(rets):
    """Build (close, ts, events) so the long event at bar 2k returns exactly rets[k].

    Events sit at bars 0,2,4,... (spaced 2 >= horizon 1), so the no-overlap engine
    never drops one and each event's net return (cost 0, horizon 1) is exactly rets[k]
    regardless of which subset is realised — the property that makes the oracle exact.
    """
    rets = list(rets)
    k = len(rets)
    close = np.empty(2 * k, dtype=float)
    close[0] = 1.0
    for j, r in enumerate(rets):
        close[2 * j + 1] = close[2 * j] * (1.0 + r)
        if 2 * j + 2 < len(close):
            close[2 * j + 2] = close[2 * j + 1]
    ts = (np.arange(2 * k) * 86_400_000).astype("int64")
    events = [(2 * j, 1) for j in range(k)]
    return close, ts, events


# --------------------------------------------------------------------------- #
# net_mean over a subset (no-overlap, cost-charged) — the realised statistic
# --------------------------------------------------------------------------- #
def test_subset_net_mean_matches_known_returns():
    rets = [0.01, 0.02, 0.03, 0.04]
    close, ts, events = _stepped(rets)
    net, n = s6.subset_net_mean(close, ts, events, horizon=1, cost_rt=0.0)
    assert n == 4
    assert net == pytest.approx(np.mean(rets))


def test_subset_net_mean_charges_cost_once_per_trade():
    rets = [0.01, 0.01]
    close, ts, events = _stepped(rets)
    net, n = s6.subset_net_mean(close, ts, events, horizon=1, cost_rt=0.005)
    assert net == pytest.approx(0.01 - 0.005)


# --------------------------------------------------------------------------- #
# shuffled-label null (X3 verbatim) — random same-size subsample of base events
# --------------------------------------------------------------------------- #
def test_shuffle_label_p_tiny_when_subset_is_the_best():
    # 20 events, increasing returns; conditioned subset = the top 10 -> almost no
    # random 10-subsample beats it -> p ~ 0.
    rets = np.linspace(0.001, 0.020, 20)
    close, ts, events = _stepped(rets)
    cond_net, n_cond = s6.subset_net_mean(close, ts, [events[i] for i in range(10, 20)], 1, 0.0)
    p = s6.shuffle_label_p(close, ts, events, n_cond, cond_net, horizon=1, cost_rt=0.0, n_perm=2000, seed=0)
    assert p < 0.05


def test_shuffle_label_p_large_when_subset_is_the_worst():
    rets = np.linspace(0.001, 0.020, 20)
    close, ts, events = _stepped(rets)
    cond_net, n_cond = s6.subset_net_mean(close, ts, [events[i] for i in range(0, 10)], 1, 0.0)
    p = s6.shuffle_label_p(close, ts, events, n_cond, cond_net, horizon=1, cost_rt=0.0, n_perm=2000, seed=0)
    assert p > 0.5


# --------------------------------------------------------------------------- #
# the full verdict — base / cond / complement triplet + nulls + survives flag
# --------------------------------------------------------------------------- #
def test_conditioning_verdict_oracle_good_split_orders_triplet():
    rets = np.linspace(0.001, 0.020, 20)
    close, ts, events = _stepped(rets)
    regime_on = np.array([i >= 10 for i in range(20)])  # top half on
    v = s6.conditioning_verdict(close, ts, events, regime_on, horizon=1, cost_rt=0.0,
                                n_perm=2000, n_sims=500, seed=0)
    assert v["cond_net"] > v["base_net"] > v["comp_net"]
    assert v["n_cond"] == 10 and v["n_comp"] == 10
    assert v["shuffle_p"] < 0.05


def test_conditioning_verdict_complement_wins_does_not_survive():
    # the X3 failure mode: the regime selects the WORSE half -> complement wins ->
    # must NOT survive, regardless of any other metric.
    rets = np.linspace(0.001, 0.020, 20)
    close, ts, events = _stepped(rets)
    regime_on = np.array([i < 10 for i in range(20)])   # bottom half on
    v = s6.conditioning_verdict(close, ts, events, regime_on, horizon=1, cost_rt=0.0,
                                n_perm=2000, n_sims=500, seed=0)
    assert v["cond_net"] < v["comp_net"]
    assert v["survives"] is False


def test_conditioning_verdict_reports_circular_null_in_unit_range():
    rets = np.linspace(0.001, 0.020, 20)
    close, ts, events = _stepped(rets)
    regime_on = np.array([i >= 10 for i in range(20)])
    v = s6.conditioning_verdict(close, ts, events, regime_on, horizon=1, cost_rt=0.0,
                                n_perm=1000, n_sims=500, seed=0)
    assert 0.0 <= v["circular_p"] <= 1.0
    assert set(["base_net", "cond_net", "comp_net", "shuffle_p", "circular_p",
                "cond_mc_p", "n_base", "n_cond", "n_comp", "survives"]).issubset(v)
