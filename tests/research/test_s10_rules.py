"""Tests for the v-next hardened methodological rules (src/hermes/research/s10_rules.py).

Each helper encodes ONE adversarial lesson the merged streams paid for, as a falsifiable guard
the other streams import:
  R1  IC-in-levels is BANNED (S6: anti-conservative, ~16% FPR at N=60) → IC on deltas/ranks.
  R2  chance-correction is MANDATORY on any stability/agreement metric (S8: 0.88 raw = chance).
  R3  orthogonality is judged on CO-ACTIVE weeks / participation ratio, not the zero-filled grid (S5).
  R4  leak-injection (future-perturbation) on every DRIVER, replayed on every SURVIVOR at merge (AM-4).
  R5  nulls are CLUSTER-AWARE for any signal that co-fires across instruments (S2: i.i.d. null
      is anti-conservative; require a macro-event block).
"""

from __future__ import annotations

import numpy as np
import pytest

from hermes.research import s10_rules as rules


# --------------------------------------------------------------------------- #
# R1 — IC on deltas/ranks, NEVER on levels
# --------------------------------------------------------------------------- #
def _ar1(n, phi, rng):
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + rng.standard_normal()
    return x


def test_level_ic_mode_is_banned():
    rng = np.random.default_rng(0)
    x, y = _ar1(60, 0.9, rng), _ar1(60, 0.9, rng)
    with pytest.raises(ValueError, match="level"):
        rules.robust_ic(x, y, mode="level")


def test_effective_sample_size_shrinks_under_autocorrelation():
    rng = np.random.default_rng(1)
    persistent = _ar1(400, 0.9, rng)
    white = rng.standard_normal(400)
    n_eff_persistent = rules.effective_sample_size(persistent)
    n_eff_white = rules.effective_sample_size(white)
    assert n_eff_persistent < 0.3 * len(persistent)  # persistence destroys independent obs
    assert n_eff_white > 0.6 * len(white)  # white noise keeps ~all of them
    assert n_eff_persistent <= len(persistent)  # never inflated above n


def test_level_ic_is_anticonservative_versus_robust_ic():
    """The headline S6 encoding: on INDEPENDENT persistent series the naive level IC fires far
    above the nominal 5%, while the delta IC (the prescribed remedy) is ~calibrated."""
    rng = np.random.default_rng(7)
    m = 500
    level_hits, robust_hits = 0, 0
    for _ in range(m):
        x, y = _ar1(60, 0.85, rng), _ar1(60, 0.85, rng)  # independent → any IC is a false positive
        # naive level IC: Pearson on the raw persistent levels, t with the full N
        r = float(np.corrcoef(x, y)[0, 1])
        t = r * np.sqrt((len(x) - 2) / max(1e-9, 1 - r * r))
        if abs(t) > 1.96:
            level_hits += 1
        # the prescribed remedy: IC on deltas with honest inference
        if rules.ic_significance(x, y, mode="delta")["p_value"] < 0.05:
            robust_hits += 1
    level_fpr, robust_fpr = level_hits / m, robust_hits / m
    assert level_fpr > 0.12  # anti-conservative (the ~16% FPR lesson)
    assert robust_fpr < level_fpr  # the delta IC is strictly better calibrated
    assert robust_fpr <= 0.10  # ~nominal


def test_robust_ic_recovers_a_real_delta_signal():
    rng = np.random.default_rng(3)
    sig = _ar1(400, 0.5, rng)
    dsig = np.diff(sig)
    fwd = np.concatenate([[0.0], 0.6 * dsig + 0.5 * rng.standard_normal(len(dsig))])  # fwd delta ~ sig delta
    res = rules.ic_significance(sig, fwd, mode="delta")
    assert res["ic"] > 0.2
    assert res["p_value"] < 0.01


def test_effective_sample_size_survives_perfect_anticorrelation():
    """A perfectly alternating (max mean-reverting) series gives r1 == -1.0; the helper must NOT
    crash on it — differencing produces exactly such series, so R1's own remedy would crash."""
    alt = np.array([1.0, -1.0] * 6)
    n_eff = rules.effective_sample_size(alt)
    assert np.isfinite(n_eff)
    assert 2.0 <= n_eff <= alt.size
    # and ic_significance (which differences first) must not raise on an alternating signal
    res = rules.ic_significance(alt, np.array([1.0, -1.0] * 6))
    assert 0.0 <= res["p_value"] <= 1.0


def test_leak_test_rejects_nondeterministic_builder():
    """A stochastic builder breaks the perturbation test's premise (its output differs run-to-run
    independent of any future perturbation) → it must error LOUDLY, not silently report a leak."""
    rng = np.random.default_rng(9)
    data = rng.standard_normal(100)

    def stochastic_causal(d):
        s = np.zeros_like(d)
        s[1:] = d[:-1]
        return s + 1e-3 * np.random.default_rng().standard_normal(d.size)  # unseeded per call

    with pytest.raises(ValueError, match="determinist"):
        rules.future_perturbation_leak_test(stochastic_causal, data, rng=rng)


def test_cluster_bootstrap_empty_input_returns_one():
    p = rules.cluster_block_bootstrap_pvalue(
        np.array([]), np.array([], dtype=int), rng=np.random.default_rng(0), n_boot=100
    )
    assert p == 1.0


# --------------------------------------------------------------------------- #
# R2 — chance-correction on any stability / agreement metric
# --------------------------------------------------------------------------- #
def test_constant_predictor_has_zero_kappa():
    """A collapsed predictor (always one label) can score high RAW agreement but its
    chance-corrected agreement is ~0 — exactly the S8 'collapsed HMM scores 1.0' trap."""
    truth = np.array([1] * 88 + [0] * 12)
    collapsed = np.ones(100, dtype=int)  # always predicts 1
    raw = float(np.mean(truth == collapsed))
    kappa = rules.chance_corrected_agreement(truth, collapsed)
    assert raw == pytest.approx(0.88)
    assert kappa == pytest.approx(0.0, abs=1e-9)


def test_collapsed_state_series_has_zero_excess_stability():
    states = np.ones(200, dtype=int)  # one state forever: raw persistence 1.0, but 0 information
    res = rules.state_stability_excess(states)
    assert res["raw_persistence"] == pytest.approx(1.0)
    assert res["excess"] == pytest.approx(0.0, abs=1e-9)


def test_genuine_stability_has_positive_excess():
    # balanced two-state series with long runs → chance ~0.5, raw ~0.9 → clear positive excess
    states = np.array(([0] * 10 + [1] * 10) * 10)
    res = rules.state_stability_excess(states)
    assert res["chance_persistence"] == pytest.approx(0.5, abs=0.02)
    assert res["excess"] > 0.5


def test_excess_over_chance_encodes_s8_lesson():
    assert rules.excess_over_chance(0.88, 0.88) == pytest.approx(0.0, abs=1e-9)
    assert rules.excess_over_chance(1.0, 0.5) == pytest.approx(1.0)
    assert rules.excess_over_chance(0.75, 0.5) == pytest.approx(0.5)


# --------------------------------------------------------------------------- #
# R3 — orthogonality on co-active weeks, not the zero-filled grid
# --------------------------------------------------------------------------- #
def test_zero_filled_grid_hides_real_correlation():
    """Two sleeves correlated WHEN BOTH FIRE but rarely co-active look orthogonal on the full
    zero-filled grid. Judge on co-active weeks (S5)."""
    rng = np.random.default_rng(11)
    n = 400
    a = np.zeros(n)
    b = np.zeros(n)
    coactive = np.zeros(n, dtype=bool)
    coactive[:30] = True  # only 30 weeks where both fire
    shared = rng.standard_normal(30)
    a[:30] = shared + 0.1 * rng.standard_normal(30)
    b[:30] = shared + 0.1 * rng.standard_normal(30)  # strongly correlated on the overlap
    # elsewhere exactly ONE fires (never both) → zeros dominate, diluting the full-grid corr
    a[30:215] = rng.standard_normal(185)
    b[215:] = rng.standard_normal(185)
    res = rules.coactive_correlation(a, b)
    assert res["n_coactive"] == 30
    assert res["corr_coactive"] > 0.9
    assert abs(res["corr_full"]) < 0.3  # the grid makes them look orthogonal
    assert res["corr_coactive"] > abs(res["corr_full"]) + 0.5


def test_participation_ratio_flags_thin_overlap():
    n = 400
    a = np.zeros(n)
    b = np.zeros(n)
    a[:5] = 1.0
    b[:5] = 1.0  # only 5 co-active weeks
    res = rules.coactive_correlation(a, b, min_coactive=20)
    assert res["participation_ratio"] == pytest.approx(5 / n)
    assert res["reliable"] is False  # too thin to judge orthogonality


# --------------------------------------------------------------------------- #
# R4 — leak-injection (future perturbation) + survivor replay at merge
# --------------------------------------------------------------------------- #
def test_future_perturbation_detects_lookahead():
    rng = np.random.default_rng(5)
    data = rng.standard_normal(200)

    def leaky(d):  # signal at t peeks one bar into the FUTURE
        s = np.zeros_like(d)
        s[:-1] = d[1:]
        return s

    def causal(d):  # signal at t uses only the PAST
        s = np.zeros_like(d)
        s[1:] = d[:-1]
        return s

    assert rules.future_perturbation_leak_test(leaky, data, rng=rng)["leaked"] is True
    assert rules.future_perturbation_leak_test(causal, data, rng=rng)["leaked"] is False


def test_assert_no_lookahead_raises_on_leak():
    rng = np.random.default_rng(6)
    data = rng.standard_normal(120)

    def leaky(d):
        s = np.zeros_like(d)
        s[:-2] = d[2:]
        return s

    with pytest.raises(rules.LeakDetected):
        rules.assert_no_lookahead(leaky, data, rng=rng)


def test_survivor_must_replay_leak_audit_at_merge():
    """AM-4: a prior clean audit is NOT enough — every survivor must REPLAY the leak audit on the
    merged artifact (composition bugs fabricate positives)."""
    survivors = [
        {"name": "sleeve_a", "leak_audit_replayed": True},
        {"name": "sleeve_b", "leak_audit_replayed": False},  # missing replay → must block
    ]
    res = rules.replay_leak_audits(survivors)
    assert res["all_clean"] is False
    assert "sleeve_b" in res["blocked"]
    assert "sleeve_a" not in res["blocked"]


# --------------------------------------------------------------------------- #
# R5 — cluster-aware nulls for co-firing signals
# --------------------------------------------------------------------------- #
def test_block_events_by_time_groups_co_firing_events():
    ts = np.array([0, 1, 2, 100, 101, 200])
    clusters = rules.block_events_by_time(ts, window=10)
    # {0,1,2}->c0 ; {100,101}->c1 ; {200}->c2
    assert clusters[0] == clusters[1] == clusters[2]
    assert clusters[3] == clusters[4]
    assert len({int(c) for c in clusters}) == 3


def test_clustering_shrinks_effective_n():
    # 100 events but they co-fire in 10 macro-event clusters of 10
    cluster_ids = np.repeat(np.arange(10), 10)
    assert rules.effective_independent_events(cluster_ids) == 10


def test_iid_null_is_anticonservative_under_clustering():
    """A squeeze co-fires across alts: the i.i.d. per-event sign-flip null treats each co-firing
    event as independent → a tiny (over-confident) p. The cluster (macro-event) null is honest."""
    rng = np.random.default_rng(21)
    n_clusters, per_cluster = 12, 10
    cluster_ids = np.repeat(np.arange(n_clusters), per_cluster)
    # each macro-event has a small common return shared by all its co-firing events; force the
    # cluster mean to +0.3 exactly so the i.i.d. vs cluster significance gap is deterministic
    raw = rng.standard_normal(n_clusters)
    cluster_shock = 0.3 + 0.8 * (raw - raw.mean())
    values = np.repeat(cluster_shock, per_cluster) + 0.02 * rng.standard_normal(n_clusters * per_cluster)

    p_iid = rules.cluster_block_bootstrap_pvalue(
        values, np.arange(len(values)), rng=np.random.default_rng(1), n_boot=2000
    )  # each event its own cluster == the i.i.d. null
    p_cluster = rules.cluster_block_bootstrap_pvalue(
        values, cluster_ids, rng=np.random.default_rng(1), n_boot=2000
    )
    assert p_cluster > p_iid  # the honest cluster null is strictly less significant
    assert p_iid < 0.05  # the i.i.d. null over-rejects
    assert p_cluster > 0.05  # the cluster-aware null does NOT (12 macro-events is under-powered)
