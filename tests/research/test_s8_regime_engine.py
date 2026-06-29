"""Stream 8 — regime engine tests.

The engine is a *conditioning layer*, never a directional signal, and its single
most load-bearing property is **causality / zero look-ahead**: the regime label
attached to bar t must be a function of bars <= t only. A regime engine that
"sees the future" silently invalidates every downstream backtest, so that
property gets its own dedicated property test (``test_*_causal*``), not just a
spot check.

All expectations are hand-derived or grounded in synthetic data with a known
generating process.
"""

from __future__ import annotations

import numpy as np
import pytest

from hermes.research.s8_regime_engine import (
    CausalGaussianHMM,
    asof_labels,
    lag_business_days,
    macro_regime_gate,
    online_bocpd,
    realized_vol,
    regime_engine,
    shuffle_label_null,
    state_stability,
    trailing_vol_bucket,
)


def _synthetic_prices(n, seed, regime_breaks=True):
    """A price path with vol regimes, for engine-level tests."""
    import pandas as pd

    rng = np.random.default_rng(seed)
    if regime_breaks:
        segs = []
        scales = [0.01, 0.04, 0.012, 0.05]
        drifts = [0.0008, -0.0010, 0.0003, -0.0015]
        per = n // len(scales) + 1
        for d, s in zip(drifts, scales):
            segs.append(rng.normal(d, s, per))
        r = np.concatenate(segs)[:n]
    else:
        r = rng.normal(0.0005, 0.02, n)
    close = 100.0 * np.exp(np.cumsum(r))
    ts = 1_600_000_000_000 + np.arange(n) * 86_400_000  # daily ms epoch
    return pd.DataFrame({"ts": ts, "close": close})


def _markov_gaussian(rng, means, var, p_stay, n):
    """Generate a 2-state persistent Gaussian Markov chain (states, observations)."""
    k = len(means)
    states = np.empty(n, dtype=int)
    states[0] = 0
    for t in range(1, n):
        if rng.random() < p_stay:
            states[t] = states[t - 1]
        else:
            states[t] = (states[t - 1] + 1) % k
    obs = rng.normal(loc=np.array(means)[states], scale=np.sqrt(var))
    return states, obs.reshape(-1, 1)


# --------------------------------------------------------------------------- #
# CausalGaussianHMM — fit (Baum-Welch) recovers the generating regimes
# --------------------------------------------------------------------------- #
def test_hmm_fit_recovers_state_means():
    rng = np.random.default_rng(0)
    _, obs = _markov_gaussian(rng, means=[-1.0, 1.0], var=0.09, p_stay=0.97, n=4000)
    hmm = CausalGaussianHMM(n_states=2, n_iter=100).fit(obs)
    recovered = sorted(float(m[0]) for m in hmm.means_)
    assert recovered[0] == pytest.approx(-1.0, abs=0.2)
    assert recovered[1] == pytest.approx(1.0, abs=0.2)


def test_hmm_canonical_order_is_by_ascending_dispersion():
    # Same mean, different variance: canonical state 0 = calm (low var), last = stress.
    rng = np.random.default_rng(1)
    n = 4000
    states = np.empty(n, dtype=int)
    states[0] = 0
    for t in range(1, n):
        states[t] = states[t - 1] if rng.random() < 0.97 else 1 - states[t - 1]
    scale = np.where(states == 0, 0.2, 1.0)
    obs = rng.normal(0.0, scale).reshape(-1, 1)
    hmm = CausalGaussianHMM(n_states=2, n_iter=100).fit(obs)
    # covars_ ordered canonically: total dispersion of state 0 < state 1.
    assert hmm.covars_[0].sum() < hmm.covars_[1].sum()


# --------------------------------------------------------------------------- #
# Causal forward filter — the zero-look-ahead guarantee
# --------------------------------------------------------------------------- #
def test_filter_posterior_at_t_uses_only_data_through_t():
    """filter(X)[i] must equal filter(X[:i+1])[-1] — the defining causal property.

    If this fails, the filtered label leaks future bars and every downstream
    backtest is invalid.
    """
    rng = np.random.default_rng(2)
    _, obs = _markov_gaussian(rng, means=[-1.0, 1.0], var=0.09, p_stay=0.95, n=300)
    hmm = CausalGaussianHMM(n_states=2, n_iter=50).fit(obs)

    states_full, probs_full = hmm.filter(obs)
    for i in (10, 50, 137, 299):
        s_prefix, p_prefix = hmm.filter(obs[: i + 1])
        assert states_full[i] == s_prefix[-1]
        np.testing.assert_allclose(probs_full[i], p_prefix[-1], atol=1e-9)


def test_filter_probs_are_valid_distributions():
    rng = np.random.default_rng(3)
    _, obs = _markov_gaussian(rng, means=[-1.0, 1.0], var=0.09, p_stay=0.95, n=200)
    hmm = CausalGaussianHMM(n_states=2, n_iter=50).fit(obs)
    _, probs = hmm.filter(obs)
    assert probs.shape == (200, 2)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-9)
    assert (probs >= -1e-12).all()


def test_filter_tracks_obvious_regime_assignment():
    rng = np.random.default_rng(4)
    _, obs = _markov_gaussian(rng, means=[-1.0, 1.0], var=0.04, p_stay=0.98, n=2000)
    hmm = CausalGaussianHMM(n_states=2, n_iter=100).fit(obs)
    # Identify which canonical state has the higher mean.
    hi = int(np.argmax([m[0] for m in hmm.means_]))
    states, _ = hmm.filter(obs)
    # On bars clearly drawn from the high-mean regime (obs >> 0), the filter should
    # mostly land on the high-mean state.
    clearly_high = obs[:, 0] > 1.5
    agree = (states[clearly_high] == hi).mean()
    assert agree > 0.8


def test_fit_is_deterministic():
    # Deterministic (quantile-seeded) init => identical fit across runs, so a
    # regime never "re-segments" merely because of RNG. OOS stability starts here.
    rng = np.random.default_rng(5)
    _, obs = _markov_gaussian(rng, means=[-1.0, 1.0], var=0.09, p_stay=0.95, n=500)
    a = CausalGaussianHMM(n_states=2, n_iter=50).fit(obs)
    b = CausalGaussianHMM(n_states=2, n_iter=50).fit(obs)
    np.testing.assert_allclose(a.means_, b.means_)
    np.testing.assert_allclose(a.transmat_, b.transmat_)


# --------------------------------------------------------------------------- #
# Online BOCPD (Adams & MacKay 2007) — changepoint probability, causal
# --------------------------------------------------------------------------- #
def test_bocpd_output_is_a_probability_series():
    rng = np.random.default_rng(10)
    x = rng.normal(0.0, 1.0, size=300)
    cp = online_bocpd(x, hazard_lambda=200.0)
    assert cp.shape == (300,)
    assert (cp >= 0.0).all() and (cp <= 1.0 + 1e-9).all()


def test_bocpd_is_causal():
    """cp_prob[t] must equal the value obtained from x[:t+1] alone."""
    rng = np.random.default_rng(11)
    x = np.concatenate([rng.normal(0, 1, 150), rng.normal(6, 1, 150)])
    cp_full = online_bocpd(x, hazard_lambda=200.0)
    for i in (5, 80, 149, 151, 220, 299):
        cp_prefix = online_bocpd(x[: i + 1], hazard_lambda=200.0)
        assert cp_full[i] == pytest.approx(cp_prefix[-1], abs=1e-9)


def test_bocpd_spikes_at_a_known_mean_shift():
    rng = np.random.default_rng(12)
    x = np.concatenate([rng.normal(0, 1, 200), rng.normal(8, 1, 200)])
    cp = online_bocpd(x, hazard_lambda=300.0)
    # Right after the break the model should see a changepoint: the max cp_prob in
    # the [200, 215) window dwarfs the calm interior of the first segment.
    near_break = cp[200:215].max()
    calm_interior = np.median(cp[50:190])
    assert near_break > 0.4
    assert near_break > 5 * calm_interior


def test_bocpd_truncation_matches_untruncated_on_short_series():
    rng = np.random.default_rng(13)
    x = rng.normal(0, 1, 120)
    full = online_bocpd(x, hazard_lambda=100.0, rmax=None)
    capped = online_bocpd(x, hazard_lambda=100.0, rmax=500)
    np.testing.assert_allclose(full, capped, atol=1e-9)


# --------------------------------------------------------------------------- #
# Causal volatility + trailing-percentile buckets (point-in-time, no leak)
# --------------------------------------------------------------------------- #
def test_realized_vol_is_trailing_std():
    import pandas as pd
    r = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02, -0.03])
    rv = realized_vol(r, window=3)
    # First two are NaN (need 3 obs); rv[2] = std of r[0:3].
    assert np.isnan(rv.iloc[0]) and np.isnan(rv.iloc[1])
    assert rv.iloc[2] == pytest.approx(r.iloc[0:3].std(ddof=1))
    assert rv.iloc[5] == pytest.approx(r.iloc[3:6].std(ddof=1))


def test_realized_vol_is_causal():
    import pandas as pd
    rng = np.random.default_rng(20)
    r = pd.Series(rng.normal(0, 1, 100))
    rv_full = realized_vol(r, window=10)
    rv_pref = realized_vol(r.iloc[:55], window=10)
    np.testing.assert_allclose(
        rv_full.iloc[:55].to_numpy(), rv_pref.to_numpy(), equal_nan=True
    )


def test_trailing_vol_bucket_is_causal():
    """bucket[t] must use only vol[:t+1] within the trailing window."""
    import pandas as pd
    rng = np.random.default_rng(21)
    vol = pd.Series(np.abs(rng.normal(1, 0.5, 200)))
    full = trailing_vol_bucket(vol, window=50, n_buckets=3)
    pref = trailing_vol_bucket(vol.iloc[:120], window=50, n_buckets=3)
    np.testing.assert_array_equal(full[:120], pref, strict=False) if False else None
    # element-wise on the overlap, ignoring warmup NaNs
    a, b = full.iloc[:120].to_numpy(), pref.to_numpy()
    mask = ~(np.isnan(a) | np.isnan(b))
    np.testing.assert_array_equal(a[mask], b[mask])


def test_trailing_vol_bucket_ranks_high_vol_into_top_bucket():
    import pandas as pd
    # A long calm stretch then a clearly elevated value -> top bucket at the spike.
    vol = pd.Series([0.1] * 100 + [5.0])
    b = trailing_vol_bucket(vol, window=100, n_buckets=3)
    assert b.iloc[-1] == 2  # the 5.0 sits in the top tercile of its trailing window


# --------------------------------------------------------------------------- #
# regime_engine — the composite. Zero look-ahead is THE invariant.
# --------------------------------------------------------------------------- #
def test_engine_row_t_identical_on_prefix_vs_full_history():
    """The defining property: the engine row for an armed bar t is identical whether
    computed on the full frame or on df[:t+1]. Any leak (full-sample fit, future
    quantile, smoothed posterior) breaks this."""
    df = _synthetic_prices(1100, seed=30)
    full = regime_engine(df, warmup=400, vol_window=20, bucket_window=252)
    for t in (850, 1000, 1099):
        pref = regime_engine(df.iloc[: t + 1], warmup=400, vol_window=20, bucket_window=252)
        row_full, row_pref = full.iloc[t], pref.iloc[t]
        for col in ("state", "p0", "p1", "p2", "cp_prob", "vol_bucket", "rv"):
            a, b = row_full[col], row_pref[col]
            if isinstance(a, float) and np.isnan(a):
                assert np.isnan(b), f"{col}: full=NaN pref={b}"
            else:
                assert a == pytest.approx(b, abs=1e-8), f"{col}: full={a} pref={b}"


def test_engine_causality_holds_on_ambiguous_nonsaturated_posteriors():
    """Stronger leak guard. On highly-separable data the posteriors saturate to
    one-hot, so a smoothed (future-dependent) posterior would be NUMERICALLY identical
    to the causal one — the saturated prefix/perturbation tests then can't discriminate
    a backward-pass leak. Here we force AMBIGUOUS regimes and assert prefix-equality on
    the LEAST-saturated armed bars, where a smoothed-posterior leak WOULD move the value.
    """
    import pandas as pd

    rng = np.random.default_rng(71)
    n = 1200
    scale = np.where((np.arange(n) // 120) % 2 == 0, 0.012, 0.018)  # mild, overlapping
    r = rng.normal(0.0, scale)
    close = 100.0 * np.exp(np.cumsum(r))
    df = pd.DataFrame({"ts": 1_600_000_000_000 + np.arange(n) * 86_400_000, "close": close})

    full = regime_engine(df, warmup=400, vol_window=20, bucket_window=252)
    armed = full.dropna(subset=["state"])
    pmax = armed[["p0", "p1", "p2"]].max(axis=1)
    assert float(pmax.min()) < 0.9, "test needs genuinely ambiguous posteriors to bite"
    for t in pmax.nsmallest(15).index:  # the most ambiguous armed bars
        pos = full.index.get_loc(t)
        pref = regime_engine(df.iloc[: pos + 1], warmup=400, vol_window=20, bucket_window=252)
        for col in ("state", "p0", "p1", "p2"):
            assert full.iloc[pos][col] == pytest.approx(pref.iloc[pos][col], abs=1e-9), (
                f"{col} differs at ambiguous bar {pos} -> look-ahead leak"
            )


def test_engine_future_bars_cannot_change_an_earlier_row():
    """Independent leak check: perturbing bars AFTER t must not move row t."""
    df = _synthetic_prices(900, seed=31)
    base = regime_engine(df, warmup=350)
    t = 700
    tampered = df.copy()
    tampered.loc[t + 1 :, "close"] *= 1.5  # rewrite the entire future
    after = regime_engine(tampered, warmup=350)
    for col in ("state", "p0", "p1", "p2", "cp_prob", "vol_bucket"):
        a, b = base.iloc[t][col], after.iloc[t][col]
        assert (np.isnan(a) and np.isnan(b)) or a == pytest.approx(b, abs=1e-8)


def test_engine_warmup_rows_have_no_hmm_state_then_arms():
    df = _synthetic_prices(900, seed=32)
    out = regime_engine(df, warmup=350)
    # Before the HMM is armed, state is NaN; after, it is a valid canonical index.
    assert out["state"].iloc[:60].isna().all()
    armed = out["state"].dropna()
    assert len(armed) > 0
    assert set(np.unique(armed.to_numpy())).issubset({0, 1, 2})


def test_engine_probs_sum_to_one_where_armed():
    df = _synthetic_prices(900, seed=33)
    out = regime_engine(df, warmup=350)
    armed = out.dropna(subset=["state"])
    psum = armed[["p0", "p1", "p2"]].sum(axis=1)
    np.testing.assert_allclose(psum.to_numpy(), 1.0, atol=1e-8)


def test_engine_emits_states_not_directional_signals():
    """Scope invariant: a regime engine NEVER emits a directional trade. Guard against
    a column that smells like a signal."""
    df = _synthetic_prices(500, seed=34)
    out = regime_engine(df, warmup=200)
    forbidden = {"side", "signal", "position", "buy", "sell", "weight", "order"}
    assert forbidden.isdisjoint({c.lower() for c in out.columns})


# --------------------------------------------------------------------------- #
# shuffle-label null — the conditioning-value honesty test (8.1 / 8.2 KILL)
# --------------------------------------------------------------------------- #
def test_shuffle_null_detects_a_genuinely_informative_gate():
    # Trades in regime 1 have a clearly higher mean than regime 0 -> the gate is real.
    rng = np.random.default_rng(40)
    labels = np.array([0, 1] * 300)
    rets = np.where(labels == 1, rng.normal(0.02, 0.05, 600), rng.normal(-0.01, 0.05, 600))
    res = shuffle_label_null(rets, labels, keep={1}, metric="mean", n_perm=2000, seed=0)
    assert res["observed"] > 0
    assert res["p_value"] < 0.05
    assert 0.0 <= res["p_value"] <= 1.0


def test_shuffle_null_gives_no_false_positive_on_a_useless_gate():
    # Constant returns: every permutation yields the same (zero) contrast -> p == 1.
    labels = np.array([0, 1, 2] * 100)
    rets = np.full(300, 0.01)
    res = shuffle_label_null(rets, labels, keep={1}, metric="mean", n_perm=500, seed=1)
    assert res["observed"] == pytest.approx(0.0, abs=1e-12)
    assert res["p_value"] == pytest.approx(1.0)


def test_shuffle_null_is_deterministic_given_seed():
    rng = np.random.default_rng(41)
    labels = rng.integers(0, 3, 400)
    rets = rng.normal(0, 0.05, 400)
    a = shuffle_label_null(rets, labels, keep={2}, n_perm=1000, seed=7)
    b = shuffle_label_null(rets, labels, keep={2}, n_perm=1000, seed=7)
    assert a["p_value"] == b["p_value"] and a["observed"] == b["observed"]


def test_shuffle_null_accepts_callable_keep():
    rng = np.random.default_rng(42)
    labels = rng.integers(0, 3, 300)
    rets = rng.normal(0, 0.05, 300)
    res = shuffle_label_null(rets, labels, keep=lambda x: x >= 1, n_perm=500, seed=0)
    assert 0.0 <= res["p_value"] <= 1.0


# --------------------------------------------------------------------------- #
# OOS state stability (8.1 KILL) — states must not re-segment on re-estimation
# --------------------------------------------------------------------------- #
def test_state_stability_high_when_regimes_are_persistent():
    rng = np.random.default_rng(43)
    # Very separable, persistent 2-regime features -> two fits on different windows
    # should agree on the held-out tail.
    n = 3000
    states = np.empty(n, dtype=int)
    states[0] = 0
    for t in range(1, n):
        states[t] = states[t - 1] if rng.random() < 0.99 else 1 - states[t - 1]
    f1 = rng.normal(np.where(states == 0, -1.0, 1.0), 0.2)
    f2 = rng.normal(np.where(states == 0, 0.0, 1.5), 0.2)
    X = np.column_stack([f1, f2])
    res = state_stability(X, frac_a=0.5, frac_b=0.7, n_states=2)
    assert 0.0 <= res["agreement"] <= 1.0
    assert res["n_oos"] > 0
    assert res["agreement"] > 0.8


def test_state_stability_returns_unit_interval():
    rng = np.random.default_rng(44)
    X = rng.normal(0, 1, (800, 2))
    res = state_stability(X, frac_a=0.5, frac_b=0.7, n_states=3)
    assert 0.0 <= res["agreement"] <= 1.0


def test_state_stability_excess_is_high_only_for_genuine_structure():
    """The HEADLINE number must be chance-corrected. Persistent regimes => large
    excess-over-chance; the raw agreement alone is not trustworthy."""
    rng = np.random.default_rng(43)
    n = 3000
    states = np.empty(n, dtype=int)
    states[0] = 0
    for t in range(1, n):
        states[t] = states[t - 1] if rng.random() < 0.99 else 1 - states[t - 1]
    f1 = rng.normal(np.where(states == 0, -1.0, 1.0), 0.2)
    f2 = rng.normal(np.where(states == 0, 0.0, 1.5), 0.2)
    res = state_stability(np.column_stack([f1, f2]), frac_a=0.5, frac_b=0.7, n_states=2)
    assert res["agreement_excess"] > 0.2
    assert 0.0 <= res["chance_baseline"] <= 1.0


def test_state_stability_excess_near_zero_on_structureless_noise():
    """iid noise has NO regime structure: raw agreement can still look high (marginal
    imbalance), but the chance-corrected excess must be ~0 — otherwise the metric is
    fooled exactly where it must not be."""
    rng = np.random.default_rng(44)
    X = rng.normal(0, 1, (1500, 2))
    res = state_stability(X, frac_a=0.5, frac_b=0.7, n_states=3)
    assert res["agreement_excess"] < 0.10


def test_state_stability_collapsed_single_state_is_not_called_stable():
    """A degenerate fit that assigns ONE state to every bar is the exact failure mode
    this metric exists to catch — its raw agreement is a perfect 1.0, so the excess
    (not the raw) must report it as NOT stable."""
    sa = np.zeros(400, dtype=int)
    sb = np.ones(400, dtype=int)  # collapsed to a different single state
    # best-match maps them onto each other -> raw 1.0; excess must be ~0.
    from hermes.research.s8_regime_engine import _best_match_agreement
    assert _best_match_agreement(sa, sb, 3) == pytest.approx(1.0)
    X = np.full((900, 2), 0.0) + np.random.default_rng(7).normal(0, 1e-9, (900, 2))
    res = state_stability(X, frac_a=0.5, frac_b=0.7, n_states=3)
    assert res["agreement_excess"] < 0.10


# --------------------------------------------------------------------------- #
# asof_labels — causal entry-bar attach (most recent label <= event)
# --------------------------------------------------------------------------- #
def test_asof_labels_picks_most_recent_past_label():
    label_ts = np.array([100, 200, 300], dtype=float)
    label_val = np.array([10, 20, 30], dtype=float)
    events = np.array([150, 200, 250, 305], dtype=float)
    got = asof_labels(events, label_ts, label_val)
    np.testing.assert_array_equal(got, [10, 20, 20, 30])


def test_asof_labels_nan_before_first_label():
    label_ts = np.array([100, 200], dtype=float)
    label_val = np.array([10, 20], dtype=float)
    got = asof_labels(np.array([50, 100], dtype=float), label_ts, label_val)
    assert np.isnan(got[0]) and got[1] == 10


# --------------------------------------------------------------------------- #
# macro gates (8.2) — business-day lag (point-in-time) + trailing bucket
# --------------------------------------------------------------------------- #
def test_lag_business_days_moves_friday_to_monday():
    import pandas as pd
    s = pd.Series([1.0], index=[pd.Timestamp("2024-01-05")])  # a Friday
    lagged = lag_business_days(s, 1)
    assert lagged.index[0] == pd.Timestamp("2024-01-08")  # Monday (skips weekend)


def test_macro_regime_gate_is_causal_trailing_bucket():
    import pandas as pd
    rng = np.random.default_rng(45)
    dates = pd.bdate_range("2018-01-01", periods=600)
    s = pd.Series(np.abs(rng.normal(3, 1, 600)), index=dates)
    gate = macro_regime_gate(s, lag_busdays=1, kind="trailing_bucket", window=252, n_buckets=3)
    # Lagged by 1 bday => availability index is shifted; values are terciles in {0,1,2}.
    vals = gate.dropna().unique()
    assert set(vals).issubset({0.0, 1.0, 2.0})
    # Causality: recomputing on a prefix gives the same gate on the overlap.
    gate_pref = macro_regime_gate(s.iloc[:400], lag_busdays=1, kind="trailing_bucket",
                                  window=252, n_buckets=3)
    common = gate.index.intersection(gate_pref.index)
    a = gate.reindex(common).to_numpy()
    b = gate_pref.reindex(common).to_numpy()
    mask = ~(np.isnan(a) | np.isnan(b))
    np.testing.assert_array_equal(a[mask], b[mask])
