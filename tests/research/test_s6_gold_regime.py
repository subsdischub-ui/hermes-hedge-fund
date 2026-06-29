"""Stream 6 — gold regime conditioner. Tests for the point-in-time state machinery.

The unit under test is a *state*, never a trade (mandate 6.1). These tests pin the
two properties the mandate makes non-negotiable: the percentile/correlation helpers
are causal (no look-ahead), and the regime state is point-in-time (publication lags
honoured) so a downstream sleeve that consumes it cannot inherit a leak.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hermes.research import s6_gold_regime as s6


# --------------------------------------------------------------------------- #
# expanding_percentile — causal rank, no look-ahead
# --------------------------------------------------------------------------- #
def test_expanding_percentile_is_fraction_of_strictly_prior_below():
    # at index i: fraction of values at indices < i that are < s[i]
    s = pd.Series([10.0, 20.0, 30.0])
    out = s6.expanding_percentile(s, min_periods=1)
    assert np.isnan(out.iloc[0])          # no prior history -> undefined
    assert out.iloc[1] == 1.0             # {10} all < 20
    assert out.iloc[2] == 1.0             # {10,20} all < 30


def test_expanding_percentile_descending_series_is_zero():
    s = pd.Series([30.0, 20.0, 10.0])
    out = s6.expanding_percentile(s, min_periods=1)
    assert out.iloc[1] == 0.0             # 30 < 20 is false
    assert out.iloc[2] == 0.0             # neither 30 nor 20 < 10


def test_expanding_percentile_respects_min_periods():
    s = pd.Series([1.0, 2.0, 3.0, 4.0])
    out = s6.expanding_percentile(s, min_periods=3)
    # need >=3 prior obs before emitting -> first non-nan at index 3
    assert out.iloc[:3].isna().all()
    assert not np.isnan(out.iloc[3])


def test_expanding_percentile_uses_only_past_not_future():
    # a late spike must NOT change earlier percentile values (causality)
    base = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    spiked = base.copy()
    spiked.iloc[-1] = 999.0
    a = s6.expanding_percentile(base, min_periods=1)
    b = s6.expanding_percentile(spiked, min_periods=1)
    pd.testing.assert_series_equal(a.iloc[:-1], b.iloc[:-1])


# --------------------------------------------------------------------------- #
# rolling_corr
# --------------------------------------------------------------------------- #
def test_rolling_corr_perfectly_correlated_is_one():
    a = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    b = 3.0 * a + 7.0
    out = s6.rolling_corr(a, b, window=3)
    assert out.iloc[-1] == pytest.approx(1.0)


def test_rolling_corr_perfectly_anticorrelated_is_minus_one():
    a = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    b = -a
    out = s6.rolling_corr(a, b, window=3)
    assert out.iloc[-1] == pytest.approx(-1.0)


# --------------------------------------------------------------------------- #
# cb_demand_pit — quarterly tonnes -> annualized, publication-lagged step series
# --------------------------------------------------------------------------- #
def test_cb_demand_pit_annualizes_trailing_four_quarters():
    rows = pd.DataFrame(
        {"year": [2021, 2021, 2021, 2021], "quarter": [1, 2, 3, 4], "tonnes": [100.0, 100.0, 100.0, 100.0]}
    )
    out = s6.cb_demand_pit(rows, lag_days=45)
    # the 4-quarter sum (400) becomes available only after Q4 close + 45d
    q4_avail = pd.Timestamp("2021-12-31") + pd.Timedelta(days=45)
    assert out.loc[q4_avail] == 400.0


def test_cb_demand_pit_availability_date_lags_quarter_end():
    rows = pd.DataFrame(
        {"year": [2022, 2022, 2022, 2022], "quarter": [1, 2, 3, 4], "tonnes": [50.0, 60.0, 70.0, 80.0]}
    )
    out = s6.cb_demand_pit(rows, lag_days=45)
    # value is NOT knowable at the quarter end itself (PIT) — index starts at avail dates
    assert out.index.min() > pd.Timestamp("2022-12-31") - pd.Timedelta(days=400)
    # the Q4 availability date is quarter_end + 45d, not the quarter end
    assert (pd.Timestamp("2022-12-31") + pd.Timedelta(days=45)) in out.index


# --------------------------------------------------------------------------- #
# gold_regime_state — the integrated PIT state {REAL_RATE, FLOW_CB}
# --------------------------------------------------------------------------- #
def _synthetic_inputs(n=1000):
    """First half: gold strongly (negatively) coupled to real rates + low CB demand
    -> REAL_RATE. Second half: decoupled (corr~0) + high CB demand -> FLOW_CB. The
    series is long enough (and the CB step date-aligned to the coupling break) that a
    labelled REAL_RATE stretch survives the rolling/percentile warm-up."""
    idx = pd.bdate_range("2014-01-01", periods=n)
    boundary = idx[n // 2]
    rng = np.random.default_rng(0)
    half = n // 2
    ddfii = rng.normal(0, 0.02, n)
    gret = np.empty(n)
    gret[:half] = -1.0 * ddfii[:half] + rng.normal(0, 0.001, half)   # tight neg coupling
    gret[half:] = rng.normal(0, 0.01, n - half)                       # decoupled
    gold = pd.Series(100.0 * np.exp(np.cumsum(gret)), index=idx)
    dfii = pd.Series(np.cumsum(ddfii) + 1.0, index=idx)
    # CB demand stepped at the *same calendar point* as the coupling break: low (200)
    # for quarters ending before the boundary, high (1100) after.
    rows = []
    for p in pd.period_range(idx[0], idx[-1], freq="Q"):
        hi = p.end_time.normalize() >= boundary
        rows.append({"year": p.year, "quarter": p.quarter, "tonnes": (1100.0 if hi else 200.0) / 4})
    cb = pd.DataFrame(rows)
    return gold, dfii, cb


def test_gold_regime_state_flips_real_rate_to_flow_cb():
    gold, dfii, cb = _synthetic_inputs()
    cb_ann = s6.cb_demand_pit(cb, lag_days=45)
    state = s6.gold_regime_state(gold, dfii, cb_ann, window=120, min_periods=40)
    labels = state["state"].dropna()
    # early window dominated by REAL_RATE, late window dominated by FLOW_CB
    early = labels.iloc[: len(labels) // 3]
    late = labels.iloc[-len(labels) // 3:]
    assert (early == "REAL_RATE").mean() > 0.5
    assert (late == "FLOW_CB").mean() > 0.5


def test_gold_regime_state_is_point_in_time_no_lookahead():
    # truncating the future must not change any past state value (causality)
    gold, dfii, cb = _synthetic_inputs()
    cb_ann = s6.cb_demand_pit(cb, lag_days=45)
    full = s6.gold_regime_state(gold, dfii, cb_ann, window=120, min_periods=40)
    cut = len(gold) - 50
    trunc = s6.gold_regime_state(
        gold.iloc[:cut], dfii.iloc[:cut], cb_ann, window=120, min_periods=40
    )
    common = full.index.intersection(trunc.index)
    pd.testing.assert_series_equal(
        full.loc[common, "flow_score"], trunc.loc[common, "flow_score"], check_names=False
    )


def test_gold_regime_state_columns_present():
    gold, dfii, cb = _synthetic_inputs()
    cb_ann = s6.cb_demand_pit(cb, lag_days=45)
    state = s6.gold_regime_state(gold, dfii, cb_ann, window=120, min_periods=40)
    for col in ("coupling", "cb_ann", "flow_score", "state"):
        assert col in state.columns
