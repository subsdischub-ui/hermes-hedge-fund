"""TDD for Stream-4 options primitives (s4_options).

The lab's recurring failure mode is look-ahead, and the S4 invariant is the
strictest of all: a sizing gate is read AT trade entry, so its VRP input must be
knowable at entry — DVOL lagged +1 bar, realized vol TRAILING (never forward).
These tests pin that no-future-leak guarantee FIRST, then the unit homogeneity of
VRP (both legs annualized identically), then the four FROZEN gate forms. The gate
statistics (Sharpe/DSR/joint-null) are already tested in tests/test_book.py and
research_fx/test_fxgate.py — we reuse, never re-verify, them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from hermes.research import s4_options as s4


# --------------------------------------------------------------------------- #
# realized_vol_cc — close-to-close, annualized to vol points, CAUSAL
# --------------------------------------------------------------------------- #
def test_realized_vol_cc_matches_trailing_log_return_std_formula():
    """rv_t = std(log returns over trailing window, ddof=1) * sqrt(ppy) * 100."""
    rets = np.array([0.01, -0.02, 0.015, -0.01, 0.02, -0.005, 0.008])
    close = pd.Series(100.0 * np.exp(np.cumsum(rets)))
    window, ppy = 4, 365.0

    rv = s4.realized_vol_cc(close, window=window, periods_per_year=ppy)

    expected = float(np.std(rets[-window:], ddof=1) * np.sqrt(ppy) * 100.0)
    assert np.isfinite(rv.iloc[-1])
    assert np.isclose(rv.iloc[-1], expected)


def test_realized_vol_cc_is_causal_future_cannot_change_past():
    """rv at index t is identical whether or not future bars exist (no look-ahead)."""
    rng = np.random.default_rng(0)
    close = pd.Series(100.0 * np.exp(np.cumsum(rng.normal(0, 0.02, size=200))))
    window = 30

    full = s4.realized_vol_cc(close, window=window)
    prefix = s4.realized_vol_cc(close.iloc[:120], window=window)

    pd.testing.assert_series_equal(full.iloc[:120], prefix, check_names=False)


def test_realized_vol_cc_zero_for_constant_log_returns():
    """A constant geometric drift has zero realized vol (std of equal returns = 0)."""
    close = pd.Series(100.0 * (1.01 ** np.arange(50)))  # constant +1% log-ish step
    rv = s4.realized_vol_cc(close, window=10)
    assert np.isclose(rv.iloc[-1], 0.0, atol=1e-9)


# --------------------------------------------------------------------------- #
# realized_vol_parkinson — range estimator cross-check, same units
# --------------------------------------------------------------------------- #
def test_realized_vol_parkinson_matches_high_low_range_formula():
    """parkinson_t = sqrt( mean( ln(H/L)^2 ) / (4 ln2) ) * sqrt(ppy) * 100."""
    high = pd.Series([105.0, 106.0, 104.0, 107.0, 103.0])
    low = pd.Series([100.0, 101.0, 99.0, 102.0, 98.0])
    window, ppy = 3, 365.0

    pk = s4.realized_vol_parkinson(high, low, window=window, periods_per_year=ppy)

    hl = np.log((high / low).to_numpy())[-window:]
    expected = float(np.sqrt(np.mean(hl**2) / (4.0 * np.log(2.0)) * ppy) * 100.0)
    assert np.isclose(pk.iloc[-1], expected)


# --------------------------------------------------------------------------- #
# compute_vrp — DVOL +1 bar lag vs trailing realized, NO future leak
# --------------------------------------------------------------------------- #
def test_compute_vrp_lags_dvol_one_bar_and_subtracts_realized():
    """VRP_t = DVOL_{t-1} - realized_t (both already in vol points)."""
    dvol = pd.Series([80.0, 82.0, 79.0, 85.0, 88.0])
    realized = pd.Series([60.0, 61.0, 62.0, 63.0, 64.0])

    vrp = s4.compute_vrp(dvol, realized, dvol_lag=1)

    # t=1: DVOL_0(80) - realized_1(61) = 19 ; t=4: DVOL_3(85) - realized_4(64) = 21
    assert np.isnan(vrp.iloc[0])  # no DVOL_{-1}
    assert np.isclose(vrp.iloc[1], 80.0 - 61.0)
    assert np.isclose(vrp.iloc[4], 85.0 - 64.0)


def test_compute_vrp_is_causal_future_cannot_change_past():
    """VRP at t uses DVOL[t-1] and realized[t] only — truncating the future is a no-op."""
    rng = np.random.default_rng(1)
    dvol = pd.Series(70 + rng.normal(0, 5, size=100).cumsum() * 0.0 + rng.normal(0, 5, 100))
    realized = pd.Series(50 + rng.normal(0, 3, size=100))

    full = s4.compute_vrp(dvol, realized, dvol_lag=1)
    prefix = s4.compute_vrp(dvol.iloc[:60], realized.iloc[:60], dvol_lag=1)

    pd.testing.assert_series_equal(full.iloc[:60], prefix, check_names=False)


# --------------------------------------------------------------------------- #
# size_multiplier — the FOUR frozen pre-registered gate forms
# --------------------------------------------------------------------------- #
def test_size_multiplier_binary_positive_sign_up_in_fear():
    """binary +1: m=1.5 when VRP_z>=0 (fear/rich vol), else 0.5; NaN warmup -> neutral 1.0."""
    vrp_z = pd.Series([np.nan, -2.0, -0.01, 0.0, 1.5])
    m = s4.size_multiplier(vrp_z, form="binary", sign=1)
    assert np.isclose(m.iloc[0], 1.0)   # warmup neutral
    assert np.isclose(m.iloc[1], 0.5)
    assert np.isclose(m.iloc[2], 0.5)
    assert np.isclose(m.iloc[3], 1.5)   # >=0
    assert np.isclose(m.iloc[4], 1.5)


def test_size_multiplier_binary_negative_sign_up_in_complacency():
    """binary -1: m=1.5 when VRP_z<0, else 0.5 (the opposite sign hypothesis)."""
    vrp_z = pd.Series([np.nan, -2.0, 0.0, 1.5])
    m = s4.size_multiplier(vrp_z, form="binary", sign=-1)
    assert np.isclose(m.iloc[0], 1.0)
    assert np.isclose(m.iloc[1], 1.5)   # <0 -> up
    assert np.isclose(m.iloc[2], 0.5)
    assert np.isclose(m.iloc[3], 0.5)


def test_size_multiplier_linear_clips_to_bounds_and_centers_on_one():
    """linear +1: m=clip(1+0.25*VRP_z, 0.5, 1.5); NaN warmup -> 1.0."""
    vrp_z = pd.Series([np.nan, 0.0, 1.0, 10.0, -10.0])
    m = s4.size_multiplier(vrp_z, form="linear", sign=1, beta=0.25)
    assert np.isclose(m.iloc[0], 1.0)
    assert np.isclose(m.iloc[1], 1.0)            # 1 + 0.25*0
    assert np.isclose(m.iloc[2], 1.25)           # 1 + 0.25*1
    assert np.isclose(m.iloc[3], 1.5)            # clipped high
    assert np.isclose(m.iloc[4], 0.5)            # clipped low


def test_size_multiplier_linear_negative_sign_flips_response():
    """linear -1: m=clip(1-0.25*VRP_z, 0.5, 1.5)."""
    vrp_z = pd.Series([1.0, -1.0])
    m = s4.size_multiplier(vrp_z, form="linear", sign=-1, beta=0.25)
    assert np.isclose(m.iloc[0], 0.75)   # 1 - 0.25
    assert np.isclose(m.iloc[1], 1.25)   # 1 + 0.25


def test_size_multiplier_is_causal_warmup_is_neutral_not_dropped():
    """Warmup (NaN VRP_z) must be neutral 1.0 so the gated series keeps every base trade."""
    vrp_z = pd.Series([np.nan, np.nan, 0.5])
    for form in ("binary", "linear"):
        m = s4.size_multiplier(vrp_z, form=form, sign=1)
        assert m.notna().all()
        assert np.isclose(m.iloc[0], 1.0) and np.isclose(m.iloc[1], 1.0)


def test_size_multiplier_rejects_unknown_form():
    import pytest
    with pytest.raises(ValueError):
        s4.size_multiplier(pd.Series([0.0]), form="sigmoid", sign=1)


# --------------------------------------------------------------------------- #
# gate_trade_returns — map the causal daily VRP_z onto each trade's entry, scale
#   the per-trade net return by the multiplier. NO future leak: a trade uses only
#   VRP_z timestamped at/before its entry; a pre-first-z trade is neutral (m=1).
# --------------------------------------------------------------------------- #
def test_gate_trade_returns_uses_last_vrpz_at_or_before_entry():
    vrp_z = pd.Series([-1.0, 0.0, 2.0])
    z_ts = np.array([1000, 2000, 3000])
    entry_ts = np.array([1500, 3500, 500])      # in z1 / past z3 / before any z
    ret_net = np.array([0.10, -0.20, 0.05])

    gated = s4.gate_trade_returns(
        entry_ts, ret_net, vrp_z, z_ts, form="binary", sign=1
    )
    # entry 1500 -> z@1000=-1 -> binary+1 m=0.5 -> 0.05
    assert np.isclose(gated[0], 0.5 * 0.10)
    # entry 3500 -> z@3000=2 -> m=1.5 -> -0.30
    assert np.isclose(gated[1], 1.5 * -0.20)
    # entry 500 -> no z yet -> neutral m=1.0 -> 0.05
    assert np.isclose(gated[2], 1.0 * 0.05)


def test_gate_trade_returns_is_causal_future_vrpz_cannot_change_past_trades():
    rng = np.random.default_rng(3)
    z_ts = np.arange(0, 200) * 1000
    vrp_z = pd.Series(rng.normal(size=200))
    entry_ts = np.array([12_000, 45_000, 99_000])      # all <= ts index 99
    ret_net = rng.normal(size=3)

    full = s4.gate_trade_returns(entry_ts, ret_net, vrp_z, z_ts, form="linear", sign=1)
    # truncate VRP_z to only the bars at/after the last entry's lookup horizon
    keep = 100
    trunc = s4.gate_trade_returns(
        entry_ts, ret_net, vrp_z.iloc[:keep], z_ts[:keep], form="linear", sign=1
    )
    np.testing.assert_allclose(full, trunc)


# --------------------------------------------------------------------------- #
# oos_mask — held-out = last `frac` of the timeline by entry date (the KILL split)
# --------------------------------------------------------------------------- #
def test_oos_mask_selects_last_fraction_by_time():
    ts = np.array([10, 20, 30, 40, 50])
    mask = s4.oos_mask(ts, frac=0.4)        # cutoff = 10 + 0.6*40 = 34 -> {40,50}
    assert list(mask) == [False, False, False, True, True]


def test_oos_mask_handles_unsorted_input_by_value_not_position():
    ts = np.array([50, 10, 40, 20, 30])
    mask = s4.oos_mask(ts, frac=0.4)        # same cutoff 34, keyed on VALUE
    assert list(mask) == [True, False, True, False, False]


# --------------------------------------------------------------------------- #
# 4.2 — aggregate-OI dealer-GEX REGIME proxy (UNSIGNED; no strikes/put-call exist)
# --------------------------------------------------------------------------- #
def test_aggregate_options_oi_sums_selected_exchanges_only():
    df = pd.DataFrame({
        "ts": [1, 1, 1, 2, 2],
        "exchange": ["Deribit", "OKX", "Binance", "Deribit", "OKX"],
        "value": [100.0, 50.0, 999.0, 120.0, 60.0],
    })
    agg = s4.aggregate_options_oi(df, exchanges=("Deribit", "OKX"))
    assert np.isclose(agg.loc[1], 150.0)    # Binance 999 excluded (2023 break)
    assert np.isclose(agg.loc[2], 180.0)


def test_aggregate_options_oi_drops_nan_value_rows():
    df = pd.DataFrame({
        "ts": [1, 1, 2],
        "exchange": ["Deribit", "OKX", "Deribit"],
        "value": [100.0, np.nan, 120.0],
    })
    agg = s4.aggregate_options_oi(df, exchanges=("Deribit", "OKX"))
    assert np.isclose(agg.loc[1], 100.0)    # NaN OKX dropped, not NaN-propagated
    assert np.isclose(agg.loc[2], 120.0)


def test_gex_oi_regime_is_causal_unsigned_zscore():
    rng = np.random.default_rng(5)
    n = 120
    df = pd.DataFrame({
        "ts": np.repeat(np.arange(n), 2),
        "exchange": ["Deribit", "OKX"] * n,
        "value": np.abs(rng.normal(1000, 50, size=2 * n)),
    })
    full = s4.gex_oi_regime(df, exchanges=("Deribit", "OKX"), window=30)
    prefix_df = df[df["ts"] < 80]
    prefix = s4.gex_oi_regime(prefix_df, exchanges=("Deribit", "OKX"), window=30)
    # causal: the regime at ts<80 is unchanged by dropping ts>=80
    pd.testing.assert_series_equal(
        full.loc[full.index < 80], prefix, check_names=False
    )


# --------------------------------------------------------------------------- #
# 4.1 — Black-Scholes delta fallback (so rr25/skew populate offline, no live
#   greeks needed) + idempotent per-UTC-day forward-log dedup for the cron.
# --------------------------------------------------------------------------- #
def test_bs_delta_atm_call_and_put_known_values():
    """ATM (S=K=100, t=1, iv=0.2, r=0): d1=0.1 -> call delta N(0.1)=0.5398, put=-0.4602."""
    dc = s4.bs_delta("C", S=100.0, K=100.0, t_years=1.0, iv=0.20, r=0.0)
    dp = s4.bs_delta("P", S=100.0, K=100.0, t_years=1.0, iv=0.20, r=0.0)
    assert np.isclose(dc, 0.539827837, atol=1e-6)
    assert np.isclose(dp, -0.460172162, atol=1e-6)


def test_bs_delta_put_call_relation_and_bounds():
    """delta_C - delta_P == 1; call delta in (0,1), put in (-1,0)."""
    dc = s4.bs_delta("C", 110.0, 100.0, 0.5, 0.6, 0.0)
    dp = s4.bs_delta("P", 110.0, 100.0, 0.5, 0.6, 0.0)
    assert np.isclose(dc - dp, 1.0, atol=1e-9)
    assert 0.0 < dc < 1.0
    assert -1.0 < dp < 0.0


def test_bs_delta_monotonic_in_spot():
    """Higher spot -> higher call delta (toward 1)."""
    lo = s4.bs_delta("C", 90.0, 100.0, 0.25, 0.5, 0.0)
    hi = s4.bs_delta("C", 120.0, 100.0, 0.25, 0.5, 0.0)
    assert hi > lo


def test_bs_delta_degenerate_returns_none():
    """Expired or zero-vol -> no usable BS delta (caller skips, never fabricates)."""
    assert s4.bs_delta("C", 100.0, 100.0, 0.0, 0.2, 0.0) is None
    assert s4.bs_delta("C", 100.0, 100.0, 1.0, 0.0, 0.0) is None


def test_dedup_forward_log_keeps_one_row_per_ccy_per_utc_day():
    """A daily cron re-run on the same UTC day OVERWRITES that day's row (idempotent)."""
    day1 = 1_700_000_000_000  # some ms in one UTC day
    day1_later = day1 + 3_600_000  # +1h, same UTC day
    day2 = day1 + 86_400_000      # +1 day
    existing = pd.DataFrame({"ts": [day1], "ccy": ["BTC"], "rr25_near": [-8.0]})
    new = pd.DataFrame({
        "ts": [day1_later, day2],
        "ccy": ["BTC", "BTC"],
        "rr25_near": [-9.0, -7.0],
    })
    merged = s4.dedup_forward_log(existing, new, ts_col="ts", key_cols=("ccy",))
    # one BTC row for day1 (the LATER value wins) + one for day2
    assert len(merged) == 2
    btc = merged.sort_values("ts")
    assert np.isclose(btc.iloc[0]["rr25_near"], -9.0)   # same-day overwrite
    assert np.isclose(btc.iloc[1]["rr25_near"], -7.0)


def test_dedup_forward_log_separates_distinct_ccys_same_day():
    day1 = 1_700_000_000_000
    new = pd.DataFrame({"ts": [day1, day1], "ccy": ["BTC", "ETH"], "rr25_near": [-8.0, -7.0]})
    merged = s4.dedup_forward_log(None, new, ts_col="ts", key_cols=("ccy",))
    assert len(merged) == 2
    assert set(merged["ccy"]) == {"BTC", "ETH"}
