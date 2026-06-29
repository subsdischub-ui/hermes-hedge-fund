"""Tests for Stream 2 v-next — cross-sectional market-neutral sleeves (s2_xsec).

The lab's recurring failure mode is look-ahead, and composition bugs in a NEW
backtester manufacture false POSITIVES — the #1 risk. So the very first tests here
are the L0 leak-injection battery on the cross-sectional engine: a signal taken FROM
the future must produce an enormous edge; the SAME signal lagged properly must destroy
it; and truncating future bars must never change a past sleeve return. The gating math
(Sharpe/DSR/PBO) is already tested under hermes.learning/validation — reused, not
re-verified.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hermes.research import s2_xsec as xs


# --------------------------------------------------------------------------- #
# cross_sectional_weights — dollar-neutral long-short construction
# --------------------------------------------------------------------------- #
def test_xs_weights_long_top_short_bottom_dollar_neutral():
    """Long the top-frac by signal, short the bottom-frac, equal weight per leg,
    each leg gross 0.5 so the book is dollar-neutral (sum 0) and gross-1."""
    sig = np.array([3.0, 1.0, 2.0, 0.0, -1.0, -2.0])  # 6 names
    w = xs.cross_sectional_weights(sig, top_frac=1 / 3, bottom_frac=1 / 3, min_names=4)
    # top 2 by value: idx0 (3), idx2 (2) -> +0.25 each ; bottom 2: idx5(-2), idx4(-1) -> -0.25
    assert w[0] == pytest.approx(0.25)
    assert w[2] == pytest.approx(0.25)
    assert w[5] == pytest.approx(-0.25)
    assert w[4] == pytest.approx(-0.25)
    assert w[1] == 0.0 and w[3] == 0.0  # the middle is flat
    assert w.sum() == pytest.approx(0.0)          # dollar-neutral
    assert np.abs(w).sum() == pytest.approx(1.0)  # gross 1


def test_xs_weights_excludes_nan_unavailable_names():
    """Names unavailable at t (NaN signal) are never ranked and stay flat — the
    variable-universe rule: the long-short at t uses only names listed at t."""
    sig = np.array([np.nan, 5.0, 1.0, -1.0, -5.0, np.nan])  # 4 available
    w = xs.cross_sectional_weights(sig, top_frac=1 / 3, bottom_frac=1 / 3, min_names=4)
    assert w[0] == 0.0 and w[5] == 0.0           # NaN -> flat, never a position
    assert w[1] == pytest.approx(0.5)            # top of 4 -> 1 name long, gross 0.5
    assert w[4] == pytest.approx(-0.5)           # bottom of 4 -> 1 name short
    assert w.sum() == pytest.approx(0.0)


def test_xs_weights_too_few_names_is_flat():
    """Below min_names available, take NO bet (all zeros) — never force a long-short
    on a skeleton universe (early-history guard)."""
    sig = np.array([1.0, np.nan, np.nan, -1.0, np.nan])  # only 2 available
    w = xs.cross_sectional_weights(sig, top_frac=1 / 3, bottom_frac=1 / 3, min_names=4)
    assert np.all(w == 0.0)


def test_xs_weights_all_equal_signal_is_flat():
    """A degenerate cross-section (all equal) has no high/low to separate -> flat,
    never an arbitrary tie-broken bet."""
    sig = np.array([2.0, 2.0, 2.0, 2.0, 2.0, 2.0])
    w = xs.cross_sectional_weights(sig, top_frac=1 / 3, bottom_frac=1 / 3, min_names=4)
    assert np.all(w == 0.0)


# --------------------------------------------------------------------------- #
# xsec_backtest — the cross-sectional engine (gross, funding, cost)
# --------------------------------------------------------------------------- #
def _flat_panel(n=30, syms=("A", "B", "C", "D", "E", "F")):
    idx = np.arange(n) * 4 * 3600 * 1000  # 4h grid in ms
    close = pd.DataFrame(100.0, index=idx, columns=list(syms))
    return idx, close


def test_xsec_backtest_gross_is_long_short_spread():
    """Gross per-rebalance return = Σ w_i·fwd_i. Plant a clean spread: top names
    rise, bottom names fall over the holding window -> positive long-short return."""
    idx, close = _flat_panel(n=20)
    # signal: A,B high (long), E,F low (short); flat C,D
    sig = pd.DataFrame(
        np.tile([2.0, 1.5, 0.0, 0.0, -1.5, -2.0], (20, 1)),
        index=idx, columns=list(close.columns),
    )
    # over the NEXT bar the longs gain +2%, the shorts lose -2%
    close.iloc[1::1, :] = 100.0
    for t in range(0, 19):
        close.iloc[t + 1, 0] = close.iloc[t, 0] * 1.02  # A up
        close.iloc[t + 1, 1] = close.iloc[t, 1] * 1.02  # B up
        close.iloc[t + 1, 4] = close.iloc[t, 4] * 0.98  # E down
        close.iloc[t + 1, 5] = close.iloc[t, 5] * 0.98  # F down
    res = xs.xsec_backtest(close, sig, funding_per_bar=None, stride=1,
                           top_frac=1 / 3, bottom_frac=1 / 3, cost_bps=0.0, min_names=4)
    g = res["gross_returns"]
    assert len(g) > 0
    # each leg 0.5 gross: long +2% short +2% -> ~ (0.5*0.02)+(0.5*0.02) = 0.02
    assert g.mean() == pytest.approx(0.02, abs=2e-3)


def test_xsec_backtest_charges_two_leg_turnover_cost():
    """Establishing the book from flat costs cost_bps × gross turnover (=1.0 here)."""
    idx, close = _flat_panel(n=12)
    sig = pd.DataFrame(np.tile([2.0, 1.0, 0.0, 0.0, -1.0, -2.0], (12, 1)),
                       index=idx, columns=list(close.columns))
    res = xs.xsec_backtest(close, sig, funding_per_bar=None, stride=1,
                           top_frac=1 / 3, bottom_frac=1 / 3, cost_bps=20.0, min_names=4)
    # flat market -> gross 0; first rebalance builds gross-1 book from flat:
    # turnover 1.0 -> cost 20bps = 0.002 on that rebalance.
    net = res["returns"]
    assert net.iloc[0] == pytest.approx(-0.002, abs=1e-9)


def test_xsec_backtest_funding_cost_charged_once_per_holding_not_per_bar():
    """Funding PnL = Σ (−w_i)·(funding settled over the holding window). A LONG of a
    positive-funding name PAYS; the carry is summed over the held bars, never double-
    counted from a forward-filled rate."""
    idx, close = _flat_panel(n=8)
    sig = pd.DataFrame(np.tile([2.0, 1.0, 0.0, 0.0, -1.0, -2.0], (8, 1)),
                       index=idx, columns=list(close.columns))
    # funding settled per bar: name A (a long) pays +0.001 at exactly one settled bar
    # within each 2-bar holding window; everything else zero.
    fpb = pd.DataFrame(0.0, index=idx, columns=list(close.columns))
    fpb.iloc[1::2, 0] = 0.001  # one settlement per 2-bar window on name A
    res = xs.xsec_backtest(close, sig, funding_per_bar=fpb, stride=2,
                           top_frac=1 / 3, bottom_frac=1 / 3, cost_bps=0.0, min_names=4)
    # A is in the long leg with weight +0.25; long pays funding -> -0.25 * 0.001 per window
    assert res["returns"].iloc[0] == pytest.approx(-0.25 * 0.001, abs=1e-12)


# --------------------------------------------------------------------------- #
# L0 — LEAK INJECTION on the cross-sectional engine (TEST #1, NON-NEGOTIABLE)
# --------------------------------------------------------------------------- #
def test_L0_future_signal_produces_huge_edge_lagged_signal_destroys_it():
    """THE leak test. A signal that IS the forward return (perfect foresight) must
    produce a large positive edge; the SAME signal shifted so it only reflects PAST
    returns must collapse to ~0. A backtester that scores the lagged (honest) version
    as profitable has a look-ahead bug — the bug that fabricates false positives.
    """
    rng = np.random.default_rng(7)
    n, syms = 240, [f"S{i}" for i in range(8)]
    idx = np.arange(n) * 4 * 3600 * 1000
    rets = rng.normal(0, 0.02, size=(n, len(syms)))
    close = pd.DataFrame(100.0 * np.exp(np.cumsum(rets, axis=0)), index=idx, columns=syms)

    stride = 1
    fwd = close.shift(-stride) / close - 1.0  # the forward target itself

    # (a) ORACLE: signal == the future return -> rank by future -> max edge.
    oracle = xs.xsec_backtest(close, fwd, funding_per_bar=None, stride=stride,
                              top_frac=1 / 4, bottom_frac=1 / 4, cost_bps=0.0, min_names=4)
    # (b) HONEST: shift the oracle signal back by `stride` so at t it equals a PAST
    #     return (knowable at t) -> no predictive power.
    lagged = fwd.shift(stride)
    honest = xs.xsec_backtest(close, lagged, funding_per_bar=None, stride=stride,
                              top_frac=1 / 4, bottom_frac=1 / 4, cost_bps=0.0, min_names=4)

    assert oracle["gross_returns"].mean() > 0.01       # perfect foresight is hugely +ve
    assert abs(honest["gross_returns"].mean()) < 0.002  # honest signal has no edge
    # and the leak is unmistakable: oracle dwarfs honest
    assert oracle["gross_returns"].mean() > 5 * abs(honest["gross_returns"].mean())


def test_L0_truncating_future_bars_does_not_change_past_sleeve_returns():
    """Causality: the sleeve return at rebalance t is fixed once bars [t, t+stride]
    exist. Appending or removing LATER bars must not change any earlier return."""
    rng = np.random.default_rng(11)
    n, syms = 120, [f"S{i}" for i in range(6)]
    idx = np.arange(n) * 4 * 3600 * 1000
    rets = rng.normal(0, 0.02, size=(n, len(syms)))
    close = pd.DataFrame(100.0 * np.exp(np.cumsum(rets, axis=0)), index=idx, columns=syms)
    sig = pd.DataFrame(rng.normal(size=(n, len(syms))), index=idx, columns=syms)

    full = xs.xsec_backtest(close, sig, funding_per_bar=None, stride=3,
                            top_frac=1 / 3, bottom_frac=1 / 3, cost_bps=5.0, min_names=4)
    cut = 90
    prefix = xs.xsec_backtest(close.iloc[:cut], sig.iloc[:cut], funding_per_bar=None,
                              stride=3, top_frac=1 / 3, bottom_frac=1 / 3,
                              cost_bps=5.0, min_names=4)
    # every rebalance whose holding window ends before the cut must match bar-for-bar.
    # The index is epoch-ms (bar position = ts / step); map back to bar position.
    step = 4 * 3600 * 1000
    common = prefix["returns"].index.intersection(full["returns"].index)
    common = [t for t in common if (t // step) + 3 < cut]
    assert len(common) > 5
    a = full["returns"].loc[common].to_numpy()
    b = prefix["returns"].loc[common].to_numpy()
    np.testing.assert_allclose(a, b, atol=1e-12)


# --------------------------------------------------------------------------- #
# mc_random_sign_pvalue — the FLOOR: is the long-short SIGN informative?
# --------------------------------------------------------------------------- #
def test_mc_sign_null_small_p_for_persistent_edge():
    """A consistently positive long-short stream is rarely beaten by random sign flips."""
    g = pd.Series(np.r_[np.full(60, 0.01), np.full(40, 0.012)])
    p = xs.mc_random_sign_pvalue(g, n_sims=4000, seed=0)
    assert p < 0.05


def test_mc_sign_null_large_p_for_noise():
    """A zero-mean noise stream: random sign flips beat it about half the time."""
    rng = np.random.default_rng(3)
    g = pd.Series(rng.normal(0, 0.01, size=200))
    p = xs.mc_random_sign_pvalue(g, n_sims=4000, seed=0)
    assert p > 0.10


# --------------------------------------------------------------------------- #
# Wave 2 — causal signal builders, funding-cost bucketing, regime
# --------------------------------------------------------------------------- #
def test_realized_vol_panel_is_causal_and_ranks_high_vol_higher():
    """Trailing std of log-returns per column; high-vol column > low-vol column; and
    truncating the future never changes a past value (no look-ahead)."""
    rng = np.random.default_rng(5)
    n = 120
    idx = np.arange(n) * 4 * 3600 * 1000
    hi = 100 * np.exp(np.cumsum(rng.normal(0, 0.05, n)))   # high vol
    lo = 100 * np.exp(np.cumsum(rng.normal(0, 0.005, n)))  # low vol
    close = pd.DataFrame({"HI": hi, "LO": lo}, index=idx)
    rv = xs.realized_vol_panel(close, window=20)
    assert rv["HI"].iloc[-1] > rv["LO"].iloc[-1]
    # causal: prefix matches
    pre = xs.realized_vol_panel(close.iloc[:80], window=20)
    pd.testing.assert_frame_equal(rv.iloc[:80], pre, check_names=False)
    # warmup NaN (need `window` returns)
    assert np.isnan(rv["HI"].iloc[0])


def test_realized_vol_known_value():
    """std (ddof=1) of the trailing window of log returns."""
    # log returns of +0.1 then -0.1 repeating
    r = np.array([0.1, -0.1, 0.1, -0.1, 0.1])
    close = pd.Series(100 * np.exp(np.r_[0.0, np.cumsum(r)]))
    df = pd.DataFrame({"X": close.to_numpy()}, index=np.arange(close.size))
    rv = xs.realized_vol_panel(df, window=2)
    # last two log returns [0.1,-0.1] -> std ddof1 = sqrt(0.02) = 0.141421
    assert rv["X"].iloc[-1] == pytest.approx(np.sqrt(0.02), abs=1e-9)


def test_oi_accel_panel_flags_accelerating_build_causally():
    """OI acceleration = causal z of OI growth, applied column-wise (reuses the tested
    s2_positioning primitive). An accelerating build -> high +z; causal."""
    idx = np.arange(33)
    oi = np.r_[np.full(30, 100.0), [110.0, 122.0, 137.0]]
    df = pd.DataFrame({"BTC": oi, "ETH": np.full(33, 50.0)}, index=idx)
    accel = xs.oi_accel_panel(df, window=20)
    assert accel["BTC"].iloc[-1] > 1.0
    pre = xs.oi_accel_panel(df.iloc[:32], window=20)
    pd.testing.assert_frame_equal(accel.iloc[:32], pre, check_names=False)


def test_funding_settled_per_bar_buckets_each_settlement_into_one_bar():
    """A native settlement at ts S is assigned to the grid bar [grid_i, grid_{i+1})
    that contains S — summed there ONCE, so the holding-window sum never double-counts."""
    step = 4 * 3600 * 1000
    grid = pd.Index(np.arange(6) * step)            # bars at 0,4h,8h,12h,16h,20h
    settle = pd.Series([0.001, 0.002], index=[2 * step, 4 * step])  # settle at 8h and 16h
    fpb = xs.funding_settled_per_bar(settle, grid)
    assert fpb.loc[2 * step] == pytest.approx(0.001)   # 8h bar
    assert fpb.loc[4 * step] == pytest.approx(0.002)   # 16h bar
    assert fpb.loc[0] == 0.0 and fpb.loc[1 * step] == 0.0 and fpb.loc[3 * step] == 0.0


def test_funding_signal_on_grid_is_causal_asof_zscore():
    """funding_z as of each grid bar = causal z over the SETTLEMENT series (last W
    settled prints), as-of-backward onto the grid. A grid bar carries the z of the most
    recent settlement at or before it, never a future one."""
    step = 4 * 3600 * 1000
    # 10 settlements every 8h, rising then a spike
    s_ts = np.arange(10) * 2 * step
    s_val = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.01])
    settle = pd.Series(s_val, index=s_ts)
    grid = pd.Index(np.arange(20) * step)
    z = xs.funding_signal_on_grid(settle, grid, window=5)
    # the grid bar just before the spike settlement (ts < 9*2*step) must NOT see the spike
    pre_spike_bar = (9 * 2 - 1) * step   # one 4h bar before the spike settlement
    spike_bar = 9 * 2 * step
    assert z.loc[pre_spike_bar] <= 0.0 or np.isnan(z.loc[pre_spike_bar])
    assert z.loc[spike_bar] > 1.0        # the spike lifts z at/after its settlement


def test_bull_bear_regime_labels_by_trailing_ma():
    up = pd.Series(np.linspace(100, 300, 300))
    down = pd.Series(np.linspace(300, 100, 300))
    assert xs.bull_bear_regime(up, ma_window=200).iloc[-1] == "bull"
    assert xs.bull_bear_regime(down, ma_window=200).iloc[-1] == "bear"
    # warmup before a full MA window is 'unknown' (never silently a regime)
    assert xs.bull_bear_regime(up, ma_window=200).iloc[0] == "unknown"


# --------------------------------------------------------------------------- #
# Wave 3 — funding_squeeze directional 4h re-test, bull vs bear (BONUS)
# --------------------------------------------------------------------------- #
def _squeeze_fixture():
    """A planted edge that lives ONLY in bull and FLIPS in bear (the bull-beta
    signature). 300 bars: bull for [0,150), bear for [150,300). A funding-z spike
    is followed by +3% over the hold in bull, -3% in bear."""
    n, hold = 300, 5
    idx = np.arange(n)
    regime = pd.Series(["bull"] * 150 + ["bear"] * 150, index=idx)
    fz = pd.Series(0.0, index=idx)
    close = pd.Series(100.0, index=idx)
    bull_spikes = [20, 60, 100]
    bear_spikes = [170, 210, 250]
    for t in bull_spikes:
        fz.iloc[t] = 3.0
        for k in range(1, hold + 1):
            close.iloc[t + k] = 100.0 * (1 + 0.03 * k / hold)  # +3% over the hold
    for t in bear_spikes:
        fz.iloc[t] = 3.0
        for k in range(1, hold + 1):
            close.iloc[t + k] = 100.0 * (1 - 0.03 * k / hold)  # -3% over the hold
    fpb = pd.Series(0.0, index=idx)
    return close, fz, regime, fpb, hold


def test_squeeze_directional_edge_lives_in_bull_flips_in_bear():
    """The decisive governance test: if the squeeze only works long in bull and the
    sign flips (or dies) in bear, it is bull-beta, not an edge."""
    close, fz, regime, fpb, hold = _squeeze_fixture()
    r = xs.funding_squeeze_directional(close, fz, regime, fpb,
                                       z_thr=2.0, hold_bars=hold, cost_bps=0.0)
    assert r["bull"]["n"] == 3 and r["bear"]["n"] == 3
    assert r["bull"]["mean"] == pytest.approx(0.03, abs=1e-9)   # +3% in bull
    assert r["bear"]["mean"] == pytest.approx(-0.03, abs=1e-9)  # flips in bear
    assert r["bull"]["mean"] > 0 > r["bear"]["mean"]


def test_squeeze_directional_no_overlap_one_event_per_hold_window():
    """Consecutive z spikes collapse to one position per hold window (no pyramiding)."""
    n, hold = 60, 5
    idx = np.arange(n)
    fz = pd.Series(0.0, index=idx)
    fz.iloc[10:20] = 3.0  # 10 consecutive armed bars
    close = pd.Series(100.0, index=idx)
    regime = pd.Series("bull", index=idx)
    fpb = pd.Series(0.0, index=idx)
    r = xs.funding_squeeze_directional(close, fz, regime, fpb,
                                       z_thr=2.0, hold_bars=hold, cost_bps=0.0)
    # 10 armed bars at 10..19, min-gap hold=5 -> entries at 10,15 -> 2 events
    assert r["all"]["n"] == 2


def test_squeeze_directional_long_pays_funding_carry():
    """A long position PAYS positive funding settled over the hold (carry headwind)."""
    n, hold = 30, 4
    idx = np.arange(n)
    fz = pd.Series(0.0, index=idx); fz.iloc[5] = 3.0
    close = pd.Series(100.0, index=idx)  # flat price
    regime = pd.Series("bull", index=idx)
    fpb = pd.Series(0.0, index=idx)
    fpb.iloc[7] = 0.001  # one settlement inside the hold window (bars 6..9)
    r = xs.funding_squeeze_directional(close, fz, regime, fpb,
                                       z_thr=2.0, hold_bars=hold, cost_bps=0.0)
    # flat price, long pays 0.001 funding -> net = -0.001
    assert r["all"]["mean"] == pytest.approx(-0.001, abs=1e-12)


# --------------------------------------------------------------------------- #
# daily_regime_on_grid — causal daily bull/bear mapped to the intraday grid
# (regression for the same-day look-ahead adversarial review found in the driver)
# --------------------------------------------------------------------------- #
def test_daily_regime_on_grid_uses_only_prior_completed_day():
    """A 4h bar on day D must carry the regime knowable at D's OPEN — derived from days
    <= D-1, never D's own end-of-day close. Construct a jump on day 4: the day-4 bars must
    still read the PRE-jump regime; the flip only appears on day 5."""
    step = 4 * 3600 * 1000
    # 6 days x 6 bars/day; close constant within each day so resample.last() picks it.
    daily_vals = [10.0, 10.0, 10.0, 10.0, 100.0, 100.0]  # jump up on day 4
    ts, vals = [], []
    base = 0
    for d, v in enumerate(daily_vals):
        for b in range(6):
            ts.append(base + (d * 6 + b) * step)
            vals.append(v)
    close = pd.Series(vals, index=ts)
    reg = xs.daily_regime_on_grid(close, ma_window=3)
    # day 4 = bars [24..29]; its label must be 'bear' (from day 3, pre-jump), NOT 'bull'
    day4 = reg.iloc[24:30]
    assert (day4 == "bear").all(), f"same-day look-ahead: day-4 bars read {day4.unique()}"
    # the bull flip only shows on day 5 (bars 30..35), using day-4's now-completed close
    assert (reg.iloc[30:36] == "bull").all()


def test_daily_regime_on_grid_is_causal_truncation_invariant():
    rng = np.random.default_rng(2)
    step = 4 * 3600 * 1000
    n_days = 40
    close = pd.Series(
        100 * np.exp(np.cumsum(rng.normal(0, 0.03, n_days * 6))),
        index=np.arange(n_days * 6) * step,
    )
    full = xs.daily_regime_on_grid(close, ma_window=5)
    pre = xs.daily_regime_on_grid(close.iloc[: 30 * 6], ma_window=5)
    # earlier bars' labels are unchanged when future bars are removed
    pd.testing.assert_series_equal(full.iloc[: 30 * 6], pre, check_names=False)
