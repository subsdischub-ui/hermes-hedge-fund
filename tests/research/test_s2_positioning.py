"""Tests for Stream 2 positioning research primitives (s2_positioning).

The lab's recurring failure mode is look-ahead. These tests pin the causality
guarantee FIRST: every signal value at time t must depend only on data at or
before t. The gating math itself (Sharpe/DSR/PBO/joint-null) is already tested
in tests/test_book.py and research_fx/test_fxgate.py — we reuse, not re-verify, it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hermes.research import s2_positioning as s2


# --------------------------------------------------------------------------- #
# causal_zscore — the no-look-ahead guarantee
# --------------------------------------------------------------------------- #
def test_causal_zscore_is_causal_future_cannot_change_past():
    """z at index t must be identical whether or not future bars exist.

    This is THE look-ahead guard: truncating the series after index t leaves
    every z value at indices <= t unchanged.
    """
    rng = np.random.default_rng(0)
    x = pd.Series(rng.normal(size=200).cumsum())
    window = 30

    full = s2.causal_zscore(x, window)
    prefix = s2.causal_zscore(x.iloc[:120], window)

    # On the overlapping prefix, the two must agree bar-for-bar (NaNs included).
    pd.testing.assert_series_equal(
        full.iloc[:120], prefix, check_names=False
    )


def test_causal_zscore_matches_trailing_window_formula():
    """z_t = (x_t - mean(trailing w incl. t)) / std(trailing w, ddof=1)."""
    x = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    window = 3
    z = s2.causal_zscore(x, window)

    # t=4 (value 5.0): window = [3,4,5], mean 4, std ddof1 = 1 -> z = (5-4)/1 = 1
    assert z.iloc[4] == pytest.approx(1.0)
    # warmup: first window-1 entries undefined
    assert np.isnan(z.iloc[0]) and np.isnan(z.iloc[1])
    assert not np.isnan(z.iloc[2])


def test_causal_zscore_zero_variance_window_is_nan_not_inf():
    """A flat trailing window (std=0) yields NaN, never inf/divide-by-zero."""
    x = pd.Series([7.0, 7.0, 7.0, 7.0, 7.0])
    z = s2.causal_zscore(x, 3)
    assert np.isnan(z.iloc[3])


# --------------------------------------------------------------------------- #
# forward_return — the prediction target (no future-bar leakage)
# --------------------------------------------------------------------------- #
def test_forward_return_simple_and_tail_is_nan():
    """fwd_t = close[t+h]/close[t] - 1, last h bars NaN (no future bar exists)."""
    close = pd.Series([100.0, 110.0, 121.0, 133.1])
    fwd = s2.forward_return(close, horizon=1)
    assert fwd.iloc[0] == pytest.approx(0.10)
    assert fwd.iloc[1] == pytest.approx(0.10)
    assert fwd.iloc[2] == pytest.approx(0.10)
    assert np.isnan(fwd.iloc[3])  # no t+1 bar


def test_forward_return_multibar_horizon():
    close = pd.Series([100.0, 110.0, 121.0, 133.1, 146.41])
    fwd = s2.forward_return(close, horizon=2)
    assert fwd.iloc[0] == pytest.approx(0.21)  # 121/100 - 1
    assert np.isnan(fwd.iloc[3]) and np.isnan(fwd.iloc[4])


# --------------------------------------------------------------------------- #
# backtest_positions — causal position -> net PnL, cost on turnover
# --------------------------------------------------------------------------- #
def test_backtest_positions_charges_cost_on_position_change_only():
    """net_t = pos_t * fwd_t - cost * |pos_t - pos_{t-1}|.

    A held (unchanged) position pays no new cost; only rebalancing does.
    """
    pos = pd.Series([1.0, 1.0, -1.0])
    fwd = pd.Series([0.02, 0.02, 0.02])
    net = s2.backtest_positions(pos, fwd, cost_per_turnover=0.0015)

    # t0: enter long from flat -> turnover 1 -> 0.02 - 0.0015
    assert net.iloc[0] == pytest.approx(0.02 - 0.0015)
    # t1: hold long -> no turnover cost
    assert net.iloc[1] == pytest.approx(0.02)
    # t2: flip long->short -> turnover 2 -> -0.02 - 2*0.0015
    assert net.iloc[2] == pytest.approx(-0.02 - 2 * 0.0015)


def test_backtest_positions_drops_nan_target_bars():
    """Bars whose forward target is NaN (the tail) contribute nothing, not NaN."""
    pos = pd.Series([1.0, 1.0])
    fwd = pd.Series([0.01, np.nan])
    net = s2.backtest_positions(pos, fwd, cost_per_turnover=0.0)
    assert net.iloc[0] == pytest.approx(0.01)
    assert len(net) == 1  # the NaN-target bar is dropped from the realised series


# --------------------------------------------------------------------------- #
# tag_regime — causal bull/bear/range labelling for the cross-regime KILL test
# --------------------------------------------------------------------------- #
def test_tag_regime_labels_trend_by_trailing_return():
    up = pd.Series(np.linspace(100, 300, 200))     # strong uptrend
    down = pd.Series(np.linspace(300, 100, 200))   # strong downtrend
    flat = pd.Series(100 + np.zeros(200))          # flat

    assert s2.tag_regime(up, window=90).iloc[-1] == "bull"
    assert s2.tag_regime(down, window=90).iloc[-1] == "bear"
    assert s2.tag_regime(flat, window=90).iloc[-1] == "range"


def test_tag_regime_warmup_is_unknown_not_a_lookahead():
    up = pd.Series(np.linspace(100, 300, 200))
    reg = s2.tag_regime(up, window=90)
    assert reg.iloc[0] == "unknown"   # no trailing window yet
    assert reg.iloc[89] == "unknown"  # still one short of a full window
    assert reg.iloc[90] != "unknown"


# --------------------------------------------------------------------------- #
# top_lsr_contrarian_signal — vector 2.1 core (CONTRARIAN: fade the crowd)
# --------------------------------------------------------------------------- #
def test_contrarian_sign_mode_fades_the_crowd():
    """High top-LSR (crowded long, z>0) -> SHORT (-1); crowded short (z<0) -> LONG (+1).

    The contrarian thesis: extreme top-trader long-skew predicts a NEGATIVE forward
    return, so the position is the opposite sign of the positioning z-score.
    """
    # A ramp then a flat tail: z is +ve while rising, then decays.
    lsr = pd.Series([1.0, 1.1, 1.3, 1.7, 2.5, 2.6, 1.0, 0.4, 0.3])
    sig = s2.top_lsr_contrarian_signal(lsr, window=4, mode="sign")
    # find a bar where z>0 (crowded long) -> signal must be -1
    z = s2.causal_zscore(lsr, 4)
    hot = z.index[z > 0.5][0]
    assert sig.loc[hot] == -1.0
    cold = z.index[z < -0.5][0]
    assert sig.loc[cold] == 1.0
    # warmup -> flat (0), never NaN
    assert sig.iloc[0] == 0.0


def test_contrarian_continuous_mode_is_bounded_and_saturates():
    lsr = pd.Series(np.r_[np.zeros(10), [10.0]])  # a spike at the end
    # A spike inflates its OWN trailing std, so its z is self-limited (~1.8 here),
    # not arbitrarily large — a real positioning-microstructure property.
    sig2 = s2.top_lsr_contrarian_signal(lsr, window=5, mode="continuous", clip=2.0)
    assert -1.0 <= sig2.iloc[-1] < 0.0            # contrarian (short the crowded long)
    assert sig2.dropna().abs().max() <= 1.0 + 1e-12  # always bounded
    assert sig2.iloc[0] == 0.0                    # warmup flat
    # With clip below the achieved |z|, the position saturates to the bound.
    sig1 = s2.top_lsr_contrarian_signal(lsr, window=5, mode="continuous", clip=1.0)
    assert sig1.iloc[-1] == pytest.approx(-1.0)


# --------------------------------------------------------------------------- #
# liquidation_flush_events — vector 2.2 (mean-revert the forced flow)
# --------------------------------------------------------------------------- #
def test_liquidation_flush_detects_zspike_and_sides_to_mean_revert():
    """A z-spike in TOTAL liquidations is a flush. Side mean-reverts the forced flow:
    long-liquidation cascade (down) -> go LONG (+1); short squeeze (up) -> SHORT (-1).
    """
    n = 40
    long_liq = pd.Series([1.0] * n)
    short_liq = pd.Series([1.0] * n)
    long_liq.iloc[30] = 200.0   # a long-dominated flush (forced selling)
    short_liq.iloc[35] = 200.0  # a short-dominated squeeze (forced buying)

    ev = s2.liquidation_flush_events(long_liq, short_liq, window=20, k=2.0)
    ev = dict(ev)  # {bar_index: side}
    assert ev.get(30) == 1.0    # longs flushed -> buy the rebound
    assert ev.get(35) == -1.0   # shorts squeezed -> fade the spike
    # quiet bars are not events
    assert 10 not in ev and 20 not in ev


def test_liquidation_flush_no_events_when_no_spike():
    flat = pd.Series([1.0] * 40)
    ev = s2.liquidation_flush_events(flat, flat, window=20, k=3.0)
    assert ev == []


def test_flush_side_array_sides_from_flush_bar_not_entry_bar():
    """REGRESSION (look-ahead): the trade side must come from the FLUSH bar's dominance,
    consistently lagged with the z-trigger — NEVER from the entry bar's dominance (which
    leaks the entry bar's own price move). This pins the composition the driver glue got
    wrong: z lagged but dom taken un-lagged at the entry bar.
    """
    n, window, k, lag = 40, 20, 2.0, 1
    ll = np.ones(n)
    sl = np.ones(n)
    ll[30] = 200.0   # long-dominant FLUSH at bar 30 -> contrarian LONG (+1)
    sl[31] = 50.0     # entry bar 31 is short-dominant: the trap (must NOT decide the side)
    side = s2.liquidation_flush_side_array(pd.Series(ll), pd.Series(sl), window=window, k=k, lag=lag)
    assert side[31] == 1.0   # from flush bar 30 (long flush), NOT entry bar 31 (short)
    assert side[30] == 0.0   # no entry on the flush bar itself when lag=1
    assert (side[:30] == 0.0).all()


def test_flush_side_array_lag_zero_enters_on_flush_bar():
    n, window, k = 40, 20, 2.0
    ll = np.ones(n); sl = np.ones(n)
    sl[30] = 200.0   # short-dominant squeeze -> contrarian SHORT (-1)
    side = s2.liquidation_flush_side_array(pd.Series(ll), pd.Series(sl), window=window, k=k, lag=0)
    assert side[30] == -1.0


# --------------------------------------------------------------------------- #
# oi_acceleration_zscore — vector 2.3 (single-venue; cross-venue is data-gated)
# --------------------------------------------------------------------------- #
def test_oi_acceleration_zscore_flags_accelerating_build_causally():
    """Acceleration = causal z of OI growth. A burst of OI growth -> high +z; causal."""
    oi = pd.Series(np.r_[np.full(30, 100.0), [110.0, 122.0, 137.0]])  # accelerating tail
    accel = s2.oi_acceleration_zscore(oi, window=20)
    assert accel.iloc[-1] > 1.0           # the build is accelerating
    # causal: truncating the future leaves earlier values unchanged
    pre = s2.oi_acceleration_zscore(oi.iloc[:32], window=20)
    pd.testing.assert_series_equal(accel.iloc[:32], pre, check_names=False)


# --------------------------------------------------------------------------- #
# basket_contrarian_gate — the de-overlapped joint-null in BAR space
# (mirrors fxgate.event_basket_gate's statistical construction, crypto-cadence-correct)
# --------------------------------------------------------------------------- #
def _planted_edge():
    """3 symbols, flat 100, with a contrarian-tradeable -3% dip 5 bars after each
    crowding spike. z is already causal+lagged (the gate does not re-lag)."""
    T, H = 240, 5
    z = {s: np.zeros(T) for s in ("A", "B", "C")}
    close = {s: np.full(T, 100.0) for s in ("A", "B", "C")}
    for t in (40, 80, 120, 160, 200):
        for s in z:
            z[s][t] = 3.0              # crowded long -> contrarian short
            close[s][t + H] = 97.0     # price falls 3% over the next H bars
    return z, close, H


def test_basket_gate_planted_edge_beats_joint_null():
    z, close, H = _planted_edge()
    r = s2.basket_contrarian_gate(z, close, thr=2.0, horizon=H, cost=0.0,
                                  leg="short", n_sims=2000, seed=0)
    assert r["n_events"] == 5
    assert r["sleeve_mean"] == pytest.approx(0.03, abs=1e-9)  # +3% per contrarian short
    assert r["mc_p"] < 0.05                                   # the joint null rarely beats it
    assert r["event_bars"] == [40, 80, 120, 160, 200]         # entry bars, for regime bucketing


def test_basket_gate_deoverlaps_persistent_signal():
    """A run of consecutive extreme bars becomes ONE held position, not many — kept
    entries are >= horizon apart (else the null variance collapses, anti-conservative)."""
    T, H = 100, 6
    z = {s: np.zeros(T) for s in ("A", "B")}
    close = {s: np.full(T, 100.0) for s in ("A", "B")}
    for s in z:
        z[s][20:40] = 3.0  # 20 consecutive crowded-long bars
    r = s2.basket_contrarian_gate(z, close, thr=2.0, horizon=H, cost=0.0,
                                  leg="short", n_sims=200, seed=0)
    # 20 consecutive bars, min-gap = horizon 6 -> entries at 20,26,32,38 => <= 4 events
    assert r["n_events"] <= 4


def test_basket_gate_no_events_when_never_extreme():
    T = 50
    z = {s: np.zeros(T) for s in ("A", "B")}
    close = {s: np.full(T, 100.0) for s in ("A", "B")}
    r = s2.basket_contrarian_gate(z, close, thr=2.0, horizon=4, cost=0.0, leg="short")
    assert r["n_events"] == 0 and r["mc_p"] == 1.0


def test_basket_event_gate_honours_explicit_side_and_joint_null():
    """vector 2.2 path: a per-bar side array (mean-revert) gated by the SAME joint null.
    Plant a +1 (long) event 5 bars before a +3% rally -> profitable, null rarely beats it."""
    T, H = 240, 5
    side = np.zeros(T)
    close = {s: np.full(T, 100.0) for s in ("A", "B", "C")}
    for t in (40, 80, 120, 160, 200):
        side[t] = 1.0               # go long
        for s in close:
            close[s][t + H] = 103.0  # +3% rally over the next H bars
    r = s2.basket_event_gate(side, close, horizon=H, cost=0.0, n_sims=2000, seed=0)
    assert r["n_events"] == 5
    assert r["sleeve_mean"] == pytest.approx(0.03, abs=1e-9)
    assert r["mc_p"] < 0.05
