"""Stream 9 — cross-book orthogonality, HRP, overlay (SHADOW). TDD, look-ahead first."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hermes.research import s9_cross_book as s9


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #
def _dt(vals, start="2021-01-01", freq="D"):
    vals = np.asarray(vals, dtype=float)
    return pd.Series(vals, index=pd.date_range(start, periods=len(vals), freq=freq))


def _funding_df(n=400, spikes=(250, 290, 330), seed=0):
    """6h OHLCV with a literal ``funding_rate`` column; positive spikes guarantee z>>2 entries."""
    rng = np.random.default_rng(seed)
    close = 100.0 * np.cumprod(1.0 + rng.normal(0, 0.01, n))
    fr = rng.normal(0.0, 0.001, n)
    for s in spikes:
        fr[s] = 0.05  # huge positive funding vs a ~N(0,0.001) trailing window → z >> 2
    ts = (1_600_000_000_000 + np.arange(n) * 21_600_000).astype("int64")  # 6h in ms
    return pd.DataFrame({"ts": ts, "open": close, "high": close * 1.01, "low": close * 0.99,
                         "close": close, "volume": np.ones(n), "funding_rate": fr})


def _crowded_pair(n_weeks=40, seed=11, w=0.4):
    """Orthogonal day-to-day, but a SHARED weekly sign accumulates → high WEEKLY corr."""
    rng = np.random.default_rng(seed)
    tilt = np.repeat(rng.choice([-1.0, 1.0], n_weeks), 7)
    n = n_weeks * 7
    a = _dt(rng.normal(0, 1, n) + w * tilt, start="2021-01-04")  # Monday-aligned weeks
    b = _dt(rng.normal(0, 1, n) + w * tilt, start="2021-01-04")
    return a, b


# --------------------------------------------------------------------------- #
# Task 1 — L0 look-ahead guard (TEST #1, written first)
# --------------------------------------------------------------------------- #
def test_L0_no_lookahead():
    n, horizon = 400, 24
    base = _funding_df(n)
    led_base = s9.run_sleeve(base, s9.funding_squeeze_continuation, horizon=horizon, warmup=200)
    spiked = base.copy()
    spiked.loc[n - 1, ["high", "low", "close", "funding_rate"]] = [1e6, 1e5, 1e6, 5.0]  # corrupt LAST bar
    led_spk = s9.run_sleeve(spiked, s9.funding_squeeze_continuation, horizon=horizon, warmup=200)
    # trades fully RESOLVED before the corrupted bar must be byte-identical (decision + realized return)
    resolved_ts = int(base["ts"].iloc[n - 2 - horizon])
    keep_b = [(t, round(r, 12)) for t, r in led_base if t <= resolved_ts]
    keep_s = [(t, round(r, 12)) for t, r in led_spk if t <= resolved_ts]
    assert keep_b and keep_b == keep_s  # non-empty AND identical


# --------------------------------------------------------------------------- #
# Task 2 — drawdown / Calmar (TEST #2)
# --------------------------------------------------------------------------- #
def test_max_drawdown_and_calmar():
    r = pd.Series([0.1, -0.2, 0.05, -0.1, 0.2])     # equity peak 1.1 → trough 0.8316
    assert s9.max_drawdown(r) == pytest.approx(0.2440, abs=1e-3)
    assert s9.max_drawdown(pd.Series([0.0, 0.0, 0.0])) == 0.0
    assert s9.calmar_ratio(pd.Series([0.0, 0.0, 0.0]), 365) == 0.0      # no DD → 0, not inf
    assert s9.calmar_ratio(pd.Series([0.01] * 100), 365) == 0.0         # monotone up → no DD → 0


# --------------------------------------------------------------------------- #
# Task 3 — regime-level orthogonality (TESTS #3, #4, #5)
# --------------------------------------------------------------------------- #
def test_weekly_view_unmasks_crowding():
    a, b = _crowded_pair()
    pair = s9.pair_ortho(a, b, s9._dt_mask_false(a))
    assert abs(pair.daily) < 0.30 <= abs(pair.weekly)          # the S1×S2 demask shape
    assert pair.rho_kill == max(abs(pair.daily), abs(pair.weekly), abs(pair.risk_off))


def test_risk_off_view_cocrash():
    n, k = 200, 50
    rng = np.random.default_rng(2)
    a = rng.normal(0, 1, n); b = rng.normal(0, 1, n)          # calm: independent
    shock = rng.normal(-2, 2, k)                              # shared shock (real covariance)
    a[-k:] = shock + rng.normal(0, 0.3, k); b[-k:] = shock + rng.normal(0, 0.3, k)
    A, B = _dt(a), _dt(b)
    mask = pd.Series([False] * (n - k) + [True] * k, index=A.index)
    pair = s9.pair_ortho(A, B, mask)
    assert pair.risk_off > 0.5 and pair.risk_off > abs(pair.daily)


def test_demask_flags_hidden_crowding():
    a, b = _crowded_pair()
    ortho = {frozenset({"a", "b"}): s9.pair_ortho(a, b, s9._dt_mask_false(a))}
    flags = s9.demask_report(ortho, bar=0.30)
    assert len(flags) == 1 and flags[0]["unmasked_by"] == "weekly"


# --------------------------------------------------------------------------- #
# Task 4 — HRP (TESTS #6, #7)
# --------------------------------------------------------------------------- #
def test_hrp_allocates_only_confirmed():
    rng = np.random.default_rng(4)
    R = {k: _dt(rng.normal(0, 1, 120)) for k in ("a", "b", "c")}
    w = s9.hrp_allocate(R, confirmed_ids={"a", "b"}, risk_off_mask=s9._dt_mask_false(R["a"]))
    assert w["c"] == 0.0 and abs(w["a"] + w["b"] - 1.0) < 1e-9


def test_hrp_downweights_colinear_block_and_psd_sizing():
    rng = np.random.default_rng(5); n = 300
    f = rng.normal(0, 1, n)
    a = _dt(f + 0.05 * rng.normal(0, 1, n))   # a,b nearly colinear
    b = _dt(f + 0.05 * rng.normal(0, 1, n))
    c = _dt(rng.normal(0, 1, n))              # c independent diversifier
    R = {"a": a, "b": b, "c": c}
    w = s9.hrp_allocate(R, confirmed_ids={"a", "b", "c"}, risk_off_mask=s9._dt_mask_false(a))
    assert all(v >= 0 for v in w.values()) and abs(sum(w.values()) - 1) < 1e-9
    assert w["c"] > w["a"] and w["c"] > w["b"]   # lone diversifier > each colinear leg


# --------------------------------------------------------------------------- #
# Task 5 — overlay (TESTS #8, #9, #10, #11)
# --------------------------------------------------------------------------- #
def test_overlay_acts_next_period_and_rearm():
    cfg = s9.OverlayConfig(vol_z_max=1.0, depeg_bps=0, gap_pct=0, cross_dd_window=0,
                           cross_dd_threshold=0, degross_fraction=0.5, rearm_clear_periods=2,
                           sharpe_cost_tol=0.1)
    idx = pd.date_range("2021-01-01", periods=6, freq="D")
    trig = pd.Series([False, True, False, False, False, False], index=idx)
    mult = s9.exposure_multiplier([trig], cfg)
    assert list(mult) == [1.0, 0.5, 0.5, 0.5, 1.0, 1.0]      # stress holds, rearms after 2 clear
    book = pd.Series([0.1] * 6, index=idx)
    overlaid = s9.apply_overlay(book, mult)
    assert overlaid.iloc[1] == pytest.approx(0.1)            # multiplier at t acts on t+1 (shift guard)
    assert overlaid.iloc[2] == pytest.approx(0.05)


def test_gap_trigger_uses_next_open():
    idx = pd.date_range("2021-01-01", periods=4, freq="D")
    ohlc = pd.DataFrame({"open": [100, 101, 130, 131], "close": [100, 100, 129, 130]}, index=idx)
    cfg = s9.OverlayConfig(vol_z_max=99, depeg_bps=0, gap_pct=0.10, cross_dd_window=0,
                           cross_dd_threshold=0, degross_fraction=0.5, rearm_clear_periods=1, sharpe_cost_tol=0.1)
    g = s9.gap_trigger(ohlc, cfg)
    assert bool(g.iloc[2]) is True and bool(g.iloc[0]) is False   # 100→130 open gap > 10%


def test_overlay_cost_drops_good_day_trigger():
    rng = np.random.default_rng(7); idx = pd.date_range("2021-01-01", periods=200, freq="D")
    book = pd.Series(rng.uniform(0.0005, 0.01, 200), index=idx)   # all-positive → never draws down
    good = book > book.quantile(0.8)                             # 'trigger' that fires with no crash to protect
    rep = s9.overlay_cost_report(book, {"bad": good},
                                 s9.OverlayConfig(0, 0, 0, 0, 0, 0.5, 1, 0.0), periods_per_year=365)
    assert rep["per_trigger"]["bad"]["verdict"] == "OUT"         # no Calmar gain (book never drew down) → reject
    assert "joint" in rep and rep["periods_per_year"] == 365


# --------------------------------------------------------------------------- #
# Task 6 — global LOO double condition (TEST #12)
# --------------------------------------------------------------------------- #
def test_global_loo_double_condition():
    rng = np.random.default_rng(8); n = 200
    base = rng.normal(0.001, 0.01, n)
    crypto = {"x": _dt(base), "y": _dt(base + rng.normal(0, 0.002, n))}     # x,y crowd natively
    fx = {"z": _dt(rng.normal(0.0005, 0.008, n))}
    res = s9.cross_book_diversifiers(crypto, fx)
    assert set(res) == {"x", "y", "z"}
    for v in res.values():
        assert v["is_diversifier"] == (v["global_loo"] > 0 and v["global_loo"] > v["native_loo"])


# --------------------------------------------------------------------------- #
# Task 7 — read-only dashboard emitter
# --------------------------------------------------------------------------- #
def test_pulse_emitter_is_shadow_readonly():
    km = pd.DataFrame([[1.0, 0.4], [0.4, 1.0]], index=["a", "b"], columns=["a", "b"])
    out = s9.s9_pulse_emitter(km, {"a": 0.6, "b": 0.4}, {"stressed": False})
    assert out["shadow"] is True and out["kind"] == "s9_cross_book"
    assert out["hrp"] == {"a": 0.6, "b": 0.4}
