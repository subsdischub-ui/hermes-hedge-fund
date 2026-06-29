"""TDD for the Stream-5 FX/metals relative-value construction core.

The whole point of S5 is sleeves that are MARKET-NEUTRAL *by construction* — the
secular-bull-beta that killed the prior arc's directional gold sleeves (B2/B3/X1)
is removed by hedging one leg against the other. So the load-bearing properties to
pin down are: (1) the hedge ratio is CAUSAL (no look-ahead — a future price never
changes a past hedge), and (2) the resulting spread really is ~zero-beta to the
common metal factor. Everything is tested on synthetic data; no data-lake file is
required to run the suite.
"""

from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd
import pytest

from hermes.research import s5_fx_metals as s5


# --------------------------------------------------------------------------- #
# log_returns
# --------------------------------------------------------------------------- #
def test_log_returns_match_manual():
    close = pd.Series([100.0, 110.0, 121.0])
    r = s5.log_returns(close)
    assert np.isnan(r.iloc[0])
    assert r.iloc[1] == pytest.approx(np.log(110 / 100))
    assert r.iloc[2] == pytest.approx(np.log(121 / 110))


# --------------------------------------------------------------------------- #
# causal_rolling_beta — the causality guarantee
# --------------------------------------------------------------------------- #
def test_causal_rolling_beta_recovers_true_beta():
    rng = np.random.default_rng(0)
    r_b = pd.Series(rng.normal(0, 0.01, 400))
    r_a = 2.0 * r_b  # exact beta = 2
    beta = s5.causal_rolling_beta(r_a, r_b, lookback=60)
    assert beta.dropna().to_numpy() == pytest.approx(2.0, abs=1e-9)


def test_causal_rolling_beta_has_no_lookahead():
    """beta_t must depend ONLY on returns strictly before t: truncating the series
    at t leaves beta_t unchanged, and a perturbed FUTURE value cannot move it."""
    rng = np.random.default_rng(1)
    r_b = pd.Series(rng.normal(0, 0.01, 200))
    r_a = pd.Series(rng.normal(0, 0.01, 200))
    lookback = 30
    beta_full = s5.causal_rolling_beta(r_a, r_b, lookback)
    t = 120
    beta_trunc = s5.causal_rolling_beta(r_a.iloc[: t + 1], r_b.iloc[: t + 1], lookback)
    assert beta_trunc.iloc[t] == pytest.approx(beta_full.iloc[t], nan_ok=True)
    # perturb a strictly-future return; beta_t is unchanged
    r_a2 = r_a.copy()
    r_a2.iloc[t + 5] += 10.0
    beta_pert = s5.causal_rolling_beta(r_a2, r_b, lookback)
    assert beta_pert.iloc[t] == pytest.approx(beta_full.iloc[t], nan_ok=True)


def test_causal_rolling_beta_warmup_is_nan():
    rng = np.random.default_rng(2)
    r_b = pd.Series(rng.normal(0, 0.01, 100))
    r_a = pd.Series(rng.normal(0, 0.01, 100))
    beta = s5.causal_rolling_beta(r_a, r_b, lookback=40)
    # nothing usable before a full lookback window has elapsed AND been shifted
    assert beta.iloc[:40].isna().all()
    assert beta.iloc[41:].notna().all()


# --------------------------------------------------------------------------- #
# neutral_spread_returns — neutrality by construction
# --------------------------------------------------------------------------- #
def test_neutral_spread_is_beta_neutral_to_the_hedge_leg():
    rng = np.random.default_rng(3)
    r_b = pd.Series(rng.normal(0, 0.01, 1500))
    noise = pd.Series(rng.normal(0, 0.01, 1500))
    r_a = 1.5 * r_b + noise  # a loads on b with beta 1.5 plus idiosyncratic noise
    spread = s5.neutral_spread_returns(r_a, r_b, lookback=120)
    neu = s5.factor_neutrality(spread, r_b)
    assert abs(neu["beta"]) < 0.1
    assert abs(neu["rho"]) < 0.15


# --------------------------------------------------------------------------- #
# causal_multi_beta / neutral_spread_returns_multi — 2-factor (both metals) hedge
# --------------------------------------------------------------------------- #
def test_causal_multi_beta_recovers_true_betas():
    rng = np.random.default_rng(10)
    f1 = pd.Series(rng.normal(0, 0.01, 800))
    f2 = pd.Series(rng.normal(0, 0.01, 800))
    factors = pd.concat([f1, f2], axis=1, keys=["f1", "f2"])
    r_a = 2.0 * f1 + 3.0 * f2  # exact betas [2, 3]
    betas = s5.causal_multi_beta(r_a, factors, lookback=80)
    last = betas.dropna().iloc[-1]
    assert last["f1"] == pytest.approx(2.0, abs=1e-6)
    assert last["f2"] == pytest.approx(3.0, abs=1e-6)


def test_causal_multi_beta_has_no_lookahead():
    rng = np.random.default_rng(11)
    factors = pd.DataFrame({"f1": rng.normal(0, 0.01, 200), "f2": rng.normal(0, 0.01, 200)})
    r_a = pd.Series(rng.normal(0, 0.01, 200))
    lb = 50
    full = s5.causal_multi_beta(r_a, factors, lb)
    t = 130
    trunc = s5.causal_multi_beta(r_a.iloc[: t + 1], factors.iloc[: t + 1], lb)
    assert trunc["f1"].iloc[t] == pytest.approx(full["f1"].iloc[t], nan_ok=True)
    r_a2 = r_a.copy(); r_a2.iloc[t + 4] += 99.0
    assert s5.causal_multi_beta(r_a2, factors, lb)["f1"].iloc[t] == pytest.approx(
        full["f1"].iloc[t], nan_ok=True
    )


def test_neutral_spread_multi_is_neutral_to_every_factor():
    rng = np.random.default_rng(12)
    f1 = pd.Series(rng.normal(0, 0.012, 1500))
    f2 = pd.Series(rng.normal(0, 0.020, 1500))
    noise = pd.Series(rng.normal(0, 0.01, 1500))
    r_a = 1.2 * f1 + 0.8 * f2 + noise
    factors = pd.concat([f1, f2], axis=1, keys=["gold", "silver"])
    spread = s5.neutral_spread_returns_multi(r_a, factors, lookback=150)
    assert abs(s5.factor_neutrality(spread, f1)["beta"]) < 0.12
    assert abs(s5.factor_neutrality(spread, f2)["beta"]) < 0.12


# --------------------------------------------------------------------------- #
# synthetic_nav
# --------------------------------------------------------------------------- #
def test_synthetic_nav_compounds_and_treats_leading_nan_as_flat():
    r = pd.Series([np.nan, 0.1, -0.1])
    nav = s5.synthetic_nav(r)
    assert nav.to_numpy() == pytest.approx([1.0, 1.1, 0.99])


# --------------------------------------------------------------------------- #
# causal_zscore
# --------------------------------------------------------------------------- #
def test_causal_zscore_no_lookahead():
    rng = np.random.default_rng(4)
    x = pd.Series(rng.normal(0, 1, 200))
    z_full = s5.causal_zscore(x, lookback=30)
    t = 100
    z_trunc = s5.causal_zscore(x.iloc[: t + 1], lookback=30)
    assert z_trunc.iloc[t] == pytest.approx(z_full.iloc[t], nan_ok=True)
    x2 = x.copy()
    x2.iloc[t + 3] += 50.0
    assert s5.causal_zscore(x2, 30).iloc[t] == pytest.approx(z_full.iloc[t], nan_ok=True)


# --------------------------------------------------------------------------- #
# threshold_events
# --------------------------------------------------------------------------- #
def test_threshold_events_trend_and_revert():
    z = pd.Series([np.nan, 0.0, 1.5, 2.0, -0.5, -2.0])
    trend = s5.threshold_events(z, entry=1.5, mode="trend")
    assert trend == [(2, 1), (3, 1), (5, -1)]
    revert = s5.threshold_events(z, entry=1.5, mode="revert")
    assert revert == [(2, -1), (3, -1), (5, 1)]


def test_threshold_events_rejects_bad_mode():
    with pytest.raises(ValueError):
        s5.threshold_events(pd.Series([2.0]), entry=1.0, mode="sideways")


# --------------------------------------------------------------------------- #
# factor_neutrality
# --------------------------------------------------------------------------- #
def test_factor_neutrality_extremes():
    rng = np.random.default_rng(5)
    f = pd.Series(rng.normal(0, 0.01, 500))
    same = s5.factor_neutrality(f, f)
    assert same["beta"] == pytest.approx(1.0, abs=1e-9)
    assert same["rho"] == pytest.approx(1.0, abs=1e-9)
    indep = s5.factor_neutrality(pd.Series(rng.normal(0, 0.01, 500)), f)
    assert abs(indep["beta"]) < 0.2
    assert abs(indep["rho"]) < 0.2


# --------------------------------------------------------------------------- #
# FX quoting convention — pair_usd_value / pair_usd_return (dollar-neutrality hinges on the sign)
# --------------------------------------------------------------------------- #
def test_pair_usd_value_direct_and_inverse():
    close = pd.Series([0.75, 0.80, 0.70])  # e.g. AUDUSD (direct) or USDCHF (inverse)
    direct = s5.pair_usd_value(close, "XXXUSD")
    assert direct.to_numpy() == pytest.approx(close.to_numpy())  # USD value of XXX is the price
    inverse = s5.pair_usd_value(close, "USDXXX")
    assert inverse.to_numpy() == pytest.approx(1.0 / close.to_numpy())  # USD value of XXX is 1/price


def test_pair_usd_return_sign():
    close = pd.Series([100.0, 110.0, 99.0])
    direct = s5.pair_usd_return(close, "XXXUSD")
    inverse = s5.pair_usd_return(close, "USDXXX")
    # the foreign currency's USD return is the NEGATIVE of a USDXXX pair's return
    assert inverse.dropna().to_numpy() == pytest.approx(-direct.dropna().to_numpy())
    assert direct.iloc[1] == pytest.approx(np.log(110 / 100))


def test_pair_usd_value_rejects_bad_quote():
    with pytest.raises(ValueError):
        s5.pair_usd_value(pd.Series([1.0]), "EURGBP")


# --------------------------------------------------------------------------- #
# load_lake_close — thin DuckDB reader (dependency-injected connection)
# --------------------------------------------------------------------------- #
def test_load_lake_close_pivots_and_inner_joins_on_ts():
    con = duckdb.connect(":memory:")
    con.execute(
        "CREATE TABLE prices (ts BIGINT, close DOUBLE, symbol VARCHAR, interval VARCHAR)"
    )
    rows = [
        (1, 10.0, "GC=F", "1d"), (2, 11.0, "GC=F", "1d"), (3, 12.0, "GC=F", "1d"),
        (1, 5.0, "SI=F", "1d"), (2, 5.5, "SI=F", "1d"),  # no ts=3 for silver
        (1, 99.0, "GC=F", "1h"),  # wrong interval must be excluded
    ]
    con.executemany("INSERT INTO prices VALUES (?,?,?,?)", rows)
    frame = s5.load_lake_close(con, ["GC=F", "SI=F"], interval="1d")
    assert list(frame.columns) == ["GC=F", "SI=F"]
    assert frame.index.tolist() == [1, 2]  # inner join drops ts=3
    assert frame.loc[1, "GC=F"] == 10.0
    assert frame.loc[2, "SI=F"] == 5.5
