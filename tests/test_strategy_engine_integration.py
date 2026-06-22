"""Integration: the Strategy Engine v1 slate is wired into the shared library,
the backtest evaluates it symbol-aware + cross-asset, and the report judges it
with the true trial count and a CPCV multi-path Sharpe distribution.

These tests lock the *wiring* (registration, symbol filtering, cross-asset
evaluation, report plumbing) — the individual components have their own units.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from hermes.skills import get_skill_library


# ----------------------------- synthetic data ------------------------------- #
def _ohlcv(close: np.ndarray, ts: np.ndarray | None = None) -> pd.DataFrame:
    n = len(close)
    rng = np.random.default_rng(0)
    high = close * (1 + np.abs(rng.standard_normal(n) * 0.001))
    low = close * (1 - np.abs(rng.standard_normal(n) * 0.001))
    open_ = close * (1 + rng.standard_normal(n) * 0.0005)
    vol = rng.uniform(10, 100, n)
    cols = {"open": open_, "high": high, "low": low, "close": close, "volume": vol}
    if ts is not None:
        cols["ts"] = ts
    return pd.DataFrame(cols)


def _btc_eth_cointegrated(n: int = 420) -> tuple[pd.DataFrame, pd.DataFrame]:
    """ETH = 2*BTC + mean-reverting (stationary) noise → cointegrated pair."""
    rng = np.random.default_rng(7)
    btc = 30_000 + np.cumsum(rng.standard_normal(n) * 50)
    # AR(1) stationary residual so Engle-Granger sees cointegration.
    noise = np.zeros(n)
    for i in range(1, n):
        noise[i] = 0.6 * noise[i - 1] + rng.standard_normal() * 30
    eth = 2.0 * btc + noise
    return _ohlcv(btc), _ohlcv(eth)


# --------------------------------- I1 --------------------------------------- #
def test_active_slate_is_strategy_engine_v1():
    ids = {s.id for s in get_skill_library().list_active()}
    assert {
        "ts_mom_vol_scaled",
        "breakout_vol_expansion",
        "mean_reversion_regime",
        "intraday_tod_btc",
    } <= ids
    # Superseded skills must NOT be in the active slate (double-counting / overfit).
    assert "trend_follow_ema" not in ids
    assert "mean_reversion_atr" not in ids


def test_intraday_tod_is_btc_only_under_symbol_filter():
    lib = get_skill_library()
    df = _ohlcv(np.linspace(100, 110, 80))
    btc_ids = {s.skill_id for s in lib.evaluate_for_symbol("BTC/USDC", df)}
    eth_ids = {s.skill_id for s in lib.evaluate_for_symbol("ETH/USDC", df)}
    assert "intraday_tod_btc" in btc_ids
    assert "intraday_tod_btc" not in eth_ids
    # A non-BTC-restricted skill is present for both.
    assert "ts_mom_vol_scaled" in btc_ids and "ts_mom_vol_scaled" in eth_ids


def test_evaluate_all_is_symbol_agnostic_backcompat():
    # Back-compat: evaluate_all(df) still returns valid signals (used by tests/LLM).
    sigs = get_skill_library().evaluate_all(_ohlcv(np.linspace(100, 120, 90)))
    assert sigs
    for s in sigs:
        assert s.side in {"BUY", "SELL", "HOLD"}
        assert s.stop_pct > 0 and s.take_profit_pct > 0


def test_cross_asset_pairs_signal_present():
    lib = get_skill_library()
    btc, eth = _btc_eth_cointegrated()
    sigs = lib.evaluate_cross_asset({"BTC/USDC": btc, "ETH/USDC": eth})
    ids = {s.skill_id for s in sigs}
    assert "pairs_eth_btc" in ids


def test_cross_asset_missing_leg_holds_gracefully():
    lib = get_skill_library()
    btc, _ = _btc_eth_cointegrated()
    sigs = lib.evaluate_cross_asset({"BTC/USDC": btc})  # ETH leg missing
    by_id = {s.skill_id: s for s in sigs}
    assert by_id["pairs_eth_btc"].side == "HOLD"


# --------------------------------- I2 --------------------------------------- #
def _hourly_frame(days: int = 12) -> pd.DataFrame:
    """Mild steady uptrend on hourly UTC bars → every day's 13:00→14:00 window is
    green, so intraday_tod_btc emits BUY at the 20:00 bar (when not filtered)."""
    n = days * 24
    base = pd.Timestamp("2024-01-01 00:00", tz="UTC")
    ts = pd.to_datetime([base + pd.Timedelta(hours=i) for i in range(n)]).values
    close = 30_000.0 * (1.0005 ** np.arange(n))
    return _ohlcv(close, ts=ts)


def test_backtest_excludes_btc_skill_for_non_btc_symbol():
    from hermes.validation.backtest import run_backtest

    res = run_backtest({"ETH/USDC": _hourly_frame()}, horizon=6, warmup=60)
    assert "intraday_tod_btc" not in res.returns_by_skill


def test_backtest_includes_btc_skill_for_btc_symbol():
    from hermes.validation.backtest import run_backtest

    res = run_backtest({"BTC/USDC": _hourly_frame()}, horizon=6, warmup=60)
    assert "intraday_tod_btc" in res.returns_by_skill


def test_backtest_includes_pairs_for_btc_eth():
    from hermes.validation.backtest import run_backtest

    btc, eth = _btc_eth_cointegrated()
    res = run_backtest({"BTC/USDC": btc, "ETH/USDC": eth}, horizon=12, warmup=60)
    assert "pairs_eth_btc" in res.returns_by_skill
    assert res.returns_by_skill["pairs_eth_btc"]
    assert all(isinstance(v, float) for v in res.returns_by_skill["pairs_eth_btc"])


# --------------------------------- I3 --------------------------------------- #
def test_cpcv_sharpe_distribution_positive_drift():
    from hermes.validation.report import cpcv_sharpe_distribution

    rng = np.random.default_rng(0)
    rets = (0.01 + rng.standard_normal(300) * 0.002).tolist()  # strong + drift
    d = cpcv_sharpe_distribution(rets)
    assert d["n_paths"] == 15  # C(6, 2)
    assert d["frac_positive"] > 0.9
    assert d["sharpe_mean"] > 0.0


def test_cpcv_sharpe_distribution_too_short_returns_zero_paths():
    from hermes.validation.report import cpcv_sharpe_distribution

    d = cpcv_sharpe_distribution([0.01, 0.02])  # fewer than n_groups bars
    assert d["n_paths"] == 0


def _pos_result():
    from hermes.validation.backtest import BacktestResult

    a = [0.02, -0.01, 0.015, -0.012, 0.018, -0.006]
    b = [0.012, -0.009, 0.02, -0.004]
    samples = (
        [{"skill_id": "a", "ret_net": r, "outcome": int(r > 0), "seq": i, "features": {}, "confidence": 0.6} for i, r in enumerate(a)]
        + [{"skill_id": "b", "ret_net": r, "outcome": int(r > 0), "seq": 6 + i, "features": {}, "confidence": 0.6} for i, r in enumerate(b)]
    )
    return BacktestResult(samples=samples, returns_by_skill={"a": a, "b": b})


def test_report_threads_true_trial_count_and_cpcv():
    from hermes.validation.report import backtest_report

    res = _pos_result()
    rep_few = backtest_report(res)  # defaults to n_trials = n_skills (2)
    rep_many = backtest_report(res, n_trials=500)
    assert rep_many["overall"]["n_trials"] == 500
    # More trials can only deflate (never inflate) the Deflated Sharpe.
    assert rep_many["overall"]["dsr"] <= rep_few["overall"]["dsr"]
    assert "cpcv" in rep_many["overall"]
    assert 0.0 <= rep_many["overall"]["cpcv"]["frac_positive"] <= 1.0
