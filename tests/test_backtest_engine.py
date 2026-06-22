"""Backtest engine: replay the real skills over historical OHLCV, label each
signal net of fees with the triple-barrier method -> training samples + returns."""

import pandas as pd

from hermes.validation.backtest import run_backtest


def _drop_df(flat: int = 60, decline: int = 30) -> pd.DataFrame:
    # Flat plateau then a sharp sustained drop. Used only for the "too short"
    # case below (history < warmup -> zero trades regardless of slate).
    closes = [100.0] * flat + [100.0 - 2.0 * k for k in range(1, decline + 1)]
    return pd.DataFrame(
        {
            "open": [c for c in closes],
            "high": [c + 1.0 for c in closes],
            "low": [c - 1.0 for c in closes],
            "close": closes,
            "volume": [1000.0] * len(closes),
        }
    )


def _trend_df(n: int = 160) -> pd.DataFrame:
    # Sustained uptrend (>96 bars so the multi-lookback ts_mom_vol_scaled skill
    # has a full panel) -> vol-scaled momentum fires BUY deterministically.
    closes = [100.0 * (1.01 ** k) for k in range(n)]
    return pd.DataFrame(
        {
            "open": closes,
            "high": [c + 1.0 for c in closes],
            "low": [c - 1.0 for c in closes],
            "close": closes,
            "volume": [1000.0] * n,
        }
    )


def test_run_backtest_produces_labeled_samples():
    res = run_backtest(
        {"BTC/USDC": _trend_df()}, fee_bps=7, slippage_bps=3, horizon=12, warmup=60
    )
    assert res.n_trades >= 1
    s = res.samples[0]
    assert {"features", "confidence", "outcome", "ret_net", "skill_id"} <= set(s)
    assert s["outcome"] in (0, 1)
    assert "confidence" in s["features"]
    assert all(isinstance(v, list) for v in res.returns_by_skill.values())


def test_run_backtest_empty_for_too_short_history():
    res = run_backtest({"BTC/USDC": _drop_df(flat=20, decline=10)}, warmup=60)
    assert res.n_trades == 0


def test_regime_gate_is_subset_and_keeps_trend_skill_in_trend():
    # With regime gating, a per-symbol skill's signal is only recorded when that
    # skill is in the active set for the bar's routed regime. Gating can only
    # REMOVE off-regime trades, never add. A clean uptrend routes to "trend", so
    # the trend skill ts_mom_vol_scaled is kept (not gated out).
    data = {"BTC/USDC": _trend_df(420)}
    ungated = run_backtest(data, horizon=12, warmup=60, step=1, window_cap=350)
    gated = run_backtest(
        data, horizon=12, warmup=60, step=1, window_cap=350, regime_gate=True
    )
    assert gated.n_trades <= ungated.n_trades
    assert len(gated.returns_by_skill.get("ts_mom_vol_scaled", [])) == len(
        ungated.returns_by_skill.get("ts_mom_vol_scaled", [])
    )


def test_skills_allowlist_restricts_active_skills():
    res = run_backtest(
        {"BTC/USDC": _trend_df(420)}, horizon=12, warmup=60, window_cap=350,
        skills_allowlist={"ts_mom_vol_scaled"},
    )
    assert set(res.returns_by_skill) <= {"ts_mom_vol_scaled"}


def test_conviction_min_keeps_only_high_confidence_trades():
    data = {"BTC/USDC": _trend_df(420)}
    base = run_backtest(
        data, horizon=12, warmup=60, window_cap=350, skills_allowlist={"ts_mom_vol_scaled"}
    )
    sel = run_backtest(
        data, horizon=12, warmup=60, window_cap=350,
        skills_allowlist={"ts_mom_vol_scaled"}, conviction_min=0.95,
    )
    assert sel.n_trades <= base.n_trades
    assert all(s["confidence"] >= 0.95 for s in sel.samples)


def test_no_overlap_consolidates_into_one_position_per_holding_window():
    # The user's insight: collapse the per-bar re-entry into one position per move.
    # With no_overlap, a skill cannot re-enter (same symbol) until `horizon` bars
    # after its last entry — so consecutive entries are spaced >= horizon.
    res = run_backtest(
        {"BTC/USDC": _trend_df(420)}, horizon=12, warmup=60, window_cap=350,
        skills_allowlist={"ts_mom_vol_scaled"}, no_overlap=True,
    )
    entries = sorted(s["t"] for s in res.samples if s["skill_id"] == "ts_mom_vol_scaled")
    gaps = [b - a for a, b in zip(entries, entries[1:])]
    assert entries  # it still trades
    assert all(g >= 12 for g in gaps)


def test_window_cap_matches_full_history_when_cap_exceeds_lookback():
    # Skills are causal and read only a trailing window (<=200 bars: MA200 etc.).
    # Capping the per-bar slice to >= that lookback must yield identical trades —
    # this is the O(n^2)->O(n) optimisation that makes multi-year backtests feasible.
    data = {"BTC/USDC": _trend_df(420)}
    full = run_backtest(data, horizon=12, warmup=60, step=1)
    capped = run_backtest(data, horizon=12, warmup=60, step=1, window_cap=350)
    assert capped.returns_by_skill == full.returns_by_skill
    assert capped.n_trades == full.n_trades
