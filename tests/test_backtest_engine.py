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
