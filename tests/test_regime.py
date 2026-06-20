"""Regime detection (Kaufman efficiency ratio) + regime-conditioned skill weights.

A fast deterministic regime signal lets the agent react to trend<->range shifts
in real time, instead of waiting for the weekly skill re-rank (which lags by
1-4 weeks). Mean-reversion is down-weighted in trends; breakout/trend-follow in ranges.
"""

import pandas as pd

from hermes.data.regime import detect_regime, regime_signal_weight


def _df(closes: list[float]) -> pd.DataFrame:
    return pd.DataFrame({"close": closes})


def test_detect_uptrend():
    out = detect_regime(_df([100 + i for i in range(48)]))
    assert out["regime"] == "trend_up"
    assert out["efficiency_ratio"] == 1.0


def test_detect_downtrend():
    out = detect_regime(_df([148 - i for i in range(48)]))
    assert out["regime"] == "trend_down"
    assert out["efficiency_ratio"] == 1.0


def test_detect_range():
    out = detect_regime(_df([100 + (i % 2) for i in range(48)]))  # zigzag, ~0 net move
    assert out["regime"] == "range"
    assert out["efficiency_ratio"] < 0.4


def test_efficiency_ratio_bounded():
    out = detect_regime(_df([100, 103, 101, 105, 102, 99, 104, 100] * 6))
    assert 0.0 <= out["efficiency_ratio"] <= 1.0


def test_regime_signal_weight_favours_mean_reversion_in_range():
    assert regime_signal_weight("mean_reversion_atr", "range") == 1.0
    assert regime_signal_weight("mean_reversion_atr", "trend_up") == 0.5


def test_regime_signal_weight_favours_trend_skills_in_trend():
    assert regime_signal_weight("trend_follow_ema", "trend_up") == 1.0
    assert regime_signal_weight("trend_follow_ema", "range") == 0.5
    assert regime_signal_weight("breakout_vol_expansion", "trend_down") == 1.0


def test_regime_signal_weight_unknown_skill_is_neutral():
    assert regime_signal_weight("some_new_skill", "range") == 1.0
