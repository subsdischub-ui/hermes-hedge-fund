"""Unit tests for the TS-MOM vol-scaled multi-lookback skill.

Each test builds a deterministic synthetic OHLCV series that should trigger
exactly one branch (BUY / SELL / HOLD) and asserts the structured signal,
mirroring the style of the existing skill tests.
"""

import numpy as np
import pandas as pd

from hermes.skills.ts_mom import ts_mom_vol_scaled


def _ohlcv_from_close(close: np.ndarray, jitter: float = 0.001) -> pd.DataFrame:
    """Wrap a close path in a plausible OHLCV frame (deterministic, tiny wicks)."""
    rng = np.random.default_rng(0)
    n = len(close)
    high = close * (1 + np.abs(rng.standard_normal(n) * jitter))
    low = close * (1 - np.abs(rng.standard_normal(n) * jitter))
    open_ = close * (1 + rng.standard_normal(n) * jitter * 0.5)
    vol = rng.uniform(10, 100, n)
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": vol})


def _uptrend(n: int = 250) -> pd.DataFrame:
    # Positive drift dominating realistic per-bar noise: every lookback's
    # vol-scaled return is clearly positive -> BUY.
    rng = np.random.default_rng(1)
    rets = 0.0015 + rng.standard_normal(n) * 0.004
    close = 50_000 * np.exp(rets.cumsum())
    return _ohlcv_from_close(close)


def _downtrend(n: int = 250) -> pd.DataFrame:
    rng = np.random.default_rng(2)
    rets = -0.0015 + rng.standard_normal(n) * 0.004
    close = 50_000 * np.exp(rets.cumsum())
    return _ohlcv_from_close(close)


def _flat(n: int = 250) -> pd.DataFrame:
    # Zero-drift noise: per-bar returns are mean-zero with realistic vol, so
    # each lookback's vol-scaled momentum is small and the averaged score sits
    # in the dead-band -> HOLD.
    rng = np.random.default_rng(10)
    rets = rng.standard_normal(n) * 0.004
    rets = rets - rets.mean()
    close = 50_000 * np.exp(rets.cumsum())
    return _ohlcv_from_close(close)


def test_uptrend_is_buy_with_positive_confidence():
    sig = ts_mom_vol_scaled(_uptrend())
    assert sig.skill_id == "ts_mom_vol_scaled"
    assert sig.side == "BUY"
    assert sig.confidence > 0.0


def test_downtrend_is_sell_with_positive_confidence():
    sig = ts_mom_vol_scaled(_downtrend())
    assert sig.side == "SELL"
    assert sig.confidence > 0.0


def test_flat_noisy_is_hold():
    sig = ts_mom_vol_scaled(_flat())
    assert sig.side == "HOLD"


def test_confidence_bounded_unit_interval():
    for build in (_uptrend, _downtrend, _flat):
        sig = ts_mom_vol_scaled(build())
        assert 0.0 <= sig.confidence <= 1.0


def test_stops_and_targets_positive():
    sig = ts_mom_vol_scaled(_uptrend())
    assert sig.stop_pct > 0.0
    assert sig.take_profit_pct > 0.0
    # TP is wider than the stop (2.5*ATR stop, 5*ATR tp).
    assert sig.take_profit_pct > sig.stop_pct


def test_short_supported_for_perps():
    # A sharp, clean downtrend must be expressible as a SELL (shorts allowed).
    sig = ts_mom_vol_scaled(_downtrend())
    assert sig.side == "SELL"
    assert sig.confidence > 0.0


def test_insufficient_history_returns_hold():
    # Fewer bars than the longest lookback -> never a sized trade on bad data.
    sig = ts_mom_vol_scaled(_uptrend(n=30))
    assert sig.side == "HOLD"
    assert sig.confidence == 0.0


def test_features_expose_score_and_atr_pct():
    sig = ts_mom_vol_scaled(_uptrend())
    assert "score" in sig.features
    assert "atr_pct" in sig.features
    assert np.isfinite(sig.features["score"])
