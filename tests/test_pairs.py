"""Unit tests for the ETH-BTC cointegration pairs strategy.

Synthetic fixtures deterministically trigger / suppress the signal:
- a cointegrated pair (eth = 1.5*btc + stationary AR(1) residual) whose spread
  is stretched at the last bar -> entry signal of the correct side;
- two independent random walks -> HOLD (no cointegration, skill disabled).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from hermes.skills.library import SkillSignal
from hermes.strategies.pairs import cointegration, pairs_eth_btc


def _ohlcv(close: np.ndarray) -> pd.DataFrame:
    close = np.asarray(close, dtype=float)
    return pd.DataFrame(
        {
            "open": close,
            "high": close * 1.001,
            "low": close * 0.999,
            "close": close,
            "volume": np.full(close.shape, 100.0),
        }
    )


def _cointegrated_pair(n: int = 220, seed: int = 7) -> tuple[np.ndarray, np.ndarray]:
    """btc is a random walk; eth = 1.5*btc + stationary AR(1) residual."""
    rng = np.random.default_rng(seed)
    btc = 30_000 + np.cumsum(rng.standard_normal(n) * 50.0)
    resid = np.zeros(n)
    for t in range(1, n):
        resid[t] = 0.5 * resid[t - 1] + rng.standard_normal() * 40.0
    eth = 1.5 * btc + resid
    return btc, eth


def test_cointegration_detects_cointegrated_pair():
    btc, eth = _cointegrated_pair()
    is_coint, hedge, pvalue = cointegration(eth, btc)
    assert is_coint is True
    assert pvalue < 0.05
    # OLS hedge ratio recovers the true 1.5 slope reasonably well.
    assert 1.2 < hedge < 1.8


def test_cointegration_rejects_independent_random_walks():
    rng = np.random.default_rng(1)
    n = 220
    btc = 30_000 + np.cumsum(rng.standard_normal(n) * 50.0)
    eth = 2_000 + np.cumsum(rng.standard_normal(n) * 40.0)
    is_coint, _hedge, pvalue = cointegration(eth, btc)
    assert is_coint is False
    assert pvalue >= 0.05


def test_pairs_sell_when_spread_stretched_up():
    btc, eth = _cointegrated_pair()
    # Stretch the spread far above its mean at the last bar: eth richly priced
    # vs btc -> z >> z_in -> SELL the spread.
    eth = eth.copy()
    eth[-1] += 6 * 40.0
    sig = pairs_eth_btc(_ohlcv(eth), _ohlcv(btc))
    assert isinstance(sig, SkillSignal)
    assert sig.skill_id == "pairs_eth_btc"
    assert sig.side == "SELL"
    assert sig.confidence > 0
    assert sig.features["z"] > 2.0


def test_pairs_buy_when_spread_stretched_down():
    btc, eth = _cointegrated_pair()
    eth = eth.copy()
    eth[-1] -= 6 * 40.0
    sig = pairs_eth_btc(_ohlcv(eth), _ohlcv(btc))
    assert sig.side == "BUY"
    assert sig.confidence > 0
    assert sig.features["z"] < -2.0


def test_pairs_hold_when_not_cointegrated():
    rng = np.random.default_rng(3)
    n = 220
    btc = 30_000 + np.cumsum(rng.standard_normal(n) * 50.0)
    eth = 2_000 + np.cumsum(rng.standard_normal(n) * 40.0)
    sig = pairs_eth_btc(_ohlcv(eth), _ohlcv(btc))
    assert sig.side == "HOLD"
    assert sig.confidence == 0.0


def test_pairs_hold_when_spread_near_mean():
    btc, eth = _cointegrated_pair()
    # Unmodified last bar: spread sits near its mean (|z| small) -> exit / HOLD.
    sig = pairs_eth_btc(_ohlcv(eth), _ohlcv(btc))
    assert sig.side == "HOLD"
    assert abs(sig.features["z"]) < 2.0


def test_pairs_hold_on_insufficient_history():
    btc, eth = _cointegrated_pair(n=20)
    sig = pairs_eth_btc(_ohlcv(eth), _ohlcv(btc), lookback=200)
    assert sig.side == "HOLD"
    assert sig.confidence == 0.0


def test_pairs_confidence_capped_at_one():
    btc, eth = _cointegrated_pair()
    eth = eth.copy()
    eth[-1] += 20 * 40.0  # absurdly large z
    sig = pairs_eth_btc(_ohlcv(eth), _ohlcv(btc))
    assert sig.confidence <= 1.0
