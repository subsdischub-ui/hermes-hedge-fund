"""Tests for the regime-gated mean-reversion skill.

Thèse testée: la réversion (RSI(2) + Bollinger %B) ne doit s'activer QU'EN range
(gate Kaufman ER) et du bon côté de la MA200. Les séries sont construites pour
déclencher déterministiquement chaque branche, sans look-ahead.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from hermes.data.regime import detect_regime
from hermes.skills.mean_reversion_regime import mean_reversion_regime


def _ohlcv(closes: list[float]) -> pd.DataFrame:
    c = np.asarray(closes, dtype=float)
    high = c * 1.001
    low = c * 0.999
    open_ = c
    vol = np.full(c.size, 50.0)
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": c, "volume": vol})


def _range_oversold_df() -> pd.DataFrame:
    """Plateau bas bruité (MA200 ancrée bas) -> régime range ; plateau récent
    haut puis chute nette en 3 barres qui reste AU-DESSUS de la MA200
    (RSI(2) bas, %B<0.1, close>MA200) -> BUY."""
    rng = np.random.default_rng(7)
    base = 98.0 + rng.standard_normal(215) * 0.05
    ramp = [99.0, 99.7, 100.4, 101.0, 101.4, 101.5]
    plateau = [101.5] * 14
    drop = [100.6, 99.7, 99.0]
    return _ohlcv(list(base) + ramp + plateau + drop)


def _range_overbought_df() -> pd.DataFrame:
    """Miroir: plateau haut (MA200 ancrée haut) -> range ; plateau récent bas
    puis flambée en 3 barres restant SOUS la MA200 (RSI(2)>90, %B>0.9,
    close<MA200) -> SELL."""
    rng = np.random.default_rng(11)
    base = 102.0 + rng.standard_normal(215) * 0.05
    ramp = [101.0, 100.3, 99.6, 99.0, 98.6, 98.5]
    plateau = [98.5] * 14
    spike = [98.6, 99.7, 101.0]
    return _ohlcv(list(base) + ramp + plateau + spike)


def test_range_oversold_triggers_buy():
    df = _range_oversold_df()
    assert detect_regime(df)["regime"] == "range"
    sig = mean_reversion_regime(df)
    assert sig.skill_id == "mean_reversion_regime"
    assert sig.side == "BUY"
    assert 0.5 <= sig.confidence <= 0.7
    assert sig.stop_pct > 0
    assert sig.take_profit_pct > 0


def test_range_overbought_triggers_sell():
    df = _range_overbought_df()
    assert detect_regime(df)["regime"] == "range"
    sig = mean_reversion_regime(df)
    assert sig.side == "SELL"
    assert 0.5 <= sig.confidence <= 0.7


def test_trend_regime_gates_to_hold():
    """Même setup survendu, mais sur une vraie tendance baissière détectée:
    le gate doit forcer HOLD (coeur de la skill)."""
    rng = np.random.default_rng(3)
    # Forte tendance baissière monotone -> detect_regime == trend_down.
    drift = np.linspace(0.0, -40.0, 235)
    closes = list(100.0 + drift + rng.standard_normal(235) * 0.02)
    # Plongeon final qui rendrait RSI(2) bas / %B bas en isolation.
    closes += [closes[-1] - 1.0, closes[-1] - 2.5, closes[-1] - 4.5]
    df = _ohlcv(closes)
    assert detect_regime(df)["regime"] in ("trend_down", "trend_up")
    sig = mean_reversion_regime(df)
    assert sig.side == "HOLD"
    assert sig.confidence == 0.0


def test_range_but_neutral_is_hold():
    """Range sans extrême RSI/%B -> HOLD (pas de sur-trading)."""
    rng = np.random.default_rng(5)
    closes = list(100.0 + rng.standard_normal(240) * 0.1)
    df = _ohlcv(closes)
    assert detect_regime(df)["regime"] == "range"
    sig = mean_reversion_regime(df)
    assert sig.side == "HOLD"


def test_insufficient_history_is_hold():
    df = _ohlcv([100.0 + (i % 3) for i in range(50)])
    sig = mean_reversion_regime(df)
    assert sig.side == "HOLD"
    assert sig.confidence == 0.0
