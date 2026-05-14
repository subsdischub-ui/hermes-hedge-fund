"""Voyager-style skill (= strategy template) library.

Each Skill is a deterministic, parameterized function from OHLCV → SkillSignal.
The LLM picks the skill, the skill produces a structured signal, and the
researcher/trader agents reason over the signal alongside news + sentiment.

Skills are first-class memory: their realized PnL is tracked weekly (skill_stats
table) so under-performers can be archived and the LLM can be asked to
propose new ones (procedural memory).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, Literal

import numpy as np
import pandas as pd


@dataclass
class SkillSignal:
    skill_id: str
    side: Literal["BUY", "SELL", "HOLD"]
    confidence: float
    stop_pct: float
    take_profit_pct: float
    rationale: str
    features: dict[str, float]


@dataclass(frozen=True)
class Skill:
    id: str
    description: str
    fn: Callable[[pd.DataFrame], SkillSignal]


def _atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]  # noqa: E741
    tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()


def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0).rolling(n).mean()
    down = -delta.clip(upper=0).rolling(n).mean()
    rs = up / down.replace(0, np.nan)
    return (100 - (100 / (1 + rs))).fillna(50)


def mean_reversion_atr(df: pd.DataFrame) -> SkillSignal:
    """Buy when price closes below lower Bollinger + RSI < 30; symmetric short."""
    c = df["close"]
    ma = c.rolling(20).mean()
    sd = c.rolling(20).std()
    lower = ma - 2 * sd
    upper = ma + 2 * sd
    rsi = _rsi(c)
    atr = _atr(df).iloc[-1]
    px = float(c.iloc[-1])
    stop_pct = float(2.0 * atr / px) if atr > 0 else 0.02
    tp_pct = float(3.0 * atr / px) if atr > 0 else 0.04

    side: Literal["BUY", "SELL", "HOLD"] = "HOLD"
    conf = 0.0
    if px < lower.iloc[-1] and rsi.iloc[-1] < 30:
        side, conf = "BUY", 0.6
    elif px > upper.iloc[-1] and rsi.iloc[-1] > 70:
        side, conf = "SELL", 0.6
    return SkillSignal(
        skill_id="mean_reversion_atr",
        side=side,
        confidence=conf,
        stop_pct=stop_pct,
        take_profit_pct=tp_pct,
        rationale=f"px={px:.2f} bb=[{lower.iloc[-1]:.2f},{upper.iloc[-1]:.2f}] rsi={rsi.iloc[-1]:.1f}",
        features={"rsi": float(rsi.iloc[-1]), "atr_pct": stop_pct / 2},
    )


def breakout_vol_expansion(df: pd.DataFrame) -> SkillSignal:
    """Donchian-20 break + ATR > 1.5×its 50-bar average."""
    h, l, c = df["high"], df["low"], df["close"]  # noqa: E741
    hh = h.rolling(20).max().shift(1)
    ll = l.rolling(20).min().shift(1)
    atr = _atr(df)
    expand = atr / atr.rolling(50).mean()
    px = float(c.iloc[-1])
    stop_pct = float(2.0 * atr.iloc[-1] / px) if atr.iloc[-1] > 0 else 0.02
    tp_pct = float(4.0 * atr.iloc[-1] / px) if atr.iloc[-1] > 0 else 0.06

    side: Literal["BUY", "SELL", "HOLD"] = "HOLD"
    conf = 0.0
    if px > hh.iloc[-1] and expand.iloc[-1] > 1.5:
        side, conf = "BUY", 0.65
    elif px < ll.iloc[-1] and expand.iloc[-1] > 1.5:
        side, conf = "SELL", 0.65
    return SkillSignal(
        skill_id="breakout_vol_expansion",
        side=side,
        confidence=conf,
        stop_pct=stop_pct,
        take_profit_pct=tp_pct,
        rationale=f"hh={hh.iloc[-1]:.2f} ll={ll.iloc[-1]:.2f} px={px:.2f} expand={expand.iloc[-1]:.2f}",
        features={"expand": float(expand.iloc[-1]), "atr_pct": stop_pct / 2},
    )


def trend_follow_ema(df: pd.DataFrame) -> SkillSignal:
    """EMA20 > EMA50 > EMA200 → long; symmetric short. Confirms with rising RSI."""
    c = df["close"]
    ema20 = c.ewm(span=20, adjust=False).mean()
    ema50 = c.ewm(span=50, adjust=False).mean()
    ema200 = c.ewm(span=200, adjust=False).mean()
    rsi = _rsi(c)
    atr = _atr(df).iloc[-1]
    px = float(c.iloc[-1])
    stop_pct = float(2.5 * atr / px) if atr > 0 else 0.025
    tp_pct = float(5.0 * atr / px) if atr > 0 else 0.05

    side: Literal["BUY", "SELL", "HOLD"] = "HOLD"
    conf = 0.0
    long_align = ema20.iloc[-1] > ema50.iloc[-1] > ema200.iloc[-1] and rsi.iloc[-1] > 55
    short_align = ema20.iloc[-1] < ema50.iloc[-1] < ema200.iloc[-1] and rsi.iloc[-1] < 45
    if long_align:
        side, conf = "BUY", 0.55
    elif short_align:
        side, conf = "SELL", 0.55
    return SkillSignal(
        skill_id="trend_follow_ema",
        side=side,
        confidence=conf,
        stop_pct=stop_pct,
        take_profit_pct=tp_pct,
        rationale=f"ema20={ema20.iloc[-1]:.2f} ema50={ema50.iloc[-1]:.2f} ema200={ema200.iloc[-1]:.2f} rsi={rsi.iloc[-1]:.1f}",
        features={"rsi": float(rsi.iloc[-1]), "atr_pct": stop_pct / 2.5},
    )


_REGISTRY: dict[str, Skill] = {
    "mean_reversion_atr": Skill(
        id="mean_reversion_atr",
        description="Buy oversold bands (BB lower + RSI < 30); short overbought. Best in range-bound regimes.",
        fn=mean_reversion_atr,
    ),
    "breakout_vol_expansion": Skill(
        id="breakout_vol_expansion",
        description="Donchian-20 break with ATR expansion (vol > 1.5× 50-bar avg). Best in trend ignitions.",
        fn=breakout_vol_expansion,
    ),
    "trend_follow_ema": Skill(
        id="trend_follow_ema",
        description="EMA 20/50/200 stack alignment + RSI bias. Best in established trends.",
        fn=trend_follow_ema,
    ),
}


class SkillLibrary:
    def __init__(self, registry: dict[str, Skill]):
        self._reg = dict(registry)
        self._archived: set[str] = set()

    def list_active(self) -> list[Skill]:
        return [s for k, s in self._reg.items() if k not in self._archived]

    def get(self, skill_id: str) -> Skill | None:
        return self._reg.get(skill_id)

    def archive(self, skill_id: str) -> None:
        self._archived.add(skill_id)

    def restore(self, skill_id: str) -> None:
        self._archived.discard(skill_id)

    def evaluate_all(self, df: pd.DataFrame) -> list[SkillSignal]:
        return [s.fn(df) for s in self.list_active()]


@lru_cache(maxsize=1)
def get_skill_library() -> SkillLibrary:
    return SkillLibrary(_REGISTRY)
