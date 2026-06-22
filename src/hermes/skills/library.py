"""Voyager-style skill (= strategy template) library.

Each Skill is a deterministic, parameterized function from OHLCV → SkillSignal.
The LLM picks the skill, the skill produces a structured signal, and the
researcher/trader agents reason over the signal alongside news + sentiment.

Skills are first-class memory: their realized PnL is tracked weekly
(skill_stats table) so under-performers can be archived and the LLM can be
asked to propose new ones. Archival is persisted across restarts via
SkillStatRow — see `_reload_archival_from_db`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from typing import Literal

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
    # Restrict the skill to symbols whose name starts with one of these prefixes
    # (case-insensitive), e.g. ("BTC",) for a BTC-only skill. None = all symbols.
    symbols: tuple[str, ...] | None = None


@dataclass(frozen=True)
class CrossAssetSkill:
    """A skill that needs more than one symbol's OHLCV at once (e.g. pairs).

    Its `fn` takes a context mapping ``symbol -> aligned OHLCV DataFrame`` and
    returns a single SkillSignal for the cross-asset position (e.g. the spread).
    """

    id: str
    description: str
    fn: Callable[[dict[str, pd.DataFrame]], SkillSignal]


def _applies_to(skill: Skill, symbol: str | None) -> bool:
    """Whether a per-symbol skill should run for `symbol`.

    `symbol=None` (the symbol-agnostic back-compat path) runs every skill. A
    skill with `symbols=None` runs everywhere; otherwise it runs only when the
    symbol name starts with one of its allowed prefixes (so "BTC" matches
    "BTC/USDC", "BTC/USDT", ...).
    """
    if skill.symbols is None or symbol is None:
        return True
    su = symbol.upper()
    return any(su.startswith(p.upper()) for p in skill.symbols)


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
    c = df["close"]
    ma = c.rolling(20).mean()
    sd = c.rolling(20).std()
    lower = ma - 2 * sd
    upper = ma + 2 * sd
    rsi = _rsi(c)
    atr = _atr(df).iloc[-1]
    px = float(c.iloc[-1])
    stop_pct = float(2.0 * atr / px) if (atr and px > 0) else 0.02
    tp_pct = float(3.0 * atr / px) if (atr and px > 0) else 0.04

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
    h, l, c = df["high"], df["low"], df["close"]  # noqa: E741
    hh = h.rolling(20).max().shift(1)
    ll = l.rolling(20).min().shift(1)
    atr = _atr(df)
    expand = atr / atr.rolling(50).mean()
    px = float(c.iloc[-1])
    stop_pct = float(2.0 * atr.iloc[-1] / px) if (atr.iloc[-1] and px > 0) else 0.02
    tp_pct = float(4.0 * atr.iloc[-1] / px) if (atr.iloc[-1] and px > 0) else 0.06

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
    c = df["close"]
    ema20 = c.ewm(span=20, adjust=False).mean()
    ema50 = c.ewm(span=50, adjust=False).mean()
    ema200 = c.ewm(span=200, adjust=False).mean()
    rsi = _rsi(c)
    atr = _atr(df).iloc[-1]
    px = float(c.iloc[-1])
    stop_pct = float(2.5 * atr / px) if (atr and px > 0) else 0.025
    tp_pct = float(5.0 * atr / px) if (atr and px > 0) else 0.05

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


# mean_reversion_atr and trend_follow_ema remain defined above (importable for
# reference / rollback) but are NOT in the active slate: the Strategy Engine v1
# replaces them with mean_reversion_regime and ts_mom_vol_scaled to avoid
# double-counting range/trend exposure and inflating the DSR trial count.
def _build_registry() -> dict[str, Skill]:
    """Build the active per-symbol slate. Lazy imports avoid the import cycle
    (the new skill modules import SkillSignal *from* this module)."""
    from hermes.skills.intraday_tod import intraday_tod_btc
    from hermes.skills.mean_reversion_regime import mean_reversion_regime
    from hermes.skills.ts_mom import ts_mom_vol_scaled

    return {
        "ts_mom_vol_scaled": Skill(
            id="ts_mom_vol_scaled",
            description="Vol-scaled time-series momentum averaged over fixed lookbacks {12,24,48,96}; long/short. Best in established trends (generalises trend_follow_ema).",
            fn=ts_mom_vol_scaled,
        ),
        "breakout_vol_expansion": Skill(
            id="breakout_vol_expansion",
            description="Donchian-20 break with ATR expansion (vol > 1.5x 50-bar avg). Best in trend ignitions.",
            fn=breakout_vol_expansion,
        ),
        "mean_reversion_regime": Skill(
            id="mean_reversion_regime",
            description="RSI(2)+Bollinger %B, active ONLY in range regime (Kaufman ER gate) and on the mean-reverting side of MA200. Refactor of mean_reversion_atr.",
            fn=mean_reversion_regime,
        ),
        "intraday_tod_btc": Skill(
            id="intraday_tod_btc",
            description="Intraday time-of-day BTC: the 13:00→14:00 UTC post-US-open return predicts the 20:00→21:00 UTC close window. ≤1 trade/day, BTC only.",
            fn=intraday_tod_btc,
            symbols=("BTC",),
        ),
    }


def _build_cross_registry() -> dict[str, CrossAssetSkill]:
    """Build the active cross-asset slate (lazy import to avoid the cycle)."""
    from hermes.strategies.pairs import pairs_eth_btc

    def _pairs_ctx(ctx: dict[str, pd.DataFrame]) -> SkillSignal:
        eth = ctx.get("ETH/USDC")
        btc = ctx.get("BTC/USDC")
        if eth is None or btc is None:
            return SkillSignal(
                skill_id="pairs_eth_btc",
                side="HOLD",
                confidence=0.0,
                stop_pct=0.02,
                take_profit_pct=0.04,
                rationale="missing ETH/USDC or BTC/USDC leg in context",
                features={},
            )
        return pairs_eth_btc(eth, btc)

    return {
        "pairs_eth_btc": CrossAssetSkill(
            id="pairs_eth_btc",
            description="Cross-asset ETH-BTC cointegration spread (Engle-Granger), market-neutral. Regime-independent (always-on).",
            fn=_pairs_ctx,
        ),
    }


class SkillLibrary:
    def __init__(
        self,
        registry: dict[str, Skill],
        cross_registry: dict[str, CrossAssetSkill] | None = None,
    ):
        self._reg = dict(registry)
        self._cross = dict(cross_registry or {})
        self._archived: set[str] = set()

    def list_active(self) -> list[Skill]:
        return [s for k, s in self._reg.items() if k not in self._archived]

    def list_active_cross(self) -> list[CrossAssetSkill]:
        return [s for k, s in self._cross.items() if k not in self._archived]

    def get(self, skill_id: str) -> Skill | None:
        return self._reg.get(skill_id)

    def archive(self, skill_id: str) -> None:
        self._archived.add(skill_id)

    def restore(self, skill_id: str) -> None:
        self._archived.discard(skill_id)

    def evaluate_for_symbol(self, symbol: str | None, df: pd.DataFrame) -> list[SkillSignal]:
        """Run the per-symbol skills that apply to `symbol` (symbol filtering)."""
        return [s.fn(df) for s in self.list_active() if _applies_to(s, symbol)]

    def evaluate_cross_asset(self, ctx: dict[str, pd.DataFrame]) -> list[SkillSignal]:
        """Run the cross-asset skills over an aligned multi-symbol context."""
        return [s.fn(ctx) for s in self.list_active_cross()]

    def evaluate_all(self, df: pd.DataFrame) -> list[SkillSignal]:
        """Back-compat symbol-agnostic evaluation (runs every per-symbol skill)."""
        return self.evaluate_for_symbol(None, df)

    async def reload_archival_from_db(self) -> None:
        """Rehydrate the in-memory archive set from the most recent SkillStatRow
        per skill. Called at startup so weekly archival decisions survive restart.
        """
        from sqlalchemy import select

        from hermes.memory.store import SkillStatRow, get_db

        db = get_db()
        async with db.session() as s:
            rows = (
                await s.execute(
                    select(SkillStatRow).order_by(SkillStatRow.week_starting.desc())
                )
            ).scalars().all()
        latest: dict[str, SkillStatRow] = {}
        for r in rows:
            latest.setdefault(r.skill_id, r)
        self._archived = {sid for sid, r in latest.items() if r.archived}


@lru_cache(maxsize=1)
def get_skill_library() -> SkillLibrary:
    # Archival state is rehydrated explicitly by the API lifespan via
    # reload_archival_from_db(); constructing the library is intentionally cheap
    # and side-effect-free so it is safe to call from sync contexts (e.g. backtest).
    return SkillLibrary(_build_registry(), _build_cross_registry())
