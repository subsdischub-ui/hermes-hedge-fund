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
    h, l, c = df["high"], df["low"], df["close"]
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
    h, l, c = df["high"], df["low"], df["close"]
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
    from hermes.research.signals import funding_carry_harvest, funding_squeeze_continuation
    from hermes.skills.intraday_tod import intraday_tod_btc
    from hermes.skills.mean_reversion_regime import mean_reversion_regime
    from hermes.skills.newdata_shadow import (
        onchain_stablecoin_growth,
        options_iv_dvol_reversion,
    )
    from hermes.skills.ts_mom import ts_mom_vol_scaled

    return {
        "ts_mom_vol_scaled": Skill(
            id="ts_mom_vol_scaled",
            description="Vol-scaled time-series momentum averaged over fixed lookbacks {12,24,48,96}; long/short. Best in established trends (generalises trend_follow_ema).",
            fn=ts_mom_vol_scaled,
        ),
        "funding_squeeze_continuation": Skill(
            id="funding_squeeze_continuation",
            description="Strategy #2 — the book's ONE real edge (basket joint-null p=0.0005 post-gatefix 2026-07-02): extreme-positive perp funding (z>=2 over 180 bars) marks leveraged-demand continuation; LONG, ~24-bar hold. Honest magnitudes per the 2026-06-25 re-audit: per-trade Sharpe ~0.3, OOS-40 ~0.3; weekly book rho to momentum was 0.35 (the old daily ~0.04 understated regime crowding). Needs a funding_rate column (attached live from Binance USDT-M).",
            fn=funding_squeeze_continuation,
            # FIDÉLITÉ (2026-08-01) : le joint-null p=0.0005 — l'unique preuve statistique du
            # livre — porte sur TROIS actifs (audit_s2_funding_squeeze.py:45 ASSETS =
            # ["BTC","ETH","SOL"] ; _s2_funding_squeeze.json n'a que ces per_asset). Sans
            # `symbols=`, le skill tournait sur les 7 perps de l'univers live : 4 d'entre eux
            # (AVAX/SUI/ARB/WIF) recevaient un signal jamais validé pour eux. Mesuré sur données
            # réelles (règle gelée, horizon 24 barres, 15 bps RT, 2026-02→2026-08) : univers
            # validé -82.1 bps/évt (n=13) contre -739.6 bps/évt (n=9) hors validation, ~9x pire.
            # Le mécanisme de restriction existait déjà et servait à 3 autres skills — il avait
            # simplement été oublié sur le seul edge validé. Ne PAS élargir cette liste sans
            # étendre d'abord le fichier de validation (test_s2_validated_universe.py le cloue),
            # et se souvenir que le ledger #41 a clos la piste d'élargissement d'univers.
            symbols=("BTC", "ETH", "SOL"),
        ),
        "funding_carry_harvest": Skill(
            id="funding_carry_harvest",
            # Strategy #3 — TRADABLE and in strategy.allowed_skills (the old
            # "DISPLAY-ONLY: always HOLD" comment here was stale/wrong — audit
            # 2026-07-10): the signal emits BUY at funding_bps>=2 and the paper
            # exchange routes it STRUCTURALLY to a delta-neutral short-perp+long-spot
            # pair (paper.py), never a naked order (naked is OOS-negative on 9y).
            # RESTATED per the 2026-06-25 re-audit: at honest 30bps two-leg taker the
            # edge is thin (+16..+23bps/trade, OOS-40 ~0) — healthy only with maker
            # execution (~12bps), hence effectively HOLD-most-of-the-time in calm
            # funding. Real-venue carry execution is NOT wired (paper-only guard).
            description="Strategy #3: delta-neutral funding carry (short perp + long spot, structurally paired — never naked). Thin at taker cost (re-audit: +16..+23bps/trade, OOS-40 ~0); arms only at funding_bps>=2.",
            fn=funding_carry_harvest,
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
        # --- SHADOW sleeves (new-data diversifiers) -------------------------- #
        # Both are DEMOTED/NOT-confirmed candidates (each FAILS the deflated bar:
        # DVOL Bonferroni p across the config grid; stablecoin null). Slow-orthogonal
        # on data OUTSIDE OHLCV+funding (options-implied vol + on-chain stablecoin
        # supply), shadowed to accrue clean forward OOS. Registered so the
        # dashboard shows their gauge, but DELIBERATELY kept OUT of
        # strategy.allowed_skills: the selective gate (hermes.risk.selective) rejects
        # any skill not in the slate, so these are computed + logged every tick yet
        # can NEVER be routed to execution. Log-only until they prove forward (same
        # SHADOW discipline as funding_carry_harvest). BTC/ETH only (where validated).
        "options_iv_dvol_reversion": Skill(
            id="options_iv_dvol_reversion",
            description="SHADOW: Deribit DVOL fear/complacency reversion (z>=+2 long, <=-2 short, ~24h). New-data diversifier (rho -0.26 vs momentum); needs a dvol_z_90d column attached live. BTC/ETH only.",
            fn=options_iv_dvol_reversion,
            symbols=("BTC", "ETH"),
        ),
        "onchain_stablecoin_growth": Skill(
            id="onchain_stablecoin_growth",
            description="SHADOW: on-chain stablecoin-supply-growth (USDT+USDC 60d z; >=+1 long, <=-1 short, ~3d). New-data diversifier (rho ~0 vs funding); needs a stbl_supply_growth_z column attached live. BTC/ETH only.",
            fn=onchain_stablecoin_growth,
            symbols=("BTC", "ETH"),
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
