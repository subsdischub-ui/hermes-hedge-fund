"""Market-analyst node: pulls OHLCV, computes ATR & indicator snapshot."""

from __future__ import annotations

import numpy as np
import pandas as pd

from hermes.data import get_market
from hermes.data.regime import detect_regime, regime_signal_weight
from hermes.graph.state import HermesState
from hermes.skills import get_skill_library
from hermes.strategies.combination import annotate_active_and_blend

# Last routed regime per symbol, so route_regime's hysteresis carries across
# ticks (each tick gets a fresh HermesState). Advisory only — no sizing depends
# on it, so a process-local cache is sufficient.
_PREV_REGIME: dict[str, str] = {}


async def run(state: HermesState) -> HermesState:
    md = get_market()
    df = await md.ohlcv(state.symbol, timeframe=state.timeframe, limit=300)
    if len(df) < 60:
        state.error = "insufficient ohlcv"
        return state

    h, l, c = df["high"], df["low"], df["close"]  # noqa: E741
    tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14).mean().iloc[-1]
    ret_1 = float(np.log(c.iloc[-1] / c.iloc[-2]))
    last = float(c.iloc[-1])
    regime = detect_regime(df)
    state.regime = regime
    state.ohlcv = df
    state.snapshot = {
        "price": last,
        "atr": float(atr),
        "atr_pct": float(atr / last),
        "ret_1": ret_1,
        "vol_avg_20": float(df["volume"].tail(20).mean()),
        "vol_last": float(df["volume"].iloc[-1]),
        "high_20": float(df["high"].tail(20).max()),
        "low_20": float(df["low"].tail(20).min()),
        "regime": regime["regime"],
        "regime_efficiency_ratio": round(regime["efficiency_ratio"], 3),
    }
    # Symbol-aware evaluation: BTC-only skills (e.g. intraday_tod_btc) are not run
    # for ETH/SOL, matching the backtest.
    sigs = [s.__dict__ for s in get_skill_library().evaluate_for_symbol(state.symbol, df)]
    # Regime-condition the skill signals: down-weight off-regime skills (e.g.
    # mean-reversion in a trend) so the trader/researchers see a regime-aware prior.
    for sig in sigs:
        w = regime_signal_weight(sig.get("skill_id", ""), regime["regime"])
        sig["regime_weight"] = w
        sig["confidence"] = float(sig.get("confidence", 0.0)) * w

    # Combination layer (SHADOW/advisory): route the regime with hysteresis and tag
    # which skills are active + their inverse-vol blend weight. Does not size or
    # place orders — it only enriches context for the researcher/trader agents.
    # Live blend weights stay 0 until per-skill journal returns are plumbed in;
    # the blend is validated in the backtest (CPCV/DSR) before it moves capital.
    prev = _PREV_REGIME.get(state.symbol)
    routed, sigs = annotate_active_and_blend(
        sigs, regime["efficiency_ratio"], prev, returns_by_skill={}
    )
    _PREV_REGIME[state.symbol] = routed
    state.skill_signals = sigs
    return state
