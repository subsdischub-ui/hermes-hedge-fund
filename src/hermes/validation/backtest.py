"""Backtest engine — replay the deterministic skills over historical OHLCV and
label every signal with the triple-barrier method (NET of round-trip fees and
slippage), producing (feature, outcome) samples for model bootstrapping and a
per-skill stream of net returns for Deflated Sharpe / PBO.

This is an OFFLINE analysis tool: it lets us judge an edge and pre-fit the
calibrator / meta-labeler from hundreds of historical trades instead of waiting
weeks for live paper trades to accumulate.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from hermes.data.regime import detect_regime
from hermes.risk.meta_labeling import meta_features
from hermes.skills import get_skill_library
from hermes.strategies.combination import (
    ALWAYS_ON,
    RANGE_SKILLS,
    TREND_SKILLS,
    route_regime,
    select_active_skills,
)
from hermes.validation.triple_barrier import triple_barrier_label


@dataclass
class BacktestResult:
    samples: list[dict] = field(default_factory=list)
    returns_by_skill: dict[str, list[float]] = field(default_factory=dict)

    @property
    def n_trades(self) -> int:
        return len(self.samples)


def _snapshot(df: pd.DataFrame, regime: dict) -> dict:
    h, l, c = df["high"], df["low"], df["close"]  # noqa: E741
    tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    atr = float(tr.rolling(14).mean().iloc[-1])
    last = float(c.iloc[-1])
    return {
        "price": last,
        "atr": atr,
        "atr_pct": (atr / last) if last else 0.0,
        "ret_1": float(np.log(c.iloc[-1] / c.iloc[-2])) if len(c) > 1 else 0.0,
        "vol_avg_20": float(df["volume"].tail(20).mean()),
        "vol_last": float(df["volume"].iloc[-1]),
        "regime": regime["regime"],
        "regime_efficiency_ratio": regime["efficiency_ratio"],
    }


def run_backtest(
    ohlcv_by_symbol: dict[str, pd.DataFrame],
    fee_bps: float = 7.0,
    slippage_bps: float = 3.0,
    horizon: int = 24,
    warmup: int = 60,
    step: int = 1,
    window_cap: int = 0,
    regime_gate: bool = False,
) -> BacktestResult:
    """Replay the engine. ``window_cap`` (>0) bounds the trailing slice fed to the
    skills each bar — since every skill is causal and reads at most ~200 trailing
    bars, a cap >= that lookback gives identical signals while turning the per-bar
    work from O(t) into O(window_cap), making multi-year histories tractable.

    ``regime_gate`` backtests the engine AS DESIGNED: at each bar the regime is
    routed with hysteresis and a per-symbol signal is recorded only if its skill
    is in the active set for that regime (trend→{ts_mom,breakout}, range→
    {mean_reversion_regime}, always-on→{pairs,intraday}). This isolates the value
    of the regime router vs. trading every skill unconditionally."""
    lib = get_skill_library()
    round_trip_cost = 2.0 * (fee_bps + slippage_bps) / 10_000.0
    samples: list[dict] = []
    rbs: dict[str, list[float]] = defaultdict(list)
    seq = 0  # monotonic order across (symbol, bar) ~ chronological, for PBO binning

    for symbol, df in ohlcv_by_symbol.items():
        df = df.reset_index(drop=True)
        highs, lows, closes = df["high"].tolist(), df["low"].tolist(), df["close"].tolist()
        n = len(df)
        prev_routed: str | None = None
        for t in range(warmup, n - 1, max(1, step)):
            lo = max(0, t + 1 - window_cap) if window_cap > 0 else 0
            window = df.iloc[lo : t + 1]
            regime = detect_regime(window)
            snap = _snapshot(window, regime)
            active: set[str] | None = None
            if regime_gate:
                prev_routed = route_regime(regime["efficiency_ratio"], prev_routed)
                active = set(
                    select_active_skills(prev_routed, TREND_SKILLS, RANGE_SKILLS, ALWAYS_ON)
                )
            # Symbol-aware: BTC-only skills (e.g. intraday_tod_btc) are skipped for
            # other symbols, exactly as in the live tick.
            for sig in lib.evaluate_for_symbol(symbol, window):
                if sig.side == "HOLD":
                    continue
                if active is not None and sig.skill_id not in active:
                    continue  # regime router excludes this skill in this regime
                _, gross = triple_barrier_label(
                    highs, lows, closes, t, closes[t], sig.side,
                    sig.stop_pct, sig.take_profit_pct, horizon,
                )
                ret_net = gross - round_trip_cost
                samples.append(
                    {
                        "features": meta_features(snap, sig.confidence),
                        "confidence": float(sig.confidence),
                        "outcome": 1 if ret_net > 0 else 0,
                        "ret_net": ret_net,
                        "skill_id": sig.skill_id,
                        "seq": seq,
                    }
                )
                rbs[sig.skill_id].append(ret_net)
                seq += 1

    # Cross-asset skills (e.g. ETH-BTC pairs): a separate, time-aligned pass over
    # the two legs. The signal is on the SPREAD, so we score it as the realised
    # market-neutral spread return over `horizon`, net of round-trip fees on BOTH
    # legs (the BTC hedge leg is scaled by the cointegration hedge ratio). These
    # do NOT enter the model-bootstrap `samples` set — pairs is market-neutral and
    # would pollute the directional confidence calibrator — but they do feed the
    # per-skill / overall return report (DSR, CPCV).
    _run_cross_asset(
        ohlcv_by_symbol, lib, rbs, round_trip_cost, horizon, warmup, step, window_cap
    )

    return BacktestResult(samples=samples, returns_by_skill=dict(rbs))


def _run_cross_asset(
    ohlcv_by_symbol: dict[str, pd.DataFrame],
    lib,
    rbs: dict[str, list[float]],
    round_trip_cost: float,
    horizon: int,
    warmup: int,
    step: int,
    window_cap: int = 0,
) -> None:
    if not ({"BTC/USDC", "ETH/USDC"} <= set(ohlcv_by_symbol)):
        return
    btc = ohlcv_by_symbol["BTC/USDC"].reset_index(drop=True)
    eth = ohlcv_by_symbol["ETH/USDC"].reset_index(drop=True)
    n = min(len(btc), len(eth))
    btc_c = btc["close"].to_numpy(dtype=float)[:n]
    eth_c = eth["close"].to_numpy(dtype=float)[:n]

    for t in range(warmup, n - 1, max(1, step)):
        lo = max(0, t + 1 - window_cap) if window_cap > 0 else 0
        ctx = {"BTC/USDC": btc.iloc[lo : t + 1], "ETH/USDC": eth.iloc[lo : t + 1]}
        for sig in lib.evaluate_cross_asset(ctx):
            if sig.side == "HOLD":
                continue
            h = min(horizon, n - 1 - t)
            if h <= 0:
                continue
            hedge = float(sig.features.get("hedge", 1.0))
            eth_ret = eth_c[t + h] / eth_c[t] - 1.0
            btc_ret = btc_c[t + h] / btc_c[t] - 1.0
            # Long the spread = long ETH, short hedge*BTC.
            spread_ret = eth_ret - hedge * btc_ret
            if sig.side == "SELL":  # short the spread
                spread_ret = -spread_ret
            cost = round_trip_cost * (1.0 + abs(hedge))  # both legs traded
            rbs[sig.skill_id].append(float(spread_ret - cost))
