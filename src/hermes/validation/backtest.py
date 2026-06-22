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
    # Chronological ledger of EVERY trade (directional + cross-asset) as
    # (event_key, ret_net), event_key being the bar timestamp when available
    # (else a monotonic seq). The report sorts on it so CPCV/DSR see a real
    # time-ordered equity stream, not skill-grouped concatenation.
    chrono: list[tuple[int, float]] = field(default_factory=list)

    @property
    def n_trades(self) -> int:
        return len(self.samples)


def _event_key(df: pd.DataFrame, t: int, fallback: int) -> int:
    """Chronological sort key for a trade entered at bar ``t``: the bar timestamp
    (epoch-ns) when the frame carries a ``ts`` column, else a monotonic counter.
    Using the timestamp makes the ledger globally time-ordered ACROSS symbols."""
    if "ts" in df.columns:
        return int(pd.Timestamp(df["ts"].iloc[t]).value)
    return fallback


def _snapshot(df: pd.DataFrame, regime: dict) -> dict:
    h, l, c = df["high"], df["low"], df["close"]
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
    conviction_min: float = 0.0,
    no_overlap: bool = False,
    skills_allowlist: set[str] | None = None,
) -> BacktestResult:
    """Replay the engine. ``window_cap`` (>0) bounds the trailing slice fed to the
    skills each bar — since every skill is causal and reads at most ~200 trailing
    bars, a cap >= that lookback gives identical signals while turning the per-bar
    work from O(t) into O(window_cap), making multi-year histories tractable.

    ``regime_gate`` backtests the engine AS DESIGNED: at each bar the regime is
    routed with hysteresis and a per-symbol signal is recorded only if its skill
    is in the active set for that regime (trend→{ts_mom,breakout}, range→
    {mean_reversion_regime}, always-on→{pairs,intraday}). This isolates the value
    of the regime router vs. trading every skill unconditionally.

    Selectivity / turnover levers (the "fewer, higher-impact trades" thesis):
    ``conviction_min`` drops any signal whose confidence is below it (trade only
    the strong tail); ``no_overlap`` consolidates the per-bar re-entry into ONE
    position per move — a (symbol, skill) cannot re-enter until ``horizon`` bars
    after its last entry, so a run of identical signals becomes a single held
    position that pays fees once, not every bar; ``skills_allowlist`` restricts
    the active slate (e.g. drop skills that lose even gross)."""
    lib = get_skill_library()
    round_trip_cost = 2.0 * (fee_bps + slippage_bps) / 10_000.0
    samples: list[dict] = []
    rbs: dict[str, list[float]] = defaultdict(list)
    chrono: list[tuple[int, float]] = []  # (event_key, ret_net) for every trade
    busy_until: dict[tuple[str, str], int] = {}  # (symbol, skill) -> next free bar
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
                if skills_allowlist is not None and sig.skill_id not in skills_allowlist:
                    continue
                if active is not None and sig.skill_id not in active:
                    continue  # regime router excludes this skill in this regime
                if sig.confidence < conviction_min:
                    continue  # selectivity: trade only the high-conviction tail
                if no_overlap and t < busy_until.get((symbol, sig.skill_id), -1):
                    continue  # already holding a position from this skill on this symbol
                if no_overlap:
                    busy_until[(symbol, sig.skill_id)] = t + horizon
                _, gross = triple_barrier_label(
                    highs, lows, closes, t, closes[t], sig.side,
                    sig.stop_pct, sig.take_profit_pct, horizon,
                )
                ret_net = gross - round_trip_cost
                samples.append(
                    {
                        "features": meta_features(snap, sig.confidence),
                        "t": t,
                        "confidence": float(sig.confidence),
                        "outcome": 1 if ret_net > 0 else 0,
                        "ret_net": ret_net,
                        "skill_id": sig.skill_id,
                        "seq": seq,
                    }
                )
                rbs[sig.skill_id].append(ret_net)
                chrono.append((_event_key(df, t, seq), ret_net))
                seq += 1

    # Cross-asset skills (e.g. ETH-BTC pairs): a separate, time-aligned pass over
    # the two legs. The signal is on the SPREAD, so we score it as the realised
    # market-neutral spread return over `horizon`, net of round-trip fees on BOTH
    # legs (the BTC hedge leg is scaled by the cointegration hedge ratio). These
    # do NOT enter the model-bootstrap `samples` set — pairs is market-neutral and
    # would pollute the directional confidence calibrator — but they do feed the
    # per-skill / overall return report (DSR, CPCV).
    if skills_allowlist is None or "pairs_eth_btc" in skills_allowlist:
        _run_cross_asset(
            ohlcv_by_symbol, lib, rbs, chrono, round_trip_cost, horizon, warmup, step, window_cap
        )

    # Trades were appended per-symbol then by the cross-asset pass; sort into a
    # single globally time-ordered ledger so the report's CPCV/DSR see real time.
    chrono.sort(key=lambda x: x[0])
    return BacktestResult(samples=samples, returns_by_skill=dict(rbs), chrono=chrono)


def _align_legs(btc: pd.DataFrame, eth: pd.DataFrame) -> pd.DataFrame:
    """Align the two legs bar-for-bar. When both carry a ``ts`` column, INNER-JOIN
    on it so each spread reads BTC and ETH from the SAME wall-clock bar (different
    start dates or internal gaps no longer silently pair mismatched bars). Without
    timestamps (synthetic fixtures) fall back to positional alignment by min length.
    Returns a frame with columns ts (if any), btc_close, eth_close."""
    if "ts" in btc.columns and "ts" in eth.columns:
        m = pd.merge(
            btc[["ts", "close"]].rename(columns={"close": "btc_close"}),
            eth[["ts", "close"]].rename(columns={"close": "eth_close"}),
            on="ts",
            how="inner",
        ).sort_values("ts").reset_index(drop=True)
        return m
    n = min(len(btc), len(eth))
    return pd.DataFrame(
        {
            "btc_close": btc["close"].to_numpy(dtype=float)[:n],
            "eth_close": eth["close"].to_numpy(dtype=float)[:n],
        }
    )


def _run_cross_asset(
    ohlcv_by_symbol: dict[str, pd.DataFrame],
    lib,
    rbs: dict[str, list[float]],
    chrono: list[tuple[int, float]],
    round_trip_cost: float,
    horizon: int,
    warmup: int,
    step: int,
    window_cap: int = 0,
) -> None:
    if not ({"BTC/USDC", "ETH/USDC"} <= set(ohlcv_by_symbol)):
        return
    aligned = _align_legs(
        ohlcv_by_symbol["BTC/USDC"].reset_index(drop=True),
        ohlcv_by_symbol["ETH/USDC"].reset_index(drop=True),
    )
    n = len(aligned)
    btc_c = aligned["btc_close"].to_numpy(dtype=float)
    eth_c = aligned["eth_close"].to_numpy(dtype=float)
    # Minimal per-leg frames carrying only what pairs_eth_btc reads (close).
    btc_leg = pd.DataFrame({"close": btc_c})
    eth_leg = pd.DataFrame({"close": eth_c})

    for t in range(warmup, n - 1, max(1, step)):
        lo = max(0, t + 1 - window_cap) if window_cap > 0 else 0
        ctx = {"BTC/USDC": btc_leg.iloc[lo : t + 1], "ETH/USDC": eth_leg.iloc[lo : t + 1]}
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
            ret_net = float(spread_ret - cost)
            rbs[sig.skill_id].append(ret_net)
            ev = int(pd.Timestamp(aligned["ts"].iloc[t]).value) if "ts" in aligned.columns else t
            chrono.append((ev, ret_net))
