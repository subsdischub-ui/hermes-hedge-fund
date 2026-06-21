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
) -> BacktestResult:
    lib = get_skill_library()
    round_trip_cost = 2.0 * (fee_bps + slippage_bps) / 10_000.0
    samples: list[dict] = []
    rbs: dict[str, list[float]] = defaultdict(list)
    seq = 0  # monotonic order across (symbol, bar) ~ chronological, for PBO binning

    for _symbol, df in ohlcv_by_symbol.items():
        df = df.reset_index(drop=True)
        highs, lows, closes = df["high"].tolist(), df["low"].tolist(), df["close"].tolist()
        n = len(df)
        for t in range(warmup, n - 1, max(1, step)):
            window = df.iloc[: t + 1]
            regime = detect_regime(window)
            snap = _snapshot(window, regime)
            for sig in lib.evaluate_all(window):
                if sig.side == "HOLD":
                    continue
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

    return BacktestResult(samples=samples, returns_by_skill=dict(rbs))
