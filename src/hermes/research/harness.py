"""Research evaluation harness for Strategy #2 candidates.

Walks a causal ``fn(df)->SkillSignal`` over history (trailing window, no-overlap
one-position-per-move like the validated Strategy #1), labels each entry with the
triple-barrier method NET of fees, and reports the honest verdict: net per-trade
edge, per-trade & annualised Sharpe, and a CPCV multi-path positive-fraction.

Crucially it also computes the ORTHOGONALITY gate vs a reference signal
(directional momentum): the Pearson correlation of the two strategies' DAILY PnL
series. A Strategy-#2 candidate must have |rho| <= ~0.3 — otherwise it is just
momentum in disguise and fails its entire purpose, however profitable.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from hermes.skills.library import SkillSignal
from hermes.validation.report import cpcv_sharpe_distribution
from hermes.validation.triple_barrier import triple_barrier_label

SignalFn = Callable[[pd.DataFrame], SkillSignal]


def evaluate_signal(
    ohlcv_by_symbol: dict[str, pd.DataFrame],
    signal_fn: SignalFn,
    fee_bps: float = 7.0,
    slippage_bps: float = 3.0,
    horizon: int = 12,
    warmup: int = 200,
    window_cap: int = 400,
    no_overlap: bool = True,
) -> list[tuple[int, float]]:
    """Return the chronological trade ledger ``[(event_ns, ret_net), ...]`` for a
    single candidate signal, net of round-trip fees, one position per move."""
    rt = 2.0 * (fee_bps + slippage_bps) / 10_000.0
    trades: list[tuple[int, float]] = []
    for df in ohlcv_by_symbol.values():
        df = df.reset_index(drop=True)
        highs, lows, closes = df["high"].tolist(), df["low"].tolist(), df["close"].tolist()
        has_ts = "ts" in df.columns
        n = len(df)
        busy = -1
        for t in range(warmup, n - 1):
            lo = max(0, t + 1 - window_cap)
            sig = signal_fn(df.iloc[lo : t + 1])
            if sig.side == "HOLD":
                continue
            if no_overlap and t < busy:
                continue
            if no_overlap:
                busy = t + horizon
            _, gross = triple_barrier_label(
                highs, lows, closes, t, closes[t], sig.side,
                sig.stop_pct, sig.take_profit_pct, horizon,
            )
            ev = int(pd.Timestamp(df["ts"].iloc[t]).value) if has_ts else t
            trades.append((ev, float(gross - rt)))
    trades.sort(key=lambda x: x[0])
    return trades


def signal_report(trades: list[tuple[int, float]], years: float = 8.8) -> dict:
    """Honest summary of a candidate's trade ledger."""
    rets = np.array([r for _, r in trades], dtype=float)
    n = int(rets.size)
    if n < 5:
        return {"n": n, "mean": 0.0, "sharpe_per_trade": 0.0, "ann_sharpe": 0.0,
                "cpcv_frac_positive": 0.0, "win_rate": 0.0}
    mean = float(rets.mean())
    std = float(rets.std(ddof=1))
    spt = mean / std if std > 0 else 0.0
    cp = cpcv_sharpe_distribution(rets.tolist())
    return {
        "n": n,
        "mean": mean,
        "win_rate": float((rets > 0).mean()),
        "sharpe_per_trade": spt,
        "ann_sharpe": spt * np.sqrt(n / years),  # trades ~independent (no-overlap)
        "cpcv_frac_positive": cp["frac_positive"],
        "cpcv_sharpe_median": cp["sharpe_median"],
    }


def _daily_pnl(trades: list[tuple[int, float]]) -> pd.Series:
    if not trades:
        return pd.Series(dtype=float)
    s = pd.Series(
        [r for _, r in trades],
        index=pd.to_datetime([ev for ev, _ in trades]).floor("D"),
    )
    return s.groupby(level=0).sum()


def orthogonality(trades_a: list[tuple[int, float]], trades_b: list[tuple[int, float]]) -> float:
    """Pearson correlation of the two strategies' DAILY PnL (0 on no-trade days,
    over the union of active days). The Strategy-#2 gate is |rho| <= ~0.3."""
    a, b = _daily_pnl(trades_a), _daily_pnl(trades_b)
    if a.empty or b.empty:
        return 0.0
    idx = a.index.union(b.index)
    av, bv = a.reindex(idx, fill_value=0.0), b.reindex(idx, fill_value=0.0)
    if av.std() == 0 or bv.std() == 0:
        return 0.0
    return float(np.corrcoef(av.to_numpy(), bv.to_numpy())[0, 1])
