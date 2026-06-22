"""Perp funding-rate history loader + as-of attachment to OHLCV bars.

Funding is the best-economically-grounded Strategy #2 input from R&D round 1: an
extreme funding rate means the crowd is over-positioned (longs pay shorts when
positive), so fading it both (a) targets the squeeze and (b) EARNS the carry while
waiting. Exogenous to price -> the most defensible orthogonality to momentum.

This module only loads + aligns (as-of, no look-ahead): each bar gets the latest
funding rate whose settlement time is <= the bar's open time. Both keyed on epoch
ms so it composes with the OHLCV loader before any datetime conversion.
"""

from __future__ import annotations

import os

import pandas as pd


def load_funding(path: str | os.PathLike[str]) -> pd.DataFrame:
    """Load a funding CSV (columns ts[ms epoch], funding_rate), sorted ascending."""
    df = pd.read_csv(path)
    return df[["ts", "funding_rate"]].sort_values("ts").reset_index(drop=True)


def attach_funding(ohlcv: pd.DataFrame, funding: pd.DataFrame) -> pd.DataFrame:
    """As-of merge the latest funding rate onto each OHLCV bar (ts in epoch ms).

    direction='backward' => bar t gets the most recent funding settled at or before
    t (never a future settlement) — causal. Bars before the first funding record get
    funding_rate = 0.0 (treated as neutral, not dropped).
    """
    left = ohlcv.sort_values("ts").reset_index(drop=True)
    right = funding.sort_values("ts").reset_index(drop=True)
    merged = pd.merge_asof(left, right, on="ts", direction="backward")
    merged["funding_rate"] = merged["funding_rate"].fillna(0.0)
    return merged
