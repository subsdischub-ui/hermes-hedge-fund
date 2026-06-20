"""HAR-RV volatility forecast + vol-targeting.

A HAR-style (Heterogeneous AutoRegressive) realized-volatility forecast blends
realized vol measured over several horizons (short/medium/long), capturing the
empirical persistence of volatility across timescales. We use it for vol-targeting:
scale the RISK BUDGET so the agent risks less when forecast vol is elevated and
(bounded) more when it is calm — distinct from the ATR-normalized per-trade stop,
which keeps the $-risk-per-trade constant.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def realized_vol(returns: Sequence[float]) -> float:
    """Population standard deviation of per-bar returns. 0.0 if < 2 observations."""
    r = np.asarray(list(returns), dtype=float)
    if r.size < 2:
        return 0.0
    return float(r.std(ddof=0))


def har_rv_forecast(
    returns: Sequence[float], windows: tuple[int, ...] = (24, 120, 480)
) -> float:
    """Equal-weighted HAR blend of realized vol over the given lookback windows
    (default ~1 day / ~5 days / ~20 days of 1h bars). 0.0 if insufficient data."""
    r = list(returns)
    if len(r) < 2:
        return 0.0
    vols = [realized_vol(r[-w:]) for w in windows]
    vols = [v for v in vols if v > 0.0]
    return float(np.mean(vols)) if vols else 0.0


def vol_target_scale(
    sigma_hat: float, target_vol: float, lo: float = 0.5, hi: float = 1.5
) -> float:
    """Risk-budget multiplier targeting `target_vol`: < 1 when forecast vol is
    elevated, > 1 (bounded by [lo, hi]) when calm. 1.0 on degenerate inputs."""
    if sigma_hat <= 0.0 or target_vol <= 0.0:
        return 1.0
    return float(max(lo, min(hi, target_vol / sigma_hat)))
