"""Size-0 shadow of the PRE-REGISTERED cross-sectional funding-reversion sleeve.

THE CELL (frozen — see ``hermes.forward.prereg.XS_FUNDING_REVERSION``):
  every 14 days, rank the perp universe by trailing-180-settlement funding-z;
  LONG the bottom tercile, SHORT the top tercile, dollar-neutral (equal gross legs).
  The sign is INVERTED (fade high funding-z) because the S2 v-next finding is that
  the cross-section MEAN-REVERTS: the squeeze direction (long-high) lost; only the
  slow, sign-inverted spread carried a +0.58 post-hoc annual lead.

WHY DOLLAR-NEUTRAL IS THE POINT: subtracting the short leg from the long leg cancels
any market-wide move, so the bull-beta that contaminates every directional backtest
is removed *by construction*. A uniform +X% move returns exactly 0.

This module is read-only on the feed, write-only on the shadow log. The eventual
forward verdict must deflate this cell at the FULL trial count (see prereg / FINDINGS).
Everything here is causal: weights use only funding-z known at the rebalance; the
period return uses prices over [rebalance, rebalance+14d] and nothing after.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# The deep-history 23-perp universe (binance_funding_native / binance_klines_4h).
UNIVERSE_23: tuple[str, ...] = (
    "ADA/USDT", "APT/USDT", "ARB/USDT", "ATOM/USDT", "AVAX/USDT", "BCH/USDT",
    "BNB/USDT", "BTC/USDT", "DOGE/USDT", "DOT/USDT", "ETH/USDT", "FIL/USDT",
    "INJ/USDT", "LINK/USDT", "LTC/USDT", "NEAR/USDT", "OP/USDT", "SOL/USDT",
    "SUI/USDT", "TRX/USDT", "UNI/USDT", "WIF/USDT", "XRP/USDT",
)

WINDOW_SETTLEMENTS = 180  # 180 funding settlements (8h native) ≈ 60 days
PERIOD_DAYS = 14          # bi-weekly rebalance (slow → controls 2-leg turnover)
N_TERCILES = 3
INVERT = True             # fade high funding-z (the pre-registered sign)
FEE_BPS = 7.0
SLIPPAGE_BPS = 3.0

LOG_NAME = "shadow/xs_funding_reversion"
KEY_COLS = ("rebalance_ts",)
TS_COL = "rebalance_ts"

_BOOK_COLS = [
    "rebalance_ts", "exit_ts", "n_long", "n_short", "mean_long_fwd", "mean_short_fwd",
    "gross_exposure", "ret_gross", "cost", "ret_net", "closed",
]


def causal_funding_z(funding: pd.Series, window: int = WINDOW_SETTLEMENTS) -> pd.Series:
    """Trailing causal z-score of a funding series (population std, ddof=0).

    Matches the live ``_ztail`` convention: z_t = (f_t - mean) / std over the trailing
    ``window`` settlements. First ``window-1`` points are NaN (no premature signal);
    a flat window (std==0) yields 0.0, never inf.
    """
    f = funding.astype(float)
    m = f.rolling(window, min_periods=window).mean()
    sd = f.rolling(window, min_periods=window).std(ddof=0)
    z = (f - m) / sd
    return z.mask(sd == 0.0, 0.0)


def assign_terciles(
    z: pd.Series, *, invert: bool = INVERT, n_terciles: int = N_TERCILES
) -> pd.Series:
    """Dollar-neutral tercile weights from a cross-section of funding-z.

    ``invert=True`` (the pre-registered sign): SHORT the top tercile (highest z),
    LONG the bottom tercile (lowest z). Weights sum to 0; each leg is equal-weight
    with gross 1 (so gross exposure 2, net 0). Too few names → all zeros.
    """
    w = pd.Series(0.0, index=z.index)
    n = int(z.notna().sum())
    k = n // n_terciles
    if k == 0:
        return w
    order = z.dropna().sort_values(kind="stable").index  # ascending z
    bottom, top = order[:k], order[-k:]
    long_leg, short_leg = (bottom, top) if invert else (top, bottom)
    w.loc[long_leg] = 1.0 / k
    w.loc[short_leg] = -1.0 / k
    return w


def _wide(panel: dict[str, pd.DataFrame], col: str, grid: np.ndarray) -> pd.DataFrame:
    out = pd.DataFrame(index=grid)
    for sym, df in panel.items():
        out[sym] = df.drop_duplicates("ts", keep="last").set_index("ts")[col].reindex(grid)
    return out


def xs_reversion_book(
    panel: dict[str, pd.DataFrame],
    *,
    period_days: int = PERIOD_DAYS,
    fee_bps: float = FEE_BPS,
    slippage_bps: float = SLIPPAGE_BPS,
    invert: bool = INVERT,
    n_terciles: int = N_TERCILES,
) -> pd.DataFrame:
    """Period-by-period dollar-neutral long-short ledger for the pre-registered cell.

    ``panel`` = ``{symbol: DataFrame(ts[ms, daily], close, funding_z)}`` where
    ``funding_z`` is precomputed at settlement granularity (see ``causal_funding_z``).
    The final, not-yet-elapsed period is logged OPEN (mark-to-market), never closed
    prematurely — the idempotent log finalises it once 14 days have passed.
    """
    if not panel:
        return pd.DataFrame(columns=_BOOK_COLS)
    grid = np.array(sorted({int(t) for df in panel.values() for t in df["ts"]}))
    if grid.size == 0:
        return pd.DataFrame(columns=_BOOK_COLS)
    close_w = _wide(panel, "close", grid)
    z_w = _wide(panel, "funding_z", grid)
    close_ff_last = close_w.ffill().iloc[-1]  # last available close per symbol (MTM)
    per_side = (fee_bps + slippage_bps) / 10_000.0
    rows: list[dict] = []
    for p in range(0, grid.size, period_days):
        exit_p = p + period_days
        closed = exit_p < grid.size
        z_row = z_w.iloc[p]
        close_now = close_w.iloc[p]
        close_exit = close_w.iloc[exit_p] if closed else close_ff_last
        exit_ts = int(grid[exit_p]) if closed else int(grid[-1])
        valid = z_row.index[z_row.notna() & close_now.notna() & close_exit.notna()]
        if len(valid) < n_terciles:
            continue
        w = assign_terciles(z_row[valid], invert=invert, n_terciles=n_terciles)
        fwd = close_exit[valid] / close_now[valid] - 1.0
        long_mask, short_mask = w > 0, w < 0
        if not closed and not (close_ff_last[valid] > 0).any():
            continue
        gross = float((w * fwd).sum())
        gross_exposure = float(w.abs().sum())
        cost = 2.0 * gross_exposure * per_side  # enter + exit, both legs (conservative)
        rows.append(
            {
                "rebalance_ts": int(grid[p]),
                "exit_ts": exit_ts,
                "n_long": int(long_mask.sum()),
                "n_short": int(short_mask.sum()),
                "mean_long_fwd": float(fwd[long_mask].mean()) if long_mask.any() else 0.0,
                "mean_short_fwd": float(fwd[short_mask].mean()) if short_mask.any() else 0.0,
                "gross_exposure": gross_exposure,
                "ret_gross": gross,
                "cost": cost,
                "ret_net": gross - cost,
                "closed": bool(closed),
            }
        )
    if not rows:
        return pd.DataFrame(columns=_BOOK_COLS)
    return pd.DataFrame(rows, columns=_BOOK_COLS)
