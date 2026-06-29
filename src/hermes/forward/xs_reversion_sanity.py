"""IN-SAMPLE (CONTAMINATED) robustness check for the XS funding-reversion cell.

⚠️  EVERYTHING THIS MODULE COMPUTES IS IN-SAMPLE AND CONTAMINATED BY BULL-BETA. It
exists for ONE decision only: is the +0.58 post-hoc lead *stable* (worth wiring to a
size-0 forward shadow) or a *lucky cell* (document and do not wire)? It is NEVER a
verdict — the verdict comes solely from the forward OOS, deflated at the full count.

Stability is judged three ways (the mandate's three axes):
  * subperiod stability — does the sign hold across chronological thirds?
  * parameter neighbourhood — cadences {7,10,14,21}d x quantiles {terciles..quintiles};
  * leave-one-instrument-out — is the edge carried by one lucky name?

The metric primitives (annualised Sharpe, subperiod split) are unit-tested; the lake
loader is a thin read-only seam validated by spot-check.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from hermes.forward.shadow_xs_reversion import (
    PERIOD_DAYS,
    UNIVERSE_23,
    WINDOW_SETTLEMENTS,
    causal_funding_z,
    xs_reversion_book,
)

_DAYS_PER_YEAR = 365.25


def _periods_per_year(period_days: int) -> float:
    return _DAYS_PER_YEAR / period_days


def annualized_sharpe(
    book: pd.DataFrame, *, periods_per_year: float = _DAYS_PER_YEAR / PERIOD_DAYS
) -> float:
    """Annualised Sharpe of the CLOSED period ledger (open MTM rows excluded)."""
    if book is None or book.empty:
        return 0.0
    rets = book.loc[book["closed"], "ret_net"].to_numpy(dtype=float)
    if rets.size < 2:
        return 0.0
    sd = rets.std(ddof=1)
    if sd == 0:
        return 0.0
    return float(rets.mean() / sd * np.sqrt(periods_per_year))


def subperiod_sharpes(
    book: pd.DataFrame,
    n_splits: int = 3,
    *,
    periods_per_year: float = _DAYS_PER_YEAR / PERIOD_DAYS,
) -> list[float]:
    """Annualised Sharpe over ``n_splits`` chronological thirds of the closed ledger.
    Splits too thin to compute a Sharpe contribute 0.0 (kept, never dropped)."""
    closed = book.loc[book["closed"]].sort_values("rebalance_ts") if not book.empty else book
    if closed is None or closed.empty:
        return [0.0] * n_splits
    pos_chunks = np.array_split(np.arange(len(closed)), n_splits)  # split positions, not the frame
    return [
        annualized_sharpe(closed.iloc[c], periods_per_year=periods_per_year)
        for c in pos_chunks
    ]


# ── lake loader (read-only seam) ───────────────────────────────────────────────── #
def load_xs_panel(
    con, symbols: tuple[str, ...] = UNIVERSE_23, *, window: int = WINDOW_SETTLEMENTS
) -> dict[str, pd.DataFrame]:
    """Build the daily ``{symbol: (ts, close, funding_z)}`` panel from the frozen lake.

    funding-z is computed on the NATIVE 8h settlement series (so "180 settlements" is
    exact), taken as the last settlement-z of each UTC day; close is the last 4h close
    of each UTC day. Read-only on ``data/hermes_datalake.duckdb``.
    """
    panel: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        fr = con.execute(
            "SELECT ts, funding_rate FROM binance_funding_native WHERE symbol=? ORDER BY ts",
            [sym],
        ).fetchdf()
        if fr.empty:
            continue
        fr["funding_z"] = causal_funding_z(fr["funding_rate"], window).to_numpy()
        fr["date"] = pd.to_datetime(fr["ts"], unit="ms", utc=True).dt.floor("D")
        z_daily = fr.dropna(subset=["funding_z"]).groupby("date").last()[["funding_z"]]
        kl = con.execute(
            "SELECT ts, close FROM binance_klines_4h WHERE symbol=? ORDER BY ts", [sym]
        ).fetchdf()
        if kl.empty:
            continue
        kl["date"] = pd.to_datetime(kl["ts"], unit="ms", utc=True).dt.floor("D")
        c_daily = kl.groupby("date").last()[["close"]]
        merged = c_daily.join(z_daily, how="inner").dropna().reset_index()
        if merged.empty:
            continue
        merged["ts"] = merged["date"].astype("int64") // 10**6  # ns → ms
        panel[sym] = merged[["ts", "close", "funding_z"]].reset_index(drop=True)
    return panel


# ── runnable sweeps (loops over the unit-tested book + sharpe) ──────────────────── #
def param_neighborhood(
    panel: dict[str, pd.DataFrame],
    *,
    cadences: tuple[int, ...] = (7, 10, 14, 21),
    n_terciles_list: tuple[int, ...] = (3, 4, 5),
    fee_bps: float = 7.0,
    slippage_bps: float = 3.0,
    invert: bool = True,
) -> list[dict]:
    """Grid of (cadence, quantile-count) → annualised Sharpe (in-sample)."""
    out: list[dict] = []
    for cad in cadences:
        for nt in n_terciles_list:
            book = xs_reversion_book(
                panel, period_days=cad, fee_bps=fee_bps, slippage_bps=slippage_bps,
                invert=invert, n_terciles=nt,
            )
            closed = int(book["closed"].sum()) if not book.empty else 0
            out.append(
                {
                    "cadence_days": cad,
                    "n_quantiles": nt,
                    "n_closed": closed,
                    "ann_sharpe": annualized_sharpe(
                        book, periods_per_year=_periods_per_year(cad)
                    ),
                }
            )
    return out


def loo_instruments(
    panel: dict[str, pd.DataFrame],
    *,
    period_days: int = PERIOD_DAYS,
    fee_bps: float = 7.0,
    slippage_bps: float = 3.0,
    invert: bool = True,
    n_terciles: int = 3,
) -> list[dict]:
    """Leave-one-instrument-out annualised Sharpe — is the edge carried by one name?"""
    ppy = _periods_per_year(period_days)
    base = annualized_sharpe(
        xs_reversion_book(
            panel, period_days=period_days, fee_bps=fee_bps, slippage_bps=slippage_bps,
            invert=invert, n_terciles=n_terciles,
        ),
        periods_per_year=ppy,
    )
    out: list[dict] = []
    for drop in panel:
        sub = {s: df for s, df in panel.items() if s != drop}
        sh = annualized_sharpe(
            xs_reversion_book(
                sub, period_days=period_days, fee_bps=fee_bps, slippage_bps=slippage_bps,
                invert=invert, n_terciles=n_terciles,
            ),
            periods_per_year=ppy,
        )
        out.append({"dropped": drop, "ann_sharpe": sh, "delta_vs_base": sh - base})
    return out
