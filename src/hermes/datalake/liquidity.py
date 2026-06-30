"""Liquidity / H1 axis (Stream 0-β, Deliverable C) — make the carry-vs-liquidity
regression CALCULABLE, and nothing more.

Boucle β's H1 hypothesis is that the realized funding-APR is a DECREASING function of
liquidity (carry persists where it cannot be arbitraged — the illiquid tail). To test
it, β needs a causal, per-market, per-bar liquidity metric joinable to the carry grid.

THE METRIC: dollar-notional volume = ``volume * close``. Base-asset ``volume`` lives in
every klines view but isn't cross-comparable (1 BTC != 1000 DOGE); multiplying by the
bar close yields a USD-notional that IS comparable across markets. It is the only
liquidity metric cleanly acquirable per-bar across BOTH venues — CCXT and HL candles
drop quote-volume and trade-count (an honest limitation, documented in COVERAGE). HL's
native ``dayNtlVlm`` / ``impactPxs`` are richer but snapshot-only (captured by the HL
exotic source's ctx view), not a per-bar history.

CAUSALITY: ``dollar_volume`` is row-local (no look-ahead possible). ``trail_dollar_volume``
is a strictly BACKWARD rolling median (a smoother regressor); it is the L0 leak-guard
target — see the test module's test #1. This module computes data only — NO regression,
NO strategy (that is Boucle β). Reuses ``annualize_funding`` and the symbol-format
guards from the alpha machinery; rebuilds nothing.
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from hermes.datalake.carry_join import annualize_funding
from hermes.research.s10_provenance import assert_joinable

__all__ = [
    "attach_dollar_volume",
    "attach_trailing_liquidity",
    "funding_vs_liquidity",
    "funding_vs_liquidity_from_lake",
]

H1_COLS = ["ts", "symbol", "funding_apr", "dollar_volume"]


# --------------------------------------------------------------------------- #
# the per-bar liquidity metric (row-local -> trivially causal)
# --------------------------------------------------------------------------- #
def attach_dollar_volume(
    frame: pd.DataFrame,
    *,
    vol_col: str = "volume",
    price_col: str = "close",
    out: str = "dollar_volume",
) -> pd.DataFrame:
    """Add ``dollar_volume = volume * close`` — the per-bar USD-notional liquidity
    proxy. Row-local: a bar's value depends only on that bar, so no look-ahead is
    possible (proven by the L0 guard in tests)."""
    df = frame.copy()
    df[out] = df[vol_col].astype("float64") * df[price_col].astype("float64")
    return df


# --------------------------------------------------------------------------- #
# a smoother, still-causal liquidity regressor (the L0 leak-guard target)
# --------------------------------------------------------------------------- #
def attach_trailing_liquidity(
    frame: pd.DataFrame,
    *,
    lookback_bars: int,
    col: str = "dollar_volume",
    out: str = "trail_dollar_volume",
    symbol_col: str = "symbol",
    ts_col: str = "ts",
) -> pd.DataFrame:
    """Add a per-symbol BACKWARD rolling-median of ``col`` over ``lookback_bars``.

    Strictly causal: bar ``t`` aggregates only ``[t-lookback+1 .. t]`` (``min_periods=1``
    so early bars are partial, never NaN-padded with the future). Row ORDER is preserved
    (the rolling result is realigned by index), so the L0 leak guard — which compares
    past rows position-wise — holds.
    """
    df = frame.copy()
    rolled = (
        df.sort_values([symbol_col, ts_col])
        .groupby(symbol_col)[col]
        .rolling(window=lookback_bars, min_periods=1)
        .median()
        .reset_index(level=0, drop=True)
    )
    df[out] = rolled  # index-aligned -> original row order preserved
    return df


# --------------------------------------------------------------------------- #
# the H1 join recipe: [ts, symbol, funding_apr, dollar_volume]
# --------------------------------------------------------------------------- #
def funding_vs_liquidity(
    klines: pd.DataFrame,
    funding: pd.DataFrame,
    *,
    funding_interval_hours: float,
    ts_col: str = "ts",
    symbol_col: str = "symbol",
) -> pd.DataFrame:
    """Join a klines frame (carrying ``volume`` + ``close``) with a funding frame
    (carrying ``funding_rate``) on ``(ts, symbol)`` into the H1-ready frame
    ``[ts, symbol, funding_apr, dollar_volume, ...]``.

    ``funding_rate`` is annualized to ``funding_apr`` on the common APR basis
    (``annualize_funding``; HL 1h -> x8760, Binance 8h -> x1095, 4h -> x2190). Raises
    :class:`ProvenanceError` if the two ``symbol`` spellings differ (a cross-format join
    silently yields zero rows — the P2 guardrail). Inner join: only bars on both legs.
    """
    assert_joinable(klines[symbol_col].tolist(), funding[symbol_col].tolist())
    k = attach_dollar_volume(klines)
    f = funding.copy()
    f["funding_apr"] = annualize_funding(f["funding_rate"].astype("float64"), funding_interval_hours)
    merged = k.merge(
        f[[ts_col, symbol_col, "funding_apr"]], on=[ts_col, symbol_col], how="inner"
    )
    rest = [c for c in merged.columns if c not in H1_COLS]
    return (
        merged[H1_COLS + rest].sort_values([symbol_col, ts_col]).reset_index(drop=True)
    )


def funding_vs_liquidity_from_lake(
    con: Any,
    *,
    view: str,
    funding_interval_hours: float,
    symbol: str | None = None,
) -> pd.DataFrame:
    """Build the H1 frame from a single ``*_funding_on_4h``-style lake view that carries
    BOTH ``volume`` and ``funding_rate`` on one grid (read-only).

    Parameterized by ``view`` so it serves the tail (``binance_tail_funding_on_4h`` /
    ``hyperliquid_exotic_funding_on_4h``) AND the majors. ``symbol`` filters one market;
    ``None`` returns the whole view. Returns ``[ts, symbol, funding_apr, dollar_volume, ...]``.
    """
    where = f" WHERE symbol = '{symbol}'" if symbol else ""
    df = con.execute(
        f"SELECT ts, symbol, close, volume, funding_rate FROM {view}{where}"
    ).df()
    df["funding_apr"] = annualize_funding(df["funding_rate"].astype("float64"), funding_interval_hours)
    df = attach_dollar_volume(df)
    rest = [c for c in df.columns if c not in H1_COLS]
    return df[H1_COLS + rest].sort_values(["symbol", "ts"]).reset_index(drop=True)
