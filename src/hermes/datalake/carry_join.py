"""Carry-join mechanic (Stream 0-alpha §2/§3) — make delta-neutral P&L CALCULABLE.

This is the bridge the cash-and-carry harvest needs, and NOTHING MORE: read-only
data alignment + the pure P&L identity. It does NOT backtest a strategy, does NOT
size, does NOT trade — the harvest research is the NEXT mandate (Boucle alpha). It only
makes the calculation *possible*.

Two recipes (mandate §2 / §3):

  (a) CASH-AND-CARRY — align the spot leg (``binance_spot_klines_4h``) with the perp
      leg + causal settled funding (``binance_funding_on_4h``) on ``(ts, symbol)``,
      both in the slash ``BTC/USDT`` spelling, then apply the identity
          pnl_dn = spot_ret + (-perp_ret) + funding_collected - costs.
      SIGN CONVENTION (documented, overridable at the merge gate): the position is
      LONG spot + SHORT perp. When perp ``funding_rate > 0`` the SHORT receives
      funding, so ``funding_collected = +funding_rate`` per settled bar. Flip the
      sign if you model the opposite leg.

  (b) CROSS-VENUE — put Binance and Hyperliquid funding on a common APR basis
      (``annualize_funding``) and align causally (backward as-of) to expose the
      harvestable differential ``apr_diff`` (short the high-APR venue, long the low).

§4 DISCIPLINE — leak-injection is the FIRST guard: :func:`assert_no_lookahead` proves
a derived-column builder never reads ``data[t+1]`` (it perturbs the future row and
asserts every earlier output is unchanged), and it CATCHES a builder that does.

Pure functions over pandas; the ``*_from_lake`` helpers run the same alignment over
real DuckDB views (read-only). No side effects at import.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pandas as pd

from hermes.research.s10_provenance import assert_joinable, normalize_symbol

__all__ = [
    "HOURS_PER_YEAR",
    "LookaheadError",
    "align_cash_and_carry",
    "align_cross_venue",
    "annualize_funding",
    "assert_no_lookahead",
    "carry_returns",
    "cash_and_carry_from_lake",
    "cross_venue_from_lake",
    "delta_neutral_pnl",
]

# 24 * 365 — HL pays funding hourly, so its per-hour rate annualizes by this factor;
# Binance 8h -> /8 of this (1095), 4h -> /4 (2190). One basis makes venues comparable.
HOURS_PER_YEAR = 8760.0


class LookaheadError(AssertionError):
    """Raised when a derived-column builder leaks future data (reads ``data[t+1]``)."""


# =========================================================================== #
# §4 — leak-injection guard (TEST #1 machinery)
# =========================================================================== #
def assert_no_lookahead(
    builder: Callable[[pd.DataFrame], pd.DataFrame],
    frame: pd.DataFrame,
    *,
    value_cols: list[str],
    ts_col: str = "ts",
) -> None:
    """Assert ``builder`` is causal: perturbing the FUTURE row (max ``ts``) must not
    change any output row at an EARLIER ``ts``.

    A backward builder (``pct_change``, backward as-of) passes; a forward-leaking one
    (``shift(-1)``, future window) is caught and raises :class:`LookaheadError`.
    """
    frame = frame.reset_index(drop=True)
    base = builder(frame).reset_index(drop=True)

    max_ts = frame[ts_col].max()
    future = frame[ts_col] == max_ts
    poisoned = frame.copy()
    for col in poisoned.columns:
        if col == ts_col:
            continue
        if pd.api.types.is_numeric_dtype(poisoned[col]):
            poisoned.loc[future, col] = poisoned.loc[future, col].astype("float64") * 1000.0 + 7.0
    after = builder(poisoned).reset_index(drop=True)

    past = (frame[ts_col] < max_ts).to_numpy()
    pre = base.loc[past, value_cols].reset_index(drop=True)
    post = after.loc[past, value_cols].reset_index(drop=True)
    if not pre.equals(post):  # NaN-aware (pandas treats NaN==NaN as equal here)
        raise LookaheadError(
            f"look-ahead detected: perturbing the future row (ts={max_ts}) changed an "
            f"earlier output in {value_cols} — the builder reads data[t+1]."
        )


# =========================================================================== #
# (a) cash-and-carry alignment + the delta-neutral identity
# =========================================================================== #
def align_cash_and_carry(spot: pd.DataFrame, perp_funding: pd.DataFrame) -> pd.DataFrame:
    """Inner-join the spot leg and the perp+funding leg on ``(ts, symbol)``.

    Raises :class:`ProvenanceError` if the two ``symbol`` columns use different
    spellings (a cross-format join silently yields zero rows — P2 guardrail).
    Returns ``[ts, symbol, spot_close, perp_close, funding_rate]`` sorted by
    ``(symbol, ts)``. Only bars present on BOTH legs survive.
    """
    assert_joinable(spot["symbol"].tolist(), perp_funding["symbol"].tolist())
    left = spot[["ts", "symbol", "close"]].rename(columns={"close": "spot_close"})
    right = perp_funding[["ts", "symbol", "close", "funding_rate"]].rename(
        columns={"close": "perp_close"}
    )
    merged = left.merge(right, on=["ts", "symbol"], how="inner")
    return (
        merged[["ts", "symbol", "spot_close", "perp_close", "funding_rate"]]
        .sort_values(["symbol", "ts"])
        .reset_index(drop=True)
    )


def carry_returns(aligned: pd.DataFrame) -> pd.DataFrame:
    """Add per-symbol BACKWARD ``spot_ret`` / ``perp_ret`` (causal pct-change).

    ``fill_method=None`` keeps NaNs honest (the first bar of each symbol has no prior)
    and avoids forward-filling — strictly t-vs-(t-1), no look-ahead.
    """
    out = aligned.copy()
    out["spot_ret"] = out.groupby("symbol")["spot_close"].pct_change(fill_method=None)
    out["perp_ret"] = out.groupby("symbol")["perp_close"].pct_change(fill_method=None)
    return out


def delta_neutral_pnl(
    *,
    spot_ret: Any,
    perp_ret: Any,
    funding_collected: Any,
    costs: Any = 0.0,
) -> Any:
    """The mandate's delta-neutral identity (scalar or vectorized over Series):

        pnl_dn = spot_ret + (-perp_ret) + funding_collected - costs

    Long spot + short perp: a positive ``funding_collected`` is what the short
    receives when perp funding is positive (see module sign convention).
    """
    return spot_ret + (-perp_ret) + funding_collected - costs


# =========================================================================== #
# §3 — cross-venue funding on a common APR basis
# =========================================================================== #
def annualize_funding(rate: Any, interval_hours: float) -> Any:
    """Annualize a per-settlement funding rate to APR: ``rate * 8760 / interval_hours``.

    HL hourly (interval_hours=1) -> x8760 ; Binance 8h -> x1095 ; Binance 4h -> x2190.
    One basis makes the two venues' funding directly comparable.
    """
    return rate * (HOURS_PER_YEAR / float(interval_hours))


def align_cross_venue(
    binance_funding: pd.DataFrame,
    hl_funding: pd.DataFrame,
    *,
    binance_interval_hours: float,
    hl_interval_hours: float = 1.0,
) -> pd.DataFrame:
    """Align Binance vs Hyperliquid funding on a common APR basis, causally.

    Normalizes both ``symbol`` columns to BARE (HL is bare; Binance is slash),
    annualizes each side, then backward as-of joins HL's last-known APR onto each
    Binance settlement ``ts`` (per symbol). Exposes ``apr_binance``, ``apr_hl`` and
    the harvestable ``apr_diff = apr_binance - apr_hl``. Backward as-of = no leak:
    a Binance ts sees only HL funding settled at or before it (NaN if none yet).
    """
    b = binance_funding.copy()
    b["symbol"] = b["symbol"].map(lambda s: normalize_symbol(s, "bare"))
    b["apr_binance"] = annualize_funding(b["funding_rate"].astype("float64"), binance_interval_hours)
    b = b.sort_values("ts").reset_index(drop=True)

    h = hl_funding.copy()
    h["symbol"] = h["symbol"].map(lambda s: normalize_symbol(s, "bare"))
    h["apr_hl"] = annualize_funding(h["funding_rate"].astype("float64"), hl_interval_hours)
    h = h[["ts", "symbol", "apr_hl"]].sort_values("ts").reset_index(drop=True)

    out = pd.merge_asof(b, h, on="ts", by="symbol", direction="backward")
    out["apr_diff"] = out["apr_binance"] - out["apr_hl"]
    return out


# =========================================================================== #
# lake conveniences — same alignment over real DuckDB views (read-only)
# =========================================================================== #
def cash_and_carry_from_lake(con: Any, symbol: str | None = None) -> pd.DataFrame:
    """Read ``binance_spot_klines_4h`` ⋈ ``binance_funding_on_4h`` from a DuckDB
    connection and return the aligned cash-and-carry frame. Read-only.

    ``symbol`` (slash, e.g. 'BTC/USDT') filters one instrument; ``None`` aligns the
    whole book. Both views share the slash spelling, so the join is lossless.
    """
    spot_where = f" WHERE symbol = '{symbol}'" if symbol else ""
    spot = con.execute(
        f"SELECT ts, symbol, close FROM binance_spot_klines_4h{spot_where}"
    ).df()
    perp = con.execute(
        f"SELECT ts, symbol, close, funding_rate FROM binance_funding_on_4h{spot_where}"
    ).df()
    return align_cash_and_carry(spot, perp)


def cross_venue_from_lake(
    con: Any, coin: str, *, binance_interval_hours: float = 8.0
) -> pd.DataFrame:
    """Read Binance native funding + Hyperliquid hourly funding for ``coin`` (bare,
    e.g. 'BTC') from a DuckDB connection and return the cross-venue APR frame.

    Binance native funding is slash-spelled (``BTC/USDT``); HL is bare (``BTC``).
    ``align_cross_venue`` normalizes both to bare before the as-of join. Read-only.
    """
    binance = con.execute(
        "SELECT ts, funding_rate, symbol FROM binance_funding_native "
        f"WHERE symbol = '{coin}/USDT'"
    ).df()
    hl = con.execute(
        "SELECT ts, funding_rate, symbol FROM hyperliquid_funding_1h "
        f"WHERE symbol = '{coin}'"
    ).df()
    return align_cross_venue(binance, hl, binance_interval_hours=binance_interval_hours)
