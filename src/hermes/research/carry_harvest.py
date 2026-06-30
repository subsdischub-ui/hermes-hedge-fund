"""Boucle-alpha carry-harvest backtester — a funding-SETTLEMENT accumulator.

Built ON ``hermes.datalake.carry_join`` (reused, not rebuilt) and mirrors the S3 paper
deploy target (``execution/paper.py``: SHORT perp + LONG spot, accrue per settlement,
fees per leg). It is NOT a price-return engine: the dominant P&L is banked funding;
the delta-neutral basis term (``spot_ret - perp_ret``) is near-zero on average but is a
real TAIL risk (basis blowout), so it is tracked explicitly, never assumed zero.

TWO correctness guards dominate (and are the first tests):

  #1  L0 leak — the trailing-funding-APR builder is causal (backward window);
      ``carry_join.assert_no_lookahead`` catches a forward window.
  #2  NO DOUBLE-COUNT — ``binance_funding_on_4h.funding_rate`` backward-CARRIES the 8h
      settled rate across a *variable* number of 4h bars (and misses ~43% of native ts),
      so summing it per bar inflates funding ~2x. ``binance_funding_native`` is the
      ground truth: ``attribute_funding`` buckets each settlement into exactly ONE 4h bar
      so ``sum(attributed) == sum(native)`` — lossless, no inflation.

All functions are pure (no I/O) except the ``*_from_lake`` loaders, which read the
frozen DuckDB read-only. No side effects at import.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from hermes.datalake import carry_join as cj

__all__ = [
    "BARS_PER_DAY_4H",
    "BARS_PER_YEAR_4H",
    "MONTHLY_BARS",
    "WEEKLY_BARS",
    "CarryConfig",
    "add_trailing_funding_apr",
    "arm_schedule",
    "attribute_funding",
    "book_returns",
    "build_book",
    "cash_and_carry_stream",
    "cross_venue_stream",
    "gated_net_returns",
    "regime_split",
]

# 4h grid arithmetic. 6 bars/day; a year is 2190 bars (matches carry_join's 4h annualize).
BARS_PER_DAY_4H = 6
WEEKLY_BARS = 7 * BARS_PER_DAY_4H      # 42
MONTHLY_BARS = 30 * BARS_PER_DAY_4H    # 180
BARS_PER_YEAR_4H = 365 * BARS_PER_DAY_4H  # 2190


@dataclass(frozen=True)
class CarryConfig:
    """One pre-registered grid cell for the cash-and-carry (CC) family."""

    tau_apr: float          # arm a coin when its trailing funding-APR >= tau_apr
    cadence_bars: int       # re-decide arm/disarm only every cadence_bars (anti-whipsaw)
    lookback_bars: int      # trailing window for the funding-APR estimate
    round_trip_cost: float  # fractional open+close cost (both legs, both sides)


# =========================================================================== #
# #2 — funding attribution: native settlements onto the 4h grid, no double-count
# =========================================================================== #
def attribute_funding(
    grid_ts: np.ndarray, native_ts: np.ndarray, native_rate: np.ndarray
) -> np.ndarray:
    """Bucket each native settlement into the 4h grid bar that CONTAINS it, sum per bar.

    Bar ``i`` spans ``[grid_ts[i], grid_ts[i+1])``; a settlement at ``native_ts`` lands in
    the bar whose open is the greatest ``grid_ts[i] <= native_ts``. Each settlement is
    counted exactly once → ``sum(out) == sum(native_rate)`` for settlements within the grid
    span (settlements before ``grid_ts[0]`` are dropped). Causal: a settlement is banked at
    the bar in which it actually settles, never earlier.
    """
    grid = np.asarray(grid_ts)
    n_ts = np.asarray(native_ts)
    rate = np.asarray(native_rate, dtype=float)
    out = np.zeros(grid.shape[0], dtype=float)
    if n_ts.size == 0:
        return out
    bucket = np.searchsorted(grid, n_ts, side="right") - 1
    valid = bucket >= 0
    np.add.at(out, bucket[valid], rate[valid])
    return out


# =========================================================================== #
# #1 — trailing funding APR (causal backward window)
# =========================================================================== #
def add_trailing_funding_apr(
    frame: pd.DataFrame,
    lookback_bars: int,
    *,
    bars_per_year: int = BARS_PER_YEAR_4H,
    col: str = "funding",
    out: str = "trail_apr",
) -> pd.DataFrame:
    """Add a causal trailing annualized-funding column.

    ``trail_apr[t] = sum(funding over the last ``lookback_bars`` bars, inclusive of t)
    * (bars_per_year / lookback_bars)``. Backward window only — perturbing a future bar
    cannot change an earlier ``trail_apr`` (passes ``assert_no_lookahead``).
    """
    df = frame.copy()
    rolled = df[col].rolling(lookback_bars, min_periods=1).sum()
    df[out] = rolled * (bars_per_year / float(lookback_bars))
    return df


# =========================================================================== #
# on/off arming — cadence-gated, causal (anti-whipsaw)
# =========================================================================== #
def arm_schedule(
    trail_apr: np.ndarray, tau: float, cadence_bars: int, warmup_bars: int = 0
) -> np.ndarray:
    """Boolean exposure mask: re-decide armed state only at cadence boundaries.

    A decision made at a boundary uses the trailing-APR known AT that bar and governs the
    NEXT block (the boundary bar itself still carries the prior decision) → strictly causal,
    and exposure only ever changes on a cadence boundary (the whipsaw guard).
    """
    trail = np.asarray(trail_apr, dtype=float)
    n = trail.shape[0]
    armed = np.zeros(n, dtype=bool)
    current = False
    for t in range(n):
        armed[t] = current
        if t >= warmup_bars and (t - warmup_bars) % cadence_bars == 0:
            current = bool(trail[t] >= tau)
    return armed


# =========================================================================== #
# gated net returns — cost charged once per open + once per close, never per bar
# =========================================================================== #
def gated_net_returns(
    basis_ret: np.ndarray,
    funding: np.ndarray,
    armed: np.ndarray,
    round_trip_cost: float,
) -> np.ndarray:
    """Per-bar net return of the gated carry leg.

    While armed: ``basis_ret + funding`` (funding already attributed once per settlement).
    A ``round_trip_cost`` (both legs, entry+exit) is split half at the open bar (armed
    flips False→True) and half at the close bar (True→False). A position still armed at the
    final bar pays its close there (you must exit eventually). Costs are event-driven, never
    accrued per bar — the classic carry over-cost/under-cost trap.
    """
    basis = np.asarray(basis_ret, dtype=float)
    fund = np.asarray(funding, dtype=float)
    arm = np.asarray(armed, dtype=bool)
    n = arm.shape[0]
    out = np.zeros(n, dtype=float)
    half = round_trip_cost / 2.0
    for t in range(n):
        prev = bool(arm[t - 1]) if t > 0 else False
        if arm[t]:
            out[t] = basis[t] + fund[t]
            if not prev:               # open
                out[t] -= half
        elif prev:                     # close on the (now flat) bar
            out[t] -= half
    if n > 0 and arm[-1]:              # force close if never exited
        out[-1] -= half
    return out


# =========================================================================== #
# book aggregation — equal weight among armed coins, flat (0) when none armed
# =========================================================================== #
def book_returns(
    per_coin_net: dict[str, np.ndarray], per_coin_armed: dict[str, np.ndarray]
) -> np.ndarray:
    """Equal-weight the coins armed each bar; 0 (in cash) when none is armed.

    Full deployment across whatever is armed — ``univers = livre`` (harvest every coin
    whose carry is on). Concentrated when few coins arm (honest: that is what you'd hold).
    """
    coins = list(per_coin_net)
    if not coins:
        return np.zeros(0)
    net = np.vstack([np.asarray(per_coin_net[c], dtype=float) for c in coins])
    arm = np.vstack([np.asarray(per_coin_armed[c], dtype=bool) for c in coins])
    active = arm.sum(axis=0)
    contrib = np.where(arm, net, 0.0).sum(axis=0)
    return np.divide(contrib, active, out=np.zeros(net.shape[1]), where=active > 0)


# =========================================================================== #
# regime decomposition — Axis B core: net conditional on the funding sign
# =========================================================================== #
def regime_split(book_ret: np.ndarray, funding_sign: np.ndarray) -> dict:
    """Split book returns by the sign of the funding regime (the decisive Axis-B test)."""
    r = np.asarray(book_ret, dtype=float)
    s = np.asarray(funding_sign, dtype=float)
    pos = s > 0
    neg = s < 0
    return {
        "mean_pos": float(r[pos].mean()) if pos.any() else 0.0,
        "mean_neg": float(r[neg].mean()) if neg.any() else 0.0,
        "frac_pos_funding": float(pos.mean()) if s.size else 0.0,
    }


# =========================================================================== #
# lake loaders — per-coin streams (read-only)
# =========================================================================== #
def _native_funding_raw(
    con, symbol_slash: str, *, view: str = "binance_funding_native"
) -> tuple[np.ndarray, np.ndarray]:
    df = con.execute(
        f"SELECT ts, funding_rate FROM {view} "
        "WHERE symbol = ? ORDER BY ts",
        [symbol_slash],
    ).df()
    return df["ts"].to_numpy(), df["funding_rate"].to_numpy(dtype=float)


def cash_and_carry_stream(
    con,
    symbol_slash: str,
    *,
    spot_view: str = "binance_spot_klines_4h",
    perp_view: str = "binance_funding_on_4h",
    native_view: str = "binance_funding_native",
) -> pd.DataFrame:
    """Per-coin CC stream on the 4h grid: ``[ts, basis_ret, funding]``.

    ``basis_ret = spot_ret - perp_ret`` (delta-neutral residual, reused from carry_join);
    ``funding`` = native settlements attributed once each onto the grid (NO double-count).
    The view names default to the 23-major spellings; pass the Boucle-β tail spellings
    (``binance_tail_spot_klines_4h`` / ``binance_tail_funding_on_4h`` /
    ``binance_tail_funding_native``) to run the harvest on the illiquid tail unchanged.
    """
    aligned = cj.carry_returns(
        cj.cash_and_carry_from_lake(
            con, symbol=symbol_slash, spot_view=spot_view, perp_view=perp_view
        )
    )
    grid_ts = aligned["ts"].to_numpy()
    n_ts, n_rate = _native_funding_raw(con, symbol_slash, view=native_view)
    funding = attribute_funding(grid_ts, n_ts, n_rate)
    basis_ret = (aligned["spot_ret"] - aligned["perp_ret"]).fillna(0.0).to_numpy()
    return pd.DataFrame({"ts": grid_ts, "basis_ret": basis_ret, "funding": funding})


def cross_venue_stream(
    con,
    coin_bare: str,
    *,
    binance_interval_hours: float = 8.0,
    hl_klines_view: str = "hyperliquid_klines",
    binance_perp_view: str = "binance_funding_on_4h",
    binance_native_view: str = "binance_funding_native",
    hl_funding_view: str = "hyperliquid_funding_1h",
) -> pd.DataFrame:
    """Per-coin XV stream on the HL 4h klines grid: ``[ts, basis_ret, funding]``.

    Position: SHORT Binance perp + LONG HL perp. ``basis_ret`` = the inter-venue price
    residual ``hl_perp_ret - binance_perp_ret`` (delta-neutral across venues). ``funding``
    = per-bar harvested differential = Binance funding received minus HL funding paid,
    attributed from each venue's native settlements (Binance native 8h; HL hourly), bucketed
    once onto the shared grid. Window is bounded by HL klines depth (~2024-03 on).
    The view names default to the major spellings; pass the Boucle-β exotic spellings
    (``hyperliquid_exotic_klines`` / ``binance_tail_funding_on_4h`` /
    ``binance_tail_funding_native`` / ``hyperliquid_exotic_funding_1h``) for a tail coin
    listed on both venues.
    """
    hlk = con.execute(
        f"SELECT ts, close FROM {hl_klines_view} WHERE symbol = ? ORDER BY ts",
        [coin_bare],
    ).df()
    bperp = con.execute(
        f"SELECT ts, close FROM {binance_perp_view} WHERE symbol = ? ORDER BY ts",
        [f"{coin_bare}/USDT"],
    ).df()
    grid = hlk.merge(bperp, on="ts", how="inner", suffixes=("_hl", "_bz")).sort_values("ts")
    grid_ts = grid["ts"].to_numpy()
    hl_ret = grid["close_hl"].pct_change(fill_method=None).fillna(0.0).to_numpy()
    bz_ret = grid["close_bz"].pct_change(fill_method=None).fillna(0.0).to_numpy()
    basis_ret = hl_ret - bz_ret

    bz_ts, bz_rate = _native_funding_raw(con, f"{coin_bare}/USDT", view=binance_native_view)
    hl = con.execute(
        f"SELECT ts, funding_rate FROM {hl_funding_view} WHERE symbol = ? ORDER BY ts",
        [coin_bare],
    ).df()
    binance_recv = attribute_funding(grid_ts, bz_ts, bz_rate)            # short Binance -> receive
    hl_pay = attribute_funding(grid_ts, hl["ts"].to_numpy(), hl["funding_rate"].to_numpy(dtype=float))
    funding = binance_recv - hl_pay                                     # long HL -> pay HL funding
    return pd.DataFrame({"ts": grid_ts, "basis_ret": basis_ret, "funding": funding})


# =========================================================================== #
# book driver — run one grid cell over a universe of per-coin streams
# =========================================================================== #
def build_book(streams: dict[str, pd.DataFrame], config: CarryConfig) -> dict:
    """Run one pre-registered grid cell over a universe of per-coin streams.

    Per-coin signals (trailing-APR, arm schedule, gated net) are computed on each coin's
    OWN contiguous grid (so pre-listing has no spurious zero-funding dilution), then placed
    onto the UNION 4h grid (net 0 / unarmed where the coin is absent). The book equal-weights
    armed coins each bar. Returns the union ts, the book net-return series, per-coin net/armed
    on the union grid, and the per-bar book funding (for the regime split).
    """
    coins = list(streams)
    union = np.array(sorted({int(t) for s in streams.values() for t in s["ts"].tolist()}))
    n = union.shape[0]
    per_net: dict[str, np.ndarray] = {}
    per_arm: dict[str, np.ndarray] = {}
    funding_u: dict[str, np.ndarray] = {}
    for coin in coins:
        s = streams[coin].sort_values("ts").reset_index(drop=True)
        tdf = add_trailing_funding_apr(s, config.lookback_bars)
        armed = arm_schedule(
            tdf["trail_apr"].to_numpy(), config.tau_apr, config.cadence_bars,
            warmup_bars=config.lookback_bars,
        )
        net = gated_net_returns(
            s["basis_ret"].to_numpy(), s["funding"].to_numpy(), armed, config.round_trip_cost
        )
        idx = np.searchsorted(union, s["ts"].to_numpy())
        net_u = np.zeros(n)
        arm_u = np.zeros(n, dtype=bool)
        fnd_u = np.zeros(n)
        net_u[idx] = net
        arm_u[idx] = armed
        fnd_u[idx] = s["funding"].to_numpy()
        per_net[coin] = net_u
        per_arm[coin] = arm_u
        funding_u[coin] = fnd_u
    book = book_returns(per_net, per_arm)
    # per-bar book funding = total funding banked across ARMED coins (sign drives the regime)
    arm_mat = np.vstack([per_arm[c] for c in coins]) if coins else np.zeros((1, n), bool)
    fnd_mat = np.vstack([funding_u[c] for c in coins]) if coins else np.zeros((1, n))
    book_funding = np.where(arm_mat, fnd_mat, 0.0).sum(axis=0)
    return {
        "ts": union,
        "book_ret": book,
        "per_coin_net": per_net,
        "per_coin_armed": per_arm,
        "book_funding": book_funding,
    }
