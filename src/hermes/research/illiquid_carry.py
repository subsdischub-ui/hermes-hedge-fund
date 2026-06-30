"""Boucle β — illiquid-niche carry (research, backtest + shadow only; nothing live).

The hypothesis chain (see research_streams/BETA/FINDINGS_BETA.md):

  H1  Is realized funding-APR a DECREASING function of liquidity? If the cash-and-carry
      is arbitraged where spot is liquid, the carry should persist in the illiquid tail
      that cannot be arbitraged. This module's H1 half MEASURES that relationship
      (a descriptive regression of funding_apr on log10(dollar_volume) across the
      ~94000x per-bar liquidity span: 23 majors -> Binance tail -> HL exotic). It is a
      MEASUREMENT, not a strategy — no registry trial is consumed by H1.

  H2  THE KILLER TEST: does the NET carry survive the illiquid tail's realistic
      (liquidity-DEPENDENT) slippage and brutal basis tail? (built in the H2 section,
      pre-registered, registry-counted.)

Reuses the alpha mechanic (carry_join / carry_harvest) and the S0-β liquidity axis
(datalake/liquidity) UNCHANGED. Funding for any ACCUMULATION is sourced from
*_funding_native (1 row = 1 settlement); the per-bar *_funding_on_4h view is used ONLY
for the H1 liquidity/level axis (its funding_rate is a backward-carried level — safe to
regress as a level, NEVER to sum, per the alpha 2x double-count lesson).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from hermes.datalake import liquidity as liq

__all__ = [
    "CAP_ROUND_TRIP",
    "H1_SPECS",
    "K_IMPACT",
    "Q_NOTIONAL",
    "TAKER_FLOOR",
    "assemble_h1_frame",
    "gated_net_returns_variable_cost",
    "infer_funding_interval_hours",
    "liquidity_funding_regression",
    "liquidity_slippage",
]


def infer_funding_interval_hours(native_ts: Any) -> float:
    """Median spacing (in hours) of consecutive NATIVE funding settlements.

    Binance moved many alts from 8h to 4h funding, so the settlement interval MUST be
    derived from the data, not assumed — annualizing a 4h-settling perp at the 8h factor
    understates its realized funding-APR by 2x. The median is robust to occasional gaps
    (missed settlements). Returns NaN if fewer than 2 settlements."""
    ts = np.sort(np.asarray(native_ts, dtype="float64"))
    if ts.size < 2:
        return float("nan")
    return float(np.median(np.diff(ts)) / 3_600_000.0)

# --------------------------------------------------------------------------- #
# H2 PRE-REGISTERED cost model (frozen 2026-06-30; see BETA/H2_PREREG.md).
# round_trip_cost(dv) = clip(taker_floor, taker_floor + k*sqrt(Q/daily_dv), cap)
# liquid major (~$100M+/day) -> ~taker floor; illiquid pays more via sqrt-impact;
# floored at the taker fee, capped so a micro-cap cannot diverge to infinity.
# --------------------------------------------------------------------------- #
TAKER_FLOOR = 0.0035   # 35 bps round-trip = alpha's TAKER cost; the liquid-major anchor
Q_NOTIONAL = 50_000.0  # $ notional per position; a SMALL book steelmans the small-book thesis
K_IMPACT = 0.025       # single central square-root-impact coefficient
CAP_ROUND_TRIP = 0.025  # 250 bps round-trip ceiling
_BARS_PER_DAY_4H = 6.0

# The full liquidity spectrum, one (venue, *_funding_on_4h view, funding interval) each.
# Binance perps settle 8h; HL exotic funding is hourly. funding_vs_liquidity_from_lake
# annualizes each to a common APR basis so the three venues are directly comparable.
# (venue, *_funding_on_4h level/volume view, *_funding_native view for interval derivation).
# The on_4h funding_rate is a backward-carried LEVEL (regress as level, never sum); the
# native view is read ONLY to DERIVE the per-symbol settlement interval — Binance moved
# many alts from 8h to 4h, so a hardcoded 8h understated their APR 2x (cf. CC audit).
H1_SPECS: list[tuple[str, str, str]] = [
    ("binance_major", "binance_funding_on_4h", "binance_funding_native"),
    ("binance_tail", "binance_tail_funding_on_4h", "binance_tail_funding_native"),
    ("hl_exotic", "hyperliquid_exotic_funding_on_4h", "hyperliquid_exotic_funding_1h"),
]


# --------------------------------------------------------------------------- #
# H1 data assembly — pool the full liquidity spectrum, tag the venue
# --------------------------------------------------------------------------- #
def assemble_h1_frame(
    con: Any,
    specs: list[tuple[str, str, float]] = H1_SPECS,
    *,
    start_ts: int | None = None,
) -> pd.DataFrame:
    """Pool the H1 frame ``[ts, symbol, funding_apr, dollar_volume, venue]`` across the
    full liquidity spectrum (read-only).

    For each ``(venue, on4h_view, native_view)`` spec, reuse
    ``liquidity.funding_vs_liquidity_from_lake`` (which annualizes funding to a common
    APR basis and attaches per-bar ``dollar_volume = volume*close``), tag the venue, and
    concatenate. ``start_ts`` (epoch ms) windows to a forward regime (e.g. 2025-01-01).
    """
    parts: list[pd.DataFrame] = []
    for venue, view, native_view in specs:
        # Derive each symbol's funding interval from its NATIVE settlement spacing (one row =
        # one real settlement), then annualize the on_4h LEVEL with that per-symbol interval.
        nat = con.execute(f"SELECT symbol, ts FROM {native_view}").df()
        intervals = {
            sym: infer_funding_interval_hours(g["ts"].to_numpy())
            for sym, g in nat.groupby("symbol")
        }
        for sym, interval in intervals.items():
            if interval != interval:  # NaN: <2 settlements, cannot annualize
                continue
            part = liq.funding_vs_liquidity_from_lake(
                con, view=view, symbol=sym, funding_interval_hours=interval
            )
            parts.append(part.assign(venue=venue))
    frame = pd.concat(parts, ignore_index=True)
    if start_ts is not None:
        frame = frame[frame["ts"] >= start_ts].reset_index(drop=True)
    return frame


# --------------------------------------------------------------------------- #
# the H1 regression: funding_apr ~ log10(dollar_volume)
# --------------------------------------------------------------------------- #
def _ols(x: np.ndarray, y: np.ndarray) -> dict:
    """Plain OLS y ~ x with slope t-stat and R². Returns NaNs if n<3 or x is degenerate."""
    n = x.shape[0]
    out = {"slope": math.nan, "intercept": math.nan, "r2": math.nan, "t_stat": math.nan, "n": int(n)}
    if n < 3:
        return out
    xbar, ybar = x.mean(), y.mean()
    sxx = float(((x - xbar) ** 2).sum())
    if sxx <= 0.0:
        return out
    slope = float(((x - xbar) * (y - ybar)).sum() / sxx)
    intercept = float(ybar - slope * xbar)
    resid = y - (intercept + slope * x)
    ss_res = float((resid**2).sum())
    ss_tot = float(((y - ybar) ** 2).sum())
    out["slope"] = slope
    out["intercept"] = intercept
    out["r2"] = (1.0 - ss_res / ss_tot) if ss_tot > 0 else math.nan
    # slope standard error -> t-stat (H0: slope==0)
    if n > 2:
        se_slope = math.sqrt((ss_res / (n - 2)) / sxx) if ss_res > 0 else 0.0
        out["t_stat"] = (slope / se_slope) if se_slope > 0 else (math.inf if slope != 0 else 0.0)
    return out


def liquidity_funding_regression(
    frame: pd.DataFrame,
    *,
    per_market: bool = True,
    funding_col: str = "funding_apr",
    liq_col: str = "dollar_volume",
    symbol_col: str = "symbol",
) -> dict:
    """Regress realized funding-APR on ``log10(dollar_volume)`` — the H1 measurement.

    ``per_market=True`` (default, the clean cross-sectional test) first collapses each
    market to its MEDIAN ``(funding_apr, dollar_volume)`` so the slope is not dominated
    by deep markets' bar counts; ``n`` = number of markets. ``per_market=False`` pools
    every bar (``n`` = bars; the t-stat is anti-conservative under within-market
    autocorrelation — use only as a cross-check). Returns
    ``{slope, intercept, r2, t_stat, n}``; a NEGATIVE slope is H1's predicted shape
    (carry rises as liquidity falls).
    """
    df = frame[[symbol_col, funding_col, liq_col]].dropna()
    df = df[df[liq_col] > 0]
    if per_market:
        df = df.groupby(symbol_col, as_index=False)[[funding_col, liq_col]].median()
    x = np.log10(df[liq_col].to_numpy(dtype="float64"))
    y = df[funding_col].to_numpy(dtype="float64")
    return _ols(x, y)


# --------------------------------------------------------------------------- #
# H2 — the pre-registered liquidity-dependent slippage (the killer cost model)
# --------------------------------------------------------------------------- #
def liquidity_slippage(
    dollar_volume: Any,
    *,
    q_notional: float = Q_NOTIONAL,
    taker_floor: float = TAKER_FLOOR,
    k_impact: float = K_IMPACT,
    cap: float = CAP_ROUND_TRIP,
    bars_per_day: float = _BARS_PER_DAY_4H,
) -> np.ndarray:
    """Per-bar round-trip slippage as a square-root-impact function of liquidity.

    ``cost = clip(taker_floor, taker_floor + k_impact*sqrt(q_notional/daily_dv), cap)``
    where ``daily_dv = dollar_volume * bars_per_day``. A liquid market (high ``dv``) pays
    ~``taker_floor`` (the q/dv participation -> 0); an illiquid market pays materially more
    via the impact term, floored at the taker fee and capped at ``cap`` so a micro-cap can
    never diverge. Row-local in ``dollar_volume`` -> trivially causal (pass the BACKWARD
    trailing-median liquidity for a smooth, still-causal cost — leak-guarded in tests)."""
    dv = np.asarray(dollar_volume, dtype="float64")
    daily_dv = dv * bars_per_day
    participation = np.divide(
        q_notional, daily_dv, out=np.full_like(daily_dv, np.inf), where=daily_dv > 0
    )
    cost = taker_floor + k_impact * np.sqrt(participation)
    return np.clip(cost, taker_floor, cap)


def gated_net_returns_variable_cost(
    basis_ret: Any, funding: Any, armed: Any, round_trip_cost: Any
) -> np.ndarray:
    """Per-bar net return of the gated carry leg with a PER-BAR (liquidity-dependent) cost.

    Identical event accounting to ``carry_harvest.gated_net_returns`` (half the round-trip
    charged at the open bar, half at the close bar; a position still armed at the end pays
    its close) — but ``round_trip_cost`` is an ARRAY, so the slippage paid is the one
    prevailing AT the open/close bar (you pay the tail's worse cost precisely when you
    transact there). Reduces to the alpha flat-cost engine when the array is constant."""
    basis = np.asarray(basis_ret, dtype="float64")
    fund = np.asarray(funding, dtype="float64")
    arm = np.asarray(armed, dtype=bool)
    cost = np.asarray(round_trip_cost, dtype="float64")
    n = arm.shape[0]
    out = np.zeros(n)
    for t in range(n):
        prev = bool(arm[t - 1]) if t > 0 else False
        if arm[t]:
            out[t] = basis[t] + fund[t]
            if not prev:                  # open: pay half the cost at this bar's liquidity
                out[t] -= cost[t] / 2.0
        elif prev:                        # close on the (now flat) bar
            out[t] -= cost[t] / 2.0
    if n > 0 and arm[-1]:                 # force close if never exited
        out[-1] -= cost[-1] / 2.0
    return out
