"""TDD — Boucle β (illiquid-niche carry). Two hypotheses:

  H1  realized funding-APR as a function of liquidity (does carry rise where liquidity
      falls — i.e. where it cannot be arbitraged?). A DESCRIPTIVE regression over the
      ~94000x per-bar dollar-volume span (HL exotic -> Binance tail -> 23 majors).
  H2  THE KILLER TEST: does the NET carry survive the illiquid tail's realistic
      (liquidity-DEPENDENT) slippage and brutal basis tail? Net-carry by liquidity band.

This module reuses the alpha carry mechanic (carry_join / carry_harvest) and the S0-β
liquidity axis (datalake/liquidity) UNCHANGED, and adds: (§2) a view-name parameter so
the lake conveniences point at the *_tail / *_exotic views; (H1) the regression; (H2)
the liquidity-dependent cost model + net-by-band harvest. Nothing live, shadow size 0.

§4 discipline: the FIRST tests are leak-injection guards (carry_join.assert_no_lookahead),
and funding is ALWAYS sourced from *_funding_native (1 row = 1 settlement), NEVER summed
from the *_funding_on_4h backward-carry (the alpha 2x double-count trap).
"""

from __future__ import annotations

import pandas as pd
import pytest

from hermes.datalake import carry_join as cj
from hermes.research import carry_harvest as ch
from hermes.research import illiquid_carry as ic

FOUR_H_MS = 14_400_000
EIGHT_H_MS = 2 * FOUR_H_MS


# --------------------------------------------------------------------------- #
# in-memory lake mirroring the real *_tail / *_exotic view names (read-only)
# --------------------------------------------------------------------------- #
def _tail_spot(sym="ANKR/USDT"):
    return pd.DataFrame({
        "ts": [0, FOUR_H_MS, EIGHT_H_MS],
        "open": [100.0, 110.0, 121.0], "high": [100.0, 110.0, 121.0],
        "low": [100.0, 110.0, 121.0], "close": [100.0, 110.0, 121.0],  # +10%, +10%
        "volume": [10.0, 10.0, 10.0], "symbol": [sym] * 3,
    })


def _tail_perp_on_4h(sym="ANKR/USDT"):
    return pd.DataFrame({
        "ts": [0, FOUR_H_MS, EIGHT_H_MS],
        "open": [100.0, 105.0, 110.25], "high": [100.0, 105.0, 110.25],
        "low": [100.0, 105.0, 110.25], "close": [100.0, 105.0, 110.25],  # +5%, +5%
        "volume": [10.0, 10.0, 10.0], "funding_rate": [0.0001, 0.0002, 0.0003],
        "symbol": [sym] * 3,
    })


def _tail_native(sym="ANKR/USDT"):
    # native 8h settlements: one at t=0 (bar0), one at t=8h (bar2). Bar1 (4h) has none.
    return pd.DataFrame({
        "ts": [0, EIGHT_H_MS], "funding_rate": [0.001, 0.001], "symbol": [sym] * 2,
    })


def _exotic_klines(sym="RENDER"):  # HL exotic = BARE symbol
    return pd.DataFrame({
        "ts": [0, FOUR_H_MS, EIGHT_H_MS],
        "open": [10.0, 10.5, 11.0], "high": [10.0, 10.5, 11.0],
        "low": [10.0, 10.5, 11.0], "close": [10.0, 10.5, 11.0],
        "volume": [100.0, 100.0, 100.0], "symbol": [sym] * 3,
    })


def _exotic_funding_1h(sym="RENDER"):
    return pd.DataFrame({
        "ts": [0, 3_600_000, 2 * 3_600_000],
        "funding_rate": [0.00002, 0.00003, 0.00004], "symbol": [sym] * 3,
    })


def _exotic_perp_on_4h(sym="TRB"):  # HL exotic = BARE symbol; hourly funding pinned to 4h grid
    return pd.DataFrame({
        "ts": [0, FOUR_H_MS, EIGHT_H_MS],
        "open": [10.0, 10.5, 11.0], "high": [10.0, 10.5, 11.0],
        "low": [10.0, 10.5, 11.0], "close": [10.0, 10.5, 11.0],
        "volume": [50.0, 50.0, 50.0], "funding_rate": [0.0001, 0.0001, 0.0001],
        "symbol": [sym] * 3,
    })


def _beta_lake_con():
    """An in-memory DuckDB carrying the β tail/exotic views for ANKR (CC) + RENDER (XV)."""
    duckdb = pytest.importorskip("duckdb")
    con = duckdb.connect(":memory:")
    reg = {
        "binance_tail_spot_klines_4h": _tail_spot(),
        "binance_tail_funding_on_4h": pd.concat(
            [_tail_perp_on_4h("ANKR/USDT"), _tail_perp_on_4h("RENDER/USDT")], ignore_index=True),
        "binance_tail_funding_native": pd.concat(
            [_tail_native("ANKR/USDT"), _tail_native("RENDER/USDT")], ignore_index=True),
        "hyperliquid_exotic_klines": _exotic_klines("RENDER"),
        "hyperliquid_exotic_funding_1h": _exotic_funding_1h("RENDER"),
    }
    for name, frame in reg.items():
        con.register(f"_{name}_df", frame)
        con.execute(f"CREATE VIEW {name} AS SELECT * FROM _{name}_df")
    return con


def _major_lake_con():
    """An in-memory DuckDB carrying ONLY the canonical MAJOR view names (retro-compat)."""
    duckdb = pytest.importorskip("duckdb")
    con = duckdb.connect(":memory:")
    reg = {
        "binance_spot_klines_4h": _tail_spot("BTC/USDT"),
        "binance_funding_on_4h": _tail_perp_on_4h("BTC/USDT"),
        "binance_funding_native": _tail_native("BTC/USDT"),
    }
    for name, frame in reg.items():
        con.register(f"_{name}_df", frame)
        con.execute(f"CREATE VIEW {name} AS SELECT * FROM _{name}_df")
    return con


# =========================================================================== #
# §2 — VIEW-NAME EXTENSION: point the alpha lake conveniences at *_tail/*_exotic
# (the pure carry functions already run on any market; this is the read wiring)
# =========================================================================== #
def test_cash_and_carry_from_lake_runs_on_a_tail_market_via_view_params():
    con = _beta_lake_con()
    out = cj.cash_and_carry_from_lake(
        con, symbol="ANKR/USDT",
        spot_view="binance_tail_spot_klines_4h",
        perp_view="binance_tail_funding_on_4h",
    )
    assert list(out.columns) == ["ts", "symbol", "spot_close", "perp_close", "funding_rate"]
    assert out["symbol"].unique().tolist() == ["ANKR/USDT"]
    assert out["spot_close"].tolist() == [100.0, 110.0, 121.0]
    assert out["perp_close"].tolist() == [100.0, 105.0, 110.25]


def test_cash_and_carry_from_lake_defaults_to_major_views_retrocompat():
    con = _major_lake_con()  # NO tail views present; default must read the major names
    out = cj.cash_and_carry_from_lake(con, symbol="BTC/USDT")
    assert len(out) == 3
    assert out["spot_close"].tolist() == [100.0, 110.0, 121.0]


def test_cross_venue_from_lake_runs_on_exotic_via_view_params():
    con = _beta_lake_con()
    out = cj.cross_venue_from_lake(
        con, "RENDER", binance_interval_hours=8.0,
        binance_view="binance_tail_funding_native",
        hl_view="hyperliquid_exotic_funding_1h",
    )
    assert "apr_diff" in out.columns
    # RENDER native funding fixture = 0.001/settlement; 8h annualizes x1095 -> 1.095 APR
    assert abs(out["apr_binance"].iloc[0] - 0.001 * 1095) < 1e-12


def test_cash_and_carry_stream_runs_on_a_tail_market_via_view_params():
    con = _beta_lake_con()
    s = ch.cash_and_carry_stream(
        con, "ANKR/USDT",
        spot_view="binance_tail_spot_klines_4h",
        perp_view="binance_tail_funding_on_4h",
        native_view="binance_tail_funding_native",
    )
    assert {"ts", "basis_ret", "funding"}.issubset(s.columns)
    # native funding attributed once per settlement: 0.001 at bar0 + 0.001 at bar2
    assert abs(s["funding"].sum() - 0.002) < 1e-12


def test_cash_and_carry_stream_defaults_to_major_views_retrocompat():
    con = _major_lake_con()
    s = ch.cash_and_carry_stream(con, "BTC/USDT")
    assert {"ts", "basis_ret", "funding"}.issubset(s.columns)
    assert abs(s["funding"].sum() - 0.002) < 1e-12


def test_cross_venue_stream_runs_on_exotic_via_view_params():
    con = _beta_lake_con()
    s = ch.cross_venue_stream(
        con, "RENDER", binance_interval_hours=8.0,
        hl_klines_view="hyperliquid_exotic_klines",
        binance_perp_view="binance_tail_funding_on_4h",
        binance_native_view="binance_tail_funding_native",
        hl_funding_view="hyperliquid_exotic_funding_1h",
    )
    assert {"ts", "basis_ret", "funding"}.issubset(s.columns)
    assert len(s) == 3


# =========================================================================== #
# H1 — funding-APR vs liquidity regression (DESCRIPTIVE; no registry bump)
# =========================================================================== #
def test_regression_recovers_a_known_slope_on_log_liquidity():
    # funding_apr = 0.5 + 2.0*log10(dollar_volume), one bar/market, no noise
    frame = pd.DataFrame({
        "symbol": [f"M{i}" for i in range(6)],
        "funding_apr": [0.5 + 2.0 * k for k in range(1, 7)],
        "dollar_volume": [10.0 ** k for k in range(1, 7)],
    })
    r = ic.liquidity_funding_regression(frame, per_market=True)
    assert abs(r["slope"] - 2.0) < 1e-9
    assert abs(r["intercept"] - 0.5) < 1e-9
    assert r["n"] == 6
    assert r["r2"] > 0.999


def test_regression_detects_a_negative_slope_when_carry_falls_with_liquidity():
    # H1's PREDICTED shape: higher liquidity -> lower realized funding-APR
    frame = pd.DataFrame({
        "symbol": [f"M{i}" for i in range(6)],
        "funding_apr": [0.30, 0.24, 0.18, 0.12, 0.06, 0.00],
        "dollar_volume": [10.0 ** k for k in range(1, 7)],
    })
    r = ic.liquidity_funding_regression(frame, per_market=True)
    assert r["slope"] < 0
    assert r["t_stat"] < -5  # a clean, significant decreasing relationship


def test_regression_per_market_collapses_bars_to_one_point_each():
    # market A = 100 bars, B = 2 bars; per_market regresses across MARKETS (n=2), not bars
    a = pd.DataFrame({"symbol": ["A"] * 100, "funding_apr": [0.1] * 100, "dollar_volume": [1e6] * 100})
    b = pd.DataFrame({"symbol": ["B"] * 2, "funding_apr": [0.2] * 2, "dollar_volume": [1e4] * 2})
    r = ic.liquidity_funding_regression(pd.concat([a, b], ignore_index=True), per_market=True)
    assert r["n"] == 2


def test_regression_pooled_uses_every_bar():
    a = pd.DataFrame({"symbol": ["A"] * 100, "funding_apr": [0.1] * 100, "dollar_volume": [1e6] * 100})
    b = pd.DataFrame({"symbol": ["B"] * 2, "funding_apr": [0.2] * 2, "dollar_volume": [1e4] * 2})
    r = ic.liquidity_funding_regression(pd.concat([a, b], ignore_index=True), per_market=False)
    assert r["n"] == 102


def _h1_lake_con():
    """In-memory lake with the three *_funding_on_4h views (volume + funding_rate)."""
    duckdb = pytest.importorskip("duckdb")
    con = duckdb.connect(":memory:")
    reg = {
        "binance_funding_on_4h": _tail_perp_on_4h("BTC/USDT"),
        "binance_tail_funding_on_4h": _tail_perp_on_4h("ANKR/USDT"),
        "hyperliquid_exotic_funding_on_4h": _exotic_perp_on_4h("TRB"),
    }
    for name, frame in reg.items():
        con.register(f"_{name}_df", frame)
        con.execute(f"CREATE VIEW {name} AS SELECT * FROM _{name}_df")
    return con


def test_assemble_h1_frame_pools_three_venues_with_a_venue_tag():
    con = _h1_lake_con()
    frame = ic.assemble_h1_frame(con)
    assert {"ts", "symbol", "funding_apr", "dollar_volume", "venue"}.issubset(frame.columns)
    assert set(frame["venue"].unique()) == {"binance_major", "binance_tail", "hl_exotic"}
    # HL is annualized at 1h (x8760), Binance at 8h (x1095) -> distinct funding_apr scales
    hl_apr = frame.loc[frame["venue"] == "hl_exotic", "funding_apr"].iloc[0]
    assert abs(hl_apr - 0.0001 * 8760) < 1e-9


def test_assemble_h1_frame_windows_by_start_ts():
    con = _h1_lake_con()
    frame = ic.assemble_h1_frame(con, start_ts=FOUR_H_MS)  # drop the t=0 bar
    assert frame["ts"].min() >= FOUR_H_MS


def test_infer_funding_interval_from_native_settlement_spacing():
    # Binance moved many alts to 4h funding; the interval must be DATA-DERIVED, not assumed
    eight_h = np.array([0, 8, 16, 24, 32]) * 3_600_000
    four_h = np.array([0, 4, 8, 12, 16]) * 3_600_000
    one_h = np.array([0, 1, 2, 3, 4]) * 3_600_000
    assert ic.infer_funding_interval_hours(eight_h) == 8.0
    assert ic.infer_funding_interval_hours(four_h) == 4.0
    assert ic.infer_funding_interval_hours(one_h) == 1.0


def test_infer_funding_interval_is_robust_to_a_few_gaps():
    # a couple of missed settlements (gaps) must not move the MEDIAN spacing off 8h
    ts = np.array([0, 8, 16, 24, 48, 56]) * 3_600_000  # one 24h gap among 8h settles
    assert ic.infer_funding_interval_hours(ts) == 8.0


# =========================================================================== #
# H2 — liquidity-DEPENDENT slippage (the pre-registered cost model) + net harvest
#   round_trip_cost(dv) = clip(taker_floor, taker_floor + k*sqrt(Q/daily_dv), cap)
#   (see research_streams/BETA/H2_PREREG.md — frozen before the run)
# =========================================================================== #
import numpy as np  # noqa: E402  (kept local to the H2 section, mirrors module style)


# ---- L0 leak-injection is the FIRST guard for β's NEW causal builder (the cost) ---- #
def _liq_frame():
    return pd.DataFrame({
        "ts": [i * FOUR_H_MS for i in range(6)],
        "symbol": ["ANKR/USDT"] * 6,
        "dollar_volume": [1e6, 2e6, 5e5, 3e6, 1e6, 4e5],
    })


def test_leak_guard_passes_the_causal_trailing_slippage_builder():
    """Slippage built on the BACKWARD trailing-median liquidity is causal — perturbing a
    FUTURE bar's dollar-volume must not change any earlier bar's slippage (TEST #1)."""
    from hermes.datalake import liquidity as liq

    def builder(df: pd.DataFrame) -> pd.DataFrame:
        out = liq.attach_trailing_liquidity(df, lookback_bars=3)
        out["slippage"] = ic.liquidity_slippage(out["trail_dollar_volume"].to_numpy())
        return out

    cj.assert_no_lookahead(builder, _liq_frame(), value_cols=["slippage"])


def test_leak_guard_catches_a_forward_looking_slippage_builder():
    """A builder pricing slippage off NEXT bar's liquidity (shift(-1)) reads data[t+1]."""
    def leaky(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        fut = out.groupby("symbol")["dollar_volume"].shift(-1).bfill().to_numpy()
        out["slippage"] = ic.liquidity_slippage(fut)
        return out

    with pytest.raises(cj.LookaheadError):
        cj.assert_no_lookahead(leaky, _liq_frame(), value_cols=["slippage"])


def test_slippage_falls_to_the_taker_floor_at_high_liquidity():
    cost = ic.liquidity_slippage(np.array([1e9, 1e9]))  # very liquid per-bar
    assert np.allclose(cost, 0.0035, atol=2e-4)  # ~ taker floor (35 bps round-trip)


def test_slippage_rises_at_low_liquidity_and_is_capped_and_floored():
    cost = ic.liquidity_slippage(np.array([1e9, 5_000.0]))  # liquid vs ultra-illiquid
    assert cost[1] > cost[0]              # the illiquid market pays MORE
    assert cost[1] <= 0.025 + 1e-12       # capped at 250 bps round-trip
    assert cost[0] >= 0.0035 - 1e-12      # never below the taker floor


def test_slippage_is_monotone_non_increasing_in_liquidity():
    dv = np.array([1e4, 1e5, 1e6, 1e7, 1e8, 1e9])
    cost = ic.liquidity_slippage(dv, cap=0.10)  # high cap so monotonicity is visible
    assert np.all(np.diff(cost) <= 1e-15)


# ---- variable (per-bar) cost gated returns: charge slippage at the prevailing liquidity ---- #
def test_variable_cost_matches_alpha_flat_cost_when_constant():
    basis = np.zeros(10)
    funding = np.full(10, 0.001)
    armed = np.array([False, True, True, True, True, False, False, False, False, False])
    flat = ch.gated_net_returns(basis, funding, armed, 0.004)
    var = ic.gated_net_returns_variable_cost(basis, funding, armed, np.full(10, 0.004))
    assert np.allclose(flat, var)


def test_variable_cost_charges_more_when_entry_liquidity_is_worse():
    basis = np.zeros(10)
    funding = np.full(10, 0.001)
    armed = np.array([False, True, True, True, True, False, False, False, False, False])
    cheap = ic.gated_net_returns_variable_cost(basis, funding, armed, np.full(10, 0.002)).sum()
    dear = ic.gated_net_returns_variable_cost(basis, funding, armed, np.full(10, 0.020)).sum()
    assert dear < cheap  # higher prevailing slippage -> strictly lower net
