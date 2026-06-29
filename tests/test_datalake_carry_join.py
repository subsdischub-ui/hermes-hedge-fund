"""TDD — carry-join mechanic (Stream 0-alpha §2/§3): make delta-neutral P&L CALCULABLE.

NOT a strategy and NOT a backtest — just the read-only data alignment + the pure
P&L identity, so the next mandate (Boucle alpha) can research the harvest. Two recipes:

  (a) cash-and-carry: align spot + perp + settled-funding on (ts, symbol), then
      pnl_dn = spot_ret + (-perp_ret) + funding_collected - costs   [the §2 formula]
  (b) cross-venue:    annualize Binance vs Hyperliquid funding to a common APR basis
      and align causally (as-of) to expose the harvestable differential.

§4 discipline: the FIRST test is leak-injection — a builder that reads data[t+1]
MUST be flagged. ``assert_no_lookahead`` is that guard (reusable), proven to PASS the
causal builder AND CATCH a deliberately leaky one.
"""

from __future__ import annotations

import math

import pandas as pd
import pytest

from hermes.datalake import carry_join as cj
from hermes.research.s10_provenance import ProvenanceError

# --------------------------------------------------------------------------- #
# fixtures — tiny spot / perp+funding frames on a shared 4h grid, slash symbols
# --------------------------------------------------------------------------- #
FOUR_H_MS = 14_400_000


def _spot_frame():
    # binance_spot_klines_4h shape (subset): ts, close, symbol (slash)
    return pd.DataFrame({
        "ts": [0, FOUR_H_MS, 2 * FOUR_H_MS],
        "open": [100.0, 101.0, 102.0],
        "high": [100.0, 101.0, 102.0],
        "low": [100.0, 101.0, 102.0],
        "close": [100.0, 110.0, 121.0],     # +10%, +10%
        "volume": [1.0, 1.0, 1.0],
        "symbol": ["BTC/USDT"] * 3,
    })


def _perp_funding_frame():
    # binance_funding_on_4h shape (subset): ts, close, funding_rate, symbol (slash)
    return pd.DataFrame({
        "ts": [0, FOUR_H_MS, 2 * FOUR_H_MS],
        "close": [100.0, 105.0, 110.25],    # +5%, +5% (perp leg moves less here)
        "funding_rate": [0.0001, 0.0002, 0.0003],
        "symbol": ["BTC/USDT"] * 3,
    })


# =========================================================================== #
# §4 — LEAK-INJECTION GUARD IS TEST #1
# =========================================================================== #
def test_leak_guard_passes_the_causal_carry_returns_builder():
    """carry_returns uses pct_change (t vs t-1) — strictly backward. Perturbing a
    FUTURE row must not change any earlier row's returns."""
    aligned = cj.align_cash_and_carry(_spot_frame(), _perp_funding_frame())
    cj.assert_no_lookahead(cj.carry_returns, aligned, value_cols=["spot_ret", "perp_ret"])


def test_leak_guard_catches_a_builder_that_reads_the_future():
    """A builder using shift(-1) reads data[t+1]; the guard MUST flag it."""
    def leaky(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["spot_ret"] = out.groupby("symbol")["spot_close"].shift(-1)  # FUTURE leak
        out["perp_ret"] = 0.0
        return out

    aligned = cj.align_cash_and_carry(_spot_frame(), _perp_funding_frame())
    with pytest.raises(cj.LookaheadError):
        cj.assert_no_lookahead(leaky, aligned, value_cols=["spot_ret", "perp_ret"])


# =========================================================================== #
# (a) cash-and-carry alignment + the delta-neutral P&L identity
# =========================================================================== #
def test_align_cash_and_carry_joins_on_ts_and_symbol():
    out = cj.align_cash_and_carry(_spot_frame(), _perp_funding_frame())
    assert list(out.columns) == ["ts", "symbol", "spot_close", "perp_close", "funding_rate"]
    assert len(out) == 3
    assert out["spot_close"].tolist() == [100.0, 110.0, 121.0]
    assert out["perp_close"].tolist() == [100.0, 105.0, 110.25]


def test_align_cash_and_carry_inner_join_drops_unmatched_bars():
    spot = _spot_frame()
    perp = _perp_funding_frame().iloc[:2]  # perp missing the last bar
    out = cj.align_cash_and_carry(spot, perp)
    assert len(out) == 2  # only bars present on BOTH legs survive


def test_align_cash_and_carry_raises_on_symbol_format_mismatch():
    spot = _spot_frame()
    perp = _perp_funding_frame()
    perp["symbol"] = "BTC"  # bare vs slash -> a silent zero-row join; must RAISE
    with pytest.raises(ProvenanceError):
        cj.align_cash_and_carry(spot, perp)


def test_carry_returns_are_per_symbol_backward_pct_change():
    aligned = cj.align_cash_and_carry(_spot_frame(), _perp_funding_frame())
    out = cj.carry_returns(aligned)
    assert math.isnan(out["spot_ret"].iloc[0])          # first bar has no prior
    assert abs(out["spot_ret"].iloc[1] - 0.10) < 1e-12  # 100 -> 110
    assert abs(out["perp_ret"].iloc[1] - 0.05) < 1e-12  # 100 -> 105


def test_delta_neutral_pnl_identity_matches_the_mandate_formula():
    # pnl_dn = spot_ret + (-perp_ret) + funding_collected - costs
    v = cj.delta_neutral_pnl(spot_ret=0.10, perp_ret=0.05, funding_collected=0.0002, costs=0.0010)
    assert abs(v - (0.10 - 0.05 + 0.0002 - 0.0010)) < 1e-15


def test_delta_neutral_pnl_is_vectorized_over_a_series():
    aligned = cj.carry_returns(cj.align_cash_and_carry(_spot_frame(), _perp_funding_frame()))
    pnl = cj.delta_neutral_pnl(
        spot_ret=aligned["spot_ret"], perp_ret=aligned["perp_ret"],
        funding_collected=aligned["funding_rate"], costs=0.0,
    )
    # bar 1: 0.10 - 0.05 + 0.0002 = 0.0502
    assert abs(pnl.iloc[1] - 0.0502) < 1e-12


# =========================================================================== #
# §3 — cross-venue funding: common APR basis + causal alignment
# =========================================================================== #
def test_annualize_funding_puts_venues_on_a_common_apr_basis():
    # HL hourly (1h) -> x8760 ; Binance 8h -> x1095 ; Binance 4h -> x2190
    assert abs(cj.annualize_funding(0.0001, 1) - 0.0001 * 8760) < 1e-12
    assert abs(cj.annualize_funding(0.0001, 8) - 0.0001 * 1095) < 1e-12
    assert abs(cj.annualize_funding(0.0001, 4) - 0.0001 * 2190) < 1e-12


def test_align_cross_venue_normalizes_symbols_and_exposes_the_differential():
    binance = pd.DataFrame({
        "ts": [0, 8 * 3_600_000],
        "funding_rate": [0.0001, 0.0001],
        "symbol": ["BTC/USDT", "BTC/USDT"],   # slash
    })
    hl = pd.DataFrame({
        "ts": [0, 3_600_000, 2 * 3_600_000],
        "funding_rate": [0.00002, 0.00003, 0.00004],
        "symbol": ["BTC", "BTC", "BTC"],       # bare
    })
    out = cj.align_cross_venue(binance, hl, binance_interval_hours=8)
    # annualized APRs on each side
    assert abs(out["apr_binance"].iloc[0] - 0.0001 * 1095) < 1e-12
    # causal as-of: at the binance ts=8h bar, HL's last-known funding is the ts=2h row
    assert abs(out["apr_hl"].iloc[1] - 0.00004 * 8760) < 1e-12
    assert "apr_diff" in out.columns
    assert abs(out["apr_diff"].iloc[1] - (out["apr_binance"].iloc[1] - out["apr_hl"].iloc[1])) < 1e-12


def test_align_cross_venue_is_causal_no_future_hl_funding_leaks():
    # binance ts=0 must NOT see an HL funding that only exists at ts>0 (backward as-of)
    binance = pd.DataFrame({"ts": [0], "funding_rate": [0.0001], "symbol": ["BTC/USDT"]})
    hl = pd.DataFrame({"ts": [3_600_000], "funding_rate": [0.5], "symbol": ["BTC"]})
    out = cj.align_cross_venue(binance, hl, binance_interval_hours=8)
    assert math.isnan(out["apr_hl"].iloc[0])  # no HL funding at/before ts=0 -> NaN, no leak


# =========================================================================== #
# lake conveniences — same alignment over real DuckDB views (in-memory, read-only)
# =========================================================================== #
def _lake_con():
    duckdb = pytest.importorskip("duckdb")
    con = duckdb.connect(":memory:")
    # mirror the real view names + the slash/bare symbol spellings of each
    con.register("spot_df", _spot_frame())
    con.register("perpf_df", _perp_funding_frame())
    con.execute("CREATE VIEW binance_spot_klines_4h AS SELECT * FROM spot_df")
    con.execute("CREATE VIEW binance_funding_on_4h AS SELECT * FROM perpf_df")
    bnative = pd.DataFrame({
        "ts": [0, 8 * 3_600_000], "funding_rate": [0.0001, 0.0001], "symbol": ["BTC/USDT"] * 2,
    })
    hlf = pd.DataFrame({
        "ts": [0, 3_600_000, 2 * 3_600_000], "funding_rate": [0.00002, 0.00003, 0.00004],
        "symbol": ["BTC"] * 3,
    })
    con.register("bnative_df", bnative)
    con.register("hlf_df", hlf)
    con.execute("CREATE VIEW binance_funding_native AS SELECT * FROM bnative_df")
    con.execute("CREATE VIEW hyperliquid_funding_1h AS SELECT * FROM hlf_df")
    return con


def test_cash_and_carry_from_lake_aligns_real_views():
    con = _lake_con()
    out = cj.cash_and_carry_from_lake(con, symbol="BTC/USDT")
    assert list(out.columns) == ["ts", "symbol", "spot_close", "perp_close", "funding_rate"]
    assert out["spot_close"].tolist() == [100.0, 110.0, 121.0]
    assert out["perp_close"].tolist() == [100.0, 105.0, 110.25]


def test_cross_venue_from_lake_uses_bare_vs_slash_correctly():
    con = _lake_con()
    out = cj.cross_venue_from_lake(con, "BTC", binance_interval_hours=8)
    assert "apr_diff" in out.columns
    assert abs(out["apr_binance"].iloc[0] - 0.0001 * 1095) < 1e-12
    # causal as-of at the 8h binance bar -> HL's last hourly funding (ts=2h)
    assert abs(out["apr_hl"].iloc[1] - 0.00004 * 8760) < 1e-12
