"""TDD for the liquidity / H1 axis (Stream 0-β, Deliverable C).

The decisive deliverable for Boucle β: a causal, per-market, per-bar liquidity metric
joinable to the carry grid, so β can regress funding-APR against liquidity (H1) directly.
Liquidity metric = dollar-notional volume (volume * close) — the only metric cleanly
acquirable per-bar across both venues (CCXT/HL klines drop quote-volume & trade-count).

TEST #1 is the L0 leak-injection guard (STREAM_COMMON §4): every derived column proves
it never reads data[t+1]. Reuses the project's assert_no_lookahead machinery.
"""
from __future__ import annotations

import pandas as pd
import pytest

from hermes.datalake import liquidity as liq
from hermes.datalake.carry_join import LookaheadError, assert_no_lookahead
from hermes.research.s10_provenance import ProvenanceError


def _klines(symbol="ZEC/USDT", n=8):
    base = 1_600_000_000_000
    return pd.DataFrame(
        {
            "ts": [base + i * 14_400_000 for i in range(n)],
            "symbol": [symbol] * n,
            "close": [10.0 + i for i in range(n)],
            "volume": [100.0 + 10 * i for i in range(n)],
        }
    )


def _funding(symbol="ZEC/USDT", n=8):
    base = 1_600_000_000_000
    return pd.DataFrame(
        {
            "ts": [base + i * 14_400_000 for i in range(n)],
            "symbol": [symbol] * n,
            "funding_rate": [0.0001 * (i + 1) for i in range(n)],
        }
    )


# --------------------------------------------------------------------------- #
# TEST #1 — L0 leak-injection guard (the FIRST guard, STREAM_COMMON §4)
# --------------------------------------------------------------------------- #
def test_leak_guard_passes_the_causal_trailing_liquidity_builder():
    """trailing dollar-volume is a BACKWARD rolling median — strictly causal."""
    frame = liq.attach_dollar_volume(_klines())

    def builder(df):
        return liq.attach_trailing_liquidity(df, lookback_bars=3)

    assert_no_lookahead(builder, frame, value_cols=["trail_dollar_volume"])


def test_leak_guard_catches_a_builder_that_reads_the_future():
    """A builder using shift(-1) reads data[t+1]; the guard MUST flag it."""
    frame = liq.attach_dollar_volume(_klines())

    def leaky(df):
        out = df.copy()
        out["trail_dollar_volume"] = out.groupby("symbol")["dollar_volume"].shift(-1)
        return out

    with pytest.raises(LookaheadError):
        assert_no_lookahead(leaky, frame, value_cols=["trail_dollar_volume"])


# --------------------------------------------------------------------------- #
# the metric itself
# --------------------------------------------------------------------------- #
def test_attach_dollar_volume_is_volume_times_close():
    out = liq.attach_dollar_volume(_klines())
    expected = _klines()["volume"] * _klines()["close"]
    assert list(out["dollar_volume"]) == pytest.approx(list(expected))


def test_attach_dollar_volume_is_row_local_no_lookahead():
    frame = _klines()
    assert_no_lookahead(liq.attach_dollar_volume, frame, value_cols=["dollar_volume"])


def test_trailing_liquidity_is_backward_rolling_median():
    frame = liq.attach_dollar_volume(_klines(n=5))
    out = liq.attach_trailing_liquidity(frame, lookback_bars=3)
    dv = list(frame["dollar_volume"])
    # bar 3 (0-indexed) = median of dv[1:4]; min_periods=1 so early bars are partial
    assert out["trail_dollar_volume"].iloc[0] == pytest.approx(dv[0])
    assert out["trail_dollar_volume"].iloc[3] == pytest.approx(
        pd.Series(dv[1:4]).median()
    )


def test_trailing_liquidity_preserves_row_order_and_length():
    frame = liq.attach_dollar_volume(_klines(n=6))
    out = liq.attach_trailing_liquidity(frame, lookback_bars=2)
    assert list(out["ts"]) == list(frame["ts"])
    assert len(out) == len(frame)


# --------------------------------------------------------------------------- #
# the H1 join recipe: [ts, symbol, funding_apr, dollar_volume]
# --------------------------------------------------------------------------- #
def test_funding_vs_liquidity_returns_h1_schema():
    out = liq.funding_vs_liquidity(_klines(), _funding(), funding_interval_hours=8.0)
    assert list(out.columns[:4]) == ["ts", "symbol", "funding_apr", "dollar_volume"]
    assert len(out) == 8


def test_funding_vs_liquidity_annualizes_funding():
    out = liq.funding_vs_liquidity(_klines(), _funding(), funding_interval_hours=8.0)
    # APR = rate * 8760/8 = rate * 1095
    assert out["funding_apr"].iloc[0] == pytest.approx(0.0001 * 1095.0)


def test_funding_vs_liquidity_raises_on_symbol_format_mismatch():
    k = _klines(symbol="ZEC/USDT")
    f = _funding(symbol="ZEC")  # bare vs slash -> silent zero-row join -> must RAISE
    with pytest.raises(ProvenanceError):
        liq.funding_vs_liquidity(k, f, funding_interval_hours=8.0)


# --------------------------------------------------------------------------- #
# lake conveniences + the carry mechanic on a TAIL market (read-only DuckDB)
# --------------------------------------------------------------------------- #
def _lake_con():
    duckdb = pytest.importorskip("duckdb")
    con = duckdb.connect(":memory:")
    return con


def test_funding_vs_liquidity_from_lake_builds_h1_frame():
    con = _lake_con()
    # a *_funding_on_4h-style view carries BOTH volume and funding_rate on one grid.
    fon4h = _klines(symbol="ZEC/USDT").assign(funding_rate=_funding()["funding_rate"])
    con.register("zec_fon4h", fon4h)
    con.execute("CREATE VIEW binance_tail_funding_on_4h AS SELECT * FROM zec_fon4h")
    out = liq.funding_vs_liquidity_from_lake(
        con, view="binance_tail_funding_on_4h", symbol="ZEC/USDT",
        funding_interval_hours=8.0,
    )
    assert list(out.columns[:4]) == ["ts", "symbol", "funding_apr", "dollar_volume"]
    assert (out["symbol"] == "ZEC/USDT").all()
    assert len(out) == 8


def test_carry_mechanic_runs_on_a_tail_market():
    """§7: the carry mechanic (alpha) is calculable on a queue/tail market UNCHANGED.

    cash_and_carry_from_lake is symbol-agnostic — registering a TAIL symbol under the
    canonical view names proves the mechanic runs on the illiquid tail with no code
    change (β only needs to point it at the *_tail views — a documented extension).
    """
    from hermes.datalake import carry_join as cj

    con = _lake_con()
    spot = _klines(symbol="ZEC/USDT")
    perp = _klines(symbol="ZEC/USDT").assign(funding_rate=_funding()["funding_rate"])
    con.register("spot_df", spot)
    con.register("perp_df", perp)
    con.execute("CREATE VIEW binance_spot_klines_4h AS SELECT * FROM spot_df")
    con.execute("CREATE VIEW binance_funding_on_4h AS SELECT * FROM perp_df")

    aligned = cj.cash_and_carry_from_lake(con, symbol="ZEC/USDT")
    aligned = cj.carry_returns(aligned)
    pnl = cj.delta_neutral_pnl(
        spot_ret=aligned["spot_ret"], perp_ret=aligned["perp_ret"],
        funding_collected=aligned["funding_rate"],
    )
    assert len(aligned) == 8
    assert (aligned["symbol"] == "ZEC/USDT").all()
    assert len(pnl) == 8
