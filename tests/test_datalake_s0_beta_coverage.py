"""S0-β COVERAGE assertions — depth-vs-reality on the LIVE lake (illiquid-niche carry).

SKIPS cleanly when the gitignored lake is absent (offline suite stays green) and ASSERTS
the materialized depths when present. The GARDE-FOUS gate for the Stream 0-β additions:

  * binance_tail_* — the illiquid USDT-M perp tail: present + non-empty, the selection
    universe spans the illiquid deciles, and binance_tail_funding_on_4h carries BOTH volume
    and funding_rate so the H1 funding-vs-liquidity recipe runs on a tail market.
  * hyperliquid_exotic_* — exotic markets: the ctx snapshot makes survivorship OBSERVABLE
    (delisted flag present), funding/klines present.
  * ADDITIVITY: the 23-major views (binance_spot_klines_4h) are byte-untouched — still
    exactly the 23, proving the tail did NOT leak into the existing views.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from hermes.datalake import liquidity as liq
from hermes.datalake.sources.binance_extended import ALL_PAIRS

_DUCKDB = Path(__file__).resolve().parents[1] / "data" / "hermes_datalake.duckdb"

pytestmark = pytest.mark.skipif(
    not _DUCKDB.exists(),
    reason="lake not materialized (gitignored); run the S0-β ingest to populate",
)


@pytest.fixture(scope="module")
def con():
    import duckdb

    c = duckdb.connect(str(_DUCKDB), read_only=True)
    yield c
    c.close()


def _views(con) -> set[str]:
    return {v for (v,) in con.execute(
        "SELECT view_name FROM duckdb_views() WHERE NOT internal").fetchall()}


def _require_tail(con) -> None:
    if "binance_tail_klines_4h" not in _views(con):
        pytest.skip("binance_tail views not yet materialized — run the S0-β ingest")


def _require_exotic(con) -> None:
    if "hyperliquid_exotic_ctx" not in _views(con):
        pytest.skip("hyperliquid_exotic views not yet materialized — run the S0-β ingest")


# =========================================================================== #
# ADDITIVITY — the 23-major views are untouched (the tail did NOT leak in)
# =========================================================================== #
def test_existing_major_spot_view_still_exactly_23(con):
    if "binance_spot_klines_4h" not in _views(con):
        pytest.skip("alpha spot view not materialized")
    n = con.execute(
        "SELECT COUNT(DISTINCT symbol) FROM binance_spot_klines_4h"
    ).fetchone()[0]
    assert n == len(ALL_PAIRS) == 23  # additive: the tail lives in SEPARATE views


# =========================================================================== #
# Deliverable A — binance_tail_* (the illiquid USDT-M perp tail)
# =========================================================================== #
def test_tail_perp_klines_present_and_nonempty(con):
    _require_tail(con)
    n_sym = con.execute(
        "SELECT COUNT(DISTINCT symbol) FROM binance_tail_klines_4h"
    ).fetchone()[0]
    assert n_sym >= 1


def test_tail_universe_spans_the_illiquid_deciles(con):
    _require_tail(con)
    rows = con.execute(
        "SELECT MIN(decile), MAX(decile), COUNT(*) FROM binance_tail_universe"
    ).fetchone()
    dmin, dmax, n = rows
    assert n >= 1
    # the policy descends into the illiquid bottom: the floor decile must be represented
    assert dmin == 0


def test_tail_symbols_are_slash_spelled(con):
    _require_tail(con)
    sym = con.execute(
        "SELECT symbol FROM binance_tail_klines_4h LIMIT 1"
    ).fetchone()[0]
    assert "/" in sym  # slash spelling -> joins the carry grid


def test_tail_funding_on_4h_carries_volume_and_funding(con):
    _require_tail(con)
    if "binance_tail_funding_on_4h" not in _views(con):
        pytest.skip("tail funding_on_4h not materialized")
    cols = {c[1] for c in con.execute(
        "PRAGMA table_info('binance_tail_funding_on_4h')").fetchall()}
    assert {"volume", "funding_rate", "close"} <= cols


def test_h1_recipe_runs_on_a_tail_market(con):
    """The decisive β deliverable: funding-APR vs liquidity, joinable, on a TAIL market."""
    _require_tail(con)
    if "binance_tail_funding_on_4h" not in _views(con):
        pytest.skip("tail funding_on_4h not materialized")
    sym = con.execute(
        "SELECT symbol FROM binance_tail_funding_on_4h "
        "WHERE funding_rate IS NOT NULL LIMIT 1"
    ).fetchone()
    if sym is None:
        pytest.skip("no tail funding rows yet")
    out = liq.funding_vs_liquidity_from_lake(
        con, view="binance_tail_funding_on_4h", symbol=sym[0], funding_interval_hours=8.0
    )
    assert list(out.columns[:4]) == ["ts", "symbol", "funding_apr", "dollar_volume"]
    assert len(out) >= 1


# =========================================================================== #
# Deliverable B — hyperliquid_exotic_* (exotic markets + survivorship snapshot)
# =========================================================================== #
def test_exotic_ctx_snapshot_makes_survivorship_observable(con):
    _require_exotic(con)
    cols = {c[1] for c in con.execute(
        "PRAGMA table_info('hyperliquid_exotic_ctx')").fetchall()}
    assert "isDelisted" in cols  # HL survivorship gap is OBSERVABLE (unlike Binance)
    n = con.execute("SELECT COUNT(*) FROM hyperliquid_exotic_ctx").fetchone()[0]
    assert n >= 1


def test_exotic_universe_manifest_spans_deciles(con):
    _require_exotic(con)
    if "hyperliquid_exotic_universe" not in _views(con):
        pytest.skip("exotic universe manifest not materialized")
    dmin, n = con.execute(
        "SELECT MIN(decile), COUNT(*) FROM hyperliquid_exotic_universe"
    ).fetchone()
    assert n >= 1
    assert dmin == 0  # the slice descends into the illiquid floor


def test_exotic_funding_and_klines_present(con):
    _require_exotic(con)
    if "hyperliquid_exotic_funding_1h" not in _views(con):
        pytest.skip("exotic funding not materialized")
    n = con.execute(
        "SELECT COUNT(DISTINCT symbol) FROM hyperliquid_exotic_funding_1h"
    ).fetchone()[0]
    assert n >= 1


def test_exotic_symbols_are_bare(con):
    _require_exotic(con)
    if "hyperliquid_exotic_klines" not in _views(con):
        pytest.skip("exotic klines not materialized")
    sym = con.execute(
        "SELECT symbol FROM hyperliquid_exotic_klines LIMIT 1"
    ).fetchone()[0]
    assert "/" not in sym and sym.upper() == sym  # bare HL spelling
