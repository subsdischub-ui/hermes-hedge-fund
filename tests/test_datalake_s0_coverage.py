"""S0 re-freeze COVERAGE assertions — depth-vs-reality, run against the LIVE lake.

The lake (data/hermes_datalake.duckdb) is gitignored / point-in-time, so on a
fresh checkout there is nothing to assert: this module SKIPS cleanly when the
lake is absent (keeping the offline suite green) and ASSERTS the materialized
depths when it is present. It is the GARDE-FOUS "minimum depth per pair" gate and
a standing regression for the two traps this stream hit:

  * the /fapi/v1/fundingRate startTime=0 clamp (deep native funding must NOT
    collapse back to the ~200-row forward stub), and
  * confusing ``binance_funding`` (forward stub) with ``binance_funding_native``
    (deep) — they must stay distinct, with native an order of magnitude deeper.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes.datalake.sources.binance_extended import ALL_PAIRS, LIVE_PAIRS

_DUCKDB = Path(__file__).resolve().parents[1] / "data" / "hermes_datalake.duckdb"

pytestmark = pytest.mark.skipif(
    not _DUCKDB.exists(),
    reason="lake not materialized (gitignored); run `python -m hermes.datalake.run` to populate",
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


# --------------------------------------------------------------------------- #
# TACHE 1 + 2 — 4h klines: every pair present + min depth; live 7 deep
# --------------------------------------------------------------------------- #
def test_klines_4h_view_exists(con):
    assert "binance_klines_4h" in _views(con)


def test_klines_4h_all_pairs_present(con):
    pairs = {r[0] for r in con.execute(
        "SELECT DISTINCT replace(symbol,'/USDT','') FROM binance_klines_4h").fetchall()}
    missing = set(ALL_PAIRS) - pairs
    assert not missing, f"4h klines missing pairs: {sorted(missing)}"


def test_klines_4h_min_depth_per_pair(con):
    # WIF (the shortest, ~2.4y) sets the floor; every pair must clear ~2y of 4h
    # bars (2y * 365 * 6 ~= 4380). A pair below this would be a listing/pull bug.
    rows = con.execute("""
        SELECT replace(symbol,'/USDT','') pair, count(*) n
        FROM binance_klines_4h GROUP BY 1
    """).fetchall()
    for pair, n in rows:
        assert n >= 4000, f"{pair} 4h klines only {n} bars (<2y) — depth/listing bug"


def test_klines_4h_live_majors_are_deep(con):
    # BTC/ETH/SOL must reach back to their ~2019-2020 listing (>=5y of 4h bars)
    for base in ("BTC", "ETH", "SOL"):
        (n,) = con.execute(
            "SELECT count(*) FROM binance_klines_4h WHERE symbol = ?",
            [f"{base}/USDT"]).fetchone()
        assert n >= 10000, f"{base} 4h only {n} bars — expected deep (>=5y)"


# --------------------------------------------------------------------------- #
# TACHE 3 — deep native funding, DISTINCT from the forward stub
# --------------------------------------------------------------------------- #
def test_funding_native_view_exists_and_distinct(con):
    vs = _views(con)
    assert "binance_funding_native" in vs
    assert "binance_funding" in vs  # the original forward stub must still exist


def test_funding_native_is_deep_not_the_200_row_stub(con):
    # every pair must be an order of magnitude deeper than the ~200-row forward
    # stub — this is the regression that catches the startTime=0 clamp.
    rows = con.execute(
        "SELECT symbol, count(*) n FROM binance_funding_native GROUP BY 1").fetchall()
    assert len(rows) == len(ALL_PAIRS)
    for sym, n in rows:
        assert n >= 3000, f"{sym} native funding only {n} rows — startTime clamp regressed?"


def test_funding_native_strictly_deeper_than_stub(con):
    # the two views must never be confused: native min depth >> stub max depth.
    (stub_max,) = con.execute(
        "SELECT max(c) FROM (SELECT count(*) c FROM binance_funding GROUP BY symbol)"
    ).fetchone()
    (native_min,) = con.execute(
        "SELECT min(c) FROM (SELECT count(*) c FROM binance_funding_native GROUP BY symbol)"
    ).fetchone()
    assert native_min > 10 * stub_max, (
        f"native_min={native_min} not >> stub_max={stub_max} — views may be confused")


def test_funding_native_majors_reach_2019_2020(con):
    # BTC native funding must start at its 2019-09 inception, not 2026.
    (min_ts,) = con.execute(
        "SELECT min(ts) FROM binance_funding_native WHERE symbol = 'BTC/USDT'").fetchone()
    # 2021-01-01 UTC in ms; BTC must predate this comfortably
    assert min_ts < 1_609_459_200_000, "BTC native funding does not reach inception"


def test_funding_on_4h_join_materialized(con):
    vs = _views(con)
    assert "binance_funding_on_4h" in vs
    # the join must cover the live universe and carry a funding_rate column
    cols = {r[0] for r in con.execute("DESCRIBE binance_funding_on_4h").fetchall()}
    assert "funding_rate" in cols
    pairs = {r[0] for r in con.execute(
        "SELECT DISTINCT replace(symbol,'/USDT','') FROM binance_funding_on_4h").fetchall()}
    assert set(LIVE_PAIRS).issubset(pairs)


def test_funding_info_snapshot_present(con):
    assert "binance_funding_info" in _views(con)
    (n,) = con.execute("SELECT count(*) FROM binance_funding_info").fetchone()
    assert n > 0


# --------------------------------------------------------------------------- #
# TACHE 4 — multi-venue dispersion substrate
# --------------------------------------------------------------------------- #
def test_multi_venue_views_present_and_multi_exchange(con):
    vs = _views(con)
    for v in ("coinglass_multi_funding_rate", "coinglass_multi_open_interest"):
        assert v in vs, f"{v} missing"
        exchanges = {r[0] for r in con.execute(
            f"SELECT DISTINCT exchange FROM {v}").fetchall()}
        # the whole point: NOT Binance-only
        assert {"OKX", "Bybit"}.issubset(exchanges), f"{v} not multi-venue: {exchanges}"
        assert len(exchanges) >= 3


def test_multi_venue_funding_is_deep(con):
    # cross-venue cells must reach 2021 or earlier (not a recent-only cap)
    (min_ts,) = con.execute(
        "SELECT min(ts) FROM coinglass_multi_funding_rate WHERE exchange='OKX' AND symbol='BTC'"
    ).fetchone()
    assert min_ts < 1_640_995_200_000, "OKX BTC funding not deep (expected <=2021)"
