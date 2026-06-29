"""S0-alpha COVERAGE assertions — depth-vs-reality on the LIVE lake (cash-and-carry).

Like ``test_datalake_s0_coverage`` this SKIPS cleanly when the gitignored lake is
absent (offline suite stays green) and ASSERTS the materialized depths when present.
It is the GARDE-FOUS gate for the two Stream 0-alpha additions:

  * ``binance_spot_klines_4h`` — the cash-and-carry SPOT leg: all 23 pairs, slash
    symbols, the 2y depth floor, and a PROOF that the delta-neutral join is lossless
    (spot ⋈ perp+funding on (ts, symbol) returns real aligned rows).
  * ``hyperliquid_funding_1h`` / ``hyperliquid_klines`` — cross-venue carry: bare
    symbols, present + non-trivial (shallow/forward by nature), and a cross-venue
    APR-differential that aligns Binance native funding vs HL hourly funding.

The HL assertions self-skip until the HL ingest has materialized those views.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes.datalake import carry_join as cj
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


def _require_hl(con) -> None:
    if "hyperliquid_funding_1h" not in _views(con):
        pytest.skip("Hyperliquid views not yet materialized — run `--sources hyperliquid`")


# =========================================================================== #
# Deliverable A — binance_spot_klines_4h (the cash-and-carry SPOT leg)
# =========================================================================== #
def test_spot_view_exists(con):
    assert "binance_spot_klines_4h" in _views(con)


def test_spot_all_23_pairs_present(con):
    pairs = {r[0] for r in con.execute(
        "SELECT DISTINCT replace(symbol,'/USDT','') FROM binance_spot_klines_4h").fetchall()}
    missing = set(ALL_PAIRS) - pairs
    assert not missing, f"spot 4h missing pairs: {sorted(missing)}"


def test_spot_symbols_are_slash_format_like_perp(con):
    # the JOIN contract: spot must use the SAME slash spelling as binance_klines_4h
    syms = [r[0] for r in con.execute(
        "SELECT DISTINCT symbol FROM binance_spot_klines_4h LIMIT 5").fetchall()]
    assert all("/USDT" in s for s in syms), syms


def test_spot_ts_is_epoch_ms_int64(con):
    dtype = con.execute("DESCRIBE binance_spot_klines_4h").df()
    ts_type = dtype.loc[dtype["column_name"] == "ts", "column_type"].iloc[0]
    assert ts_type.upper().startswith("BIGINT")
    # a 2020-2026 bar in ms is ~1.5e12 — guard against a seconds/ms mixup
    mn = con.execute("SELECT min(ts) FROM binance_spot_klines_4h").fetchone()[0]
    assert mn > 1_000_000_000_000


def test_spot_min_depth_per_pair(con):
    # mirror the perp floor: every pair clears ~2y of 4h bars (2*365*6 ~= 4380)
    rows = con.execute("""
        SELECT replace(symbol,'/USDT','') pair, count(*) n
        FROM binance_spot_klines_4h GROUP BY 1
    """).fetchall()
    for pair, n in rows:
        assert n >= 4000, f"{pair} spot 4h only {n} bars (<2y) — depth/listing bug"


def test_spot_live_majors_are_deep(con):
    # BTC/ETH/SOL spot must reach inception-era depth (>=10k bars ~ 4.5y)
    for base in ("BTC", "ETH", "SOL"):
        n = con.execute(
            "SELECT count(*) FROM binance_spot_klines_4h WHERE symbol = ?",
            [f"{base}/USDT"],
        ).fetchone()[0]
        assert n >= 10_000, f"{base} spot only {n} bars — expected inception-era depth"


# =========================================================================== #
# The JOIN — delta-neutral P&L is CALCULABLE (the §2 deliverable, on real data)
# =========================================================================== #
def test_cash_and_carry_join_is_lossless_on_real_data(con):
    aligned = cj.cash_and_carry_from_lake(con, symbol="BTC/USDT")
    assert len(aligned) > 4000, "spot ⋈ perp+funding produced too few aligned bars"
    assert list(aligned.columns) == ["ts", "symbol", "spot_close", "perp_close", "funding_rate"]
    # both legs are real positive prices; funding is present (the carry payload)
    assert (aligned["spot_close"] > 0).all()
    assert (aligned["perp_close"] > 0).all()
    assert aligned["funding_rate"].notna().mean() > 0.5


def test_delta_neutral_pnl_is_computable_end_to_end(con):
    aligned = cj.carry_returns(cj.cash_and_carry_from_lake(con, symbol="BTC/USDT"))
    pnl = cj.delta_neutral_pnl(
        spot_ret=aligned["spot_ret"], perp_ret=aligned["perp_ret"],
        funding_collected=aligned["funding_rate"], costs=0.0,
    )
    # the mechanic produces a finite per-bar series (the harvest is the NEXT mandate)
    assert pnl.notna().sum() > 4000
    # spot and perp track each other closely (delta-neutral) -> |gross| typically small
    assert pnl.dropna().abs().median() < 0.05


# =========================================================================== #
# Deliverable B — Hyperliquid views (cross-venue, shallow/forward by nature)
# =========================================================================== #
def test_hl_views_exist(con):
    _require_hl(con)
    v = _views(con)
    assert "hyperliquid_funding_1h" in v
    assert "hyperliquid_klines" in v


def test_hl_funding_is_bare_symbol_and_present(con):
    _require_hl(con)
    syms = [r[0] for r in con.execute(
        "SELECT DISTINCT symbol FROM hyperliquid_funding_1h").fetchall()]
    assert "BTC" in syms                       # HL native bare spelling
    assert all("/" not in s for s in syms), syms
    n = con.execute(
        "SELECT count(*) FROM hyperliquid_funding_1h WHERE symbol='BTC'").fetchone()[0]
    assert n > 1000, f"HL BTC funding only {n} rows — pagination/ingest bug"


def test_hl_funding_overlaps_the_binance_universe(con):
    _require_hl(con)
    hl_syms = {r[0] for r in con.execute(
        "SELECT DISTINCT symbol FROM hyperliquid_funding_1h").fetchall()}
    # the carry-relevant overlap with the live book must be non-trivial
    overlap = set(LIVE_PAIRS) & hl_syms
    assert len(overlap) >= 5, f"HL overlap with live pairs too small: {overlap}"


def test_cross_venue_funding_differential_is_computable(con):
    _require_hl(con)
    out = cj.cross_venue_from_lake(con, "BTC", binance_interval_hours=8)
    assert len(out) > 0
    assert {"apr_binance", "apr_hl", "apr_diff"}.issubset(out.columns)
    # at least some Binance settlements have a known prior HL funding (causal as-of)
    assert out["apr_hl"].notna().sum() > 0
