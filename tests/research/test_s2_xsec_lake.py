"""Lake-seam integration tests for s2_xsec (skip if the frozen lake is absent).

These verify the loaders against the REAL frozen DuckDB — the schema/symbol-format
contract the pure tests cannot see. The lake is gitignored/point-in-time, so on a
clean checkout these skip rather than fail. Pattern mirrors tests/test_datalake_s0_coverage.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from hermes.research import s2_xsec as xs

_DUCKDB = Path(__file__).resolve().parents[2] / "data" / "hermes_datalake.duckdb"

pytestmark = pytest.mark.skipif(
    not _DUCKDB.exists(),
    reason="lake not materialized (gitignored); run `python -m hermes.datalake.run`",
)


@pytest.fixture(scope="module")
def con():
    import duckdb

    c = duckdb.connect(str(_DUCKDB), read_only=True)
    yield c
    c.close()


def test_build_panel_klines_4h_uses_slash_symbols_and_aligns(con):
    """binance_klines_4h symbols are BASE/USDT; the panel pivots onto a shared ts grid
    with NaN before each name's listing (variable universe)."""
    panel = xs.build_panel(
        con, ["BTC/USDT", "ETH/USDT", "SOL/USDT"], view="binance_klines_4h",
        interval="4h", value_col="close",
    )
    assert list(panel.columns) == ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
    assert len(panel) > 14000               # BTC's full 4h depth sets the grid length
    # SOL lists later than BTC -> leading NaN for SOL while BTC is already trading
    assert panel["SOL/USDT"].isna().sum() > 0
    assert panel["BTC/USDT"].iloc[-1] > 0   # last bar present


def test_build_panel_rejects_non_allowlisted_view(con):
    with pytest.raises(ValueError):
        xs.build_panel(con, ["BTC/USDT"], view="sqlite_master", interval="4h")


def test_load_native_funding_is_deep_and_settlement_cadenced(con):
    """Native funding is deep (thousands of settlements), not the ~200-row stub."""
    f = xs.load_native_funding(con, ["BTC/USDT"])
    btc = f["BTC/USDT"]
    assert len(btc) > 7000                   # ~3 settlements/day since 2019-09
    assert np.isfinite(btc.to_numpy()).all()
    # values are small fractions (per-8h funding), |rate| well under 1%
    assert float(np.nanmax(np.abs(btc.to_numpy()))) < 0.01


def test_build_panel_oi_uses_concat_symbols(con):
    """coinglass_open_interest symbols are concatenated BTCUSDT (NOT slashed) — the
    join trap; the loader passes them through verbatim."""
    panel = xs.build_panel(
        con, ["BTCUSDT", "ETHUSDT"], view="coinglass_open_interest",
        interval="1d", value_col="close",
    )
    assert list(panel.columns) == ["BTCUSDT", "ETHUSDT"]
    assert len(panel) > 1000 and (panel["BTCUSDT"].dropna() > 0).all()


def test_build_venue_funding_has_four_venues(con):
    """coinglass_multi_funding_rate symbols are bare base (BTC) across 4 venues."""
    v = xs.build_venue_funding(con, "BTC", interval="1d")
    assert {"Binance", "OKX", "Bybit", "Bitget"}.issubset(set(v.columns))
    assert len(v) > 1000
