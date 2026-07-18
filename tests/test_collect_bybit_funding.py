"""FORWARD (infra, hors-essai) — Bybit 8h funding-history collector.

Pure DATA accrual for the cross-venue Bybit revive_condition named on ledger
entry #41 ("re-test >= 2026-11"). Collection is $0, no strategy attached, no
new trial registered. Tests exercise the pure parsing transform and the
fail-soft / idempotent collection wrapper with an injected fake fetcher — NO
real network call is ever made here.
"""

from __future__ import annotations

import pandas as pd
import pytest

from hermes.forward import collect_bybit_funding as bf


def _raw(symbol: str, timestamps: list[int], rate: float = 0.0001):
    """Shape a Bybit-style page: newest first (as the real API returns it)."""
    return [
        {"symbol": symbol, "fundingRate": str(rate), "fundingRateTimestamp": str(ts)}
        for ts in sorted(timestamps, reverse=True)
    ]


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------
def test_parse_funding_rows_shapes_a_well_formed_page():
    raw = _raw("BTCUSDT", [1_000, 29_800, 58_600], rate=0.00015)
    rows = bf.parse_funding_rows(raw)
    assert set(rows) == {(1_000, 0.00015), (29_800, 0.00015), (58_600, 0.00015)}


def test_parse_funding_rows_drops_malformed_entries_never_fabricates():
    raw = [
        {"symbol": "BTCUSDT", "fundingRate": "0.0001", "fundingRateTimestamp": "1000"},
        {"symbol": "BTCUSDT", "fundingRate": None, "fundingRateTimestamp": "2000"},
        {"symbol": "BTCUSDT", "fundingRate": "0.0002", "fundingRateTimestamp": None},
        {"symbol": "BTCUSDT", "fundingRate": "not-a-number", "fundingRateTimestamp": "3000"},
    ]
    rows = bf.parse_funding_rows(raw)
    assert rows == [(1000, 0.0001)]


def test_parse_funding_rows_empty_yields_empty_no_fabrication():
    assert bf.parse_funding_rows([]) == []
    assert bf.parse_funding_rows(None) == []


# ---------------------------------------------------------------------------
# Idempotent CSV append
# ---------------------------------------------------------------------------
def test_append_rows_writes_new_rows_and_skips_exact_ts_duplicates(tmp_path):
    path = tmp_path / "BTCUSDT.csv"
    n1 = bf.append_rows(path, [(1000, 0.0001), (2000, 0.0002)])
    assert n1 == 2
    n2 = bf.append_rows(path, [(1000, 0.0001), (3000, 0.0003)])  # 1000 is a dup
    assert n2 == 1
    df = pd.read_csv(path)
    assert list(df["ts_ms"]) == [1000, 2000, 3000]
    assert list(df.columns) == ["ts_ms", "funding_rate"]


def test_append_rows_on_empty_batch_is_a_noop(tmp_path):
    path = tmp_path / "BTCUSDT.csv"
    assert bf.append_rows(path, []) == 0
    assert not path.exists()


# ---------------------------------------------------------------------------
# Pagination / backfill (fake fetcher, no network)
# ---------------------------------------------------------------------------
def test_backfill_symbol_paginates_until_a_short_page_ends_history(tmp_path):
    # Two full pages of 3 (limit=3), then a short final page of 1 -> stop.
    pages = [
        _raw("BTCUSDT", [700, 800, 900]),   # newest
        _raw("BTCUSDT", [400, 500, 600]),
        _raw("BTCUSDT", [100]),             # short page -> end of history
    ]
    calls = []

    def fake_fetch(symbol, *, end_time=None, limit=200):
        calls.append(end_time)
        return pages[len(calls) - 1]

    n = bf.backfill_symbol(
        "BTCUSDT", data_dir=tmp_path, fetch=fake_fetch, limit=3, sleep_s=0.0
    )
    assert n == 7
    df = pd.read_csv(tmp_path / "bybit_funding" / "BTCUSDT.csv")
    assert list(df["ts_ms"]) == [100, 400, 500, 600, 700, 800, 900]
    assert calls[0] is None          # first page: no endTime
    assert calls[1] == 700 - 1       # next page starts just before page 1's oldest (700)
    assert calls[2] == 400 - 1       # then just before page 2's oldest (400)


def test_backfill_symbol_stops_early_once_reaching_already_known_territory(tmp_path):
    # Simulate a SECOND run: the CSV already has [400..900]; only [700,800,900]
    # (the newest page) should be re-fetched before the collector recognizes
    # it has reached previously-backfilled ground and stops (idempotent + cheap).
    existing = tmp_path / "bybit_funding" / "BTCUSDT.csv"
    existing.parent.mkdir(parents=True)
    pd.DataFrame(
        {"ts_ms": [400, 500, 600, 700, 800, 900], "funding_rate": [0.0001] * 6}
    ).to_csv(existing, index=False)

    pages = [_raw("BTCUSDT", [700, 800, 900])]  # entirely already-known -> stop after page 1
    calls = []

    def fake_fetch(symbol, *, end_time=None, limit=200):
        calls.append(end_time)
        assert len(calls) <= 1, "must not paginate past an all-duplicate page"
        return pages[0]

    n = bf.backfill_symbol(
        "BTCUSDT", data_dir=tmp_path, fetch=fake_fetch, limit=3, sleep_s=0.0
    )
    assert n == 0
    assert len(calls) == 1


# ---------------------------------------------------------------------------
# collect(): fail-soft per symbol, idempotence at the collect() level.
# ---------------------------------------------------------------------------
def test_collect_is_fail_soft_per_symbol():
    def fetch(symbol, *, end_time=None, limit=200):
        if symbol == "ZECUSDT":
            raise RuntimeError("symbol not listed on Bybit")
        return _raw(symbol, [1000, 2000])

    out = bf.collect(("BTCUSDT", "ZECUSDT"), data_dir=None, fetch=fetch)
    assert "ZECUSDT" in out["blockers"]
    assert out["written"]["BTCUSDT"] == 2
    assert "BTCUSDT" not in out["blockers"]


def test_collect_second_run_writes_zero_new_rows(tmp_path):
    def fetch(symbol, *, end_time=None, limit=200):
        return _raw(symbol, [1000, 2000, 3000])

    r1 = bf.collect(("BTCUSDT",), data_dir=tmp_path, fetch=fetch)
    assert r1["written"]["BTCUSDT"] == 3
    r2 = bf.collect(("BTCUSDT",), data_dir=tmp_path, fetch=fetch)
    assert r2["written"]["BTCUSDT"] == 0  # idempotent: nothing new the 2nd time
    assert not r2["blockers"]


def test_default_symbols_include_the_breadth_universe():
    expected = {
        "BTCUSDT", "ETHUSDT", "SOLUSDT", "AVAXUSDT", "SUIUSDT", "ARBUSDT", "WIFUSDT",
        "XRPUSDT", "BNBUSDT", "DOGEUSDT", "ZECUSDT",
    }
    assert expected <= set(bf.DEFAULT_SYMBOLS)


# ---------------------------------------------------------------------------
# No real network call anywhere in this module's test file.
# ---------------------------------------------------------------------------
def test_fetch_funding_page_is_a_separate_pragma_no_cover_network_seam():
    # The default `fetch` param of collect()/backfill_symbol IS the real network
    # seam (bf.fetch_funding_page) — but every test above injects a fake fetch,
    # so importing/using this module in the test suite never touches the network.
    assert callable(bf.fetch_funding_page)


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__])


def test_monkeypatching_the_module_attr_is_sufficient(tmp_path, monkeypatch):
    """Régression du 2026-07-18 : `fetch=fetch_funding_page` en DÉFAUT DE PARAMÈTRE
    se lie à l'import — un monkeypatch de l'attribut de module était alors sans
    effet et les tests CLI (tests/forward/test_run.py) frappaient l'API RÉELLE :
    140 s de backfill complet PAR TEST, mesuré. La couture doit se résoudre à
    l'APPEL : patcher `bf.fetch_funding_page` doit suffire, sans passer `fetch=`."""
    calls = []

    def _fake(symbol, *, end_time=None, limit=200):
        calls.append(symbol)
        return []

    monkeypatch.setattr(bf, "fetch_funding_page", _fake)
    out = bf.collect(symbols=("BTCUSDT",), data_dir=tmp_path, sleep_s=0.0)
    assert calls == ["BTCUSDT"], "le patch de l'attribut de module doit être vu par collect()"
    assert out["written"] == {"BTCUSDT": 0} and out["blockers"] == {}
