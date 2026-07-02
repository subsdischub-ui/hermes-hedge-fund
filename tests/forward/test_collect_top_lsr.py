"""FORWARD — Binance top-trader long/short-ratio collector.

Pure DATA accrual, no signal: the DeadEndsLedger already killed the DIRECTIONAL
top-LSR contrarian use ("NEVER re-run continuous top-LSR" as a signal), but Binance's
free endpoints only retain ~30 rolling days, so this collector exists purely to stop
that history from being lost — shadow size-0, nothing traded. Tests exercise the pure
row-shaping transform and the fail-soft/idempotent collection wrapper with an injected
fake fetcher; no network call is ever made in test.
"""

from __future__ import annotations

from hermes.forward import collect_top_lsr as lsr
from hermes.forward import logio

_DAY_MS = 86_400_000
_NOW = 1_780_000_000_000


def _raw(symbol: str, n=3, start_ts=_NOW):
    return [
        {
            "symbol": symbol,
            "longShortRatio": str(2.0 + i * 0.1),
            "longAccount": str(0.66 + i * 0.01),
            "shortAccount": str(0.34 - i * 0.01),
            "timestamp": start_ts + i * _DAY_MS,
        }
        for i in range(n)
    ]


def test_ratio_rows_shapes_raw_payload():
    out = lsr.ratio_rows("BTCUSDT", "account", _raw("BTCUSDT", n=2))
    assert list(out["symbol"]) == ["BTCUSDT", "BTCUSDT"]
    assert list(out["ratio_type"]) == ["account", "account"]
    assert out["long_short_ratio"].iloc[0] == 2.0
    assert out["long_account"].iloc[0] == 0.66
    assert out["ts"].iloc[0] == _NOW


def test_ratio_rows_empty_raw_yields_empty_no_fabrication():
    out = lsr.ratio_rows("BTCUSDT", "account", [])
    assert out.empty
    assert list(out.columns) == ["ts", "symbol", "ratio_type", "long_short_ratio", "long_account", "short_account"]


def test_ratio_rows_drops_rows_missing_ratio_or_timestamp():
    raw = [
        *_raw("BTCUSDT", n=1),
        {"symbol": "BTCUSDT", "longShortRatio": None, "longAccount": "0.5", "shortAccount": "0.5", "timestamp": _NOW},
        {"symbol": "BTCUSDT", "longShortRatio": "1.5", "longAccount": "0.5", "shortAccount": "0.5", "timestamp": None},
    ]
    out = lsr.ratio_rows("BTCUSDT", "account", raw)
    assert len(out) == 1  # only the well-formed row survives


def test_collect_is_fail_soft_per_symbol_and_ratio_type():
    # SUIUSDT position ratio is "unavailable" (raises); everything else succeeds.
    def fetch(symbol, ratio_type):
        if symbol == "SUIUSDT" and ratio_type == "position":
            raise RuntimeError("451: geo-blocked")
        return _raw(symbol, n=1)

    out = lsr.collect(("BTCUSDT", "SUIUSDT"), ("account", "position"), fetch=fetch)
    assert "SUIUSDT:position" in out["blockers"]
    assert set(out["rows"]["symbol"]) == {"BTCUSDT", "SUIUSDT"}
    # SUIUSDT still contributes its "account" row even though "position" failed
    sui_types = set(out["rows"].loc[out["rows"]["symbol"] == "SUIUSDT", "ratio_type"])
    assert sui_types == {"account"}


def test_collect_skips_empty_symbol_with_blocker_not_crash():
    def fetch(symbol, ratio_type):
        if symbol == "XAGUSDT":
            return []  # Binance returns an empty list for an unsupported/empty symbol
        return _raw(symbol, n=1)

    out = lsr.collect(("BTCUSDT", "XAGUSDT"), ("account",), fetch=fetch)
    assert "XAGUSDT:account" in out["blockers"]
    assert set(out["rows"]["symbol"]) == {"BTCUSDT"}


def test_collect_all_symbols_unavailable_yields_empty_frame_and_blockers_only():
    def fetch(symbol, ratio_type):
        raise RuntimeError("feed down")

    out = lsr.collect(("BTCUSDT",), ("account", "position"), fetch=fetch)
    assert out["rows"].empty
    assert set(out["blockers"]) == {"BTCUSDT:account", "BTCUSDT:position"}


def test_collect_default_symbols_include_live_universe_and_metals():
    assert set(("BTCUSDT", "ETHUSDT", "SOLUSDT", "AVAXUSDT", "SUIUSDT", "ARBUSDT", "WIFUSDT")) <= set(
        lsr.DEFAULT_SYMBOLS
    )
    assert "XAUUSDT" in lsr.DEFAULT_SYMBOLS
    assert "XAGUSDT" in lsr.DEFAULT_SYMBOLS


def test_end_to_end_write_through_logio_is_idempotent_no_duplicates(tmp_path):
    # Simulates the daily cron: collect, write via the shared PIT log, re-run same day.
    def fetch(symbol, ratio_type):
        return _raw(symbol, n=1, start_ts=_NOW)

    res1 = lsr.collect(("BTCUSDT", "ETHUSDT"), ("account", "position"), fetch=fetch)
    logio.append_snapshot(
        "top_lsr", res1["rows"], key_cols=("symbol", "ratio_type"), data_dir=tmp_path
    )
    # same UTC day re-run (a cron double-fire) must not duplicate rows
    res2 = lsr.collect(("BTCUSDT", "ETHUSDT"), ("account", "position"), fetch=fetch)
    logio.append_snapshot(
        "top_lsr", res2["rows"], key_cols=("symbol", "ratio_type"), data_dir=tmp_path
    )
    back = logio.read_log("top_lsr", data_dir=tmp_path)
    assert len(back) == 4  # 2 symbols x 2 ratio types, deduped per (day, symbol, ratio_type)


def test_end_to_end_distinct_days_accumulate(tmp_path):
    def fetch_day(day_ts):
        def fetch(symbol, ratio_type):
            return _raw(symbol, n=1, start_ts=day_ts)
        return fetch

    r1 = lsr.collect(("BTCUSDT",), ("account",), fetch=fetch_day(_NOW))
    logio.append_snapshot("top_lsr", r1["rows"], key_cols=("symbol", "ratio_type"), data_dir=tmp_path)
    r2 = lsr.collect(("BTCUSDT",), ("account",), fetch=fetch_day(_NOW + _DAY_MS))
    logio.append_snapshot("top_lsr", r2["rows"], key_cols=("symbol", "ratio_type"), data_dir=tmp_path)
    back = logio.read_log("top_lsr", data_dir=tmp_path)
    assert len(back) == 2  # two distinct UTC days both retained
