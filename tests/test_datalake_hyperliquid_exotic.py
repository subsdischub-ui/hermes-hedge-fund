"""TDD for the Hyperliquid EXOTIC-markets source (Stream 0-β, Deliverable B).

The core of β's hypothesis: HL exotic markets (beyond the 23 cross-venue overlap) are
young, low-capital, generous-funding — the most likely home of un-arbitraged carry. HL
is also KINDER on survivorship than Binance: delisted markets stay in metaAndAssetCtxs
flagged isDelisted, so the ctx snapshot makes the survivorship gap VISIBLE.

Pure logic (parse ctxs / select / snapshot over INJECTED payloads) tested offline; run()
+ live /info transport never executed here. Symbols are HL-native BARE (BTC).
"""
from __future__ import annotations

from hermes.datalake.sources import hyperliquid_exotic as hx

YR_MS = 365 * 86_400_000
NOW = 1_751_241_600_000


def _meta():
    universe = [
        {"name": "BTC", "maxLeverage": 50},                       # ingested -> excluded
        {"name": "HYPE", "maxLeverage": 10},                      # exotic, liquid
        {"name": "SMALL", "maxLeverage": 5},                      # exotic, illiquid
        {"name": "DEAD", "maxLeverage": 5},                       # exotic, zero volume
        {"name": "GONE", "maxLeverage": 3, "isDelisted": True},   # exotic, delisted
    ]
    ctxs = [
        {"dayNtlVlm": "1000000", "openInterest": "10", "markPx": "60000", "funding": "0.00001"},
        {"dayNtlVlm": "507000000", "openInterest": "20000", "markPx": "40", "funding": "0.00002"},
        {"dayNtlVlm": "250000", "openInterest": "5000", "markPx": "2", "funding": "0.00005"},
        {"dayNtlVlm": "0", "openInterest": "0", "markPx": "1", "funding": "0"},
        {"dayNtlVlm": "300000", "openInterest": "100", "markPx": "10", "funding": "0.0001"},
    ]
    return [{"universe": universe}, ctxs]


def test_parse_asset_ctxs_extracts_liquidity_fields():
    rows = hx.parse_hl_asset_ctxs(_meta())
    hype = next(r for r in rows if r["name"] == "HYPE")
    assert hype["dayNtlVlm"] == 507_000_000.0
    assert hype["oi_usd"] == 20000.0 * 40.0  # openInterest(coins) * markPx
    assert hype["isDelisted"] is False
    gone = next(r for r in rows if r["name"] == "GONE")
    assert gone["isDelisted"] is True


def test_select_excludes_ingested_delisted_and_zero_volume():
    rows = hx.parse_hl_asset_ctxs(_meta())
    out = hx.select_exotic_universe(rows, ingested={"BTC"}, n_target=10)
    names = {r["name"] for r in out}
    assert names == {"HYPE", "SMALL"}  # BTC ingested, DEAD zero-vol, GONE delisted


def test_select_applies_depth_filter_when_inception_given():
    rows = hx.parse_hl_asset_ctxs(_meta())
    inception = {"HYPE": NOW - 2 * YR_MS, "SMALL": NOW - 100 * 86_400_000}  # SMALL < 1y
    out = hx.select_exotic_universe(
        rows, ingested={"BTC"}, n_target=10,
        inception_ms=inception, now_ms=NOW, min_history_ms=YR_MS,
    )
    assert {r["name"] for r in out} == {"HYPE"}  # SMALL fails the >=1y depth floor


def test_select_stratifies_and_annotates_decile():
    universe = [{"name": f"X{i:02d}", "maxLeverage": 5} for i in range(30)]
    ctxs = [
        {"dayNtlVlm": str((i + 1) * 100000), "openInterest": "1", "markPx": "1", "funding": "0"}
        for i in range(30)
    ]
    rows = hx.parse_hl_asset_ctxs([{"universe": universe}, ctxs])
    out = hx.select_exotic_universe(rows, ingested=set(), n_target=10)
    assert len(out) == 10
    assert all("decile" in r for r in out)
    assert min(r["decile"] for r in out) == 0  # reaches the illiquid floor


def test_ctx_snapshot_frame_makes_survivorship_visible():
    rows = hx.parse_hl_asset_ctxs(_meta())
    frame = hx.ctx_snapshot_frame(rows, run_as_of="2026-06-30")
    # the delisted market is PRESENT in the snapshot (HL survivorship is observable)
    assert "GONE" in set(frame["name"])
    assert bool(frame.loc[frame["name"] == "GONE", "isDelisted"].iloc[0]) is True
    assert (frame["source"] == hx.SOURCE).all()


def test_selection_frame_records_decile_and_inception():
    selected = [
        {"name": "BIGTIME", "decile": 4, "dayNtlVlm": 2e6, "oi_usd": 1e6,
         "funding": 1e-5, "inception_ms": NOW - 2 * YR_MS, "maxLeverage": 5},
    ]
    f = hx.selection_frame(selected, run_as_of="2026-06-30")
    assert list(f["name"]) == ["BIGTIME"]
    assert int(f["decile"].iloc[0]) == 4
    assert int(f["inception_ms"].iloc[0]) == NOW - 2 * YR_MS
    assert (f["source"] == hx.SOURCE).all()
