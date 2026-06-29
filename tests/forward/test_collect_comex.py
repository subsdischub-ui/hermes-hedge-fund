"""FORWARD — COMEX futures-curve forward collection (PIT calendar spreads).

The one axis S5 named twice as able to break the 2-D metals wall is the term
structure — blocked in backtest by yfinance survivorship (the continuous front
splices contracts and loses the curve). The escape is to accrue the DATED-contract
curve forward: snapshot the 8-maturity chain daily, derive point-in-time calendar
spreads, and in 12-18 months own a clean, survivorship-free term-structure panel.

The pure transforms are unit-tested (the network fetch is a thin fail-soft seam):
  * the curve snapshot orders by expiry, ranks maturities, and stamps days-to-expiry
    against the COLLECTION time (PIT), not with hindsight;
  * calendar spreads are consecutive-maturity (far - near), sign = contango/backwardation.
"""

from __future__ import annotations

from hermes.forward import collect_comex as cx

_DAY_MS = 86_400_000
_NOW = 1_780_000_000_000  # collection time


def _curve():  # gold curve: 4 maturities, rising settles (contango)
    return [
        {"contract": "GCQ26", "expiry_ms": _NOW + 60 * _DAY_MS, "settle": 2000.0},
        {"contract": "GCV26", "expiry_ms": _NOW + 120 * _DAY_MS, "settle": 2010.0},
        {"contract": "GCZ26", "expiry_ms": _NOW + 180 * _DAY_MS, "settle": 2025.0},
        {"contract": "GCG27", "expiry_ms": _NOW + 240 * _DAY_MS, "settle": 2045.0},
    ]


def test_curve_snapshot_orders_ranks_and_dates_pit():
    shuffled = [_curve()[2], _curve()[0], _curve()[3], _curve()[1]]
    snap = cx.curve_snapshot("gold", shuffled, _NOW)
    assert list(snap["seq"]) == [0, 1, 2, 3]
    assert list(snap["contract"]) == ["GCQ26", "GCV26", "GCZ26", "GCG27"]
    # days-to-expiry computed against the collection ts (PIT), nearest first
    assert snap["dte"].iloc[0] == 60
    assert (snap["ts"] == _NOW).all()


def test_curve_snapshot_caps_at_eight_maturities():
    many = [
        {"contract": f"C{i}", "expiry_ms": _NOW + (i + 1) * 30 * _DAY_MS, "settle": 100.0 + i}
        for i in range(12)
    ]
    snap = cx.curve_snapshot("silver", many, _NOW)
    assert len(snap) == 8  # nearest 8 maturities only


def test_calendar_spreads_consecutive_and_contango_positive():
    snap = cx.curve_snapshot("gold", _curve(), _NOW)
    sp = cx.calendar_spreads(snap)
    assert len(sp) == 3  # n-1 consecutive spreads
    assert list(sp["near_contract"]) == ["GCQ26", "GCV26", "GCZ26"]
    assert (sp["spread"] > 0).all()  # rising settles → contango → positive far-near
    assert sp["spread"].iloc[0] == 10.0  # 2010 - 2000


def test_backwardation_gives_negative_spread():
    inv = [
        {"contract": "A", "expiry_ms": _NOW + 30 * _DAY_MS, "settle": 100.0},
        {"contract": "B", "expiry_ms": _NOW + 60 * _DAY_MS, "settle": 95.0},
    ]
    sp = cx.calendar_spreads(cx.curve_snapshot("pgm", inv, _NOW))
    assert sp["spread"].iloc[0] == -5.0


def test_too_few_maturities_yields_no_spreads():
    one = [{"contract": "A", "expiry_ms": _NOW + 30 * _DAY_MS, "settle": 100.0}]
    assert cx.calendar_spreads(cx.curve_snapshot("gold", one, _NOW)).empty


def test_drops_rows_missing_settle_or_expiry():
    bad = [
        *_curve(),
        {"contract": "BADX", "expiry_ms": None, "settle": 1.0},
        {"contract": "BADY", "expiry_ms": _NOW + 300 * _DAY_MS, "settle": None},
    ]
    snap = cx.curve_snapshot("gold", bad, _NOW)
    assert "BADX" not in set(snap["contract"])
    assert "BADY" not in set(snap["contract"])


def test_collect_is_fail_soft_per_metal():
    # one metal's fetch raises → recorded as a blocker, the other still collects
    def fetch(metal):
        if metal == "silver":
            raise RuntimeError("feed down")
        return _curve()

    out = cx.collect(("gold", "silver"), fetch=fetch, now_ms=_NOW)
    assert not out["spreads"].empty
    assert set(out["spreads"]["metal"]) == {"gold"}
    assert "silver" in out["blockers"]
