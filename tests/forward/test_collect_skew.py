"""FORWARD — Deribit 25Δ skew forward-log (reuses the S4 surface-snapshot transform).

S4 already started this forward-log; the unified cron just wraps it so all three
flux share one fail-soft, idempotent collection path. The numeric RR25/skew logic is
owned + tested by S4 (``compute_surface_snapshot``); here we test only the wrapper's
own behaviour: delegation, empty-chain → empty (no fabrication), and fail-soft per ccy.
"""

from __future__ import annotations

import datetime as dt

from hermes.forward import collect_skew as sk

_NOW = dt.datetime(2026, 6, 29, tzinfo=dt.UTC)
_NOW_MS = int(_NOW.timestamp() * 1000)


def test_empty_book_yields_empty_snapshot_no_fabrication():
    out = sk.skew_snapshot("BTC", [], now=_NOW, ts_ms=_NOW_MS, delta_fn=lambda inst: None)
    assert out.empty


def test_collect_is_fail_soft_per_ccy():
    def fetch_book(ccy):
        if ccy == "ETH":
            raise RuntimeError("deribit unreachable")
        return []  # BTC: reachable but empty chain → recorded, not crashed

    out = sk.collect(
        ("BTC", "ETH"),
        fetch_book=fetch_book,
        delta_fn_factory=lambda ccy: (lambda inst: None),
        now=_NOW,
        now_ms=_NOW_MS,
    )
    assert "ETH" in out["blockers"]
    assert "BTC" in out["blockers"]  # empty chain is a soft blocker, not a crash
    assert out["skew"].empty
