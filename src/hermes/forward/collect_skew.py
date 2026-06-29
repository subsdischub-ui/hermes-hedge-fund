"""Deribit 25Δ skew forward-log — thin wrapper over the S4 surface-snapshot transform.

S4 owns the numeric RR25/skew/term-slope computation (``compute_surface_snapshot``,
already tested + fail-soft, never fabricates NaN fields). This wrapper plugs it into
the unified forward cron so the skew flux shares one collection/idempotency path with
COMEX and macro. Read-only on the Deribit public API.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd

from hermes.datalake.sources.deribit import (
    book_summary_to_rows,
    compute_surface_snapshot,
)

DEFAULT_CCYS = ("BTC", "ETH")


def skew_snapshot(
    ccy: str,
    book_summary: list[dict],
    *,
    now: dt.datetime,
    ts_ms: int,
    delta_fn,
) -> pd.DataFrame:
    """One PIT skew row for ``ccy`` from a Deribit book summary — empty if unfillable."""
    rows = book_summary_to_rows(book_summary or [])
    snap = compute_surface_snapshot(ccy, rows, now=now, ts_ms=ts_ms, delta_fn=delta_fn)
    if snap is None or snap.empty:
        return pd.DataFrame()
    return snap


def collect(
    ccys=DEFAULT_CCYS,
    *,
    fetch_book,
    delta_fn_factory,
    now: dt.datetime,
    now_ms: int,
) -> dict:
    """Collect skew snapshots for ``ccys``, fail-soft per ccy.

    ``fetch_book(ccy)`` returns the public book summary; ``delta_fn_factory(ccy)``
    returns a greeks-delta lookup. Returns ``{"skew": df, "blockers": {ccy: reason}}``.
    """
    snaps, blockers = [], {}
    for ccy in ccys:
        try:
            book = fetch_book(ccy)
            snap = skew_snapshot(
                ccy, book, now=now, ts_ms=now_ms, delta_fn=delta_fn_factory(ccy)
            )
            if snap.empty:
                blockers[ccy] = "empty / unfillable chain"
                continue
            snaps.append(snap)
        except Exception as e:  # fail-soft: one ccy's outage never aborts the rest
            blockers[ccy] = f"{type(e).__name__}: {e}"
    skew_df = pd.concat(snaps, ignore_index=True) if snaps else pd.DataFrame()
    return {"skew": skew_df, "blockers": blockers}
