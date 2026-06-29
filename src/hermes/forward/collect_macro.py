"""Macro forward collection: SEP dot-plot vs Fed-Funds-futures-implied path.

S7.3's forward-collectable frontier: snapshot the Fed's SEP median projected policy
path and the market-implied path from Fed-Funds futures, then log the per-horizon gap
(Fed minus market) — a point-in-time measure of policy disagreement. Forward accrual
only (the SEP updates quarterly at FOMC; FF futures move daily).

The fetchers are thin, swappable, fail-soft seams (SEP from the Fed's release; FF path
from CME FedWatch / ZQ futures). The matching/gap transform below is pure + tested.
"""

from __future__ import annotations

import pandas as pd

_GAP_COLS = ["ts", "horizon", "sep_median", "ff_implied", "gap"]


def dot_vs_ff_gap(sep: list[dict], ff: list[dict], ts_ms: int) -> pd.DataFrame:
    """Per-horizon gap = SEP median minus FF-implied, for horizons present in BOTH.

    ``sep`` rows: ``horizon`` (e.g. "2026"), ``median`` (projected policy rate, %).
    ``ff`` rows:  ``horizon``, ``implied`` (futures-implied rate, %). Positive gap =
    Fed projects above the market (hawkish); negative = dovish.
    """
    ff_by_h = {r["horizon"]: float(r["implied"]) for r in ff}
    rows = []
    for r in sep:
        h = r["horizon"]
        if h not in ff_by_h:
            continue  # only horizons priced by BOTH the Fed and the market
        med = float(r["median"])
        rows.append(
            {
                "ts": int(ts_ms),
                "horizon": h,
                "sep_median": med,
                "ff_implied": ff_by_h[h],
                "gap": med - ff_by_h[h],
            }
        )
    return pd.DataFrame(rows, columns=_GAP_COLS)


def fetch_sep() -> list[dict]:  # pragma: no cover - network seam
    """Fetch the latest SEP dot-plot median path (read-only). Thin swappable seam."""
    raise NotImplementedError("wire the Fed SEP release fetcher (quarterly)")


def fetch_ff() -> list[dict]:  # pragma: no cover - network seam
    """Fetch the Fed-Funds-futures-implied path (read-only). Thin swappable seam."""
    raise NotImplementedError("wire a free FF-futures path fetcher (CME FedWatch / ZQ)")


def collect(*, fetch_sep=fetch_sep, fetch_ff=fetch_ff, now_ms: int) -> dict:
    """Collect the dot-vs-FF gap snapshot, fail-soft. Returns ``{"gap", "blockers"}``."""
    blockers = {}
    sep = ff = None
    try:
        sep = fetch_sep()
    except Exception as e:
        blockers["sep"] = f"{type(e).__name__}: {e}"
    try:
        ff = fetch_ff()
    except Exception as e:
        blockers["ff"] = f"{type(e).__name__}: {e}"
    if sep is None or ff is None:
        return {"gap": pd.DataFrame(columns=_GAP_COLS), "blockers": blockers}
    return {"gap": dot_vs_ff_gap(sep, ff, now_ms), "blockers": blockers}
