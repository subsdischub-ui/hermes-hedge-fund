"""COMEX futures-curve forward collection → point-in-time calendar spreads.

S5 named the metals term structure twice as "the only axis that can break the 2-D
wall", blocked in backtest by yfinance survivorship. This collector escapes that by
accruing the DATED-contract curve forward: each UTC day it snapshots the (up to)
8-maturity chain per metal and derives consecutive-maturity calendar spreads, stamped
at collection time. No backfill — clean PIT term structure builds over 12-18 months.

The network fetch is a thin, swappable, fail-soft seam (``fetch_curve``); the curve→
spread transforms below are pure and unit-tested. Swapping the fetcher (Yahoo dated
contracts → CME settlement files) never touches the PIT/spread logic.
"""

from __future__ import annotations

import pandas as pd

MAX_MATURITIES = 8  # the 8-maturity chain (gold/silver/PGM)
DEFAULT_METALS = ("gold", "silver")  # PGM added when a free source provides it

_CURVE_COLS = ["ts", "metal", "contract", "expiry_ms", "settle", "dte", "seq"]
_SPREAD_COLS = [
    "ts", "metal", "near_contract", "far_contract", "near_settle", "far_settle",
    "near_dte", "far_dte", "spread", "spread_pct", "ann_carry",
]


def curve_snapshot(metal: str, curve: list[dict], ts_ms: int) -> pd.DataFrame:
    """Order a raw curve by expiry, rank maturities, stamp PIT days-to-expiry.

    ``curve`` rows carry ``contract``, ``expiry_ms`` (epoch-ms), ``settle``. Rows with
    a missing expiry or settle are dropped (never fabricated). Keeps the nearest
    ``MAX_MATURITIES`` contracts.
    """
    rows = [
        r for r in curve
        if r.get("expiry_ms") is not None and r.get("settle") is not None
    ]
    rows = sorted(rows, key=lambda r: r["expiry_ms"])[:MAX_MATURITIES]
    out = []
    for seq, r in enumerate(rows):
        out.append(
            {
                "ts": int(ts_ms),
                "metal": metal,
                "contract": r["contract"],
                "expiry_ms": int(r["expiry_ms"]),
                "settle": float(r["settle"]),
                "dte": round((int(r["expiry_ms"]) - int(ts_ms)) / 86_400_000),
                "seq": seq,
            }
        )
    return pd.DataFrame(out, columns=_CURVE_COLS)


def calendar_spreads(snapshot: pd.DataFrame) -> pd.DataFrame:
    """Consecutive-maturity calendar spreads (far - near) from a curve snapshot.

    spread>0 = contango (far richer), spread<0 = backwardation. ``ann_carry`` is the
    spread-pct annualised by the day-gap between the two maturities.
    """
    if snapshot is None or len(snapshot) < 2:
        return pd.DataFrame(columns=_SPREAD_COLS)
    snap = snapshot.sort_values("seq").reset_index(drop=True)
    rows = []
    for i in range(len(snap) - 1):
        near, far = snap.iloc[i], snap.iloc[i + 1]
        spread = float(far["settle"] - near["settle"])
        spread_pct = spread / float(near["settle"]) if near["settle"] else float("nan")
        dt_years = (int(far["dte"]) - int(near["dte"])) / 365.0
        rows.append(
            {
                "ts": int(near["ts"]),
                "metal": near["metal"],
                "near_contract": near["contract"],
                "far_contract": far["contract"],
                "near_settle": float(near["settle"]),
                "far_settle": float(far["settle"]),
                "near_dte": int(near["dte"]),
                "far_dte": int(far["dte"]),
                "spread": spread,
                "spread_pct": spread_pct,
                "ann_carry": spread_pct / dt_years if dt_years > 0 else float("nan"),
            }
        )
    return pd.DataFrame(rows, columns=_SPREAD_COLS)


def fetch_curve(metal: str) -> list[dict]:  # pragma: no cover - network seam
    """Fetch the current dated-contract curve for ``metal`` (read-only).

    Thin, swappable seam. The default targets free Yahoo dated metal contracts; if
    unavailable it raises (the caller is fail-soft per metal). The cleaner upgrade is
    the CME daily settlement file — swap here without touching the transforms above.
    """
    raise NotImplementedError(
        "wire a free dated-contract fetcher (Yahoo dated contracts / CME settlement)"
    )


def collect(metals=DEFAULT_METALS, *, fetch=fetch_curve, now_ms: int) -> dict:
    """Collect curve + calendar-spread snapshots for ``metals``, fail-soft per metal.

    Returns ``{"curve": df, "spreads": df, "blockers": {metal: reason}}``.
    """
    curves, spreads, blockers = [], [], {}
    for metal in metals:
        try:
            raw = fetch(metal)
            snap = curve_snapshot(metal, raw, now_ms)
            if snap.empty:
                blockers[metal] = "empty curve"
                continue
            curves.append(snap)
            sp = calendar_spreads(snap)
            if not sp.empty:
                spreads.append(sp)
        except Exception as e:  # fail-soft: one metal's outage never aborts the rest
            blockers[metal] = f"{type(e).__name__}: {e}"
    curve_df = pd.concat(curves, ignore_index=True) if curves else pd.DataFrame(columns=_CURVE_COLS)
    spread_df = pd.concat(spreads, ignore_index=True) if spreads else pd.DataFrame(columns=_SPREAD_COLS)
    return {"curve": curve_df, "spreads": spread_df, "blockers": blockers}
