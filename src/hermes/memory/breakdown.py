"""Per-key (symbol / skill) aggregation of closed trades for the dashboard.

Pure function over closed journal rows so the per-asset view (used to watch the
SOL-concentration of the validated edge) is unit-tested and never hard-codes an
asset cut — it only surfaces the breakdown for monitoring.
"""

from __future__ import annotations

from typing import Any


def _get(row: Any, key: str) -> Any:
    return getattr(row, key) if hasattr(row, key) else (row.get(key) if isinstance(row, dict) else None)


def aggregate_breakdown(rows: list[Any], key: str) -> dict[str, dict[str, float]]:
    """Group closed trades by ``key`` ('symbol' or 'skill_id') -> {n, pnl_usd,
    win_rate}. Accepts rows as objects (JournalRow) or dicts; None PnL counts as 0."""
    out: dict[str, dict[str, float]] = {}
    for r in rows:
        k = _get(r, key) or "?"
        pnl = float(_get(r, "pnl_usd") or 0.0)
        b = out.setdefault(k, {"n": 0, "pnl_usd": 0.0, "_wins": 0})
        b["n"] += 1
        b["pnl_usd"] += pnl
        if pnl > 0:
            b["_wins"] += 1
    for b in out.values():
        b["win_rate"] = (b["_wins"] / b["n"]) if b["n"] else 0.0
        b["pnl_usd"] = round(b["pnl_usd"], 4)
        del b["_wins"]
    return out
