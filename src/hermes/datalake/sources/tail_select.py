"""Illiquid-tail universe selector (Stream 0-β) — shared by the Binance-tail and
Hyperliquid-exotic sources.

THE POLICY (Malachie, S0-β): maximize liquidity VARIANCE across the whole spectrum,
NOT pick the most-liquid. β's hypothesis H1 is that carry persists where it cannot be
arbitraged — the illiquid tail — so the studied slice MUST descend into that tail.
Concretely:

  * The caller applies the only HARD filter first: data viability (>=1y continuous
    funding+klines history) + a positive liquidity value. That is backtestability,
    not liquidity selection.
  * This selector then stratifies the eligible candidates into liquidity DECILES and
    samples ~``n_target`` markets ACROSS all deciles, deliberately OVER-WEIGHTING the
    bottom (illiquid) deciles (``bottom_fraction`` of the picks land in the bottom
    ``bottom_deciles``). The remainder spreads over the upper deciles.

Within a decile it takes the most-liquid representatives first — the "most alive" of
each illiquid stratum — so we descend into the illiquid tail without picking literally
dead (near-zero-volume / delisted) markets, which the caller has already screened out.

Pure + DETERMINISTIC (no RNG): the same candidate set always yields the same slice, so
the chosen universe is reproducible and auditable. The "sweet spot" (liquid-enough-to-
harvest vs illiquid-enough-to-be-unarbitraged) is a RESULT β discovers downstream, never
an entry filter here.
"""
from __future__ import annotations

from collections.abc import Iterable
from typing import Any


def _round_robin(
    by_decile: dict[int, list[dict[str, Any]]],
    cursors: dict[int, int],
    decile_range: Iterable[int],
    count: int,
    out: list[dict[str, Any]],
) -> None:
    """Take up to ``count`` rows by cycling through ``decile_range`` one decile at a
    time, advancing per-decile cursors. Exhausted deciles are skipped; stops when
    ``count`` is reached or no decile can yield more."""
    drange = list(decile_range)
    picked = 0
    progressed = True
    while picked < count and progressed:
        progressed = False
        for d in drange:
            if picked >= count:
                break
            pool = by_decile[d]
            c = cursors[d]
            if c < len(pool):
                out.append(pool[c])
                cursors[d] = c + 1
                picked += 1
                progressed = True


def stratified_decile_sample(
    rows: list[dict[str, Any]],
    *,
    liquidity_key: str,
    n_target: int,
    n_deciles: int = 10,
    bottom_deciles: int = 3,
    bottom_fraction: float = 0.4,
) -> list[dict[str, Any]]:
    """Select ~``n_target`` rows stratified by liquidity decile, over-weighting the
    illiquid bottom. Each returned row is a COPY annotated with its ``decile`` (0 =
    most illiquid .. ``n_deciles``-1 = most liquid). Deterministic. See module docstring.

    ``rows`` are the already-eligible candidates (caller screened depth + positive
    liquidity). Returns all of them (annotated) when fewer than ``n_target`` exist.
    """
    if not rows or n_target <= 0:
        return []

    ordered = sorted((dict(r) for r in rows), key=lambda r: float(r[liquidity_key]))
    n = len(ordered)
    for i, r in enumerate(ordered):
        r["decile"] = min(n_deciles - 1, (i * n_deciles) // n)

    if n <= n_target:
        return ordered

    by_decile: dict[int, list[dict[str, Any]]] = {d: [] for d in range(n_deciles)}
    for r in ordered:
        by_decile[r["decile"]].append(r)
    # within a decile: most-liquid (most "alive") first — ordered is ascending, reverse it.
    for d in by_decile:
        by_decile[d] = list(reversed(by_decile[d]))

    n_bottom = round(n_target * bottom_fraction)
    n_rest = n_target - n_bottom
    cursors = {d: 0 for d in range(n_deciles)}
    selected: list[dict[str, Any]] = []
    _round_robin(by_decile, cursors, range(bottom_deciles), n_bottom, selected)
    _round_robin(by_decile, cursors, range(bottom_deciles, n_deciles), n_rest, selected)

    # If some deciles were short of their allocation, backfill from the remaining
    # candidates (deepest-liquidity-first via the ascending ``ordered``) so we still
    # return n_target whenever enough candidates exist.
    if len(selected) < n_target:
        chosen = {id(r) for r in selected}
        for r in ordered:
            if len(selected) >= n_target:
                break
            if id(r) not in chosen:
                selected.append(r)

    return selected[:n_target]
