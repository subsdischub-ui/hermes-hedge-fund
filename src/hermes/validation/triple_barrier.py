"""Triple-barrier labeling (López de Prado, Advances in Financial ML).

From an entry, scan forward up to `horizon` bars and return which barrier is hit
first: take-profit (win), stop-loss (loss), or — if neither — a time-out labelled
by the sign of the final return. When a single bar's range straddles both
barriers we resolve to the STOP (conservative: assume the adverse move first).
"""

from __future__ import annotations

from collections.abc import Sequence


def triple_barrier_label(
    highs: Sequence[float],
    lows: Sequence[float],
    closes: Sequence[float],
    entry_idx: int,
    entry_px: float,
    side: str,
    stop_pct: float,
    tp_pct: float,
    horizon: int,
) -> tuple[int, float]:
    """Return (outcome, gross_return_fraction). outcome 1 = win, 0 = loss."""
    long = side == "BUY"
    if long:
        tp, sl = entry_px * (1 + tp_pct), entry_px * (1 - stop_pct)
    else:
        tp, sl = entry_px * (1 - tp_pct), entry_px * (1 + stop_pct)

    end = min(entry_idx + horizon, len(closes) - 1)
    for i in range(entry_idx + 1, end + 1):
        hi, lo = highs[i], lows[i]
        stop_hit = (lo <= sl) if long else (hi >= sl)
        tp_hit = (hi >= tp) if long else (lo <= tp)
        if stop_hit:  # stop precedence covers the same-bar straddle case
            return 0, -stop_pct
        if tp_hit:
            return 1, tp_pct

    final = closes[end]
    ret = (final - entry_px) / entry_px if long else (entry_px - final) / entry_px
    return (1 if ret > 0 else 0), float(ret)
