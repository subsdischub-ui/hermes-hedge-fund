"""Size-0 shadow of the LIVE ``funding_squeeze_continuation`` sleeve.

WHY THIS EXISTS — the live edge was validated over a period of secular crypto
appreciation; the open question (logged in ``signals.py``) is whether it survives a
bear/range regime or is drift-exposure in disguise. The frozen lake cannot answer
that (it ends today and is contaminated by the same bull-beta). So we reproduce the
*exact* live rule on the live feed going forward and log size-0 theoretical trades:
this accrues the clean, un-contaminated bear OOS that will eventually settle it.

FAITHFULNESS CONTRACT (so the OOS measures the live sleeve, not a cousin):
  * the entry decision calls the live ``funding_squeeze_continuation`` — never a copy;
  * the trade return mirrors ``harness.evaluate_signal`` byte-for-byte (same triple
    barrier, ``no_overlap`` via ``busy=t+horizon``, same round-trip cost);
  * the only deliberate difference is FORWARD semantics: a position whose 24-bar
    horizon has not yet elapsed (and no barrier hit) is logged OPEN with mark-to-
    market, never prematurely closed — the idempotent log finalises it later.

This module is read-only on the feed and write-only on the shadow log. It does NOT
register a new trial (the cell is the already-counted live sleeve, not a new search).
"""

from __future__ import annotations

import pandas as pd

from hermes.research.signals import _FUNDC_HOLD_BARS, funding_squeeze_continuation

# Live 7-asset slate (the funding-attached symbols the runtime actually trades).
LIVE_UNIVERSE: tuple[str, ...] = (
    "BTC/USDT", "ETH/USDT", "SOL/USDT", "AVAX/USDT", "SUI/USDT", "ARB/USDT", "WIF/USDT",
)

HOLD_BARS = _FUNDC_HOLD_BARS  # 24 — mirror the frozen live constant, never hard-code
WARMUP = 200
WINDOW_CAP = 400
# Honest crypto round-trip cost = 2*(fee+slippage)/1e4 = 20 bps (validated-harness default).
FEE_BPS = 7.0
SLIPPAGE_BPS = 3.0

LOG_NAME = "shadow/funding_squeeze_continuation"
KEY_COLS = ("symbol", "entry_ts")
TS_COL = "entry_ts"

_TRADE_COLS = [
    "symbol", "entry_ts", "entry_idx", "entry_px", "side", "funding_z", "conviction",
    "stop_pct", "tp_pct", "hold_bars", "exit_ts", "exit_idx", "exit_px",
    "exit_reason", "ret_gross", "ret_net", "closed",
]


def _resolve_long(highs, lows, closes, t, entry_px, stop_pct, tp_pct, horizon, n):
    """Resolve a LONG entry at bar ``t`` (entry=closes[t]).

    Mirrors ``triple_barrier_label`` (stop precedence on same-bar straddle) but also
    distinguishes a genuinely OPEN position from a time-out. Returns
    ``(exit_idx, exit_px, gross, reason, closed)``.
    """
    tp, sl = entry_px * (1 + tp_pct), entry_px * (1 - stop_pct)
    end_avail = min(t + horizon, n - 1)
    for i in range(t + 1, end_avail + 1):
        if lows[i] <= sl:  # stop precedence covers the straddle case
            return i, sl, -stop_pct, "stop", True
        if highs[i] >= tp:
            return i, tp, tp_pct, "tp", True
    if t + horizon <= n - 1:  # full horizon elapsed with no barrier → time exit
        final = closes[t + horizon]
        return t + horizon, final, (final - entry_px) / entry_px, "time", True
    # horizon not yet elapsed and no barrier → still OPEN, mark-to-market at last bar
    final = closes[n - 1]
    return n - 1, final, (final - entry_px) / entry_px, "open", False


def funding_squeeze_trades(
    df: pd.DataFrame,
    symbol: str,
    *,
    hold_bars: int = HOLD_BARS,
    fee_bps: float = FEE_BPS,
    slippage_bps: float = SLIPPAGE_BPS,
    warmup: int = WARMUP,
    window_cap: int = WINDOW_CAP,
    no_overlap: bool = True,
) -> pd.DataFrame:
    """The size-0 shadow trade log for one symbol's OHLCV+funding history.

    ``df`` must carry epoch-ms ``ts`` and a causal ``funding_rate`` column already
    as-of-attached (see ``hermes.research.funding.attach_funding``).
    """
    df = df.reset_index(drop=True)
    n = len(df)
    if n < warmup + 2 or "funding_rate" not in df.columns:
        return pd.DataFrame(columns=_TRADE_COLS)
    highs = df["high"].tolist()
    lows = df["low"].tolist()
    closes = df["close"].tolist()
    ts = df["ts"].tolist()
    rt = 2.0 * (fee_bps + slippage_bps) / 10_000.0
    rows: list[dict] = []
    busy = -1
    for t in range(warmup, n - 1):
        if no_overlap and t < busy:
            continue
        lo = max(0, t + 1 - window_cap)
        sig = funding_squeeze_continuation(df.iloc[lo : t + 1])
        if sig.side != "BUY":
            continue
        if no_overlap:
            busy = t + hold_bars
        entry_px = float(closes[t])
        exit_idx, exit_px, gross, reason, closed = _resolve_long(
            highs, lows, closes, t, entry_px, sig.stop_pct, sig.take_profit_pct, hold_bars, n
        )
        rows.append(
            {
                "symbol": symbol,
                "entry_ts": int(ts[t]),
                "entry_idx": int(t),
                "entry_px": entry_px,
                "side": "BUY",
                "funding_z": float(sig.features["funding_z"]),
                "conviction": float(sig.confidence),
                "stop_pct": float(sig.stop_pct),
                "tp_pct": float(sig.take_profit_pct),
                "hold_bars": int(hold_bars),
                "exit_ts": int(ts[exit_idx]),
                "exit_idx": int(exit_idx),
                "exit_px": float(exit_px),
                "exit_reason": reason,
                "ret_gross": float(gross),
                "ret_net": float(gross - rt),
                "closed": bool(closed),
            }
        )
    if not rows:
        return pd.DataFrame(columns=_TRADE_COLS)
    return pd.DataFrame(rows, columns=_TRADE_COLS)


def shadow_step(frames_by_symbol: dict[str, pd.DataFrame], **kw) -> pd.DataFrame:
    """Concatenate per-symbol shadow trade logs into the full size-0 trade set."""
    parts = [
        funding_squeeze_trades(df, sym, **kw) for sym, df in frames_by_symbol.items()
    ]
    parts = [p for p in parts if not p.empty]
    if not parts:
        return pd.DataFrame(columns=_TRADE_COLS)
    return pd.concat(parts, ignore_index=True).sort_values("entry_ts").reset_index(drop=True)
