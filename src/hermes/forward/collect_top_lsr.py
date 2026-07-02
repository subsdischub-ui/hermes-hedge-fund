"""Binance top-trader long/short-ratio forward collection — PIT positioning accrual.

The DeadEndsLedger already buried the DIRECTIONAL use of top-LSR (S2.1 top-lsr
contrarian: "the -0.27 rho(top_lsr, fwd) was a ~30-day MIRAGE ... NEVER re-run
continuous top-LSR" as a signal). This module does NOT resurrect that signal — it is
a pure DATA COLLECTOR, shadow size-0, no strategy attached. The reason to collect
anyway: Binance's public futures-data endpoints only retain ~30 ROLLING days of
history (``limit=30`` is the max window), so every day this cron does not run is
irrecoverable history lost forever. Accruing forward now is what would let a FUTURE,
genuinely new positioning axis (the ledger's own revive_condition: "a positioning
axis with a stable forward sign across regimes ... NEVER re-run continuous top-LSR"
[for the OLD contrarian form]) be evaluated on deep history instead of starting from
zero the day someone finally asks.

Two ratio flavours are both free, keyless, and worth accruing (they diverge — account
ratio counts unique accounts, position ratio is volume/size-weighted):
  * topLongShortAccountRatio — long/short ACCOUNT-count ratio among top traders;
  * topLongShortPositionRatio — long/short POSITION-SIZE ratio among top traders.

Both endpoints are read-only, free, no API key (public Binance USDT-M futures data
API). The network fetch is a thin, swappable, fail-soft seam (``fetch_ratio``); the
row-shaping transform below is pure and unit-tested.
"""

from __future__ import annotations

import pandas as pd

# The live 7-asset slate (mapped {COIN}/USDT:USDT -> {COIN}USDT) plus the two metals
# perps that are a collect-now extension of the forward/ dossier (XAU/XAG, no live
# strategy attached — shadow accrual only).
DEFAULT_SYMBOLS = (
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "AVAXUSDT", "SUIUSDT", "ARBUSDT", "WIFUSDT",
    "XAUUSDT", "XAGUSDT",
)
RATIO_TYPES = ("account", "position")

_ROW_COLS = ["ts", "symbol", "ratio_type", "long_short_ratio", "long_account", "short_account"]


def ratio_rows(symbol: str, ratio_type: str, raw: list[dict]) -> pd.DataFrame:
    """Shape Binance's raw top-LSR payload into PIT rows — empty if unfillable.

    ``raw`` is the endpoint's JSON list of ``{symbol, longShortRatio, longAccount,
    shortAccount, timestamp}``. Rows missing a ratio or timestamp are dropped (never
    fabricated); ``ts`` is Binance's own snapshot ``timestamp`` (already epoch-ms and
    already PIT — this is a read of the exchange's own daily aggregate, not a
    same-day recompute), so no collection-time re-stamping is applied.
    """
    if not raw:
        return pd.DataFrame(columns=_ROW_COLS)
    rows = []
    for r in raw:
        ts = r.get("timestamp")
        ratio = r.get("longShortRatio")
        if ts is None or ratio is None:
            continue
        rows.append(
            {
                "ts": int(ts),
                "symbol": symbol,
                "ratio_type": ratio_type,
                "long_short_ratio": float(ratio),
                "long_account": float(r.get("longAccount")) if r.get("longAccount") is not None else float("nan"),
                "short_account": float(r.get("shortAccount")) if r.get("shortAccount") is not None else float("nan"),
            }
        )
    return pd.DataFrame(rows, columns=_ROW_COLS)


def fetch_top_lsr(symbol: str, ratio_type: str) -> list[dict]:  # pragma: no cover - network seam
    """Fetch the last 30 daily top-LSR points for ``symbol``/``ratio_type`` (read-only).

    Thin, swappable seam over Binance's free USDT-M futures data API:
      * account  -> /futures/data/topLongShortAccountRatio
      * position -> /futures/data/topLongShortPositionRatio
    Both take ``symbol``, ``period=1d``, ``limit=30`` (Binance's own rolling-window
    cap; this is NOT a collector-imposed limit). Raises on any transport/HTTP error —
    the caller (``collect``) is fail-soft per (symbol, ratio_type).
    """
    import httpx

    endpoint = {
        "account": "topLongShortAccountRatio",
        "position": "topLongShortPositionRatio",
    }[ratio_type]
    resp = httpx.get(
        f"https://fapi.binance.com/futures/data/{endpoint}",
        params={"symbol": symbol, "period": "1d", "limit": 30},
        timeout=30.0,
    )
    resp.raise_for_status()
    return resp.json()


def collect(symbols=DEFAULT_SYMBOLS, ratio_types=RATIO_TYPES, *, fetch=fetch_top_lsr) -> dict:
    """Collect top-LSR snapshots for every (symbol, ratio_type), fail-soft per pair.

    Returns ``{"rows": df, "blockers": {"SYM:ratio_type": reason}}``. A symbol Binance
    doesn't publish top-LSR for (e.g. a metals perp not yet covered) is skipped
    cleanly with a blocker instead of aborting the whole run.
    """
    frames, blockers = [], {}
    for symbol in symbols:
        for ratio_type in ratio_types:
            key = f"{symbol}:{ratio_type}"
            try:
                raw = fetch(symbol, ratio_type)
                rows = ratio_rows(symbol, ratio_type, raw)
                if rows.empty:
                    blockers[key] = "empty / no data returned"
                    continue
                frames.append(rows)
            except Exception as e:  # fail-soft: one pair's outage never aborts the rest
                blockers[key] = f"{type(e).__name__}: {e}"
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=_ROW_COLS)
    return {"rows": out, "blockers": blockers}
