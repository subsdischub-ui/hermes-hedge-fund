"""Binance SPOT 4h klines lake client (Stream 0-alpha — the cash-and-carry SPOT leg).

Purely ADDITIVE, layered on the proven ``binance.py`` engine (which it imports,
never forks). It writes ONE new Parquet directory ``raw/binance_spot/klines_4h/``
-> the auto-discovered DuckDB view ``binance_spot_klines_4h``. The existing perp /
funding views are untouched.

WHY this exists: a delta-neutral cash-and-carry (long spot + short perp, equal
notionals) needs the SPOT price leg to compute the real P&L
``pnl_dn = spot_ret + (-perp_ret) + funding_collected - costs``. The lake already
has the perp leg + causal settled funding on the 4h grid
(``binance_klines_4h`` / ``binance_funding_on_4h``); the spot leg is the only gap.

THE JOIN CONTRACT (the whole point): this view uses the SAME slash symbol spelling
(``BTC/USDT``) and the SAME ``ts`` (epoch ms int64) 4h grid as ``binance_klines_4h``,
so the carry frame is ``spot.symbol == perp.symbol`` on ``ts`` — a cross-format join
silently yields zero rows (P2 of the S10 provenance guardrails).

VENUE note: this is the Binance SPOT venue (``ccxt.binance()`` default), tagged
``exchange='binance'`` — distinct from ``binanceusdm`` (the USDT-M PERP venue in
``binance.py``). Symbol ``BTC/USDT`` resolves to the spot market on the spot client.

All real logic is PURE (``fetch_spot_ohlcv_paginated`` over an INJECTED client) and
unit-tested offline; ``run()`` / ``_make_spot_ccxt_client`` do live network and are
never executed by tests. ts is epoch ms int64 everywhere (the lake convention).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from hermes.datalake.core import Manifest, RateLimiter, cached_pull
from hermes.datalake.sources.binance import (
    KLINE_LIMIT,
    gap_count,
    normalize_ohlcv,
)
from hermes.datalake.sources.binance_extended import ALL_PAIRS, FOUR_H_MS

# --------------------------------------------------------------------------- #
# constants
# --------------------------------------------------------------------------- #
SOURCE = "binance_spot"  # raw/binance_spot/klines_4h -> view binance_spot_klines_4h
SPOT_EXCHANGE = "binance"  # the SPOT venue (NOT 'binanceusdm', the perp venue)

# Mirror the perp universe EXACTLY: every perp needs its spot leg, same order.
PAIRS = ALL_PAIRS

# 4h over ~9y of BTC spot history is ~20k bars -> ~20 pages of 1000; give headroom.
KLINE_4H_MAX_BATCHES = 96


# --------------------------------------------------------------------------- #
# PURE transform — unit-tested offline (transport injected)
# --------------------------------------------------------------------------- #
def fetch_spot_ohlcv_paginated(
    client: Any,
    symbol: str,
    timeframe: str,
    *,
    since_ms: int,
    interval_ms: int,
    limit: int = KLINE_LIMIT,
    max_batches: int = KLINE_4H_MAX_BATCHES,
) -> pd.DataFrame:
    """Forward-paginate ``client.fetch_ohlcv`` (SPOT venue) from ``since_ms``.

    Mirrors ``binance.fetch_ohlcv_paginated`` (same proven forward-window walk:
    each batch advances ``since`` to ``last_ts + interval_ms``; the boundary bar
    may re-appear and is dropped by the ``ts`` dedup in :func:`normalize_ohlcv`;
    stops on an empty batch, no forward progress, or ``max_batches``) — but tags
    the frame with the SPOT venue (``exchange='binance'``) instead of the perp
    venue. No look-ahead: it only reshapes already-closed candles.
    """
    collected: list[list[float]] = []
    since = int(since_ms)
    last_seen: int | None = None

    for _ in range(max_batches):
        batch = client.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
        if not batch:
            break
        collected.extend(list(r) for r in batch)
        batch_last = int(batch[-1][0])
        if last_seen is not None and batch_last <= last_seen:
            break  # no forward progress -> exchange has nothing newer
        last_seen = batch_last
        since = batch_last + interval_ms  # next window starts just past last bar

    return normalize_ohlcv(
        collected, symbol=symbol, exchange=SPOT_EXCHANGE, interval=timeframe
    )


# --------------------------------------------------------------------------- #
# LIVE transport (constructed in run(); never executed by tests)
# --------------------------------------------------------------------------- #
def _make_spot_ccxt_client() -> Any:
    """Construct the Binance SPOT ccxt client (default market type = spot)."""
    import ccxt

    return ccxt.binance({"enableRateLimit": True, "options": {"defaultType": "spot"}})


# --------------------------------------------------------------------------- #
# public entrypoint — DO NOT execute in tests (live network)
# --------------------------------------------------------------------------- #
def run(raw_root: Path, manifest: Manifest, *, run_as_of: str) -> None:
    """Pull Binance SPOT 4h klines at MAX depth (``since=0`` -> each pair's listing)
    for the 23-perp universe into ``raw_root/binance_spot/klines_4h/<BASE>_4h.parquet``,
    idempotently and fail-soft. One symbol failure becomes a blocker; the loop
    continues. Public endpoint — no API key required.
    """
    raw_root = Path(raw_root)

    try:
        client = _make_spot_ccxt_client()
        rate = RateLimiter(1200)  # ccxt enforces its own limit; this is a soft cap
    except Exception as e:  # setup failure -> one blocker, nothing pulled
        manifest.blocker(
            source=SOURCE,
            endpoint="klines_4h",
            params={"venue": SPOT_EXCHANGE},
            error=repr(e),
            action="ccxt SPOT client failed to construct; check ccxt install / network",
        )
        return

    for base in PAIRS:
        symbol = f"{base}/USDT"
        target = raw_root / SOURCE / "klines_4h" / f"{base}_4h.parquet"

        def _fetch(_sym=symbol):
            return fetch_spot_ohlcv_paginated(
                client, _sym, "4h",
                since_ms=0, interval_ms=FOUR_H_MS,
                limit=KLINE_LIMIT, max_batches=KLINE_4H_MAX_BATCHES,
            )

        df = cached_pull(
            target, _fetch,
            source=SOURCE, endpoint="klines_4h",
            params={"symbol": symbol, "interval": "4h", "venue": SPOT_EXCHANGE},
            manifest=manifest, ts_col="ts", rate=rate,
        )
        if df is not None and len(df):
            g = gap_count(df, FOUR_H_MS)
            manifest.log(
                source=SOURCE, endpoint="klines_4h",
                params={"symbol": symbol, "interval": "4h"},
                status="partial" if g else "ok",
                rows=len(df), gaps=g, path=str(target),
                note=f"gap_count={g}",
            )
