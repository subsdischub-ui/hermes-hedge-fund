"""Binance USDT-M PERP klines + funding lake client (Stream 0).

Ports the *proven* loops:
  * the forward `since`-pagination + dedup-on-ts + stop-on-empty/no-progress
    engine from ``hermes.data.history.fetch_history`` and
    ``scripts/intraday_fetch.fetch_ohlcv_full`` (1d full history ~3200 bars; 6h
    capped to now-360d);
  * the deep ``startTime`` walk of ``fapi/v1/fundingRate`` from
    ``scripts/intraday_fetch.fetch_funding`` (funding is fully retained).

CRITICAL venue note: the existing ``data/intraday/*.csv`` were pulled from
Binance **SPOT**. Here we pull the **USDT-M PERP** venue
(``ccxt.binance({'options':{'defaultType':'future'}})``, matching
``settings.exchange_primary='binanceusdm'``) so price shares a venue with funding
and (later) CoinGlass OI/LSR. OI/LSR is NOT pulled here — Binance retains it only
~30d (see ``scripts/oi_fetch.py``); that comes from CoinGlass.

All real logic lives in PURE functions (``normalize_ohlcv``, ``normalize_funding``,
``fetch_ohlcv_paginated``, ``fetch_funding_paginated``, ``gap_count``) that take
already-fetched data and return DataFrames — these are unit-tested offline. The
transport (ccxt client, funding HTTP page fetcher) is INJECTABLE so tests fake it;
``run()`` constructs the real transports and does the live pulls via
``cached_pull`` (idempotent, fail-soft).
"""

from __future__ import annotations

import contextlib
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd

from hermes.datalake.core import Manifest, RateLimiter, cached_pull
from hermes.datalake.core import detect_gaps as _detect_gaps

# --------------------------------------------------------------------------- #
# constants
# --------------------------------------------------------------------------- #
SOURCE = "binance"
EXCHANGE = "binanceusdm"  # matches settings.exchange_primary; the USDT-M PERP venue

DAY_MS = 86_400_000
SIX_H_MS = 21_600_000
INTERVAL_MS = {"1d": DAY_MS, "6h": SIX_H_MS}

# Universe: base coins. Symbol on the PERP venue is e.g. 'BTC/USDT'; the funding
# endpoint uses the compact 'BTCUSDT' form.
PAIRS = ["BTC", "ETH", "SOL", "AVAX", "SUI", "ARB", "WIF"]

# 6h is capped to the last 360 days; 1d goes back to each pair's listing (since=0,
# ccxt returns from listing — recent pairs ARB/SUI/WIF naturally start at listing).
SIXH_LOOKBACK_DAYS = 360

# Binance caps klines at 1000 bars/call. 1d full history is ~3200 bars, so bump
# max_batches well above the ~4 pages it needs.
KLINE_LIMIT = 1000
KLINE_MAX_BATCHES = 64

# fapi/v1/fundingRate caps at 1000 rows/call; funding is fully retained from
# listing, so allow many pages (3y of 8h funding ~ 3285 rows -> ~4 pages, but
# headroom is cheap and the short-page stop ends it early).
FUNDING_BASE = "https://fapi.binance.com/fapi/v1/fundingRate"
FUNDING_LIMIT = 1000
FUNDING_MAX_PAGES = 64

_OHLCV_COLS = ["ts", "open", "high", "low", "close", "volume"]
_FUNDING_COLS = ["ts", "funding_rate"]


# --------------------------------------------------------------------------- #
# PURE transforms — unit-tested offline
# --------------------------------------------------------------------------- #
def normalize_ohlcv(
    rows: list[list[float]], *, symbol: str, exchange: str, interval: str
) -> pd.DataFrame:
    """Canonicalise raw ccxt OHLCV rows ``[ts, o, h, l, c, v]`` into the lake frame.

    ``ts`` is epoch **milliseconds** as int64 (ccxt's native kline unit). Adds
    ``symbol``/``exchange``/``interval`` tag columns. Dedups on ``ts`` (keep
    first) and sorts ascending. An empty input yields the canonical empty frame.
    No look-ahead: this only reshapes already-closed candles.
    """
    if not rows:
        df = pd.DataFrame(columns=_OHLCV_COLS)
        df["ts"] = df["ts"].astype("int64")
        df["symbol"] = pd.Series(dtype="object")
        df["exchange"] = pd.Series(dtype="object")
        df["interval"] = pd.Series(dtype="object")
        return df

    df = pd.DataFrame([list(r) for r in rows], columns=_OHLCV_COLS)
    df["ts"] = df["ts"].astype("int64")
    for c in ("open", "high", "low", "close", "volume"):
        df[c] = df[c].astype("float64")
    df = df.drop_duplicates(subset="ts").sort_values("ts").reset_index(drop=True)
    df["symbol"] = symbol
    df["exchange"] = exchange
    df["interval"] = interval
    return df


def normalize_funding(rows: list[dict[str, Any]], *, symbol: str) -> pd.DataFrame:
    """Canonicalise raw ``fapi/v1/fundingRate`` JSON rows into the lake frame.

    Maps ``fundingTime`` -> ``ts`` (epoch ms int64) and ``fundingRate`` (a string
    in the API) -> ``funding_rate`` (float). Adds a ``symbol`` tag, dedups on
    ``ts``, sorts ascending. Empty input -> canonical empty frame.
    """
    if not rows:
        df = pd.DataFrame(columns=_FUNDING_COLS)
        df["ts"] = df["ts"].astype("int64")
        df["funding_rate"] = df["funding_rate"].astype("float64")
        df["symbol"] = pd.Series(dtype="object")
        return df

    df = pd.DataFrame(rows)
    df["ts"] = df["fundingTime"].astype("int64")
    df["funding_rate"] = df["fundingRate"].astype(float)
    df = (
        df[["ts", "funding_rate"]]
        .drop_duplicates(subset="ts")
        .sort_values("ts")
        .reset_index(drop=True)
    )
    df["symbol"] = symbol
    return df


def fetch_ohlcv_paginated(
    client: Any,
    symbol: str,
    timeframe: str,
    *,
    since_ms: int,
    interval_ms: int,
    limit: int = KLINE_LIMIT,
    max_batches: int = KLINE_MAX_BATCHES,
) -> pd.DataFrame:
    """Forward-paginate ``client.fetch_ohlcv`` from ``since_ms`` into one frame.

    Ported from ``hermes.data.history.fetch_history`` /
    ``scripts/intraday_fetch.fetch_ohlcv_full``: each batch advances ``since`` to
    ``last_ts + interval_ms`` (CCXT-style; the boundary bar may re-appear and is
    dropped by the ``ts`` dedup in :func:`normalize_ohlcv`). Stops on an empty
    batch, on no forward progress, or at ``max_batches``.

    Returns the *canonical* OHLCV frame (tag columns included) — accumulation +
    normalization in one place so callers get a ready-to-write frame.
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
        collected, symbol=symbol, exchange=EXCHANGE, interval=timeframe
    )


def fetch_funding_paginated(
    page_fetcher: Callable[[str, int, int], list[dict[str, Any]]],
    symbol: str,
    api_symbol: str,
    *,
    start_ms: int,
    limit: int = FUNDING_LIMIT,
    max_pages: int = FUNDING_MAX_PAGES,
) -> pd.DataFrame:
    """Deep ``startTime``-paginate funding rows into one canonical frame.

    Ported from ``scripts/intraday_fetch.fetch_funding``: ``page_fetcher(api_symbol,
    start_ms, limit)`` returns a list of raw funding dicts with ``fundingTime``
    >= ``start_ms``. We advance ``start`` to ``last_fundingTime + 1`` each page,
    and stop when a page is empty or shorter than ``limit`` (the API returned
    everything it had). ``ts`` dedup guards any boundary overlap.
    """
    collected: list[dict[str, Any]] = []
    start = int(start_ms)

    for _ in range(max_pages):
        rows = page_fetcher(api_symbol, start, limit)
        if not rows:
            break
        collected.extend(rows)
        last = int(rows[-1]["fundingTime"])
        if len(rows) < limit:
            break  # short page -> nothing newer available
        start = last + 1

    return normalize_funding(collected, symbol=symbol)


def gap_count(df: pd.DataFrame, interval_ms: int) -> int:
    """Number of holes (>1.5x interval between consecutive bars) in ``df['ts']``.

    Thin wrapper over ``core.detect_gaps`` returning just the count, suitable for
    a manifest note.
    """
    if df is None or "ts" not in df.columns or len(df) < 2:
        return 0
    return len(_detect_gaps(df["ts"].tolist(), interval_ms))


# --------------------------------------------------------------------------- #
# LIVE transports (constructed in run(); never executed by tests)
# --------------------------------------------------------------------------- #
def _make_ccxt_client() -> Any:
    """Construct the USDT-M PERP ccxt client. Matches settings.exchange_primary."""
    import ccxt

    return ccxt.binance(
        {"enableRateLimit": True, "options": {"defaultType": "future"}}
    )


def _make_funding_page_fetcher(
    client: Any | None = None,
) -> Callable[[str, int, int], list[dict[str, Any]]]:
    """Build a ``page_fetcher(api_symbol, start_ms, limit)`` over fapi/v1/fundingRate.

    Uses an injected httpx-style client when given (tests), else constructs a real
    ``httpx.Client``. Returns the parsed JSON list (raises on transport/HTTP error
    so ``cached_pull`` records a blocker).
    """
    import httpx

    owns = client is None
    cli = client if client is not None else httpx.Client(timeout=60.0)

    def fetch(api_symbol: str, start_ms: int, limit: int) -> list[dict[str, Any]]:
        resp = cli.get(
            FUNDING_BASE,
            params={"symbol": api_symbol, "startTime": int(start_ms), "limit": int(limit)},
        )
        resp.raise_for_status()
        data = resp.json()
        return data if isinstance(data, list) else []

    # Stash the client so run() can close it after the funding loop.
    fetch._client = cli  # type: ignore[attr-defined]
    fetch._owns = owns  # type: ignore[attr-defined]
    return fetch


# --------------------------------------------------------------------------- #
# public entrypoint — DO NOT execute in tests (live network)
# --------------------------------------------------------------------------- #
def run(raw_root: Path, manifest: Manifest, *, run_as_of: str) -> None:
    """Pull Binance PERP klines (1d full history, 6h last-360d) + funding for the
    universe into ``raw_root/binance/...`` Parquet, idempotently and fail-soft.

    Layout:
      raw_root/binance/klines/<BASE>_1d.parquet
      raw_root/binance/klines/<BASE>_6h.parquet
      raw_root/binance/funding/<BASE>.parquet

    Each successful frame gets a ``detect_gaps`` count passed into the manifest
    note. One symbol/series failure becomes a blocker and the loop continues.
    API keys (none required for these public endpoints) are read from os.environ
    at call time; the orchestrator loads .env first.
    """
    raw_root = Path(raw_root)
    now_ms = int(time.time() * 1000)

    # Read any creds at call time (public endpoints need none, but honor the
    # contract / future-proof a keyed client).
    _api_key = os.environ.get("BINANCE_API_KEY")

    # --- klines (USDT-M PERP via ccxt) ---
    try:
        ccxt_client = _make_ccxt_client()
        rate = RateLimiter(1200)  # ccxt enforces its own limit; this is a soft cap
    except Exception as e:  # setup failure -> one blocker, skip klines, still try funding
        manifest.blocker(
            source=SOURCE,
            endpoint="klines",
            params={"venue": EXCHANGE},
            error=repr(e),
            action="ccxt USDT-M PERP client failed to construct; check ccxt install / network",
        )
        ccxt_client = None
        rate = None

    if ccxt_client is not None:
        for base in PAIRS:
            symbol = f"{base}/USDT"
            for interval in ("1d", "6h"):
                interval_ms = INTERVAL_MS[interval]
                since_ms = 0 if interval == "1d" else now_ms - SIXH_LOOKBACK_DAYS * DAY_MS
                target = raw_root / SOURCE / "klines" / f"{base}_{interval}.parquet"

                def _fetch(_sym=symbol, _iv=interval, _ivms=interval_ms, _since=since_ms):
                    return fetch_ohlcv_paginated(
                        ccxt_client, _sym, _iv,
                        since_ms=_since, interval_ms=_ivms,
                        limit=KLINE_LIMIT, max_batches=KLINE_MAX_BATCHES,
                    )

                df = cached_pull(
                    target, _fetch,
                    source=SOURCE, endpoint="klines",
                    params={"symbol": symbol, "interval": interval, "venue": EXCHANGE},
                    manifest=manifest, ts_col="ts", rate=rate,
                )
                if df is not None and len(df):
                    g = gap_count(df, interval_ms)
                    manifest.log(
                        source=SOURCE, endpoint="klines",
                        params={"symbol": symbol, "interval": interval},
                        status="partial" if g else "ok",
                        rows=len(df), gaps=g, path=str(target),
                        note=f"gap_count={g}",
                    )

    # --- funding (fapi/v1/fundingRate, deep startTime walk) ---
    try:
        page_fetcher = _make_funding_page_fetcher()
        frate = RateLimiter(240)  # well under fapi's IP weight ceiling
    except Exception as e:
        manifest.blocker(
            source=SOURCE, endpoint="fundingRate", params={},
            error=repr(e),
            action="httpx client failed to construct for funding; check httpx install",
        )
        return

    try:
        for base in PAIRS:
            symbol = f"{base}/USDT"
            api_symbol = f"{base}USDT"
            target = raw_root / SOURCE / "funding" / f"{base}.parquet"

            def _fetch_f(_sym=symbol, _api=api_symbol):
                return fetch_funding_paginated(
                    page_fetcher, _sym, _api,
                    start_ms=0, limit=FUNDING_LIMIT, max_pages=FUNDING_MAX_PAGES,
                )

            df = cached_pull(
                target, _fetch_f,
                source=SOURCE, endpoint="fundingRate",
                params={"symbol": api_symbol},
                manifest=manifest, ts_col="ts", rate=frate,
            )
            if df is not None and len(df):
                # Funding cadence on Binance is 8h (3 * 3_600_000 ms); flag holes.
                g = gap_count(df, 8 * 3_600_000)
                manifest.log(
                    source=SOURCE, endpoint="fundingRate",
                    params={"symbol": api_symbol},
                    status="partial" if g else "ok",
                    rows=len(df), gaps=g, path=str(target),
                    note=f"gap_count={g}",
                )
    finally:
        cli = getattr(page_fetcher, "_client", None)
        if cli is not None and getattr(page_fetcher, "_owns", False):
            with contextlib.suppress(Exception):
                cli.close()
