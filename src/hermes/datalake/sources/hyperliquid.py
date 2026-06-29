"""Hyperliquid PUBLIC-API lake client (Stream 0-alpha — cross-venue carry + hourly funding).

DATALAKE source (read-only history). DISTINCT from ``hermes.execution.hyperliquid``,
which is the live-order EXECUTION adapter (off-limits, the human merge gate). This
module only GETs public market data via ``POST https://api.hyperliquid.xyz/info``
(no API key, no KYC; reachable from France for data). It NEVER places an order.

Two additive Parquet directories -> two auto-discovered DuckDB views:

  * ``hyperliquid_funding_1h`` — HOURLY settled funding. HL pays funding every hour
    at 1/8 of the 8h-computed rate; ``fundingHistory`` returns the per-HOUR rate, so
    the stored ``funding_rate`` is hourly. To compare against Binance (8h/4h), put
    both on an APR basis: ``annualize_funding(rate, interval_hours)`` with HL
    interval_hours=1 (-> x8760). [VÉRIFIÉ doc: "paid every hour at one eighth of the
    computed rate"].

  * ``hyperliquid_klines`` — perp OHLCV at 4h (aligned to the Binance carry grid).
    ``candleSnapshot`` is HARD-CAPPED at 5000 rows/request -> fetch paginates by time.

DEPTH CAVEAT (document loudly, do not let downstream over-fit a short backtest): HL
perps are SHALLOW vs Binance's 7y. The real inception is PROBED at ingest (the mandate
guessed ~Nov 2024; secondary research suggested ~Feb 2023 [PROBABLE]); the COVERAGE
doc reports the actual earliest ts the API returns. This is a PRIMARILY-FORWARD axis.

Symbol is HL-native BARE (``BTC``). The cross-venue funding join normalizes both venues
to bare via ``s10_provenance.normalize_symbol``. ts is epoch ms int64 (lake convention).

All real logic is PURE (normalize_* + fetch_*_paginated over an INJECTED ``poster``)
and unit-tested offline; ``run()`` / ``_make_hl_poster`` do live network, never in tests.
"""

from __future__ import annotations

import contextlib
import time
from pathlib import Path
from typing import Any

import pandas as pd

from hermes.datalake.core import Manifest, RateLimiter, cached_pull
from hermes.datalake.sources.binance_extended import ALL_PAIRS

# --------------------------------------------------------------------------- #
# constants
# --------------------------------------------------------------------------- #
SOURCE = "hyperliquid"
EXCHANGE = "hyperliquid"
INFO_URL = "https://api.hyperliquid.xyz/info"

HOUR_MS = 3_600_000
FOUR_H_MS = 14_400_000
HL_INTERVAL_MS = {"1h": HOUR_MS, "4h": FOUR_H_MS}

# Conservative pre-inception floor to walk forward FROM. The API returns from the
# perp's TRUE inception within any window, so an early floor never breaks the walk
# (the first page is non-empty at real inception); the actual earliest ts is what
# we record. 2023-01-01 UTC predates every plausible HL inception estimate.
HL_HISTORY_START_MS = 1_672_531_200_000

# candleSnapshot caps at 5000 rows/call; 2y of 4h is ~4400 bars -> ~1 page, but a
# deeper history or 1h cadence needs several. Headroom is cheap; the empty/no-progress
# stop ends it early.
CANDLE_MAX_BATCHES = 64
FUNDING_MAX_PAGES = 256  # hourly funding over 2y is ~17.5k rows; pages may cap ~500

# Carry-relevant universe: the Binance perp bases. run() intersects this with HL's
# live universe (metaAndAssetCtxs) and records any base absent on HL.
HL_COINS = ALL_PAIRS

_KLINE_COLS = ["ts", "open", "high", "low", "close", "volume"]
_FUNDING_COLS = ["ts", "funding_rate", "premium", "symbol"]


# --------------------------------------------------------------------------- #
# PURE transforms — unit-tested offline
# --------------------------------------------------------------------------- #
def normalize_hl_klines(
    rows: list[dict[str, Any]], *, symbol: str, interval: str
) -> pd.DataFrame:
    """Canonicalise ``candleSnapshot`` rows into the lake OHLCV frame.

    HL candles are ``{t,T,o,h,l,c,v,n,i,s}`` with prices as strings; ``t`` is the
    OPEN time (epoch ms). Maps to ``[ts, open, high, low, close, volume]`` + tag
    columns, dedups on ``ts`` (keep first), sorts ascending. Empty -> canonical
    empty frame. No look-ahead: only reshapes already-closed candles.
    """
    if not rows:
        df = pd.DataFrame(columns=_KLINE_COLS)
        df["ts"] = df["ts"].astype("int64")
        for c in ("open", "high", "low", "close", "volume"):
            df[c] = df[c].astype("float64")
        df["symbol"] = pd.Series(dtype="object")
        df["exchange"] = pd.Series(dtype="object")
        df["interval"] = pd.Series(dtype="object")
        return df

    raw = pd.DataFrame(list(rows))
    df = pd.DataFrame()
    df["ts"] = raw["t"].astype("int64")
    df["open"] = raw["o"].astype("float64")
    df["high"] = raw["h"].astype("float64")
    df["low"] = raw["l"].astype("float64")
    df["close"] = raw["c"].astype("float64")
    df["volume"] = raw["v"].astype("float64")
    df = df.drop_duplicates(subset="ts").sort_values("ts").reset_index(drop=True)
    df["symbol"] = symbol
    df["exchange"] = EXCHANGE
    df["interval"] = interval
    return df


def normalize_hl_funding(rows: list[dict[str, Any]], *, coin: str) -> pd.DataFrame:
    """Canonicalise ``fundingHistory`` rows into the lake funding frame.

    HL rows are ``{coin, fundingRate, premium, time}`` (rates as strings); ``time``
    is epoch ms, ``fundingRate`` is the per-HOUR settled rate. Maps to
    ``[ts, funding_rate, premium, symbol]``, dedups on ``ts``, sorts ascending.
    Empty -> canonical empty frame.
    """
    if not rows:
        df = pd.DataFrame(columns=_FUNDING_COLS)
        df["ts"] = df["ts"].astype("int64")
        df["funding_rate"] = df["funding_rate"].astype("float64")
        df["premium"] = df["premium"].astype("float64")
        df["symbol"] = pd.Series(dtype="object")
        return df

    raw = pd.DataFrame(list(rows))
    df = pd.DataFrame()
    df["ts"] = raw["time"].astype("int64")
    df["funding_rate"] = pd.to_numeric(raw["fundingRate"], errors="coerce").astype("float64")
    df["premium"] = pd.to_numeric(raw.get("premium"), errors="coerce").astype("float64")
    df = df.drop_duplicates(subset="ts").sort_values("ts").reset_index(drop=True)
    df["symbol"] = coin
    return df[_FUNDING_COLS]


def parse_hl_universe(meta_and_ctxs: Any) -> list[str]:
    """Extract perp coin names from a ``metaAndAssetCtxs`` response.

    Canonical shape is ``[{"universe": [{"name": ...}, ...]}, [assetCtxs...]]``;
    also tolerates the universe list given directly as ``meta_and_ctxs[0]``.
    """
    first = meta_and_ctxs[0]
    universe = first["universe"] if isinstance(first, dict) and "universe" in first else first
    return [asset["name"] for asset in universe]


# --------------------------------------------------------------------------- #
# PURE paginators — forward walk over an INJECTED poster(payload) -> json
# --------------------------------------------------------------------------- #
def fetch_hl_candles_paginated(
    poster: Any,
    coin: str,
    interval: str,
    *,
    start_ms: int,
    end_ms: int,
    interval_ms: int,
    max_batches: int = CANDLE_MAX_BATCHES,
) -> pd.DataFrame:
    """BACKWARD-paginate ``candleSnapshot`` (5000-row cap) into one canonical frame.

    ``candleSnapshot`` is END-ANCHORED: it returns the most-recent <=5000 candles in
    [startTime, endTime] and IGNORES an early ``startTime`` (verified live — a far-past
    ``startTime`` still yields only the last ~5000 ending at ``endTime``). So to recover
    OLD history we hold ``startTime`` at the floor and step ``endTime`` BACK to just
    before the earliest candle each batch. Stops on an empty batch, no backward
    progress, reaching the ``start_ms`` floor, or ``max_batches``. The ``ts`` dedup in
    :func:`normalize_hl_klines` guards any boundary overlap. ``interval_ms`` is unused
    here (the window walk is driven by the earliest returned candle) and kept for a
    uniform signature.
    """
    collected: list[dict[str, Any]] = []
    end = int(end_ms)
    floor = int(start_ms)
    earliest_seen: int | None = None

    for _ in range(max_batches):
        batch = poster(
            {
                "type": "candleSnapshot",
                "req": {"coin": coin, "interval": interval,
                        "startTime": floor, "endTime": end},
            }
        )
        if not batch:
            break
        collected.extend(batch)
        batch_first = int(batch[0]["t"])  # candles are ascending -> [0] is earliest
        if earliest_seen is not None and batch_first >= earliest_seen:
            break  # no backward progress -> API has nothing older
        earliest_seen = batch_first
        if batch_first <= floor:
            break  # reached the requested floor
        end = batch_first - 1  # next window ends just before the earliest seen

    return normalize_hl_klines(collected, symbol=coin, interval=interval)


def fetch_hl_funding_paginated(
    poster: Any,
    coin: str,
    *,
    start_ms: int,
    end_ms: int,
    max_pages: int = FUNDING_MAX_PAGES,
) -> pd.DataFrame:
    """Forward-paginate ``fundingHistory`` into one canonical frame.

    Advances ``startTime`` to ``last_time + 1`` each page; stops on an empty page,
    no forward progress, or ``max_pages``. ``ts`` dedup guards boundary overlap.
    """
    collected: list[dict[str, Any]] = []
    start = int(start_ms)
    last_seen: int | None = None

    for _ in range(max_pages):
        rows = poster(
            {"type": "fundingHistory", "coin": coin,
             "startTime": start, "endTime": int(end_ms)}
        )
        if not rows:
            break
        collected.extend(rows)
        last = int(rows[-1]["time"])
        if last_seen is not None and last <= last_seen:
            break
        last_seen = last
        start = last + 1

    return normalize_hl_funding(collected, coin=coin)


# --------------------------------------------------------------------------- #
# LIVE transport (constructed in run(); never executed by tests)
# --------------------------------------------------------------------------- #
def hl_request(
    client: Any,
    payload: dict,
    *,
    rate: Any | None = None,
    sleep: Any = time.sleep,
    max_retries: int = 6,
) -> Any:
    """POST ``payload`` to /info with PER-REQUEST throttle + 429 backoff.

    Calls ``rate.wait()`` (if given) before EACH attempt — this is the fix for the
    burst-429: the rate limiter must space every pagination request, not just the
    once-per-coin ``cached_pull`` call. On HTTP 429 it sleeps an exponential backoff
    (capped 30s) and retries; after ``max_retries`` it surfaces the last error.
    Returns the parsed JSON on success.
    """
    resp = None
    for attempt in range(max_retries):
        if rate is not None:
            rate.wait()
        resp = client.post(INFO_URL, json=payload)
        if getattr(resp, "status_code", 200) == 429:
            sleep(min(2.0 ** attempt, 30.0))
            continue
        resp.raise_for_status()
        return resp.json()
    if resp is not None:  # exhausted -> surface the last 429
        resp.raise_for_status()
    return []


def _make_hl_poster(client: Any | None = None, rate: Any | None = None):
    """Build a ``poster(payload: dict) -> parsed json`` over POST /info.

    Uses an injected httpx-style client when given (tests), else a real one. Routes
    every request through :func:`hl_request` so throttling + 429 backoff apply to each
    pagination call. Raises on non-429 transport/HTTP error so ``cached_pull`` records
    a blocker.
    """
    import httpx

    owns = client is None
    cli = client if client is not None else httpx.Client(timeout=60.0)

    def poster(payload: dict) -> Any:
        return hl_request(cli, payload, rate=rate)

    poster._client = cli  # type: ignore[attr-defined]
    poster._owns = owns  # type: ignore[attr-defined]
    return poster


# --------------------------------------------------------------------------- #
# public entrypoint — DO NOT execute in tests (live network)
# --------------------------------------------------------------------------- #
def run(raw_root: Path, manifest: Manifest, *, run_as_of: str) -> None:
    """Pull HL hourly funding + 4h klines for the carry-relevant universe into
    ``raw_root/hyperliquid/{funding_1h,klines}/`` Parquet, idempotently + fail-soft.

    Discovers the live universe via ``metaAndAssetCtxs`` (logged for COVERAGE / HIP-3
    visibility), intersects it with the Binance perp bases, and records any base
    absent on HL. One coin failure becomes a blocker; the loop continues. Public
    endpoint — no key required.
    """
    raw_root = Path(raw_root)
    now_ms = int(time.time() * 1000)

    try:
        # PER-REQUEST throttle lives inside the poster (every pagination call is
        # spaced), well under HL's 1200 weight/min/IP ceiling; hl_request adds 429
        # backoff on top. ~45/min ≈ 1.33s/req keeps a margin for candleSnapshot weight.
        rate = RateLimiter(45)
        poster = _make_hl_poster(rate=rate)
    except Exception as e:
        manifest.blocker(
            source=SOURCE, endpoint="info", params={},
            error=repr(e),
            action="httpx client failed to construct for Hyperliquid /info; check httpx install",
        )
        return

    # --- universe discovery (fail-soft; fall back to our bases) ---
    try:
        meta = poster({"type": "metaAndAssetCtxs"})
        universe = parse_hl_universe(meta)
        manifest.log(
            source=SOURCE, endpoint="metaAndAssetCtxs", params={}, status="ok",
            rows=len(universe), note=f"hl_universe={len(universe)} markets (incl. HIP-3 if listed)",
        )
    except Exception as e:
        manifest.blocker(
            source=SOURCE, endpoint="metaAndAssetCtxs", params={},
            error=repr(e),
            action="HL universe discovery failed; falling back to the Binance base list",
        )
        universe = list(HL_COINS)

    uni_set = set(universe)
    coins = [c for c in HL_COINS if c in uni_set]
    absent = [c for c in HL_COINS if c not in uni_set]
    if not coins:  # discovery returned something unexpected -> try our bases anyway
        coins = list(HL_COINS)
    if absent:
        manifest.log(
            source=SOURCE, endpoint="universe_overlap", params={}, status="ok",
            rows=len(coins), note=f"binance_bases_absent_on_hl={absent}",
        )

    try:
        for coin in coins:
            # --- hourly funding ---
            ftarget = raw_root / SOURCE / "funding_1h" / f"{coin}.parquet"

            def _fetch_f(_c=coin):
                return fetch_hl_funding_paginated(
                    poster, _c, start_ms=HL_HISTORY_START_MS, end_ms=now_ms,
                )

            fdf = cached_pull(
                ftarget, _fetch_f,
                source=SOURCE, endpoint="fundingHistory",
                params={"coin": coin}, manifest=manifest, ts_col="ts", rate=None,
            )
            if fdf is not None and len(fdf):
                manifest.log(
                    source=SOURCE, endpoint="fundingHistory", params={"coin": coin},
                    status="ok", rows=len(fdf), path=str(ftarget),
                    note=f"earliest_ts={int(fdf['ts'].min())}",
                )

            # --- 4h klines (carry-grid-aligned) ---
            ktarget = raw_root / SOURCE / "klines" / f"{coin}_4h.parquet"

            def _fetch_k(_c=coin):
                return fetch_hl_candles_paginated(
                    poster, _c, "4h",
                    start_ms=HL_HISTORY_START_MS, end_ms=now_ms, interval_ms=FOUR_H_MS,
                )

            kdf = cached_pull(
                ktarget, _fetch_k,
                source=SOURCE, endpoint="candleSnapshot",
                params={"coin": coin, "interval": "4h"},
                manifest=manifest, ts_col="ts", rate=None,
            )
            if kdf is not None and len(kdf):
                manifest.log(
                    source=SOURCE, endpoint="candleSnapshot",
                    params={"coin": coin, "interval": "4h"},
                    status="ok", rows=len(kdf), path=str(ktarget),
                    note=f"earliest_ts={int(kdf['ts'].min())}",
                )
    finally:
        cli = getattr(poster, "_client", None)
        if cli is not None and getattr(poster, "_owns", False):
            with contextlib.suppress(Exception):
                cli.close()
