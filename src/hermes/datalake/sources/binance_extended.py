"""Binance USDT-M PERP — EXTENDED lake client (Stream 0 re-freeze élargi).

This module ADDS the data that unblocks the crypto core of Round 2, layered on
top of the proven ``binance.py`` pagination engine (which it imports, never
forks). It is purely additive: it writes NEW Parquet directories, so the
existing ``binance_klines`` / ``binance_funding`` views are untouched.

Three new lake artifacts (each a directory -> one auto-discovered DuckDB view):

  * ``binance_klines_4h``    — 4h OHLCV at MAX depth (``since=0``; NOT the 360d
    cap the 6h series uses) for the 7 live pairs + an extended liquid universe.
    TACHE 1 + TACHE 2. 4h is the missing cadence: the validated S2 funding edge
    is 4h, and S3 needs >=3 deep instruments for its cross-asset killer-filter.

  * ``binance_funding_native`` — DEEP native funding from /fapi/v1/fundingRate,
    walked from each perp's inception. TACHE 3.
    !! DISTINCT from the existing ``binance_funding`` view, which is FORWARD-ONLY
    (~200 rows/symbol from 2026-04; backtesting on it tests on emptiness). These
    two MUST NEVER be confused — ``binance_funding`` = the live forward stub,
    ``binance_funding_native`` = the deep backtestable history. The settled rate
    is pinned to 4h bars by :func:`attach_funding_to_bars` (causal, lagged, no
    look-ahead) and materialized as ``binance_funding_on_4h``.

  * ``binance_funding_info`` — a /fapi/v1/fundingInfo snapshot recording each
    perp's CURRENT funding cadence (some moved 8h -> 4h) and rate caps. This is
    the "variable interval" handling: history is ~8h, recent prints may be 4h.

All real logic lives in PURE functions (``normalize_funding_info``,
``funding_interval_hours_for``, ``attach_funding_to_bars``) unit-tested offline;
``run()`` constructs the live transports (reused from ``binance.py``) and drives
every pull through ``cached_pull`` (idempotent + fail-soft). ts is epoch ms int64
everywhere (the lake convention).
"""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Any

import pandas as pd

from hermes.datalake.core import Manifest, RateLimiter, cached_pull
from hermes.datalake.sources.binance import (
    DAY_MS,
    EXCHANGE,
    KLINE_LIMIT,
    SOURCE,
    _make_ccxt_client,
    _make_funding_page_fetcher,
    fetch_funding_paginated,
    fetch_ohlcv_paginated,
    gap_count,
)

# --------------------------------------------------------------------------- #
# constants — interval + universe
# --------------------------------------------------------------------------- #
FOUR_H_MS = 14_400_000  # 4 * 3600 * 1000

# The 7 instruments the book trades live.
LIVE_PAIRS = ["BTC", "ETH", "SOL", "AVAX", "SUI", "ARB", "WIF"]

# Extended universe — deep, liquid USDT-M perps chosen for history + liquidity
# (the approved cross-asset breadth angle). The brief's candidate list plus two
# extra deep majors (TRX, UNI). None overlap LIVE_PAIRS. The >=2-3yr depth floor
# is enforced empirically: anything that lands too shallow is reported, and the
# coverage test asserts the floor on what actually materialized.
EXTENDED_PAIRS = [
    "LINK", "DOGE", "BNB", "XRP", "ADA", "DOT", "LTC", "BCH",
    "ATOM", "NEAR", "FIL", "INJ", "APT", "OP", "TRX", "UNI",
]

ALL_PAIRS = LIVE_PAIRS + EXTENDED_PAIRS

# 4h over ~7y of BTC perp history is ~15k bars -> ~15 pages of 1000; give headroom.
KLINE_4H_MAX_BATCHES = 80

# fapi/v1/fundingInfo (snapshot of current per-symbol funding settings).
FUNDING_INFO_URL = "https://fapi.binance.com/fapi/v1/fundingInfo"

# !! The /fapi/v1/fundingRate clamp: startTime<=0 (or absent) makes the endpoint
# return ONLY the most-recent ~200 prints, NOT inception-to-now — the silent trap
# that made the existing ``binance_funding`` view a forward-only stub. A strictly
# POSITIVE startTime escapes the clamp (Binance then clamps it up to the perp's
# actual listing). 1ms-past-epoch is universal and predates every USDT-M perp.
FUNDING_DEEP_START_MS = 1

# Funding-gap accounting uses the historical MAX cadence (8h): a perp that moved
# 8h->4h then has *denser* prints (never a false gap), while a genuinely missing
# 8h print still shows as a >12h hole. See funding_interval_hours_for for the
# per-symbol CURRENT cadence used by research, not by this data-quality note.
FUNDING_GAP_INTERVAL_MS = 8 * 3_600_000

# The publication-lag used to pin settled funding to 4h bars in the materialized
# ``binance_funding_on_4h``. Binance publishes the SETTLED rate at fundingTime,
# so the real latency is ~0: a bar opening at T may carry funding settled at
# fundingTime <= T (already realized + published). Research wanting a stricter
# embargo can re-join binance_funding_native against binance_klines_4h with a
# larger lag via attach_funding_to_bars.
FUNDING_ON_4H_LAG_MS = 0

_INFO_COLS = [
    "symbol", "funding_interval_hours", "funding_rate_cap",
    "funding_rate_floor", "source", "run_as_of",
]


# --------------------------------------------------------------------------- #
# PURE transforms — unit-tested offline
# --------------------------------------------------------------------------- #
def normalize_funding_info(rows: list[dict[str, Any]], *, run_as_of: str) -> pd.DataFrame:
    """Canonicalise /fapi/v1/fundingInfo rows into a tidy per-symbol frame.

    Binance lists only perps with NON-default settings here; numeric caps arrive
    as strings, ``fundingIntervalHours`` as an int. Maps to
    ``symbol / funding_interval_hours / funding_rate_cap / funding_rate_floor``
    plus ``source`` + ``run_as_of`` provenance. Empty input -> canonical empty
    frame (never raises).
    """
    if not rows:
        df = pd.DataFrame(columns=_INFO_COLS)
        df["funding_interval_hours"] = df["funding_interval_hours"].astype("float64")
        return df

    raw = pd.DataFrame(list(rows))
    out = pd.DataFrame()
    out["symbol"] = raw.get("symbol")
    out["funding_interval_hours"] = pd.to_numeric(
        raw.get("fundingIntervalHours"), errors="coerce"
    )
    out["funding_rate_cap"] = pd.to_numeric(
        raw.get("adjustedFundingRateCap"), errors="coerce"
    )
    out["funding_rate_floor"] = pd.to_numeric(
        raw.get("adjustedFundingRateFloor"), errors="coerce"
    )
    out["source"] = SOURCE
    out["run_as_of"] = run_as_of
    return out[_INFO_COLS]


def funding_interval_hours_for(
    info_df: pd.DataFrame, api_symbol: str, *, default: int = 8
) -> int:
    """Resolve ``api_symbol``'s current funding cadence (hours).

    Returns the value from a fundingInfo frame, or ``default`` (Binance's 8h
    default) when the symbol is absent — which it is for every perp still on
    default settings. Never raises; falls back to ``default`` on any bad value.
    """
    if info_df is None or len(info_df) == 0 or "symbol" not in info_df.columns:
        return default
    hit = info_df[info_df["symbol"] == api_symbol]
    if len(hit) == 0:
        return default
    val = hit["funding_interval_hours"].iloc[0]
    try:
        if pd.isna(val):
            return default
        return int(val)
    except (TypeError, ValueError):
        return default


def attach_funding_to_bars(
    bars: pd.DataFrame, funding: pd.DataFrame, *, lag_ms: int
) -> pd.DataFrame:
    """Pin each settled funding print to its 4h bar — CAUSAL, lagged, no look-ahead.

    For a bar opening at ``ts = T``, attach the most recent funding whose
    *effective availability* ``fundingTime + lag_ms <= T``. With ``lag_ms = 0``
    this is "the funding settled at or before the bar open"; a positive
    ``lag_ms`` embargoes a print until ``lag_ms`` after it settles, so a funding
    settling exactly at a bar's open then lands on the NEXT bar (proven by the
    boundary tests). Bars before the first available print get ``NaN``
    (fail-soft). Output is sorted ascending by ``ts``; every input bar survives.

    Implemented as a backward as-of merge on the lag-shifted funding clock.
    """
    out = bars.sort_values("ts").reset_index(drop=True).copy()
    if funding is None or len(funding) == 0:
        out["funding_rate"] = pd.Series([float("nan")] * len(out), dtype="float64")
        return out

    right = pd.DataFrame(
        {
            "ts_eff": funding["ts"].astype("int64") + int(lag_ms),
            "funding_rate": funding["funding_rate"].astype("float64"),
        }
    ).sort_values("ts_eff").reset_index(drop=True)

    out["ts"] = out["ts"].astype("int64")
    merged = pd.merge_asof(
        out, right, left_on="ts", right_on="ts_eff", direction="backward",
    )
    merged = merged.drop(columns=["ts_eff"])
    return merged


# --------------------------------------------------------------------------- #
# live transport — fundingInfo (klines/funding transports reused from binance.py)
# --------------------------------------------------------------------------- #
def _fetch_funding_info(client: Any | None = None) -> list[dict[str, Any]]:
    """GET /fapi/v1/fundingInfo -> list of per-symbol setting dicts.

    Uses an injected httpx-style client when given (tests), else a real one.
    Raises on transport/HTTP error so ``cached_pull`` records a blocker.
    """
    import httpx

    owns = client is None
    cli = client if client is not None else httpx.Client(timeout=60.0)
    try:
        resp = cli.get(FUNDING_INFO_URL)
        resp.raise_for_status()
        data = resp.json()
        return data if isinstance(data, list) else []
    finally:
        if owns:
            with contextlib.suppress(Exception):
                cli.close()


# --------------------------------------------------------------------------- #
# public entrypoint — DO NOT execute in tests (live network)
# --------------------------------------------------------------------------- #
def run(raw_root: Path, manifest: Manifest, *, run_as_of: str) -> None:
    """Pull 4h klines (max depth) + deep native funding + a fundingInfo snapshot
    for the live + extended universe, idempotently and fail-soft, then
    materialize the causal funding-on-4h join.

    Layout (each dir -> one auto-discovered view):
      raw_root/binance/klines_4h/<BASE>_4h.parquet     -> binance_klines_4h
      raw_root/binance/funding_native/<BASE>.parquet   -> binance_funding_native
      raw_root/binance/funding_info/all.parquet        -> binance_funding_info
      raw_root/binance/funding_on_4h/<BASE>.parquet    -> binance_funding_on_4h
    """
    raw_root = Path(raw_root)

    # --- fundingInfo snapshot first (cheap; informs gap notes) ----------------
    info_df = _pull_funding_info(raw_root, manifest, run_as_of=run_as_of)

    # --- 4h klines (USDT-M PERP via ccxt), since=0 -> max depth ---------------
    try:
        ccxt_client = _make_ccxt_client()
        krate = RateLimiter(1200)
    except Exception as e:  # setup failure -> one blocker, skip klines, still try funding
        manifest.blocker(
            source=SOURCE, endpoint="klines_4h", params={"venue": EXCHANGE},
            error=repr(e),
            action="ccxt USDT-M PERP client failed to construct; check ccxt install / network",
        )
        ccxt_client = None
        krate = None

    if ccxt_client is not None:
        for base in ALL_PAIRS:
            symbol = f"{base}/USDT"
            target = raw_root / SOURCE / "klines_4h" / f"{base}_4h.parquet"

            def _fetch(_sym=symbol):
                return fetch_ohlcv_paginated(
                    ccxt_client, _sym, "4h",
                    since_ms=0, interval_ms=FOUR_H_MS,
                    limit=KLINE_LIMIT, max_batches=KLINE_4H_MAX_BATCHES,
                )

            df = cached_pull(
                target, _fetch,
                source=SOURCE, endpoint="klines_4h",
                params={"symbol": symbol, "interval": "4h", "venue": EXCHANGE},
                manifest=manifest, ts_col="ts", rate=krate,
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

    # --- deep native funding (fapi/v1/fundingRate, walked from inception) ------
    try:
        page_fetcher = _make_funding_page_fetcher()
        frate = RateLimiter(240)
    except Exception as e:
        manifest.blocker(
            source=SOURCE, endpoint="fundingRate_native", params={},
            error=repr(e),
            action="httpx client failed to construct for native funding; check httpx install",
        )
        return

    try:
        for base in ALL_PAIRS:
            symbol = f"{base}/USDT"
            api_symbol = f"{base}USDT"
            target = raw_root / SOURCE / "funding_native" / f"{base}.parquet"

            def _fetch_f(_sym=symbol, _api=api_symbol):
                return fetch_funding_paginated(
                    page_fetcher, _sym, _api, start_ms=FUNDING_DEEP_START_MS,
                )

            df = cached_pull(
                target, _fetch_f,
                source=SOURCE, endpoint="fundingRate_native",
                params={"symbol": api_symbol},
                manifest=manifest, ts_col="ts", rate=frate,
            )
            if df is not None and len(df):
                g = gap_count(df, FUNDING_GAP_INTERVAL_MS)
                manifest.log(
                    source=SOURCE, endpoint="fundingRate_native",
                    params={"symbol": api_symbol,
                            "current_interval_h": funding_interval_hours_for(info_df, api_symbol)},
                    status="partial" if g else "ok",
                    rows=len(df), gaps=g, path=str(target),
                    note=f"gap_count={g} (8h grid)",
                )
    finally:
        cli = getattr(page_fetcher, "_client", None)
        if cli is not None and getattr(page_fetcher, "_owns", False):
            with contextlib.suppress(Exception):
                cli.close()

    # --- materialize the causal funding-on-4h join ----------------------------
    _materialize_funding_on_4h(raw_root, manifest)


# --------------------------------------------------------------------------- #
# per-unit helpers
# --------------------------------------------------------------------------- #
def _pull_funding_info(
    raw_root: Path, manifest: Manifest, *, run_as_of: str
) -> pd.DataFrame:
    """Pull the fundingInfo snapshot to one Parquet; return the normalized frame
    (empty frame on any failure — never raises)."""
    target = raw_root / SOURCE / "funding_info" / "all.parquet"

    def _fetch() -> pd.DataFrame:
        return normalize_funding_info(_fetch_funding_info(), run_as_of=run_as_of)

    df = cached_pull(
        target, _fetch,
        source=SOURCE, endpoint="fundingInfo", params={},
        manifest=manifest, ts_col=None, rate=RateLimiter(240),
    )
    return df if df is not None else normalize_funding_info([], run_as_of=run_as_of)


def _materialize_funding_on_4h(raw_root: Path, manifest: Manifest) -> None:
    """For each pair with BOTH 4h klines and native funding, write the causal
    funding-on-4h join. Idempotent (cached_pull skips an existing target);
    per-pair fail-soft."""
    from hermes.datalake.core import read_parquet

    kdir = raw_root / SOURCE / "klines_4h"
    fdir = raw_root / SOURCE / "funding_native"
    for base in ALL_PAIRS:
        kpath = kdir / f"{base}_4h.parquet"
        fpath = fdir / f"{base}.parquet"
        target = raw_root / SOURCE / "funding_on_4h" / f"{base}.parquet"
        if target.exists() or not (kpath.exists() and fpath.exists()):
            continue

        def _join(_k=kpath, _f=fpath):
            bars = read_parquet(_k)
            funding = read_parquet(_f)
            return attach_funding_to_bars(bars, funding, lag_ms=FUNDING_ON_4H_LAG_MS)

        cached_pull(
            target, _join,
            source=SOURCE, endpoint="funding_on_4h",
            params={"symbol": f"{base}USDT", "lag_ms": FUNDING_ON_4H_LAG_MS},
            manifest=manifest, ts_col="ts", rate=None,
        )


# Re-export the day constant so callers can reason about depth without importing
# two modules.
__all__ = [
    "ALL_PAIRS",
    "DAY_MS",
    "EXTENDED_PAIRS",
    "FOUR_H_MS",
    "LIVE_PAIRS",
    "attach_funding_to_bars",
    "funding_interval_hours_for",
    "normalize_funding_info",
    "run",
]
