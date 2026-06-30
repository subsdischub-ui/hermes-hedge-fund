"""Binance USDT-M ILLIQUID-TAIL lake client (Stream 0-β, Deliverable A).

The illiquid tail = USDT-M perpetuals BEYOND the ingested 23 majors. This is where
Boucle β's H1 predicts carry persists (un-arbitraged because un-arbitrageable). Purely
ADDITIVE, layered on the proven ``binance.py`` / ``binance_spot.py`` engines (imported,
never forked): it writes NEW ``raw/binance_tail/*`` directories -> NEW auto-discovered
views, so the 23-major views (and their "exactly 23" coverage tests) are byte-untouched.

Five lake artifacts (each dir -> one view):
  * ``binance_tail_klines_4h``       — perp 4h OHLCV, max depth (since=0)
  * ``binance_tail_spot_klines_4h``  — spot 4h OHLCV (the cash-and-carry leg, where listed)
  * ``binance_tail_funding_native``  — DEEP native funding (walked from inception)
  * ``binance_tail_funding_on_4h``   — causal funding pinned to 4h bars (volume + funding)
  * ``binance_tail_universe``        — the selection manifest (base, liquidity decile,
    onboard date, spot availability) + the survivorship note

UNIVERSE POLICY (Malachie, S0-β): hard >=1y depth filter (data viability), then
liquidity-decile stratification that OVER-SAMPLES the illiquid bottom (``tail_select``).
Liquidity for SELECTION = current 24h quote-volume (a snapshot proxy); the per-bar H1
liquidity metric is computed later by ``datalake.liquidity`` from the klines.

SURVIVORSHIP (trap #1): ``/fapi/v1/exchangeInfo`` lists only currently-TRADING contracts
— DELISTED perps are absent and therefore CANNOT be enumerated (their klines remain
fetchable only by a known symbol we don't have a list of). The selected slice is thus a
SURVIVOR set; β must bound its conclusions accordingly. Documented loudly in COVERAGE.

All real logic is PURE (discovery/selection over INJECTED payloads) and unit-tested
offline; ``run()`` + live ccxt/httpx transports are never executed by tests. Symbols use
the slash spelling (BTC/USDT) — the carry-join contract. ts is epoch ms int64.
"""
from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Any

import pandas as pd

from hermes.datalake.core import Manifest, RateLimiter, cached_pull, read_parquet
from hermes.datalake.sources.binance import (
    EXCHANGE,
    KLINE_LIMIT,
    _make_ccxt_client,
    _make_funding_page_fetcher,
    fetch_funding_paginated,
    fetch_ohlcv_paginated,
    gap_count,
)
from hermes.datalake.sources.binance_extended import (
    ALL_PAIRS,
    FOUR_H_MS,
    FUNDING_DEEP_START_MS,
    FUNDING_GAP_INTERVAL_MS,
    FUNDING_ON_4H_LAG_MS,
    KLINE_4H_MAX_BATCHES,
    attach_funding_to_bars,
)
from hermes.datalake.sources.binance_spot import (
    KLINE_4H_MAX_BATCHES as SPOT_KLINE_4H_MAX_BATCHES,
)
from hermes.datalake.sources.binance_spot import (
    SPOT_EXCHANGE,
    _make_spot_ccxt_client,
    fetch_spot_ohlcv_paginated,
)
from hermes.datalake.sources.tail_select import stratified_decile_sample

# --------------------------------------------------------------------------- #
# constants
# --------------------------------------------------------------------------- #
SOURCE = "binance_tail"  # raw/binance_tail/<dir> -> view binance_tail_<dir>

INGESTED = set(ALL_PAIRS)  # the 23 majors already in the lake — excluded from the tail
YEAR_MS = 365 * 86_400_000
MIN_HISTORY_MS = YEAR_MS  # hard >=1y depth floor (data viability, NOT liquidity selection)
N_TARGET = 18  # the Binance share of the ~30-40 stratified slice (HL adds the rest)

EXCHANGE_INFO_URL = "https://fapi.binance.com/fapi/v1/exchangeInfo"
SPOT_EXCHANGE_INFO_URL = "https://api.binance.com/api/v3/exchangeInfo"
TICKER_24HR_URL = "https://fapi.binance.com/fapi/v1/ticker/24hr"

_UNIVERSE_COLS = [
    "base", "symbol", "api_symbol", "onboard_ms", "quote_volume",
    "decile", "has_spot", "source", "run_as_of",
]


# --------------------------------------------------------------------------- #
# PURE transforms — unit-tested offline
# --------------------------------------------------------------------------- #
def discover_tail_perps(
    exchange_info: dict[str, Any],
    *,
    ingested: set[str],
    now_ms: int,
    min_history_ms: int,
) -> list[dict[str, Any]]:
    """Filter ``/fapi/v1/exchangeInfo`` to the depth-qualified USDT-M perp TAIL.

    Keeps symbols that are PERPETUAL + USDT-quoted + TRADING + not already ingested +
    listed at least ``min_history_ms`` ago (``onboardDate <= now_ms - min_history_ms``).
    The depth floor is the only HARD filter (backtestability), not a liquidity gate.
    Returns ``[{base, symbol(slash), api_symbol(concat), onboard_ms}]``.
    """
    out: list[dict[str, Any]] = []
    cutoff = now_ms - min_history_ms
    for s in exchange_info.get("symbols", []):
        if s.get("contractType") != "PERPETUAL":
            continue
        if s.get("quoteAsset") != "USDT":
            continue
        if s.get("status") != "TRADING":
            continue
        base = s.get("baseAsset")
        if not base or base in ingested:
            continue
        onboard = int(s.get("onboardDate", 0) or 0)
        if onboard <= 0 or onboard > cutoff:
            continue  # unknown or too-young -> fails the >=1y depth floor
        out.append(
            {"base": base, "symbol": f"{base}/USDT", "api_symbol": f"{base}USDT",
             "onboard_ms": onboard}
        )
    return out


def liquidity_map_from_ticker(ticker_24hr: list[dict[str, Any]]) -> dict[str, float]:
    """Map each API symbol to its 24h quote-volume (USD-notional liquidity snapshot)
    used as the SELECTION stratification key. Bad/absent values -> 0.0 (screened out)."""
    m: dict[str, float] = {}
    for t in ticker_24hr:
        sym = t.get("symbol")
        if sym is None:
            continue
        try:
            m[sym] = float(t.get("quoteVolume", 0) or 0)
        except (TypeError, ValueError):
            m[sym] = 0.0
    return m


def select_tail_universe(
    candidates: list[dict[str, Any]],
    liquidity_by_api: dict[str, float],
    *,
    n_target: int,
    **kw: Any,
) -> list[dict[str, Any]]:
    """Attach liquidity, drop non-positive/zero-volume (dead/delisted) markets, then
    stratify by liquidity decile over-sampling the illiquid bottom (``tail_select``).
    Returns the selected candidates, each annotated with ``quote_volume`` + ``decile``."""
    rows: list[dict[str, Any]] = []
    for c in candidates:
        v = liquidity_by_api.get(c["api_symbol"])
        if v is not None and v > 0:
            rows.append({**c, "quote_volume": float(v)})
    return stratified_decile_sample(rows, liquidity_key="quote_volume", n_target=n_target, **kw)


def selection_frame(
    selected: list[dict[str, Any]], *, spot_bases: set[str], run_as_of: str
) -> pd.DataFrame:
    """Build the ``binance_tail_universe`` provenance frame from the selected slice,
    flagging spot-leg availability (cash-and-carry needs it; perp-only -> cross-venue)."""
    if not selected:
        return pd.DataFrame(columns=_UNIVERSE_COLS)
    rows = []
    for c in selected:
        rows.append(
            {
                "base": c["base"], "symbol": c["symbol"], "api_symbol": c["api_symbol"],
                "onboard_ms": int(c["onboard_ms"]), "quote_volume": float(c.get("quote_volume", 0.0)),
                "decile": int(c.get("decile", -1)), "has_spot": c["base"] in spot_bases,
                "source": SOURCE, "run_as_of": run_as_of,
            }
        )
    return pd.DataFrame(rows)[_UNIVERSE_COLS]


# --------------------------------------------------------------------------- #
# LIVE transport — DO NOT execute in tests (network)
# --------------------------------------------------------------------------- #
def _http_get_json(url: str, client: Any | None = None) -> Any:
    """GET ``url`` -> parsed JSON via an injected or fresh httpx client. Raises on HTTP
    error so the caller records a blocker."""
    import httpx

    owns = client is None
    cli = client if client is not None else httpx.Client(timeout=60.0)
    try:
        resp = cli.get(url)
        resp.raise_for_status()
        return resp.json()
    finally:
        if owns:
            with contextlib.suppress(Exception):
                cli.close()


def _spot_usdt_bases(spot_exchange_info: dict[str, Any]) -> set[str]:
    """Bases with a currently-TRADING Binance USDT spot pair (cash-and-carry leg)."""
    return {
        s.get("baseAsset")
        for s in spot_exchange_info.get("symbols", [])
        if s.get("quoteAsset") == "USDT" and s.get("status") == "TRADING"
    }


# --------------------------------------------------------------------------- #
# public entrypoint — DO NOT execute in tests (live network)
# --------------------------------------------------------------------------- #
def run(raw_root: Path, manifest: Manifest, *, run_as_of: str) -> None:
    """Discover + stratify the illiquid tail, then pull spot+perp klines + deep native
    funding for the selected slice, idempotently + fail-soft, and materialize the causal
    funding-on-4h join. Public endpoints — no key required.

    Layout (each dir -> one auto-discovered view):
      raw_root/binance_tail/universe/selection.parquet  -> binance_tail_universe
      raw_root/binance_tail/klines_4h/<BASE>_4h.parquet -> binance_tail_klines_4h
      raw_root/binance_tail/spot_klines_4h/<BASE>_4h... -> binance_tail_spot_klines_4h
      raw_root/binance_tail/funding_native/<BASE>...    -> binance_tail_funding_native
      raw_root/binance_tail/funding_on_4h/<BASE>...     -> binance_tail_funding_on_4h
    """
    import time

    raw_root = Path(raw_root)
    now_ms = int(time.time() * 1000)

    # --- discovery (fail-soft: any failure aborts the tail cleanly) -----------
    try:
        xinfo = _http_get_json(EXCHANGE_INFO_URL)
        ticker = _http_get_json(TICKER_24HR_URL)
        spot_xinfo = _http_get_json(SPOT_EXCHANGE_INFO_URL)
    except Exception as e:
        manifest.blocker(
            source=SOURCE, endpoint="discovery", params={},
            error=repr(e),
            action="Binance exchangeInfo/ticker discovery failed; check network/geo-access",
        )
        return

    candidates = discover_tail_perps(
        xinfo, ingested=INGESTED, now_ms=now_ms, min_history_ms=MIN_HISTORY_MS
    )
    liq = liquidity_map_from_ticker(ticker)
    selected = select_tail_universe(candidates, liq, n_target=N_TARGET)
    spot_bases = _spot_usdt_bases(spot_xinfo)

    manifest.log(
        source=SOURCE, endpoint="discovery", params={}, status="ok",
        rows=len(selected),
        note=(f"tail_candidates={len(candidates)} (>=1y depth) -> selected={len(selected)} "
              f"stratified; SURVIVOR-SET ONLY (delisted perps absent from exchangeInfo)"),
    )

    # --- the selection manifest view (always written, even if pulls fail) -----
    cached_pull(
        raw_root / SOURCE / "universe" / "selection.parquet",
        lambda: selection_frame(selected, spot_bases=spot_bases, run_as_of=run_as_of),
        source=SOURCE, endpoint="universe", params={}, manifest=manifest, ts_col=None,
    )

    if not selected:
        return

    # --- transports (perp klines, spot klines, deep funding) ------------------
    try:
        perp_client = _make_ccxt_client()
        spot_client = _make_spot_ccxt_client()
        page_fetcher = _make_funding_page_fetcher()
        krate = RateLimiter(1200)
        srate = RateLimiter(1200)
        frate = RateLimiter(240)
    except Exception as e:
        manifest.blocker(
            source=SOURCE, endpoint="setup", params={},
            error=repr(e),
            action="ccxt/httpx transports failed to construct for binance_tail; check installs",
        )
        return

    try:
        for c in selected:
            base, symbol, api_symbol = c["base"], c["symbol"], c["api_symbol"]

            # perp 4h klines (max depth)
            ktarget = raw_root / SOURCE / "klines_4h" / f"{base}_4h.parquet"
            df = cached_pull(
                ktarget,
                lambda _s=symbol: fetch_ohlcv_paginated(
                    perp_client, _s, "4h", since_ms=0, interval_ms=FOUR_H_MS,
                    limit=KLINE_LIMIT, max_batches=KLINE_4H_MAX_BATCHES,
                ),
                source=SOURCE, endpoint="klines_4h",
                params={"symbol": symbol, "venue": EXCHANGE},
                manifest=manifest, ts_col="ts", rate=krate,
            )
            if df is not None and len(df):
                g = gap_count(df, FOUR_H_MS)
                manifest.log(
                    source=SOURCE, endpoint="klines_4h", params={"symbol": symbol},
                    status="partial" if g else "ok", rows=len(df), gaps=g,
                    path=str(ktarget), note=f"decile={c.get('decile')} gap_count={g}",
                )

            # spot 4h klines (cash-and-carry leg) — only where a spot pair exists
            if base in spot_bases:
                starget = raw_root / SOURCE / "spot_klines_4h" / f"{base}_4h.parquet"
                cached_pull(
                    starget,
                    lambda _s=symbol: fetch_spot_ohlcv_paginated(
                        spot_client, _s, "4h", since_ms=0, interval_ms=FOUR_H_MS,
                        limit=KLINE_LIMIT, max_batches=SPOT_KLINE_4H_MAX_BATCHES,
                    ),
                    source=SOURCE, endpoint="spot_klines_4h",
                    params={"symbol": symbol, "venue": SPOT_EXCHANGE},
                    manifest=manifest, ts_col="ts", rate=srate,
                )

            # deep native funding (walked from inception)
            ftarget = raw_root / SOURCE / "funding_native" / f"{base}.parquet"
            fdf = cached_pull(
                ftarget,
                lambda _s=symbol, _a=api_symbol: fetch_funding_paginated(
                    page_fetcher, _s, _a, start_ms=FUNDING_DEEP_START_MS,
                ),
                source=SOURCE, endpoint="fundingRate_native",
                params={"symbol": api_symbol}, manifest=manifest, ts_col="ts", rate=frate,
            )
            if fdf is not None and len(fdf):
                g = gap_count(fdf, FUNDING_GAP_INTERVAL_MS)
                manifest.log(
                    source=SOURCE, endpoint="fundingRate_native", params={"symbol": api_symbol},
                    status="partial" if g else "ok", rows=len(fdf), gaps=g, path=str(ftarget),
                    note=f"earliest_ts={int(fdf['ts'].min())} gap_count={g}",
                )
    finally:
        cli = getattr(page_fetcher, "_client", None)
        if cli is not None and getattr(page_fetcher, "_owns", False):
            with contextlib.suppress(Exception):
                cli.close()

    # --- materialize causal funding-on-4h for the tail (perp bars + funding) --
    _materialize_funding_on_4h(raw_root, manifest, [c["base"] for c in selected])


def _materialize_funding_on_4h(raw_root: Path, manifest: Manifest, bases: list[str]) -> None:
    """For each selected base with BOTH 4h perp klines and native funding, write the
    causal funding-on-4h join (perp OHLCV incl. volume + settled funding). Idempotent,
    per-base fail-soft. Mirrors binance_extended; reuses attach_funding_to_bars."""
    kdir = raw_root / SOURCE / "klines_4h"
    fdir = raw_root / SOURCE / "funding_native"
    for base in bases:
        kpath = kdir / f"{base}_4h.parquet"
        fpath = fdir / f"{base}.parquet"
        target = raw_root / SOURCE / "funding_on_4h" / f"{base}.parquet"
        if target.exists() or not (kpath.exists() and fpath.exists()):
            continue
        cached_pull(
            target,
            lambda _k=kpath, _f=fpath: attach_funding_to_bars(
                read_parquet(_k), read_parquet(_f), lag_ms=FUNDING_ON_4H_LAG_MS
            ),
            source=SOURCE, endpoint="funding_on_4h",
            params={"symbol": f"{base}USDT", "lag_ms": FUNDING_ON_4H_LAG_MS},
            manifest=manifest, ts_col="ts", rate=None,
        )


__all__ = [
    "MIN_HISTORY_MS",
    "N_TARGET",
    "SOURCE",
    "discover_tail_perps",
    "liquidity_map_from_ticker",
    "run",
    "select_tail_universe",
    "selection_frame",
]
