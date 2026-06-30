"""Hyperliquid EXOTIC-markets lake client (Stream 0-β, Deliverable B).

The heart of Boucle β: HL markets BEYOND the 23 cross-venue overlap — young, low-capital,
hourly-funding small-caps / HIP-3 — the most likely home of un-arbitraged carry. Purely
ADDITIVE, layered on the proven ``hyperliquid.py`` engine (imported, never forked): writes
NEW ``raw/hyperliquid_exotic/*`` dirs -> NEW views; the 23-overlap views are untouched.
DISTINCT from ``hermes.execution.hyperliquid`` (the off-limits live adapter).

Four lake artifacts (each dir -> one view):
  * ``hyperliquid_exotic_ctx``           — full-exotic-universe liquidity SNAPSHOT
    (dayNtlVlm, openInterest, funding, isDelisted...) for ALL exotic markets, the
    survivorship + liquidity-distribution artifact. HL keeps DELISTED markets in
    metaAndAssetCtxs (flagged) -> the survivorship gap is OBSERVABLE here (unlike Binance).
  * ``hyperliquid_exotic_funding_1h``    — hourly settled funding for the selected slice
  * ``hyperliquid_exotic_klines``        — 4h perp OHLCV for the selected slice
  * ``hyperliquid_exotic_funding_on_4h`` — hourly funding pinned causally to 4h bars
    (volume + funding) -> the H1 liquidity-vs-funding recipe applies directly.

UNIVERSE POLICY (Malachie, S0-β): exclude the 23 + delisted + zero-volume (dead) markets,
then stratify by dayNtlVlm decile over-sampling the illiquid bottom. The hard >=1y depth
filter is applied via a bounded funding-inception probe over a stratified shortlist (HL
inception is not in the ctx snapshot, so it must be probed — cheaply, one call per coin).

DEPTH CAVEAT: HL exotics are even younger than the majors -> a STRONGLY-FORWARD axis;
per-market inception is recorded at ingest. Symbols are HL-native BARE (BTC).
"""
from __future__ import annotations

import contextlib
import time
from pathlib import Path
from typing import Any

import pandas as pd

from hermes.datalake.core import Manifest, RateLimiter, cached_pull, read_parquet
from hermes.datalake.sources.binance_extended import (
    FUNDING_ON_4H_LAG_MS,
    attach_funding_to_bars,
)
from hermes.datalake.sources.hyperliquid import (
    ALL_PAIRS,
    FOUR_H_MS,
    HL_HISTORY_START_MS,
    _make_hl_poster,
    fetch_hl_candles_paginated,
    fetch_hl_funding_paginated,
)
from hermes.datalake.sources.tail_select import stratified_decile_sample

# --------------------------------------------------------------------------- #
# constants
# --------------------------------------------------------------------------- #
SOURCE = "hyperliquid_exotic"  # raw/hyperliquid_exotic/<dir> -> hyperliquid_exotic_<dir>

INGESTED = set(ALL_PAIRS)  # the 23-overlap already in hyperliquid_* -> excluded
YEAR_MS = 365 * 86_400_000
MIN_HISTORY_MS = YEAR_MS  # hard >=1y depth floor (data viability)
N_TARGET = 18  # the HL share of the ~30-40 stratified slice
PROBE_MULT = 3  # probe inception over a shortlist this many x N_TARGET (bounded cost)

_CTX_COLS = [
    "name", "dayNtlVlm", "openInterest", "markPx", "oi_usd", "funding",
    "premium", "midPx", "isDelisted", "maxLeverage", "source", "run_as_of",
]


# --------------------------------------------------------------------------- #
# PURE transforms — unit-tested offline
# --------------------------------------------------------------------------- #
def _f(v: Any, default: float = 0.0) -> float:
    try:
        if v is None:
            return default
        return float(v)
    except (TypeError, ValueError):
        return default


def parse_hl_asset_ctxs(meta_and_ctxs: Any) -> list[dict[str, Any]]:
    """Extract the per-market liquidity snapshot from a ``metaAndAssetCtxs`` response.

    Canonical shape ``[{"universe": [...]}, [assetCtxs...]]`` (also tolerates the
    universe list directly). Captures the native liquidity fields the base HL source
    discards: ``dayNtlVlm`` (USD day volume), ``openInterest`` (coins) -> ``oi_usd``
    (notional), ``funding`` (hourly), ``premium``, ``midPx``, and the ``isDelisted`` flag.
    """
    first = meta_and_ctxs[0]
    universe = first["universe"] if isinstance(first, dict) and "universe" in first else first
    ctxs = meta_and_ctxs[1]
    rows: list[dict[str, Any]] = []
    for u, c in zip(universe, ctxs):
        mark = _f(c.get("markPx"))
        oi = _f(c.get("openInterest"))
        rows.append(
            {
                "name": u.get("name"),
                "dayNtlVlm": _f(c.get("dayNtlVlm")),
                "openInterest": oi,
                "markPx": mark,
                "oi_usd": oi * mark,
                "funding": _f(c.get("funding")),
                "premium": _f(c.get("premium")),
                "midPx": _f(c.get("midPx")),
                "isDelisted": bool(u.get("isDelisted", False)),
                "maxLeverage": u.get("maxLeverage"),
            }
        )
    return rows


def select_exotic_universe(
    ctx_rows: list[dict[str, Any]],
    *,
    ingested: set[str],
    n_target: int,
    inception_ms: dict[str, int] | None = None,
    now_ms: int = 0,
    min_history_ms: int = 0,
    exclude_delisted: bool = True,
    **kw: Any,
) -> list[dict[str, Any]]:
    """Filter exotic candidates (exclude ingested 23 + delisted + zero-volume dead),
    optionally apply the >=1y depth floor (when ``inception_ms`` is supplied), then
    stratify by ``dayNtlVlm`` decile over-sampling the illiquid bottom (``tail_select``).
    Returns the selected candidates annotated with ``decile``.
    """
    cands: list[dict[str, Any]] = []
    for r in ctx_rows:
        name = r.get("name")
        if not name or name in ingested:
            continue
        if exclude_delisted and r.get("isDelisted"):
            continue
        if not (_f(r.get("dayNtlVlm")) > 0):
            continue
        if inception_ms is not None:
            inc = inception_ms.get(name)
            if inc is None or inc > now_ms - min_history_ms:
                continue  # unknown or < min history -> fails the depth floor
        cands.append(dict(r))
    return stratified_decile_sample(cands, liquidity_key="dayNtlVlm", n_target=n_target, **kw)


_SEL_COLS = [
    "name", "decile", "dayNtlVlm", "oi_usd", "funding", "inception_ms",
    "maxLeverage", "source", "run_as_of",
]


def selection_frame(selected: list[dict[str, Any]], *, run_as_of: str) -> pd.DataFrame:
    """Build the ``hyperliquid_exotic_universe`` manifest from the selected slice,
    recording the liquidity decile + probed funding inception (parity with
    ``binance_tail_universe``)."""
    if not selected:
        return pd.DataFrame(columns=_SEL_COLS)
    rows = [
        {
            "name": c["name"], "decile": int(c.get("decile", -1)),
            "dayNtlVlm": _f(c.get("dayNtlVlm")), "oi_usd": _f(c.get("oi_usd")),
            "funding": _f(c.get("funding")), "inception_ms": int(c.get("inception_ms") or 0),
            "maxLeverage": c.get("maxLeverage"), "source": SOURCE, "run_as_of": run_as_of,
        }
        for c in selected
    ]
    return pd.DataFrame(rows)[_SEL_COLS]


def ctx_snapshot_frame(ctx_rows: list[dict[str, Any]], *, run_as_of: str) -> pd.DataFrame:
    """Build the ``hyperliquid_exotic_ctx`` snapshot frame (the survivorship + liquidity
    distribution artifact). Includes delisted markets so the survivorship gap is visible."""
    if not ctx_rows:
        return pd.DataFrame(columns=_CTX_COLS)
    keys = _CTX_COLS[:-2]
    rows = [{**{k: r.get(k) for k in keys}, "source": SOURCE, "run_as_of": run_as_of} for r in ctx_rows]
    return pd.DataFrame(rows)[_CTX_COLS]


# --------------------------------------------------------------------------- #
# LIVE transport — DO NOT execute in tests (network)
# --------------------------------------------------------------------------- #
def _probe_funding_inception(poster: Any, coin: str) -> int | None:
    """Cheap inception probe: the first ``fundingHistory`` page from startTime=0 returns
    the earliest settlement; its time is the listing-ish inception. None on empty/failure."""
    try:
        rows = poster({"type": "fundingHistory", "coin": coin, "startTime": 0})
    except Exception:
        return None
    if isinstance(rows, list) and rows:
        return int(rows[0]["time"])
    return None


# --------------------------------------------------------------------------- #
# public entrypoint — DO NOT execute in tests (live network)
# --------------------------------------------------------------------------- #
def run(raw_root: Path, manifest: Manifest, *, run_as_of: str) -> None:
    """Snapshot the exotic universe, depth-filter a stratified slice, then pull hourly
    funding + 4h klines for it, idempotently + fail-soft, and materialize funding-on-4h.

    Layout (each dir -> one view):
      raw_root/hyperliquid_exotic/ctx/snapshot.parquet      -> hyperliquid_exotic_ctx
      raw_root/hyperliquid_exotic/funding_1h/<COIN>.parquet -> hyperliquid_exotic_funding_1h
      raw_root/hyperliquid_exotic/klines/<COIN>_4h.parquet  -> hyperliquid_exotic_klines
      raw_root/hyperliquid_exotic/funding_on_4h/<COIN>...   -> hyperliquid_exotic_funding_on_4h
    """
    raw_root = Path(raw_root)
    now_ms = int(time.time() * 1000)

    try:
        rate = RateLimiter(45)  # per-request throttle inside the poster (HL ceiling)
        poster = _make_hl_poster(rate=rate)
    except Exception as e:
        manifest.blocker(
            source=SOURCE, endpoint="info", params={}, error=repr(e),
            action="httpx client failed to construct for Hyperliquid /info; check httpx install",
        )
        return

    try:
        # --- universe + ctx snapshot ---
        try:
            meta = poster({"type": "metaAndAssetCtxs"})
            ctx_rows = parse_hl_asset_ctxs(meta)
        except Exception as e:
            manifest.blocker(
                source=SOURCE, endpoint="metaAndAssetCtxs", params={}, error=repr(e),
                action="HL exotic universe discovery failed; nothing pulled this run",
            )
            return

        exotic_all = [r for r in ctx_rows if r.get("name") not in INGESTED]
        cached_pull(
            raw_root / SOURCE / "ctx" / "snapshot.parquet",
            lambda: ctx_snapshot_frame(exotic_all, run_as_of=run_as_of),
            source=SOURCE, endpoint="ctx_snapshot", params={}, manifest=manifest, ts_col=None,
        )
        n_delisted = sum(1 for r in exotic_all if r.get("isDelisted"))
        manifest.log(
            source=SOURCE, endpoint="metaAndAssetCtxs", params={}, status="ok",
            rows=len(exotic_all),
            note=(f"exotic_markets={len(exotic_all)} ({n_delisted} delisted, VISIBLE in ctx snapshot "
                  f"-> HL survivorship is observable, unlike Binance)"),
        )

        # --- stratified shortlist -> bounded inception probe -> depth-filtered final ---
        positive = [r for r in exotic_all if not r.get("isDelisted") and _f(r.get("dayNtlVlm")) > 0]
        probe_n = min(len(positive), PROBE_MULT * N_TARGET)
        shortlist = select_exotic_universe(exotic_all, ingested=INGESTED, n_target=probe_n)
        inception = {r["name"]: _probe_funding_inception(poster, r["name"]) for r in shortlist}
        selected = select_exotic_universe(
            shortlist, ingested=set(), n_target=N_TARGET,
            inception_ms=inception, now_ms=now_ms, min_history_ms=MIN_HISTORY_MS,
        )
        manifest.log(
            source=SOURCE, endpoint="selection", params={}, status="ok", rows=len(selected),
            note=(f"probed={len(shortlist)} -> selected={len(selected)} (>=1y funding depth), "
                  f"stratified illiquid-overweight"),
        )
        if not selected:
            return

        # --- persist the selection manifest (decile + probed inception) ---
        for r in selected:
            r["inception_ms"] = inception.get(r["name"])
        cached_pull(
            raw_root / SOURCE / "universe" / "selection.parquet",
            lambda: selection_frame(selected, run_as_of=run_as_of),
            source=SOURCE, endpoint="universe", params={}, manifest=manifest, ts_col=None,
        )

        # --- pull hourly funding + 4h klines for the selected slice ---
        for r in selected:
            coin = r["name"]
            ftarget = raw_root / SOURCE / "funding_1h" / f"{coin}.parquet"
            fdf = cached_pull(
                ftarget,
                lambda _c=coin: fetch_hl_funding_paginated(
                    poster, _c, start_ms=HL_HISTORY_START_MS, end_ms=now_ms
                ),
                source=SOURCE, endpoint="fundingHistory", params={"coin": coin},
                manifest=manifest, ts_col="ts", rate=None,
            )
            if fdf is not None and len(fdf):
                manifest.log(
                    source=SOURCE, endpoint="fundingHistory", params={"coin": coin},
                    status="ok", rows=len(fdf), path=str(ftarget),
                    note=f"decile={r.get('decile')} earliest_ts={int(fdf['ts'].min())}",
                )

            ktarget = raw_root / SOURCE / "klines" / f"{coin}_4h.parquet"
            cached_pull(
                ktarget,
                lambda _c=coin: fetch_hl_candles_paginated(
                    poster, _c, "4h", start_ms=HL_HISTORY_START_MS, end_ms=now_ms,
                    interval_ms=FOUR_H_MS,
                ),
                source=SOURCE, endpoint="candleSnapshot", params={"coin": coin, "interval": "4h"},
                manifest=manifest, ts_col="ts", rate=None,
            )
    finally:
        cli = getattr(poster, "_client", None)
        if cli is not None and getattr(poster, "_owns", False):
            with contextlib.suppress(Exception):
                cli.close()

    _materialize_funding_on_4h(raw_root, manifest, [r["name"] for r in selected])


def _materialize_funding_on_4h(raw_root: Path, manifest: Manifest, coins: list[str]) -> None:
    """For each selected coin with BOTH 4h klines and hourly funding, pin the hourly
    funding causally onto the 4h bars (volume + funding) -> the H1 recipe applies. The
    attached rate is the most-recent HOURLY settle <= bar open (annualize with
    interval_hours=1). Idempotent, per-coin fail-soft; reuses attach_funding_to_bars."""
    kdir = raw_root / SOURCE / "klines"
    fdir = raw_root / SOURCE / "funding_1h"
    for coin in coins:
        kpath = kdir / f"{coin}_4h.parquet"
        fpath = fdir / f"{coin}.parquet"
        target = raw_root / SOURCE / "funding_on_4h" / f"{coin}.parquet"
        if target.exists() or not (kpath.exists() and fpath.exists()):
            continue
        cached_pull(
            target,
            lambda _k=kpath, _f=fpath: attach_funding_to_bars(
                read_parquet(_k), read_parquet(_f), lag_ms=FUNDING_ON_4H_LAG_MS
            ),
            source=SOURCE, endpoint="funding_on_4h",
            params={"coin": coin, "lag_ms": FUNDING_ON_4H_LAG_MS},
            manifest=manifest, ts_col="ts", rate=None,
        )


__all__ = [
    "MIN_HISTORY_MS",
    "N_TARGET",
    "SOURCE",
    "ctx_snapshot_frame",
    "parse_hl_asset_ctxs",
    "run",
    "select_exotic_universe",
    "selection_frame",
]
