"""Bybit 8h funding-history collector — infra, $0, hors-essai (no strategy attached).

WHY THIS EXISTS — ledger entry #41's ``revive_condition`` names "cross-venue Bybit
(re-test >= 2026-11)" for the funding_squeeze family. The pre-registration for THAT
re-test will happen at that date; this module is only the DATA COLLECTION half, done
now because it is free and idempotent, and because Bybit's public funding-history
endpoint has no documented retention guarantee — every day this does not run is history
that may become unrecoverable. It registers NO new trial and attaches NO signal.

The frozen coinglass_multi lake only has 5 coins at 1d resolution — too coarse to date
8h-cadence funding events. This collector pulls Bybit's OWN 8h funding prints directly
(no aggregation, no vendor), giving genuine per-settlement (~3/day) granularity.

Storage is deliberately NOT the shared ``hermes.forward.logio`` day-level dedup (that
collapses to one row per UTC day, which would destroy 2 of every 3 funding settlements).
Instead: one CSV per symbol at ``<data_dir>/bybit_funding/<SYMBOL>.csv``, columns
``ts_ms,funding_rate``, deduplicated on EXACT ``ts_ms`` — funding settles at fixed
UTC times (00:00/08:00/16:00 for most symbols) so exact-ts dedup is the correct
idempotency key, not a day floor.

Network: GET https://api.bybit.com/v5/market/funding/history (category=linear, no key,
limit<=200/page), paginated backward via ``endTime``. Rate-limited (~0.2s/request).
Fail-soft per symbol: one absent/erroring symbol never blocks the others.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pandas as pd

from hermes.logging_setup import get_logger
from hermes.settings import get_settings

log = get_logger(__name__)

# Live 7-asset slate + the breadth-verifiable extension named in the mission brief.
DEFAULT_SYMBOLS: tuple[str, ...] = (
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "AVAXUSDT", "SUIUSDT", "ARBUSDT", "WIFUSDT",
    "XRPUSDT", "BNBUSDT", "DOGEUSDT", "ZECUSDT",
)

_CSV_COLS = ["ts_ms", "funding_rate"]
_DEFAULT_LIMIT = 200
_DEFAULT_SLEEP_S = 0.2
_DEFAULT_MAX_PAGES = 500  # generous guard against a runaway loop; real history is ~O(30)


# ---- pure transform ---------------------------------------------------------
def parse_funding_rows(raw: list[dict] | None) -> list[tuple[int, float]]:
    """Shape Bybit's raw ``result.list`` payload into ``(ts_ms, funding_rate)`` tuples.

    Rows missing/unparsable ``fundingRateTimestamp``/``fundingRate`` are dropped —
    never fabricated. Order is NOT guaranteed here (callers that care sort explicitly).
    """
    if not raw:
        return []
    out: list[tuple[int, float]] = []
    for r in raw:
        ts = r.get("fundingRateTimestamp")
        rate = r.get("fundingRate")
        if ts is None or rate is None:
            continue
        try:
            out.append((int(ts), float(rate)))
        except (TypeError, ValueError):
            continue
    return out


# ---- network seam ------------------------------------------------------------
def fetch_funding_page(
    symbol: str, *, end_time: int | None = None, limit: int = _DEFAULT_LIMIT
) -> list[dict]:  # pragma: no cover - network seam
    """Fetch one page of Bybit funding history (read-only, no API key).

    Returns the raw ``result.list`` (newest-first, per Bybit's own ordering).
    Raises on transport error or a non-zero ``retCode`` (e.g. an unlisted symbol) —
    the caller (``collect``/``backfill_symbol``) is fail-soft per symbol.
    """
    import httpx

    params: dict[str, Any] = {"category": "linear", "symbol": symbol, "limit": limit}
    if end_time is not None:
        params["endTime"] = int(end_time)
    resp = httpx.get(
        "https://api.bybit.com/v5/market/funding/history", params=params, timeout=30.0
    )
    resp.raise_for_status()
    data = resp.json()
    if data.get("retCode") != 0:
        raise RuntimeError(f"bybit retCode={data.get('retCode')}: {data.get('retMsg')}")
    return data.get("result", {}).get("list", []) or []


# ---- storage ------------------------------------------------------------------
def _base_dir(data_dir: Path | str | None = None) -> Path:
    if data_dir is not None:
        return Path(data_dir)
    return Path(get_settings().data_dir)


def _csv_path(symbol: str, data_dir: Path | str | None = None) -> Path:
    d = _base_dir(data_dir) / "bybit_funding"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{symbol}.csv"


def append_rows(path: Path, rows: list[tuple[int, float]]) -> int:
    """Idempotently append ``(ts_ms, funding_rate)`` rows to ``path``.

    At most one row survives per exact ``ts_ms`` (never a duplicate); the file is
    kept sorted ascending by ``ts_ms``. Returns the count of genuinely NEW rows
    written (0 if every row in ``rows`` was already present — the noop case that
    makes a same-data re-run cheap and detectable).
    """
    if not rows:
        return 0
    if path.exists():
        existing = pd.read_csv(path)
        existing_ts = set(existing["ts_ms"].astype("int64").tolist()) if len(existing) else set()
    else:
        existing = pd.DataFrame(columns=_CSV_COLS)
        existing_ts = set()
    new = [(t, r) for t, r in rows if t not in existing_ts]
    if not new:
        return 0
    new_df = pd.DataFrame(new, columns=_CSV_COLS)
    frames = [df for df in (existing, new_df) if len(df)]
    merged = (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates(subset=["ts_ms"], keep="last")
        .sort_values("ts_ms")
        .reset_index(drop=True)
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(path, index=False)
    return len(new)


# ---- pagination / backfill -----------------------------------------------------
def backfill_symbol(
    symbol: str,
    *,
    data_dir: Path | str | None = None,
    fetch=None,
    limit: int = _DEFAULT_LIMIT,
    sleep_s: float = _DEFAULT_SLEEP_S,
    max_pages: int = _DEFAULT_MAX_PAGES,
) -> int:
    """Backfill one symbol's full funding history, walking ``endTime`` backward.

    Paginates until any of: an empty page, a short page (< ``limit`` rows — Bybit
    has no more history), a page whose rows are ALL already on disk (we've reached
    previously-backfilled territory — makes a second run cheap: it re-fetches only
    the newest page(s) before recognizing known ground and stopping), or
    ``max_pages`` as a runaway guard. Returns the total count of NEW rows written.
    """
    # Liaison TARDIVE de la couture réseau (2026-07-18) : un défaut de paramètre
    # `fetch=fetch_funding_page` se lie à l'IMPORT — un monkeypatch de l'attribut de
    # module devient alors sans effet et les tests CLI frappaient l'API réelle
    # (mesuré : 140 s/test de backfill complet dans la suite). Résolu à l'appel.
    if fetch is None:
        fetch = fetch_funding_page
    path = _csv_path(symbol, data_dir)
    end_time: int | None = None
    total_new = 0
    prev_min_ts: int | None = None
    for _ in range(max_pages):
        raw = fetch(symbol, end_time=end_time, limit=limit)
        rows = parse_funding_rows(raw)
        if not rows:
            break
        new_count = append_rows(path, rows)
        total_new += new_count
        page_min_ts = min(t for t, _ in rows)
        if prev_min_ts is not None and page_min_ts >= prev_min_ts:
            break  # no backward progress -> guard against an infinite loop
        prev_min_ts = page_min_ts
        if new_count == 0:
            # Every row on this page was already logged: older pages would be too
            # (Bybit returns strictly-decreasing timestamps) -> stop here.
            break
        if len(rows) < limit:
            break  # short page: reached the earliest history Bybit serves
        end_time = page_min_ts - 1
        if sleep_s:
            time.sleep(sleep_s)
    return total_new


# ---- fail-soft multi-symbol entry point ----------------------------------------
def collect(
    symbols: tuple[str, ...] = DEFAULT_SYMBOLS,
    *,
    data_dir: Path | str | None = None,
    fetch=None,
    sleep_s: float = _DEFAULT_SLEEP_S,
    max_pages: int = _DEFAULT_MAX_PAGES,
) -> dict:
    """Backfill every symbol, fail-soft per symbol.

    Returns ``{"written": {symbol: n_new_rows}, "blockers": {symbol: reason}}``. A
    symbol Bybit doesn't list (or a transient transport error) is skipped cleanly
    with a blocker instead of aborting the whole run.
    """
    written: dict[str, int] = {}
    blockers: dict[str, str] = {}
    for symbol in symbols:
        try:
            written[symbol] = backfill_symbol(
                symbol, data_dir=data_dir, fetch=fetch, sleep_s=sleep_s, max_pages=max_pages
            )
        except Exception as e:  # fail-soft: one symbol's outage never aborts the rest
            blockers[symbol] = f"{type(e).__name__}: {e}"
            log.warning("bybit_funding_symbol_failed", symbol=symbol, error=str(e))
    return {"written": written, "blockers": blockers}
