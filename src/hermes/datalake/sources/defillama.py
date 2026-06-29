"""DeFiLlama stablecoin-supply lake client (ported from data/onchain/fetch_onchain.py).

Public, no-key feeds under ``https://stablecoins.llama.fi``. The Mozilla
User-Agent header is MANDATORY: bare clients get a 403.

Pulls three series and outer-merges them on the calendar date:

  * ``usdt_supply`` — per-coin daily circulating from ``/stablecoin/1``
    (``.tokens[] -> {date: INT unix-seconds, circulating.peggedUSD}``).
  * ``usdc_supply`` — same shape, from ``/stablecoin/2``.
  * ``total_supply`` — from ``/stablecoincharts/all``
    (``[] -> {date: STR unix-seconds, totalCirculating.peggedUSD}``).
    We use ``totalCirculating.peggedUSD`` — the USD-PEGGED total — NOT
    ``totalCirculatingUSD`` which folds non-USD-pegged stables into the figure.

NOTE the date types differ across endpoints: ``/stablecoincharts/all`` gives
``date`` as a STRING of unix-seconds, ``/stablecoin/{id}`` as an INT — both are
cast through ``int(x)`` before ``pd.to_datetime(unit='s')``.

The merged frame is sorted ascending and ``net_change = total_supply.diff()``.
Output columns: ``[date, total_supply, usdt_supply, usdc_supply, net_change]``,
plus a ``ts`` epoch-MILLISECONDS int64 column (the lake time-series convention).

CAVEAT: treat the latest 1-2 days as PROVISIONAL — DeFiLlama revises the tail of
the as-reported series for a couple of days after first publication.

Acquisition only: pure transforms (``parse_total`` / ``parse_per_coin`` /
``build_stablecoin_supply``) hold all the logic and are unit-tested offline; the
network wrappers inject their HTTP client so tests fake the transport.
"""

from __future__ import annotations

import pathlib
from collections.abc import Callable
from typing import Any

import pandas as pd

from hermes.datalake.core import Manifest, cached_pull

# --------------------------------------------------------------------------- #
# constants (ported)
# --------------------------------------------------------------------------- #
SL_BASE = "https://stablecoins.llama.fi"
# bare urllib/httpx gets 403 from DeFiLlama — the Mozilla UA is MANDATORY.
_HEADERS = {"User-Agent": "Mozilla/5.0"}
_HTTP_TIMEOUT = 60.0

SOURCE = "defillama"
_SERIES = "stablecoin_supply"

# DeFiLlama stablecoin ids (verified live): USDT=1, USDC=2.
_USDT_ID = 1
_USDC_ID = 2

# Legacy CSV some research_fx consumers still read.
_LEGACY_CSV = pathlib.Path(
    "/Users/lacarbonnerie/Developer/hermes-hedge-fund/data/onchain/stablecoin_supply_daily.csv"
)


# --------------------------------------------------------------------------- #
# pure transforms (unit-tested offline — NO network here)
# --------------------------------------------------------------------------- #
def parse_total(charts: list[dict]) -> pd.DataFrame:
    """``/stablecoincharts/all`` -> DataFrame[date, total_supply].

    ``date`` arrives as a STRING of unix-seconds; cast ``int(x)`` before
    ``pd.to_datetime(unit='s')``. Uses ``totalCirculating.peggedUSD`` (the
    USD-pegged total) — NOT ``totalCirculatingUSD`` (which folds non-USD pegs).
    Sorted ascending.
    """
    rows = [
        {
            "date": pd.to_datetime(int(x["date"]), unit="s"),
            "total_supply": (x.get("totalCirculating") or {}).get("peggedUSD"),
        }
        for x in (charts or [])
    ]
    df = pd.DataFrame(rows, columns=["date", "total_supply"])
    if df.empty:
        return df
    return df.sort_values("date").reset_index(drop=True)


def parse_per_coin(payload: dict, value_col: str) -> pd.DataFrame:
    """``/stablecoin/{id}`` -> DataFrame[date, <value_col>].

    ``.tokens[] -> {date: INT unix-seconds, circulating.peggedUSD}``. ``date`` is
    cast through ``int(x)`` (robust whether it arrives as int or str). A missing
    optional ``circulating`` / ``peggedUSD`` degrades to NaN. Sorted ascending.
    """
    tokens = (payload or {}).get("tokens", []) or []
    rows = [
        {
            "date": pd.to_datetime(int(x["date"]), unit="s"),
            value_col: (x.get("circulating") or {}).get("peggedUSD"),
        }
        for x in tokens
    ]
    df = pd.DataFrame(rows, columns=["date", value_col])
    if df.empty:
        return df
    return df.sort_values("date").reset_index(drop=True)


def build_stablecoin_supply(
    total: pd.DataFrame, usdt: pd.DataFrame, usdc: pd.DataFrame
) -> pd.DataFrame:
    """Outer-merge the three series on ``date``, sort ascending, add ``net_change``.

    Returns columns ``[date, total_supply, usdt_supply, usdc_supply, net_change]``
    with ``net_change = total_supply.diff()`` (first row NaN).
    """
    out = (
        total.merge(usdt, on="date", how="outer")
        .merge(usdc, on="date", how="outer")
        .sort_values("date")
        .reset_index(drop=True)
    )
    # Guarantee all value columns exist even if a feed came back empty.
    for col in ("total_supply", "usdt_supply", "usdc_supply"):
        if col not in out.columns:
            out[col] = pd.NA
    out["net_change"] = out["total_supply"].diff()
    return out[["date", "total_supply", "usdt_supply", "usdc_supply", "net_change"]]


def add_ts_ms(df: pd.DataFrame) -> pd.DataFrame:
    """Add the lake time-series column ``ts`` = epoch MILLISECONDS int64 (from ``date``)."""
    out = df.copy()
    out["ts"] = (pd.to_datetime(out["date"]).astype("int64") // 1_000_000).astype("int64")
    return out


# --------------------------------------------------------------------------- #
# network wrappers (injectable client; default constructs a real httpx.Client)
# --------------------------------------------------------------------------- #
def _default_client_factory() -> Any:
    import httpx

    return httpx.Client(timeout=_HTTP_TIMEOUT, headers=_HEADERS, follow_redirects=True)


def _get_json(client_factory: Callable[[], Any], path: str) -> Any:
    """GET ``SL_BASE + path`` as JSON via a fresh client from ``client_factory``.

    The Mozilla UA is applied here too (belt-and-suspenders for clients whose
    factory did not set default headers)."""
    url = f"{SL_BASE}{path}"
    client = client_factory()
    with client as cx:
        r = cx.get(url, headers=_HEADERS)
        r.raise_for_status()
        return r.json()


def _fetch_supply_frame(client_factory: Callable[[], Any]) -> pd.DataFrame:
    """Fetch all three series, transform, merge, and tag with the ms ``ts`` column.

    This is the zero-arg-style fetch body handed to ``cached_pull`` (closed over
    ``client_factory``). Any network/parse error propagates so ``cached_pull``
    fail-softs it into a blocker.
    """
    charts = _get_json(client_factory, "/stablecoincharts/all")
    usdt_raw = _get_json(client_factory, f"/stablecoin/{_USDT_ID}")
    usdc_raw = _get_json(client_factory, f"/stablecoin/{_USDC_ID}")

    total = parse_total(charts)
    usdt = parse_per_coin(usdt_raw, "usdt_supply")
    usdc = parse_per_coin(usdc_raw, "usdc_supply")

    merged = build_stablecoin_supply(total, usdt, usdc)
    if merged.empty:
        return merged
    return add_ts_ms(merged)


def _maybe_emit_legacy_csv(df: pd.DataFrame, csv_path: pathlib.Path | None) -> None:
    """Best-effort re-emit of the legacy per-coin CSV for research_fx consumers.

    Columns mirror the old fetch_onchain.py output: date, usdt_circ_usd,
    usdc_circ_usd. ``csv_path=None`` disables the re-emit (tests pass None so the
    offline run() never writes outside its tmp dir). Never raises into the pull —
    a write failure is swallowed."""
    try:
        if csv_path is None or df is None or df.empty:
            return
        legacy = pd.DataFrame(
            {
                "date": pd.to_datetime(df["date"]).dt.date,
                "usdt_circ_usd": df["usdt_supply"],
                "usdc_circ_usd": df["usdc_supply"],
            }
        )
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        legacy.to_csv(csv_path, index=False)
    except Exception:
        pass


# --------------------------------------------------------------------------- #
# public entrypoint (DO NOT execute — it does live network unless faked)
# --------------------------------------------------------------------------- #
def run(
    raw_root: pathlib.Path,
    manifest: Manifest,
    *,
    run_as_of: str,
    client_factory: Callable[[], Any] | None = None,
    legacy_csv_path: pathlib.Path | None = _LEGACY_CSV,
) -> None:
    """Pull the DeFiLlama stablecoin-supply series into the lake (idempotent, fail-soft).

    Writes ``raw_root/defillama/stablecoin_supply/stablecoin_supply.parquet`` with
    columns ``[date, total_supply, usdt_supply, usdc_supply, net_change, ts]``.
    ``client_factory`` is injectable for tests (default: real ``httpx.Client``).
    ``legacy_csv_path`` (default: the real research_fx CSV) is also re-emitted as a
    convenience; pass ``None`` to disable. The latest 1-2 rows are PROVISIONAL
    (DeFiLlama revises the tail).
    """
    factory = client_factory or _default_client_factory
    raw_root = pathlib.Path(raw_root)
    target = raw_root / SOURCE / _SERIES / f"{_SERIES}.parquet"

    df = cached_pull(
        target,
        lambda: _fetch_supply_frame(factory),
        source=SOURCE,
        endpoint="/stablecoincharts/all + /stablecoin/{1,2}",
        params={"ids": [_USDT_ID, _USDC_ID], "metric": "totalCirculating.peggedUSD"},
        manifest=manifest,
        ts_col="ts",
    )

    if df is not None and not df.empty:
        _maybe_emit_legacy_csv(df, legacy_csv_path)
