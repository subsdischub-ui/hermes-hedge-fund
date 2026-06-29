"""Gold / FX / metals lake client (Stream-0 source ``gold``).

Acquires three things into ``raw_root/`` Parquet, idempotently + fail-soft via
``hermes.datalake.core.cached_pull``:

  (a) DAILY PRICES for the gold complex + FX + DXY:
        XAUUSD (GC=F proxy), XAGUSD (SI=F proxy), GC=F, SI=F, GDX, DX-Y.NYB (DXY),
        and the FX crosses EURUSD / USDJPY / AUDUSD.
      We REUSE what already exists under ``data/fx/`` (and ``data/futures/``) —
      those files were produced by ``data/fx/fetch_fx.py`` using **yfinance** and
      already carry the canonical ``ts(ms),open,high,low,close,volume`` schema.
      For a symbol with an on-disk file we READ + REWRITE it into the lake (no
      network). Only genuinely missing symbols (GDX, DX-Y.NYB) are fetched live,
      using the SAME mechanism fetch_fx.py uses: ``yfinance.download(period='max',
      interval='1d')``.
      TARGET: ``raw_root/prices/<symbol>_1d.parquet`` (ts = epoch MILLISECONDS).

  (b) GLD daily ETF flow / holdings from SPDR's public historical archive
      (https://api.spdrgoldshares.com/api/v1/historical-archive). The endpoint
      returns an .xlsx workbook; sheet "US GLD Historical Archive" has one row per
      NYSE trading day (non-trading days are literal "US Holiday" rows, dropped).
      TARGET: ``raw_root/etf/gold_etf_flows.parquet``.
      NOTE: iShares IAU has no equivalent free keyless archive endpoint -> recorded
      as a blocker (see ``run``), not faked.

  (c) WGC central-bank gold demand (Goldhub): no keyless URL exists; it sits
      behind a free Goldhub registration. Recorded as a blocker (see ``run``),
      not faked.

Schema (time series): ``ts`` = epoch MILLISECONDS (int64). Prices keep
open/high/low/close/volume + symbol/interval. SPDR flow keeps close/ounces/tonnes/
nav_usd/share_volume + symbol. All real logic lives in the PURE functions below
(``canonicalize_ohlcv``, ``canonicalize_price_csv``, ``parse_spdr_archive_*``),
which are what the offline tests exercise. ``run`` only wires those into
``cached_pull``; it is NOT executed in tests (it does live network).
"""

from __future__ import annotations

import io
import os
import pathlib

import pandas as pd

from hermes.datalake.core import Manifest, RateLimiter, cached_pull

# --------------------------------------------------------------------------- #
# constants
# --------------------------------------------------------------------------- #
PRICE_COLS = ["ts", "open", "high", "low", "close", "volume", "symbol", "interval"]

# The on-disk lake produced by data/fx/fetch_fx.py. We reuse these instead of
# re-fetching. Absolute so cwd doesn't matter.
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[4]
DATA_FX_DIR = _REPO_ROOT / "data" / "fx"

# yahoo ticker -> the on-disk canonical-CSV key under data/fx/ (None = not on disk,
# must be fetched live). fetch_fx.py wrote gold/silver futures under the SPOT keys
# XAUUSD/XAGUSD (GC=F/SI=F are the underlying proxy), so GC=F and SI=F reuse those.
PRICE_SYMBOLS: dict[str, str | None] = {
    "XAUUSD": "XAUUSD",       # COMEX gold futures proxy, on disk
    "XAGUSD": "XAGUSD",       # COMEX silver futures proxy, on disk
    "GC=F": "XAUUSD",         # same series as XAUUSD on disk (fetch_fx.py mapping)
    "SI=F": "XAGUSD",         # same series as XAGUSD on disk
    "GDX": None,              # VanEck Gold Miners ETF — NOT on disk, fetch live
    "DX-Y.NYB": None,         # ICE US Dollar Index (DXY) — NOT on disk, fetch live
    "EURUSD": "EURUSD",
    "USDJPY": "USDJPY",
    "AUDUSD": "AUDUSD",
}

SPDR_URL = "https://api.spdrgoldshares.com/api/v1/historical-archive"
SPDR_PARAMS = {"product": "gld", "exchange": "NYSE", "lang": "en"}
SPDR_SHEET = "US GLD Historical Archive"
_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)

# yfinance is a free Yahoo client; default rate is conservative (Yahoo is unkeyed).
_YF_MAX_PER_MIN = 30.0

# SPDR ounces<->tonnes reconciliation constant (troy oz per metric tonne).
_OUNCES_PER_TONNE = 32150.7466


# --------------------------------------------------------------------------- #
# PURE TRANSFORMS (unit-tested offline)
# --------------------------------------------------------------------------- #
def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """yfinance returns a MultiIndex column frame for single tickers in newer
    versions; collapse it to the OHLCV field names."""
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = [c[0] for c in df.columns]
    return df


def canonicalize_ohlcv(df: pd.DataFrame, *, symbol: str, interval: str) -> pd.DataFrame:
    """yfinance OHLCV frame (DatetimeIndex) -> canonical PRICE_COLS frame.

    ``ts`` = epoch MILLISECONDS (int64, UTC). Mirrors data/fx/fetch_fx.py exactly:
    rename Open/High/Low/Close/Volume, ns->ms on the index, drop NaN closes, dedup
    on ts, sort. Empty in -> empty (typed) out so cached_pull fail-softs cleanly.
    """
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=PRICE_COLS)
    df = _flatten_columns(df)
    df = df.rename(
        columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adj Close": "adj_close",
            "Volume": "volume",
        }
    )
    idx = pd.to_datetime(df.index, utc=True)
    out = pd.DataFrame(
        {
            "ts": (idx.astype("int64") // 1_000_000).astype("int64"),  # ns -> ms
            "open": df["open"].astype(float).to_numpy(),
            "high": df["high"].astype(float).to_numpy(),
            "low": df["low"].astype(float).to_numpy(),
            "close": df["close"].astype(float).to_numpy(),
            "volume": df.get("volume", pd.Series(0.0, index=df.index))
            .fillna(0.0)
            .astype(float)
            .to_numpy(),
        }
    )
    out = (
        out.dropna(subset=["close"])
        .drop_duplicates(subset="ts")
        .sort_values("ts")
        .reset_index(drop=True)
    )
    out["symbol"] = symbol
    out["interval"] = interval
    return out[PRICE_COLS]


def canonicalize_price_csv(
    path: str | pathlib.Path, *, symbol: str, interval: str
) -> pd.DataFrame:
    """Read an EXISTING on-disk price file (ts already in ms) and re-emit the
    canonical PRICE_COLS frame. Reuses the data/fx/ lake without any network.

    Accepts ``.csv`` or ``.parquet`` (chosen by suffix). The on-disk schema is
    ``ts,open,high,low,close,volume`` (ts = epoch ms). We re-stamp symbol/interval
    (the on-disk key may differ from the requested symbol, e.g. GC=F reuses XAUUSD).
    """
    p = pathlib.Path(path)
    df = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=PRICE_COLS)
    out = pd.DataFrame(
        {
            "ts": pd.to_numeric(df["ts"], errors="coerce").astype("Int64"),
            "open": pd.to_numeric(df.get("open"), errors="coerce"),
            "high": pd.to_numeric(df.get("high"), errors="coerce"),
            "low": pd.to_numeric(df.get("low"), errors="coerce"),
            "close": pd.to_numeric(df.get("close"), errors="coerce"),
            "volume": pd.to_numeric(
                df.get("volume", pd.Series(0.0, index=df.index)), errors="coerce"
            ).fillna(0.0),
        }
    )
    out = out.dropna(subset=["ts", "close"]).copy()
    out["ts"] = out["ts"].astype("int64")
    out = (
        out.drop_duplicates(subset="ts").sort_values("ts").reset_index(drop=True)
    )
    out["symbol"] = symbol
    out["interval"] = interval
    return out[PRICE_COLS]


# ---- SPDR GLD historical archive ---- #
_SPDR_RENAME = {
    "Date": "date",
    "Closing Price": "close",
    "Total Ounces of Gold in the Trust": "ounces",
    "Tonnes of Gold": "tonnes",
    "Total Net Asset Value in the Trust": "nav_usd",
    "Daily Share Volume": "share_volume",
}
_SPDR_NUMERIC = ["close", "ounces", "tonnes", "nav_usd", "share_volume"]
_SPDR_OUT_COLS = ["ts", "date", "close", "ounces", "tonnes", "nav_usd", "share_volume", "symbol"]


def _normalize_spdr_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Shared core: a frame with SPDR's verbatim columns -> canonical flow frame.

    Drops "US Holiday" placeholder rows (their numeric cells are the literal
    string "US Holiday" -> coerce to NaN -> dropped on the ounces key). ``ts`` =
    epoch MILLISECONDS at the row's UTC midnight; ``date`` kept as a readable
    ISO date string for convenience.
    """
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=_SPDR_OUT_COLS)
    df = df.rename(columns=_SPDR_RENAME)
    for col in ["date", *_SPDR_NUMERIC]:
        if col not in df.columns:
            df[col] = pd.NA
    df = df[["date", *_SPDR_NUMERIC]].copy()

    dt = pd.to_datetime(df["date"], format="%d-%b-%Y", errors="coerce", utc=True)
    for col in _SPDR_NUMERIC:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["dt"] = dt
    df = df.dropna(subset=["dt", "ounces"]).reset_index(drop=True)
    if len(df) == 0:
        return pd.DataFrame(columns=_SPDR_OUT_COLS)
    df["ts"] = (df["dt"].astype("int64") // 1_000_000).astype("int64")
    df["date"] = df["dt"].dt.strftime("%Y-%m-%d")
    df = (
        df.drop_duplicates(subset="ts").sort_values("ts").reset_index(drop=True)
    )
    df["symbol"] = "GLD"
    return df[_SPDR_OUT_COLS]


def parse_spdr_archive_records(records: list[dict]) -> pd.DataFrame:
    """List-of-dict rows (SPDR's verbatim column names) -> canonical flow frame."""
    return _normalize_spdr_frame(pd.DataFrame(records))


def parse_spdr_archive_xlsx(xlsx_bytes: bytes, *, sheet_name: str = SPDR_SHEET) -> pd.DataFrame:
    """SPDR .xlsx workbook bytes -> canonical flow frame (parses the data sheet)."""
    raw = pd.read_excel(
        io.BytesIO(xlsx_bytes), sheet_name=sheet_name, header=0, engine="openpyxl"
    )
    return _normalize_spdr_frame(raw)


# --------------------------------------------------------------------------- #
# live fetchers (NOT called in tests)
# --------------------------------------------------------------------------- #
def _fetch_yf_daily(ticker: str, *, symbol: str, interval: str = "1d", period: str = "max"):
    """Zero-arg-able fetcher (bind with a closure in run). Uses yfinance exactly
    like data/fx/fetch_fx.py. Returns the canonical frame or an empty frame."""
    import yfinance as yf  # lazy: only needed on a live fetch

    raw = yf.download(
        ticker,
        period=period,
        interval=interval,
        progress=False,
        auto_adjust=False,
        threads=False,
    )
    return canonicalize_ohlcv(raw, symbol=symbol, interval=interval)


def _fetch_spdr_archive(client=None):
    """Pull the SPDR GLD historical archive and normalize it. ``client`` is an
    injectable httpx.Client-like object (default None -> construct a real one)."""
    own = False
    if client is None:
        import httpx  # lazy

        client = httpx.Client(
            follow_redirects=True,
            timeout=60.0,
            headers={"User-Agent": _USER_AGENT, "Referer": "https://www.spdrgoldshares.com/"},
        )
        own = True
    try:
        r = client.get(SPDR_URL, params=SPDR_PARAMS)
        r.raise_for_status()
        return parse_spdr_archive_xlsx(r.content, sheet_name=SPDR_SHEET)
    finally:
        if own:
            client.close()


# --------------------------------------------------------------------------- #
# public entrypoint — DO NOT execute in tests (live network)
# --------------------------------------------------------------------------- #
def run(raw_root: pathlib.Path, manifest: Manifest, *, run_as_of: str) -> None:
    """Acquire prices + GLD ETF flow into ``raw_root`` Parquet; record WGC + IAU
    blockers. Per-unit failures become manifest blockers and the loop continues
    (fail-soft). API keys (none required here) would be read from os.environ at
    call time.
    """
    raw_root = pathlib.Path(raw_root)
    rate = RateLimiter(_YF_MAX_PER_MIN)

    # (a) DAILY PRICES ------------------------------------------------------- #
    prices_dir = raw_root / "prices"
    for symbol, disk_key in PRICE_SYMBOLS.items():
        # filesystem-safe target name (DX-Y.NYB / GC=F contain shell-hostile chars)
        safe = symbol.replace("=", "_").replace(".", "_").replace("/", "_")
        target = prices_dir / f"{safe}_1d.parquet"

        if disk_key is not None:
            # REUSE the existing data/fx lake: prefer parquet, fall back to csv.
            src_pq = DATA_FX_DIR / f"{disk_key}_1d.parquet"
            src_csv = DATA_FX_DIR / f"{disk_key}_1d.csv"
            src = src_pq if src_pq.exists() else (src_csv if src_csv.exists() else None)
            if src is None:
                manifest.blocker(
                    source="gold",
                    endpoint="reuse:data/fx",
                    params={"symbol": symbol, "disk_key": disk_key},
                    error=f"expected on-disk price file {disk_key}_1d.(parquet|csv) not found",
                    action=(
                        f"run data/fx/fetch_fx.py to produce {disk_key}_1d, or add "
                        f"'{symbol}' to a live-fetch list"
                    ),
                )
                continue
            cached_pull(
                target,
                lambda s=src, sym=symbol: canonicalize_price_csv(s, symbol=sym, interval="1d"),
                source="gold",
                endpoint="reuse:data/fx",
                params={"symbol": symbol, "src": str(src)},
                manifest=manifest,
                ts_col="ts",
            )
        else:
            # genuinely missing -> fetch live with yfinance (same as fetch_fx.py)
            cached_pull(
                target,
                lambda t=symbol: _fetch_yf_daily(t, symbol=t, interval="1d", period="max"),
                source="gold",
                endpoint="yfinance.download",
                params={"symbol": symbol, "interval": "1d", "period": "max"},
                manifest=manifest,
                ts_col="ts",
                rate=rate,
            )

    # (b) GLD ETF FLOW ------------------------------------------------------- #
    etf_target = raw_root / "etf" / "gold_etf_flows.parquet"
    cached_pull(
        etf_target,
        _fetch_spdr_archive,
        source="gold",
        endpoint=SPDR_URL,
        params=SPDR_PARAMS,
        manifest=manifest,
        ts_col="ts",
        rate=rate,
    )
    # IAU has no free keyless archive endpoint. It is now provided by manual_gold
    # (manual iShares download). Only blocker if that manual ingestion is absent.
    iau_manual = raw_root / "etf" / "iau_flows" / "iau_flows.parquet"
    if iau_manual.exists():
        manifest.log(
            source="gold", endpoint="ishares:IAU-holdings", params={"etf": "IAU"},
            status="skip", path=str(iau_manual),
            note="IAU flows provided by manual_gold (manual iShares download) — no keyless archive needed",
        )
    else:
        manifest.blocker(
            source="gold",
            endpoint="ishares:IAU-holdings",
            params={"etf": "IAU"},
            error="no free keyless daily-holdings archive endpoint for iShares IAU",
            action=(
                "Malachie: IAU daily holdings are only on the iShares product page as a "
                "JS-rendered table / gated download. Either (1) accept GLD-only ETF flow, "
                "or (2) manually export IAU 'Fund Holdings' history from "
                "https://www.ishares.com/us/products/239561/ and drop a CSV the orchestrator "
                "can ingest. SKIPPED for now."
            ),
        )

    # (c) WGC central-bank demand ------------------------------------------- #
    # No keyless URL (Goldhub needs login). Now provided by manual_gold (manual
    # Goldhub download). Only blocker if that manual ingestion is absent.
    wgc_manual = raw_root / "wgc" / "cb_demand_gdt" / "cb_demand_gdt.parquet"
    if wgc_manual.exists():
        manifest.log(
            source="wgc", endpoint="goldhub:central-bank-net-purchases",
            params={"series": "central_bank_gold_demand"}, status="skip", path=str(wgc_manual),
            note="WGC CB demand provided by manual_gold (manual Goldhub download) — no keyless URL needed",
        )
    else:
        manifest.blocker(
            source="wgc",
            endpoint="goldhub:central-bank-net-purchases",
            params={"series": "central_bank_gold_demand"},
            error="WGC Goldhub central-bank demand requires a free login; no keyless URL",
            action=(
                "Malachie: register a FREE account at https://www.gold.org/goldhub (Goldhub), "
                "open Data > 'Central bank holdings' / 'Gold demand by sector', and download the "
                "quarterly central-bank NET PURCHASES (tonnes) series as XLSX/CSV; drop it where "
                "the orchestrator can read it -> raw_root/wgc/cb_demand.parquet. SKIPPED (not faked)."
            ),
        )

    # touch os.environ at call time so a future keyed variant fails loudly, not silently
    _ = os.environ.get("HERMES_GOLD_TOKEN")
