"""FRED / ALFRED macro-lake client (Stream-0 source).

Ports the proven research_fx logic:
  * keyless fredgraph CSV pull through a hardened transport (scout_realrate.py /
    scout_gvz.py) — FRED's Akamai edge HTTP/2-resets unusual UAs, so we send a
    browser UA and fall through httpx -> urllib -> curl;
  * the ALFRED vintage observations parser + _key() / 0.7s-throttle + 429 backoff
    (audit_b2_vintage.py).

Two acquisition modes, RAW daily observations only (publication-lag is a
downstream sleeve concern, never applied here):

  MODE A — LATEST (default for all five series): keyless
    GET https://fred.stlouisfed.org/graph/fredgraph.csv?id=<ID>
    CSV columns: ``observation_date,<ID>``. Holidays are ''/'.'  ->  coerced to
    NaN and dropped. DFII10 keyless==vintage was audited at 0.000 bp;
    DGS10/T10Y2Y/DTWEXBGS are market-observed/arithmetic with immaterial
    revisions, so LATEST is point-in-time defensible for them.

  MODE B — VINTAGE / ALFRED (optional, for BAMLH0A0HYM2 if $FRED_API_KEY is set):
    keyed GET https://api.stlouisfed.org/fred/series/observations  with
    realtime_start==realtime_end==<run_as_of>. ``value == '.'`` means missing.
    Throttle 0.7s/call + backoff on HTTP 429. No key -> skip vintage, fall back
    to LATEST and log a manifest note (NOT a hard blocker).

Target Parquet: ``raw_root / "fred" / "<series_id>.parquet"`` with columns
``[date, value, series_id, ts]`` (``ts`` = epoch MILLISECONDS int64, per the
shared time-series schema).

This module performs NO network at import time. ``run()`` does live network and
is exercised only by the orchestrator.
"""

from __future__ import annotations

import io
import os
import subprocess
import time
import urllib.request
from collections.abc import Callable
from pathlib import Path

import pandas as pd

from hermes.datalake.core import Manifest, RateLimiter, cached_pull

try:  # httpx is the preferred transport; urllib/curl are keyless fallbacks
    import httpx

    _HAS_HTTPX = True
except Exception:  # pragma: no cover - httpx is present in this env
    _HAS_HTTPX = False

SOURCE = "fred"

# Five series. All default to MODE A (LATEST). BAMLH0A0HYM2 additionally tries
# MODE B (ALFRED vintage) when a key is present.
SERIES_LATEST: list[str] = ["DFII10", "DGS10", "T10Y2Y", "BAMLH0A0HYM2", "DTWEXBGS"]
VINTAGE_SERIES: list[str] = ["BAMLH0A0HYM2"]

FREDGRAPH_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
ALFRED_OBS_URL = "https://api.stlouisfed.org/fred/series/observations"

# FRED/Akamai HTTP/2-resets unusual UAs (a custom "research-scout" UA reproducibly
# triggers err 92 / timeouts). A standard browser UA is accepted; curl uses its own
# default UA (no -A), which FRED also accepts.
_UA = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0 Safari/537.36"
    )
}

_TS_SCHEMA = ["date", "value", "series_id", "ts"]


# --------------------------------------------------------------------------- #
# hardened keyless transport (httpx -> urllib -> curl), all browser-UA, keyless
# --------------------------------------------------------------------------- #
def _get(url: str, timeout: int = 90) -> str:
    """Keyless GET returning the response body as text. No credentials.

    Transport order httpx -> urllib -> curl: the Python socket path can stall on
    first-byte for FRED while the system ``curl`` (proxy-aware) succeeds, so curl
    is the reliable last resort. All three carry a browser UA (or curl's default).
    """
    if _HAS_HTTPX:
        try:
            r = httpx.get(url, headers=_UA, timeout=timeout, follow_redirects=True)
            r.raise_for_status()
            return r.text
        except Exception:
            pass
    try:
        req = urllib.request.Request(url, headers=_UA)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read().decode("utf-8", "replace")
    except Exception:
        pass
    out = subprocess.run(
        ["curl", "-sSL", "--max-time", str(timeout), url],
        capture_output=True,
        text=True,
        check=True,
    )
    return out.stdout


# --------------------------------------------------------------------------- #
# PURE transforms (all real logic lives here; these are unit-tested offline)
# --------------------------------------------------------------------------- #
def _add_ts_ms(df: pd.DataFrame) -> pd.DataFrame:
    """Append ``ts`` = epoch MILLISECONDS int64 derived from the ``date`` column
    (interpreted as UTC midnight)."""
    if len(df) == 0:
        df = df.copy()
        df["ts"] = pd.Series([], dtype="int64")
        return df
    ts = pd.to_datetime(df["date"]).dt.tz_localize("UTC").astype("int64") // 1_000_000
    df = df.copy()
    df["ts"] = ts.astype("int64")
    return df


def _empty_frame(series_id: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.Series([], dtype="datetime64[ns]"),
            "value": pd.Series([], dtype="float64"),
            "series_id": pd.Series([], dtype="object"),
            "ts": pd.Series([], dtype="int64"),
        }
    )


def normalize_fredgraph_csv(text: str, series_id: str) -> pd.DataFrame:
    """Normalize a keyless fredgraph CSV body to ``[date, value, series_id, ts]``.

    The CSV has columns ``observation_date,<series_id>``; holidays are encoded as
    ''/'.' . We coerce the value column with ``errors='coerce'`` (turning ''/'.'/
    junk into NaN), drop missing rows, and sort ascending by date. Returns a
    well-typed empty frame (with the full schema) when there are no real rows.
    """
    raw = pd.read_csv(io.StringIO(text))
    if raw.shape[1] < 2 or len(raw) == 0:
        return _empty_frame(series_id)

    # First column is the date; second is the series value (name varies in case).
    date_col, val_col = raw.columns[0], raw.columns[1]
    out = pd.DataFrame()
    out["date"] = pd.to_datetime(raw[date_col], errors="coerce")
    out["value"] = pd.to_numeric(raw[val_col], errors="coerce")  # ''/'.' -> NaN
    out = out.dropna(subset=["date", "value"]).reset_index(drop=True)
    if len(out) == 0:
        return _empty_frame(series_id)
    out = out.sort_values("date").reset_index(drop=True)
    out["value"] = out["value"].astype("float64")
    out["series_id"] = series_id
    return _add_ts_ms(out)[_TS_SCHEMA]


def parse_alfred_observations(payload: dict, series_id: str) -> pd.DataFrame:
    """Parse an ALFRED ``series/observations`` JSON body to ``[date, value,
    series_id, ts]``. ``value == '.'`` (or non-numeric) means missing -> dropped.
    Safe on missing/empty ``observations``."""
    obs = (payload or {}).get("observations") or []
    if not obs:
        return _empty_frame(series_id)
    df = pd.DataFrame(obs)
    if "date" not in df.columns or "value" not in df.columns:
        return _empty_frame(series_id)
    out = pd.DataFrame()
    out["date"] = pd.to_datetime(df["date"], errors="coerce")
    out["value"] = pd.to_numeric(df["value"], errors="coerce")  # '.' -> NaN
    out = out.dropna(subset=["date", "value"]).reset_index(drop=True)
    if len(out) == 0:
        return _empty_frame(series_id)
    out = out.sort_values("date").reset_index(drop=True)
    out["value"] = out["value"].astype("float64")
    out["series_id"] = series_id
    return _add_ts_ms(out)[_TS_SCHEMA]


# --------------------------------------------------------------------------- #
# fetch helpers (thin shells over the pure transforms + the injected transport)
# --------------------------------------------------------------------------- #
def fetch_latest_frame(
    series_id: str,
    *,
    get: Callable[[str], str] = _get,
) -> pd.DataFrame:
    """MODE A: keyless GET of the fredgraph CSV for ``series_id`` -> normalized
    frame. ``get`` is injectable so tests fake the transport (no live network)."""
    url = FREDGRAPH_CSV_URL.format(series_id=series_id)
    text = get(url)
    return normalize_fredgraph_csv(text, series_id)


def _fred_key() -> str | None:
    """FRED API key from $FRED_API_KEY (read at call time), else ~/.fred_api_key,
    else None. Never logged/printed."""
    k = os.environ.get("FRED_API_KEY")
    if not k:
        p = os.path.expanduser("~/.fred_api_key")
        if os.path.exists(p):
            with open(p) as fh:
                k = fh.read().strip()
    return k or None


def fetch_vintage_frame(
    series_id: str,
    *,
    api_key: str,
    vintage: str,
    sleep: Callable[[float], None] = time.sleep,
    session=None,
) -> pd.DataFrame:
    """MODE B: keyed ALFRED pull of the full series AS PUBLISHED on ``vintage``
    (realtime_start==realtime_end==vintage). Throttle 0.7s/call + backoff on 429.
    ``session`` is an injectable object with ``.get(url, params=, timeout=) ->
    resp(.status_code/.json()/.raise_for_status())``; default uses httpx."""
    # ALFRED requires realtime_start/_end as YYYY-MM-DD; a full ISO run_as_of
    # timestamp 400s. Truncate to the date.
    vintage = str(vintage)[:10]
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "realtime_start": vintage,
        "realtime_end": vintage,
    }
    client = session if session is not None else httpx
    for attempt in range(5):
        sleep(0.7)  # stay under FRED's per-minute cap
        r = client.get(ALFRED_OBS_URL, params=params, timeout=30)
        if r.status_code == 429:
            sleep(2.0 * (attempt + 1))
            continue
        r.raise_for_status()
        return parse_alfred_observations(r.json(), series_id)
    raise RuntimeError(f"FRED 429 backoff exhausted for {series_id}")


# --------------------------------------------------------------------------- #
# public entrypoint — DO NOT EXECUTE in tests (live network)
# --------------------------------------------------------------------------- #
def run(raw_root: Path, manifest: Manifest, *, run_as_of: str) -> None:
    """Pull all five FRED series into ``raw_root/fred/<series_id>.parquet``.

    MODE A LATEST for all five (keyless). For BAMLH0A0HYM2, if $FRED_API_KEY is
    present, additionally write an ALFRED vintage snapshot frozen at ``run_as_of``
    to ``raw_root/fred/BAMLH0A0HYM2_vintage.parquet``; if no key, skip vintage and
    log a manifest note (fall back to LATEST, NOT a hard blocker).

    Fail-soft per series: cached_pull soft-handles fetch errors; we wrap setup so
    one series' failure becomes a blocker and the loop continues.
    """
    raw_root = Path(raw_root)
    rate = RateLimiter(60.0)  # be polite to fred.stlouisfed.org

    # --- MODE A: LATEST for all five ---------------------------------------
    for series_id in SERIES_LATEST:
        target = raw_root / SOURCE / f"{series_id}.parquet"
        params = {"id": series_id, "mode": "latest"}
        try:
            cached_pull(
                target,
                lambda sid=series_id: fetch_latest_frame(sid),
                source=SOURCE,
                endpoint="fredgraph.csv",
                params=params,
                manifest=manifest,
                ts_col="ts",
                rate=rate,
            )
        except Exception as e:  # setup/validation guard (fetch itself is fail-soft)
            manifest.blocker(
                source=SOURCE,
                endpoint="fredgraph.csv",
                params=params,
                error=repr(e),
                action=f"investigate fred LATEST setup for {series_id}",
            )

    # --- MODE B: ALFRED vintage for BAMLH0A0HYM2 (optional) -----------------
    key = _fred_key()
    for series_id in VINTAGE_SERIES:
        if not key:
            manifest.log(
                source=SOURCE,
                endpoint="series/observations",
                params={"series_id": series_id, "mode": "vintage"},
                status="skip",
                note=(
                    "no $FRED_API_KEY -> ALFRED vintage skipped; using keyless "
                    "LATEST for this series (set FRED_API_KEY to enable point-in-"
                    "time vintages for credit-spread series)"
                ),
            )
            continue
        target = raw_root / SOURCE / f"{series_id}_vintage.parquet"
        params = {"series_id": series_id, "mode": "vintage", "vintage": run_as_of}
        try:
            cached_pull(
                target,
                lambda sid=series_id: fetch_vintage_frame(
                    sid, api_key=key, vintage=run_as_of
                ),
                source=SOURCE,
                endpoint="series/observations",
                params=params,
                manifest=manifest,
                ts_col="ts",
                rate=rate,
            )
        except Exception as e:
            manifest.blocker(
                source=SOURCE,
                endpoint="series/observations",
                params=params,
                error=repr(e),
                action=f"investigate fred ALFRED vintage setup for {series_id}",
            )
