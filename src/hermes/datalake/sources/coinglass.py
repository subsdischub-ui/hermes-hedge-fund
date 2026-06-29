"""CoinGlass V4 lake client — the rate-limited centerpiece of the Stream-0 lake.

Written FRESH (there is no CoinGlass code to port; the repo's
``hunt_h5_liqcascade.py`` only builds a causal liquidation proxy because the
free Binance feed is dead and CoinGlass needs a paid key).

Design (mirrors the SHARED CONTRACT in core.py):
  * ``CoinGlassClient(api_key, transport=None)`` — a thin sync wrapper. Every
    request carries ``CG-API-KEY``. The JSON envelope ``{code,'msg',data}`` is
    asserted (``code=='0'``) and unwrapped by ``_get`` to a ``list[dict]`` (or
    ``[]`` on a non-success / missing-data / shape-mismatch — fail-soft, no
    raise). It self-throttles two ways: a hard ``RateLimiter`` (Hobbyist =
    30 req/min, never exceeded) AND a soft back-off that reads the
    ``API-KEY-USE-LIMIT`` / ``API-KEY-MAX-LIMIT`` response headers.
  * Pure transforms (``normalize_futures`` / ``normalize_aggregated_oi`` /
    ``normalize_options`` / ``normalize_options_info`` / ``paginate_backward``)
    take already-fetched data and return DataFrames. These are unit-tested
    offline; ``_get`` is injectable so the client is too.
  * ``run(raw_root, manifest, *, run_as_of)`` — the live entrypoint. DEFINED,
    not executed here. It validates the universe against
    ``supported-exchange-pairs`` / ``supported-coins`` first (so the 30/min
    budget is never burnt on non-listed pairs), then drives every pull through
    ``core.cached_pull`` (idempotent resume + fail-soft) writing Parquet under
    ``raw_root/coinglass/...``.

ts convention: the API ts field is ``time`` in MILLISECONDS -> normalized to an
int64 column named ``ts`` everywhere.
"""

from __future__ import annotations

import contextlib
import datetime as _dt
import os
import pathlib
import time
from collections.abc import Callable
from typing import Any

import pandas as pd

from hermes.datalake.core import Manifest, RateLimiter, cached_pull

# --------------------------------------------------------------------------- #
# constants
# --------------------------------------------------------------------------- #
SOURCE = "coinglass"
BASE_URL = "https://open-api-v4.coinglass.com"
AUTH_HEADER = "CG-API-KEY"
ENV_KEY = "COINGLASS_API_KEY"

# Hobbyist tier: 30 req/min hard ceiling — NEVER exceed.
MAX_PER_MIN = 30
# Hobbyist interval floor is >=4h, so 6h/1d are fine; 1h/30m/15m return empty.
PAGE_LIMIT = 1000  # max bars per page (~2.7y of daily bars)

PAIRS = ["BTC", "ETH", "SOL", "AVAX", "SUI", "ARB", "WIF"]
OPTION_COINS = ["BTC", "ETH"]  # alts 404/empty on the options venue — skip + log
EXCHANGE = "Binance"  # per-exchange futures venue
INTERVALS = ["1d", "6h"]  # 1d = all-time (paginated); 6h = last 360d
SIXH_WINDOW_DAYS = 360  # Hobbyist 6h history cap

# Per-exchange futures series: endpoint path + the value columns to keep.
FUTURES_SERIES: dict[str, str] = {
    "top_lsr_account": "/api/futures/top-long-short-account-ratio/history",
    "top_lsr_position": "/api/futures/top-long-short-position-ratio/history",
    "global_lsr_account": "/api/futures/global-long-short-account-ratio/history",
    "taker_volume": "/api/futures/v2/taker-buy-sell-volume/history",  # NOTE the /v2/
    "open_interest": "/api/futures/open-interest/history",
    "funding_rate": "/api/futures/funding-rate/history",
    "liquidation": "/api/futures/liquidation/history",
}
# open-interest is requested in USD.
FUTURES_EXTRA_PARAMS: dict[str, dict[str, Any]] = {
    "open_interest": {"unit": "usd"},
}

AGG_OI_SERIES = "aggregated_open_interest"
AGG_OI_PATH = "/api/futures/open-interest/aggregated-history"

OPTIONS_SERIES: dict[str, str] = {
    "options_oi": "/api/option/exchange-oi-history",
    "options_vol": "/api/option/exchange-vol-history",
}
OPTIONS_INFO_PATH = "/api/option/info"
OPTIONS_INFO_SERIES = "options_info"

SUPPORTED_PAIRS_PATH = "/api/futures/supported-exchange-pairs"
SUPPORTED_COINS_PATH = "/api/futures/supported-coins"


# --------------------------------------------------------------------------- #
# pure transforms (unit-tested offline)
# --------------------------------------------------------------------------- #
def _stamp_ts(df: pd.DataFrame) -> pd.DataFrame:
    """Rename the API 'time' field (epoch ms) to an int64 'ts' column."""
    if "time" in df.columns:
        df = df.rename(columns={"time": "ts"})
    if "ts" in df.columns and len(df):
        df["ts"] = df["ts"].astype("int64")
    return df


def normalize_futures(
    data: list[dict[str, Any]],
    *,
    symbol: str,
    exchange: str,
    interval: str,
) -> pd.DataFrame:
    """Unwrap a futures history list-of-dicts into a long frame.

    ``time`` (epoch ms) -> int64 ``ts``; the source value columns are kept
    verbatim; provenance columns source/symbol/exchange/interval are stamped.
    An empty ``data`` yields a 0-row frame (never raises).
    """
    df = pd.DataFrame(list(data or []))
    df = _stamp_ts(df)
    # CoinGlass returns numeric value fields as a MIX of str and float in the same
    # column (live-verified on OHLC / funding / liquidation). Coerce every value
    # column to float64 so the frame is Parquet-writeable and dtype-uniform across
    # symbols; 'ts' stays int64.
    for col in df.columns:
        if col != "ts":
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["source"] = SOURCE
    df["symbol"] = symbol
    df["exchange"] = exchange
    df["interval"] = interval
    return df


def normalize_aggregated_oi(
    data: list[dict[str, Any]],
    *,
    symbol: str,
    interval: str,
) -> pd.DataFrame:
    """Aggregated OI OHLC (no single exchange) -> long frame with
    exchange='aggregated'."""
    return normalize_futures(data, symbol=symbol, exchange="aggregated", interval=interval)


def normalize_options(
    payload: dict[str, Any],
    *,
    symbol: str,
) -> pd.DataFrame:
    """TRANSPOSE the wide per-exchange options payload into LONG rows.

    Input shape (``data`` of /api/option/exchange-{oi,vol}-history)::

        {time_list: [ms,...], price_list: [px,...], data_map: {<exchange>: [v,...]}}

    Output rows: ``[ts, exchange, value, price, symbol, source]`` — one row per
    (timestamp, exchange). A series shorter than ``time_list`` only contributes
    its overlapping prefix (ragged-safe). Missing keys -> empty frame.
    """
    if not isinstance(payload, dict):
        return _empty_options_frame()
    time_list = payload.get("time_list") or []
    price_list = payload.get("price_list") or []
    data_map = payload.get("data_map") or {}
    if not time_list or not isinstance(data_map, dict) or not data_map:
        return _empty_options_frame()

    frames: list[pd.DataFrame] = []
    for exchange, series in data_map.items():
        series = series or []
        n = min(len(time_list), len(series))
        if n == 0:
            continue
        sub = pd.DataFrame(
            {
                "ts": list(time_list[:n]),
                "exchange": exchange,
                "value": list(series[:n]),
                "price": list(price_list[:n]) if len(price_list) >= n else [None] * n,
            }
        )
        frames.append(sub)

    if not frames:
        return _empty_options_frame()
    df = pd.concat(frames, ignore_index=True)
    df["ts"] = df["ts"].astype("int64")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["symbol"] = symbol
    df["source"] = SOURCE
    return df


def _empty_options_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=["ts", "exchange", "value", "price", "symbol", "source"])


def normalize_options_info(
    payload: list[dict[str, Any]] | dict[str, Any],
    *,
    symbol: str,
    run_as_of: str,
) -> pd.DataFrame:
    """The /api/option/info SNAPSHOT (no history): one row per exchange, stamped
    with run_as_of. ``payload`` is a list of per-exchange dicts (or a single
    dict). Empty -> 0-row frame."""
    rows = [payload] if isinstance(payload, dict) else list(payload or [])
    df = pd.DataFrame(rows)
    # Snapshot mixes numeric metrics + string fields (e.g. exchange_name). Coerce
    # numeric-looking columns to float, keep genuine string columns as string, so
    # the frame is Parquet-writeable with homogeneous per-column dtypes.
    for col in list(df.columns):
        num = pd.to_numeric(df[col], errors="coerce")
        if num.notna().any():
            df[col] = num
        elif df[col].notna().any():
            df[col] = df[col].astype("string")
    df["symbol"] = symbol
    df["source"] = SOURCE
    df["run_as_of"] = run_as_of
    return df


def paginate_backward(
    fetch_page: Callable[[int | None], pd.DataFrame],
    *,
    ts_col: str = "ts",
    max_pages: int = 200,
    floor_ts: int | None = None,
) -> pd.DataFrame:
    """Backward-paginate a series via ``end_time``, newest page first.

    Loop: fetch the newest page (``end_time=None``), then set
    ``end_time = earliest_ts - 1`` and fetch again, dedup on ``ts_col``; STOP
    when a page adds no NEW ts (or is empty). The returned frame is unique on
    ``ts_col`` and sorted ascending. ``fetch_page`` already returns a normalized
    frame (so ``ts_col`` is int64 ms). Capped at ``max_pages`` as a safety valve.

    ``floor_ts`` bounds a windowed pull (e.g. 6h capped at now-360d): walk back
    only until a page reaches the floor, then trim anything older. Because it
    paginates from NOW, the most-recent bars are always captured — unlike a
    single forward page from ``start_time``, which returns only the oldest 1000.
    """
    seen: set[Any] = set()
    collected: list[pd.DataFrame] = []
    end_time: int | None = None

    for _ in range(max_pages):
        page = fetch_page(end_time)
        if page is None or len(page) == 0 or ts_col not in page.columns:
            break
        new_mask = ~page[ts_col].isin(seen)
        new_rows = page[new_mask]
        if len(new_rows) == 0:
            break  # no progress -> stop
        collected.append(new_rows)
        seen.update(new_rows[ts_col].tolist())
        earliest = int(page[ts_col].min())
        end_time = earliest - 1
        if floor_ts is not None and earliest <= floor_ts:
            break  # reached the windowed floor (e.g. now-360d)

    if not collected:
        return fetch_page_empty(fetch_page, ts_col)
    out = pd.concat(collected, ignore_index=True)
    out = out.drop_duplicates(subset=[ts_col]).sort_values(ts_col).reset_index(drop=True)
    if floor_ts is not None:
        out = out[out[ts_col] >= floor_ts].reset_index(drop=True)
    return out


def fetch_page_empty(fetch_page: Callable[[int | None], pd.DataFrame], ts_col: str) -> pd.DataFrame:
    """Return an empty frame with the same columns the fetcher would produce
    (so downstream schema-stamping is preserved even on a no-data series)."""
    try:
        probe = fetch_page(None)
        if probe is not None:
            return probe.iloc[0:0]
    except Exception:
        pass
    return pd.DataFrame(columns=[ts_col])


# --------------------------------------------------------------------------- #
# the client
# --------------------------------------------------------------------------- #
class CoinGlassClient:
    """Sync CoinGlass V4 client with an injectable transport.

    ``transport`` must expose ``get(url, *, headers, params) -> response`` where
    ``response`` has ``.json()`` and ``.headers`` (a mapping). Default ``None``
    builds a real ``httpx.Client``. ``_get`` unwraps the envelope and returns a
    ``list[dict]`` (``[]`` on any non-success). The ``RateLimiter`` (30/min) is
    enforced before every request; ``monotonic``/``sleep`` are injected so tests
    drive a fake clock.
    """

    def __init__(
        self,
        api_key: str,
        transport: Any | None = None,
        *,
        timeout: float = 30.0,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.api_key = api_key
        self._sleep = sleep
        self._owns_transport = transport is None
        if transport is None:
            import httpx  # lazy: only needed for live runs

            transport = httpx.Client(timeout=timeout)
        self._transport = transport
        self._rate = RateLimiter(MAX_PER_MIN, monotonic=monotonic, sleep=sleep)

    # -- low-level ---------------------------------------------------------- #
    def _get(self, path: str, **params: Any) -> list[dict[str, Any]]:
        """GET ``path`` with auth header + params; return the unwrapped ``data``
        list, or ``[]`` on any non-success (code!='0', missing data, wrong
        shape, or transport error). Fail-soft: never raises."""
        self._rate.wait()  # hard 30/min throttle
        url = BASE_URL + path
        headers = {AUTH_HEADER: self.api_key, "accept": "application/json"}
        clean = {k: v for k, v in params.items() if v is not None}
        try:
            resp = self._transport.get(url, headers=headers, params=clean)
        except Exception:
            return []

        # Soft self-throttle from rate-limit headers (back off when near the cap).
        self._respect_rate_headers(getattr(resp, "headers", {}) or {})

        try:
            envelope = resp.json()
        except Exception:
            return []
        if not isinstance(envelope, dict):
            return []
        if str(envelope.get("code")) != "0":
            return []
        data = envelope.get("data")
        if data is None:
            return []
        if isinstance(data, dict):
            return [data]  # caller decides (e.g. supported_pairs / options payload)
        if isinstance(data, list):
            return data
        return []

    def _get_payload(self, path: str, **params: Any) -> Any:
        """Like ``_get`` but returns the raw ``data`` object (dict or list) for
        endpoints whose ``data`` is a wide map (options) — ``{}`` on failure."""
        self._rate.wait()
        url = BASE_URL + path
        headers = {AUTH_HEADER: self.api_key, "accept": "application/json"}
        clean = {k: v for k, v in params.items() if v is not None}
        try:
            resp = self._transport.get(url, headers=headers, params=clean)
        except Exception:
            return {}
        self._respect_rate_headers(getattr(resp, "headers", {}) or {})
        try:
            envelope = resp.json()
        except Exception:
            return {}
        if not isinstance(envelope, dict) or str(envelope.get("code")) != "0":
            return {}
        return envelope.get("data") or {}

    def _respect_rate_headers(self, headers: Any) -> None:
        """Read API-KEY-USE-LIMIT / API-KEY-MAX-LIMIT; if we are at/over the
        per-minute cap, sleep to the top of the next minute window."""
        try:
            use = headers.get("API-KEY-USE-LIMIT")
            mx = headers.get("API-KEY-MAX-LIMIT")
            if use is None or mx is None:
                return
            use_i = int(use)
            max_i = int(mx)
        except Exception:
            return
        if max_i > 0 and use_i >= max_i:
            # We've consumed the window; pause a full min-interval to be safe.
            self._sleep(60.0 / MAX_PER_MIN)

    def close(self) -> None:
        if self._owns_transport:
            with contextlib.suppress(Exception):
                self._transport.close()

    # -- validation helpers ------------------------------------------------- #
    def supported_pairs(self, *, exchange: str) -> set[str]:
        """Return the set of instrument_ids listed on ``exchange`` (so run() can
        skip non-listed pairs like WIF/SUI/ARB on a venue without burning
        budget). Empty set on failure."""
        data = self._get_payload(SUPPORTED_PAIRS_PATH)
        if not isinstance(data, dict):
            return set()
        venue = data.get(exchange) or []
        out: set[str] = set()
        for item in venue:
            if not isinstance(item, dict):
                continue
            iid = item.get("instrument_id") or item.get("symbol") or item.get("pair")
            if iid:
                out.add(str(iid))
        return out

    def supported_coins(self) -> set[str]:
        """Return the set of supported coin symbols (bare, e.g. 'BTC'). Empty set
        on failure."""
        data = self._get_payload(SUPPORTED_COINS_PATH)
        if isinstance(data, list):
            return {str(c) for c in data if c}
        if isinstance(data, dict):
            # some shapes wrap under a key
            for v in data.values():
                if isinstance(v, list):
                    return {str(c) for c in v if c}
        return set()


# --------------------------------------------------------------------------- #
# time helpers
# --------------------------------------------------------------------------- #
def _now_ms() -> int:
    return int(time.time() * 1000)


def _sixh_start_ms(now_ms: int | None = None) -> int:
    now_ms = now_ms if now_ms is not None else _now_ms()
    return now_ms - SIXH_WINDOW_DAYS * 86_400_000


def _parse_run_as_of_ms(run_as_of: str) -> int:
    """Best-effort: parse run_as_of (ISO date/datetime) to epoch ms; fall back to
    wall-clock now."""
    try:
        s = run_as_of.replace("Z", "+00:00")
        dt = _dt.datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=_dt.UTC)
        return int(dt.timestamp() * 1000)
    except Exception:
        return _now_ms()


# --------------------------------------------------------------------------- #
# live entrypoint — DEFINED, not executed here (does live network)
# --------------------------------------------------------------------------- #
def run(raw_root: pathlib.Path, manifest: Manifest, *, run_as_of: str) -> None:
    """Pull every CoinGlass series for the universe into
    ``raw_root/coinglass/...`` Parquet, idempotently + fail-soft.

    Reads the API key from ``os.environ`` AT CALL TIME (the orchestrator loads
    ``.env`` first). One missing key is a single blocker, not a crash. Each
    per-symbol/per-series unit is wrapped so one failure becomes a
    ``manifest.blocker`` and the loop continues.
    """
    raw_root = pathlib.Path(raw_root)
    root = raw_root / SOURCE

    api_key = os.environ.get(ENV_KEY)
    if not api_key:
        manifest.blocker(
            source=SOURCE,
            endpoint="(auth)",
            params={"env": ENV_KEY},
            error=f"{ENV_KEY} not set in environment",
            action=f"Malachie: export {ENV_KEY}=<CoinGlass Hobbyist key> in .env, then re-run.",
        )
        return

    client = CoinGlassClient(api_key=api_key)
    rate = client._rate  # share the one throttle with cached_pull
    now_ms = _parse_run_as_of_ms(run_as_of)
    sixh_start = _sixh_start_ms(now_ms)

    try:
        # ----- validate universe first (cheap; protects the 30/min budget) ---
        try:
            listed_pairs = client.supported_pairs(exchange=EXCHANGE)
        except Exception as e:  # never abort the run on a validation hiccup
            listed_pairs = set()
            manifest.blocker(
                source=SOURCE, endpoint=SUPPORTED_PAIRS_PATH, params={"exchange": EXCHANGE},
                error=repr(e), action="could not validate pairs; proceeding optimistically",
            )
        try:
            listed_coins = client.supported_coins()
        except Exception:
            listed_coins = set()

        # ----- per-exchange futures series ----------------------------------
        for coin in PAIRS:
            sym = f"{coin}USDT"
            if listed_pairs and sym not in listed_pairs:
                manifest.log(
                    source=SOURCE, endpoint=SUPPORTED_PAIRS_PATH,
                    params={"exchange": EXCHANGE, "symbol": sym},
                    status="empty", note=f"{sym} not supported on {EXCHANGE} — skipped (research note)",
                )
                continue
            for series_name, path in FUTURES_SERIES.items():
                for interval in INTERVALS:
                    _pull_futures_series(
                        client, manifest, rate, root,
                        series_name=series_name, path=path, coin=coin, symbol=sym,
                        interval=interval, now_ms=now_ms, sixh_start=sixh_start,
                    )

        # ----- aggregated OI (bare coin, no exchange) -----------------------
        for coin in PAIRS:
            if listed_coins and coin not in listed_coins:
                manifest.log(
                    source=SOURCE, endpoint=SUPPORTED_COINS_PATH, params={"symbol": coin},
                    status="empty", note=f"{coin} not a supported coin — skipped aggregated OI",
                )
                continue
            for interval in INTERVALS:
                _pull_aggregated_oi(
                    client, manifest, rate, root,
                    coin=coin, interval=interval, now_ms=now_ms, sixh_start=sixh_start,
                )

        # ----- options (BTC/ETH only) ---------------------------------------
        for coin in OPTION_COINS:
            for series_name, path in OPTIONS_SERIES.items():
                _pull_options_series(
                    client, manifest, rate, root,
                    series_name=series_name, path=path, coin=coin,
                )
            _pull_options_info(client, manifest, rate, root, coin=coin, run_as_of=run_as_of)
    finally:
        client.close()


# --------------------------------------------------------------------------- #
# per-unit pulls (each fail-soft via cached_pull)
# --------------------------------------------------------------------------- #
def _pull_futures_series(
    client: CoinGlassClient, manifest: Manifest, rate: RateLimiter, root: pathlib.Path,
    *, series_name: str, path: str, coin: str, symbol: str, interval: str,
    now_ms: int, sixh_start: int,
) -> None:
    target = root / series_name / f"{symbol}_{interval}.parquet"
    extra = FUTURES_EXTRA_PARAMS.get(series_name, {})

    def _page(end_time: int | None, start_time: int | None) -> pd.DataFrame:
        data = client._get(
            path, exchange=EXCHANGE, symbol=symbol, interval=interval,
            limit=PAGE_LIMIT, start_time=start_time, end_time=end_time, **extra,
        )
        return normalize_futures(data, symbol=symbol, exchange=EXCHANGE, interval=interval)

    if interval == "6h":  # windowed: backward from now, floored at now-360d (recent bars first)
        def fetch() -> pd.DataFrame:
            return paginate_backward(lambda et: _page(et, None), ts_col="ts", floor_ts=sixh_start)
    else:  # 1d all-time -> backward pagination
        def fetch() -> pd.DataFrame:
            return paginate_backward(lambda et: _page(et, None), ts_col="ts")

    cached_pull(
        target, fetch,
        source=SOURCE, endpoint=path,
        params={"exchange": EXCHANGE, "symbol": symbol, "interval": interval, **extra},
        manifest=manifest, ts_col="ts", rate=rate,
    )


def _pull_aggregated_oi(
    client: CoinGlassClient, manifest: Manifest, rate: RateLimiter, root: pathlib.Path,
    *, coin: str, interval: str, now_ms: int, sixh_start: int,
) -> None:
    target = root / AGG_OI_SERIES / f"{coin}_{interval}.parquet"

    def _page(end_time: int | None, start_time: int | None) -> pd.DataFrame:
        data = client._get(
            AGG_OI_PATH, symbol=coin, interval=interval, unit="usd",
            limit=PAGE_LIMIT, start_time=start_time, end_time=end_time,
        )
        return normalize_aggregated_oi(data, symbol=coin, interval=interval)

    if interval == "6h":  # windowed: backward from now, floored at now-360d (recent bars first)
        def fetch() -> pd.DataFrame:
            return paginate_backward(lambda et: _page(et, None), ts_col="ts", floor_ts=sixh_start)
    else:
        def fetch() -> pd.DataFrame:
            return paginate_backward(lambda et: _page(et, None), ts_col="ts")

    cached_pull(
        target, fetch,
        source=SOURCE, endpoint=AGG_OI_PATH,
        params={"symbol": coin, "interval": interval, "unit": "usd"},
        manifest=manifest, ts_col="ts", rate=rate,
    )


def _pull_options_series(
    client: CoinGlassClient, manifest: Manifest, rate: RateLimiter, root: pathlib.Path,
    *, series_name: str, path: str, coin: str,
) -> None:
    target = root / series_name / f"{coin}.parquet"

    def fetch() -> pd.DataFrame:
        payload = client._get_payload(path, symbol=coin, unit="usd", range="all")
        if not isinstance(payload, dict):
            return _empty_options_frame()
        return normalize_options(payload, symbol=coin)

    cached_pull(
        target, fetch,
        source=SOURCE, endpoint=path,
        params={"symbol": coin, "unit": "usd", "range": "all"},
        manifest=manifest, ts_col="ts", rate=rate,
    )


def _pull_options_info(
    client: CoinGlassClient, manifest: Manifest, rate: RateLimiter, root: pathlib.Path,
    *, coin: str, run_as_of: str,
) -> None:
    target = root / OPTIONS_INFO_SERIES / f"{coin}.parquet"

    def fetch() -> pd.DataFrame:
        data = client._get(OPTIONS_INFO_PATH, symbol=coin)
        return normalize_options_info(data, symbol=coin, run_as_of=run_as_of)

    # Snapshot: no expected_range (always 'skip' once present); ts_col=run_as_of-stamped.
    cached_pull(
        target, fetch,
        source=SOURCE, endpoint=OPTIONS_INFO_PATH,
        params={"symbol": coin},
        manifest=manifest, ts_col=None, rate=rate,
    )
