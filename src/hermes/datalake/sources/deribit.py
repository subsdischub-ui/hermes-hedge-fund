"""Deribit options-surface lake client (BTC & ETH only) — PORTED.

Two surfaces, with very different backfill properties:

  (A) DVOL HISTORY — the only multi-year backfillable options-surface signal.
      ``public/get_volatility_index_data`` returns hourly OHLC of Deribit's
      implied-vol index, paginated BACKWARD via ``result.continuation`` to
      inception (BTC & ETH both 2021-03-24). Stored RAW & UNLAGGED at
      ``raw_root/deribit/dvol/<ccy>.parquet`` with columns
      ``[ts, dvol_o, dvol_h, dvol_l, dvol]``.

      CAUSALITY NOTE: the DVOL bar labelled ``ts`` is OPEN-timestamped — its
      close only prints at ts+1h — so any feature built off this column MUST lag
      it by one bar (the convention applied in
      ``research_options/deribit_iv.py`` and ``hermes/data/newdata.py``). We do
      NOT lag here: the raw lake stores ground-truth OHLC; the 1-bar causal lag
      is a downstream feature-engineering concern.

  (B) DAILY LIVE SURFACE SNAPSHOT — CANNOT be backfilled, so it is a forward
      log: one dated row APPENDED per run to
      ``raw_root/deribit/surface_snapshot/<ccy>.parquet``. A single
      ``public/get_book_summary_by_currency`` returns the whole option chain
      (mark_iv, open_interest, underlying_price); we compute, per currency, the
      ATM IV, the 25-delta risk-reversal (RR25 = IV(25dC) - IV(25dP)) at the 1M
      and 3M tenors, the normalized skew (RR25 / ATM_IV), and the term slope.
      The 25-delta strikes need per-instrument greeks, fetched via
      ``public/ticker`` only for the handful of strikes near |delta|=0.25.

Public REST, GET, payload under ``.result``; NO auth. SOL / alts have no options
market on Deribit — skipped, not a blocker.

ALL real logic is in PURE FUNCTIONS (parse, paginate+dedup, RR25/ATM/term math)
that take already-fetched data and return frames; ``run()`` is the only thing
that touches the network and it is fail-soft per currency.
"""

from __future__ import annotations

import datetime as dt
import os
import pathlib
import time
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from hermes.datalake.core import Manifest, RateLimiter, cached_pull

# --------------------------------------------------------------------------- #
# constants
# --------------------------------------------------------------------------- #
API = "https://www.deribit.com/api/v2/public"
SOURCE = "deribit"
CCYS = ("BTC", "ETH")
# Deribit DVOL inception for BTC & ETH (both 2021-03-24). Used as the backward
# pagination floor and the stop condition.
INCEPTION_MS = int(dt.datetime(2021, 3, 24, tzinfo=dt.UTC).timestamp() * 1000)

_HOUR_MS = 3_600_000
_DVOL_COLS = ["ts", "dvol_o", "dvol_h", "dvol_l", "dvol"]
_HTTP_TIMEOUT = 30.0
# Deribit public REST is generous; keep a polite ceiling well under any limit.
_RATE_PER_MIN = 120.0
# How many strikes per wing we ticker for greeks (closest-to-ATM first, both
# wings) — enough to bracket |delta|=0.25 without tickering the whole chain.
_DELTA_STRIKES_PER_WING = 6


# --------------------------------------------------------------------------- #
# pure: instrument-name parsing
# --------------------------------------------------------------------------- #
def parse_inst(name: str) -> tuple[dt.datetime, float, str] | None:
    """Parse a Deribit option name ``CCY-DDMMMYY-STRIKE-C/P``.

    Returns ``(expiry_utc, strike, cp)`` or ``None`` for non-options / malformed.
    """
    parts = name.split("-")  # CCY-DDMMMYY-STRIKE-C/P
    if len(parts) != 4:
        return None
    try:
        exp = dt.datetime.strptime(parts[1], "%d%b%y").replace(tzinfo=dt.UTC)
        return exp, float(parts[2]), parts[3]
    except ValueError:
        return None


# --------------------------------------------------------------------------- #
# pure: DVOL pages -> frame, and backward paginator
# --------------------------------------------------------------------------- #
def dvol_pages_to_frame(rows: list[list]) -> pd.DataFrame:
    """Concatenated raw DVOL rows ``[ts_ms, open, high, low, close]`` ->
    deduped, ts-sorted frame with the canonical columns. ``ts`` is int64 ms.
    """
    if not rows:
        return pd.DataFrame(columns=_DVOL_COLS).astype({"ts": "int64"})
    df = pd.DataFrame(rows, columns=_DVOL_COLS)
    df = df.drop_duplicates(subset="ts").sort_values("ts").reset_index(drop=True)
    df["ts"] = df["ts"].astype("int64")
    return df


def paginate_dvol(
    ccy: str,
    *,
    client: Any | None = None,
    now_ms: int | None = None,
    start_ms: int = INCEPTION_MS,
    resolution: int = 3600,
    max_calls: int = 200,
    sleep: Callable[[float], None] | None = None,
) -> pd.DataFrame:
    """Pull the full hourly DVOL OHLC history for ``ccy`` BACKWARD via
    ``continuation`` until exhausted / inception.

    ``client`` is an httpx.Client-like object with ``.get(url, params=...)``
    returning a response exposing ``.raise_for_status()`` and ``.json()``. If
    ``None``, a real ``httpx.Client`` is constructed (network — used only by
    ``run()``; never in tests). Returns the canonical 5-column frame; empty frame
    on no data.
    """
    own_client = client is None
    if own_client:
        import httpx

        client = httpx.Client(timeout=_HTTP_TIMEOUT)

    if now_ms is None:
        now_ms = int(time.time() * 1000)

    url = f"{API}/get_volatility_index_data"
    end = now_ms
    chunks: list[list] = []
    try:
        for _ in range(max_calls):
            resp = client.get(
                url,
                params={
                    "currency": ccy,
                    "start_timestamp": start_ms,
                    "end_timestamp": end,
                    "resolution": resolution,
                },
            )
            resp.raise_for_status()
            res = resp.json().get("result", {}) or {}
            data = res.get("data", []) or []
            if not data:
                break
            chunks.extend(data)
            earliest = data[0][0]
            cont = res.get("continuation")
            # Stop at the natural end (no continuation) OR once we have reached
            # inception, even if Deribit still hands back a continuation token.
            if cont is None or earliest <= start_ms:
                break
            end = earliest - 1
            if sleep is not None:
                sleep(0.12)
    finally:
        if own_client:
            client.close()

    return dvol_pages_to_frame(chunks)


# --------------------------------------------------------------------------- #
# pure: book-summary normalization
# --------------------------------------------------------------------------- #
def book_summary_to_rows(summary: list[dict]) -> list[dict]:
    """Filter+parse a ``get_book_summary_by_currency`` option chain.

    Keeps only rows that parse as options AND carry a ``mark_iv``. Each kept row
    becomes ``{inst, exp, strike, cp, iv, oi, u}``.
    """
    rows: list[dict] = []
    for s in summary:
        name = s.get("instrument_name")
        if not name:
            continue
        pi = parse_inst(name)
        iv = s.get("mark_iv")
        if pi is None or iv is None:
            continue
        exp, strike, cp = pi
        rows.append(
            {
                "inst": name,
                "exp": exp,
                "strike": float(strike),
                "cp": cp,
                "iv": float(iv),
                "oi": s.get("open_interest"),
                "u": s.get("underlying_price"),
            }
        )
    return rows


# --------------------------------------------------------------------------- #
# pure: 25-delta IV interpolation
# --------------------------------------------------------------------------- #
def interp_25d_iv(sub: list[dict], cp: str) -> float | None:
    """Interpolate IV at ``|delta|=0.25`` for calls ('C') or puts ('P') within
    one expiry. ``sub`` rows carry ``cp``, ``delta`` (signed), ``iv``.

    Requires 0.25 to be bracketed by the available |delta| points (no
    extrapolation), and at least two valid points.
    """
    pts = [
        (abs(r["delta"]), r["iv"])
        for r in sub
        if r.get("cp") == cp and r.get("delta") is not None
    ]
    pts = sorted(pts)
    if len(pts) < 2:
        return None
    deltas = np.array([p[0] for p in pts])
    ivs = np.array([p[1] for p in pts])
    if deltas.min() > 0.25 or deltas.max() < 0.25:
        return None  # 0.25 not bracketed -> unreliable extrapolation
    return float(np.interp(0.25, deltas, ivs))


# --------------------------------------------------------------------------- #
# pure: term structure + ATM
# --------------------------------------------------------------------------- #
def _dte(exp: dt.datetime, now: dt.datetime) -> float:
    return (exp - now).total_seconds() / 86400.0


def _atm_iv_for_expiry(rows: list[dict], underlying: float) -> float:
    atm = min(rows, key=lambda r: abs(r["strike"] - underlying))
    return float(atm["iv"])


def _term_points(rows: list[dict], underlying: float, now: dt.datetime):
    """List of ``(dte, atm_iv)`` per expiry with dte>0, sorted by dte."""
    exps = sorted({r["exp"] for r in rows})
    term = []
    for e in exps:
        se = [r for r in rows if r["exp"] == e]
        term.append((_dte(e, now), _atm_iv_for_expiry(se, underlying)))
    term = [t for t in term if t[0] > 0]
    term.sort(key=lambda t: t[0])
    return term


def _interp_atm_at(term, target_days: float) -> float:
    """ATM IV interpolated at ``target_days`` along the term structure.

    Single point -> that point; otherwise piecewise-linear (np.interp clamps at
    the ends, which is the desired behaviour for a tenor outside the listed range).
    """
    if not term:
        return float("nan")
    if len(term) == 1:
        return float(term[0][1])
    d = np.array([t[0] for t in term])
    v = np.array([t[1] for t in term])
    return float(np.interp(target_days, d, v))


def _nearest_expiry(rows: list[dict], now: dt.datetime, target_days: float):
    """The expiry whose dte is closest to ``target_days`` (dte>0 only)."""
    exps = sorted({r["exp"] for r in rows})
    cand = [(e, _dte(e, now)) for e in exps]
    cand = [(e, d) for e, d in cand if d > 0]
    if not cand:
        return None
    return min(cand, key=lambda ed: abs(ed[1] - target_days))[0]


def _rr25_for_tenor(
    rows: list[dict],
    now: dt.datetime,
    target_days: float,
    delta_fn: Callable[[str], float | None],
) -> float:
    """RR25 = IV(25dC) - IV(25dP) for the expiry bracketing ``target_days``.

    Tickers (via ``delta_fn``) only the strikes nearest ATM on each wing of the
    chosen expiry, enough to bracket |delta|=0.25, then interpolates. Returns NaN
    if either wing cannot bracket 0.25.
    """
    exp = _nearest_expiry(rows, now, target_days)
    if exp is None:
        return float("nan")
    sub = [r for r in rows if r["exp"] == exp]
    if not sub:
        return float("nan")
    u = next((r["u"] for r in sub if r.get("u") is not None), None)
    if u is None:
        return float("nan")
    # Limit greek lookups to the strikes near ATM on each wing.
    calls = sorted((r for r in sub if r["cp"] == "C"), key=lambda r: abs(r["strike"] - u))
    puts = sorted((r for r in sub if r["cp"] == "P"), key=lambda r: abs(r["strike"] - u))
    chosen = calls[:_DELTA_STRIKES_PER_WING] + puts[:_DELTA_STRIKES_PER_WING]
    enriched = []
    for r in chosen:
        d = delta_fn(r["inst"])
        enriched.append({"cp": r["cp"], "iv": r["iv"], "delta": d})
    iv_c = interp_25d_iv(enriched, "C")
    iv_p = interp_25d_iv(enriched, "P")
    if iv_c is None or iv_p is None:
        return float("nan")
    return iv_c - iv_p


def compute_surface_snapshot(
    ccy: str,
    rows: list[dict],
    *,
    now: dt.datetime,
    ts_ms: int,
    delta_fn: Callable[[str], float | None],
) -> pd.DataFrame | None:
    """Build one dated surface-snapshot row for ``ccy`` from a parsed chain.

    Columns:
      ts (run snapshot ms), ccy, underlying, atm_iv (strike nearest underlying
      on the near tenor), atm_iv_1m / atm_iv_3m (term-interpolated at 30/90d),
      rr25_1m / rr25_3m (IV(25dC)-IV(25dP) at the 30d/90d-bracketing expiries),
      skew_1m / skew_3m (rr25 / atm_iv, normalized), term_slope (atm_iv_3m -
      atm_iv_1m), n_options.

    Returns ``None`` (fail-soft) if the chain is empty / has no underlying.
    """
    if not rows:
        return None
    u = next((r["u"] for r in rows if r.get("u") is not None), None)
    if u is None:
        return None
    u = float(u)

    term = _term_points(rows, u, now)
    if not term:
        return None
    atm_iv = float(term[0][1])  # near-tenor ATM
    atm_1m = _interp_atm_at(term, 30.0)
    atm_3m = _interp_atm_at(term, 90.0)
    term_slope = atm_3m - atm_1m

    rr25_1m = _rr25_for_tenor(rows, now, 30.0, delta_fn)
    rr25_3m = _rr25_for_tenor(rows, now, 90.0, delta_fn)

    def _norm(rr: float) -> float:
        return rr / atm_iv if (np.isfinite(rr) and atm_iv) else float("nan")

    snap = pd.DataFrame(
        [
            {
                "ts": int(ts_ms),
                "ccy": ccy,
                "underlying": u,
                "atm_iv": atm_iv,
                "atm_iv_1m": atm_1m,
                "atm_iv_3m": atm_3m,
                "rr25_1m": rr25_1m,
                "rr25_3m": rr25_3m,
                "skew_1m": _norm(rr25_1m),
                "skew_3m": _norm(rr25_3m),
                "term_slope": term_slope,
                "n_options": len(rows),
                "exchange": SOURCE,
            }
        ]
    )
    snap["ts"] = snap["ts"].astype("int64")
    return snap


# --------------------------------------------------------------------------- #
# network helper (used only by run(); pure functions never call it)
# --------------------------------------------------------------------------- #
def _get(client: Any, path: str, **params) -> Any:
    """GET ``API/path`` and return the ``.result`` payload (raises on error)."""
    resp = client.get(f"{API}/{path}", params=params)
    resp.raise_for_status()
    return resp.json()["result"]


def _make_delta_fn(client: Any, *, sleep: Callable[[float], None] | None = None):
    """A ``delta_fn`` that tickers an instrument for its greeks.delta (live)."""

    def _fn(inst: str) -> float | None:
        try:
            tk = _get(client, "ticker", instrument_name=inst)
            g = tk.get("greeks") or {}
            if sleep is not None:
                sleep(0.05)
            return g.get("delta")
        except Exception:
            return None

    return _fn


# --------------------------------------------------------------------------- #
# public entrypoint (LIVE — do NOT execute in tests)
# --------------------------------------------------------------------------- #
def run(raw_root: pathlib.Path, manifest: Manifest, *, run_as_of: str) -> None:
    """Pull Deribit DVOL history + append a live surface snapshot for BTC & ETH.

    Fail-soft per currency and per series: any setup/validation failure becomes a
    ``manifest.blocker`` and the loop continues. ``cached_pull`` already
    fail-softs the fetch itself.

    API keys are not required (public REST); we read os.environ lazily anyway so
    a future authenticated path stays orchestrator-controlled.
    """
    raw_root = pathlib.Path(raw_root)
    _ = os.environ  # public endpoint — no keys needed; touch env at call time only.

    rate = RateLimiter(_RATE_PER_MIN)

    import httpx

    client = httpx.Client(timeout=_HTTP_TIMEOUT)
    try:
        # ---- (A) DVOL history: idempotent, multi-year backfill ---------------- #
        for ccy in CCYS:
            target = raw_root / SOURCE / "dvol" / f"{ccy}.parquet"
            params = {"currency": ccy, "resolution": 3600, "endpoint": "dvol"}

            def _fetch_dvol(_ccy=ccy):
                df = paginate_dvol(_ccy, client=client, sleep=time.sleep)
                if df.empty:
                    return None
                df = df.copy()
                df["symbol"] = _ccy
                df["exchange"] = SOURCE
                df["interval"] = "1h"
                return df

            try:
                cached_pull(
                    target,
                    _fetch_dvol,
                    source=SOURCE,
                    endpoint="get_volatility_index_data",
                    params=params,
                    manifest=manifest,
                    # DVOL is append-only/growing; expected_range=None means
                    # "pull if missing, otherwise skip" — re-runs refresh via a
                    # missing-file or an orchestrator-driven delete, keeping the
                    # multi-year backfill idempotent and cheap.
                    expected_range=None,
                    ts_col="ts",
                    rate=rate,
                )
            except Exception as e:  # setup/validation guard (cached_pull is fail-soft)
                manifest.blocker(
                    source=SOURCE,
                    endpoint="get_volatility_index_data",
                    params=params,
                    error=repr(e),
                    action=f"investigate deribit dvol pull for {ccy}",
                )

        # ---- (B) Live surface snapshot: forward log (append one dated row) ---- #
        now = dt.datetime.now(dt.UTC)
        snap_ms = int(now.timestamp() * 1000)
        delta_fn = _make_delta_fn(client, sleep=time.sleep)
        for ccy in CCYS:
            params = {"currency": ccy, "kind": "option", "endpoint": "surface_snapshot"}
            target = raw_root / SOURCE / "surface_snapshot" / f"{ccy}.parquet"
            try:
                rate.wait()
                summary = _get(client, "get_book_summary_by_currency",
                               currency=ccy, kind="option")
                rows = book_summary_to_rows(summary or [])
                snap = compute_surface_snapshot(
                    ccy, rows, now=now, ts_ms=snap_ms, delta_fn=delta_fn
                )
                if snap is None or snap.empty:
                    manifest.blocker(
                        source=SOURCE,
                        endpoint="get_book_summary_by_currency",
                        params=params,
                        error="empty/unusable option chain",
                        action=f"confirm deribit option chain for {ccy} is live",
                    )
                    continue
                # Forward log: append the dated row, idempotent per UTC day.
                prev = pd.read_parquet(target) if target.exists() else None
                snap = merge_surface_snapshot(prev, snap)
                _write_parquet(snap, target)
                manifest.log(
                    source=SOURCE,
                    endpoint="get_book_summary_by_currency",
                    params=params,
                    status="ok",
                    rows=len(snap),
                    min_ts=int(snap["ts"].min()),
                    max_ts=int(snap["ts"].max()),
                    path=str(target),
                    note="forward-logged surface snapshot (not backfillable)",
                )
            except Exception as e:
                manifest.blocker(
                    source=SOURCE,
                    endpoint="get_book_summary_by_currency",
                    params=params,
                    error=repr(e),
                    action=f"investigate deribit surface snapshot for {ccy}",
                )
    finally:
        client.close()


def merge_surface_snapshot(prev: pd.DataFrame | None, new: pd.DataFrame) -> pd.DataFrame:
    """Append a forward-log surface snapshot, keeping at most ONE row per UTC day
    (the latest). Idempotent: re-running the freeze the same day overwrites that
    day's row instead of duplicating it."""
    snap = new if prev is None or prev.empty else pd.concat([prev, new], ignore_index=True)
    day = pd.to_datetime(snap["ts"], unit="ms").dt.date
    return (
        snap.assign(_d=day)
        .sort_values("ts")
        .drop_duplicates("_d", keep="last")
        .drop(columns="_d")
        .reset_index(drop=True)
    )


def _write_parquet(df: pd.DataFrame, path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
