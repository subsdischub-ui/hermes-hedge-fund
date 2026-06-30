"""Unified forward-collection cron: one idempotent, fail-soft job, many flux.

Run daily (`python -m hermes.forward.run`). Each flux is read-only on its feed and
write-only on ``<data_dir>/forward/<flux>/``; a failure in one flux is recorded and
never aborts the others; a same-day re-run overwrites that day's rows (idempotent).

Flux:
  * skew   — Deribit 25Δ skew (reuses the S4 surface-snapshot transform);
  * comex  — COMEX dated-contract curve → PIT calendar spreads;
  * macro  — SEP dot-plot vs Fed-Funds-futures gap.
Shadow size-0 sleeves:
  * funding_squeeze_shadow — the LIVE edge, always accrues (clean bear/range OOS);
  * xs_reversion_shadow    — GATED OFF by default (fragile in-sample; one flag to wire).

This module NEVER touches the live trading runtime — wiring a confirmed sleeve into
market_analyst/pulse/allowed_skills is the human merge gate (see FINDINGS).
"""

from __future__ import annotations

import datetime as dt
import json
import time

import pandas as pd

from hermes.forward import (
    collect_comex,
    collect_macro,
    collect_skew,
    logio,
)
from hermes.forward import (
    shadow_funding_squeeze as sfs,
)
from hermes.forward import (
    shadow_xs_reversion as xsr,
)

# Fragile in-sample (decays to negative in the recent regime; parameter-sensitive;
# ETH-concentrated). Per the mandate's sanity gate: documented, NOT wired. Flip to
# True only if the "forward-is-the-only-clean-test" argument is adopted at merge.
WIRE_XS_REVERSION = False


def default_collector_names(*, wire_xs_reversion: bool = WIRE_XS_REVERSION) -> list[str]:
    names = ["skew", "comex", "macro", "funding_squeeze_shadow"]
    if wire_xs_reversion:
        names.append("xs_reversion_shadow")
    return names


# ── live read-only fetchers (thin seams; fail-soft when network/creds absent) ───── #
def fetch_live_frames(universe=sfs.LIVE_UNIVERSE, *, limit=400):  # pragma: no cover - network
    """Recent 4h OHLCV + causal as-of funding per symbol from Binance USDT-M (public)."""
    import ccxt

    from hermes.research.funding import attach_funding

    ex = ccxt.binanceusdm({"enableRateLimit": True})
    frames = {}
    for sym in universe:
        ohlcv = ex.fetch_ohlcv(sym, timeframe="4h", limit=limit)
        df = pd.DataFrame(ohlcv, columns=["ts", "open", "high", "low", "close", "volume"])
        hist = ex.fetch_funding_rate_history(sym, limit=1000)
        fdf = pd.DataFrame(
            [{"ts": r["timestamp"], "funding_rate": r["fundingRate"]} for r in hist]
        )
        frames[sym] = attach_funding(df, fdf) if len(fdf) else df.assign(funding_rate=0.0)
    return frames


# ── collectors (each returns {logs:[(name, df, key_cols, ts_col)], blockers:{}}) ── #
def _skew_collector(now_ms):  # pragma: no cover - network seam (reuses S4 public fetch)
    def c():
        import time as _t

        import httpx

        from hermes.datalake.sources.deribit import _get, _make_delta_fn

        now = dt.datetime.fromtimestamp(now_ms / 1000, tz=dt.UTC)
        client = httpx.Client(timeout=30.0)
        try:
            delta_fn = _make_delta_fn(client, sleep=_t.sleep)
            res = collect_skew.collect(
                fetch_book=lambda ccy: _get(
                    client, "get_book_summary_by_currency", currency=ccy, kind="option"
                ),
                delta_fn_factory=lambda ccy: delta_fn,
                now=now,
                now_ms=now_ms,
            )
        finally:
            client.close()
        return {"logs": [("skew", res["skew"], ("ccy",), "ts")], "blockers": res["blockers"]}

    c.flux_name = "skew"
    return c


def _comex_collector(now_ms):
    def c():
        res = collect_comex.collect(now_ms=now_ms)  # default fetcher is the unwired seam
        return {
            "logs": [
                ("comex_curve", res["curve"], ("metal", "contract"), "ts"),
                ("comex_spreads", res["spreads"], ("metal", "near_contract", "far_contract"), "ts"),
            ],
            "blockers": res["blockers"],
        }

    c.flux_name = "comex"
    return c


def _macro_collector(now_ms):
    def c():
        res = collect_macro.collect(now_ms=now_ms)
        return {"logs": [("macro_dot_vs_ff", res["gap"], ("horizon",), "ts")], "blockers": res["blockers"]}

    c.flux_name = "macro"
    return c


def _funding_squeeze_collector(now_ms):
    def c():
        frames = fetch_live_frames()
        trades = sfs.shadow_step(frames)
        return {"logs": [(sfs.LOG_NAME, trades, sfs.KEY_COLS, sfs.TS_COL)], "blockers": {}}

    c.flux_name = "funding_squeeze_shadow"
    return c


def _xs_reversion_collector(now_ms):
    def c():
        frames = fetch_live_frames(universe=xsr.UNIVERSE_23)
        panel = _frames_to_daily_panel(frames)
        book = xsr.xs_reversion_book(panel)
        return {"logs": [(xsr.LOG_NAME, book, xsr.KEY_COLS, xsr.TS_COL)], "blockers": {}}

    c.flux_name = "xs_reversion_shadow"
    return c


def _frames_to_daily_panel(frames):  # pragma: no cover - exercised only when wired
    """4h OHLCV+funding frames → daily (ts, close, funding_z) panel for XS reversion."""
    panel = {}
    for sym, df in frames.items():
        if "funding_rate" not in df.columns or df.empty:
            continue
        z = xsr.causal_funding_z(df["funding_rate"])
        d = df.assign(funding_z=z)
        d["date"] = pd.to_datetime(d["ts"], unit="ms", utc=True).dt.floor("D")
        daily = d.dropna(subset=["funding_z"]).groupby("date").last()[["close", "funding_z"]]
        if daily.empty:
            continue
        daily = daily.reset_index()
        daily["ts"] = daily["date"].astype("int64") // 10**6
        panel[sym] = daily[["ts", "close", "funding_z"]].reset_index(drop=True)
    return panel


def _raise_unwired(what):  # pragma: no cover
    def _f(*a, **k):
        raise NotImplementedError(f"wire the read-only fetcher for {what}")
    return _f


def _default_collectors(now_ms, *, wire_xs_reversion: bool):
    built = {
        "skew": _skew_collector(now_ms),
        "comex": _comex_collector(now_ms),
        "macro": _macro_collector(now_ms),
        "funding_squeeze_shadow": _funding_squeeze_collector(now_ms),
        "xs_reversion_shadow": _xs_reversion_collector(now_ms),
    }
    return [built[n] for n in default_collector_names(wire_xs_reversion=wire_xs_reversion)]


def run_forward(
    *,
    data_dir=None,
    now_ms: int | None = None,
    collectors=None,
    wire_xs_reversion: bool = WIRE_XS_REVERSION,
) -> dict:
    """Run every collector once, fail-soft, idempotent. Returns a status manifest."""
    if now_ms is None:
        now_ms = int(time.time() * 1000)
    if collectors is None:
        collectors = _default_collectors(now_ms, wire_xs_reversion=wire_xs_reversion)
    manifest = {"now_ms": now_ms, "flux": {}}
    for coll in collectors:
        name = getattr(coll, "flux_name", getattr(coll, "__name__", "unknown"))
        try:
            res = coll()
            written = 0
            for log_name, df, key_cols, ts_col in res.get("logs", []):
                if df is not None and len(df):
                    logio.append_snapshot(
                        log_name, df, key_cols=key_cols, ts_col=ts_col, data_dir=data_dir
                    )
                    written += len(df)
            blockers = res.get("blockers", {})
            manifest["flux"][name] = {
                "status": "ok" if not blockers else "partial",
                "rows_written": written,
                "blockers": blockers,
            }
        except Exception as e:  # fail-soft: one flux's failure never aborts the cron
            manifest["flux"][name] = {
                "status": "error",
                "rows_written": 0,
                "blockers": {name: f"{type(e).__name__}: {e}"},
            }
    return manifest


def main():  # pragma: no cover - CLI entry
    import argparse

    p = argparse.ArgumentParser(description="Unified forward-collection cron (read-only feeds).")
    p.add_argument("--data-dir", default="data", help="root for <data_dir>/forward/ logs")
    p.add_argument("--wire-xs-reversion", action="store_true", help="activate the fragile XS sleeve")
    args = p.parse_args()
    from hermes.forward import prereg

    # NOTE: deliberately do NOT pass --data-dir here. That flag is the root for THIS
    # deployment's forward-collection logs (<data_dir>/forward/<flux>/); the trial
    # registry is a SEPARATE, deployment-scoped concern that resolves on its own via
    # governance_registry_dir() (HERMES_GOVERNANCE_DIR env var, or the committed repo
    # dir in local/dev). Passing data_dir through here previously made the registry
    # silently fork onto an ad-hoc path instead of its one persistent, intentional home.
    trials = prereg.register_trials()  # idempotent pre-reg bump
    manifest = run_forward(data_dir=args.data_dir, wire_xs_reversion=args.wire_xs_reversion)
    manifest["trial_registry"] = trials
    print(json.dumps(manifest, indent=2, default=str))


if __name__ == "__main__":  # pragma: no cover
    main()
