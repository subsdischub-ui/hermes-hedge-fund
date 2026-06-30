"""Stream-0 orchestrator: freeze the Hermès point-in-time data lake.

Sequence (§8): one source at a time (so CoinGlass never exceeds 30 req/min and
sources don't contend), each fail-soft, then build DuckDB views over the frozen
Parquet tree and write the provenance manifests.

    python -m hermes.datalake.run                 # full freeze, all sources
    python -m hermes.datalake.run --sources fred,stablecoins   # subset / resume
    python -m hermes.datalake.run --list          # show source order and stop

Acquisition only — no research, no backtests. Re-runnable: cached_pull skips any
Parquet that already covers its range, so a crashed run resumes instead of
re-pulling.
"""

from __future__ import annotations

import argparse
import datetime as dt
import importlib
import sys
from pathlib import Path

from hermes.datalake.core import Manifest, build_views, discover_views, write_manifest

REPO_ROOT = Path(__file__).resolve().parents[3]
RAW_ROOT = REPO_ROOT / "data" / "raw"
MANIFEST_DIR = REPO_ROOT / "data" / "manifest"
DUCKDB_PATH = REPO_ROOT / "data" / "hermes_datalake.duckdb"

# §8 order. ForexFactory (B4) is deliberately absent — it has no freezable
# history; it is recorded as a dead source in the manifest (see _record_dead).
SOURCE_ORDER = [
    "coinglass",     # rate-limited centerpiece (deep OI/funding/LSR/options) — first
    "coinglass_multi",   # cross-venue funding+OI (OKX/Bybit/Bitget+Binance) — S0 re-freeze scout
    "binance",       # perp klines 1d/6h + deep funding
    "binance_extended",  # 4h klines (max depth) + deep native funding + extended universe (S0 re-freeze)
    "binance_spot",  # SPOT 4h klines (cash-and-carry spot leg) — Stream 0-α
    "hyperliquid",   # HL hourly funding + 4h klines (cross-venue carry, forward axis) — Stream 0-α
    "binance_tail",  # USDT-M illiquid TAIL (perps beyond 23, stratified) — Stream 0-β
    "hyperliquid_exotic",  # HL exotic markets (beyond 23, stratified + ctx snapshot) — Stream 0-β
    "deribit",       # DVOL history (backfill) + forward surface snapshot
    "defillama",     # stablecoin total/USDT/USDC supply
    "fred",          # macro: DFII10/DGS10/T10Y2Y/BAMLH0A0HYM2/DTWEXBGS
    "manual_gold",   # manual Goldhub (WGC CB demand) + iShares IAU flows (before gold)
    "gold",          # XAU/XAG/FX/miners/DXY prices + GLD flows (IAU/WGC now via manual_gold)
    "cbspeak",       # BIS bulk corpus + FOMC statements (corpus only, no scoring)
]


def _load_env() -> None:
    """Load .env into os.environ if python-dotenv is available (keys for CoinGlass/FRED)."""
    try:
        from dotenv import load_dotenv

        load_dotenv(REPO_ROOT / ".env")
    except Exception:  # dotenv optional; keys may already be exported
        pass


def _record_dead_forexfactory(manifest: Manifest) -> None:
    manifest.blocker(
        source="forexfactory",
        endpoint="ff_calendar (faireconomy mirror / forexfactory.com)",
        params="thisweek.json",
        error="no freezable history: mirror serves current-week only and carries NO 'actual' "
        "field (surprise=actual-forecast uncomputable); range siblings 404; official site Cloudflare-403",
        action="B4 is a SETTLED DEAD source (LAB_JOURNAL + memory). To get a macro-surprise feed "
        "either (a) stand up a forward-accruing cron that snapshots ff_calendar_thisweek.json weekly "
        "AND re-pulls post-release to capture 'actual' (zero backfill), or (b) buy a calendar-history "
        "vendor (Econoday/Haver/TradingEconomics). ALFRED backfills actuals-as-vintages but has no "
        "consensus forecast, so it alone cannot reconstruct a surprise.",
    )


def _lake_size_bytes(root: Path) -> int:
    return sum(p.stat().st_size for p in root.rglob("*") if p.is_file()) if root.exists() else 0


def run_lake(sources: list[str]) -> Manifest:
    _load_env()
    run_as_of = dt.datetime.now(dt.UTC).isoformat()
    RAW_ROOT.mkdir(parents=True, exist_ok=True)
    manifest = Manifest(run_as_of=run_as_of)
    print(f"[lake] run_as_of={run_as_of}  raw_root={RAW_ROOT}", flush=True)

    for name in sources:
        print(f"\n[lake] === source: {name} ===", flush=True)
        try:
            mod = importlib.import_module(f"hermes.datalake.sources.{name}")
            mod.run(RAW_ROOT, manifest, run_as_of=run_as_of)
        except Exception as e:  # defense-in-depth; per-pull errors are already fail-soft
            manifest.blocker(
                source=name, endpoint="run()", params="",
                error=repr(e),
                action=f"source module {name}.run() raised at the top level — inspect; "
                f"other sources were unaffected (fail-soft).",
            )
            print(f"[lake] !! {name} top-level error: {e!r}", flush=True)

    # B4 / ForexFactory: explicitly recorded as dead (no data materialized).
    if "forexfactory" not in sources:
        _record_dead_forexfactory(manifest)

    # DuckDB views over whatever Parquet landed.
    try:
        views = discover_views(RAW_ROOT)
        if views:
            build_views(DUCKDB_PATH, views)
            print(f"\n[lake] built {len(views)} DuckDB views -> {DUCKDB_PATH}", flush=True)
    except Exception as e:
        manifest.blocker(source="duckdb", endpoint="build_views", params="", error=repr(e),
                         action="DuckDB view build failed — Parquet is intact; rebuild with build_views().")

    write_manifest(manifest, MANIFEST_DIR)
    _print_summary(manifest)
    return manifest


def _print_summary(manifest: Manifest) -> None:
    from hermes.datalake.core import summarize_manifest

    s = summarize_manifest(manifest.entries)
    size_mb = _lake_size_bytes(RAW_ROOT) / 1e6
    print("\n" + "=" * 64, flush=True)
    print("STREAM 0 — DATA LAKE: FINAL SUMMARY", flush=True)
    print("=" * 64, flush=True)
    print(f"  ✅ ok={s['ok']}  ⏭️ skip={s['skip']}  ⚠️ partial={s['partial']}  ❌ blocked={s['blocked']}", flush=True)
    print(f"  total rows pulled : {s['total_rows']:,}", flush=True)
    print(f"  lake size         : {size_mb:,.1f} MB  ({RAW_ROOT})", flush=True)
    print(f"  manifests         : {MANIFEST_DIR}", flush=True)
    print(f"  duckdb            : {DUCKDB_PATH}", flush=True)
    if manifest.blockers:
        print(f"\n  ⚠️  {len(manifest.blockers)} BLOCKER(S) need Malachie — see {MANIFEST_DIR / 'BLOCKERS.md'}", flush=True)
        for b in manifest.blockers:
            print(f"     - {b['source']}: {b['error'][:90]}", flush=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Freeze the Hermès Stream-0 data lake.")
    ap.add_argument("--sources", default=",".join(SOURCE_ORDER),
                    help="comma-separated subset of: " + ",".join(SOURCE_ORDER))
    ap.add_argument("--list", action="store_true", help="print source order and exit")
    args = ap.parse_args(argv)

    if args.list:
        print("source order:", ", ".join(SOURCE_ORDER))
        print("(forexfactory = dead, recorded as a blocker, no data)")
        return 0

    requested = [s.strip() for s in args.sources.split(",") if s.strip()]
    unknown = [s for s in requested if s not in SOURCE_ORDER]
    if unknown:
        print(f"unknown sources: {unknown}; valid: {SOURCE_ORDER}", file=sys.stderr)
        return 2
    ordered = [s for s in SOURCE_ORDER if s in requested]
    run_lake(ordered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
