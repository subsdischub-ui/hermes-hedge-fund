"""Manual gold ingestion: iShares IAU flows + WGC Goldhub series.

These are MANUALLY downloaded inputs (logged-in Goldhub + iShares, 2026-06-26)
that close 2 of the 4 Stream-0 blockers (IAU flows + WGC central-bank demand).
They are ingested into the SAME lake with the SAME core (cached_pull / Manifest /
discover_views) — acquisition only, no research.

Each table is written to its OWN subdirectory under ``data/raw/`` so the
directory-based ``discover_views`` emits exactly one correctly-named view per
table (e.g. ``wgc/cb_demand_gdt/…parquet`` -> view ``wgc_cb_demand_gdt``).

Caveats (from data/manual/wgc_DATA_NOTES.md) are propagated into the manifest:
 * ``cb_demand_gdt`` is the AUTHORITATIVE published headline (reported + estimate
   of unreported buying) — the Stream-6 conditioner.
 * ``cb_changes_monthly`` is REPORTED-ONLY (≪ headline; ~66% recently unreported)
   and contains the documented ``Turkey``/``Turkey*`` twins — never sum naively.
 * ``official_holdings_snapshot`` is a levels SNAPSHOT, not a time series.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd

from hermes.datalake.core import Manifest, cached_pull

GOLDHUB = "manual:goldhub"
ISHARES = "manual:ishares"
INGEST_METHOD = "manual download (logged-in), 2026-06-26"

_NUM = "coerce"


def _stamp(df: pd.DataFrame, *, source: str) -> pd.DataFrame:
    df["source"] = source
    df["ingest_method"] = INGEST_METHOD
    return df


# --------------------------------------------------------------------------- #
# pure normalizers (clean tidy CSV -> typed frame + provenance)
# --------------------------------------------------------------------------- #
def normalize_iau(df: pd.DataFrame, *, run_as_of: str) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    for c in ["nav_per_share", "shares_outstanding", "aum_usd", "flow_shares", "flow_usd"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors=_NUM)
    df["ts"] = df["date"].astype("int64") // 1_000_000  # ns -> epoch ms
    df["symbol"] = "IAU"
    return _stamp(df, source=ISHARES)


def normalize_cb_demand_gdt(df: pd.DataFrame, *, run_as_of: str) -> pd.DataFrame:
    df = df.copy()
    df["period_type"] = df["period_type"].astype(str)
    df["year"] = pd.to_numeric(df["year"], errors=_NUM).astype("int64")
    df["quarter"] = pd.to_numeric(df["quarter"], errors=_NUM)  # NaN for annual rows
    df["tonnes"] = pd.to_numeric(df["tonnes"], errors=_NUM)
    return _stamp(df, source=GOLDHUB)


def normalize_gdt_balance(df: pd.DataFrame, *, run_as_of: str) -> pd.DataFrame:
    df = df.copy()
    df["line_item"] = df["line_item"].astype(str)
    df["period"] = df["period"].astype(str)  # mixed years/quarters/"YoY % change"
    df["tonnes"] = pd.to_numeric(df["tonnes"], errors=_NUM)
    return _stamp(df, source=GOLDHUB)


def normalize_cb_changes_monthly(df: pd.DataFrame, *, run_as_of: str) -> pd.DataFrame:
    df = df.copy()
    df["month"] = pd.to_datetime(df["month"])
    df["country"] = df["country"].astype(str)
    df["net_tonnes"] = pd.to_numeric(df["net_tonnes"], errors=_NUM)
    return _stamp(df, source=GOLDHUB)


def normalize_cb_changes_annual(df: pd.DataFrame, *, run_as_of: str) -> pd.DataFrame:
    df = df.copy()
    df["year"] = pd.to_numeric(df["year"], errors=_NUM).astype("int64")
    df["country"] = df["country"].astype(str)
    df["net_tonnes"] = pd.to_numeric(df["net_tonnes"], errors=_NUM)
    return _stamp(df, source=GOLDHUB)


def normalize_cb_reported_proxy(df: pd.DataFrame, *, run_as_of: str) -> pd.DataFrame:
    df = df.copy()
    df["month"] = pd.to_datetime(df["month"])
    df["reported_net_tonnes_proxy"] = pd.to_numeric(df["reported_net_tonnes_proxy"], errors=_NUM)
    return _stamp(df, source=GOLDHUB)


def normalize_official_holdings(df: pd.DataFrame, *, run_as_of: str) -> pd.DataFrame:
    df = df.copy()
    df["rank"] = pd.to_numeric(df["rank"], errors=_NUM)  # NaN for aggregate rows
    df["country"] = df["country"].astype(str)
    df["tonnes"] = pd.to_numeric(df["tonnes"], errors=_NUM)
    df["pct_of_reserves"] = pd.to_numeric(df["pct_of_reserves"], errors=_NUM)
    # as_of is MIXED-format: most rows "YYYY-MM-DD HH:MM:SS", but some countries
    # report less often and carry a "Mon YYYY" latest-available stamp.
    df["as_of"] = pd.to_datetime(df["as_of"], format="mixed", errors="coerce")
    return _stamp(df, source=GOLDHUB)


# --------------------------------------------------------------------------- #
# ingestion plan: (csv, out-subdir == view name, parquet, normalizer, ts_col, caveat)
# --------------------------------------------------------------------------- #
_SPECS = [
    dict(csv="iau_flows.csv", subdir="etf/iau_flows", out="iau_flows.parquet",
         norm=normalize_iau, source=ISHARES, ts_col="ts",
         caveat="IAU daily flows = delta shares x NAV (manual iShares download)"),
    dict(csv="wgc_gdt_central_bank_demand.csv", subdir="wgc/cb_demand_gdt",
         out="cb_demand_gdt.parquet", norm=normalize_cb_demand_gdt, source=GOLDHUB, ts_col=None,
         caveat="AUTHORITATIVE published headline = reported + estimate of unreported; "
                "Q1'26=243.7t, 2024=1092t. Stream-6 CB-demand conditioner."),
    dict(csv="wgc_gdt_supply_demand_balance.csv", subdir="wgc/gdt_balance",
         out="gdt_supply_demand_balance.parquet", norm=normalize_gdt_balance, source=GOLDHUB, ts_col=None,
         caveat="full GDT supply/demand (long); incl 'ETFs and Similar Products' line "
                "(complements GLD/IAU flow) + Central Bank line. period is a string (years+quarters)."),
    dict(csv="wgc_cb_net_changes_monthly_by_country.csv", subdir="wgc/cb_changes_monthly",
         out="cb_changes_monthly_by_country.parquet", norm=normalize_cb_changes_monthly,
         source=GOLDHUB, ts_col=None,
         caveat="REPORTED ONLY (much lower than cb_demand_gdt; ~66% recently unreported). "
                "Turkey/Turkey* are TWINS — do NOT sum naively. SOFAZ/Jordan* special; Euro Area (EA)=ECB."),
    dict(csv="wgc_cb_net_changes_annual_by_country.csv", subdir="wgc/cb_changes_annual",
         out="cb_changes_annual_by_country.parquet", norm=normalize_cb_changes_annual,
         source=GOLDHUB, ts_col=None,
         caveat="REPORTED ONLY, annual by country (same twin/special-entity caveats as monthly)."),
    dict(csv="wgc_cb_reported_changes_monthly_global_PROXY.csv",
         subdir="wgc/cb_reported_monthly_global_proxy", out="cb_reported_monthly_global_proxy.parquet",
         norm=normalize_cb_reported_proxy, source=GOLDHUB, ts_col=None,
         caveat="PROXY: monthly global = sum of by-country after deduping starred twins + excl Euro Area. "
                "Reported-basis, NOT the headline — for the headline use cb_demand_gdt."),
    dict(csv="wgc_official_holdings_snapshot.csv", subdir="wgc/official_holdings_snapshot",
         out="official_holdings_snapshot.parquet", norm=normalize_official_holdings,
         source=GOLDHUB, ts_col=None,
         caveat="SNAPSHOT of levels (~Mar/Apr 2026), NOT a time series. Aggregate rows "
                "('World6)', 'Euro Area (incl. ECB)', IMF/BIS) have null rank."),
]

NOTES_FILE = "wgc_DATA_NOTES.md"


def run(raw_root: Path, manifest: Manifest, *, run_as_of: str, manual_root: Path | None = None) -> None:
    """Ingest the manually-downloaded IAU + WGC files from ``manual_root``
    (default ``<raw_root>/../manual``) into the lake. Idempotent (cached_pull
    skips existing); fail-soft per file (a missing/unreadable file is one blocker,
    the rest continue)."""
    raw_root = Path(raw_root)
    manual_root = Path(manual_root) if manual_root is not None else raw_root.parent / "manual"

    for spec in _SPECS:
        src = manual_root / spec["csv"]
        target = raw_root / spec["subdir"] / spec["out"]
        if not src.exists():
            manifest.blocker(
                source=spec["source"], endpoint=spec["csv"], params={"manual_root": str(manual_root)},
                error=f"manual file not found: {src}",
                action=f"drop {spec['csv']} into {manual_root} (manual Goldhub/iShares download) and re-run",
            )
            continue

        def _fetch(s=src, nf=spec["norm"]):
            return nf(pd.read_csv(s), run_as_of=run_as_of)

        cached_pull(
            target, _fetch,
            source=spec["source"], endpoint=spec["csv"],
            params={"method": INGEST_METHOD, "caveat": spec["caveat"]},
            manifest=manifest, ts_col=spec["ts_col"],
        )

    # provenance notes — copied verbatim, NOT parquetized
    notes_src = manual_root / NOTES_FILE
    if notes_src.exists():
        dst = raw_root / "wgc" / NOTES_FILE
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(notes_src, dst)
        manifest.log(
            source=GOLDHUB, endpoint=NOTES_FILE, params="", status="ok",
            path=str(dst), note="WGC provenance notes (copied, not parquetized)",
        )
