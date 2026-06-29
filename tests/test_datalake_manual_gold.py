"""Manual gold ingestion (iShares IAU flows + WGC Goldhub) — offline normalizer
tests. The source CSVs are already clean/tidy, so per the addendum these are
light: expected columns + non-empty + one known-value/dtype assertion each.
The heavier 'against reality' checks run as live spot-checks on the ingested
Parquet (addendum §4), not here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from hermes.datalake.sources.manual_gold import (
    GOLDHUB,
    ISHARES,
    normalize_cb_changes_annual,
    normalize_cb_changes_monthly,
    normalize_cb_demand_gdt,
    normalize_cb_reported_proxy,
    normalize_gdt_balance,
    normalize_iau,
    normalize_official_holdings,
)

RA = "2026-06-26T00:00:00Z"


def test_normalize_iau_dtypes_provenance_and_blank_flows():
    raw = pd.DataFrame({
        "date": ["2005-01-21", "2005-01-24", "2026-06-25"],
        "nav_per_share": [8.534, 8.537718, 75.268114],
        "shares_outstanding": [750000.0, 750000.0, 797150000.0],
        "aum_usd": [6400500.0, 6403288.5, 59999977075.1],
        "flow_shares": [np.nan, 0.0, -4900000.0],   # first day has no prior -> blank
        "flow_usd": [np.nan, 0.0, -368813758.6],
    })
    df = normalize_iau(raw, run_as_of=RA)
    assert {"date", "ts", "nav_per_share", "shares_outstanding", "aum_usd",
            "flow_shares", "flow_usd", "symbol", "source"} <= set(df.columns)
    assert len(df) == 3
    assert df["date"].dtype == np.dtype("datetime64[ns]")
    assert df["ts"].dtype == np.int64
    assert (df["symbol"] == "IAU").all()
    assert (df["source"] == ISHARES).all()
    # latest shares_outstanding ~ 797M (spot-check anchor)
    assert abs(df["shares_outstanding"].iloc[-1] - 797_150_000) < 1


def test_normalize_cb_demand_gdt_authoritative_series():
    raw = pd.DataFrame({
        "period_type": ["annual", "annual", "quarter"],
        "year": [2022, 2024, 2026],
        "quarter": [np.nan, np.nan, 1.0],
        "tonnes": [1080.0, 1092.0, 243.66413452],
    })
    df = normalize_cb_demand_gdt(raw, run_as_of=RA)
    assert {"period_type", "year", "quarter", "tonnes", "source"} <= set(df.columns)
    assert df["tonnes"].dtype == np.float64
    assert (df["source"] == GOLDHUB).all()
    # Q1'26 authoritative figure ~ 243.7t
    q = df[(df["period_type"] == "quarter") & (df["year"] == 2026)]
    assert abs(q["tonnes"].iloc[0] - 243.7) < 0.5


def test_normalize_gdt_balance_keeps_string_period_and_etf_line():
    raw = pd.DataFrame({
        "line_item": ["ETFs and Similar Products", "Central Bank and Other Institutions"],
        "period": ["Q1'26", "Q1'26"],
        "tonnes": [62.0, 243.7],
    })
    df = normalize_gdt_balance(raw, run_as_of=RA)
    assert {"line_item", "period", "tonnes", "source"} <= set(df.columns)
    assert df["period"].dtype == object or str(df["period"].dtype).startswith("string")
    assert "ETFs and Similar Products" in set(df["line_item"])


def test_normalize_cb_changes_monthly_month_datetime_and_twins_preserved():
    raw = pd.DataFrame({
        "month": ["2002-01-01", "2025-01-01", "2025-01-01"],
        "country": ["China, P.R.: Mainland", "Turkey", "Turkey*"],
        "net_tonnes": [0.0, 5.0, 7.0],
    })
    df = normalize_cb_changes_monthly(raw, run_as_of=RA)
    assert df["month"].dtype == np.dtype("datetime64[ns]")
    assert df["net_tonnes"].dtype == np.float64
    # the documented Turkey/Turkey* twins must BOTH survive (no naive merge)
    assert {"Turkey", "Turkey*"} <= set(df["country"])


def test_normalize_cb_changes_annual_year_int():
    raw = pd.DataFrame({"year": [2002, 2025], "country": ["Poland", "Poland"],
                        "net_tonnes": [0.0, 90.0]})
    df = normalize_cb_changes_annual(raw, run_as_of=RA)
    assert {"year", "country", "net_tonnes", "source"} <= set(df.columns)
    assert str(df["year"].dtype).startswith("int")


def test_normalize_cb_reported_proxy_month_and_value():
    raw = pd.DataFrame({"month": ["2002-01-01", "2026-04-01"],
                        "reported_net_tonnes_proxy": [-50.6, 56.72]})
    df = normalize_cb_reported_proxy(raw, run_as_of=RA)
    assert df["month"].dtype == np.dtype("datetime64[ns]")
    assert df["reported_net_tonnes_proxy"].dtype == np.float64
    assert len(df) == 2


def test_normalize_official_holdings_snapshot_nullable_rank():
    raw = pd.DataFrame({
        "rank": [1.0, 60.0, np.nan],
        "country": ["United States", "WAEMU3)", "World6)"],
        "tonnes": [8133.46, 36.52, 36558.54],
        "pct_of_reserves": [0.833, 0.208, 0.277],
        # MIXED as_of formats: full timestamp + month-year (some countries report less often)
        "as_of": ["2026-03-31 00:00:00", "Mar 2025", "2026-03-31 00:00:00"],
    })
    df = normalize_official_holdings(raw, run_as_of=RA)
    assert {"rank", "country", "tonnes", "pct_of_reserves", "as_of", "source"} <= set(df.columns)
    assert df["as_of"].dtype == np.dtype("datetime64[ns]")
    assert df["as_of"].notna().all()  # both "Mar 2025" and the full timestamp parse
    # US rank-1 ~ 8133t; aggregate row has NaN rank but is kept
    us = df[df["country"] == "United States"]
    assert abs(us["tonnes"].iloc[0] - 8133) < 5
    assert df["rank"].isna().any()  # aggregate rows kept with null rank
