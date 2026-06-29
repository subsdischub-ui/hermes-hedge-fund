"""Offline tests for the DeFiLlama stablecoin-supply lake client.

NO live network. We feed synthetic /stablecoincharts/all and /stablecoin/{id}
payloads to the PURE transforms, and inject a fake httpx transport for the
fetch wrappers. We assert:

  * the 5-column schema [date, total_supply, usdt_supply, usdc_supply, net_change],
  * the str/int unix-second date convention is normalised identically,
  * ascending sort + net_change == total_supply.diff(),
  * total uses totalCirculating.peggedUSD (NOT totalCirculatingUSD),
  * a missing optional key degrades to NaN (no crash),
  * the ts-millis time-series convention on the parquet rows,
  * fail-soft: a broken transport never raises out of run().
"""

from __future__ import annotations

import math

import pandas as pd

from hermes.datalake.core import Manifest, read_parquet
from hermes.datalake.sources.defillama import (
    build_stablecoin_supply,
    parse_per_coin,
    parse_total,
    run,
)


# --------------------------------------------------------------------------- #
# synthetic payloads
# --------------------------------------------------------------------------- #
def _charts_all() -> list[dict]:
    """/stablecoincharts/all — date as STRING unix-seconds. totalCirculating is
    the USD-pegged total we WANT; totalCirculatingUSD folds non-USD pegs and must
    be ignored. Deliberately UNSORTED to prove the transform sorts ascending."""
    day = 86400
    base = 1_600_000_000
    rows = [
        {"date": str(base + 2 * day),
         "totalCirculating": {"peggedUSD": 1.30e11},
         "totalCirculatingUSD": {"peggedUSD": 9.99e11}},
        {"date": str(base + 0 * day),
         "totalCirculating": {"peggedUSD": 1.00e11},
         "totalCirculatingUSD": {"peggedUSD": 9.99e11}},
        {"date": str(base + 1 * day),
         "totalCirculating": {"peggedUSD": 1.15e11},
         "totalCirculatingUSD": {"peggedUSD": 9.99e11}},
    ]
    return rows


def _per_coin(circ: list[float], *, drop_last_key: bool = False) -> dict:
    """/stablecoin/{id} — date as INT unix-seconds, circulating.peggedUSD per day."""
    day = 86400
    base = 1_600_000_000
    tokens = []
    for i, c in enumerate(circ):
        if drop_last_key and i == len(circ) - 1:
            tokens.append({"date": base + i * day, "circulating": {}})  # missing peggedUSD
        else:
            tokens.append({"date": base + i * day, "circulating": {"peggedUSD": c}})
    return {"tokens": tokens}


# --------------------------------------------------------------------------- #
# pure-transform tests
# --------------------------------------------------------------------------- #
def test_parse_total_uses_pegged_usd_and_casts_str_date():
    df = parse_total(_charts_all())
    assert list(df.columns) == ["date", "total_supply"]
    # cast from STRING unix-seconds -> datetime
    assert pd.api.types.is_datetime64_any_dtype(df["date"])
    # ascending sort applied
    assert df["date"].is_monotonic_increasing
    # uses totalCirculating.peggedUSD, NOT totalCirculatingUSD (9.99e11)
    assert df["total_supply"].tolist() == [1.00e11, 1.15e11, 1.30e11]


def test_parse_per_coin_casts_int_date():
    df = parse_per_coin(_per_coin([1.0, 2.0, 3.0]), "usdt_supply")
    assert list(df.columns) == ["date", "usdt_supply"]
    assert pd.api.types.is_datetime64_any_dtype(df["date"])
    assert df["usdt_supply"].tolist() == [1.0, 2.0, 3.0]


def test_parse_per_coin_missing_key_becomes_nan():
    df = parse_per_coin(_per_coin([1.0, 2.0, 3.0], drop_last_key=True), "usdc_supply")
    assert len(df) == 3
    assert math.isnan(df["usdc_supply"].iloc[-1])


def test_build_merge_schema_sort_and_net_change():
    total = parse_total(_charts_all())
    usdt = parse_per_coin(_per_coin([5.0, 6.0, 7.0]), "usdt_supply")
    usdc = parse_per_coin(_per_coin([1.0, 2.0, 3.0]), "usdc_supply")
    out = build_stablecoin_supply(total, usdt, usdc)

    # exact 5-column schema, exact order
    assert list(out.columns) == [
        "date", "total_supply", "usdt_supply", "usdc_supply", "net_change",
    ]
    # ascending
    assert out["date"].is_monotonic_increasing
    # net_change == total_supply.diff()
    expected = out["total_supply"].diff()
    pd.testing.assert_series_equal(
        out["net_change"].reset_index(drop=True),
        expected.reset_index(drop=True),
        check_names=False,
    )
    # first net_change is NaN (diff of first row)
    assert math.isnan(out["net_change"].iloc[0])
    # the three feeds aligned on the same dates -> 3 rows
    assert len(out) == 3


def test_build_outer_merge_unions_disjoint_dates():
    """Outer merge: a coin missing a tail date still yields a union row (NaN there)."""
    total = parse_total(_charts_all())          # 3 dates
    usdt = parse_per_coin(_per_coin([5.0, 6.0]), "usdt_supply")  # only first 2 dates
    usdc = parse_per_coin(_per_coin([1.0, 2.0, 3.0]), "usdc_supply")
    out = build_stablecoin_supply(total, usdt, usdc)
    assert len(out) == 3
    assert math.isnan(out["usdt_supply"].iloc[-1])  # union row, usdt absent
    assert out["usdc_supply"].iloc[-1] == 3.0


# --------------------------------------------------------------------------- #
# fake-transport / run() tests (offline httpx)
# --------------------------------------------------------------------------- #
class _Resp:
    def __init__(self, payload):
        self._p = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._p


class _FakeClient:
    """Routes by URL: /stablecoincharts/all -> charts; /stablecoin/{id} -> per-coin."""

    def __init__(self, *a, **k):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def get(self, url, params=None, headers=None):
        if "stablecoincharts" in url:
            return _Resp(_charts_all())
        sid = int(url.rstrip("/").split("/")[-1])
        circ = [1.0e10, 1.1e10, 1.2e10] if sid == 1 else [5.0e9, 5.5e9, 6.0e9]
        return _Resp(_per_coin(circ))


class _BoomClient:
    def __init__(self, *a, **k):
        pass

    def __enter__(self):
        raise RuntimeError("network down")

    def __exit__(self, *a):
        return False


def test_run_writes_parquet_with_ms_ts(tmp_path):
    mani = Manifest(run_as_of="2026-06-26")
    run(tmp_path, mani, run_as_of="2026-06-26",
        client_factory=lambda: _FakeClient(),
        legacy_csv_path=tmp_path / "legacy.csv")

    target = tmp_path / "defillama" / "stablecoin_supply" / "stablecoin_supply.parquet"
    assert target.exists()
    df = read_parquet(target)
    # 5 logical columns present
    for col in ["date", "total_supply", "usdt_supply", "usdc_supply", "net_change"]:
        assert col in df.columns
    # time-series schema: ts = epoch MILLISECONDS int64
    assert "ts" in df.columns
    assert str(df["ts"].dtype) == "int64"
    # 1_600_000_000 s -> 1_600_000_000_000 ms
    assert int(df["ts"].iloc[0]) == 1_600_000_000 * 1000
    # ascending + net_change == diff
    assert df["ts"].is_monotonic_increasing
    pd.testing.assert_series_equal(
        df["net_change"].reset_index(drop=True),
        df["total_supply"].diff().reset_index(drop=True),
        check_names=False,
    )
    # manifest recorded an ok pull, no blockers
    assert mani.blockers == []
    assert any(e["status"] == "ok" for e in mani.entries)


def test_run_is_failsoft_on_broken_transport(tmp_path):
    mani = Manifest(run_as_of="2026-06-26")
    # Must NOT raise; records a blocker instead.
    run(tmp_path, mani, run_as_of="2026-06-26",
        client_factory=lambda: _BoomClient(),
        legacy_csv_path=tmp_path / "legacy.csv")
    target = tmp_path / "defillama" / "stablecoin_supply" / "stablecoin_supply.parquet"
    assert not target.exists()
    assert len(mani.blockers) >= 1


def test_run_idempotent_skip_on_rerun(tmp_path):
    mani1 = Manifest(run_as_of="2026-06-26")
    run(tmp_path, mani1, run_as_of="2026-06-26",
        client_factory=lambda: _FakeClient(),
        legacy_csv_path=tmp_path / "legacy.csv")

    # Second run with a broken transport: the cached parquet means fetch_fn is
    # never called, so no blocker and a skip is logged.
    mani2 = Manifest(run_as_of="2026-06-26")
    run(tmp_path, mani2, run_as_of="2026-06-26",
        client_factory=lambda: _BoomClient(),
        legacy_csv_path=tmp_path / "legacy.csv")
    assert mani2.blockers == []
    assert any(e["status"] == "skip" for e in mani2.entries)
