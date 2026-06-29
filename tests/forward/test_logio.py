"""FORWARD — shared append-only PIT log writer.

The unit under test is ``hermes.forward.logio`` — the single seam every flux and
shadow log writes through. It MUST be:

* point-in-time / append-only — a snapshot is stamped at collection and never
  rewritten retroactively;
* idempotent — re-running the cron on the same UTC day overwrites that day's row
  (per key), never appends a duplicate (so a cron that fires twice is safe);
* fail-soft on read — a missing log reads back as an empty frame, not an error.
"""

from __future__ import annotations

import pandas as pd

from hermes.forward import logio

_DAY_MS = 86_400_000


def test_append_then_read_round_trips(tmp_path):
    rows = pd.DataFrame({"ts": [10 * _DAY_MS], "ccy": ["BTC"], "val": [1.5]})
    logio.append_snapshot("skew", rows, key_cols=("ccy",), data_dir=tmp_path)
    back = logio.read_log("skew", data_dir=tmp_path)
    assert list(back["ccy"]) == ["BTC"]
    assert list(back["val"]) == [1.5]


def test_read_missing_log_is_empty_not_error(tmp_path):
    out = logio.read_log("never_written", data_dir=tmp_path)
    assert isinstance(out, pd.DataFrame)
    assert out.empty


def test_rerun_same_utc_day_overwrites_not_duplicates(tmp_path):
    # Two collections on the SAME UTC day for the same key → one row, last wins.
    first = pd.DataFrame({"ts": [11 * _DAY_MS], "ccy": ["BTC"], "val": [1.0]})
    second = pd.DataFrame({"ts": [11 * _DAY_MS + 3_600_000], "ccy": ["BTC"], "val": [2.0]})
    logio.append_snapshot("skew", first, key_cols=("ccy",), data_dir=tmp_path)
    logio.append_snapshot("skew", second, key_cols=("ccy",), data_dir=tmp_path)
    back = logio.read_log("skew", data_dir=tmp_path)
    assert len(back) == 1
    assert float(back["val"].iloc[0]) == 2.0  # the later collection wins


def test_distinct_days_accumulate(tmp_path):
    d1 = pd.DataFrame({"ts": [11 * _DAY_MS], "ccy": ["BTC"], "val": [1.0]})
    d2 = pd.DataFrame({"ts": [12 * _DAY_MS], "ccy": ["BTC"], "val": [2.0]})
    logio.append_snapshot("skew", d1, key_cols=("ccy",), data_dir=tmp_path)
    logio.append_snapshot("skew", d2, key_cols=("ccy",), data_dir=tmp_path)
    back = logio.read_log("skew", data_dir=tmp_path)
    assert len(back) == 2
    assert sorted(back["val"]) == [1.0, 2.0]


def test_distinct_keys_same_day_kept_separately(tmp_path):
    rows = pd.DataFrame(
        {"ts": [11 * _DAY_MS, 11 * _DAY_MS], "ccy": ["BTC", "ETH"], "val": [1.0, 9.0]}
    )
    logio.append_snapshot("skew", rows, key_cols=("ccy",), data_dir=tmp_path)
    back = logio.read_log("skew", data_dir=tmp_path)
    assert len(back) == 2
    assert dict(zip(back["ccy"], back["val"])) == {"BTC": 1.0, "ETH": 9.0}


def test_latest_ts_wins_within_an_out_of_order_batch(tmp_path):
    # A single append carrying same-(day,key) rows OUT OF ORDER must keep the latest
    # ts (not an arbitrary df-order row) — hardens against the dedup_forward_log
    # keep="last" footgun flagged by the adversarial audit.
    rows = pd.DataFrame(
        {
            "ts": [11 * _DAY_MS + 10 * 3_600_000, 11 * _DAY_MS + 8 * 3_600_000, 11 * _DAY_MS + 9 * 3_600_000],
            "ccy": ["BTC", "BTC", "BTC"],
            "val": [100.0, 50.0, 75.0],
        }
    )
    logio.append_snapshot("skew", rows, key_cols=("ccy",), data_dir=tmp_path)
    back = logio.read_log("skew", data_dir=tmp_path)
    assert len(back) == 1
    assert float(back["val"].iloc[0]) == 100.0  # the 10:00 row (latest ts) wins


def test_empty_key_cols_is_rejected(tmp_path):
    rows = pd.DataFrame({"ts": [10 * _DAY_MS], "ccy": ["BTC"], "val": [1.0]})
    try:
        logio.append_snapshot("skew", rows, key_cols=(), data_dir=tmp_path)
        raise AssertionError("expected ValueError for empty key_cols")
    except ValueError:
        pass


def test_non_epoch_ms_ts_is_rejected(tmp_path):
    rows = pd.DataFrame(
        {"ts": pd.to_datetime(["2026-01-01", "2026-01-02"]), "ccy": ["BTC", "ETH"], "val": [1.0, 2.0]}
    )
    try:
        logio.append_snapshot("skew", rows, key_cols=("ccy",), data_dir=tmp_path)
        raise AssertionError("expected TypeError for non-epoch-ms ts")
    except TypeError:
        pass


def test_forward_dir_is_under_data_dir(tmp_path):
    d = logio.forward_dir("comex", data_dir=tmp_path)
    assert d.exists()
    assert d.name == "comex"
    assert d.parent.name == "forward"
    assert d.parent.parent == tmp_path
