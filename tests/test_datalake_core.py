"""Stream-0 data-lake core infrastructure: rate-limit spacing, point-in-time
idempotency (covers_range), gap detection, provenance/manifest aggregation, the
idempotent cached_pull contract, Parquet round-trips, and DuckDB view building.

These are the load-bearing guarantees of S0: a strict <=30 req/min throttle, a
resume-after-crash that never re-pulls what is already complete, and a provenance
trail. They are pure logic / local I/O, so they are unit-tested with injected
fakes (a fake monotonic clock for the limiter, a sentinel fetch_fn for cached_pull)
and tmp_path for Parquet/DuckDB — no network.
"""

from __future__ import annotations

import pandas as pd
import pytest

from hermes.datalake.core import (
    Manifest,
    RateLimiter,
    build_views,
    cached_pull,
    covers_range,
    detect_gaps,
    discover_views,
    frame_ts_bounds,
    read_parquet,
    render_blockers_md,
    summarize_manifest,
    write_parquet,
)


# --------------------------------------------------------------------------- #
# fake clock so RateLimiter timing is deterministic (no real sleeping)
# --------------------------------------------------------------------------- #
class FakeClock:
    def __init__(self, t: float = 1000.0) -> None:
        self.t = t
        self.sleeps: list[float] = []

    def now(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt

    def sleep(self, dt: float) -> None:
        self.sleeps.append(dt)
        self.t += dt


def _limiter(clock: FakeClock, max_per_min: int = 30) -> RateLimiter:
    return RateLimiter(max_per_min, monotonic=clock.now, sleep=clock.sleep)


# --------------------------------------------------------------------------- #
# RateLimiter
# --------------------------------------------------------------------------- #
def test_min_interval_is_sixty_over_rate():
    assert _limiter(FakeClock(), 30).min_interval == pytest.approx(2.0)
    assert _limiter(FakeClock(), 120).min_interval == pytest.approx(0.5)


def test_first_wait_never_sleeps():
    clock = FakeClock()
    rl = _limiter(clock, 30)
    rl.wait()
    assert clock.sleeps == []


def test_back_to_back_waits_are_spaced_to_min_interval():
    clock = FakeClock()
    rl = _limiter(clock, 30)  # min_interval = 2.0
    rl.wait()  # first: no sleep
    rl.wait()  # immediately after, 0s elapsed -> must sleep the full 2.0
    assert clock.sleeps == [pytest.approx(2.0)]


def test_no_sleep_when_enough_time_already_elapsed():
    clock = FakeClock()
    rl = _limiter(clock, 30)  # min_interval = 2.0
    rl.wait()
    clock.advance(5.0)  # caller did 5s of other work
    rl.wait()  # 5s > 2s -> no throttle needed
    assert clock.sleeps == []


# --------------------------------------------------------------------------- #
# covers_range — the idempotency predicate
# --------------------------------------------------------------------------- #
def test_covers_range_full_containment():
    assert covers_range(10, 100, 20, 90) is True


def test_covers_range_false_when_starts_too_late():
    assert covers_range(30, 100, 20, 90) is False


def test_covers_range_false_when_ends_too_early():
    assert covers_range(10, 80, 20, 90) is False


def test_covers_range_false_when_no_data():
    assert covers_range(None, None, 20, 90) is False


def test_covers_range_open_ended_bounds():
    assert covers_range(10, 100, None, 90) is True   # start unconstrained
    assert covers_range(10, 100, 20, None) is True   # end unconstrained
    assert covers_range(30, 100, 20, None) is False  # start still checked


# --------------------------------------------------------------------------- #
# detect_gaps
# --------------------------------------------------------------------------- #
def test_detect_gaps_none_on_regular_grid():
    day = 86400
    ts = [0, day, 2 * day, 3 * day]
    assert detect_gaps(ts, day) == []


def test_detect_gaps_flags_a_missing_bar():
    day = 86400
    ts = [0, day, 3 * day]  # 2*day jump between bar 1 and 2
    gaps = detect_gaps(ts, day, tolerance=1.5)
    assert gaps == [(day, 3 * day)]


def test_detect_gaps_exact_interval_is_not_a_gap():
    day = 86400
    assert detect_gaps([0, day], day) == []


# --------------------------------------------------------------------------- #
# manifest aggregation + blockers rendering
# --------------------------------------------------------------------------- #
def test_summarize_manifest_counts_and_sums_rows():
    entries = [
        {"status": "ok", "rows": 100},
        {"status": "ok", "rows": 50},
        {"status": "skip", "rows": 30},
        {"status": "blocked"},
        {"status": "partial", "rows": 5},
    ]
    s = summarize_manifest(entries)
    assert s["ok"] == 2
    assert s["skip"] == 1
    assert s["blocked"] == 1
    assert s["partial"] == 1
    assert s["total_rows"] == 185


def test_render_blockers_md_has_required_action_line():
    md = render_blockers_md(
        [
            {
                "source": "deribit",
                "endpoint": "public/auth",
                "params": {"x": 1},
                "error": "invalid_scope",
                "action": "Create a READ-ONLY Deribit key (scope account:read).",
            }
        ]
    )
    assert "deribit" in md
    assert "invalid_scope" in md
    assert "ACTION REQUISE DE MALACHIE" in md
    assert "account:read" in md


# --------------------------------------------------------------------------- #
# Parquet helpers
# --------------------------------------------------------------------------- #
def test_parquet_round_trip(tmp_path):
    df = pd.DataFrame({"ts": [1, 2, 3], "v": [10.0, 20.0, 30.0]})
    p = tmp_path / "nested" / "x.parquet"
    write_parquet(df, p)  # must create parent dirs
    assert p.exists()
    back = read_parquet(p)
    pd.testing.assert_frame_equal(back, df)


def test_frame_ts_bounds():
    df = pd.DataFrame({"ts": [50, 10, 30], "v": [1, 2, 3]})
    assert frame_ts_bounds(df, "ts") == (10, 50)


def test_frame_ts_bounds_empty_is_none():
    assert frame_ts_bounds(pd.DataFrame({"ts": []}), "ts") == (None, None)


# --------------------------------------------------------------------------- #
# cached_pull — idempotent, fail-soft, provenance-logging
# --------------------------------------------------------------------------- #
def _boom():  # a fetch_fn that must NOT be called
    raise AssertionError("fetch_fn should not have been called")


def test_cached_pull_skips_when_target_covers_range(tmp_path):
    target = tmp_path / "lsr.parquet"
    write_parquet(pd.DataFrame({"ts": [10, 50, 100], "v": [1, 2, 3]}), target)
    m = Manifest(run_as_of="2026-06-26T00:00:00Z")

    out = cached_pull(
        target, _boom, source="coinglass", endpoint="lsr", params={},
        manifest=m, expected_range=(20, 90), ts_col="ts",
    )
    assert out is not None and len(out) == 3
    assert any(e["status"] == "skip" for e in m.entries)


def test_cached_pull_fetches_and_writes_when_missing(tmp_path):
    target = tmp_path / "lsr.parquet"
    df = pd.DataFrame({"ts": [10, 20, 30], "v": [1, 2, 3]})
    m = Manifest(run_as_of="2026-06-26T00:00:00Z")

    out = cached_pull(
        target, lambda: df, source="coinglass", endpoint="lsr", params={"symbol": "BTC"},
        manifest=m, expected_range=(10, 30), ts_col="ts",
    )
    assert target.exists()
    assert len(out) == 3
    ok = [e for e in m.entries if e["status"] == "ok"]
    assert len(ok) == 1 and ok[0]["rows"] == 3
    assert ok[0]["min_ts"] == 10 and ok[0]["max_ts"] == 30


def test_cached_pull_refetches_when_existing_does_not_cover(tmp_path):
    target = tmp_path / "lsr.parquet"
    write_parquet(pd.DataFrame({"ts": [10, 50], "v": [1, 2]}), target)  # ends at 50
    full = pd.DataFrame({"ts": [10, 50, 90], "v": [1, 2, 3]})
    m = Manifest(run_as_of="2026-06-26T00:00:00Z")

    out = cached_pull(
        target, lambda: full, source="coinglass", endpoint="lsr", params={},
        manifest=m, expected_range=(10, 90), ts_col="ts",
    )
    assert len(out) == 3  # refetched the fuller series
    assert any(e["status"] == "ok" for e in m.entries)


def test_cached_pull_empty_response_records_blocker_and_returns_none(tmp_path):
    target = tmp_path / "lsr.parquet"
    m = Manifest(run_as_of="2026-06-26T00:00:00Z")

    out = cached_pull(
        target, lambda: pd.DataFrame(), source="coinglass", endpoint="lsr",
        params={"symbol": "WIF"}, manifest=m, expected_range=(10, 30), ts_col="ts",
    )
    assert out is None
    assert not target.exists()
    assert len(m.blockers) == 1
    assert m.blockers[0]["source"] == "coinglass"


def test_cached_pull_failsoft_on_exception(tmp_path):
    target = tmp_path / "lsr.parquet"
    m = Manifest(run_as_of="2026-06-26T00:00:00Z")

    def explode():
        raise RuntimeError("429 rate limited")

    out = cached_pull(
        target, explode, source="coinglass", endpoint="lsr", params={},
        manifest=m, expected_range=(10, 30), ts_col="ts",
    )
    assert out is None  # fail-soft: does not raise
    assert len(m.blockers) == 1
    assert "429" in m.blockers[0]["error"]


def test_cached_pull_failsoft_on_write_error(tmp_path):
    """A non-empty frame that pyarrow cannot serialize (mixed bytes/float object
    column — the real CoinGlass failure mode) must become a per-unit blocker and
    return None, NOT propagate and abort the whole source run."""
    target = tmp_path / "bad.parquet"
    bad = pd.DataFrame({"ts": [1, 2], "v": [b"x", 1.0]})  # mixed bytes/float -> Arrow error
    m = Manifest(run_as_of="2026-06-26T00:00:00Z")

    out = cached_pull(
        target, lambda: bad, source="coinglass", endpoint="oi", params={},
        manifest=m, expected_range=None, ts_col="ts",
    )
    assert out is None
    assert not target.exists()  # no corrupt partial file left behind
    assert len(m.blockers) == 1


# --------------------------------------------------------------------------- #
# DuckDB view layer
# --------------------------------------------------------------------------- #
def test_build_views_makes_queryable_views(tmp_path):
    raw = tmp_path / "raw"
    write_parquet(pd.DataFrame({"ts": [1, 2], "symbol": ["BTC", "BTC"]}),
                  raw / "coinglass" / "lsr_top" / "BTC_1d.parquet")
    write_parquet(pd.DataFrame({"ts": [3], "symbol": ["ETH"]}),
                  raw / "coinglass" / "lsr_top" / "ETH_1d.parquet")
    db = tmp_path / "lake.duckdb"

    build_views(db, {"lsr_top": str(raw / "coinglass" / "lsr_top" / "*.parquet")})

    import duckdb

    con = duckdb.connect(str(db))
    n = con.execute("SELECT COUNT(*) FROM lsr_top").fetchone()[0]
    assert n == 3
    con.close()


def test_discover_views_groups_parquet_by_directory(tmp_path):
    raw = tmp_path / "raw"
    write_parquet(pd.DataFrame({"ts": [1]}), raw / "coinglass" / "lsr_top" / "BTC_1d.parquet")
    write_parquet(pd.DataFrame({"ts": [2]}), raw / "coinglass" / "lsr_top" / "ETH_1d.parquet")
    write_parquet(pd.DataFrame({"date": [1]}), raw / "fred" / "DFII10.parquet")
    write_parquet(pd.DataFrame({"date": [2]}), raw / "fred" / "DGS10.parquet")

    views = discover_views(raw)
    assert set(views) == {"coinglass_lsr_top", "fred"}
    assert views["fred"].endswith("fred/*.parquet")
    # the discovered mapping must be directly usable by build_views
    build_views(tmp_path / "lake.duckdb", views)
    import duckdb

    con = duckdb.connect(str(tmp_path / "lake.duckdb"))
    assert con.execute("SELECT COUNT(*) FROM coinglass_lsr_top").fetchone()[0] == 2
    assert con.execute("SELECT COUNT(*) FROM fred").fetchone()[0] == 2
    con.close()
