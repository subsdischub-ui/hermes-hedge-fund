"""Persisted agent-activity feed — so the dashboard shows what the agent DID at
each tick (decision + reason), not only closed trades. A bounded ring buffer in
/data survives restarts."""

from __future__ import annotations

from hermes.runtime.activity import ActivityLog


def test_ring_buffer_caps_and_returns_newest_first(tmp_path):
    log = ActivityLog(tmp_path / "activity.json", cap=3)
    for i in range(5):
        log.record({"i": i, "kind": "tick"})
    items = log.recent(10)
    assert [x["i"] for x in items] == [4, 3, 2]  # newest first, capped at 3


def test_record_stamps_ts_and_persists(tmp_path):
    path = tmp_path / "activity.json"
    ActivityLog(path, cap=10).record({"symbol": "BTC/USDC", "decision": "HOLD"})
    # A fresh instance reads the persisted entry, and ts was stamped.
    items = ActivityLog(path, cap=10).recent(5)
    assert len(items) == 1
    assert items[0]["symbol"] == "BTC/USDC"
    assert "ts" in items[0]


def test_recent_limit(tmp_path):
    log = ActivityLog(tmp_path / "activity.json", cap=100)
    for i in range(20):
        log.record({"i": i})
    assert len(log.recent(5)) == 5


def test_corrupt_file_is_tolerated(tmp_path):
    path = tmp_path / "activity.json"
    path.write_text("{ not json")
    log = ActivityLog(path, cap=10)
    assert log.recent(5) == []
    log.record({"i": 1})  # still works after corruption
    assert log.recent(5)[0]["i"] == 1
