"""Weekly re-rank: row -> per-skill return grouping (pure helper)."""

from types import SimpleNamespace as NS

from hermes.learning.weekly import returns_by_skill


def test_returns_by_skill_groups_and_normalizes_by_notional():
    rows = [
        NS(skill_id="a", pnl_usd=2.0, entry_price=100.0, qty=0.1),   # notional 10 -> +0.2
        NS(skill_id="a", pnl_usd=-1.0, entry_price=50.0, qty=0.1),   # notional 5  -> -0.2
        NS(skill_id=None, pnl_usd=1.0, entry_price=10.0, qty=1.0),   # notional 10 -> +0.1, "unknown"
    ]
    out = returns_by_skill(rows)
    assert out["a"] == [0.2, -0.2]
    assert out["unknown"] == [0.1]


def test_returns_by_skill_handles_missing_pnl():
    rows = [NS(skill_id="x", pnl_usd=None, entry_price=10.0, qty=1.0)]
    out = returns_by_skill(rows)
    assert out["x"] == [0.0]
