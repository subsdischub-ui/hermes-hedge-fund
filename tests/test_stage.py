from hermes.risk.stage import StageManager


def test_advance_blocked_when_gate_missing():
    mgr = StageManager()
    rep = mgr.evaluate({})
    assert rep.can_advance is False


def test_max_drawdown_is_upper_bound():
    mgr = StageManager()
    rep = mgr.evaluate({
        "paper_trades": 250,
        "walk_forward_sharpe": 1.5,
        "max_drawdown_pct": 9.0,   # below 15% target -> pass
    })
    # Verify max_drawdown gate uses upper-bound semantics.
    if "max_drawdown_pct" in rep.gates:
        _, observed, passed = rep.gates["max_drawdown_pct"]
        assert passed is True
