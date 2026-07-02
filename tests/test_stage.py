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


def test_stage0_gate_counts_edge_trades_not_any_trades():
    """RESTRUCTURATION 2026-07-02 (docs/RESTRUCTURATION_2026-07.md §3): the stage_0
    gate must count ONLY current-slate trades (edge_paper_trades), never the old
    any-skill paper_trades counter — a gate fed by a retired drag skill would advance
    stages on bull-beta noise."""
    from hermes.settings import get_settings

    rules = (get_settings().yaml("stages", default={}) or {})["stage_0_paper"]["advance_when"]
    assert "edge_paper_trades" in rules
    assert "paper_trades" not in rules
    assert rules["edge_paper_trades"] == 40


def test_edge_gate_ignores_legacy_trade_count():
    """A flood of legacy-skill trades must NOT open the gate: only edge_paper_trades
    (current-slate count) is consulted by the config rules."""
    mgr = StageManager()
    rep = mgr.evaluate({
        "paper_trades": 10_000,      # any-skill count — must be irrelevant
        "edge_paper_trades": 3,      # current-slate count — below the 40 target
        "walk_forward_sharpe": 2.0,
        "max_drawdown_pct": 5.0,
    })
    target, observed, passed = rep.gates["edge_paper_trades"]
    assert (target, observed, passed) == (40, 3, False)
    assert rep.can_advance is False
