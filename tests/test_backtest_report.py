"""Backtest report: per-skill + overall Deflated Sharpe, win-rate (net of fees)
and Probability of Backtest Overfitting."""

from hermes.validation.backtest import BacktestResult
from hermes.validation.report import backtest_report


def _sample(skill, ret, seq):
    return {
        "skill_id": skill, "ret_net": ret, "outcome": 1 if ret > 0 else 0,
        "seq": seq, "features": {}, "confidence": 0.6,
    }


def test_report_structure_and_metrics():
    samples = [_sample("a", 0.02, 0), _sample("a", -0.01, 1), _sample("b", 0.03, 2)]
    res = BacktestResult(
        samples=samples, returns_by_skill={"a": [0.02, -0.01], "b": [0.03]}
    )
    rep = backtest_report(res)
    assert rep["overall"]["n"] == 3
    assert 0.0 <= rep["overall"]["win_rate"] <= 1.0
    assert 0.0 <= rep["overall"]["dsr"] <= 1.0
    assert 0.0 <= rep["overall"]["pbo"] <= 1.0
    assert rep["per_skill"]["a"]["n"] == 2
    assert rep["per_skill"]["a"]["win_rate"] == 0.5
    assert 0.0 <= rep["per_skill"]["b"]["dsr"] <= 1.0


def test_report_empty_result():
    rep = backtest_report(BacktestResult())
    assert rep["overall"]["n"] == 0
    assert rep["per_skill"] == {}
