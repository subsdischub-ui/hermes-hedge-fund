"""Turn a BacktestResult into an honest edge report: per-skill and overall
Deflated Sharpe (deflated by the number of skills tried), win-rate net of fees,
and the Probability of Backtest Overfitting."""

from __future__ import annotations

import numpy as np

from hermes.learning.stats import deflated_sharpe_ratio, sharpe_ratio
from hermes.validation.backtest import BacktestResult
from hermes.validation.pbo import probability_backtest_overfitting


def _win_rate(rets: list[float]) -> float:
    return float(np.mean([1.0 if r > 0 else 0.0 for r in rets])) if rets else 0.0


def _pbo_from_samples(samples: list[dict], skills: list[str], n_blocks: int) -> float:
    if len(skills) < 2 or len(samples) < n_blocks:
        return 0.0
    seqs = [s["seq"] for s in samples]
    lo, hi = min(seqs), max(seqs)
    if hi == lo:
        return 0.0
    col = {s: i for i, s in enumerate(skills)}
    edges = np.linspace(lo, hi + 1, n_blocks + 1)
    sums = np.zeros((n_blocks, len(skills)))
    cnts = np.zeros((n_blocks, len(skills)))
    for smp in samples:
        b = min(max(int(np.searchsorted(edges, smp["seq"], side="right") - 1), 0), n_blocks - 1)
        j = col[smp["skill_id"]]
        sums[b, j] += smp["ret_net"]
        cnts[b, j] += 1
    matrix = np.where(cnts > 0, sums / np.where(cnts > 0, cnts, 1), 0.0)
    return probability_backtest_overfitting(matrix.tolist(), n_splits=n_blocks)


def backtest_report(result: BacktestResult, n_blocks: int = 10) -> dict:
    rbs = result.returns_by_skill
    skills = sorted(rbs)
    srs = [sharpe_ratio(rbs[s]) for s in skills]
    sr_var = float(np.var(srs, ddof=1)) if len(skills) >= 2 else 0.0
    n_trials = max(1, len(skills))

    per_skill = {
        s: {
            "n": len(rbs[s]),
            "win_rate": _win_rate(rbs[s]),
            "mean_ret": float(np.mean(rbs[s])) if rbs[s] else 0.0,
            "dsr": deflated_sharpe_ratio(rbs[s], sr_var, n_trials),
        }
        for s in skills
    }
    pooled = [x for s in skills for x in rbs[s]]
    return {
        "per_skill": per_skill,
        "overall": {
            "n": len(pooled),
            "win_rate": _win_rate(pooled),
            "mean_ret": float(np.mean(pooled)) if pooled else 0.0,
            "total_ret": float(np.sum(pooled)) if pooled else 0.0,
            "dsr": deflated_sharpe_ratio(pooled, sr_var, n_trials),
            "pbo": _pbo_from_samples(result.samples, skills, n_blocks),
        },
    }
