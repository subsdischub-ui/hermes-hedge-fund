"""Turn a BacktestResult into an honest edge report: per-skill and overall
Deflated Sharpe (deflated by the number of skills tried), win-rate net of fees,
and the Probability of Backtest Overfitting."""

from __future__ import annotations

import numpy as np

from hermes.learning.stats import deflated_sharpe_ratio, sharpe_ratio
from hermes.validation.backtest import BacktestResult
from hermes.validation.cpcv import combinatorial_purged_splits
from hermes.validation.pbo import probability_backtest_overfitting


def _win_rate(rets: list[float]) -> float:
    return float(np.mean([1.0 if r > 0 else 0.0 for r in rets])) if rets else 0.0


def cpcv_sharpe_distribution(
    returns: list[float],
    n_groups: int = 6,
    k_test: int = 2,
    embargo: int = 0,
) -> dict:
    """CPCV multi-path Sharpe distribution over a single return stream.

    Partition the chronological returns into ``n_groups`` blocks and, for every
    combination of ``k_test`` blocks held out as test, compute the test-fold
    Sharpe. The *distribution* of those C(n_groups, k_test) Sharpes — not a
    single point — is the engine's robustness arbiter: a real edge stays positive
    across most paths; an overfit one does not. Returns zero paths when there is
    too little history to split.
    """
    rets = list(returns)
    zero = {
        "n_paths": 0,
        "sharpe_mean": 0.0,
        "sharpe_std": 0.0,
        "sharpe_median": 0.0,
        "frac_positive": 0.0,
    }
    if len(rets) < n_groups:
        return zero

    arr = np.asarray(rets, dtype=float)
    splits = combinatorial_purged_splits(len(rets), n_groups, k_test, embargo)
    sharpes = [
        sharpe_ratio(arr[test].tolist()) for _train, test in splits if test.size >= 2
    ]
    if not sharpes:
        return zero
    sh = np.asarray(sharpes, dtype=float)
    return {
        "n_paths": int(sh.size),
        "sharpe_mean": float(np.mean(sh)),
        "sharpe_std": float(np.std(sh, ddof=1)) if sh.size >= 2 else 0.0,
        "sharpe_median": float(np.median(sh)),
        "frac_positive": float(np.mean(sh > 0.0)),
    }


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


def backtest_report(
    result: BacktestResult, n_blocks: int = 10, n_trials: int | None = None
) -> dict:
    """Honest edge report. ``n_trials`` is the TRUE number of backtested
    configurations (from the persistent TrialRegistry) used to deflate the
    Sharpe; when omitted it falls back to the number of skills, which
    under-counts and so over-states the DSR (anti-overfit trap #1)."""
    rbs = result.returns_by_skill
    skills = sorted(rbs)
    srs = [sharpe_ratio(rbs[s]) for s in skills]
    sr_var = float(np.var(srs, ddof=1)) if len(skills) >= 2 else 0.0
    effective_trials = int(n_trials) if n_trials else max(1, len(skills))

    per_skill = {
        s: {
            "n": len(rbs[s]),
            "win_rate": _win_rate(rbs[s]),
            "mean_ret": float(np.mean(rbs[s])) if rbs[s] else 0.0,
            "dsr": deflated_sharpe_ratio(rbs[s], sr_var, effective_trials),
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
            "dsr": deflated_sharpe_ratio(pooled, sr_var, effective_trials),
            "pbo": _pbo_from_samples(result.samples, skills, n_blocks),
            "cpcv": cpcv_sharpe_distribution(pooled),
            "n_trials": effective_trials,
        },
    }
