"""Statistical-rigor helpers for skill/strategy evaluation.

Implements the Probabilistic Sharpe Ratio (PSR), the Deflated Sharpe Ratio
(DSR) and the Minimum Track Record Length (MinTRL) from Bailey & López de
Prado, "The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest
Overfitting and Non-Normality" (2014).

Why this exists: ranking skills by a naive annualized Sharpe and pruning on
Sharpe < 0 is exactly the cherry-picking the DSR was designed to defeat — with
3+ skills evaluated on a handful of trades, the "best" Sharpe is almost
certainly noise. PSR/DSR turn a point estimate into P(true Sharpe > benchmark)
while accounting for sample size, skewness, kurtosis AND the number of trials.

All Sharpe ratios here are PER-PERIOD (per-trade), never annualized — there is
no spurious sqrt(252) factor.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from scipy.stats import kurtosis as _kurtosis
from scipy.stats import norm
from scipy.stats import skew as _skew

_EULER_MASCHERONI = 0.5772156649015329


def sharpe_ratio(returns: Sequence[float]) -> float:
    """Per-period Sharpe = mean / sample-std (ddof=1). 0.0 if undefined."""
    r = np.asarray(list(returns), dtype=float)
    if r.size < 2:
        return 0.0
    sd = float(r.std(ddof=1))
    if sd <= 0.0:
        return 0.0
    return float(r.mean() / sd)


def _moments(returns: Sequence[float]) -> tuple[float, float, float, int]:
    """Return (sharpe, skew_g1, kurtosis_g2_pearson, n). Robust to zero variance."""
    r = np.asarray(list(returns), dtype=float)
    n = int(r.size)
    if n < 2 or r.std(ddof=1) <= 0.0:
        return 0.0, 0.0, 3.0, n  # normal-distribution defaults when variance is degenerate
    sr = float(r.mean() / r.std(ddof=1))
    g1 = float(_skew(r, bias=True))
    g2 = float(_kurtosis(r, fisher=False, bias=True))  # Pearson: normal == 3
    return sr, g1, g2, n


def probabilistic_sharpe_ratio(returns: Sequence[float], sr_star: float = 0.0) -> float:
    """P(true per-period Sharpe > sr_star), in [0, 1]."""
    sr, g1, g2, n = _moments(returns)
    if n < 2:
        return 0.0
    inside = 1.0 - g1 * sr + ((g2 - 1.0) / 4.0) * sr * sr
    denom = math.sqrt(inside) if inside > 1e-12 else 1e-6
    z = (sr - sr_star) * math.sqrt(n - 1) / denom
    return float(norm.cdf(z))


def expected_max_sharpe(sr_variance: float, n_trials: int) -> float:
    """E[max Sharpe] across n_trials independent strategies of variance sr_variance.

    This is the deflation benchmark: the Sharpe you'd expect to see purely from
    running many trials, even with zero true skill. 0.0 when there is nothing to
    deflate (single trial or no cross-trial dispersion).
    """
    if n_trials < 2 or sr_variance <= 0.0:
        return 0.0
    g = _EULER_MASCHERONI
    max_z = (1.0 - g) * float(norm.ppf(1.0 - 1.0 / n_trials)) + g * float(
        norm.ppf(1.0 - 1.0 / (n_trials * math.e))
    )
    return math.sqrt(sr_variance) * max_z


def deflated_sharpe_ratio(
    returns: Sequence[float], sr_variance: float, n_trials: int
) -> float:
    """PSR deflated by the expected-maximum-Sharpe benchmark for n_trials. In [0, 1]."""
    sr_star = expected_max_sharpe(sr_variance, n_trials)
    return probabilistic_sharpe_ratio(returns, sr_star=sr_star)


def min_track_record_length(
    returns: Sequence[float], sr_star: float = 0.0, prob: float = 0.95
) -> float:
    """Number of observations needed for PSR(sr_star) >= prob. inf if no edge."""
    sr, g1, g2, n = _moments(returns)
    if n < 2 or sr <= sr_star:
        return math.inf
    z = float(norm.ppf(prob))
    inside = 1.0 - g1 * sr + ((g2 - 1.0) / 4.0) * sr * sr
    return 1.0 + inside * (z / (sr - sr_star)) ** 2


def score_skills(
    returns_by_skill: dict[str, list[float]],
    min_trades: int = 10,
    archive_dsr_below: float = 0.5,
) -> dict[str, dict]:
    """Score each skill with PSR/DSR and flag archival, deflating by the number
    of skills evaluated (the trials). A skill is archived ONLY when it has a real
    track record (>= min_trades), is cumulatively negative, AND its deflated
    Sharpe is below coin-flip — i.e. no statistical evidence of genuine edge.
    """
    items = list(returns_by_skill.items())
    n_trials = len(items)
    srs = [sharpe_ratio(r) for _, r in items]
    sr_variance = float(np.var(srs, ddof=1)) if n_trials >= 2 else 0.0

    out: dict[str, dict] = {}
    for skill_id, rets in items:
        n = len(rets)
        pnl_sum = float(sum(rets))
        dsr = deflated_sharpe_ratio(rets, sr_variance, n_trials)
        archived = n >= min_trades and pnl_sum < 0.0 and dsr < archive_dsr_below
        out[skill_id] = {
            "trades": n,
            "sharpe": sharpe_ratio(rets),
            "psr": probabilistic_sharpe_ratio(rets),
            "dsr": dsr,
            "pnl_sum": pnl_sum,
            "archived": archived,
        }
    return out
